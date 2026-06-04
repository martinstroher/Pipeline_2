import json
import os
import time

import pandas as pd
from tqdm import tqdm

from src.utils.gemini_client import generate
from src.utils import log
from src.utils.prompt_loader import load_prompt
from src.utils.ontology_config import get_config


def _extract_category_names(text: str) -> set:
    """Extract category names from definition file (format: 'Name: description')."""
    names = set()
    for line in text.strip().splitlines():
        line = line.strip()
        if line and ':' in line:
            name = line.split(':')[0].strip()
            if name:
                names.add(name)
    return names


def run_term_categorization():
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))
    MODEL_NAME = os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    MODEL_TEMPERATURE = float(os.environ.get("LLM_GENERATION_TEMPERATURE", 0))
    INPUT_FILE_PATH = os.environ["CONSOLIDATED_LLM_RESULTS_WITH_NLDS"]
    OUTPUT_FILE_PATH = os.environ["CATEGORIZED_LLM_TERMS"]

    log.info(f"Categorizing in batches of {BATCH_SIZE}.")

    cfg = get_config()
    georeservoir_definitions = cfg.llm_definitions_block("georeservoir")
    geocore_definitions = cfg.llm_definitions_block("geocore")
    bfo_definitions = cfg.llm_definitions_block("bfo")
    if not geocore_definitions or not bfo_definitions:
        raise RuntimeError("Required ontology definition blocks are empty in ontology_config.yaml.")

    valid_categories = (
        _extract_category_names(georeservoir_definitions)
        | _extract_category_names(geocore_definitions)
        | _extract_category_names(bfo_definitions)
    )


    def load_nlds_from_csv(filepath):
        if not os.path.exists(filepath):
            log.error(f"File '{filepath}' not found.")
            return None
        try:
            df = pd.read_csv(filepath, encoding='utf-8-sig', delimiter=',', header=0,
                             usecols=['Term', 'NLD', 'Context_Used'])

            log.info(f"{len(df)} terms loaded for categorization.")
            return df
        except Exception as e:
            log.error(f"Reading CSV: {e}")
            return None


    system_instruction, prompt_template = load_prompt("term_categorization.txt")

    df_nlds = load_nlds_from_csv(INPUT_FILE_PATH)

    if df_nlds is not None:
        classification_results = []
        total_terms = len(df_nlds)

        pbar = tqdm(total=total_terms, desc="Categorizing")
        for i in range(0, total_terms, BATCH_SIZE):
            batch_df = df_nlds.iloc[i:i + BATCH_SIZE]

            batch_list = []
            for index, row in batch_df.iterrows():
                batch_list.append({
                    "term": row['Term'],
                    "nld": row['NLD']
                })
            json_batch_str = json.dumps(batch_list, indent=2)

            try:
                final_prompt = prompt_template.format(geocore_definitions=geocore_definitions,
                                                      bfo_definitions=bfo_definitions,
                                                      georeservoir_definitions= georeservoir_definitions,
                                                      json_batch=json_batch_str)
                response_text = generate(
                    final_prompt,
                    model=MODEL_NAME,
                    system_instruction=system_instruction,
                    temperature=MODEL_TEMPERATURE,
                    response_mime_type="application/json",
                )
                response_json = json.loads(response_text)

                if len(response_json) != len(batch_df):
                    raise ValueError("LLM response length does not match batch size.")

                for idx, result_item in enumerate(response_json):
                    original_row = batch_df.iloc[idx]
                    category = result_item.get('category', 'ERROR_PARSE')
                    if (category not in valid_categories
                            and category != "NOT_CLASSIFIED"
                            and not category.startswith("ERROR")):
                        log.warn(f"Unknown category '{category}' for '{original_row['Term']}' — may break IRI lookup")

                    classification_results.append({
                        'Term': original_row['Term'],
                        'RAG_Context_Used': original_row['Context_Used'],
                        'Category': category,
                        'Reasoning': result_item.get('reasoning', ''),
                        'NLD': original_row['NLD']
                    })

            except json.JSONDecodeError:
                tqdm.write("")
                log.error("LLM returned invalid JSON. Batch flagged.")
                for item in batch_list:
                    classification_results.append({
                        'Term': item['term'],
                        'RAG_Context_Used': '',
                        'Category': 'ERROR_INVALID_JSON',
                        'Reasoning': 'LLM response was not valid JSON.',
                        'NLD': item['nld']
                    })
            except Exception as e:
                tqdm.write("")
                log.error(f"Classifying batch: {e}")
                for item in batch_list:
                    classification_results.append({
                        'Term': item['term'],
                        'RAG_Context_Used': '',
                        'Category': 'ERROR_GENERAL',
                        'Reasoning': f'Error: {str(e)}',
                        'NLD': item['nld']
                    })

            pbar.update(len(batch_df))
            time.sleep(2)

        pbar.close()

        output_dir = os.path.dirname(OUTPUT_FILE_PATH)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                log.detail(f"Created directory: {output_dir}")
            except OSError as e:
                raise RuntimeError(f"Creating directory {output_dir}: {e}")

        try:
            final_df = pd.DataFrame(classification_results)

            final_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
            log.success(f"{len(final_df)} terms categorized -> '{OUTPUT_FILE_PATH}'")
        except Exception as e:
            log.error(f"Saving CSV '{OUTPUT_FILE_PATH}': {e}")
