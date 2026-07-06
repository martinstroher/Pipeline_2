import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.utils.csv_io import read_csv, write_csv
from tqdm import tqdm

from src.utils.llm_client import generate, parse_json_array
from src.utils import log
from src.utils.prompt_loader import load_prompt
from src.utils.ontology_config import get_config


def run_term_categorization():
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))
    MAX_WORKERS = int(os.environ.get("MAX_CONCURRENT_CATEGORIZE", 5))
    MODEL_NAME = os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    MODEL_TEMPERATURE = float(os.environ.get("LLM_GENERATION_TEMPERATURE", 0))
    INPUT_FILE_PATH = os.environ["CONSOLIDATED_LLM_RESULTS_WITH_NLDS"]
    OUTPUT_FILE_PATH = os.environ["CATEGORIZED_LLM_TERMS"]

    log.info(f"Categorizing in batches of {BATCH_SIZE} ({MAX_WORKERS} workers).")

    cfg = get_config()
    categories_block = cfg.categorization_block()
    if not categories_block.strip():
        raise RuntimeError("categorization_block() returned empty — check waterfall: in ontology_config.yaml.")

    valid_categories = {
        label
        for key in cfg.waterfall_ontologies()
        for label in cfg.categories_for(key)
    }


    def load_nlds_from_csv(filepath):
        if not os.path.exists(filepath):
            log.error(f"File '{filepath}' not found.")
            return None
        try:
            df = read_csv(filepath, delimiter=',', header=0,
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
        lock = threading.Lock()

        def _process_batch(batch_df: pd.DataFrame) -> list[dict]:
            batch_list = [
                {"term": row["Term"], "nld": row["NLD"]}
                for _, row in batch_df.iterrows()
            ]
            json_batch_str = json.dumps(batch_list, indent=2)
            try:
                final_prompt = prompt_template.format(
                    categories_block=categories_block,
                    json_batch=json_batch_str,
                )
                response_text = generate(
                    final_prompt,
                    model=MODEL_NAME,
                    system_instruction=system_instruction,
                    temperature=MODEL_TEMPERATURE,
                    response_mime_type="application/json",
                )
                response_json = parse_json_array(response_text)

                if len(response_json) != len(batch_df):
                    raise ValueError("LLM response length does not match batch size.")

                rows = []
                for idx, result_item in enumerate(response_json):
                    original_row = batch_df.iloc[idx]
                    category = result_item.get("category", "ERROR_PARSE")
                    if (
                        category not in valid_categories
                        and category != "NOT_CLASSIFIED"
                        and not category.startswith("ERROR")
                    ):
                        tqdm.write(
                            f"  [warn] Unknown category '{category}' for "
                            f"'{original_row['Term']}' — may break IRI lookup"
                        )
                    rows.append({
                        "Term": original_row["Term"],
                        "RAG_Context_Used": original_row["Context_Used"],
                        "Category": category,
                        "Reasoning": result_item.get("reasoning", ""),
                        "NLD": original_row["NLD"],
                    })
                return rows

            except json.JSONDecodeError:
                tqdm.write("  [error] LLM returned invalid JSON. Batch flagged.")
                return [
                    {
                        "Term": item["term"],
                        "RAG_Context_Used": "",
                        "Category": "ERROR_INVALID_JSON",
                        "Reasoning": "LLM response was not valid JSON.",
                        "NLD": item["nld"],
                    }
                    for item in batch_list
                ]
            except Exception as e:
                tqdm.write(f"  [error] Classifying batch: {e}")
                return [
                    {
                        "Term": item["term"],
                        "RAG_Context_Used": "",
                        "Category": "ERROR_GENERAL",
                        "Reasoning": f"Error: {str(e)}",
                        "NLD": item["nld"],
                    }
                    for item in batch_list
                ]

        batches = [
            df_nlds.iloc[i : i + BATCH_SIZE]
            for i in range(0, total_terms, BATCH_SIZE)
        ]
        max_workers = min(MAX_WORKERS, len(batches)) if batches else 1
        log.info(f"Dispatching {len(batches)} batch(es) to {max_workers} worker(s)")

        pbar = tqdm(total=total_terms, desc="Categorizing")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_batch, b): b for b in batches}
            for future in as_completed(futures):
                rows = future.result()
                with lock:
                    classification_results.extend(rows)
                pbar.update(len(rows))
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

            write_csv(final_df, OUTPUT_FILE_PATH)
            log.success(f"{len(final_df)} terms categorized -> '{OUTPUT_FILE_PATH}'")
        except Exception as e:
            log.error(f"Saving CSV '{OUTPUT_FILE_PATH}': {e}")
