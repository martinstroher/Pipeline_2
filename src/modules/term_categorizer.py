import json
import os
import time

import pandas as pd
from tqdm import tqdm

from src.utils.gemini_client import generate
from src.utils import log


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
    GEORESERVOIR_DEFS_PATH = os.environ["GEORESERVOIR_DEFS_PATH"]
    GEOCORE_DEFS_PATH = os.environ["GEOCORE_DEFS_PATH"]
    BFO_DEFS_PATH = os.environ["BFO_DEFS_PATH"]

    log.info(f"Categorizing in batches of {BATCH_SIZE}.")


    def load_definitions_from_file(filepath):
        if not os.path.exists(filepath):
            log.error(f"Definition file not found: '{filepath}'")
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            log.error(f"Reading definition file '{filepath}': {e}")
            return None


    georeservoir_definitions= load_definitions_from_file(GEORESERVOIR_DEFS_PATH)
    geocore_definitions = load_definitions_from_file(GEOCORE_DEFS_PATH)
    bfo_definitions = load_definitions_from_file(BFO_DEFS_PATH)
    if not geocore_definitions or not bfo_definitions:
        raise RuntimeError("Required ontology definition files could not be loaded.")

    valid_categories = (
        _extract_category_names(georeservoir_definitions or "")
        | _extract_category_names(geocore_definitions or "")
        | _extract_category_names(bfo_definitions or "")
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


    system_instruction = "You are an expert ontology engineer specializing in foundational (BFO) and geological (GeoCore and GeoReservoir) ontologies. You process data in batches and your response format MUST be a valid JSON array of objects."
    prompt_template = """Your task is to classify a batch of geological terms based on their Natural Language Definitions (NLDs).

    **METHODOLOGY (Follow Strictly for each item):**
    1.  **Analyze Data:** Read the Term and, if present, its NLD.
    2.  **Prioritize GeoReservoir:** First, attempt to classify the term into one of the `### GeoReservoir Categories`.
    3.  **Fallback to GeoCore:** If and only if no GeoReservoir category is a good fit, then attempt to classify it into one of the `### GeoCore Categories`.
    4.  **Fallback to BFO:** If and only if no GeoCore category fits, then attempt to classify it into one of the `### BFO Categories`.
    5.  **Final Fallback:** If the term does not fit well into ANY of the provided categories (GeoReservoir, GeoCore, or BFO), you MUST use the string `NOT_CLASSIFIED`. Reserve NOT_CLASSIFIED for physical analytical instruments treated as objects (e.g., 'Microscope'). Characterization methods and analytical processes that describe geological observations or workflows (e.g., 'Petrographic Analysis', 'Core Analysis') may fit 'Geological Process' in GeoCore — prefer a real category over NOT_CLASSIFIED when the NLD describes a geological action, observation, or property.
    6.  **Provide Reasoning:** In one short sentence, explain WHY you chose that category based on the NLD.

    **INPUT/OUTPUT FORMAT:**
    -   **INPUT:** A JSON array of objects, where each object has a "term" and optionally an "nld" field.
    -   **OUTPUT:** Your response MUST BE a valid JSON array. Each object in the array must contain the "term", the assigned "category", and a "reasoning" string.
    -   The value of "category" MUST exactly match one of the category name strings listed above, verbatim, including capitalization (e.g., "Sedimentary Rock" not "sedimentary rock" or "Sedimentary Rocks"). The only exception is "NOT_CLASSIFIED".

    ---
    **ONTOLOGY CATEGORIES REFERENCE:**

    ### GeoReservoir Categories:
    {georeservoir_definitions}

    ### GeoCore Categories:
    {geocore_definitions}

    ### BFO Categories:
    {bfo_definitions}

    ---
    **EXAMPLES:**

    Input:  [{{"term": "Grainstone",
              "nld": "Grainstone is a grain-supported sedimentary carbonate rock that lacks micrite matrix, with allochems typically consisting of bivalves, ostracods, or ooids."}}]
    Output: [{{"term": "Grainstone",
              "category": "Sedimentary Rock",
              "reasoning": "Directly describes a type of sedimentary rock classified under GeoReservoir."}}]

    Input:  [{{"term": "Normal Fault",
              "nld": "Normal Fault is a geological structure formed by extensional tectonics where the hanging wall moves down relative to the footwall along a dip-slip fault plane."}}]
    Output: [{{"term": "Normal Fault",
              "category": "Geological Structure",
              "reasoning": "Describes the internal structural arrangement of a geological object, fitting GeoCore."}}]

    Input:  [{{"term": "Core Sample"}}]
    Output: [{{"term": "Core Sample",
              "category": "NOT_CLASSIFIED",
              "reasoning": "Analytical instrument used to characterize formations, not an ontological geological concept."}}]

    ---
    **DATA TO CLASSIFY:**
    {json_batch}
    """

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
