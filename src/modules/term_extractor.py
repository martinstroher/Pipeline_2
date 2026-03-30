import json
import os

import pandas as pd
from tqdm import tqdm

from src.utils import log
from src.utils.gemini_client import generate


def run_llm_term_extraction():
    LLM_MODEL_NAME = os.environ.get("LLM_EXTRACTION_MODEL", "gemini-2.5-flash")
    LLM_MODEL_TEMPERATURE = float(os.environ.get("LLM_EXTRACTION_TEMPERATURE", 0.0))
    LLM_INPUT_DIR = os.environ["LLM_INPUT_DIR"]
    LLM_OUTPUT_FILE = os.environ["LLM_OUTPUT_FILE"]

    def load_papers_from_dir(directory):
        papers = []
        if not os.path.exists(directory):
             log.error(f"Directory '{directory}' not found.")
             return []

        files = [f for f in os.listdir(directory) if f.endswith('.md')]
        if not files:
             log.error(f"No .md files found in '{directory}'.")
             return []

        for filename in files:
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                if content.strip():
                    papers.append(content)
                log.detail(f"Loaded: {filename} ({len(content)} chars)")
            except Exception as e:
                log.error(f"Reading {filename}: {e}")
        return papers

    system_instruction = """You are an expert geologist and ontology engineer specializing in South Atlantic Pre-Salt petroleum systems.
                            Your task is to extract core geological concepts from scientific texts suitable for building a domain ontology.
                          This ontology's primary purpose is to assist geologists in describing and comparing analog reservoirs geological settings."""

    prompt_template = """**METHODOLOGY**
    1.  **Identify Conceptual Entities:** Identify all terms or phrases representing geological concepts.
    Focus on identifying *types* or *classes* of entities relevant to petroleum geology and pre-salt context.
    You MUST only extract terms that are explicitly mentioned in the provided text. Do NOT generate terms from your own knowledge.
    2.  **Normalize Terms:** Return all extracted concepts translated to English and, where appropriate, in their singular, base form (e.g., "carbonates" -> "Carbonate", "faults" -> "Fault").
    Use title case for concepts.
    3.  **Strict Filtering:** You MUST exclude:
        * Highly specific identifiers with no ontological value: individual well names (e.g., 'Well 1-BRSA-123'), author names, company names.
        * Units of measure, numerical values, and codes (e.g., 'mD', 'API', '10%', 'SiO2').
        * Analytical methods, laboratory techniques, or observational instruments (e.g., 'Thin Section', 'Core Plug', 'Seismic Survey', 'Well Log').
        Note: Named petroleum fields (e.g., 'Lula Field', 'Búzios'), basin names (e.g., 'Santos Basin'), and named geological formations ARE valid — extract them. They will be classified as OWL named individuals downstream.
    4.  **Focus:** Prioritize terms that represent reusable classes within an ontology framework. Be selective — it is better to miss a marginal term than to include noise. Note: named geological time periods (e.g., Aptian, Cretaceous) are valid and should be extracted — they will be classified as OWL individuals downstream.

    **OUTPUT FORMAT:**
    Your response MUST BE a valid JSON array of unique strings.

    **Example of output array:**
    ["Microbial Carbonate", "Diagenesis", "Source Rock", "Structural Trap", "Porosity", "Lacustrine Environment", "Diagenetic Alteration"]

    ---
    **TEXT TO ANALYZE:**
    {chunk_text}
"""


    log.info(f"Loading papers from {LLM_INPUT_DIR}...")
    papers = load_papers_from_dir(LLM_INPUT_DIR)

    if papers:
        all_extracted_terms = []
        log.info(f"Found {len(papers)} papers to process.")

        for paper_text in tqdm(papers, desc="Extracting terms"):
            try:
                final_prompt = prompt_template.format(chunk_text=paper_text)

                response_text = generate(
                    final_prompt,
                    model=LLM_MODEL_NAME,
                    system_instruction=system_instruction,
                    temperature=LLM_MODEL_TEMPERATURE,
                    response_mime_type="application/json",
                )

                terms_from_paper = json.loads(response_text)
                all_extracted_terms.extend(terms_from_paper)

            except Exception as e:
                tqdm.write("")
                log.error(f"Processing paper: {e}")

        df_raw_results = pd.DataFrame(all_extracted_terms, columns=['Entity'])
        df_raw_results.to_csv(LLM_OUTPUT_FILE, index=False, encoding='utf-8-sig')

        log.success(f"{len(all_extracted_terms)} raw terms extracted -> '{LLM_OUTPUT_FILE}'")
