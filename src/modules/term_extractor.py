import json
import os

import pandas as pd
from tqdm import tqdm

from src.utils import log
from src.utils.gemini_client import generate
from src.utils.prompt_loader import load_prompt


def run_llm_term_extraction():
    LLM_MODEL_NAME = os.environ.get("LLM_EXTRACTION_MODEL", "gemini-2.5-pro")
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

    system_instruction, prompt_template = load_prompt("term_extraction.txt")


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
