import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    N_WORKERS = int(os.environ.get("EXTRACTION_WORKERS", 5))

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
                    papers.append((filename, content))
                log.detail(f"Loaded: {filename} ({len(content)} chars)")
            except Exception as e:
                log.error(f"Reading {filename}: {e}")
        return papers

    system_instruction, prompt_template = load_prompt("term_extraction.txt")

    def _extract_one(paper_tuple):
        filename, paper_text = paper_tuple
        final_prompt = prompt_template.format(chunk_text=paper_text)
        response_text = generate(
            final_prompt,
            model=LLM_MODEL_NAME,
            system_instruction=system_instruction,
            temperature=LLM_MODEL_TEMPERATURE,
            response_mime_type="application/json",
        )
        return filename, json.loads(response_text)

    log.info(f"Loading papers from {LLM_INPUT_DIR}...")
    papers = load_papers_from_dir(LLM_INPUT_DIR)

    if papers:
        all_extracted_terms = []
        log.info(f"Found {len(papers)} papers to process.")
        log.info(f"Concurrent extraction: {N_WORKERS} workers")

        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(_extract_one, p): p[0] for p in papers}
            with tqdm(total=len(papers), desc="Extracting terms") as pbar:
                for future in as_completed(futures):
                    fname = futures[future]
                    try:
                        _, terms_from_paper = future.result()
                        all_extracted_terms.extend(terms_from_paper)
                        pbar.set_postfix_str(fname[:40])
                    except Exception as e:
                        tqdm.write("")
                        log.error(f"Processing {fname}: {e}")
                    pbar.update(1)

        df_raw_results = pd.DataFrame(all_extracted_terms, columns=['Entity'])
        df_raw_results.to_csv(LLM_OUTPUT_FILE, index=False, encoding='utf-8-sig')

        log.success(f"{len(all_extracted_terms)} raw terms extracted -> '{LLM_OUTPUT_FILE}'")
