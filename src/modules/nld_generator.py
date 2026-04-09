"""
Step 4 — NLD (Natural Language Definition) Generation.

For each filtered term, retrieves top-5 corpus chunks via hybrid RAG,
then generates an Aristotelian NLD ("X is a Y that Z") using an LLM.
Supports concurrent generation with checkpoint/resume.
"""

import json
import os
import threading
import time

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.utils import log
from src.utils.gemini_client import generate
from src.utils.rag_setup import get_relevant_documents, load_vector_store

MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_NLD", 5))

_SYSTEM_INSTRUCTION = (
    "You are a senior geoscientist and ontology engineer. Your expertise spans "
    "oil and gas exploration geology, structural geology, stratigraphy, and "
    "petroleum systems, with a specific focus on the carbonate reservoirs of "
    "the Brazilian Pre-Salt."
)

_PROMPT_TEMPLATE = """Generate a concise and precise Natural Language Definition (NLD) for the provided geological term.

Mandatory Instructions:
1. The FIRST sentence MUST follow the Aristotelian pattern "X is a Y that Z", where Y is the proximate genus and Z is the differentia. Up to two additional sentences may provide domain-specific elaboration. Minimum two sentences total.
2. The definition MUST be written in English.
3. If a term is polysemous, define the sense most relevant to Pre-Salt petroleum geology.
4. Primary Knowledge Source: Base the definition PRIMARILY on the provided context, as it contains the most up-to-date and domain-specific knowledge. Use your internal knowledge of Brazilian Pre-Salt geology only to structure the definition correctly, fill in minor conceptual gaps, or if the provided context does not define the term geologically. If the context section contains no geologically relevant information about the term (including when it reads 'No additional context available.'), rely entirely on your domain expertise and set Context_Used to false.
5. Output ONLY a valid JSON object with exactly two keys:
   - "Definition": strictly the string containing the generated NLD.
   - "Context_Used": boolean (true if the provided context was relevant and used as the primary source, false if you had to fallback entirely to internal knowledge).

**EXAMPLES:**
Term: "Grainstone"
Context: [paper.md > Carbonate Classification] Grainstones are grain-supported carbonate rocks lacking mud matrix...
Output: {{"Definition": "Grainstone is a grain-supported sedimentary carbonate rock that is characterized by the absence of micrite matrix, with grains typically consisting of allochems such as bivalves, ostracods, gastropods, ooids, or intraclasts.", "Context_Used": true}}

Term: "Coquina"
Context: [paper.md > Reservoir Facies] Coquinas from the Santos Basin are bioclastic carbonates composed predominantly of bivalve shells...
Output: {{"Definition": "Coquina is a sedimentary rock that is primarily composed of accumulated mollusk shells (bivalves and gastropods) and their fragments, deposited in brackish to saline lacustrine environments, commonly occurring as fragmented shell beds and serving as a significant reservoir rock type.", "Context_Used": true}}

Term: "Dolomitization"
Context: No additional context available.
Output: {{"Definition": "Dolomitization is a diagenetic process that replaces calcium carbonate minerals with dolomite through the substitution of calcium ions by magnesium ions from Mg-rich fluids, commonly occurring in burial or hydrothermal settings and frequently enhancing reservoir porosity and permeability.", "Context_Used": false}}

---
Term to be defined: "{term}"

Relevant context:
{context}
"""


def format_docs_for_context(docs) -> str:
    """Format retrieved documents into a context string with source breadcrumbs."""
    if not docs:
        return "No additional context available."
    parts = []
    for item in docs:
        doc = item[0] if isinstance(item, tuple) else item
        meta = doc.metadata
        source = os.path.basename(meta.get("source", "Unknown"))
        headers = [meta.get(f"Header {i}", "") for i in (1, 2, 3)]
        breadcrumb = " > ".join(filter(None, [source] + headers))
        parts.append(f"[{breadcrumb}]\n{doc.page_content}")
    return "\n\n".join(parts)


def generate_nld(term: str, context: str) -> tuple[str, str]:
    """Generate an NLD for a single term. Returns (response_json, full_prompt)."""
    prompt = _PROMPT_TEMPLATE.format(term=term, context=context)
    response = generate(
        prompt,
        system_instruction=_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
    )
    return response.strip(), prompt


def _load_checkpoint(path: str) -> tuple[set, list]:
    """Load completed terms and rows from a checkpoint CSV."""
    if not os.path.exists(path):
        return set(), []
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        return set(df["Term"].tolist()), df.to_dict("records")
    except Exception:
        return set(), []


def run_nld_generation(vector_store=None, bm25_retriever=None):
    """Run NLD generation for all filtered terms with concurrent processing."""
    input_file = os.environ.get("FILTERED_TERMS_OUTPUT")
    output_file = os.environ.get("CONSOLIDATED_LLM_RESULTS_WITH_NLDS")
    failure_file = os.environ.get("OUTPUT_FAILURE_FILE")

    for var, name in [(input_file, "FILTERED_TERMS_OUTPUT"),
                      (output_file, "CONSOLIDATED_LLM_RESULTS_WITH_NLDS"),
                      (failure_file, "OUTPUT_FAILURE_FILE")]:
        if not var:
            raise RuntimeError(f"Environment variable {name} is not set.")

    # Load terms
    df_terms = pd.read_csv(input_file, encoding="utf-8-sig", usecols=["Readable_Term"])
    all_terms = df_terms["Readable_Term"].tolist()
    log.info(f"{len(all_terms)} terms loaded from '{input_file}'.")

    # Load checkpoint
    completed, results = _load_checkpoint(output_file)
    if completed:
        log.info(f"Resuming: {len(completed)} terms already processed.")

    # Setup RAG
    if vector_store is None:
        log.warn("No vector_store passed, loading from disk (dense-only).")
        vector_store = load_vector_store()
    log.info(f"BM25 hybrid retrieval: {'ACTIVE' if bm25_retriever else 'OFF (dense-only)'}")

    pending = [t for t in all_terms if t not in completed]
    sleep_secs = float(os.environ.get("NLD_SLEEP_SECONDS", 0))
    max_workers = min(MAX_CONCURRENT, len(pending)) if pending else 1
    log.info(f"Concurrent NLD: {max_workers} workers, {len(pending)} pending / {len(all_terms)} total")

    lock = threading.Lock()
    errors = []

    def _process(term):
        docs = get_relevant_documents(term, vector_store, bm25_retriever=bm25_retriever)
        context = format_docs_for_context(docs)
        nld_json, _ = generate_nld(term, context)
        try:
            data = json.loads(nld_json)
            nld = data.get("Definition", "")
            ctx_used = data.get("Context_Used", True)
        except json.JSONDecodeError:
            nld, ctx_used = nld_json, False
        if sleep_secs > 0:
            time.sleep(sleep_secs)
        return {"Term": term, "NLD": nld, "Context_Used": ctx_used, "Context": context}

    pbar = tqdm(total=len(all_terms), desc="Generating NLDs",
                initial=len(all_terms) - len(pending))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, t): t for t in pending}
        for future in as_completed(futures):
            term = futures[future]
            try:
                row = future.result()
                with lock:
                    results.append(row)
                    pd.DataFrame([row]).to_csv(
                        output_file, mode="a",
                        header=(len(results) == 1), index=False, encoding="utf-8-sig",
                    )
                pbar.set_postfix_str(term[:30])
            except Exception as e:
                log.error(f"Term '{term}': {e}")
                errors.append({"Term": term, "Error": str(e)})
            pbar.update(1)
    pbar.close()

    # Write final consolidated output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(results).to_csv(output_file, index=False, encoding="utf-8-sig")
    log.success(f"{len(results)} definitions saved to '{output_file}'")

    if errors:
        os.makedirs(os.path.dirname(failure_file), exist_ok=True)
        pd.DataFrame(errors).to_csv(failure_file, index=False, encoding="utf-8-sig")
        log.warn(f"{len(errors)} terms need review -> '{failure_file}'")
