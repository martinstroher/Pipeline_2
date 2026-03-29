"""
Ablation Study Runner — 4-condition ablation for PreSaltOntoLearn pipeline.

Runs Steps 4-5 (NLD generation + categorization) under 4 conditions on the SAME term set:
  A: Full pipeline    — RAG context -> NLD -> categorizer (Term + NLD)
  B: No RAG           — "No additional context available." -> NLD -> categorizer (Term + NLD)
  C: No NLD           — Skip NLD, categorizer receives (Term + "No definition available.")
  D: Raw RAG          — Skip NLD, categorizer receives (Term + raw RAG chunks)
"""

import json
import os
import time
import argparse

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.utils.rag_setup import setup_rag, get_relevant_documents
from src.modules.nld_generator import (
    generate_nld,
    format_docs_for_context,
    _ensure_genai_configured,
)
from src.utils.gemini_client import generate as gemini_generate


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONDITIONS = ["A", "B", "C", "D"]
CONDITION_LABELS = {
    "A": "Full Pipeline (RAG+NLD)",
    "B": "No RAG (NLD only)",
    "C": "No NLD (Term only)",
    "D": "Raw RAG (no NLD)",
}

OUTPUT_DIR = os.environ.get("ABLATION_OUTPUT_DIR", "output/ablation")
SLEEP_SECONDS = float(os.environ.get("NLD_SLEEP_SECONDS", 4))


# ---------------------------------------------------------------------------
# Categorizer (self-contained, supports all 4 conditions)
# ---------------------------------------------------------------------------

def _load_definitions():
    """Load ontology definition files."""
    paths = {
        "georeservoir": os.environ["GEORESERVOIR_DEFS_PATH"],
        "geocore": os.environ["GEOCORE_DEFS_PATH"],
        "bfo": os.environ["BFO_DEFS_PATH"],
    }
    defs = {}
    for key, path in paths.items():
        with open(path, "r", encoding="utf-8") as f:
            defs[key] = f.read()
    return defs


def _build_categorizer_prompt(defs: dict, is_raw_rag: bool = False):
    """Return (system_instruction, prompt_template) for the categorizer."""
    system_instruction = (
        "You are an expert ontology engineer specializing in foundational (BFO) "
        "and geological (GeoCore and GeoReservoir) ontologies. "
        "You process data in batches and your response format MUST be a valid JSON array of objects."
    )

    if is_raw_rag:
        data_description = (
            '**INPUT:** A JSON array of objects, where each object has a "term" and '
            '"context" field (raw corpus passages retrieved via RAG).\n'
            "    -   **OUTPUT:** Your response MUST BE a valid JSON array. Each object "
            'must contain the "term", the assigned "category", and a "reasoning" string.'
        )
        data_instruction = (
            "1.  **Analyze Data:** Read the Term and its retrieved corpus context.\n"
            "    2.  **Classify** the term based on the corpus context provided."
        )
    else:
        data_description = (
            '**INPUT:** A JSON array of objects, where each object has a "term" and '
            '"nld" field.\n'
            "    -   **OUTPUT:** Your response MUST BE a valid JSON array. Each object "
            'must contain the "term", the assigned "category", and a "reasoning" string.'
        )
        data_instruction = (
            "1.  **Analyze Data:** Read the Term and its NLD.\n"
            "    2.  **Classify** the term based on its Natural Language Definition."
        )

    prompt_template = f"""Your task is to classify a batch of geological terms.

    **METHODOLOGY (Follow Strictly for each item):**
    {data_instruction}
    3.  **Prioritize GeoReservoir:** First, attempt to classify the term into one of the `### GeoReservoir Categories`.
    4.  **Fallback to GeoCore:** If and only if no GeoReservoir category is a good fit, then attempt to classify it into one of the `### GeoCore Categories`.
    5.  **Fallback to BFO:** If and only if no GeoCore category fits, then attempt to classify it into one of the `### BFO Categories`.
    6.  **Final Fallback:** If the term does not fit well into ANY of the provided categories (GeoReservoir, GeoCore, or BFO), you MUST use the string `NOT_CLASSIFIED`.
    7.  **Provide Reasoning:** In one short sentence, explain WHY you chose that category.

    **INPUT/OUTPUT FORMAT:**
    -   {data_description}

    ---
    **ONTOLOGY CATEGORIES REFERENCE:**

    ### GeoReservoir Categories:
    {defs["georeservoir"]}

    ### GeoCore Categories:
    {defs["geocore"]}

    ### BFO Categories:
    {defs["bfo"]}

    ---
    **DATA TO CLASSIFY:**
    {{json_batch}}
    """
    return system_instruction, prompt_template


def categorize_batch(
    batch_items: list[dict],
    prompt_template: str,
    system_instruction: str,
    model_name: str,
    model_temperature: float,
) -> list[dict]:
    """Classify a batch of terms. Returns list of result dicts."""
    json_batch_str = json.dumps(batch_items, indent=2)
    final_prompt = prompt_template.format(json_batch=json_batch_str)

    try:
        response_text = gemini_generate(
            final_prompt,
            model=model_name,
            system_instruction=system_instruction,
            temperature=model_temperature,
            response_mime_type="application/json",
        )
        response_json = json.loads(response_text)

        if len(response_json) != len(batch_items):
            raise ValueError(
                f"LLM response length ({len(response_json)}) != batch size ({len(batch_items)})"
            )

        results = []
        for idx, result_item in enumerate(response_json):
            results.append({
                "Term": batch_items[idx]["term"],
                "Category": result_item.get("category", "ERROR_PARSE"),
                "Reasoning": result_item.get("reasoning", ""),
            })
        return results

    except json.JSONDecodeError:
        return [
            {
                "Term": item["term"],
                "Category": "ERROR_INVALID_JSON",
                "Reasoning": "LLM response was not valid JSON.",
            }
            for item in batch_items
        ]
    except Exception as e:
        return [
            {
                "Term": item["term"],
                "Category": "ERROR_GENERAL",
                "Reasoning": f"Error: {str(e)}",
            }
            for item in batch_items
        ]


# ---------------------------------------------------------------------------
# Per-condition runners
# ---------------------------------------------------------------------------

def _checkpoint_path(condition: str) -> str:
    return os.path.join(OUTPUT_DIR, f"nld_{condition}.csv")


def _cat_checkpoint_path(condition: str) -> str:
    return os.path.join(OUTPUT_DIR, f"cat_{condition}.csv")


def _load_checkpoint(path: str) -> tuple[set, list[dict]]:
    """Load checkpoint CSV if it exists. Returns (completed_terms, rows)."""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
            return set(df["Term"].tolist()), df.to_dict("records")
        except Exception:
            pass
    return set(), []


def _append_row(path: str, row: dict, is_first: bool):
    """Atomic single-row append to CSV."""
    pd.DataFrame([row]).to_csv(
        path, mode="a", header=is_first, index=False, encoding="utf-8-sig"
    )


def run_condition_a(terms: list[str], vector_store, bm25_retriever) -> pd.DataFrame:
    """Full pipeline: RAG context -> NLD -> categorize(term + NLD)."""
    print("\n=== Condition A: Full Pipeline (RAG + NLD) ===")
    nld_path = _checkpoint_path("A")
    completed, nld_rows = _load_checkpoint(nld_path)
    if completed:
        print(f"  Resuming: {len(completed)} terms already done.")

    total = len(terms)
    for i, term in enumerate(terms):
        if term in completed:
            continue
        print(f"  [{i+1}/{total}] A: NLD for '{term}'...")
        try:
            docs = get_relevant_documents(term, vector_store, bm25_retriever=bm25_retriever)
            context = format_docs_for_context(docs)
            nld_json_str, _ = generate_nld(term, context)
            nld_data = json.loads(nld_json_str)
            nld = nld_data.get("Definition", nld_json_str)
            ctx_used = nld_data.get("Context_Used", True)
        except json.JSONDecodeError:
            nld = nld_json_str
            ctx_used = "Error"
        except Exception as e:
            print(f"    ERROR: {e}")
            nld = f"ERROR: {e}"
            ctx_used = "Error"

        row = {"Term": term, "NLD": nld, "Context_Used": ctx_used, "Context": context if 'context' in dir() else ""}
        nld_rows.append(row)
        completed.add(term)
        _append_row(nld_path, row, len(nld_rows) == 1)
        time.sleep(SLEEP_SECONDS)

    df = pd.DataFrame(nld_rows)
    df.to_csv(nld_path, index=False, encoding="utf-8-sig")
    print(f"  Condition A NLD: {len(df)} terms -> {nld_path}")
    return df


def run_condition_b(terms: list[str]) -> pd.DataFrame:
    """No RAG: generate NLD with 'No additional context available.'"""
    print("\n=== Condition B: No RAG (Parametric NLD only) ===")
    nld_path = _checkpoint_path("B")
    completed, nld_rows = _load_checkpoint(nld_path)
    if completed:
        print(f"  Resuming: {len(completed)} terms already done.")

    total = len(terms)
    for i, term in enumerate(terms):
        if term in completed:
            continue
        print(f"  [{i+1}/{total}] B: NLD for '{term}'...")
        try:
            nld_json_str, _ = generate_nld(term, "No additional context available.")
            nld_data = json.loads(nld_json_str)
            nld = nld_data.get("Definition", nld_json_str)
            ctx_used = False
        except json.JSONDecodeError:
            nld = nld_json_str
            ctx_used = "Error"
        except Exception as e:
            print(f"    ERROR: {e}")
            nld = f"ERROR: {e}"
            ctx_used = "Error"

        row = {"Term": term, "NLD": nld, "Context_Used": ctx_used, "Context": ""}
        nld_rows.append(row)
        completed.add(term)
        _append_row(nld_path, row, len(nld_rows) == 1)
        time.sleep(SLEEP_SECONDS)

    df = pd.DataFrame(nld_rows)
    df.to_csv(nld_path, index=False, encoding="utf-8-sig")
    print(f"  Condition B NLD: {len(df)} terms -> {nld_path}")
    return df


def run_condition_c(terms: list[str]) -> pd.DataFrame:
    """No NLD: categorizer receives 'No definition available.'"""
    print("\n=== Condition C: No NLD (Term only) ===")
    rows = [{"Term": t, "NLD": "No definition available.", "Context_Used": False, "Context": ""} for t in terms]
    df = pd.DataFrame(rows)
    path = _checkpoint_path("C")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Condition C placeholder NLDs: {len(df)} terms -> {path}")
    return df


def run_condition_d(terms: list[str], vector_store, bm25_retriever) -> pd.DataFrame:
    """Raw RAG: skip NLD, feed raw chunks to categorizer."""
    print("\n=== Condition D: Raw RAG (no NLD) ===")
    nld_path = _checkpoint_path("D")
    completed, nld_rows = _load_checkpoint(nld_path)
    if completed:
        print(f"  Resuming: {len(completed)} terms already done.")

    total = len(terms)
    for i, term in enumerate(terms):
        if term in completed:
            continue
        print(f"  [{i+1}/{total}] D: Retrieving RAG for '{term}'...")
        try:
            docs = get_relevant_documents(term, vector_store, bm25_retriever=bm25_retriever)
            context = format_docs_for_context(docs)
        except Exception as e:
            print(f"    ERROR: {e}")
            context = f"ERROR: {e}"

        # For condition D, the "NLD" column stores raw RAG context (used as categorizer input)
        row = {"Term": term, "NLD": context, "Context_Used": True, "Context": context}
        nld_rows.append(row)
        completed.add(term)
        _append_row(nld_path, row, len(nld_rows) == 1)

    df = pd.DataFrame(nld_rows)
    df.to_csv(nld_path, index=False, encoding="utf-8-sig")
    print(f"  Condition D RAG context: {len(df)} terms -> {nld_path}")
    return df


def run_categorization(
    condition: str, nld_df: pd.DataFrame, defs: dict, batch_size: int = 5
) -> pd.DataFrame:
    """Run categorization for a given condition's NLD output."""
    print(f"\n--- Categorizing Condition {condition} ({CONDITION_LABELS[condition]}) ---")

    cat_path = _cat_checkpoint_path(condition)
    completed, cat_rows = _load_checkpoint(cat_path)
    if completed:
        print(f"  Resuming: {len(completed)} terms already categorized.")

    is_raw_rag = condition == "D"
    sys_instr, prompt_tmpl = _build_categorizer_prompt(defs, is_raw_rag=is_raw_rag)

    MODEL_NAME = os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    MODEL_TEMPERATURE = float(os.environ.get("LLM_GENERATION_TEMPERATURE", 0))

    # Filter to uncategorized terms
    remaining_df = nld_df[~nld_df["Term"].isin(completed)]
    total = len(remaining_df)

    for i in range(0, total, batch_size):
        batch_df = remaining_df.iloc[i : i + batch_size]
        batch_items = []
        for _, row in batch_df.iterrows():
            if is_raw_rag:
                batch_items.append({"term": row["Term"], "context": row["NLD"]})
            else:
                batch_items.append({"term": row["Term"], "nld": row["NLD"]})

        print(f"  Categorizing {i+1}-{min(i+batch_size, total)} of {total}...")
        results = categorize_batch(batch_items, prompt_tmpl, sys_instr, MODEL_NAME, MODEL_TEMPERATURE)

        for result in results:
            # Enrich with NLD/context info from the nld_df
            nld_row = nld_df[nld_df["Term"] == result["Term"]].iloc[0]
            result["NLD"] = nld_row["NLD"]
            result["Context_Used"] = nld_row.get("Context_Used", "")
            result["Condition"] = condition
            cat_rows.append(result)
            _append_row(cat_path, result, len(cat_rows) == len(results) and i == 0)

        time.sleep(2)

    df = pd.DataFrame(cat_rows)
    df.to_csv(cat_path, index=False, encoding="utf-8-sig")
    print(f"  Condition {condition} categorized: {len(df)} terms -> {cat_path}")
    return df


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_ablation(conditions: list[str] | None = None):
    """
    Run the full ablation study.

    Args:
        conditions: list of condition letters to run (default: all 4).
                    Useful for resuming specific conditions.
    """
    if conditions is None:
        conditions = CONDITIONS

    _ensure_genai_configured()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load terms (Steps 1-3 output)
    terms_file = os.environ.get("FILTERED_TERMS_OUTPUT", "output/3_filtered_top_terms.csv")
    df_terms = pd.read_csv(terms_file, encoding="utf-8")
    terms = df_terms["Readable_Term"].tolist()
    print(f"\nAblation study: {len(terms)} terms, conditions: {conditions}")

    # Setup RAG (needed for A and D)
    vector_store, bm25 = None, None
    if "A" in conditions or "D" in conditions:
        print("\nSetting up RAG infrastructure...")
        vector_store, bm25 = setup_rag()

    defs = _load_definitions()
    batch_size = int(os.environ.get("BATCH_SIZE", 5))

    # --- Run NLD generation per condition ---
    nld_results = {}

    if "A" in conditions:
        nld_results["A"] = run_condition_a(terms, vector_store, bm25)

    if "B" in conditions:
        nld_results["B"] = run_condition_b(terms)

    if "C" in conditions:
        nld_results["C"] = run_condition_c(terms)

    if "D" in conditions:
        nld_results["D"] = run_condition_d(terms, vector_store, bm25)

    # --- Run categorization per condition ---
    cat_results = {}
    for cond in conditions:
        if cond in nld_results:
            cat_results[cond] = run_categorization(
                cond, nld_results[cond], defs, batch_size
            )

    # --- Merge all results into a single analysis-ready file ---
    all_cat = []
    for cond, df in cat_results.items():
        df_copy = df.copy()
        df_copy["Condition"] = cond
        all_cat.append(df_copy)

    if all_cat:
        merged = pd.concat(all_cat, ignore_index=True)
        merged_path = os.path.join(OUTPUT_DIR, "ablation_merged.csv")
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        print(f"\n=== Ablation complete. Merged results: {merged_path} ===")

        # Quick summary
        for cond in conditions:
            df_c = merged[merged["Condition"] == cond]
            n_classified = len(df_c[~df_c["Category"].str.startswith("ERROR")])
            n_not = len(df_c[df_c["Category"] == "NOT_CLASSIFIED"])
            n_err = len(df_c[df_c["Category"].str.startswith("ERROR")])
            print(
                f"  {cond} ({CONDITION_LABELS[cond]}): "
                f"{n_classified} classified, {n_not} NOT_CLASSIFIED, {n_err} errors"
            )

    return cat_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--conditions",
        type=str,
        default="A,B,C,D",
        help="Comma-separated list of conditions to run (default: A,B,C,D)",
    )
    args = parser.parse_args()
    conds = [c.strip().upper() for c in args.conditions.split(",")]
    run_ablation(conditions=conds)
