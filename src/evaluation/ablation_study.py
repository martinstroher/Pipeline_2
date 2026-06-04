"""
Ablation Study Runner — 4-condition ablation for PreSaltOntoLearn pipeline.

Runs Steps 4-5 (NLD generation + categorization) under 4 conditions on the SAME term set:
  A: Full pipeline    — RAG context -> NLD -> categorizer (Term + NLD)
  B: No RAG           — "No additional context available." -> NLD -> categorizer (Term + NLD)
  C: No NLD           — Skip NLD, categorizer receives only (Term), no "nld" field
  D: Raw RAG          — Skip NLD, categorizer receives (Term + raw RAG chunks)
"""

import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import pandas as pd

from src.utils.csv_io import read_csv, write_csv
from src.utils.checkpoint import Checkpoint
from dotenv import load_dotenv

load_dotenv()

from src.utils.rag_setup import setup_rag, get_relevant_documents
from src.modules.nld_generator import generate_nld, format_docs_for_context
from src.utils.gemini_client import get_client, generate as gemini_generate
from src.utils.prompt_loader import load_prompt
from src.utils.ontology_config import get_config


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
SLEEP_SECONDS = float(os.environ.get("NLD_SLEEP_SECONDS", 0))
MAX_CONCURRENT_NLD = int(os.environ.get("MAX_CONCURRENT_NLD", 5))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _nld_path(condition: str) -> str:
    return os.path.join(OUTPUT_DIR, f"nld_{condition}.csv")


def _cat_path(condition: str) -> str:
    return os.path.join(OUTPUT_DIR, f"cat_{condition}.csv")


# ---------------------------------------------------------------------------
# Generic NLD runner (shared by conditions A, B, D)
# ---------------------------------------------------------------------------

def _run_nld_generation(
    condition: str,
    terms: list[str],
    process_fn: Callable[[str], dict],
) -> pd.DataFrame:
    """Run NLD generation with checkpoint/resume and concurrent workers."""
    print(f"\n=== Condition {condition}: {CONDITION_LABELS[condition]} ===")
    path = _nld_path(condition)
    ckpt = Checkpoint(path)
    completed, rows = ckpt.load()
    if completed:
        print(f"  Resuming: {len(completed)} terms already done.")

    pending = [t for t in terms if t not in completed]
    if not pending:
        return pd.DataFrame(rows)

    total = len(terms)
    lock = threading.Lock()
    max_workers = min(MAX_CONCURRENT_NLD, len(pending))
    print(f"  Concurrent NLD: {max_workers} workers, {len(pending)} pending")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_fn, t): t for t in pending}
        for future in as_completed(futures):
            term = futures[future]
            try:
                row = future.result()
            except Exception as e:
                print(f"    ERROR: {e}")
                row = {"Term": term, "NLD": f"ERROR: {e}", "Context_Used": "Error", "Context": ""}
            with lock:
                rows.append(row)
                completed.add(term)
                ckpt.append(row, is_first=(len(rows) == 1))
            print(f"  [{len(completed)}/{total}] {condition}: '{term}' done")

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df
    print(f"  Condition {condition} NLD: {len(df)} terms -> {path}")
    return df


def _parse_nld_response(nld_json_str: str, with_context: bool = False) -> tuple[str, object]:
    """Parse LLM NLD JSON response into (nld_text, context_used)."""
    try:
        nld_data = json.loads(nld_json_str)
        nld = nld_data.get("Definition", nld_json_str)
        ctx_used = nld_data.get("Context_Used", with_context)
    except json.JSONDecodeError:
        nld = nld_json_str
        ctx_used = "Error" if with_context else False
    return nld, ctx_used


# ---------------------------------------------------------------------------
# Per-condition NLD runners
# ---------------------------------------------------------------------------

def run_condition_a(terms: list[str], vector_store, bm25_retriever) -> pd.DataFrame:
    """Full pipeline: RAG context -> NLD."""
    path = _nld_path("A")
    pipeline_nld = os.environ.get("NLD_OUTPUT", "output/4_nld_generated_definitions.csv")
    if not os.path.exists(path) and os.path.exists(pipeline_nld):
        pipeline_df = read_csv(pipeline_nld)
        if set(terms).issubset(set(pipeline_df["Term"].tolist())):
            write_csv(pipeline_df, path)
            print(f"  Reused pipeline output ({pipeline_nld}) as Condition A NLD.")
            return pipeline_df

    def _process(term):
        docs = get_relevant_documents(term, vector_store, bm25_retriever=bm25_retriever)
        context = format_docs_for_context(docs)
        nld_json_str, _ = generate_nld(term, context)
        nld, ctx_used = _parse_nld_response(nld_json_str, with_context=True)
        if SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)
        return {"Term": term, "NLD": nld, "Context_Used": ctx_used, "Context": context}

    return _run_nld_generation("A", terms, _process)


def run_condition_b(terms: list[str]) -> pd.DataFrame:
    """No RAG: generate NLD with parametric knowledge only."""
    def _process(term):
        nld_json_str, _ = generate_nld(term, "No additional context available.")
        nld, _ = _parse_nld_response(nld_json_str)
        if SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)
        return {"Term": term, "NLD": nld, "Context_Used": False, "Context": ""}

    return _run_nld_generation("B", terms, _process)


def run_condition_c(terms: list[str]) -> pd.DataFrame:
    """No NLD: placeholder rows (no LLM calls)."""
    print("\n=== Condition C: No NLD (Term only) ===")
    rows = [{"Term": t, "NLD": "", "Context_Used": False, "Context": ""} for t in terms]
    df = pd.DataFrame(rows)
    write_csv(df, _nld_path("C"))
    print(f"  Condition C placeholder NLDs: {len(df)} terms -> {_nld_path('C')}")
    return df


def run_condition_d(terms: list[str], vector_store, bm25_retriever) -> pd.DataFrame:
    """Raw RAG: retrieve context chunks (no NLD generation)."""
    def _process(term):
        try:
            docs = get_relevant_documents(term, vector_store, bm25_retriever=bm25_retriever)
            context = format_docs_for_context(docs)
        except Exception as e:
            context = f"ERROR: {e}"
        return {"Term": term, "NLD": context, "Context_Used": True, "Context": context}

    return _run_nld_generation("D", terms, _process)


# ---------------------------------------------------------------------------
# Categorizer (self-contained, supports all 4 conditions)
# ---------------------------------------------------------------------------

def _load_definitions() -> dict:
    cfg = get_config()
    return {
        "georeservoir": cfg.llm_definitions_block("georeservoir"),
        "geocore": cfg.llm_definitions_block("geocore"),
        "bfo": cfg.llm_definitions_block("bfo"),
    }


def _build_categorizer_prompt(defs: dict, is_raw_rag: bool = False):
    """Return (system_instruction, prompt_template) for the categorizer."""
    filename = "ablation_categorization_rag.txt" if is_raw_rag else "ablation_categorization_nld.txt"
    system_instruction, prompt_template = load_prompt(filename)
    prompt_template = prompt_template.format(
        georeservoir_definitions=defs["georeservoir"],
        geocore_definitions=defs["geocore"],
        bfo_definitions=defs["bfo"],
        json_batch="{json_batch}",
    )
    return system_instruction, prompt_template


def _categorize_batch(
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

        return [
            {
                "Term": batch_items[idx]["term"],
                "Category": item.get("category", "ERROR_PARSE"),
                "Reasoning": item.get("reasoning", ""),
            }
            for idx, item in enumerate(response_json)
        ]

    except json.JSONDecodeError:
        return [
            {"Term": i["term"], "Category": "ERROR_INVALID_JSON", "Reasoning": "LLM response was not valid JSON."}
            for i in batch_items
        ]
    except Exception as e:
        return [
            {"Term": i["term"], "Category": "ERROR_GENERAL", "Reasoning": f"Error: {e}"}
            for i in batch_items
        ]


def run_categorization(
    condition: str, nld_df: pd.DataFrame, defs: dict, batch_size: int = 5
) -> pd.DataFrame:
    """Run categorization for a given condition's NLD output."""
    print(f"\n--- Categorizing Condition {condition} ({CONDITION_LABELS[condition]}) ---")
    cat_csv = _cat_path(condition)

    # Reuse main pipeline categorization output for Condition A if available
    if condition == "A" and not os.path.exists(cat_csv):
        pipeline_cat = os.environ.get("CATEGORIZED_OUTPUT", "output/5_categorized_ontology.csv")
        if os.path.exists(pipeline_cat):
            pcat = read_csv(pipeline_cat)
            cat_a = pd.DataFrame({
                "Term": pcat["Term"], "Category": pcat["Category"],
                "Reasoning": pcat["Reasoning"], "NLD": pcat["NLD"],
                "Context_Used": pcat.get("RAG_Context_Used", ""),
                "Condition": "A",
            })
            write_csv(cat_a, cat_csv)
            print(f"  Reused pipeline output ({pipeline_cat}) as Condition A categorization.")
            return cat_a

    cat_ckpt = Checkpoint(cat_csv)
    completed, cat_rows = cat_ckpt.load()
    if completed:
        print(f"  Resuming: {len(completed)} terms already categorized.")

    is_raw_rag = condition == "D"
    sys_instr, prompt_tmpl = _build_categorizer_prompt(defs, is_raw_rag=is_raw_rag)
    model_name = os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    model_temp = float(os.environ.get("LLM_GENERATION_TEMPERATURE", 0))

    remaining = nld_df[~nld_df["Term"].isin(completed)]
    total = len(remaining)

    for i in range(0, total, batch_size):
        batch_df = remaining.iloc[i : i + batch_size]
        batch_items = []
        for _, row in batch_df.iterrows():
            if is_raw_rag:
                batch_items.append({"term": row["Term"], "context": row["NLD"]})
            elif condition == "C":
                batch_items.append({"term": row["Term"], "nld": ""})
            else:
                batch_items.append({"term": row["Term"], "nld": row["NLD"]})

        print(f"  Categorizing {i+1}-{min(i+batch_size, total)} of {total}...")
        results = _categorize_batch(batch_items, prompt_tmpl, sys_instr, model_name, model_temp)

        for result in results:
            nld_row = nld_df[nld_df["Term"] == result["Term"]].iloc[0]
            result["NLD"] = nld_row["NLD"]
            result["Context_Used"] = nld_row.get("Context_Used", "")
            result["Condition"] = condition
            cat_rows.append(result)
            cat_ckpt.append(result, is_first=not os.path.exists(cat_csv))

        time.sleep(2)

    df = pd.DataFrame(cat_rows)
    write_csv(df, cat_csv)
    print(f"  Condition {condition} categorized: {len(df)} terms -> {cat_csv}")
    return df


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_ablation(conditions: list[str] | None = None):
    """Run the full ablation study."""
    if conditions is None:
        conditions = CONDITIONS

    get_client()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load terms (Steps 1-3 output)
    terms_file = os.environ.get("FILTERED_TERMS_OUTPUT", "output/3_filtered_top_terms.csv")
    terms = read_csv(terms_file)["Readable_Term"].tolist()
    print(f"\nAblation study: {len(terms)} terms, conditions: {conditions}")

    # Setup RAG (needed for A and D)
    vector_store, bm25 = None, None
    if "A" in conditions or "D" in conditions:
        print("\nSetting up RAG infrastructure...")
        vector_store, bm25 = setup_rag()

    defs = _load_definitions()
    batch_size = int(os.environ.get("BATCH_SIZE", 5))

    # --- NLD generation per condition ---
    nld_results = {}

    if "A" in conditions:
        nld_results["A"] = run_condition_a(terms, vector_store, bm25)

    # C is instant (no LLM calls)
    if "C" in conditions:
        nld_results["C"] = run_condition_c(terms)

    # B and D are independent — run in parallel
    parallel_nld = {}
    if "B" in conditions:
        parallel_nld["B"] = lambda: run_condition_b(terms)
    if "D" in conditions:
        parallel_nld["D"] = lambda: run_condition_d(terms, vector_store, bm25)

    if parallel_nld:
        print(f"\n  Running NLD generation for conditions {list(parallel_nld.keys())} in parallel...")
        with ThreadPoolExecutor(max_workers=len(parallel_nld)) as executor:
            futures = {executor.submit(fn): cond for cond, fn in parallel_nld.items()}
            for future in as_completed(futures):
                cond = futures[future]
                try:
                    nld_results[cond] = future.result()
                except Exception as e:
                    print(f"  ERROR in condition {cond}: {e}")

    # --- Categorization per condition (parallel for all) ---
    cat_results = {}
    cat_conditions = [c for c in conditions if c in nld_results]
    if cat_conditions:
        print(f"\n  Running categorization for conditions {cat_conditions} in parallel...")
        with ThreadPoolExecutor(max_workers=len(cat_conditions)) as executor:
            futures = {
                executor.submit(run_categorization, cond, nld_results[cond], defs, batch_size): cond
                for cond in cat_conditions
            }
            for future in as_completed(futures):
                cond = futures[future]
                try:
                    cat_results[cond] = future.result()
                except Exception as e:
                    print(f"  ERROR categorizing condition {cond}: {e}")

    # --- Auto-include Condition A from pipeline output or checkpoint ---
    if "A" not in cat_results:
        cat_a_csv = _cat_path("A")
        pipeline_cat = os.environ.get("CATEGORIZED_OUTPUT", "output/5_categorized_ontology.csv")
        if os.path.exists(cat_a_csv):
            cat_results["A"] = read_csv(cat_a_csv)
            print(f"\n  Auto-included Condition A from checkpoint: {cat_a_csv}")
        elif os.path.exists(pipeline_cat):
            pcat = read_csv(pipeline_cat)
            cat_a = pd.DataFrame({
                "Term": pcat["Term"], "Category": pcat["Category"],
                "Reasoning": pcat["Reasoning"], "NLD": pcat["NLD"],
                "Context_Used": pcat.get("RAG_Context_Used", ""),
                "Condition": "A",
            })
            write_csv(cat_a, cat_a_csv)
            cat_results["A"] = cat_a
            print(f"\n  Auto-included Condition A from pipeline output: {pipeline_cat}")

    # --- Merge all results ---
    all_cat = []
    for cond, df in cat_results.items():
        df_copy = df.copy()
        df_copy["Condition"] = cond
        all_cat.append(df_copy)

    if all_cat:
        merged = pd.concat(all_cat, ignore_index=True)
        merged_path = os.path.join(OUTPUT_DIR, "ablation_merged.csv")
        write_csv(merged, merged_path)
        print(f"\n=== Ablation complete. Merged results: {merged_path} ===")

        for cond in sorted(cat_results.keys()):
            df_c = merged[merged["Condition"] == cond]
            n_classified = len(df_c[~df_c["Category"].str.startswith("ERROR")])
            n_not = len(df_c[df_c["Category"] == "NOT_CLASSIFIED"])
            n_err = len(df_c[df_c["Category"].str.startswith("ERROR")])
            print(
                f"  {cond} ({CONDITION_LABELS[cond]}): "
                f"{n_classified} classified, {n_not} NOT_CLASSIFIED, {n_err} errors"
            )

    return cat_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ablation Study Runner")
    parser.add_argument(
        "--conditions", type=str, default="A,B,C,D",
        help="Comma-separated conditions to run (default: A,B,C,D)",
    )
    args = parser.parse_args()
    conds = [c.strip().upper() for c in args.conditions.split(",")]
    run_ablation(conditions=conds)
