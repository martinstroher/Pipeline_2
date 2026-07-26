"""
Ablation Study Runner — 4-condition ablation for PreSaltOntoLearn pipeline.

Runs Steps 4-5 (NLD generation + categorization) under 4 conditions on the SAME term set:
  A: Full pipeline    — RAG context -> NLD -> categorizer (Term + NLD)
  B: No RAG           — "No additional context available." -> NLD -> categorizer (Term + NLD)
    C: No NLD           — Skip NLD, categorizer receives (Term + empty NLD field)
    D: Raw RAG          — Skip NLD, categorizer receives A's exact stored RAG chunks
"""

import json
import os
import hashlib
import shutil
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from src.utils.csv_io import read_csv, write_csv
from src.utils.checkpoint import Checkpoint
from dotenv import load_dotenv

load_dotenv()

from src.modules.define.nld_generator import generate_nld
from src.utils.llm_client import get_client, generate as llm_generate, parse_json_array
from evaluation_study.paths import (
    ABLATION_OUTPUT,
    FILTERED_TERMS,
    FROZEN_A_CATEGORIES,
    FROZEN_A_NLD,
)
from evaluation_study.prompt_loader import load_prompt
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

OUTPUT_DIR = os.environ.get("ABLATION_OUTPUT_DIR", str(ABLATION_OUTPUT))
SLEEP_SECONDS = float(os.environ.get("NLD_SLEEP_SECONDS", 0))
ABLATION_WORKERS = 3
EXPECTED_REASONING_EFFORT = "high"
EXPECTED_SEED = 42


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _nld_path(condition: str) -> str:
    return os.path.join(OUTPUT_DIR, f"nld_{condition}.csv")


def _cat_path(condition: str) -> str:
    return os.path.join(OUTPUT_DIR, f"cat_{condition}.csv")


def _copy_frozen_artifact(source: str, destination: str) -> pd.DataFrame:
    """Copy a production artifact byte-for-byte into the ablation directory."""
    if not os.path.exists(source):
        raise FileNotFoundError(f"Frozen Condition A artifact not found: {source}")
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    if os.path.abspath(source) != os.path.abspath(destination):
        shutil.copy2(source, destination)
        if _sha256_file(source) != _sha256_file(destination):
            raise OSError(f"Frozen artifact checksum mismatch after copy: {source}")
    return read_csv(destination)


def _validate_term_set(df: pd.DataFrame, terms: list[str], label: str) -> None:
    """Fail when an artifact is not a complete one-row-per-term matrix."""
    if "Term" not in df.columns:
        raise ValueError(f"{label}: missing Term column")
    observed = df["Term"].astype(str)
    duplicates = sorted(set(observed[observed.duplicated(keep=False)].tolist()))
    if duplicates:
        raise ValueError(f"{label}: duplicate terms: {duplicates[:10]}")
    expected_set = set(terms)
    observed_set = set(observed)
    if observed_set != expected_set:
        missing = sorted(expected_set - observed_set)
        extra = sorted(observed_set - expected_set)
        raise ValueError(
            f"{label}: term-set mismatch (missing={missing[:10]}, extra={extra[:10]})"
        )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _term_set_hash(terms: list[str]) -> str:
    return _sha256_text("\n".join(sorted(str(term).strip() for term in terms)))


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _validate_nld_output(df: pd.DataFrame, terms: list[str], label: str) -> None:
    _validate_term_set(df, terms, label)
    missing = {"NLD", "Context_Used", "Context"} - set(df.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    errors = df["NLD"].fillna("").astype(str).str.startswith("ERROR")
    if errors.any():
        raise ValueError(f"{label}: {int(errors.sum())} ERROR rows")


def _valid_categories() -> set[str]:
    cfg = get_config()
    return {
        label
        for ontology_key in cfg.waterfall_ontologies()
        for label in cfg.categories_for(ontology_key)
    } | {"NOT_CLASSIFIED"}


def _category_to_tier() -> dict[str, str]:
    cfg = get_config()
    mapping = {"NOT_CLASSIFIED": "NOT_CLASSIFIED"}
    for ontology_key in cfg.waterfall_ontologies():
        tier = cfg.ontologies[ontology_key].eval_tier.upper()
        for category in cfg.categories_for(ontology_key):
            mapping[category] = tier
    return mapping


def _validate_category_output(df: pd.DataFrame, terms: list[str], label: str) -> None:
    _validate_term_set(df, terms, label)
    if "Category" not in df.columns:
        raise ValueError(f"{label}: missing Category column")
    categories = df["Category"].fillna("").astype(str)
    errors = categories.str.startswith("ERROR")
    invalid = sorted(set(categories) - _valid_categories())
    if errors.any() or invalid:
        raise ValueError(
            f"{label}: invalid output (errors={int(errors.sum())}, invalid={invalid[:10]})"
        )


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
    max_workers = min(ABLATION_WORKERS, len(pending))
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
    write_csv(df, path)
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

def run_condition_a(terms: list[str]) -> pd.DataFrame:
    """Materialize frozen production A without regenerating or rewriting it."""
    source = os.environ.get(
        "ABLATION_FROZEN_A_NLD",
        os.environ.get("CONSOLIDATED_LLM_RESULTS_WITH_NLDS", str(FROZEN_A_NLD)),
    )
    df = _copy_frozen_artifact(source, _nld_path("A"))
    _validate_term_set(df, terms, "Condition A NLD")
    required = {"NLD", "Context_Used", "Context"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Condition A NLD: missing columns {sorted(missing)}")
    print(f"  Frozen Condition A NLD copied byte-for-byte from {source}.")
    return df


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


def run_condition_d(terms: list[str], condition_a: pd.DataFrame) -> pd.DataFrame:
    """Raw RAG: reuse the exact stored chunks supplied to frozen A."""
    _validate_term_set(condition_a, terms, "Condition A NLD source for D")
    if "Context" not in condition_a.columns:
        raise ValueError("Condition A NLD source for D: missing Context column")
    if condition_a["Context"].isna().any():
        missing_terms = condition_a.loc[condition_a["Context"].isna(), "Term"].tolist()
        raise ValueError(
            f"Condition A NLD source for D has missing stored contexts: {missing_terms[:10]}"
        )
    df = condition_a[["Term", "Context_Used", "Context"]].copy()
    df["NLD"] = df["Context"]
    df = df[["Term", "NLD", "Context_Used", "Context"]]
    write_csv(df, _nld_path("D"))
    print(f"  Condition D reused frozen A contexts for {len(df)} terms (no retrieval).")
    return df


# ---------------------------------------------------------------------------
# Categorizer (self-contained, supports all 4 conditions)
# ---------------------------------------------------------------------------

def _build_categorizer_prompt(is_raw_rag: bool = False):
    """Return (system_instruction, prompt_template) for the categorizer."""
    filename = "ablation_categorization_rag.txt" if is_raw_rag else "term_categorization.txt"
    system_instruction, prompt_template = load_prompt(filename)
    prompt_template = prompt_template.format(
        categories_block=get_config().categorization_block(),
        json_batch="{json_batch}",
    )
    return system_instruction, prompt_template


def _render_categorizer_prompt(
    prompt_template: str,
    batch_items: list[dict],
) -> str:
    """Insert one batch without re-formatting embedded JSON examples."""
    marker = "{json_batch}"
    if prompt_template.count(marker) != 1:
        raise ValueError("Categorizer prompt must contain exactly one {json_batch} marker")
    return prompt_template.replace(marker, json.dumps(batch_items, indent=2))


def _categorize_batch(
    batch_items: list[dict],
    prompt_template: str,
    system_instruction: str,
    model_name: str,
    model_temperature: float,
) -> list[dict]:
    """Classify a batch of terms. Returns list of result dicts."""
    final_prompt = _render_categorizer_prompt(prompt_template, batch_items)

    try:
        response_text = llm_generate(
            final_prompt,
            model=model_name,
            system_instruction=system_instruction,
            temperature=model_temperature,
            response_mime_type="application/json",
        )
        response_json = parse_json_array(response_text)

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
    condition: str, nld_df: pd.DataFrame, batch_size: int = 5
) -> pd.DataFrame:
    """Run categorization for a given condition's NLD output."""
    print(f"\n--- Categorizing Condition {condition} ({CONDITION_LABELS[condition]}) ---")
    cat_csv = _cat_path(condition)

    if condition == "A":
        source = os.environ.get(
            "ABLATION_FROZEN_A_CATEGORY",
            os.environ.get("CATEGORIZED_LLM_TERMS", str(FROZEN_A_CATEGORIES)),
        )
        df = _copy_frozen_artifact(source, cat_csv)
        _validate_category_output(df, nld_df["Term"].astype(str).tolist(), "Condition A")
        print(f"  Frozen Condition A categorization copied byte-for-byte from {source}.")
        return df

    cat_ckpt = Checkpoint(cat_csv)
    completed, cat_rows = cat_ckpt.load()
    if completed:
        print(f"  Resuming: {len(completed)} terms already categorized.")

    is_raw_rag = condition == "D"
    sys_instr, prompt_tmpl = _build_categorizer_prompt(is_raw_rag=is_raw_rag)
    model_name = os.environ.get("LLM_GENERATION_MODEL", "gpt-5.4")
    model_temp = float(os.environ.get("LLM_GENERATION_TEMPERATURE", 0))

    remaining = nld_df[~nld_df["Term"].isin(completed)]
    total = len(remaining)

    batches = [
        remaining.iloc[i : i + batch_size]
        for i in range(0, total, batch_size)
    ]
    lock = threading.Lock()

    def _process_batch(batch_df: pd.DataFrame) -> list[dict]:
        batch_items = []
        for _, row in batch_df.iterrows():
            if is_raw_rag:
                batch_items.append({"term": row["Term"], "context": row["NLD"]})
            elif condition == "C":
                batch_items.append({"term": row["Term"], "nld": ""})
            else:
                batch_items.append({"term": row["Term"], "nld": row["NLD"]})

        results = _categorize_batch(batch_items, prompt_tmpl, sys_instr, model_name, model_temp)
        for result in results:
            if str(result["Category"]).startswith("ERROR"):
                raise RuntimeError(
                    f"Condition {condition} batch failed for {result['Term']}: "
                    f"{result['Reasoning']}"
                )
            nld_row = nld_df[nld_df["Term"] == result["Term"]].iloc[0]
            result["NLD"] = nld_row["NLD"]
            result["Context_Used"] = nld_row.get("Context_Used", "")
            result["Condition"] = condition
        return results

    workers = min(ABLATION_WORKERS, len(batches)) if batches else 1
    print(f"  Categorizing {total} terms in {len(batches)} batches with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_batch, batch) for batch in batches]
        for future in as_completed(futures):
            results = future.result()
            with lock:
                for result in results:
                    cat_rows.append(result)
                    cat_ckpt.append(result, is_first=not os.path.exists(cat_csv))
            print(f"  [{len(cat_rows)}/{len(nld_df)}] Condition {condition} categorized")

    df = pd.DataFrame(cat_rows)
    order = {term: index for index, term in enumerate(nld_df["Term"].astype(str))}
    df = df.sort_values("Term", key=lambda values: values.astype(str).map(order)).reset_index(drop=True)
    write_csv(df, cat_csv)
    _validate_category_output(
        df,
        nld_df["Term"].astype(str).tolist(),
        f"Condition {condition} categorization",
    )
    print(f"  Condition {condition} categorized: {len(df)} terms -> {cat_csv}")
    return df


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_ablation(conditions: list[str] | None = None):
    """Run the full ablation study."""
    if conditions is None:
        conditions = CONDITIONS.copy()
    conditions = [condition.upper() for condition in conditions]
    invalid_conditions = sorted(set(conditions) - set(CONDITIONS))
    if invalid_conditions:
        raise ValueError(f"Unknown ablation conditions: {invalid_conditions}")

    reasoning_effort = os.environ.get("LLM_REASONING_EFFORT", "high").lower()
    seed = int(os.environ.get("LLM_SEED", EXPECTED_SEED))
    if reasoning_effort != EXPECTED_REASONING_EFFORT:
        raise RuntimeError(
            f"Ablation requires LLM_REASONING_EFFORT={EXPECTED_REASONING_EFFORT} "
            f"to match frozen A (got {reasoning_effort})"
        )
    if seed != EXPECTED_SEED:
        raise RuntimeError(f"Ablation requires LLM_SEED={EXPECTED_SEED} (got {seed})")

    if any(condition in conditions for condition in ("B", "C", "D")):
        get_client()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load terms (Steps 1-3 output)
    terms_file = os.environ.get("FILTERED_TERMS_OUTPUT", str(FILTERED_TERMS))
    terms_df = read_csv(terms_file)
    if "Readable_Term" not in terms_df.columns:
        raise ValueError(f"Filtered terms file has no Readable_Term column: {terms_file}")
    terms = terms_df["Readable_Term"].astype(str).tolist()
    expected_count = int(os.environ.get("ABLATION_EXPECTED_TERM_COUNT", 407))
    if len(terms) != expected_count:
        raise ValueError(f"Ablation requires {expected_count} terms; found {len(terms)}")
    if len(terms) != len(set(terms)):
        raise ValueError("Filtered ablation term set contains duplicate terms")
    print(f"\nAblation study: {len(terms)} terms, conditions: {conditions}")

    batch_size = int(os.environ.get("BATCH_SIZE", 5))
    cfg = get_config()
    nld_system, nld_prompt = load_prompt("nld_generation.txt")
    abc_system, abc_prompt = _build_categorizer_prompt(is_raw_rag=False)
    d_system, d_prompt = _build_categorizer_prompt(is_raw_rag=True)
    frozen_nld_source = os.environ.get(
        "ABLATION_FROZEN_A_NLD",
        os.environ.get("CONSOLIDATED_LLM_RESULTS_WITH_NLDS", str(FROZEN_A_NLD)),
    )
    frozen_cat_source = os.environ.get(
        "ABLATION_FROZEN_A_CATEGORY",
        os.environ.get("CATEGORIZED_LLM_TERMS", str(FROZEN_A_CATEGORIES)),
    )
    manifest_path = os.path.join(OUTPUT_DIR, "experiment_manifest.json")
    manifest = {
        "study": "PreSaltOntoLearn A/B/C/D representation ablation",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "model": os.environ.get("LLM_GENERATION_MODEL", "gpt-5.4"),
        "reasoning_effort": reasoning_effort,
        "seed": seed,
        "batch_size": batch_size,
        "workers_within_condition": ABLATION_WORKERS,
        "term_count": len(terms),
        "term_set_sha256": _term_set_hash(terms),
        "ontology_config_path": str(cfg._source_path),
        "ontology_config_sha256": _sha256_file(cfg._source_path),
        "prompt_sha256": {
            "nld_generation": _sha256_text(nld_system + "\n" + nld_prompt),
            "categorization_abc": _sha256_text(abc_system + "\n" + abc_prompt),
            "categorization_d": _sha256_text(d_system + "\n" + d_prompt),
        },
        "frozen_condition_a": {
            "nld_source": frozen_nld_source,
            "nld_source_sha256": _sha256_file(frozen_nld_source),
            "category_source": frozen_cat_source,
            "category_source_sha256": _sha256_file(frozen_cat_source),
        },
        "condition_d_context_source": "Condition A Context column; no retrieval",
        "conditions_requested": conditions,
        "condition_artifacts": {},
    }
    fingerprint_fields = {
        "git_commit": manifest["git_commit"],
        "model": manifest["model"],
        "reasoning_effort": reasoning_effort,
        "seed": seed,
        "batch_size": batch_size,
        "workers_within_condition": ABLATION_WORKERS,
        "term_set_sha256": manifest["term_set_sha256"],
        "ontology_config_sha256": manifest["ontology_config_sha256"],
        "prompt_sha256": manifest["prompt_sha256"],
        "frozen_condition_a": manifest["frozen_condition_a"],
    }
    manifest["design_fingerprint_sha256"] = _sha256_text(
        json.dumps(fingerprint_fields, sort_keys=True, separators=(",", ":"))
    )

    resumable_paths = [
        path_fn(condition)
        for condition in ("B", "C", "D")
        for path_fn in (_nld_path, _cat_path)
    ]
    existing_resumable = [path for path in resumable_paths if os.path.exists(path)]
    if existing_resumable:
        if not os.path.exists(manifest_path):
            raise RuntimeError(
                "Ablation checkpoints exist without an experiment manifest; "
                "move them to a separate directory before starting this study"
            )
        with open(manifest_path, "r", encoding="utf-8") as handle:
            previous_manifest = json.load(handle)
        previous_fingerprint = previous_manifest.get("design_fingerprint_sha256")
        if previous_fingerprint != manifest["design_fingerprint_sha256"]:
            raise RuntimeError(
                "Existing ablation checkpoints were produced by a different study design; "
                "use a new ABLATION_OUTPUT_DIR"
            )
        manifest["condition_artifacts"] = previous_manifest.get("condition_artifacts", {})

    def _write_manifest() -> None:
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

    def _record_failure(exc: Exception) -> None:
        manifest["status"] = "failed"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest()

    def _record_condition(condition: str, nld_df: pd.DataFrame, cat_df: pd.DataFrame) -> None:
        _validate_nld_output(nld_df, terms, f"Condition {condition} NLD")
        _validate_category_output(cat_df, terms, f"Condition {condition} categorization")
        nld_path = _nld_path(condition)
        cat_path = _cat_path(condition)
        manifest["condition_artifacts"][condition] = {
            "nld_path": nld_path,
            "nld_sha256": _sha256_file(nld_path),
            "category_path": cat_path,
            "category_sha256": _sha256_file(cat_path),
            "term_set_sha256": _term_set_hash(cat_df["Term"].astype(str).tolist()),
            "reused_frozen_production": condition == "A",
        }
        _write_manifest()

    _write_manifest()
    nld_results: dict[str, pd.DataFrame] = {}
    cat_results: dict[str, pd.DataFrame] = {}
    try:
        # Frozen A is always materialized so every downstream matrix has its reference.
        nld_results["A"] = run_condition_a(terms)
        cat_results["A"] = run_categorization("A", nld_results["A"], batch_size)
        _record_condition("A", nld_results["A"], cat_results["A"])

        # Conditions run sequentially; concurrency is limited to batches within one condition.
        for condition in ("B", "C", "D"):
            if condition not in conditions:
                continue
            if condition == "B":
                nld_results[condition] = run_condition_b(terms)
            elif condition == "C":
                nld_results[condition] = run_condition_c(terms)
            else:
                nld_results[condition] = run_condition_d(terms, nld_results["A"])
            _validate_nld_output(nld_results[condition], terms, f"Condition {condition} NLD")
            cat_results[condition] = run_categorization(
                condition,
                nld_results[condition],
                batch_size,
            )
            _record_condition(condition, nld_results[condition], cat_results[condition])
    except Exception as exc:
        _record_failure(exc)
        raise

    # --- Merge all results ---
    try:
        all_cat = []
        tier_mapping = _category_to_tier()
        for cond, df in cat_results.items():
            df_copy = df.copy()
            df_copy["Condition"] = cond
            df_copy["Tier"] = df_copy["Category"].map(tier_mapping)
            if df_copy["Tier"].isna().any():
                unknown = sorted(df_copy.loc[df_copy["Tier"].isna(), "Category"].unique())
                raise ValueError(f"Condition {cond} has categories without frozen tiers: {unknown}")
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
    except Exception as exc:
        _record_failure(exc)
        raise

    manifest["status"] = "complete"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_manifest()
    print(f"  Experiment manifest: {manifest_path}")

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
