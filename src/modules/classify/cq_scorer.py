"""
Step 5b — CQ-Driven Refinement.

Cleans encoding/synonym duplicates, scores each term against 10 competency
questions, and produces filtered categorized CSVs at multiple CQ-count
thresholds (T=0, T=1, T=2, T=3).

Sub-steps:
  A. Deterministic cleanup — encoding dupes, surface-form variants.
  B. Synonym triage — LLM-assisted 3-way classification of near-synonym clusters.
  C. CQ scoring — parallel batched scoring (5 terms/call) against all 10 CQs.
  D. Threshold split — write filtered CSVs per threshold into output/refined/t{N}/.
"""

import json
import os
import threading
import unicodedata

import pandas as pd

from src.utils.csv_io import read_csv, write_csv
from src.utils.checkpoint import Checkpoint
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.utils import log
from src.utils.gemini_client import generate
from src.utils.prompt_loader import load_prompt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFINED_DIR = os.path.join("output", "refined")
CLEANUP_REPORT = os.path.join(REFINED_DIR, "5b_cleanup_report.csv")
SPECIALIZATION_HINTS = os.path.join(REFINED_DIR, "5b_specialization_hints.csv")
CQ_MATRIX_FILE = os.path.join(REFINED_DIR, "5b_cq_matrix.csv")
THRESHOLD_SUMMARY = os.path.join(REFINED_DIR, "threshold_summary.csv")
THRESHOLDS = [0, 1, 2, 3]

CQ_BATCH_SIZE = int(os.environ.get("CQ_BATCH_SIZE", 5))
MAX_CONCURRENT_CQ = int(os.environ.get("MAX_CONCURRENT_CQ", 5))
VALID_CQS = {f"CQ{i}" for i in range(1, 11)}

_SYSTEM_INSTRUCTION, _PROMPT_TEMPLATE = load_prompt("cq_scoring.txt")
_SYNONYM_SYSTEM, _SYNONYM_PROMPT = load_prompt("cq_synonym_triage.txt")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Unicode NFKD normalise and lowercase."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower().strip()




# ---------------------------------------------------------------------------
# Sub-step A: Deterministic encoding / surface-form cleanup
# ---------------------------------------------------------------------------


def _detect_encoding_dupes(df: pd.DataFrame) -> list[dict]:
    """Find terms that are encoding/accent variants of each other."""
    norm_map: dict[str, list[str]] = {}
    for term in df["Term"]:
        key = _normalize(term)
        norm_map.setdefault(key, []).append(term)

    merges = []
    for key, variants in norm_map.items():
        if len(variants) <= 1:
            continue
        # Prefer the accented / longer form as canonical
        canonical = max(variants, key=lambda t: (len(t), t))
        for v in variants:
            if v != canonical:
                merges.append({
                    "Dropped_Term": v,
                    "Canonical_Term": canonical,
                    "Classification": "ENCODING_DUPLICATE",
                    "Reasoning": f"Accent/encoding variant of '{canonical}'",
                })
    return merges


def _detect_hyphen_variants(df: pd.DataFrame) -> list[dict]:
    """Find terms differing only by hyphenation (e.g. build-up / buildup)."""
    dehyphen_map: dict[str, list[str]] = {}
    for term in df["Term"]:
        key = term.lower().replace("-", "").replace(" ", "")
        dehyphen_map.setdefault(key, []).append(term)

    merges = []
    for key, variants in dehyphen_map.items():
        if len(variants) <= 1:
            continue
        # Already handled by encoding dedup?
        norms = {_normalize(v) for v in variants}
        if len(norms) == 1:
            continue  # Already caught by _detect_encoding_dupes
        canonical = max(variants, key=lambda t: (len(t), t))
        for v in variants:
            if v != canonical:
                merges.append({
                    "Dropped_Term": v,
                    "Canonical_Term": canonical,
                    "Classification": "HYPHEN_VARIANT",
                    "Reasoning": f"Hyphenation variant of '{canonical}'",
                })
    return merges


# ---------------------------------------------------------------------------
# Sub-step B: LLM-assisted synonym triage
# ---------------------------------------------------------------------------


def _build_synonym_candidates(df: pd.DataFrame) -> list[list[dict]]:
    """Build near-synonym clusters by grouping terms that share head nouns
    within the same category."""
    clusters = []
    for category, group in df.groupby("Category"):
        # Group by last word (head noun)
        head_map: dict[str, list[dict]] = {}
        for _, row in group.iterrows():
            words = row["Term"].lower().split()
            if not words:
                continue
            head = words[-1]
            head_map.setdefault(head, []).append({
                "term": row["Term"],
                "category": row["Category"],
                "definition": row["NLD"],
            })
        for head, terms in head_map.items():
            if len(terms) >= 2:
                # Only flag groups where at least 2 terms share >50% of words
                # to avoid grouping "porosity" with "vuggy porosity" etc.
                for i in range(len(terms)):
                    for j in range(i + 1, len(terms)):
                        words_i = set(terms[i]["term"].lower().split())
                        words_j = set(terms[j]["term"].lower().split())
                        overlap = len(words_i & words_j) / max(1, min(len(words_i), len(words_j)))
                        if overlap >= 0.5 and len(words_i | words_j) <= len(words_i) + 1:
                            # These two are close enough to be candidates
                            # Check if they're already in a cluster
                            found = False
                            for cl in clusters:
                                cl_terms = {t["term"] for t in cl}
                                if terms[i]["term"] in cl_terms or terms[j]["term"] in cl_terms:
                                    cl_terms.add(terms[i]["term"])
                                    cl_terms.add(terms[j]["term"])
                                    # Rebuild cluster with all terms
                                    all_in = {t["term"]: t for t in cl}
                                    all_in[terms[i]["term"]] = terms[i]
                                    all_in[terms[j]["term"]] = terms[j]
                                    cl.clear()
                                    cl.extend(all_in.values())
                                    found = True
                                    break
                            if not found:
                                clusters.append([terms[i], terms[j]])

    # Deduplicate within clusters and filter to ≥2
    deduped = []
    for cl in clusters:
        seen = set()
        unique = []
        for t in cl:
            if t["term"] not in seen:
                seen.add(t["term"])
                unique.append(t)
        if len(unique) >= 2:
            deduped.append(unique)
    return deduped


def _run_synonym_triage(clusters: list[list[dict]]) -> tuple[list[dict], list[dict]]:
    """Send synonym clusters to LLM for 3-way classification.

    Returns:
        (merges, specializations) — merges are SYNONYM drops;
        specializations are (General_Term, Specific_Term) pairs.
    """
    if not clusters:
        return [], []

    # Build batched prompt (all clusters in one call if ≤20, else split)
    all_merges = []
    all_specializations = []
    batch_size = 20
    for batch_start in range(0, len(clusters), batch_size):
        batch = clusters[batch_start:batch_start + batch_size]
        clusters_payload = []
        for idx, cl in enumerate(batch):
            clusters_payload.append({
                "cluster_id": idx,
                "terms": [
                    {"term": t["term"], "definition": t["definition"][:300]}
                    for t in cl
                ],
            })

        prompt = _SYNONYM_PROMPT.format(
            clusters_json=json.dumps(clusters_payload, indent=2)
        )
        response = generate(
            prompt,
            system_instruction=_SYNONYM_SYSTEM,
            response_mime_type="application/json",
        )
        try:
            results = json.loads(response)
        except json.JSONDecodeError:
            log.warn(f"Synonym triage: failed to parse LLM response, skipping batch")
            continue

        for result in results:
            cid = result.get("cluster_id", -1)
            if cid < 0 or cid >= len(batch):
                continue
            classification = result.get("classification", "DISTINCT")
            if classification == "SYNONYM":
                canonical = result.get("canonical_term")
                cluster_terms = [t["term"] for t in batch[cid]]
                if canonical not in cluster_terms:
                    canonical = cluster_terms[0]
                for t in cluster_terms:
                    if t != canonical:
                        all_merges.append({
                            "Dropped_Term": t,
                            "Canonical_Term": canonical,
                            "Classification": "SYNONYM",
                            "Reasoning": result.get("reasoning", ""),
                        })
            elif classification == "SPECIALIZATION":
                # Record parent-child hint for taxonomy builder
                cluster_terms = [t["term"] for t in batch[cid]]
                canonical = result.get("canonical_term")
                if canonical and canonical in cluster_terms:
                    for t in cluster_terms:
                        if t != canonical:
                            all_specializations.append({
                                "General_Term": canonical,
                                "Specific_Term": t,
                                "Reasoning": result.get("reasoning", ""),
                            })
            # DISTINCT → keep both, no action needed

    return all_merges, all_specializations


# ---------------------------------------------------------------------------
# Sub-step C: CQ scoring (parallel, batched)
# ---------------------------------------------------------------------------


def _score_batch(batch_terms: list[dict], batch_size: int) -> list[dict]:
    """Score a single batch of terms against CQs. Returns list of row dicts."""
    terms_json = json.dumps(batch_terms, indent=2)
    prompt = _PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        terms_json=terms_json,
    )
    response = generate(
        prompt,
        system_instruction=_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
    )

    parsed = json.loads(response)
    if not isinstance(parsed, list) or len(parsed) != batch_size:
        raise ValueError(
            f"Expected {batch_size} results, got {len(parsed) if isinstance(parsed, list) else type(parsed)}"
        )

    rows = []
    for i, item in enumerate(parsed):
        term = item.get("term", batch_terms[i]["term"])
        raw_cqs = item.get("relevant_cqs", [])
        # Validate CQ identifiers
        valid = sorted([cq for cq in raw_cqs if cq in VALID_CQS])
        reasoning = item.get("reasoning", "")

        row = {"Term": term, "Reasoning": reasoning, "CQ_Count": len(valid)}
        for cq in VALID_CQS:
            row[cq] = 1 if cq in valid else 0
        rows.append(row)
    return rows


def _run_cq_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Score all terms against CQs with parallel batched calls."""
    # Load checkpoint
    ckpt = Checkpoint(CQ_MATRIX_FILE)
    completed, results = ckpt.load()
    all_terms = df["Term"].tolist()
    if completed:
        log.info(f"CQ scoring: resuming, {len(completed)} terms already scored.")

    # Build batch payloads for pending terms
    pending_rows = df[~df["Term"].isin(completed)]
    pending_items = [
        {"term": row["Term"], "category": row["Category"], "definition": row["NLD"]}
        for _, row in pending_rows.iterrows()
    ]

    # Split into batches
    batches = []
    for i in range(0, len(pending_items), CQ_BATCH_SIZE):
        batches.append(pending_items[i:i + CQ_BATCH_SIZE])

    if not batches:
        log.info("CQ scoring: all terms already scored.")
        return pd.DataFrame(results)

    max_workers = min(MAX_CONCURRENT_CQ, len(batches))
    log.info(
        f"CQ scoring: {len(pending_items)} pending terms in {len(batches)} batches, "
        f"{max_workers} workers"
    )

    lock = threading.Lock()
    errors = []

    pbar = tqdm(
        total=len(all_terms),
        desc="CQ Scoring",
        initial=len(all_terms) - len(pending_items),
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_score_batch, batch, len(batch)): batch
            for batch in batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            batch_terms_str = ", ".join(t["term"][:20] for t in batch[:2])
            try:
                batch_rows = future.result()
                with lock:
                    is_first = len(results) == 0
                    results.extend(batch_rows)
                    ckpt.append_batch(batch_rows, is_first=is_first)
                pbar.set_postfix_str(batch_terms_str)
            except Exception as e:
                tqdm.write("")
                log.error(f"CQ batch [{batch_terms_str}...]: {e}")
                for t in batch:
                    errors.append({"Term": t["term"], "Error": str(e)})
            pbar.update(len(batch))
    pbar.close()

    # Write consolidated output
    df_result = pd.DataFrame(results)
    write_csv(df_result, CQ_MATRIX_FILE)
    log.success(f"CQ matrix: {len(df_result)} terms scored → '{CQ_MATRIX_FILE}'")

    if errors:
        log.warn(f"CQ scoring: {len(errors)} term(s) failed. Check logs.")

    return df_result


# ---------------------------------------------------------------------------
# Sub-step D: Threshold split
# ---------------------------------------------------------------------------


def _run_threshold_split(
    df_categorized: pd.DataFrame,
    df_cq: pd.DataFrame,
) -> dict[int, str]:
    """Write filtered categorized CSVs for each threshold. Returns {T: path}."""
    # Merge CQ_Count into categorized data
    cq_counts = df_cq[["Term", "CQ_Count"]].drop_duplicates(subset="Term")
    df_merged = df_categorized.merge(cq_counts, on="Term", how="left")
    df_merged["CQ_Count"] = df_merged["CQ_Count"].fillna(0).astype(int)

    paths = {}
    summary_rows = []
    for t in THRESHOLDS:
        t_dir = os.path.join(REFINED_DIR, f"t{t}")
        os.makedirs(t_dir, exist_ok=True)
        out_path = os.path.join(t_dir, "5_categorized_ontology.csv")

        df_filtered = df_merged[df_merged["CQ_Count"] >= t].copy()
        # Drop CQ_Count — not part of the standard Step 5 schema
        df_filtered = df_filtered.drop(columns=["CQ_Count"])
        write_csv(df_filtered, out_path)

        n_terms = len(df_filtered)
        n_cats = df_filtered["Category"].nunique()
        log.info(f"  T≥{t}: {n_terms} terms, {n_cats} categories → '{out_path}'")

        paths[t] = out_path
        summary_rows.append({
            "Threshold": t,
            "Term_Count": n_terms,
            "Category_Count": n_cats,
        })

    write_csv(pd.DataFrame(summary_rows), THRESHOLD_SUMMARY)
    return paths


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_cq_refinement(
    categorized_csv: str | None = None,
) -> dict[int, str]:
    """Run the full CQ-driven refinement: cleanup → scoring → threshold split.

    Args:
        categorized_csv: Path to Step 5 output. Defaults to CATEGORIZED_LLM_TERMS env var.

    Returns:
        Dict mapping threshold T → path to filtered categorized CSV.
    """
    if categorized_csv is None:
        categorized_csv = os.environ.get("CATEGORIZED_LLM_TERMS")
        if not categorized_csv:
            raise RuntimeError("CATEGORIZED_LLM_TERMS env var is not set.")

    os.makedirs(REFINED_DIR, exist_ok=True)

    # Load Step 5 output
    df = read_csv(categorized_csv)
    log.info(f"Loaded {len(df)} terms from '{categorized_csv}'")

    # Filter out error categories
    error_mask = df["Category"].str.startswith("ERROR", na=False) | (df["Category"] == "NOT_CLASSIFIED")
    if error_mask.any():
        log.warn(f"Removing {error_mask.sum()} terms with error/unclassified categories")
        df = df[~error_mask].copy()

    # ----- Sub-step A: Deterministic cleanup -----
    log.banner("5b-A", "Encoding & Surface-Form Cleanup")
    all_merges = []
    all_merges.extend(_detect_encoding_dupes(df))
    all_merges.extend(_detect_hyphen_variants(df))

    # Apply deterministic merges
    dropped_terms = {m["Dropped_Term"] for m in all_merges}
    df_clean = df[~df["Term"].isin(dropped_terms)].copy()
    log.info(f"Deterministic cleanup: {len(all_merges)} duplicates removed, {len(df_clean)} terms remain")

    # ----- Sub-step B: Synonym triage -----
    log.banner("5b-B", "Synonym Triage")
    clusters = _build_synonym_candidates(df_clean)
    log.info(f"Found {len(clusters)} candidate synonym clusters")

    if clusters:
        synonym_merges, specializations = _run_synonym_triage(clusters)
        all_merges.extend(synonym_merges)
        syn_dropped = {m["Dropped_Term"] for m in synonym_merges}
        df_clean = df_clean[~df_clean["Term"].isin(syn_dropped)].copy()
        log.info(f"Synonym triage: {len(synonym_merges)} merges, {len(specializations)} specializations, {len(df_clean)} terms remain")

        # Write specialization hints for taxonomy builder
        if specializations:
            write_csv(pd.DataFrame(specializations), SPECIALIZATION_HINTS)
            log.info(f"Specialization hints: {len(specializations)} pairs → '{SPECIALIZATION_HINTS}'")
    else:
        log.info("No synonym candidates found.")

    # Write cleanup report
    if all_merges:
        write_csv(pd.DataFrame(all_merges), CLEANUP_REPORT)
        log.success(f"Cleanup report: {len(all_merges)} merges → '{CLEANUP_REPORT}'")
    else:
        log.info("No merges needed.")

    # ----- Sub-step C: CQ scoring -----
    log.banner("5b-C", "CQ Scoring")
    df_cq = _run_cq_scoring(df_clean)

    # ----- Sub-step D: Threshold split -----
    log.banner("5b-D", "Threshold Split")
    paths = _run_threshold_split(df_clean, df_cq)

    log.success(f"Step 5b complete. {len(THRESHOLDS)} threshold variants written to '{REFINED_DIR}/'")
    return paths
