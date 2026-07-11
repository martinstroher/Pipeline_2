"""
Relation Extractor — Step 6b of the PreSaltOntoLearn pipeline.

Extracts ontological relations from NLDs using an LLM, validates them
against upper-ontology domain/range constraints, and outputs accepted
relations.

Input:  classify_categories.csv (Term, Category, NLD)
Output: construct_relations.csv (Term, Category, Property, Property_IRI,
        Filler, Filler_Source, Confidence, Evidence, Validation_Status,
        Validation_Reason)

Design decisions (from 5-subagent consensus):
  - 16 Tier 1 properties offered to the LLM (covers >95% of geological NLDs)
  - Flat sequential batches of 10 terms
  - 2-level confidence: 1.0 (explicit) or 0.8 (implied); <0.8 = don't extract
  - Post-hoc property specialization: LLM emits a generic property (e.g.,
    has_part); Python upgrades it to the upper-ontology-specific variant
    (e.g., has_continuant_part) using rules declared under
    `property_specializations:` in domains/<name>/ontology_config.yaml.
  - Deterministic filler_source resolution in Python (not LLM)
  - Checkpoint/resume with single flat CSV
"""

import json
import os
import time

import pandas as pd

from src.utils.csv_io import read_csv, write_csv
from src.utils.checkpoint import Checkpoint
from tqdm import tqdm

from src.utils import log
from src.utils.llm_client import generate, parse_json_array
from src.utils.ontology_config import get_config
from src.utils.prompt_loader import load_prompt
from src.utils.relation_validator import (
    PROPERTY_CONSTRAINTS,
    get_metatypes,
    specialize_property,
    validate_relation_full,
    ValidationResult,
)

_CFG = get_config()

# ────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────

BATCH_SIZE = int(os.environ.get("RELATION_BATCH_SIZE", 10))
CONFIDENCE_THRESHOLD = float(os.environ.get("RELATION_CONFIDENCE_THRESHOLD", 0.7))

_SYSTEM_INSTRUCTION, _PROMPT_TEMPLATE = load_prompt("relation_extraction.txt")


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def _resolve_filler_source(filler: str, known_terms_lower: set[str]) -> str:
    """Deterministic filler source resolution."""
    if filler.strip().lower() in known_terms_lower:
        return "domain_term"
    return "external"


def _specialize_property(
    property_name: str,
    subject_cat: str,
    filler_cat: str | None,
) -> str:
    """Thin wrapper around the shared specializer in `relation_validator` (kept
    for call-site stability). The canonical logic lives there so the validate
    step can re-normalise the same way."""
    return specialize_property(property_name, subject_cat, filler_cat)


# ────────────────────────────────────────────────────────────────────────
# Core extraction
# ────────────────────────────────────────────────────────────────────────

def _extract_batch(
    batch: list[dict],
    known_terms_str: str,
) -> list[dict]:
    """Call LLM to extract relations for a batch of terms."""
    json_batch = json.dumps(batch, indent=2)
    prompt = _PROMPT_TEMPLATE.format(
        known_terms=known_terms_str,
        json_batch=json_batch,
        batch_size=len(batch),
    )

    response_text = generate(
        prompt,
        system_instruction=_SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
    )

    result = parse_json_array(response_text)

    if len(result) != len(batch):
        raise ValueError(
            f"LLM response length ({len(result)}) != batch size ({len(batch)})"
        )

    return result


def _materialize_llm_relations(
    batch: list[dict],
    llm_results: list[dict],
    known_terms_lower: set[str],
    term_to_cat: dict[str, str],
    accepted_relations: set[tuple[str, str, str]],
) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    stats = {"accepted": 0, "rejected": 0, "empty": 0}
    for i, item in enumerate(llm_results):
        term = batch[i]["term"]
        category = batch[i]["category"]
        nld = batch[i]["nld"]
        relations = item.get("relations", [])
        if not relations:
            stats["empty"] += 1
        for rel in relations:
            prop_name = rel.get("property", "")
            filler = rel.get("filler", "")
            confidence = float(rel.get("confidence", 0.0))
            evidence = rel.get("evidence", "")
            if prop_name not in PROPERTY_CONSTRAINTS:
                rows.append(_make_row(
                    term, category, prop_name, "", filler, "unknown",
                    confidence, evidence, "REJECTED", f"Unknown property '{prop_name}'",
                ))
                stats["rejected"] += 1
                continue
            filler_source = _resolve_filler_source(filler, known_terms_lower)
            filler_cat = term_to_cat.get(filler.strip().lower(), "")
            specialized_prop = _specialize_property(prop_name, category, filler_cat)
            prop_constraint = PROPERTY_CONSTRAINTS.get(specialized_prop)
            prop_iri = prop_constraint.iri if prop_constraint else ""
            result = validate_relation_full(
                subject_term=term,
                subject_cat=category,
                property_name=specialized_prop,
                object_term=filler,
                object_cat=filler_cat if filler_cat else "unknown",
                confidence=confidence,
                evidence=evidence,
                nld_text=nld,
                existing_relations=accepted_relations,
                confidence_threshold=CONFIDENCE_THRESHOLD,
            )
            status = "ACCEPTED" if result.is_valid else "REJECTED"
            reason = result.reason
            if result.warnings:
                reason += " | Warnings: " + "; ".join(result.warnings)
            rows.append(_make_row(
                term, category, specialized_prop, prop_iri, filler,
                filler_source, confidence, evidence, status, reason,
            ))
            if result.is_valid:
                accepted_relations.add((term, specialized_prop, filler))
                stats["accepted"] += 1
            else:
                stats["rejected"] += 1
    return rows, stats


def extract_relations_for_terms(
    term_rows: list[dict],
    known_terms: pd.DataFrame,
    existing_rows: list[dict] | None = None,
) -> pd.DataFrame:
    """Run standard Step-6b extraction/validation only for supplied new terms."""
    if not term_rows:
        return pd.DataFrame(columns=[
            "Term", "Category", "Property", "Property_IRI", "Filler",
            "Filler_Source", "Confidence", "Evidence", "Validation_Status",
            "Validation_Reason",
        ])
    all_terms = known_terms["Term"].dropna().astype(str).tolist()
    known_terms_lower = {term.strip().lower() for term in all_terms}
    known_terms_str = ", ".join(sorted(all_terms))
    term_to_cat = dict(zip(
        known_terms["Term"].astype(str).str.strip().str.lower(),
        known_terms["Category"].astype(str),
    ))
    accepted_relations = {
        (str(row.get("Term", "")), str(row.get("Property", "")), str(row.get("Filler", "")))
        for row in (existing_rows or []) if row.get("Validation_Status") == "ACCEPTED"
    }
    output: list[dict] = []
    for start in range(0, len(term_rows), BATCH_SIZE):
        batch = term_rows[start:start + BATCH_SIZE]
        llm_results = _extract_batch(
            [{"term": row["term"], "nld": row["nld"]} for row in batch],
            known_terms_str,
        )
        rows, _ = _materialize_llm_relations(
            batch, llm_results, known_terms_lower, term_to_cat, accepted_relations,
        )
        output.extend(rows)
    return pd.DataFrame(output)


def run_relation_extraction(
    categorized_csv: str | None = None,
    output_path: str | None = None,
):
    """
    Extract relations from categorized terms.

    Args:
        categorized_csv: Path to classify_categories.csv
        output_path: Output path for construct_relations.csv
    """
    if categorized_csv is None:
        categorized_csv = os.environ.get("CATEGORIZED_LLM_TERMS")
        if not categorized_csv:
            raise RuntimeError("CATEGORIZED_LLM_TERMS env var not set and no path provided.")

    if output_path is None:
        base_dir = os.path.dirname(categorized_csv)
        output_path = os.path.join(base_dir, "construct_relations.csv")

    log.banner("6b", "Relation Extraction")

    # Load categorized terms
    df = read_csv(categorized_csv)
    log.info(f"Loaded {len(df)} terms from {categorized_csv}")

    # Filter out errors and NOT_CLASSIFIED, and rows without NLD
    df_valid = df[
        ~df["Category"].str.startswith("ERROR", na=False)
        & (df["Category"] != "NOT_CLASSIFIED")
        & df["NLD"].notna()
        & (df["NLD"].str.strip() != "")
    ].copy()
    log.info(f"{len(df_valid)} valid terms (with NLD, excluding errors)")

    if df_valid.empty:
        log.warn("No valid terms for relation extraction.")
        return output_path

    # Build known terms set (for filler resolution)
    all_terms = df["Term"].tolist()
    known_terms_lower = {t.strip().lower() for t in all_terms}
    known_terms_str = ", ".join(sorted(all_terms))

    # Build category lookup for filler specialization
    term_to_cat = dict(zip(
        df["Term"].str.strip().str.lower(),
        df["Category"],
    ))

    # Load checkpoint
    ckpt = Checkpoint(output_path)
    completed, existing_rows = ckpt.load()
    if completed:
        log.info(f"Resuming: {len(completed)} terms already processed.")

    # Prepare batches (flat sequential)
    pending_df = df_valid[~df_valid["Term"].isin(completed)]
    pending_list = []
    for _, row in pending_df.iterrows():
        pending_list.append({
            "term": row["Term"],
            "nld": row["NLD"],
            "category": row["Category"],
        })

    total_batches = (len(pending_list) + BATCH_SIZE - 1) // BATCH_SIZE
    log.info(f"Processing {len(pending_list)} pending terms in {total_batches} batches of {BATCH_SIZE}")

    all_rows = list(existing_rows)
    accepted_relations: set[tuple[str, str, str]] = set()
    # Rebuild accepted set from checkpoint
    for row in existing_rows:
        if row.get("Validation_Status") == "ACCEPTED":
            accepted_relations.add((row["Term"], row["Property"], row["Filler"]))

    stats = {"accepted": 0, "rejected": 0, "errors": 0, "empty": 0}

    pbar = tqdm(
        total=len(df_valid),
        desc="Extracting relations",
        initial=len(completed),
    )

    for batch_start in range(0, len(pending_list), BATCH_SIZE):
        batch = pending_list[batch_start : batch_start + BATCH_SIZE]
        batch_for_llm = [{"term": b["term"], "nld": b["nld"]} for b in batch]

        try:
            llm_results = _extract_batch(batch_for_llm, known_terms_str)
            batch_rows, batch_stats = _materialize_llm_relations(
                batch, llm_results, known_terms_lower, term_to_cat, accepted_relations,
            )
            all_rows.extend(batch_rows)
            for key in ("accepted", "rejected", "empty"):
                stats[key] += batch_stats[key]

        except Exception as e:
            tqdm.write("")
            log.error(f"Batch starting at '{batch[0]['term']}': {e}")
            for b in batch:
                row = _make_row(
                    b["term"], b["category"], "", "", "", "",
                    0.0, "", "ERROR", str(e),
                )
                all_rows.append(row)
            stats["errors"] += len(batch)

        pbar.update(len(batch))

        # Checkpoint: save after each batch
        _save_checkpoint(all_rows, output_path)

        time.sleep(1)

    pbar.close()

    # Final save
    _save_checkpoint(all_rows, output_path)

    log.success(
        f"Relation extraction complete: {stats['accepted']} accepted, "
        f"{stats['rejected']} rejected, {stats['empty']} terms with no relations, "
        f"{stats['errors']} errors"
    )
    log.detail(f"Output: {output_path}")

    return output_path


def _make_row(
    term, category, prop, prop_iri, filler, filler_source,
    confidence, evidence, status, reason,
) -> dict:
    return {
        "Term": term,
        "Category": category,
        "Property": prop,
        "Property_IRI": prop_iri,
        "Filler": filler,
        "Filler_Source": filler_source,
        "Confidence": confidence,
        "Evidence": evidence,
        "Validation_Status": status,
        "Validation_Reason": reason,
    }


def _save_checkpoint(rows: list[dict], path: str) -> None:
    """Save all rows to CSV (overwrites)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_csv(pd.DataFrame(rows), path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Relation Extraction (Step 6b)")
    parser.add_argument("input_csv", help="Path to categorized CSV (classify_categories.csv)")
    parser.add_argument("--output", default=None, help="Output path for relations CSV")
    args = parser.parse_args()
    run_relation_extraction(args.input_csv, args.output)
