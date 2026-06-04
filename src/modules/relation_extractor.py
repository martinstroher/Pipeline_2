"""
Relation Extractor — Step 6b of the PreSaltOntoLearn pipeline.

Extracts ontological relations from NLDs using an LLM, validates them
against BFO domain/range constraints, and outputs accepted relations.

Input:  5_categorized_ontology.csv (Term, Category, NLD)
Output: 6b_relations.csv (Term, Category, Property, Property_IRI,
        Filler, Filler_Source, Confidence, Evidence, Validation_Status,
        Validation_Reason)

Design decisions (from 5-subagent consensus):
  - 16 Tier 1 properties offered to the LLM (covers >95% of geological NLDs)
  - Flat sequential batches of 10 terms
  - 2-level confidence: 1.0 (explicit) or 0.8 (implied); <0.8 = don't extract
  - Post-hoc property specialization: LLM extracts generic has_part →
    validator upgrades to has_continuant_part / has_occurrent_part
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
from src.utils.gemini_client import generate
from src.utils.prompt_loader import load_prompt
from src.utils.relation_validator import (
    PROPERTY_CONSTRAINTS,
    validate_relation_full,
    ValidationResult,
)

# ────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────

BATCH_SIZE = int(os.environ.get("RELATION_BATCH_SIZE", 10))
CONFIDENCE_THRESHOLD = float(os.environ.get("RELATION_CONFIDENCE_THRESHOLD", 0.7))

_SYSTEM_INSTRUCTION, _PROMPT_TEMPLATE = load_prompt("relation_extraction.txt")

# Allowed property names in the prompt (for validation of LLM output)
_ALLOWED_PROPERTIES = {
    "has_part", "part_of", "has_participant", "participates_in",
    "occurs_in", "located_in", "derives_from", "derives_into",
    "generated_by", "constituted_by", "has_quality", "inheres_in",
    "preceded_by", "precedes", "generated_in", "has_age",
}


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
    """Post-hoc property specialization.

    Upgrades generic has_part/part_of to BFO-precise variants
    based on metatypes of subject and filler.
    """
    from src.utils.relation_validator import get_metatypes, BFOMeta

    if property_name not in ("has_part", "part_of"):
        return property_name

    subj_meta = get_metatypes(subject_cat) or frozenset()
    filler_meta = get_metatypes(filler_cat) if filler_cat else frozenset()

    if not filler_meta:
        return property_name

    subj_is_occ = bool(subj_meta & {BFOMeta.OCCURRENT, BFOMeta.PROCESS, BFOMeta.PROCESS_BOUNDARY})
    filler_is_occ = bool(filler_meta & {BFOMeta.OCCURRENT, BFOMeta.PROCESS, BFOMeta.PROCESS_BOUNDARY})

    if property_name == "has_part":
        if subj_is_occ and filler_is_occ:
            return "has_occurrent_part"
        elif not subj_is_occ and not filler_is_occ:
            return "has_continuant_part"
    elif property_name == "part_of":
        if subj_is_occ and filler_is_occ:
            return "occurrent_part_of"
        elif not subj_is_occ and not filler_is_occ:
            return "continuant_part_of"

    return property_name


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

    result = json.loads(response_text)

    if len(result) != len(batch):
        raise ValueError(
            f"LLM response length ({len(result)}) != batch size ({len(batch)})"
        )

    return result


def run_relation_extraction(
    categorized_csv: str | None = None,
    output_path: str | None = None,
):
    """
    Extract relations from categorized terms.

    Args:
        categorized_csv: Path to 5_categorized_ontology.csv
        output_path: Output path for 6b_relations.csv
    """
    if categorized_csv is None:
        categorized_csv = os.environ.get("CATEGORIZED_LLM_TERMS")
        if not categorized_csv:
            raise RuntimeError("CATEGORIZED_LLM_TERMS env var not set and no path provided.")

    if output_path is None:
        base_dir = os.path.dirname(categorized_csv)
        output_path = os.path.join(base_dir, "6b_relations.csv")

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
        batch_cats = {b["term"]: b["category"] for b in batch}

        try:
            llm_results = _extract_batch(batch_for_llm, known_terms_str)

            for i, item in enumerate(llm_results):
                term = batch[i]["term"]
                category = batch_cats[term]
                nld = batch[i]["nld"]
                relations = item.get("relations", [])

                if not relations:
                    stats["empty"] += 1

                for rel in relations:
                    prop_name = rel.get("property", "")
                    filler = rel.get("filler", "")
                    confidence = float(rel.get("confidence", 0.0))
                    evidence = rel.get("evidence", "")

                    # Skip unknown properties
                    if prop_name not in _ALLOWED_PROPERTIES:
                        row = _make_row(
                            term, category, prop_name, "", filler, "unknown",
                            confidence, evidence, "REJECTED",
                            f"Unknown property '{prop_name}'",
                        )
                        all_rows.append(row)
                        stats["rejected"] += 1
                        continue

                    # Resolve filler source
                    filler_source = _resolve_filler_source(filler, known_terms_lower)

                    # Resolve filler category (for validation)
                    filler_cat = term_to_cat.get(filler.strip().lower(), "")

                    # Specialize property
                    specialized_prop = _specialize_property(prop_name, category, filler_cat)

                    # Get IRI
                    prop_constraint = PROPERTY_CONSTRAINTS.get(specialized_prop)
                    prop_iri = prop_constraint.iri if prop_constraint else ""

                    # Validate
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

                    row = _make_row(
                        term, category, specialized_prop, prop_iri, filler,
                        filler_source, confidence, evidence, status, reason,
                    )
                    all_rows.append(row)

                    if result.is_valid:
                        accepted_relations.add((term, specialized_prop, filler))
                        stats["accepted"] += 1
                    else:
                        stats["rejected"] += 1

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
    parser.add_argument("input_csv", help="Path to categorized CSV (5_categorized_ontology.csv)")
    parser.add_argument("--output", default=None, help="Output path for relations CSV")
    args = parser.parse_args()
    run_relation_extraction(args.input_csv, args.output)
