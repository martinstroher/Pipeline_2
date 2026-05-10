"""
Relation Reclassifier — Step 6d of the PreSaltOntoLearn pipeline.

Deterministic post-processing step that uses accepted relations from Step 6b
to infer and correct BFO metatype classifications. Reverses the domain/range
logic from relation_validator.py: instead of "is this relation valid for
these categories?" → "given these relations, what categories are valid?"

No LLM calls. No hardcoded inference rules. Fully dynamic — reads
PROPERTY_CONSTRAINTS and _CATEGORY_TO_METATYPES from relation_validator.py.

Input:  6c_taxonomy_cleaned.csv + 6c_relations_cleaned.csv
Output: 6d_taxonomy_reclassified.csv + 6d_reclassification_log.csv
"""

import os
from collections import defaultdict

import pandas as pd

from src.utils import log
from src.utils.relation_validator import (
    PROPERTY_CONSTRAINTS,
    _CATEGORY_TO_METATYPES,
    get_metatypes,
)
from src.modules.taxonomy_builder import UPPER_IRIS

# Case-insensitive category lookup: lowered key → original key
_CATEGORY_LOOKUP_ORIG = {k.lower(): k for k in _CATEGORY_TO_METATYPES}
# Upper-ontology lookup (case-insensitive)
_UPPER_LOWER = {k.lower(): k for k in UPPER_IRIS}

# Property name lookup (case-insensitive) for relation CSV matching
_PROPERTY_LOOKUP = {k.lower(): k for k in PROPERTY_CONSTRAINTS}


# ── Evidence collection ────────────────────────────────────────────────

def _collect_evidence(
    term: str,
    relations_by_subject: dict[str, list[dict]],
    relations_by_filler: dict[str, list[dict]],
) -> list[frozenset[str]]:
    """Collect metatype constraint sets from all accepted relations involving a term.

    For each relation where the term is the subject, appends the property's
    domain constraint set. For each relation where the term is the filler,
    appends the property's range constraint set.

    Returns a list of frozensets (one per relation occurrence).
    """
    evidence: list[frozenset[str]] = []
    term_lower = term.strip().lower()

    for rel in relations_by_subject.get(term_lower, []):
        prop_name = _PROPERTY_LOOKUP.get(rel["Property"].strip().lower())
        if prop_name and prop_name in PROPERTY_CONSTRAINTS:
            evidence.append(PROPERTY_CONSTRAINTS[prop_name].domain)

    for rel in relations_by_filler.get(term_lower, []):
        prop_name = _PROPERTY_LOOKUP.get(rel["Property"].strip().lower())
        if prop_name and prop_name in PROPERTY_CONSTRAINTS:
            evidence.append(PROPERTY_CONSTRAINTS[prop_name].range)

    return evidence


# ── Category matching ──────────────────────────────────────────────────

# Overly generic metatypes that don't carry classification information.
# Excluded from parent-category compatibility checks because they make
# everything look compatible (every category includes Continuant or Occurrent).
_GENERIC_METATYPES = frozenset({"Continuant", "Occurrent"})


def _distinguishing_metatypes(metatypes: frozenset[str]) -> frozenset[str]:
    """Return the metatype set with overly generic ancestors removed."""
    return metatypes - _GENERIC_METATYPES


def _find_best_category(
    implied_metatypes: frozenset[str],
    current_category: str,
) -> str | None:
    """Find a more specific category compatible with the metatype evidence.

    Compatibility means the category's metatypes INTERSECT with the implied
    metatypes (not subset — because categories include ancestor chains like
    Continuant but property constraints don't).

    Returns None if current category is already the best match.
    Never downgrades to a less specific category.
    Prefers domain-specific categories (GeoCore/GeoReservoir) over raw BFO.
    """
    current_metatypes = get_metatypes(current_category)

    # Check if current category is compatible (metatypes intersect with evidence)
    if current_metatypes is not None and (current_metatypes & implied_metatypes):
        # Current is compatible — only reclassify if we find something MORE specific
        pass
    elif current_metatypes is not None:
        # Current category is INCOMPATIBLE with evidence (no intersection)
        # Must reclassify
        pass

    # Find all compatible categories (metatypes intersect with evidence)
    candidates: list[tuple[str, frozenset[str], int]] = []
    for cat_name, cat_metatypes in _CATEGORY_TO_METATYPES.items():
        overlap = cat_metatypes & implied_metatypes
        if overlap:
            # Score: number of distinguishing metatypes in common with evidence.
            # More specific overlap = better candidate.
            dist_overlap = _distinguishing_metatypes(overlap)
            candidates.append((cat_name, cat_metatypes, len(dist_overlap)))

    if not candidates:
        return None

    # Rank: most distinguishing overlap first, then fewer total metatypes (more specific),
    # then prefer domain-specific names (longer names = GeoCore/GeoReservoir)
    candidates.sort(key=lambda c: (-c[2], len(c[1]), -len(c[0])))

    best_name = candidates[0][0]
    best_metatypes = candidates[0][1]

    # Don't reclassify to the same category
    if best_name.lower() == current_category.lower():
        return None

    # Don't reclassify if current is already compatible and equally or more specific
    if current_metatypes is not None and (current_metatypes & implied_metatypes):
        current_dist = len(_distinguishing_metatypes(current_metatypes & implied_metatypes))
        best_dist = candidates[0][2]
        if current_dist >= best_dist:
            return None
        # Only reclassify if new category is strictly more specific
        if len(best_metatypes) >= len(current_metatypes):
            return None

    return best_name


# ── Main entry point ───────────────────────────────────────────────────

def run_relation_reclassification(
    taxonomy_csv: str,
    relations_csv: str,
    output_path: str | None = None,
) -> str:
    """
    Reclassify taxonomy terms using BFO metatype evidence from accepted relations.

    Args:
        taxonomy_csv: Path to 6c_taxonomy_cleaned.csv
        relations_csv: Path to 6c_relations_cleaned.csv
        output_path: Output path for reclassified taxonomy (default: 6d_taxonomy_reclassified.csv)

    Returns:
        Path to the reclassified taxonomy CSV
    """
    base_dir = os.path.dirname(taxonomy_csv)
    if output_path is None:
        output_path = os.path.join(base_dir, "6d_taxonomy_reclassified.csv")
    log_path = os.path.join(base_dir, "6d_reclassification_log.csv")

    df = pd.read_csv(taxonomy_csv, encoding="utf-8-sig")
    rel_df = pd.read_csv(relations_csv, encoding="utf-8-sig")

    # Filter to accepted relations only
    accepted = rel_df[rel_df["Validation_Status"] == "ACCEPTED"]
    log.info(
        f"Relation reclassifier: {len(df)} taxonomy entries, "
        f"{len(accepted)} accepted relations"
    )

    # Build lookup indices (case-insensitive)
    relations_by_subject: dict[str, list[dict]] = defaultdict(list)
    relations_by_filler: dict[str, list[dict]] = defaultdict(list)
    for _, rel in accepted.iterrows():
        row_dict = rel.to_dict()
        term_lower = str(rel["Term"]).strip().lower()
        filler_lower = str(rel["Filler"]).strip().lower()
        relations_by_subject[term_lower].append(row_dict)
        relations_by_filler[filler_lower].append(row_dict)

    # Process each term
    log_rows: list[dict] = []
    n_reclassified = 0
    n_contradictions = 0
    n_reparented = 0

    for idx, row in df.iterrows():
        term = str(row["Term"]).strip()
        current_category = str(row.get("Category", "")).strip()

        # Step A: Collect evidence
        evidence = _collect_evidence(term, relations_by_subject, relations_by_filler)
        if not evidence:
            continue  # No relations for this term

        # Step B: Intersect all evidence sets
        implied_metatypes = evidence[0]
        for ev in evidence[1:]:
            implied_metatypes = implied_metatypes & ev

        if not implied_metatypes:
            # CONTRADICTION: relations imply incompatible metatypes
            n_contradictions += 1
            # Collect the conflicting property names for the log
            subject_props = [
                r["Property"] for r in relations_by_subject.get(term.strip().lower(), [])
            ]
            filler_props = [
                r["Property"] for r in relations_by_filler.get(term.strip().lower(), [])
            ]
            log_rows.append({
                "Action": "CONTRADICTION",
                "Term": term,
                "Old_Category": current_category,
                "New_Category": current_category,
                "Detail": (
                    f"Empty metatype intersection — incompatible relations. "
                    f"As subject: {', '.join(set(subject_props))}. "
                    f"As filler: {', '.join(set(filler_props))}"
                ),
                "Evidence_Count": len(evidence),
            })
            log.warn(f"  CONTRADICTION: '{term}' — relations imply incompatible metatypes")
            continue

        # Step C: Find best category
        best_category = _find_best_category(implied_metatypes, current_category)
        if best_category is None:
            continue  # Current category is fine

        # Reclassify
        df.at[idx, "Category"] = best_category
        n_reclassified += 1
        log_rows.append({
            "Action": "RECLASSIFY",
            "Term": term,
            "Old_Category": current_category,
            "New_Category": best_category,
            "Detail": (
                f"Metatype evidence from {len(evidence)} relations. "
                f"Implied: {{{', '.join(sorted(implied_metatypes))}}}"
            ),
            "Evidence_Count": len(evidence),
        })

    # Step D: Category/Parent consistency repair
    for idx, row in df.iterrows():
        term = str(row["Term"]).strip()
        parent = row.get("Parent_Term", "")
        category = str(row.get("Category", "")).strip()

        if not parent or (isinstance(parent, float) and pd.isna(parent)) or not str(parent).strip():
            continue

        parent_str = str(parent).strip()

        # Fix self-referential parents (term = parent)
        if parent_str == term:
            upper_key = category if category in UPPER_IRIS else _UPPER_LOWER.get(category.lower(), "")
            if upper_key and upper_key in UPPER_IRIS:
                df.at[idx, "Parent_Term"] = upper_key
                n_reparented += 1
                log_rows.append({
                    "Action": "REPARENT",
                    "Term": term,
                    "Old_Category": category,
                    "New_Category": category,
                    "Detail": (
                        f"Self-referential parent removed. "
                        f"Reparented to '{upper_key}'"
                    ),
                    "Evidence_Count": 0,
                })
            continue

        parent_str = str(parent).strip()

        # Find parent's category
        parent_rows = df[df["Term"] == parent_str]
        if parent_rows.empty:
            # Parent is an upper-ontology term or external — skip
            continue

        # If parent is a known upper-ontology term, only skip if it's the
        # upper-ontology anchor for THIS term's category.  When the parent is
        # a different upper-ontology branch (e.g., Geological Structure parent
        # for a Geological Object term), the incompatibility is real.
        parent_is_upper = parent_str in UPPER_IRIS or parent_str.lower() in _UPPER_LOWER
        if parent_is_upper:
            category_upper = UPPER_IRIS.get(category, "")
            parent_upper = UPPER_IRIS.get(parent_str, "")
            if category_upper == parent_upper:
                continue  # Same upper anchor — compatible by definition

        parent_cat = str(parent_rows.iloc[0].get("Category", "")).strip()
        # Use the parent's own name as its effective category when it IS an
        # upper-ontology term (its Category column may be misleading for
        # intermediate nodes placed in a different branch).
        effective_parent_cat = parent_str if parent_is_upper else parent_cat

        # If the parent's name IS the term's category name, it's correctly placed
        # (e.g., term with Category="Geological Structure" under parent named "Geological Structure")
        if parent_str.lower() == category.lower():
            continue

        # Check metatype compatibility between term and parent using
        # DISTINGUISHING metatypes only — excluding Continuant/Occurrent
        # which are too generic and would make everything look compatible.
        term_metatypes = get_metatypes(category)
        parent_metatypes = get_metatypes(effective_parent_cat)

        if term_metatypes is None or parent_metatypes is None:
            continue

        term_dist = _distinguishing_metatypes(term_metatypes)
        parent_dist = _distinguishing_metatypes(parent_metatypes)

        # Incompatible if distinguishing metatypes share nothing
        # (e.g., Object under GDC parent, or Site under SpatialRegion parent)
        if term_dist and parent_dist and not (term_dist & parent_dist):
            # Reparent to category root via UPPER_IRIS
            upper_key = category if category in UPPER_IRIS else _UPPER_LOWER.get(category.lower(), "")
            if upper_key and upper_key in UPPER_IRIS:
                old_parent = parent_str
                df.at[idx, "Parent_Term"] = upper_key
                n_reparented += 1
                log_rows.append({
                    "Action": "REPARENT",
                    "Term": term,
                    "Old_Category": category,
                    "New_Category": category,
                    "Detail": (
                        f"Parent '{old_parent}' (effective category: {effective_parent_cat}) "
                        f"incompatible with term category '{category}'. "
                        f"Reparented to '{upper_key}'"
                    ),
                    "Evidence_Count": 0,
                })

    # ── Save outputs ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    if log_rows:
        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
        log.detail(f"Reclassification log: {log_path}")

    log.success(
        f"Relation reclassifier: {n_reclassified} reclassified, "
        f"{n_contradictions} contradictions, {n_reparented} reparented"
    )
    log.detail(f"Output: {output_path}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reclassify taxonomy terms using relation-based BFO metatype evidence"
    )
    parser.add_argument("taxonomy_csv", help="Path to 6c_taxonomy_cleaned.csv")
    parser.add_argument("--relations", required=True, help="Path to 6c_relations_cleaned.csv")
    parser.add_argument("--output", default=None, help="Output path for reclassified taxonomy")
    args = parser.parse_args()
    run_relation_reclassification(
        args.taxonomy_csv,
        args.relations,
        output_path=args.output,
    )
