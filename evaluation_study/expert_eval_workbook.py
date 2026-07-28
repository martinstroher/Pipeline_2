"""Modular, blinded expert-evaluation workbooks for PreSaltOntoLearn.

The study separates three estimands:

* Representation evaluation: relevance and A/B NLD quality for a seeded,
    stratified 100-term sample.
* Category evaluation: A/B/C/D correctness for a separate seeded 60-term
    sample enriched for A-vs-baseline proposal disagreements.
* Final-ontology evaluation: taxonomy, defined classes, general relations,
  named entities, and critic exclusion/demotion decisions.

All experts receive the same sampled items. Row order and A/B definition order
are independently randomized per expert. Stable row IDs and all hidden fields
are written to one separate blinding key.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.formatting.rule import CellIsRule
from openpyxl.formatting.rule import FormulaRule
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from evaluation_study.layer1_analysis import validate_paired_results
from evaluation_study.paths import (
    ABLATION_OUTPUT,
    APPROVED_ONTOLOGY_DIR,
    DISPLAY_TEXT_CONFIG,
    FILTERED_TERMS,
    STUDY_CONFIG,
)
from evaluation_study.study_config import display_label, get_display_registry, get_study_config
from src.utils.csv_io import read_csv, write_csv
from src.utils.ontology_config import get_config


CONDITIONS = ("A", "B", "C", "D")
DEFAULT_TERM_SAMPLE = 100
DEFAULT_CATEGORY_TERM_SAMPLE = 60
DEFAULT_EXPERTS = 3
DEFAULT_SEED = 42
DATA_HEADER_ROW = 5
DATA_START_ROW = DATA_HEADER_ROW + 1

VISIBLE_SHEET_NAMES = {
    "Category_Guide": "Category Guide",
    "Category_Correct": "Category Review",
    "Defined_Classes": "Definition Review",
    "Individuals": "Named Items",
    "Meaning_Preservation": "Removed or Rewritten Terms",
}

COLUMN_DISPLAY_NAMES = {
    "Category_Guide": {
        "Positive_Example": "Example",
        "Not_This": "Different from",
    },
    "Representation": {
        "Relevance (1-5/Unsure)": "Pre-Salt relevance",
        "Definition_1": "Definition 1",
        "Definition_2": "Definition 2",
        "Quality_1 (1-5/Unsure)": "Definition 1 accuracy",
        "Quality_2 (1-5/Unsure)": "Definition 2 accuracy",
        "Preference (1/2/Tie/Unsure)": "Which definition is better?",
        "Notes": "Optional notes",
    },
    "Category_Correct": {
        "Reference_Definition": "Meaning of the term",
        "Proposed_Category": "Suggested category",
        "Category_Definition": "What the category means",
        "Correct (Yes/Partial/No/Unsure)": "Does the term fit this category?",
        "Notes": "Optional notes",
    },
    "Taxonomy": {
        "Child_Concept": "More specific term",
        "Parent_Concept": "Broader term",
        "Relationship_Correct (Yes/Partial/No/Unsure)": "Is the first term a type of the broader term?",
        "Useful_PreSalt_Distinction (Yes/No/Unsure)": "Is this separate term useful for Pre-Salt geology?",
        "Notes": "Optional notes",
    },
    "Defined_Classes": {
        "Concept": "Term",
        "Proposed_Definition": "Definition to review",
        "Definition_Verdict (Correct/Partly correct/Incorrect/Unsure)": "Is this definition geologically correct?",
        "Issue_Reason (select for Partly/Incorrect)": "Main problem with the definition",
        "Notes": "Optional notes",
    },
    "Relations": {
        "Relation_Statement": "Statement to review",
        "Relation_Verdict": "How accurate is this statement?",
        "Notes": "Optional notes",
    },
    "Individuals": {
        "Named_Entity": "Named item",
        "Proposed_Type": "Suggested type",
        "Specific_Named_Entity (Yes/No/Unsure)": "Is this one specific named item?",
        "Type_Correct (Yes/Partial/No/Unsure)": "Is the suggested type correct?",
        "Notes": "Optional notes",
    },
    "Meaning_Preservation": {
        "Concept": "Term",
        "Reference_Definition": "Meaning of the term",
        "Before": "How it was represented before",
        "After": "How it is represented now",
        "Meaning_Preserved (Fully/Mostly/No/Unsure)": "Does the new treatment keep the term's geological meaning?",
        "Appropriate_for_Lean_Core (Yes/With concern/No/Unsure)": "Is the new treatment suitable for the main Pre-Salt model?",
        "Preferred_Outcome (for Mostly/No)": "What should happen instead?",
        "Notes": "Optional notes",
    },
}
FINAL_SAMPLE_SIZES = {
    "Taxonomy": 40,
    "Defined_Classes": 13,
    "Relations": 25,
    "Individuals": 15,
    "Meaning_Preservation": 40,
}
APPROVED_POPULATIONS = {
    "Taxonomy": 185,
    "Defined_Classes": 13,
    "Relations": 280,
    "Individuals": 58,
    "Meaning_Preservation": 103,
}


@dataclass(frozen=True)
class StudyInputs:
    terms: pd.DataFrame
    nld: dict[str, pd.DataFrame]
    categories: dict[str, pd.DataFrame]
    refined_categories: pd.DataFrame
    taxonomy: pd.DataFrame
    defined_classes: pd.DataFrame
    relations: pd.DataFrame
    individuals: pd.DataFrame
    demotions: pd.DataFrame
    class_fates: pd.DataFrame


def visible_sheet_name(internal_name: str) -> str:
    return VISIBLE_SHEET_NAMES.get(internal_name, internal_name)


def visible_column_name(sheet_name: str, internal_name: str) -> str:
    return COLUMN_DISPLAY_NAMES.get(sheet_name, {}).get(internal_name, internal_name)


def canonicalize_workbook_frame(sheet_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    inverse = {
        visible: internal
        for internal, visible in COLUMN_DISPLAY_NAMES.get(sheet_name, {}).items()
    }
    return frame.rename(columns=inverse)


def _normalise(value: object) -> str:
    return str(value).strip().casefold()


def _accentfold(value: object) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", _normalise(value))
        if not unicodedata.combining(character)
    )


def _collapse_accent_variants(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one canonical spelling for terms that differ only by accents."""
    work = frame.copy()
    work["_accent_key"] = work["Readable_Term"].map(_accentfold)
    work["_diacritic_count"] = work["Readable_Term"].map(
        lambda value: sum(
            bool(unicodedata.combining(character))
            for character in unicodedata.normalize("NFKD", str(value))
        )
    )
    work = work.sort_values(
        ["_accent_key", "_diacritic_count", "Frequency", "Readable_Term"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    return work.drop_duplicates("_accent_key", keep="first").drop(
        columns=["_accent_key", "_diacritic_count"]
    )


def _require_columns(df: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_required(path: str | Path, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return read_csv(path)


def load_reviewed_reference_definitions(
    path: str | Path,
    required_terms: set[str],
) -> dict[str, str]:
    """Load one approved, condition-independent definition per required term."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Reviewed reference definitions not found: {source}")
    frame = (
        pd.read_excel(source, sheet_name="Review", header=2, engine="openpyxl")
        if source.suffix.lower() == ".xlsx"
        else read_csv(source)
    )
    _require_columns(
        frame,
        {"Term", "Reference_Definition", "Review_Status"},
        "Reviewed reference definitions",
    )
    work = frame[["Term", "Reference_Definition", "Review_Status"]].copy()
    work["_key"] = work["Term"].map(_normalise)
    if work["_key"].duplicated().any():
        duplicates = work.loc[work["_key"].duplicated(keep=False), "Term"].tolist()
        raise ValueError(f"Reviewed reference definitions contain duplicates: {duplicates[:10]}")
    required_keys = {_normalise(term) for term in required_terms}
    available = set(work["_key"])
    missing = sorted(required_keys - available)
    if missing:
        raise ValueError(f"Reviewed reference definitions missing terms: {missing[:10]}")
    required = work[work["_key"].isin(required_keys)].copy()
    unapproved = required[
        required["Review_Status"].fillna("").astype(str).str.strip().str.upper() != "APPROVED"
    ]["Term"].tolist()
    if unapproved:
        raise ValueError(f"Reference definitions are not APPROVED: {unapproved[:10]}")
    empty = required[
        required["Reference_Definition"].fillna("").astype(str).str.strip().eq("")
    ]["Term"].tolist()
    if empty:
        raise ValueError(f"Approved reference definitions are empty: {empty[:10]}")
    return dict(
        zip(
            required["_key"],
            required["Reference_Definition"].astype(str).str.strip(),
        )
    )


def load_study_inputs(
    ablation_dir: str,
    ontology_dir: str,
    terms_path: str,
    expected_term_count: int = 407,
) -> StudyInputs:
    """Load and validate the complete ablation and approved ontology artifacts."""
    terms = _read_required(terms_path, "Filtered terms")
    _require_columns(terms, {"Readable_Term", "Frequency"}, "Filtered terms")
    if terms["Readable_Term"].astype(str).duplicated().any():
        raise ValueError("Filtered terms contain duplicate Readable_Term values")
    if len(terms) != expected_term_count:
        raise ValueError(f"Expected {expected_term_count} terms; found {len(terms)}")

    nld: dict[str, pd.DataFrame] = {}
    categories: dict[str, pd.DataFrame] = {}
    merged_categories = []
    expected_terms = set(terms["Readable_Term"].astype(str))
    for condition in CONDITIONS:
        nld_path = os.path.join(ablation_dir, f"nld_{condition}.csv")
        category_path = os.path.join(ablation_dir, f"cat_{condition}.csv")
        nld[condition] = _read_required(nld_path, f"Condition {condition} NLD")
        categories[condition] = _read_required(
            category_path,
            f"Condition {condition} categories",
        )
        _require_columns(nld[condition], {"Term", "NLD"}, f"Condition {condition} NLD")
        if set(nld[condition]["Term"].astype(str)) != expected_terms:
            raise ValueError(f"Condition {condition} NLD term set does not match filtered terms")
        category_copy = categories[condition].copy()
        category_copy["Condition"] = condition
        merged_categories.append(category_copy)

    validate_paired_results(
        pd.concat(merged_categories, ignore_index=True),
        expected_term_count=expected_term_count,
    )

    ontology_path = Path(ontology_dir)
    return StudyInputs(
        terms=terms,
        nld=nld,
        categories=categories,
        refined_categories=_read_required(
            ontology_path / "classify_categories.csv",
            "CQ-filtered categories",
        ),
        taxonomy=_read_required(ontology_path / "validate_taxonomy.csv", "Validated taxonomy"),
        defined_classes=_read_required(
            ontology_path / "validate_defined_classes.csv",
            "Validated defined classes",
        ),
        relations=_read_required(ontology_path / "validate_relations.csv", "Validated relations"),
        individuals=_read_required(
            ontology_path / "validate_instances.csv",
            "Validated individuals",
        ),
        demotions=_read_required(ontology_path / "validate_demotions.csv", "Validated demotions"),
        class_fates=_read_required(
            ontology_path / "validate_class_fates.csv",
            "Validated class fates",
        ),
    )


def _category_to_tier() -> dict[str, str]:
    cfg = get_config()
    mapping = {"NOT_CLASSIFIED": "NOT_CLASSIFIED"}
    for ontology_key in cfg.waterfall_ontologies():
        tier = cfg.ontologies[ontology_key].eval_tier.upper()
        for category in cfg.categories_for(ontology_key):
            mapping[category] = tier
    return mapping


def _frequency_bands(frequencies: pd.Series) -> pd.Series:
    """Assign equal-count low/middle/high corpus-frequency bands."""
    if len(frequencies) < 3:
        return pd.Series(["Middle"] * len(frequencies), index=frequencies.index)
    ranks = pd.to_numeric(frequencies, errors="raise").rank(method="first")
    return pd.qcut(ranks, q=3, labels=["Low", "Middle", "High"]).astype(str)


def _proportional_sample(
    frame: pd.DataFrame,
    n_rows: int,
    strata: list[str],
    seed: int,
) -> pd.DataFrame:
    """Take an exact-size seeded sample using largest-remainder allocation."""
    if n_rows < 1:
        raise ValueError("Sample size must be positive")
    if len(frame) < n_rows:
        raise ValueError(f"Cannot sample {n_rows} rows from a population of {len(frame)}")
    if len(frame) == n_rows:
        return frame.copy().reset_index(drop=True)
    _require_columns(frame, set(strata), "Sampling frame")

    work = frame.copy()
    work["_Stratum"] = work[strata].fillna("MISSING").astype(str).agg(" | ".join, axis=1)
    sizes = work.groupby("_Stratum", sort=True).size()
    ideals = sizes * n_rows / len(work)
    quotas = ideals.apply(math.floor).astype(int)
    remaining = n_rows - int(quotas.sum())
    while remaining:
        candidates = [name for name in sizes.index if quotas[name] < sizes[name]]
        if not candidates:
            raise RuntimeError("Unable to allocate the requested stratified sample")
        candidates.sort(key=lambda name: (-(ideals[name] - quotas[name]), str(name)))
        for name in candidates:
            if remaining == 0:
                break
            quotas[name] += 1
            remaining -= 1

    samples = []
    for name in sizes.index:
        group = work[work["_Stratum"] == name]
        group_seed = (
            seed + int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8], 16)
        ) % (2**32 - 1)
        samples.append(group.sample(n=int(quotas[name]), random_state=group_seed))
    return pd.concat(samples, ignore_index=True).drop(columns="_Stratum")


def _repair_contrast_coverage(
    sampled: pd.DataFrame,
    population: pd.DataFrame,
    contrast_columns: list[str],
    minimum: int,
    seed: int,
) -> pd.DataFrame:
    """Swap the fewest rows needed to meet overlapping contrast quotas."""
    if any(int(population[column].sum()) < minimum for column in contrast_columns):
        unavailable = {
            column: int(population[column].sum())
            for column in contrast_columns
            if int(population[column].sum()) < minimum
        }
        raise ValueError(f"Insufficient population for category contrast quotas: {unavailable}")

    result = sampled.copy().reset_index(drop=True)
    for target in contrast_columns:
        while int(result[target].sum()) < minimum:
            selected = set(result["_key"])
            counts = {column: int(result[column].sum()) for column in contrast_columns}
            candidates = population[
                population[target] & ~population["_key"].isin(selected)
            ]
            removable = result[~result[target]]
            choices = []
            for candidate_index, candidate in candidates.iterrows():
                for remove_index, removed in removable.iterrows():
                    new_counts = {
                        column: counts[column]
                        + int(candidate[column])
                        - int(removed[column])
                        for column in contrast_columns
                    }
                    if any(
                        new_counts[column]
                        < (minimum if counts[column] >= minimum else counts[column])
                        for column in contrast_columns
                    ):
                        continue
                    deficit_reduction = sum(
                        max(0, minimum - counts[column])
                        - max(0, minimum - new_counts[column])
                        for column in contrast_columns
                    )
                    stratum_penalty = sum(
                        candidate[column] != removed[column]
                        for column in ("Tier_A", "Frequency_Band")
                    )
                    stable_rank = hashlib.sha256(
                        f"{seed}|{target}|{candidate['_key']}|{removed['_key']}".encode(
                            "utf-8"
                        )
                    ).hexdigest()
                    choices.append(
                        (
                            -deficit_reduction,
                            stratum_penalty,
                            -sum(new_counts.values()),
                            stable_rank,
                            candidate_index,
                            remove_index,
                        )
                    )
            if not choices:
                raise ValueError(f"Unable to satisfy category contrast quota for {target}")
            *_, candidate_index, remove_index = min(choices)
            result.loc[remove_index] = population.loc[candidate_index]
    return result


def select_representation_terms(
    terms: pd.DataFrame,
    condition_a: pd.DataFrame,
    n_terms: int = DEFAULT_TERM_SAMPLE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Sample proportionally by Condition-A ontology tier and frequency band."""
    _require_columns(terms, {"Readable_Term", "Frequency"}, "Filtered terms")
    _require_columns(condition_a, {"Term", "Category"}, "Condition A categories")
    if len(condition_a) != len(terms):
        raise ValueError("Condition A and filtered terms have different row counts")

    a_categories = condition_a[["Term", "Category"]].copy()
    a_categories["_key"] = a_categories["Term"].map(_normalise)
    if a_categories["_key"].duplicated().any():
        raise ValueError("Condition A contains duplicate terms")
    frame = terms[["Readable_Term", "Frequency"]].copy()
    frame["_key"] = frame["Readable_Term"].map(_normalise)
    frame = frame.merge(a_categories[["_key", "Category"]], on="_key", how="left", validate="one_to_one")
    if frame["Category"].isna().any():
        missing = frame.loc[frame["Category"].isna(), "Readable_Term"].tolist()[:10]
        raise ValueError(f"Condition A categories missing for terms: {missing}")
    frame["Tier_A"] = frame["Category"].map(_category_to_tier())
    if frame["Tier_A"].isna().any():
        unknown = sorted(frame.loc[frame["Tier_A"].isna(), "Category"].unique())
        raise ValueError(f"Unknown Condition A categories: {unknown}")
    frame["Frequency_Band"] = _frequency_bands(frame["Frequency"])

    sampled = _proportional_sample(
        frame,
        n_rows=n_terms,
        strata=["Tier_A", "Frequency_Band"],
        seed=seed,
    )
    sampled = sampled.sort_values("Readable_Term", key=lambda values: values.str.casefold()).reset_index(drop=True)
    sampled.insert(0, "Row_ID", [f"REP-{index:03d}" for index in range(1, len(sampled) + 1)])
    return sampled.drop(columns="_key")


def select_category_terms(
    terms: pd.DataFrame,
    categories: dict[str, pd.DataFrame],
    n_terms: int = DEFAULT_CATEGORY_TERM_SAMPLE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Sample A-vs-baseline disagreements across overlap, tier, and frequency."""
    _require_columns(terms, {"Readable_Term", "Frequency"}, "Filtered terms")
    if set(categories) != set(CONDITIONS):
        raise ValueError(f"Category conditions must be exactly {CONDITIONS}")

    frame = terms[["Readable_Term", "Frequency"]].copy()
    frame["_key"] = frame["Readable_Term"].map(_normalise)
    for condition in CONDITIONS:
        condition_frame = categories[condition][["Term", "Category"]].copy()
        condition_frame["_key"] = condition_frame["Term"].map(_normalise)
        if condition_frame["_key"].duplicated().any():
            raise ValueError(f"Condition {condition} contains duplicate terms")
        frame = frame.merge(
            condition_frame[["_key", "Category"]].rename(
                columns={"Category": f"Category_{condition}"}
            ),
            on="_key",
            how="left",
            validate="one_to_one",
        )
    category_columns = [f"Category_{condition}" for condition in CONDITIONS]
    if frame[category_columns].isna().any().any():
        raise ValueError("One or more conditions are missing filtered terms")

    for comparator in ("B", "C", "D"):
        frame[f"A_vs_{comparator}"] = (
            frame["Category_A"] != frame[f"Category_{comparator}"]
        )
    frame = frame[frame[["A_vs_B", "A_vs_C", "A_vs_D"]].any(axis=1)].copy()
    frame = _collapse_accent_variants(frame)
    frame["Disagreement_Pattern"] = [
        "".join(
            comparator
            for comparator in ("B", "C", "D")
            if row[f"A_vs_{comparator}"]
        )
        for row in frame.to_dict("records")
    ]
    frame["Tier_A"] = frame["Category_A"].map(_category_to_tier())
    if frame["Tier_A"].isna().any():
        unknown = sorted(frame.loc[frame["Tier_A"].isna(), "Category_A"].unique())
        raise ValueError(f"Unknown Condition A categories: {unknown}")
    frame["Frequency_Band"] = _frequency_bands(frame["Frequency"])

    sampled = _proportional_sample(
        frame,
        n_rows=n_terms,
        strata=["Disagreement_Pattern", "Tier_A", "Frequency_Band"],
        seed=seed + 200,
    )
    contrast_columns = [f"A_vs_{comparator}" for comparator in ("B", "C", "D")]
    minimum_coverage = min(25, math.ceil(n_terms * 5 / 12))
    sampled = _repair_contrast_coverage(
        sampled,
        frame,
        contrast_columns,
        minimum_coverage,
        seed + 201,
    )
    sampled = sampled.sort_values(
        "Readable_Term",
        key=lambda values: values.str.casefold(),
    ).reset_index(drop=True)
    sampled.insert(
        0,
        "Row_ID",
        [f"CATERM-{index:03d}" for index in range(1, len(sampled) + 1)],
    )
    return sampled.drop(columns="_key")


def build_final_fates(inputs: StudyInputs) -> dict[str, str]:
    """Partition every original term into one hidden downstream final fate."""
    _require_columns(inputs.refined_categories, {"Term"}, "CQ-filtered categories")
    _require_columns(inputs.taxonomy, {"Term"}, "Validated taxonomy")
    _require_columns(inputs.individuals, {"Term"}, "Validated individuals")
    _require_columns(inputs.demotions, {"Term"}, "Validated demotions")
    _require_columns(inputs.class_fates, {"term", "action"}, "Validated class fates")

    original = {_normalise(term) for term in inputs.terms["Readable_Term"]}
    refined = {_normalise(term) for term in inputs.refined_categories["Term"]}
    groups = {
        "FINAL_CLASS": {_normalise(term) for term in inputs.taxonomy["Term"]} & original,
        "FINAL_INDIVIDUAL": {_normalise(term) for term in inputs.individuals["Term"]} & original,
        "DEMOTED": {_normalise(term) for term in inputs.demotions["Term"]} & original,
        "CRITIC_EXCLUDED": {
            _normalise(term)
            for term in inputs.class_fates.loc[
                inputs.class_fates["action"].astype(str).str.startswith("DROP"),
                "term",
            ]
        } & original,
        "CQ_EXCLUDED": original - refined,
    }
    memberships: dict[str, list[str]] = {}
    for fate, terms in groups.items():
        for term in terms:
            memberships.setdefault(term, []).append(fate)
    overlaps = {term: fates for term, fates in memberships.items() if len(fates) != 1}
    missing = sorted(original - set(memberships))
    if overlaps or missing:
        raise ValueError(
            f"Final fates do not partition original terms (overlaps={overlaps}, missing={missing[:10]})"
        )
    return {term: fates[0] for term, fates in memberships.items()}


def _lookup_by_term(df: pd.DataFrame, value_column: str, label: str) -> dict[str, object]:
    _require_columns(df, {"Term", value_column}, label)
    work = df[["Term", value_column]].copy()
    work["_key"] = work["Term"].map(_normalise)
    if work["_key"].duplicated().any():
        raise ValueError(f"{label} has duplicate terms")
    return dict(zip(work["_key"], work[value_column]))


def build_representation_items(
    sample: pd.DataFrame,
    inputs: StudyInputs,
    final_fates: dict[str, str],
) -> pd.DataFrame:
    """Attach hidden A/B definitions and final fates to sampled terms."""
    nld_a = _lookup_by_term(inputs.nld["A"], "NLD", "Condition A NLD")
    nld_b = _lookup_by_term(inputs.nld["B"], "NLD", "Condition B NLD")
    context_column = "Context_Used" if "Context_Used" in inputs.nld["A"].columns else None
    context_a = (
        _lookup_by_term(inputs.nld["A"], context_column, "Condition A NLD")
        if context_column
        else {}
    )
    rows = []
    for item in sample.to_dict("records"):
        term = item["Readable_Term"]
        key = _normalise(term)
        if key not in nld_a or key not in nld_b or key not in final_fates:
            raise ValueError(f"Incomplete representation data for '{term}'")
        rows.append({
            "Row_ID": item["Row_ID"],
            "Term": term,
            "Frequency": item["Frequency"],
            "Frequency_Band": item["Frequency_Band"],
            "Category_A": item["Category"],
            "Tier_A": item["Tier_A"],
            "Final_Fate": final_fates[key],
            "Context_Used_A": context_a.get(key, ""),
            "NLD_A": nld_a[key],
            "NLD_B": nld_b[key],
        })
    return pd.DataFrame(rows)


def _category_descriptions() -> dict[str, str]:
    cfg = get_config()
    descriptions = {
        "NOT_CLASSIFIED": "The term does not fit any category offered by the upper-ontology waterfall."
    }
    for ontology_key in cfg.waterfall_ontologies():
        for line in cfg.llm_definitions_block(ontology_key, categorizer_only=True).splitlines():
            if ":" in line:
                label, description = line.split(":", 1)
                descriptions[label.strip()] = description.strip()
    for category, entry in get_display_registry().categories.items():
        if entry.definition:
            matching = next(
                (label for label in descriptions if label.casefold() == category),
                category,
            )
            descriptions[matching] = entry.definition
    return descriptions


def build_term_glosses(nld: pd.DataFrame) -> dict[str, str]:
    """Return shared context only for terms found ambiguous in the pilot."""
    _require_columns(nld, {"Term", "NLD"}, "NLD gloss source")
    registry = get_display_registry()
    nld_lookup = _lookup_by_term(nld, "NLD", "NLD gloss source")
    glosses: dict[str, str] = {}
    for key in registry.ambiguous_terms:
        if key in registry.term_glosses:
            glosses[key] = registry.term_glosses[key]
            continue
        text = str(nld_lookup.get(key, "")).strip()
        if not text:
            continue
        glosses[key] = _first_sentence(text)
    return glosses


def _first_sentence(value: object) -> str:
    text = str(value).strip()
    sentence_end = next(
        (index + 1 for index, character in enumerate(text) if character in ".!?"),
        len(text),
    )
    return text[:sentence_end].strip()


def _quoted_replacement(value: object) -> str:
    match = re.search(r"'(.*?)'", str(value))
    return match.group(1).strip() if match else ""


def _meaning_after_state(row: pd.Series) -> str:
    action = str(row["action"])
    category = display_label(row["category"], "category").lower()
    if action == "DEMOTE_TO_PROPERTY":
        base = display_label(row["Base_Class"], "category").lower()
        relation = display_label(row["Property"], "property")
        filler = display_label(row["Filler"], "category").lower()
        return (
            f"Not kept as a separate concept. Its meaning is retained as "
            f"{base} that {relation} {filler}."
        )
    if action == "DROP_AS_REDUNDANT":
        replacement = _quoted_replacement(row.get("reason", ""))
        if replacement:
            return (
                f"Not kept separately. Its meaning is covered by "
                f"{display_label(replacement, 'category')}."
            )
        return (
            f"Not kept separately because the broader {category} concept "
            "already covers the same meaning."
        )
    if action == "DROP_AS_MIXIN":
        return (
            f"Not kept as a separate type. The broader {category} concept "
            "remains, but this combined distinction is not a separate core concept."
        )
    basis = str(row.get("drop_basis", ""))
    if basis == "NARROW_EXTENSION_DETAIL":
        return (
            "Not kept in the lean core. This narrower distinction is deferred "
            "to a possible domain extension."
        )
    if basis == "STACKED_CONTEXT":
        return (
            "Not kept separately. Its combined context can be expressed using "
            "the retained broader concepts."
        )
    return (
        f"Not kept separately in the lean core. Broader {category} concepts "
        "remain, while this detail is outside the core vocabulary."
    )


def build_category_guide() -> pd.DataFrame:
    """Build the visible category reference from formal config plus display text."""
    descriptions = _category_descriptions()
    tiers = _category_to_tier()
    registry = get_display_registry()
    rows = []
    for formal_label, definition in sorted(
        descriptions.items(),
        key=lambda item: display_label(item[0], "category").casefold(),
    ):
        entry = registry.categories.get(formal_label.casefold())
        rows.append({
            "Category": display_label(formal_label, "category"),
            "Meaning": definition,
            "Positive_Example": entry.example if entry else "",
            "Not_This": entry.counterexample if entry else "",
            "Source": entry.source if entry and entry.source else tiers.get(formal_label, ""),
        })
    return pd.DataFrame(rows)


def build_category_items(
    sample: pd.DataFrame,
    categories: dict[str, pd.DataFrame],
    final_fates: dict[str, str],
    term_glosses: dict[str, str] | None = None,
    reference_definitions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Deduplicate identical term/category assignments across A/B/C/D."""
    sampled_terms = {_normalise(term): term for term in sample["Readable_Term"]}
    assignments: dict[tuple[str, str], list[str]] = {}
    for condition in CONDITIONS:
        frame = categories[condition]
        _require_columns(frame, {"Term", "Category"}, f"Condition {condition} categories")
        for row in frame[["Term", "Category"]].to_dict("records"):
            term_key = _normalise(row["Term"])
            if term_key in sampled_terms:
                assignments.setdefault((term_key, str(row["Category"])), []).append(condition)

    descriptions = _category_descriptions()
    tiers = _category_to_tier()
    term_glosses = term_glosses or {}
    reference_definitions = reference_definitions or {}
    rows = []
    ordered = sorted(assignments.items(), key=lambda item: (sampled_terms[item[0][0]].casefold(), item[0][1]))
    for index, ((term_key, category), conditions) in enumerate(ordered, 1):
        rows.append({
            "Row_ID": f"CAT-{index:04d}",
            "Term": sampled_terms[term_key],
            "Term_Gloss": term_glosses.get(term_key, ""),
            "Reference_Definition": reference_definitions.get(term_key, ""),
            "Assigned_Category": category,
            "Category_Description": descriptions.get(category, ""),
            "Conditions": ",".join(sorted(conditions)),
            "Tier": tiers[category],
            "Final_Fate": final_fates[term_key],
        })
    result = pd.DataFrame(rows)
    if result["Term"].nunique() != len(sampled_terms):
        raise ValueError("Category items do not cover every sampled representation term")
    return result


def _assign_stable_ids(
    frame: pd.DataFrame,
    prefix: str,
    sort_columns: list[str],
) -> pd.DataFrame:
    result = frame.sort_values(
        sort_columns,
        key=lambda values: values.astype(str).str.casefold(),
    ).reset_index(drop=True)
    result.insert(0, "Row_ID", [f"{prefix}-{index:03d}" for index in range(1, len(result) + 1)])
    return result


def select_final_ontology_items(
    inputs: StudyInputs,
    seed: int = DEFAULT_SEED,
    strict_approved_populations: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build the five approved final-ontology sampling frames and samples."""
    taxonomy = inputs.taxonomy[
        (inputs.taxonomy["Relationship_Type"] == "rdfs:subClassOf")
        & inputs.taxonomy["Parent_Term"].notna()
    ].copy()
    defined = inputs.defined_classes.copy()
    relations = inputs.relations[
        (inputs.relations["Validation_Status"].astype(str).str.upper() == "ACCEPTED")
    ].copy()
    individuals = inputs.individuals.copy()
    source_terms = set(inputs.terms["Readable_Term"].map(_normalise))
    decisions = inputs.class_fates[
        inputs.class_fates["action"].astype(str).str.startswith(("DROP", "DEMOTE"))
        & inputs.class_fates["term"].map(_normalise).isin(source_terms)
    ].copy()
    decisions["Decision_Type"] = np.where(
        decisions["action"].astype(str).str.startswith("DEMOTE"),
        "DEMOTE",
        "EXCLUDE",
    )
    demotion_details = inputs.demotions[
        ["Term", "Base_Class", "Property", "Filler"]
    ].copy()
    demotion_details["_term_key"] = demotion_details["Term"].map(_normalise)
    decisions["_term_key"] = decisions["term"].map(_normalise)
    decisions = decisions.merge(
        demotion_details.drop(columns="Term"),
        on="_term_key",
        how="left",
        validate="one_to_one",
    ).drop(columns="_term_key")
    nld_lookup = _lookup_by_term(inputs.nld["A"], "NLD", "Condition A NLD")
    decisions["Reference_Definition"] = decisions["term"].map(
        lambda value: _first_sentence(nld_lookup.get(_normalise(value), ""))
    )
    decisions["Before_State"] = decisions["category"].map(
        lambda value: (
            "Represented as a separate concept of the kind: "
            f"{display_label(value, 'category')}."
        )
    )
    decisions["After_State"] = decisions.apply(_meaning_after_state, axis=1)
    populations = {
        "Taxonomy": taxonomy,
        "Defined_Classes": defined,
        "Relations": relations,
        "Individuals": individuals,
        "Meaning_Preservation": decisions,
    }
    required_columns = {
        "Taxonomy": {"Term", "Parent_Term", "Category", "Is_Intermediate"},
        "Defined_Classes": {"Bearer", "Genus", "Property", "Filler"},
        "Relations": {"Term", "Property", "Filler", "Evidence", "Relation_Scope"},
        "Individuals": {"Term", "Target_Class", "Reason"},
        "Meaning_Preservation": {
            "term", "category", "action", "reason", "Decision_Type",
            "Reference_Definition", "Before_State", "After_State",
        },
    }
    for name, frame in populations.items():
        _require_columns(frame, required_columns[name], f"{name} population")
        if frame[list(required_columns[name])].isna().any().any():
            raise ValueError(f"{name} population has missing required values")
    if strict_approved_populations:
        observed = {name: len(frame) for name, frame in populations.items()}
        if observed != APPROVED_POPULATIONS:
            raise ValueError(
                f"Approved ontology populations changed: expected {APPROVED_POPULATIONS}, observed {observed}"
            )

    samples = {
        "Taxonomy": _proportional_sample(
            taxonomy,
            FINAL_SAMPLE_SIZES["Taxonomy"],
            ["Category", "Is_Intermediate"],
            seed + 300,
        ),
        "Defined_Classes": defined.copy(),
        "Relations": _proportional_sample(
            relations,
            FINAL_SAMPLE_SIZES["Relations"],
            ["Relation_Scope"],
            seed + 400,
        ),
        "Individuals": _proportional_sample(
            individuals,
            FINAL_SAMPLE_SIZES["Individuals"],
            ["Target_Class"],
            seed + 500,
        ),
        "Meaning_Preservation": _proportional_sample(
            decisions,
            FINAL_SAMPLE_SIZES["Meaning_Preservation"],
            ["Decision_Type", "action"],
            seed + 600,
        ),
    }
    samples["Taxonomy"] = _assign_stable_ids(
        samples["Taxonomy"], "TAX", ["Term", "Parent_Term"]
    )
    samples["Defined_Classes"] = _assign_stable_ids(
        samples["Defined_Classes"], "DEF", ["Bearer", "Property", "Filler"]
    )
    samples["Relations"] = _assign_stable_ids(
        samples["Relations"], "REL", ["Term", "Property", "Filler"]
    )
    samples["Individuals"] = _assign_stable_ids(
        samples["Individuals"], "IND", ["Term", "Target_Class"]
    )
    samples["Meaning_Preservation"] = _assign_stable_ids(
        samples["Meaning_Preservation"], "DEC", ["term", "action"]
    )
    return samples


def _lower_initial(value: object) -> str:
    text = str(value).strip()
    return text[:1].lower() + text[1:] if text else text


def _defined_class_sentence(
    bearer: object,
    genus: object,
    property_name: object,
    filler: object,
) -> str:
    registry = get_display_registry()
    sentence = registry.defined_class_sentences.get(str(bearer).casefold())
    if sentence:
        return sentence
    subject = _lower_initial(display_label(bearer, "category"))
    base_kind = display_label(genus, "category").lower()
    feature = (
        f"{display_label(property_name, 'property')} "
        f"{display_label(filler, 'category').lower()}"
    )
    article = "an" if base_kind[:1] in "aeiou" else "a"
    subject_article = "An" if subject[:1] in "aeiou" else "A"
    return f"{subject_article} {subject} is {article} {base_kind} that {feature}."


def _relation_statement(
    subject: object,
    property_name: object,
    filler: object,
    scope: object,
) -> str:
    prefix = {
        "generic": "Generally, ",
        "corpus_context": "In some reported Pre-Salt contexts, ",
        "individual_fact": "For this named entity, ",
    }.get(str(scope).strip().casefold(), "In the proposed model, ")
    return (
        f"{prefix}{_lower_initial(subject)} "
        f"{display_label(property_name, 'property')} "
        f"{display_label(filler, 'category').lower()}."
    )


def _shuffle(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    return frame.sample(frac=1, random_state=seed).reset_index(drop=True)


def _representation_for_expert(
    items: pd.DataFrame,
    expert_id: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    visible_rows = []
    key_rows = []
    for row in items.to_dict("records"):
        if rng.random() < 0.5:
            first, second = row["NLD_A"], row["NLD_B"]
            first_condition, second_condition = "A", "B"
        else:
            first, second = row["NLD_B"], row["NLD_A"]
            first_condition, second_condition = "B", "A"
        visible_rows.append({
            "Row_ID": row["Row_ID"],
            "Term": row["Term"],
            "Relevance (1-5/Unsure)": "",
            "Definition_1": first,
            "Definition_2": second,
            "Quality_1 (1-5/Unsure)": "",
            "Quality_2 (1-5/Unsure)": "",
            "Preference (1/2/Tie/Unsure)": "",
            "Notes": "",
        })
        key_rows.append({
            "Expert_ID": expert_id,
            "Sheet": "Representation",
            **row,
            "Definition_1_Condition": first_condition,
            "Definition_2_Condition": second_condition,
        })
    visible = _shuffle(pd.DataFrame(visible_rows), seed + 1)
    return visible, pd.DataFrame(key_rows)


def _category_for_expert(
    items: pd.DataFrame,
    expert_id: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    visible = items[
        ["Row_ID", "Term", "Reference_Definition", "Assigned_Category", "Category_Description"]
    ].copy()
    visible = visible.rename(columns={
        "Assigned_Category": "Proposed_Category",
        "Category_Description": "Category_Definition",
    })
    unclassified = visible["Proposed_Category"] == "NOT_CLASSIFIED"
    visible["Proposed_Category"] = visible["Proposed_Category"].map(
        lambda value: display_label(value, "category")
    )
    visible.loc[unclassified, "Proposed_Category"] = "Leave unclassified"
    visible.loc[unclassified, "Category_Definition"] = (
        "The system proposes leaving this term without an ontology category. "
        "Mark Yes when it is outside scope or not a reusable geological concept; "
        "mark No when it is a meaningful Pre-Salt concept that should receive some category."
    )
    visible["Correct (Yes/Partial/No/Unsure)"] = ""
    visible["Notes"] = ""
    key = items.copy()
    key.insert(0, "Sheet", "Category_Correct")
    key.insert(0, "Expert_ID", expert_id)
    return _shuffle(visible, seed), key


def _final_frames_for_expert(
    samples: dict[str, pd.DataFrame],
    expert_id: str,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], list[pd.DataFrame]]:
    taxonomy = samples["Taxonomy"]
    taxonomy_visible = pd.DataFrame({
        "Row_ID": taxonomy["Row_ID"],
        "Child_Concept": taxonomy["Term"],
        "Parent_Concept": taxonomy["Parent_Term"].map(
            lambda value: display_label(value, "category")
        ),
        "Relationship_Correct (Yes/Partial/No/Unsure)": "",
        "Useful_PreSalt_Distinction (Yes/No/Unsure)": "",
        "Notes": "",
    })

    defined = samples["Defined_Classes"]
    defined_visible = pd.DataFrame({
        "Row_ID": defined["Row_ID"],
        "Concept": defined["Bearer"],
        "Proposed_Definition": [
            _defined_class_sentence(bearer, genus, prop, filler)
            for bearer, genus, prop, filler in zip(
                defined["Bearer"],
                defined["Genus"],
                defined["Property"],
                defined["Filler"],
            )
        ],
        "Definition_Verdict (Correct/Partly correct/Incorrect/Unsure)": "",
        "Issue_Reason (select for Partly/Incorrect)": "",
        "Notes": "",
    })

    relations = samples["Relations"]
    relation_visible = pd.DataFrame({
        "Row_ID": relations["Row_ID"],
        "Relation_Statement": [
            _relation_statement(subject, prop, filler, scope)
            for subject, prop, filler, scope in zip(
                relations["Term"],
                relations["Property"],
                relations["Filler"],
                relations["Relation_Scope"],
            )
        ],
        "Relation_Verdict": "",
        "Notes": "",
    })

    individuals = samples["Individuals"]
    individual_visible = pd.DataFrame({
        "Row_ID": individuals["Row_ID"],
        "Named_Entity": individuals["Term"],
        "Proposed_Type": individuals["Target_Class"].map(
            lambda value: display_label(value, "category")
        ),
        "Specific_Named_Entity (Yes/No/Unsure)": "",
        "Type_Correct (Yes/Partial/No/Unsure)": "",
        "Notes": "",
    })

    decisions = samples["Meaning_Preservation"]
    decision_visible = pd.DataFrame({
        "Row_ID": decisions["Row_ID"],
        "Concept": decisions["term"],
        "Reference_Definition": decisions["Reference_Definition"],
        "Before": decisions["Before_State"],
        "After": decisions["After_State"],
        "Meaning_Preserved (Fully/Mostly/No/Unsure)": "",
        "Appropriate_for_Lean_Core (Yes/With concern/No/Unsure)": "",
        "Preferred_Outcome (for Mostly/No)": "",
        "Notes": "",
    })

    frames = {
        "Taxonomy": _shuffle(taxonomy_visible, seed + 10),
        "Defined_Classes": _shuffle(defined_visible, seed + 20),
        "Relations": _shuffle(relation_visible, seed + 30),
        "Individuals": _shuffle(individual_visible, seed + 40),
        "Meaning_Preservation": _shuffle(decision_visible, seed + 50),
    }
    keys = []
    for sheet_name, sample in samples.items():
        key = sample.copy()
        key.insert(0, "Sheet", sheet_name)
        key.insert(0, "Expert_ID", expert_id)
        keys.append(key)
    return frames, keys


_HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
_HEADER_FONT = Font(bold=True, color="FFFFFF")
_INPUT_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
_MISSING_FILL = PatternFill(start_color="F4CCCC", end_color="F4CCCC", fill_type="solid")
_WRAP = Alignment(wrap_text=True, vertical="top")
_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
_BORDER = Border(
    left=Side(style="thin", color="D9E2F3"),
    right=Side(style="thin", color="D9E2F3"),
    top=Side(style="thin", color="D9E2F3"),
    bottom=Side(style="thin", color="D9E2F3"),
)


_INPUT_VALIDATIONS = {
    "Representation": {
        "Relevance (1-5/Unsure)": "1,2,3,4,5,Unsure",
        "Quality_1 (1-5/Unsure)": "1,2,3,4,5,Unsure",
        "Quality_2 (1-5/Unsure)": "1,2,3,4,5,Unsure",
        "Preference (1/2/Tie/Unsure)": "1,2,Tie,Unsure",
        "Notes": None,
    },
    "Category_Correct": {
        "Correct (Yes/Partial/No/Unsure)": "Yes,Partial,No,Unsure",
        "Notes": None,
    },
    "Taxonomy": {
        "Relationship_Correct (Yes/Partial/No/Unsure)": "Yes,Partial,No,Unsure",
        "Useful_PreSalt_Distinction (Yes/No/Unsure)": "Yes,No,Unsure",
        "Notes": None,
    },
    "Defined_Classes": {
        "Definition_Verdict (Correct/Partly correct/Incorrect/Unsure)": "Correct,Partly correct,Incorrect,Unsure",
        "Issue_Reason (select for Partly/Incorrect)": "Wrong general type,Missing or wrong defining feature,Too broad,Too narrow,Unclear wording,Other,Unsure",
        "Notes": None,
    },
    "Relations": {
        "Relation_Verdict": "Generally true,Context-specific,Partly wrong,Incorrect,Unsure",
        "Notes": None,
    },
    "Individuals": {
        "Specific_Named_Entity (Yes/No/Unsure)": "Yes,No,Unsure",
        "Type_Correct (Yes/Partial/No/Unsure)": "Yes,Partial,No,Unsure",
        "Notes": None,
    },
    "Meaning_Preservation": {
        "Meaning_Preserved (Fully/Mostly/No/Unsure)": "Fully,Mostly,No,Unsure",
        "Appropriate_for_Lean_Core (Yes/With concern/No/Unsure)": "Yes,With concern,No,Unsure",
        "Preferred_Outcome (for Mostly/No)": "Keep as separate concept,Keep information but not as separate concept,Leave out,Unsure",
        "Notes": None,
    },
}

_OPTIONAL_INPUTS = {
    ("Defined_Classes", "Issue_Reason (select for Partly/Incorrect)"),
    ("Meaning_Preservation", "Preferred_Outcome (for Mostly/No)"),
    ("Representation", "Notes"),
    ("Category_Correct", "Notes"),
    ("Taxonomy", "Notes"),
    ("Defined_Classes", "Notes"),
    ("Relations", "Notes"),
    ("Individuals", "Notes"),
    ("Meaning_Preservation", "Notes"),
}


def _format_instructions(ws) -> None:
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 100
    ws.freeze_panes = "A2"
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = _WRAP
            cell.border = _BORDER
        first = str(row[0].value or "")
        if first and first == first.upper():
            row[0].font = Font(bold=True, color="1F4E78", size=12)


def _format_data_sheet(
    ws,
    input_validations: dict[str, str | None],
    sheet_name: str,
    instructions,
) -> None:
    last_column_letter = get_column_letter(ws.max_column)
    panel_rows = (
        (1, instructions.title, "1F4E78", "FFFFFF", 16, 30),
        (2, f"YOUR TASK\n{instructions.task}", "D9EAF7", "1F1F1F", 11, 48),
        (3, f"HOW TO ANSWER\n{instructions.guidance}", "EAF2F8", "1F1F1F", 10, 60),
        (
            4,
            instructions.answer_cue
            or "Complete every red cell. Yellow cells are optional. Some yellow cells may turn red after certain answers. Notes are optional.",
            "F3F6F8",
            "404040",
            9,
            28,
        ),
    )
    for row, text, fill, color, size, height in panel_rows:
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ws.max_column)
        cell = ws.cell(row, 1)
        cell.value = text
        cell.font = Font(bold=row <= 2, color=color, size=size)
        cell.fill = PatternFill(start_color=fill, end_color=fill, fill_type="solid")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = _BORDER
        ws.row_dimensions[row].height = height
    ws.freeze_panes = f"A{DATA_START_ROW}"
    ws.auto_filter.ref = (
        f"A{DATA_HEADER_ROW}:{last_column_letter}{ws.max_row}"
    )
    ws.sheet_view.showGridLines = False
    headers = {str(cell.value): cell.column for cell in ws[DATA_HEADER_ROW]}
    if "Row_ID" in headers:
        ws.column_dimensions[get_column_letter(headers["Row_ID"])].hidden = True
    for cell in ws[DATA_HEADER_ROW]:
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = _CENTER
        cell.border = _BORDER
    for column_index in range(1, ws.max_column + 1):
        header = str(ws.cell(DATA_HEADER_ROW, column_index).value or "")
        width = 15
        if any(token in header for token in ("Definition", "Description", "Meaning", "Rationale", "Evidence", "Excerpt", "Question", "Decision", "Statement", "represented", "treatment")):
            width = 70
        elif any(token in header for token in ("Term", "Concept", "Category", "Parent", "Subject", "Entity", "Type", "Broader", "specific", "Named item")):
            width = 28
        elif header in {"Optional notes", "Optional comments"} or "Outcome" in header or header.startswith("Suggested") or header.startswith("What should"):
            width = 35
        elif len(header) > 32:
            width = 28
        ws.column_dimensions[get_column_letter(column_index)].width = width
        for row_index in range(DATA_START_ROW, ws.max_row + 1):
            cell = ws.cell(row_index, column_index)
            cell.alignment = _WRAP
            cell.border = _BORDER

    for header, options in input_validations.items():
        visible_header = visible_column_name(sheet_name, header)
        column_index = headers[visible_header]
        column_letter = get_column_letter(column_index)
        optional = (sheet_name, header) in _OPTIONAL_INPUTS
        for row_index in range(DATA_START_ROW, ws.max_row + 1):
            ws.cell(row_index, column_index).fill = _INPUT_FILL
        if options:
            validation = DataValidation(
                type="list",
                formula1=f'"{options}"',
                allow_blank=optional,
                showErrorMessage=True,
                errorTitle="Response required",
                error="Select one of the listed responses before submitting the workbook.",
            )
            validation.add(
                f"{column_letter}{DATA_START_ROW}:{column_letter}{ws.max_row}"
            )
            ws.add_data_validation(validation)
        if not optional:
            ws.conditional_formatting.add(
                f"{column_letter}{DATA_START_ROW}:{column_letter}{ws.max_row}",
                CellIsRule(operator="equal", formula=['""'], fill=_MISSING_FILL),
            )

    if sheet_name == "Defined_Classes":
        verdict = get_column_letter(headers[visible_column_name(
            sheet_name,
            "Definition_Verdict (Correct/Partly correct/Incorrect/Unsure)",
        )])
        issue = get_column_letter(headers[visible_column_name(
            sheet_name,
            "Issue_Reason (select for Partly/Incorrect)",
        )])
        ws.conditional_formatting.add(
            f"{issue}{DATA_START_ROW}:{issue}{ws.max_row}",
            FormulaRule(
                formula=[
                    f'AND(OR(${verdict}{DATA_START_ROW}="Partly correct",'
                    f'${verdict}{DATA_START_ROW}="Incorrect"),'
                    f'${issue}{DATA_START_ROW}="")'
                ],
                fill=_MISSING_FILL,
            ),
        )
    if sheet_name == "Meaning_Preservation":
        meaning = get_column_letter(headers[visible_column_name(
            sheet_name,
            "Meaning_Preserved (Fully/Mostly/No/Unsure)",
        )])
        outcome = get_column_letter(headers[visible_column_name(
            sheet_name,
            "Preferred_Outcome (for Mostly/No)",
        )])
        ws.conditional_formatting.add(
            f"{outcome}{DATA_START_ROW}:{outcome}{ws.max_row}",
            FormulaRule(
                formula=[
                    f'AND(OR(${meaning}{DATA_START_ROW}="Mostly",'
                    f'${meaning}{DATA_START_ROW}="No"),'
                    f'${outcome}{DATA_START_ROW}="")'
                ],
                fill=_MISSING_FILL,
            ),
        )


def _write_workbook(
    path: str,
    instructions: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> None:
    sheet_instructions = get_study_config().sheet_instructions
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="Instructions", index=False)
        for sheet_name, frame in frames.items():
            visible_frame = frame.rename(columns=COLUMN_DISPLAY_NAMES.get(sheet_name, {}))
            visible_frame.to_excel(
                writer,
                sheet_name=visible_sheet_name(sheet_name),
                index=False,
                startrow=DATA_HEADER_ROW - 1,
            )
        workbook = writer.book
        _format_instructions(workbook["Instructions"])
        for sheet_name in frames:
            _format_data_sheet(
                workbook[visible_sheet_name(sheet_name)],
                _INPUT_VALIDATIONS.get(sheet_name, {}),
                sheet_name,
                sheet_instructions[sheet_name],
            )


def _write_optional_model_changes_workbook(
    path: str,
    instructions: pd.DataFrame,
    frame: pd.DataFrame,
) -> None:
    optional_instructions = pd.concat(
        [
            pd.DataFrame([
                {
                    "Section": "OPTIONAL SPECIALIST REVIEW",
                    "Details": (
                        "This separate file is for a geologist or ontologist who is "
                        "comfortable reviewing how terms were removed or rewritten. "
                        "It is not part of the main three-expert study."
                    ),
                }
            ]),
            instructions,
        ],
        ignore_index=True,
    )
    _write_workbook(
        path,
        optional_instructions,
        {"Meaning_Preservation": frame},
    )


def generate_modular_evaluation(
    n_terms: int = DEFAULT_TERM_SAMPLE,
    n_category_terms: int = DEFAULT_CATEGORY_TERM_SAMPLE,
    seed: int = DEFAULT_SEED,
    output_dir: str | None = None,
    n_experts: int = DEFAULT_EXPERTS,
    ontology_dir: str | None = None,
    terms_path: str | None = None,
    reference_definitions_path: str | None = None,
    require_reviewed_definitions: bool = True,
) -> tuple[list[str], str]:
    """Generate modular workbooks with identical sampled items and one key."""
    if n_experts < 1:
        raise ValueError("n_experts must be positive")
    ablation_dir = output_dir or os.environ.get("ABLATION_OUTPUT_DIR", str(ABLATION_OUTPUT))
    workbook_dir = os.path.join(ablation_dir, "expert_workbooks")
    private_dir = os.path.join(ablation_dir, "private")
    ontology_dir = ontology_dir or os.environ.get("EXPERT_ONTOLOGY_DIR", str(APPROVED_ONTOLOGY_DIR))
    terms_path = terms_path or os.environ.get("EXPERT_TERMS_PATH", str(FILTERED_TERMS))
    expected_terms = int(os.environ.get("ABLATION_EXPECTED_TERM_COUNT", 407))
    strict_populations = os.environ.get("EXPERT_STRICT_APPROVED_POPULATIONS", "true").lower() == "true"
    os.makedirs(ablation_dir, exist_ok=True)
    os.makedirs(workbook_dir, exist_ok=True)
    os.makedirs(private_dir, exist_ok=True)

    inputs = load_study_inputs(
        ablation_dir=ablation_dir,
        ontology_dir=ontology_dir,
        terms_path=terms_path,
        expected_term_count=expected_terms,
    )
    final_fates = build_final_fates(inputs)
    term_sample = select_representation_terms(inputs.terms, inputs.categories["A"], n_terms, seed)
    category_term_sample = select_category_terms(
        inputs.terms,
        inputs.categories,
        n_terms=n_category_terms,
        seed=seed,
    )
    sampled_fates = {
        final_fates[_normalise(term)]
        for term in term_sample["Readable_Term"]
    }
    expected_fates = {
        "FINAL_CLASS",
        "FINAL_INDIVIDUAL",
        "DEMOTED",
        "CRITIC_EXCLUDED",
        "CQ_EXCLUDED",
    }
    if sampled_fates != expected_fates:
        raise ValueError(
            f"Fixed representation sample does not cover all final fates: "
            f"missing={sorted(expected_fates - sampled_fates)}"
        )
    representation_items = build_representation_items(term_sample, inputs, final_fates)
    final_samples = select_final_ontology_items(
        inputs,
        seed=seed,
        strict_approved_populations=strict_populations,
    )
    reference_definitions_path = reference_definitions_path or os.environ.get(
        "EXPERT_REFERENCE_DEFINITIONS"
    )
    reference_definitions: dict[str, str] = {}
    required_reference_terms = set(term_sample["Readable_Term"].astype(str)) | set(
        category_term_sample["Readable_Term"].astype(str)
    ) | set(
        final_samples["Meaning_Preservation"]["term"].astype(str)
    )
    if reference_definitions_path:
        reference_definitions = load_reviewed_reference_definitions(
            reference_definitions_path,
            required_reference_terms,
        )
    elif require_reviewed_definitions:
        raise RuntimeError(
            "Real expert workbooks require EXPERT_REFERENCE_DEFINITIONS pointing "
            "to an approved CSV or XLSX covering all sampled terms."
        )
    else:
        nld_lookup = _lookup_by_term(inputs.nld["A"], "NLD", "Condition A NLD")
        reference_definitions = {
            _normalise(term): _first_sentence(nld_lookup[_normalise(term)])
            for term in required_reference_terms
        }
    category_items = build_category_items(
        category_term_sample,
        inputs.categories,
        final_fates,
        reference_definitions=reference_definitions,
    )
    final_samples["Meaning_Preservation"]["Reference_Definition"] = (
        final_samples["Meaning_Preservation"]["term"].map(
            lambda value: reference_definitions[_normalise(value)]
        )
    )

    instructions = pd.DataFrame(
        get_study_config().instruction_rows,
        columns=["Section", "Details"],
    )
    category_guide = build_category_guide()
    workbook_paths = []
    key_parts = []
    optional_review_frame: pd.DataFrame | None = None
    for expert_number in range(1, n_experts + 1):
        expert_id = f"expert_{expert_number}"
        expert_seed = seed + expert_number * 10_000
        representation, representation_key = _representation_for_expert(
            representation_items,
            expert_id,
            expert_seed,
        )
        category, category_key = _category_for_expert(
            category_items,
            expert_id,
            expert_seed + 100,
        )
        final_frames, final_keys = _final_frames_for_expert(
            final_samples,
            expert_id,
            expert_seed + 200,
        )
        if optional_review_frame is None:
            optional_review_frame = final_frames["Meaning_Preservation"].copy()
        frames = {
            "Category_Guide": category_guide,
            "Representation": representation,
            "Category_Correct": category,
            **{
                name: frame
                for name, frame in final_frames.items()
                if name != "Meaning_Preservation"
            },
        }
        workbook_path = os.path.join(
            workbook_dir,
            f"expert_evaluation_{expert_number}.xlsx",
        )
        _write_workbook(workbook_path, instructions, frames)
        workbook_paths.append(workbook_path)
        key_parts.extend([representation_key, category_key, *final_keys])

    if optional_review_frame is None:
        raise RuntimeError("Optional model-changes review was not generated")
    optional_review_path = os.path.join(
        ablation_dir,
        "model_changes_review.xlsx",
    )
    _write_optional_model_changes_workbook(
        optional_review_path,
        instructions,
        optional_review_frame,
    )

    key_path = os.path.join(private_dir, f"blinding_key_{seed}.csv")
    key = pd.concat(key_parts, ignore_index=True, sort=False)
    write_csv(key, key_path)

    source_paths = {
        "terms": terms_path,
        **{
            f"nld_{condition}": os.path.join(ablation_dir, f"nld_{condition}.csv")
            for condition in CONDITIONS
        },
        **{
            f"cat_{condition}": os.path.join(ablation_dir, f"cat_{condition}.csv")
            for condition in CONDITIONS
        },
        "cq_filtered_categories": os.path.join(ontology_dir, "classify_categories.csv"),
        "taxonomy": os.path.join(ontology_dir, "validate_taxonomy.csv"),
        "defined_classes": os.path.join(ontology_dir, "validate_defined_classes.csv"),
        "relations": os.path.join(ontology_dir, "validate_relations.csv"),
        "individuals": os.path.join(ontology_dir, "validate_instances.csv"),
        "demotions": os.path.join(ontology_dir, "validate_demotions.csv"),
        "class_fates": os.path.join(ontology_dir, "validate_class_fates.csv"),
        "ontology_config": str(get_config()._source_path),
        "study_config": os.environ.get("STUDY_CONFIG_PATH", str(STUDY_CONFIG)),
        "display_text_config": os.environ.get(
            "DISPLAY_TEXT_CONFIG_PATH",
            str(DISPLAY_TEXT_CONFIG),
        ),
    }
    if reference_definitions_path:
        source_paths["reviewed_reference_definitions"] = reference_definitions_path
    manifest = {
        "study": "PreSaltOntoLearn modular expert evaluation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_experts": n_experts,
        "representation_sample": len(representation_items),
        "category_term_sample": len(category_term_sample),
        "category_rows": len(category_items),
        "category_disagreement_coverage": {
            comparator: int(category_term_sample[f"A_vs_{comparator}"].sum())
            for comparator in ("B", "C", "D")
        },
        "final_sample_sizes": {name: len(frame) for name, frame in final_samples.items()},
        "final_fate_counts": pd.Series(final_fates).value_counts().sort_index().to_dict(),
        "reviewed_reference_definitions": bool(reference_definitions_path),
        "source_paths": {
            name: str(Path(path).resolve())
            for name, path in source_paths.items()
        },
        "source_sha256": {name: _sha256_file(path) for name, path in source_paths.items()},
        "workbooks": workbook_paths,
        "optional_model_changes_review": optional_review_path,
        "blinding_key": key_path,
    }
    manifest_path = os.path.join(ablation_dir, "expert_evaluation_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"\nExpert evaluation: {len(representation_items)} representation terms")
    print(f"  Category rows: {len(category_items)} unique term/category assignments")
    for sheet_name, frame in final_samples.items():
        print(f"  {sheet_name}: {len(frame)} sampled rows")
    for path in workbook_paths:
        print(f"  Workbook: {path}")
    print(f"  Optional model-changes review: {optional_review_path}")
    print(f"  Blinding key: {key_path} (do not share with experts)")
    print(f"  Manifest: {manifest_path}")
    return workbook_paths, key_path