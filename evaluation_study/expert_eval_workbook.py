"""Modular, blinded expert-evaluation workbooks for PreSaltOntoLearn.

The study separates two estimands:

* Representation evaluation: relevance, A/B NLD quality, and A/B/C/D category
  correctness for a seeded 100-term sample.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.formatting.rule import CellIsRule
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
DEFAULT_EXPERTS = 3
DEFAULT_SEED = 42
FINAL_SAMPLE_SIZES = {
    "Taxonomy": 40,
    "Defined_Classes": 13,
    "Relations": 25,
    "Individuals": 15,
    "Critic_Decisions": 40,
}
APPROVED_POPULATIONS = {
    "Taxonomy": 185,
    "Defined_Classes": 13,
    "Relations": 280,
    "Individuals": 58,
    "Critic_Decisions": 116,
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


def _normalise(value: object) -> str:
    return str(value).strip().casefold()


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
        sentence_end = next(
            (index + 1 for index, character in enumerate(text) if character in ".!?"),
            len(text),
        )
        glosses[key] = text[:sentence_end].strip()
    return glosses


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


def build_timing_sheet() -> pd.DataFrame:
    """Collect actual completion time by module during the human pilot."""
    modules = [
        "Representation",
        "Category Correct and Taxonomy",
        "Defined Classes, Relations, Individuals, and Critic Decisions",
    ]
    return pd.DataFrame({
        "Row_ID": [f"TIME-{index:02d}" for index in range(1, len(modules) + 1)],
        "Session": [1, 2, 3],
        "Module": modules,
        "Minutes": ["", "", ""],
        "Comments": ["", "", ""],
    })


def build_category_items(
    sample: pd.DataFrame,
    categories: dict[str, pd.DataFrame],
    final_fates: dict[str, str],
    term_glosses: dict[str, str] | None = None,
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
    rows = []
    ordered = sorted(assignments.items(), key=lambda item: (sampled_terms[item[0][0]].casefold(), item[0][1]))
    for index, ((term_key, category), conditions) in enumerate(ordered, 1):
        rows.append({
            "Row_ID": f"CAT-{index:04d}",
            "Term": sampled_terms[term_key],
            "Term_Gloss": term_glosses.get(term_key, ""),
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
    decisions = inputs.class_fates[
        inputs.class_fates["action"].astype(str).str.startswith(("DROP", "DEMOTE"))
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
    decisions["Resulting_Treatment"] = [
        (
            f"No longer a separate class; represented as {base_class} "
            f"with {display_label(property_name, 'property')} "
            f"{display_label(filler, 'category').lower()}."
            if decision_type == "DEMOTE"
            else "Not included as a separate concept in the final ontology."
        )
        for decision_type, base_class, property_name, filler in zip(
            decisions["Decision_Type"],
            decisions["Base_Class"],
            decisions["Property"],
            decisions["Filler"],
        )
    ]
    populations = {
        "Taxonomy": taxonomy,
        "Defined_Classes": defined,
        "Relations": relations,
        "Individuals": individuals,
        "Critic_Decisions": decisions,
    }
    required_columns = {
        "Taxonomy": {"Term", "Parent_Term", "Category", "Is_Intermediate"},
        "Defined_Classes": {"Bearer", "Genus", "Property", "Filler"},
        "Relations": {"Term", "Property", "Filler", "Evidence", "Relation_Scope"},
        "Individuals": {"Term", "Target_Class", "Reason"},
        "Critic_Decisions": {
            "term", "category", "action", "reason", "Decision_Type",
            "Resulting_Treatment",
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
        "Critic_Decisions": _proportional_sample(
            decisions,
            FINAL_SAMPLE_SIZES["Critic_Decisions"],
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
    samples["Critic_Decisions"] = _assign_stable_ids(
        samples["Critic_Decisions"], "DEC", ["term", "action"]
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
    subject = _lower_initial(display_label(bearer, "category"))
    base_kind = display_label(genus, "category").lower()
    feature = display_label(bearer, "defined_class")
    if str(bearer).casefold() not in get_display_registry().defined_class_features:
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
        ["Row_ID", "Term", "Assigned_Category", "Category_Description"]
    ].copy()
    visible.insert(2, "Term_Gloss", items.get("Term_Gloss", ""))
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
    term_glosses: dict[str, str] | None = None,
) -> tuple[dict[str, pd.DataFrame], list[pd.DataFrame]]:
    term_glosses = term_glosses or {}
    taxonomy = samples["Taxonomy"]
    taxonomy_visible = pd.DataFrame({
        "Row_ID": taxonomy["Row_ID"],
        "Child_Concept": taxonomy["Term"],
        "Term_Gloss": taxonomy["Term"].map(lambda value: term_glosses.get(_normalise(value), "")),
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
        "Term_Gloss": defined["Bearer"].map(
            lambda value: term_glosses.get(_normalise(value), "")
        ),
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
        "Term_Gloss": relations["Term"].map(
            lambda value: term_glosses.get(_normalise(value), "")
        ),
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
        "Corpus_Excerpt (context only)": relations["Evidence"],
        "Notes": "",
    })

    individuals = samples["Individuals"]
    individual_visible = pd.DataFrame({
        "Row_ID": individuals["Row_ID"],
        "Named_Entity": individuals["Term"],
        "Term_Gloss": individuals["Term"].map(
            lambda value: term_glosses.get(_normalise(value), "")
        ),
        "Proposed_Type": individuals["Target_Class"].map(
            lambda value: display_label(value, "category")
        ),
        "Specific_Named_Entity (Yes/No/Unsure)": "",
        "Type_Correct (Yes/Partial/No/Unsure)": "",
        "Notes": "",
    })

    decisions = samples["Critic_Decisions"]
    decision_text = decisions["action"].map({
        "DEMOTE_TO_PROPERTY": "Represent as a characteristic or relation",
        "DROP_AS_OVER_SPECIFIC": "Exclude as a separate concept",
        "DROP_AS_REDUNDANT": "Exclude as a separate concept",
        "DROP_AS_MIXIN": "Exclude as a separate concept",
    }).fillna(decisions["action"].astype(str))
    decision_visible = pd.DataFrame({
        "Row_ID": decisions["Row_ID"],
        "Concept": decisions["term"],
        "Term_Gloss": decisions["term"].map(
            lambda value: term_glosses.get(_normalise(value), "")
        ),
        "Current_Category": decisions["category"].map(
            lambda value: display_label(value, "category")
        ),
        "Critic_Decision": decision_text,
        "Resulting_Treatment": decisions["Resulting_Treatment"],
        "Decision_Acceptability (Accept/Accept with concern/Reject/Unsure)": "",
        "Preferred_Treatment (for Concern/Reject)": "",
        "Notes": "",
    })

    frames = {
        "Taxonomy": _shuffle(taxonomy_visible, seed + 10),
        "Defined_Classes": _shuffle(defined_visible, seed + 20),
        "Relations": _shuffle(relation_visible, seed + 30),
        "Individuals": _shuffle(individual_visible, seed + 40),
        "Critic_Decisions": _shuffle(decision_visible, seed + 50),
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
        "Issue_Reason (select for Partly/Incorrect)": "Base kind is wrong,Feature is not defining,Too broad,Too narrow,Wording unclear,Other,Unsure",
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
    "Critic_Decisions": {
        "Decision_Acceptability (Accept/Accept with concern/Reject/Unsure)": "Accept,Accept with concern,Reject,Unsure",
        "Preferred_Treatment (for Concern/Reject)": "Keep as separate concept,Keep information but not as separate concept,Leave out,Unsure",
        "Notes": None,
    },
    "Timing": {
        "Minutes": None,
        "Comments": None,
    },
}

_OPTIONAL_INPUTS = {
    ("Defined_Classes", "Issue_Reason (select for Partly/Incorrect)"),
    ("Critic_Decisions", "Preferred_Treatment (for Concern/Reject)"),
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
) -> None:
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.sheet_view.showGridLines = False
    headers = {str(cell.value): cell.column for cell in ws[1]}
    if "Row_ID" in headers:
        ws.column_dimensions[get_column_letter(headers["Row_ID"])].hidden = True
    for cell in ws[1]:
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = _CENTER
        cell.border = _BORDER
    for column_index in range(1, ws.max_column + 1):
        header = str(ws.cell(1, column_index).value or "")
        width = 15
        if any(token in header for token in ("Definition", "Description", "Rationale", "Evidence", "Excerpt", "Question", "Decision", "Statement")):
            width = 70
        elif any(token in header for token in ("Term", "Concept", "Category", "Parent", "Subject", "Entity", "Type")):
            width = 28
        elif header == "Notes" or header.startswith("Suggested"):
            width = 35
        ws.column_dimensions[get_column_letter(column_index)].width = width
        for row_index in range(2, ws.max_row + 1):
            cell = ws.cell(row_index, column_index)
            cell.alignment = _WRAP
            cell.border = _BORDER

    for header, options in input_validations.items():
        column_index = headers[header]
        column_letter = get_column_letter(column_index)
        for row_index in range(2, ws.max_row + 1):
            ws.cell(row_index, column_index).fill = _INPUT_FILL
        if options:
            optional = (sheet_name, header) in _OPTIONAL_INPUTS
            validation = DataValidation(
                type="list",
                formula1=f'"{options}"',
                allow_blank=optional,
                showErrorMessage=True,
                errorTitle="Response required",
                error="Select one of the listed responses before submitting the workbook.",
            )
            validation.add(f"{column_letter}2:{column_letter}{ws.max_row}")
            ws.add_data_validation(validation)
            if not optional:
                ws.conditional_formatting.add(
                    f"{column_letter}2:{column_letter}{ws.max_row}",
                    CellIsRule(operator="equal", formula=['""'], fill=_MISSING_FILL),
                )


def _write_workbook(
    path: str,
    instructions: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="Instructions", index=False)
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
        workbook = writer.book
        _format_instructions(workbook["Instructions"])
        for sheet_name in frames:
            _format_data_sheet(
                workbook[sheet_name],
                _INPUT_VALIDATIONS.get(sheet_name, {}),
                sheet_name,
            )


def generate_modular_evaluation(
    n_terms: int = DEFAULT_TERM_SAMPLE,
    seed: int = DEFAULT_SEED,
    output_dir: str | None = None,
    n_experts: int = DEFAULT_EXPERTS,
    ontology_dir: str | None = None,
    terms_path: str | None = None,
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
    term_glosses = build_term_glosses(inputs.nld["A"])
    category_items = build_category_items(
        term_sample,
        inputs.categories,
        final_fates,
        term_glosses=term_glosses,
    )
    final_samples = select_final_ontology_items(
        inputs,
        seed=seed,
        strict_approved_populations=strict_populations,
    )

    instructions = pd.DataFrame(
        get_study_config().instruction_rows,
        columns=["Section", "Details"],
    )
    category_guide = build_category_guide()
    timing = build_timing_sheet()
    workbook_paths = []
    key_parts = []
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
            term_glosses=term_glosses,
        )
        frames = {
            "Category_Guide": category_guide,
            "Representation": representation,
            "Category_Correct": category,
            **final_frames,
            "Timing": timing,
        }
        workbook_path = os.path.join(
            workbook_dir,
            f"expert_evaluation_{expert_number}.xlsx",
        )
        _write_workbook(workbook_path, instructions, frames)
        workbook_paths.append(workbook_path)
        key_parts.extend([representation_key, category_key, *final_keys])

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
    manifest = {
        "study": "PreSaltOntoLearn modular expert evaluation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_experts": n_experts,
        "representation_sample": len(representation_items),
        "category_rows": len(category_items),
        "final_sample_sizes": {name: len(frame) for name, frame in final_samples.items()},
        "final_fate_counts": pd.Series(final_fates).value_counts().sort_index().to_dict(),
        "source_paths": {
            name: str(Path(path).resolve())
            for name, path in source_paths.items()
        },
        "source_sha256": {name: _sha256_file(path) for name, path in source_paths.items()},
        "workbooks": workbook_paths,
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
    print(f"  Blinding key: {key_path} (do not share with experts)")
    print(f"  Manifest: {manifest_path}")
    return workbook_paths, key_path