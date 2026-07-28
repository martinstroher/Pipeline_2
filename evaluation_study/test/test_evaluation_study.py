"""Focused regression checks for the thesis evaluation study.

Run:
    python evaluation_study/test/test_evaluation_study.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evaluation_study.ablation_study as ablation
from evaluation_study.expert_eval_analysis import (
    _numeric_rating,
    _summarize_judgments,
    analyze_categories,
    analyze_final_ontology,
    analyze_representation,
)
from evaluation_study.expert_eval_workbook import (
    StudyInputs,
    _category_for_expert,
    _collapse_accent_variants,
    _representation_for_expert,
    build_final_fates,
    load_reviewed_reference_definitions,
    select_category_terms,
    select_representation_terms,
)
from evaluation_study.layer1_analysis import (
    agreement_matrix,
    cochrans_q_test,
    mcnemar_posthoc,
    sensitivity_flags,
    stuart_maxwell_tests,
    validate_paired_results,
)
from evaluation_study.relation_analysis import generate_precision_sample
from src.utils.csv_io import read_csv, write_csv
from src.utils.ontology_config import get_config
from src.utils.prompt_loader import load_prompt


_FAILURES: list[str] = []


def _expect(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"[OK]   {label}")
        return
    message = f"[FAIL] {label}"
    if detail:
        message += f" :: {detail}"
    _FAILURES.append(message)
    print(message)


def _expect_raises(label: str, exception_type: type[Exception], function) -> None:
    try:
        function()
    except exception_type:
        _expect(label, True)
    except Exception as exc:
        _expect(label, False, f"raised {type(exc).__name__}: {exc}")
    else:
        _expect(label, False, f"expected {exception_type.__name__}")


def test_ablation_artifacts_and_prompts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "production_a.csv"
        write_csv(pd.DataFrame([
            {"Term": "alpha", "NLD": "Definition A", "Context_Used": True, "Context": "chunk one"},
            {"Term": "beta", "NLD": "Definition B", "Context_Used": False, "Context": "chunk two"},
        ]), source)
        old_output = ablation.OUTPUT_DIR
        old_source = os.environ.get("ABLATION_FROZEN_A_NLD")
        try:
            ablation.OUTPUT_DIR = str(root / "ablation")
            os.environ["ABLATION_FROZEN_A_NLD"] = str(source)
            frozen = ablation.run_condition_a(["alpha", "beta"])
            copied = root / "ablation" / "nld_A.csv"
            _expect(
                "Condition A is copied byte-for-byte",
                ablation._sha256_file(source) == ablation._sha256_file(copied),
            )
            raw = ablation.run_condition_d(["alpha", "beta"], frozen)
            _expect(
                "Condition D reuses Condition A contexts",
                raw["NLD"].fillna("").tolist() == frozen["Context"].fillna("").tolist(),
            )
        finally:
            ablation.OUTPUT_DIR = old_output
            if old_source is None:
                os.environ.pop("ABLATION_FROZEN_A_NLD", None)
            else:
                os.environ["ABLATION_FROZEN_A_NLD"] = old_source

    production_system, production_template = load_prompt("term_categorization.txt")
    expected_template = production_template.format(
        categories_block=get_config().categorization_block(),
        json_batch="{json_batch}",
    )
    actual_system, actual_template = ablation._build_categorizer_prompt(False)
    _expect(
        "A/B/C categorization uses the production prompt",
        actual_system == production_system and actual_template == expected_template,
    )
    _, raw_template = ablation._build_categorizer_prompt(True)
    _expect(
        "Condition D worked examples use raw context fields",
        '"context":' in raw_template and '"nld":' not in raw_template,
    )
    rendered = ablation._render_categorizer_prompt(
        actual_template,
        [{"term": "sample", "nld": "sample definition"}],
    )
    _expect(
        "Categorizer batch insertion preserves embedded JSON examples",
        '"term": "sample"' in rendered
        and '"term": "Grainstone"' in rendered
        and "{json_batch}" not in rendered,
    )


def _layer1_fixture() -> pd.DataFrame:
    categories = {
        "A": ["Sedimentary Rock", "Sedimentary Rock", "Geological Process", "NOT_CLASSIFIED", "Earth Material", "Earth Material"],
        "B": ["Sedimentary Rock", "Geological Process", "Geological Process", "NOT_CLASSIFIED", "Earth Material", "Sedimentary Rock"],
        "C": ["Geological Process", "Geological Process", "Geological Process", "NOT_CLASSIFIED", "Earth Material", "Earth Material"],
        "D": ["Sedimentary Rock", "Sedimentary Rock", "Geological Process", "Sedimentary Rock", "Earth Material", "Geological Process"],
    }
    return pd.DataFrame([
        {"Term": f"term-{index}", "Condition": condition, "Category": category}
        for condition, values in categories.items()
        for index, category in enumerate(values)
    ])


def test_layer1_contracts() -> None:
    frame = _layer1_fixture()
    validate_paired_results(frame, expected_term_count=6)
    _expect("Complete A/B/C/D matrix is accepted", True)
    _expect_raises(
        "Incomplete paired matrix is rejected",
        ValueError,
        lambda: validate_paired_results(frame.iloc[:-1]),
    )
    flags = sensitivity_flags(frame)
    _expect(
        "Sensitivity axes are independent",
        {"RAG_Sensitive", "NLD_Sensitive", "Structuring_Sensitive"}.issubset(flags.columns),
    )
    _expect("Exact agreement includes Cohen kappa", "Cohens_Kappa" in agreement_matrix(frame).columns)
    frozen_tiers = frame.copy()
    frozen_tiers["Tier"] = "FROZEN_TIER"
    _expect(
        "Layer 1 prefers tiers frozen by the ablation run",
        agreement_matrix(frozen_tiers, level="tier")["Agreement_Rate"].eq(1.0).all(),
    )
    omnibus = cochrans_q_test(frame)
    _expect("Global Cochran Q uses three A-anchored indicators", omnibus["df"] == 2)
    _expect("McNemar post-hoc family has three comparisons", len(mcnemar_posthoc(frame, omnibus["p_value"])) == 3)
    _expect("Stuart-Maxwell tier family has three comparisons", len(stuart_maxwell_tests(frame)) == 3)


def _fate_fixture() -> StudyInputs:
    terms = pd.DataFrame({
        "Readable_Term": ["final class", "named place", "quality term", "critic drop", "cq drop"],
        "Frequency": [10, 9, 8, 7, 6],
    })
    return StudyInputs(
        terms=terms,
        nld={},
        categories={},
        refined_categories=pd.DataFrame({"Term": ["final class", "named place", "quality term", "critic drop"]}),
        taxonomy=pd.DataFrame({"Term": ["final class"]}),
        defined_classes=pd.DataFrame(),
        relations=pd.DataFrame(),
        individuals=pd.DataFrame({"Term": ["named place"]}),
        demotions=pd.DataFrame({"Term": ["quality term"]}),
        class_fates=pd.DataFrame({"term": ["critic drop"], "action": ["DROP_AS_OVER_SPECIFIC"]}),
    )


def test_sampling_and_fates() -> None:
    fates = build_final_fates(_fate_fixture())
    _expect(
        "Final fates form the five-way partition",
        Counter(fates.values()) == {
            "FINAL_CLASS": 1,
            "FINAL_INDIVIDUAL": 1,
            "DEMOTED": 1,
            "CRITIC_EXCLUDED": 1,
            "CQ_EXCLUDED": 1,
        },
    )

    terms = pd.DataFrame({
        "Readable_Term": [f"term-{index:02d}" for index in range(30)],
        "Frequency": list(range(1, 31)),
    })
    categories = ["Sedimentary Rock", "Geological Process", "Earth Material"]
    condition_a = pd.DataFrame({
        "Term": terms["Readable_Term"],
        "Category": [categories[index % len(categories)] for index in range(len(terms))],
    })
    first = select_representation_terms(terms, condition_a, n_terms=12, seed=42)
    second = select_representation_terms(terms, condition_a, n_terms=12, seed=42)
    _expect("Representation sample has exact requested size", len(first) == 12)
    _expect("Representation sample is seed-reproducible", first.equals(second))
    _expect(
        "Representation sample carries both strata",
        {"Tier_A", "Frequency_Band"}.issubset(first.columns),
    )

    category_frames = {
        condition: pd.DataFrame({
            "Term": terms["Readable_Term"],
            "Category": ["Sedimentary Rock"] * len(terms),
        })
        for condition in ("A", "B", "C", "D")
    }
    for index in range(len(terms)):
        unchanged = {"A", ("D", "C", "B")[index % 3]}
        for condition in set(category_frames) - unchanged:
            category_frames[condition].loc[index, "Category"] = "Geological Process"
    category_terms = pd.concat([terms] * 3, ignore_index=True)
    category_terms["Readable_Term"] = [
        f"category-term-{index:03d}" for index in range(len(category_terms))
    ]
    for condition, frame in category_frames.items():
        category_frames[condition] = pd.concat([frame] * 3, ignore_index=True)
        category_frames[condition]["Term"] = category_terms["Readable_Term"]
    category_first = select_category_terms(
        category_terms,
        category_frames,
        n_terms=60,
        seed=42,
    )
    category_second = select_category_terms(
        category_terms,
        category_frames,
        n_terms=60,
        seed=42,
    )
    _expect("Category sample has exact requested size", len(category_first) == 60)
    _expect("Category sample is seed-reproducible", category_first.equals(category_second))
    _expect(
        "Category sample contains only A disagreements",
        category_first[["A_vs_B", "A_vs_C", "A_vs_D"]].any(axis=1).all(),
    )
    _expect(
        "Category sample covers every prespecified contrast",
        all(int(category_first[f"A_vs_{condition}"].sum()) >= 25 for condition in ("B", "C", "D")),
    )
    accent_variants = _collapse_accent_variants(pd.DataFrame([
        {"Readable_Term": "florianopolis high", "Frequency": 8},
        {"Readable_Term": "florianópolis high", "Frequency": 6},
        {"Readable_Term": "basement high", "Frequency": 5},
    ]))
    _expect(
        "Category sampling collapses accent-only duplicate concepts",
        set(accent_variants["Readable_Term"])
        == {"florianópolis high", "basement high"},
    )


def test_item_level_nld_inference() -> None:
    _expect(
        "Representation Unsure is explicit missingness",
        pd.isna(_numeric_rating("Unsure", "test rating")),
    )
    frame = pd.DataFrame([
        {
            "Term": term,
            "Expert": f"expert_{expert}",
            "Relevance": 4,
            "Quality_A": quality_a,
            "Quality_B": quality_b,
            "Preference_A": 1 if quality_a > quality_b else 0,
            "Final_Fate": "FINAL_CLASS",
        }
        for term, quality_a, quality_b in (("alpha", 5, 3), ("beta", 4, 4))
        for expert in (1, 2, 3)
    ])
    results = analyze_representation(frame, bootstrap_iterations=50, seed=42)
    nld = results["nld_quality"]
    _expect("NLD inference unit is two terms, not six ratings", nld["n_terms"] == 2)
    _expect("Term-level A-B difference is preserved", nld["mean_difference_A_minus_B"] == 1.0)
    _expect("Rank-biserial effect is reported", "rank_biserial" in nld["wilcoxon"])
    _expect_raises(
        "Zero bootstrap iterations are rejected",
        ValueError,
        lambda: analyze_representation(frame, bootstrap_iterations=0, seed=42),
    )


def test_modular_blinding_and_analysis() -> None:
    with tempfile.TemporaryDirectory() as directory:
        reference_path = Path(directory) / "references.csv"
        write_csv(pd.DataFrame([
            {"Term": "grainstone", "Reference_Definition": "A grainstone is a grain-supported carbonate sedimentary rock without a mud matrix.", "Review_Status": "APPROVED"},
            {"Term": "rift", "Reference_Definition": "A rift is an extensional tectonic setting formed by faulting and subsidence.", "Review_Status": "PENDING"},
        ]), reference_path)
        references = load_reviewed_reference_definitions(
            reference_path,
            {"grainstone"},
        )
        _expect(
            "Approved reference definitions load by normalized term",
            references["grainstone"].startswith("A grainstone"),
        )
        _expect_raises(
            "Pending reference definitions are rejected",
            ValueError,
            lambda: load_reviewed_reference_definitions(reference_path, {"rift"}),
        )
        _expect_raises(
            "Missing reference definitions are rejected",
            ValueError,
            lambda: load_reviewed_reference_definitions(reference_path, {"dolomite"}),
        )

    representation_items = pd.DataFrame([
        {
            "Row_ID": f"REP-{index:03d}",
            "Term": f"term-{index}",
            "Frequency": index,
            "Frequency_Band": "Middle",
            "Category_A": "Sedimentary Rock",
            "Tier_A": "GEORESERVOIR",
            "Final_Fate": "FINAL_CLASS",
            "Context_Used_A": True,
            "NLD_A": f"A definition {index}",
            "NLD_B": f"B definition {index}",
        }
        for index in range(1, 21)
    ])
    first, first_key = _representation_for_expert(representation_items, "expert_1", 100)
    second, second_key = _representation_for_expert(representation_items, "expert_2", 200)
    _expect(
        "Experts receive identical stable representation IDs",
        set(first["Row_ID"]) == set(second["Row_ID"]),
    )
    _expect(
        "Experts receive independently shuffled representation rows",
        first["Row_ID"].tolist() != second["Row_ID"].tolist(),
    )
    order_changed = (
        first_key.set_index("Row_ID")["Definition_1_Condition"]
        != second_key.set_index("Row_ID")["Definition_1_Condition"]
    ).any()
    _expect("A/B display order is independently randomized", bool(order_changed))

    category_items = pd.DataFrame([
        {
            "Row_ID": "CAT-0001",
            "Term": "grainstone",
            "Reference_Definition": "A grainstone is a grain-supported carbonate sedimentary rock without a mud matrix.",
            "Assigned_Category": "Sedimentary Rock",
            "Category_Description": "A sedimentary rock.",
            "Conditions": "A,B",
            "Tier": "GEORESERVOIR",
            "Final_Fate": "FINAL_CLASS",
        }
    ])
    category_visible, _ = _category_for_expert(category_items, "expert_1", 42)
    _expect(
        "Visible category sheet hides condition, tier, and final fate",
        not {"Conditions", "Tier", "Final_Fate"} & set(category_visible.columns),
    )
    _expect(
        "Visible category sheet does not request a hidden-list replacement",
        "Suggested_Category" not in category_visible.columns
        and "Proposed_Category" in category_visible.columns,
    )
    not_classified_items = category_items.copy()
    not_classified_items.loc[0, "Assigned_Category"] = "NOT_CLASSIFIED"
    not_classified_visible, _ = _category_for_expert(
        not_classified_items,
        "expert_1",
        42,
    )
    _expect(
        "NOT_CLASSIFIED is presented as an evaluable leave-unclassified decision",
        "Leave unclassified" in set(not_classified_visible["Proposed_Category"]),
    )

    category_rows = []
    raw_score = {
        "A": ("yes", 1.0),
        "B": ("partial", 0.5),
        "C": ("no", 0.0),
        "D": ("no", 0.0),
    }
    proposed_category = {
        "A": "Sedimentary Rock",
        "B": "Rock",
        "C": "Earth Material",
        "D": "Sedimentary Rock",
    }
    for term_index in range(1, 7):
        for condition, (raw, score) in raw_score.items():
            for expert_index in range(1, 4):
                category_rows.append({
                    "Assignment_ID": f"CAT-{term_index}-{condition}",
                    "Term": f"term-{term_index}",
                    "Condition": condition,
                    "Assigned_Category": proposed_category[condition],
                    "Expert": f"expert_{expert_index}",
                    "Correct_Raw": raw,
                    "Correct_Score": score,
                })
    category_results = analyze_categories(
        pd.DataFrame(category_rows),
        bootstrap_iterations=50,
        seed=42,
    )
    _expect(
        "Category inference aggregates six terms before Friedman",
        category_results["friedman"]["n_complete_terms"] == 6,
    )
    _expect(
        "Significant Friedman gate runs three Holm comparisons",
        len(category_results["posthoc_wilcoxon_holm"]) == 3,
    )
    disagreement = {
        row["comparison"]: row
        for row in category_results["disagreement_contrasts"]
    }
    _expect(
        "Category contrasts include only differing proposals",
        disagreement["A vs B"]["n_disagreement_terms"] == 6
        and disagreement["A vs C"]["n_disagreement_terms"] == 6
        and disagreement["A vs D"]["n_disagreement_terms"] == 0,
    )
    _expect(
        "Category contrast family applies Holm correction",
        all("wilcoxon_p_value_holm" in row for row in disagreement.values()),
    )

    def repeated(sheet_rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame([
            {**row, "Expert": f"expert_{expert}"}
            for row in sheet_rows
            for expert in (1, 2, 3)
        ])

    final_frames = {
        "Taxonomy": repeated([
            {"Row_ID": "TAX-001", "Relationship_Correct (Yes/Partial/No/Unsure)": "Yes", "Useful_PreSalt_Distinction (Yes/No/Unsure)": "Yes"},
            {"Row_ID": "TAX-002", "Relationship_Correct (Yes/Partial/No/Unsure)": "Partial", "Useful_PreSalt_Distinction (Yes/No/Unsure)": "No"},
        ]),
        "Defined_Classes": repeated([
            {"Row_ID": "DEF-001", "Definition_Verdict (Correct/Partly correct/Incorrect/Unsure)": "Correct", "Issue_Reason (select for Partly/Incorrect)": ""},
        ]),
        "Relations": repeated([
            {"Row_ID": "REL-001", "Relation_Scope": "generic", "Relation_Verdict": "Generally true"},
            {"Row_ID": "REL-002", "Relation_Scope": "corpus_context", "Relation_Verdict": "Context-specific"},
        ]),
        "Individuals": repeated([
            {"Row_ID": "IND-001", "Specific_Named_Entity (Yes/No/Unsure)": "Yes", "Type_Correct (Yes/Partial/No/Unsure)": "Yes"},
        ]),
        "Meaning_Preservation": repeated([
            {"Row_ID": "DEC-001", "Decision_Type": "DEMOTE", "Meaning_Preserved (Fully/Mostly/No/Unsure)": "Fully", "Appropriate_for_Lean_Core (Yes/With concern/No/Unsure)": "Yes", "Preferred_Outcome (for Mostly/No)": ""},
            {"Row_ID": "DEC-002", "Decision_Type": "EXCLUDE", "Meaning_Preserved (Fully/Mostly/No/Unsure)": "Mostly", "Appropriate_for_Lean_Core (Yes/With concern/No/Unsure)": "With concern", "Preferred_Outcome (for Mostly/No)": "Leave out"},
        ]),
    }
    final_results = analyze_final_ontology(final_frames, bootstrap_iterations=50, seed=42)
    _expect(
        "Final ontology task families remain separate",
        set(final_results) == {"taxonomy", "defined_classes", "relations", "individuals", "meaning_preservation", "core_appropriateness"},
    )
    _expect(
        "Meaning preservation is separated by treatment type",
        set(final_results["meaning_preservation"]) == {"DEMOTE", "EXCLUDE"},
    )
    _expect(
        "Core appropriateness is analyzed separately by treatment type",
        set(final_results["core_appropriateness"]) == {"DEMOTE", "EXCLUDE"},
    )
    invalid_preferred_outcome = {
        name: frame.copy() for name, frame in final_frames.items()
    }
    invalid_preferred_outcome["Meaning_Preservation"].loc[
        invalid_preferred_outcome["Meaning_Preservation"]["Row_ID"] == "DEC-002",
        "Preferred_Outcome (for Mostly/No)",
    ] = "Keep information but not as a separate concept"
    _expect_raises(
        "Invalid preferred outcome labels are rejected",
        ValueError,
        lambda: analyze_final_ontology(
            invalid_preferred_outcome,
            bootstrap_iterations=20,
            seed=42,
        ),
    )
    invalid_issue_reason = {
        name: frame.copy() for name, frame in final_frames.items()
    }
    invalid_issue_reason["Defined_Classes"].loc[
        invalid_issue_reason["Defined_Classes"]["Row_ID"] == "DEF-001",
        "Definition_Verdict (Correct/Partly correct/Incorrect/Unsure)",
    ] = "Partly correct"
    invalid_issue_reason["Defined_Classes"].loc[
        invalid_issue_reason["Defined_Classes"]["Row_ID"] == "DEF-001",
        "Issue_Reason (select for Partly/Incorrect)",
    ] = "Wrong wording"
    _expect_raises(
        "Invalid definition issue reason labels are rejected",
        ValueError,
        lambda: analyze_final_ontology(
            invalid_issue_reason,
            bootstrap_iterations=20,
            seed=42,
        ),
    )
    _expect(
        "Relation verdicts align with explicit generic and contextual scopes",
        final_results["relations"]["scope_alignment"]["generic"]["alignment_rate"] == 1.0
        and final_results["relations"]["scope_alignment"]["corpus_context"]["alignment_rate"] == 1.0,
    )

    unsure = pd.DataFrame([
        {"Row_ID": "X-001", "Expert": f"expert_{expert}", "Judgment": "Unsure"}
        for expert in (1, 2, 3)
    ])
    unsure_summary = _summarize_judgments(
        unsure,
        item_column="Row_ID",
        raw_column="Judgment",
        allowed=("yes", "no", "unsure"),
        bootstrap_iterations=20,
        seed=42,
        partial=False,
    )
    _expect(
        "All-Unsure outcomes remain JSON-safe",
        unsure_summary["mean_score"] is None and unsure_summary["proportion_yes"] is None,
    )
    unanimous = pd.DataFrame([
        {"Row_ID": row_id, "Expert": f"expert_{expert}", "Judgment": verdict}
        for row_id, verdict in (("X-001", "Yes"), ("X-002", "No"))
        for expert in (1, 2, 3)
    ])
    unanimous_summary = _summarize_judgments(
        unanimous,
        item_column="Row_ID",
        raw_column="Judgment",
        allowed=("yes", "no", "unsure"),
        bootstrap_iterations=20,
        seed=42,
        partial=False,
        ordinal=True,
    )
    sensitivity = unanimous_summary["agreement_sensitivity"]
    _expect(
        "Raw agreement and Gwet AC1 report unanimous ratings",
        sensitivity["raw_pairwise_agreement"] == 1.0
        and sensitivity["gwet_ac1"]["coefficient"] == 1.0,
    )
    _expect(
        "Ordinal Gwet AC2 reports Unsure exclusions",
        sensitivity["gwet_ac2"]["unsure_ratings_excluded"] == 0,
    )


def test_relation_status_case() -> None:
    relations = pd.DataFrame([
        {"Term": "a", "Property": "has_part", "Filler": "b", "Evidence": "e", "Validation_Status": "ACCEPTED"},
        {"Term": "c", "Property": "has_part", "Filler": "d", "Evidence": "e", "Validation_Status": "REJECTED"},
    ])
    sample = generate_precision_sample(relations, n=10)
    _expect("Uppercase production ACCEPTED relation is sampled", len(sample) == 1 and sample.iloc[0]["Term"] == "a")


def main() -> int:
    test_ablation_artifacts_and_prompts()
    test_layer1_contracts()
    test_sampling_and_fates()
    test_item_level_nld_inference()
    test_modular_blinding_and_analysis()
    test_relation_status_case()
    print()
    if _FAILURES:
        print(f"=== EVALUATION STUDY TESTS FAILED ({len(_FAILURES)}) ===")
        return 1
    print("=== EVALUATION STUDY TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())