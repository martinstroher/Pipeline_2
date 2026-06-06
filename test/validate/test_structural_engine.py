"""Unit tests for src.validate.engines.structural.

Runs against the real Pre-Salt ontology config (no fixtures needed for
metatype/property data — that's stable across this branch).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.validate.engines.structural import (  # noqa: E402
    RULE_FUNCTIONS,
    StructuralContext,
    Verdict,
    bfo_disjointness,
    broken_parent,
    cq_coverage_gate,
    orphan_intermediate,
    property_domain_range,
    relation_evidence_refinement,
    self_loop,
    single_child_intermediate,
)


def _ok(actual: str, expected: str, label: str) -> None:
    assert actual == expected, f"{label}: expected {expected}, got {actual}"


def test_registry_has_eight_entries():
    expected = {
        "bfo_disjointness",
        "property_domain_range",
        "relation_evidence_refinement",
        "cq_coverage_gate",
        "single_child_intermediate",
        "orphan_intermediate",
        "self_loop",
        "broken_parent",
    }
    assert set(RULE_FUNCTIONS) == expected, f"Registry mismatch: {set(RULE_FUNCTIONS) ^ expected}"
    print("  ✓ registry has 8 entries")


def test_self_loop_detects_and_passes():
    ctx = StructuralContext()
    v = self_loop({"term": "X", "parent_term": "X"}, ctx)
    _ok(v.verdict, "FAIL", "self_loop FAIL")
    v = self_loop({"term": "X", "parent_term": "Y"}, ctx)
    _ok(v.verdict, "PASS", "self_loop PASS")
    print("  ✓ self_loop works")


def test_broken_parent_abstain_without_taxonomy():
    ctx = StructuralContext()
    v = broken_parent({"term": "A", "parent_term": "B"}, ctx)
    _ok(v.verdict, "ABSTAIN", "broken_parent ABSTAIN")
    print("  ✓ broken_parent abstains without taxonomy")


def test_broken_parent_with_taxonomy():
    df = pd.DataFrame(
        [
            {"Term": "Sandstone", "Parent_Term": "Sedimentary Rock", "Category": "Earth Material"},
            {"Term": "Sedimentary Rock", "Parent_Term": "Rock", "Category": "Earth Material"},
        ]
    )
    ctx = StructuralContext(taxonomy_df=df)
    # Parent exists as Term — PASS
    v = broken_parent({"term": "Sandstone", "parent_term": "Sedimentary Rock"}, ctx)
    _ok(v.verdict, "PASS", "broken_parent PASS by term match")
    # Parent exists only as Category — PASS
    v = broken_parent({"term": "Sandstone", "parent_term": "Earth Material"}, ctx)
    _ok(v.verdict, "PASS", "broken_parent PASS by category match")
    # Parent unknown — FAIL
    v = broken_parent({"term": "Sandstone", "parent_term": "Nonexistent Parent"}, ctx)
    _ok(v.verdict, "FAIL", "broken_parent FAIL on unknown parent")
    print("  ✓ broken_parent works with taxonomy")


def test_single_child_intermediate():
    df = pd.DataFrame(
        [
            {"Term": "Mineral Wrapper", "Parent_Term": "Earth Material", "Category": "Earth Material"},
            {"Term": "Calcite", "Parent_Term": "Mineral Wrapper", "Category": "Earth Material"},
        ]
    )
    ctx = StructuralContext(taxonomy_df=df)
    v = single_child_intermediate(
        {"term": "Mineral Wrapper", "is_intermediate": True}, ctx
    )
    _ok(v.verdict, "FAIL", "single_child_intermediate FAIL")
    # Non-intermediate → PASS regardless of child count
    v = single_child_intermediate(
        {"term": "Mineral Wrapper", "is_intermediate": False}, ctx
    )
    _ok(v.verdict, "PASS", "single_child_intermediate PASS non-intermediate")
    print("  ✓ single_child_intermediate works")


def test_orphan_intermediate():
    df = pd.DataFrame(
        [
            {"Term": "Orphan Group", "Parent_Term": "X", "Category": "Y"},
            {"Term": "Leaf", "Parent_Term": "X", "Category": "Y"},
        ]
    )
    ctx = StructuralContext(taxonomy_df=df)
    v = orphan_intermediate({"term": "Orphan Group", "is_intermediate": True}, ctx)
    _ok(v.verdict, "FAIL", "orphan_intermediate FAIL")
    # Leaf has no children but is not intermediate → PASS
    v = orphan_intermediate({"term": "Leaf", "is_intermediate": False}, ctx)
    _ok(v.verdict, "PASS", "orphan_intermediate PASS non-intermediate")
    print("  ✓ orphan_intermediate works")


def test_cq_coverage_gate():
    ctx = StructuralContext(cq_covered_terms={"covered_term"})
    v = cq_coverage_gate({"term": "Covered_Term"}, ctx)
    _ok(v.verdict, "PASS", "cq_coverage_gate PASS (case-insensitive)")
    v = cq_coverage_gate({"term": "Uncovered"}, ctx)
    _ok(v.verdict, "FAIL", "cq_coverage_gate FAIL")
    v = cq_coverage_gate({"term": "Whatever"}, StructuralContext())
    _ok(v.verdict, "ABSTAIN", "cq_coverage_gate ABSTAIN when no data")
    print("  ✓ cq_coverage_gate works")


def test_bfo_disjointness_abstain_on_unknown_categories():
    ctx = StructuralContext()
    v = bfo_disjointness(
        {"term": "X", "parent_term": "Y", "category": "Unknown", "parent_category": ""},
        ctx,
    )
    _ok(v.verdict, "ABSTAIN", "bfo_disjointness ABSTAIN")
    print("  ✓ bfo_disjointness abstains on unknown categories")


def test_property_domain_range_abstain_on_unknown_property():
    ctx = StructuralContext()
    v = property_domain_range(
        {
            "term": "X",
            "property": "made_up_property_xyz",
            "filler": "Y",
            "category": "Earth Material",
            "filler_category": "Earth Material",
        },
        ctx,
    )
    _ok(v.verdict, "ABSTAIN", "property_domain_range ABSTAIN on unknown property")
    print("  ✓ property_domain_range abstains on unknown property")


def test_relation_evidence_refinement_abstain_without_relations():
    ctx = StructuralContext()
    v = relation_evidence_refinement(
        {"term": "X", "category": "Earth Material"}, ctx
    )
    _ok(v.verdict, "ABSTAIN", "relation_evidence_refinement ABSTAIN")
    # With empty relations DataFrame
    ctx2 = StructuralContext(relations_df=pd.DataFrame())
    v = relation_evidence_refinement({"term": "X", "category": "Earth Material"}, ctx2)
    _ok(v.verdict, "ABSTAIN", "relation_evidence_refinement ABSTAIN on empty df")
    print("  ✓ relation_evidence_refinement abstains without relations")


def test_relation_evidence_refinement_passes_below_threshold():
    df = pd.DataFrame(
        [
            {
                "Term": "Term_X",
                "Property": "made_up_property_xyz",  # unknown → no evidence collected
                "Filler": "Y",
            }
        ]
    )
    ctx = StructuralContext(relations_df=df)
    v = relation_evidence_refinement({"term": "Term_X", "category": "Earth Material"}, ctx)
    _ok(v.verdict, "PASS", "relation_evidence_refinement PASS (zero evidence)")
    print("  ✓ relation_evidence_refinement passes below min_evidence")


def main() -> int:
    tests = [
        test_registry_has_eight_entries,
        test_self_loop_detects_and_passes,
        test_broken_parent_abstain_without_taxonomy,
        test_broken_parent_with_taxonomy,
        test_single_child_intermediate,
        test_orphan_intermediate,
        test_cq_coverage_gate,
        test_bfo_disjointness_abstain_on_unknown_categories,
        test_property_domain_range_abstain_on_unknown_property,
        test_relation_evidence_refinement_abstain_without_relations,
        test_relation_evidence_refinement_passes_below_threshold,
    ]
    print(f"=== test_structural_engine.py — {len(tests)} tests ===")
    for t in tests:
        t()
    print(f"=== ALL {len(tests)} TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
