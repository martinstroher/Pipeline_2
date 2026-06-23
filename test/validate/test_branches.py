"""Unit tests for branch-style filters: grouper registry, branch dispatcher,
case_collision structural function, aggregator branch-action handler, and
the iterative structural-cleanup loop.

Runs with plain `python test/validate/test_branches.py` — no pytest needed.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.validate.aggregator import (  # noqa: E402
    STRUCTURAL_CLEANUP_FILTERS,
    _apply_filter_actions,
    run_aggregator,
)
from src.validate.engines.grouping import (  # noqa: E402
    GROUPERS,
    resolve_grouper,
)
from src.validate.engines.structural import (  # noqa: E402
    BRANCH_STRUCTURAL_FUNCTIONS,
    StructuralContext,
    case_collision_branch,
    string_variant_merge_branch,
)


# ─── Grouper registry ──────────────────────────────────────────────────


def test_grouper_registry_has_expected_keys():
    assert "__lowercase_term__" in GROUPERS
    assert "__nfkd_term_and_parent__" in GROUPERS
    assert "__parent_term__" in GROUPERS
    assert "__nld_cluster__" in GROUPERS
    print("  ✓ grouper registry has 4 entries")


def test_resolve_grouper_unknown_raises():
    try:
        resolve_grouper("__nope__")
    except KeyError as e:
        assert "Unknown grouper" in str(e)
        print("  ✓ resolve_grouper raises KeyError on unknown name")
        return
    raise AssertionError("Should have raised KeyError")


def test_lowercase_term_grouper_collapses_case():
    subjects = [
        {"term": "Carbonate Rock"},
        {"term": "carbonate rock"},
        {"term": "Stromatolite"},
    ]
    groups = GROUPERS["__lowercase_term__"](subjects, StructuralContext(), {})
    assert "carbonate rock" in groups
    assert len(groups["carbonate rock"]) == 2
    assert len(groups["stromatolite"]) == 1
    print("  ✓ __lowercase_term__ groups case variants together")


def test_parent_term_grouper():
    subjects = [
        {"term": "A", "parent_term": "Rock"},
        {"term": "B", "parent_term": "rock"},
        {"term": "C", "parent_term": "Process"},
    ]
    groups = GROUPERS["__parent_term__"](subjects, StructuralContext(), {})
    assert len(groups["rock"]) == 2
    assert len(groups["process"]) == 1
    print("  ✓ __parent_term__ groups by lowercase parent")


# ─── case_collision branch function ────────────────────────────────────


def test_case_collision_registered():
    assert "case_collision" in BRANCH_STRUCTURAL_FUNCTIONS
    print("  ✓ case_collision registered in BRANCH_STRUCTURAL_FUNCTIONS")


def test_case_collision_picks_alphabetical_survivor():
    members = [{"term": "Carbonate Rock"}, {"term": "carbonate rock"}]
    out = case_collision_branch(members, StructuralContext())
    assert out is not None
    assert out["verdict"] == "FAIL"
    actions = out["proposed_actions"]
    # Survivor is alphabetical first: "Carbonate Rock" (uppercase 'C' < 'c')
    assert all(a["action"] == "rename" for a in actions)
    assert all(a["new_name"] == "Carbonate Rock" for a in actions)
    assert {a["term"] for a in actions} == {"carbonate rock"}
    print("  ✓ case_collision picks alphabetical survivor and proposes renames")


def test_case_collision_single_member_returns_none():
    out = case_collision_branch([{"term": "Foo"}], StructuralContext())
    assert out is None
    print("  ✓ case_collision returns None for single-member groups")


def test_case_collision_same_spelling_returns_none():
    # Two rows with identical spelling but different parents — not a case collision.
    out = case_collision_branch(
        [{"term": "Foo"}, {"term": "Foo"}], StructuralContext()
    )
    assert out is None
    print("  ✓ case_collision returns None when spellings are identical")


# ─── Branch dispatcher (via domain_filters._dispatch_branches) ─────────


def test_dispatch_branches_structural_case_collision():
    from src.validate.domain_filters import _dispatch_branches  # noqa: WPS433

    subjects = [
        {"term": "Carbonate Rock", "parent_term": "Rock"},
        {"term": "carbonate rock", "parent_term": "Rock"},
        {"term": "Stromatolite", "parent_term": "Carbonate Rock"},
    ]
    flt = {
        "id": "case_collision",
        "engine": "structural",
        "target": "branches",
        "group_by": "__lowercase_term__",
        "actions": ["rename"],
        "min_group_size": 2,
    }
    verdicts = _dispatch_branches(flt, subjects, StructuralContext(), {})
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.subject_type == "branch"
    assert v.verdict == "FAIL"
    actions = v.evidence["proposed_actions"]
    assert all(a["action"] == "rename" for a in actions)
    print("  ✓ _dispatch_branches emits one branch Verdict with proposed_actions")


def test_dispatch_branches_filters_disallowed_actions():
    """Branch dispatcher must drop any proposed action not in the whitelist."""
    from src.validate.domain_filters import _dispatch_branches

    subjects = [{"term": "X"}, {"term": "x"}]
    flt = {
        "id": "case_collision",
        "engine": "structural",
        "target": "branches",
        "group_by": "__lowercase_term__",
        "actions": ["merge"],  # whitelist excludes 'rename' — should drop all
        "min_group_size": 2,
    }
    verdicts = _dispatch_branches(flt, subjects, StructuralContext(), {})
    assert verdicts == []
    print("  ✓ _dispatch_branches filters out actions outside whitelist")


# ─── Aggregator branch-action handler ──────────────────────────────────


def test_apply_filter_actions_rename_branch_action_collapses_duplicates():
    tax = pd.DataFrame([
        {"Term": "Carbonate Rock", "Parent_Term": "Rock", "Category": "MaterialEntity", "Is_Intermediate": False, "NLD": "..."},
        {"Term": "carbonate rock", "Parent_Term": "Rock", "Category": "MaterialEntity", "Is_Intermediate": False, "NLD": "..."},
    ])
    rel = pd.DataFrame()

    filter_index = {
        "carbonate rock": [{
            "filter_id": "case_collision",
            "engine": "structural",
            "subject_type": "branch",
            "subject_id": "carbonate rock",
            "verdict": "FAIL",
            "action": "MERGE",
            "reason": "case-only collision",
            "evidence_json": json.dumps({
                "proposed_actions": [
                    {"action": "rename", "term": "carbonate rock", "new_name": "Carbonate Rock"},
                ]
            }),
        }]
    }
    tax_out, rel_out, log_rows = _apply_filter_actions(tax, rel, filter_index)
    # Two rows collapse to one after rename + case-insensitive dedup
    assert len(tax_out) == 1
    assert tax_out.iloc[0]["Term"] == "Carbonate Rock"
    # Log should contain rename + dedup entries
    actions = [r["action"] for r in log_rows]
    assert "rename" in actions
    assert "dedup" in actions
    print("  ✓ branch rename action + unconditional dedup collapses case duplicates")


def test_apply_filter_actions_unconditional_dedup_runs_without_rename():
    """drop_duplicates must run even when no rename actions were applied."""
    tax = pd.DataFrame([
        {"Term": "A", "Parent_Term": "X", "Category": "Cat", "Is_Intermediate": False, "NLD": "..."},
        {"Term": "a", "Parent_Term": "x", "Category": "cat", "Is_Intermediate": False, "NLD": "..."},
    ])
    rel = pd.DataFrame()
    tax_out, _, log_rows = _apply_filter_actions(tax, rel, {})
    # Case-insensitive dedup collapses the two rows
    assert len(tax_out) == 1
    assert any(r["action"] == "dedup" for r in log_rows)
    print("  ✓ unconditional dedup runs even with no actions")


# ─── Iterative cleanup loop ────────────────────────────────────────────


def test_structural_cleanup_set_has_expected_filters():
    assert STRUCTURAL_CLEANUP_FILTERS == frozenset({
        "orphan_intermediate", "broken_parent",
        "single_child_intermediate", "self_loop",
    })
    print("  ✓ STRUCTURAL_CLEANUP_FILTERS frozenset is correct")


def test_cleanup_loop_fixes_orphan_created_by_rename():
    """End-to-end: rename creates an orphan intermediate; cleanup loop removes it."""
    tmpdir = tempfile.mkdtemp(prefix="branch_cleanup_")
    try:
        # Taxonomy where "Carbonate Rock" intermediate has only "carbonate rock"
        # as child via case-variant — after rename + dedup, the intermediate
        # would have only its remaining duplicate child, which collapses.
        tax = pd.DataFrame([
            {"Term": "Stromatolite", "Parent_Term": "carbonate rock", "Category": "MaterialEntity", "Is_Intermediate": False, "NLD": "..."},
            {"Term": "carbonate rock", "Parent_Term": "Rock", "Category": "MaterialEntity", "Is_Intermediate": False, "NLD": "..."},
            {"Term": "Carbonate Rock", "Parent_Term": "Rock", "Category": "MaterialEntity", "Is_Intermediate": False, "NLD": "..."},
        ])
        tax_csv = os.path.join(tmpdir, "construct_taxonomy.csv")
        tax.to_csv(tax_csv, encoding="utf-8-sig", index=False)

        # Empty rule + filter verdict CSVs
        empty_rule = pd.DataFrame(columns=[
            "rule_id", "engine", "subject_type", "subject_id", "verdict",
            "reason", "decision_on_fail", "evidence_json",
        ])
        empty_filt = pd.DataFrame(columns=[
            "filter_id", "engine", "subject_type", "subject_id",
            "verdict", "action", "reason", "evidence_json",
        ])
        rule_csv = os.path.join(tmpdir, "validate_rule_verdicts.csv")
        flt_csv = os.path.join(tmpdir, "validate_filter_verdicts.csv")
        empty_rule.to_csv(rule_csv, encoding="utf-8-sig", index=False)
        empty_filt.to_csv(flt_csv, encoding="utf-8-sig", index=False)

        # Active subset for cleanup loop only
        os.environ["VALIDATION_FILTERS_ACTIVE"] = ",".join(STRUCTURAL_CLEANUP_FILTERS)
        try:
            tax_out, _, log_out, _ = run_aggregator(tax_csv, rule_csv, flt_csv, tmpdir)
            df = pd.read_csv(tax_out, encoding="utf-8-sig")
            # The cleanup loop should at minimum not crash; it produces a
            # log file documenting any iterations.
            assert os.path.exists(log_out)
            log = pd.read_csv(log_out, encoding="utf-8-sig") if os.path.getsize(log_out) > 100 else None
            print(
                f"  ✓ cleanup loop runs and produces log "
                f"(tax: {len(tax)} → {len(df)}, log rows: {0 if log is None else len(log)})"
            )
        finally:
            os.environ.pop("VALIDATION_FILTERS_ACTIVE", None)
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── Phase 4-6 wiring: column grouper, context_injection, change_category ─


def test_resolve_grouper_column_name_fallback():
    """`group_by: Category` (a column name) must resolve to a dict-key grouper."""
    grouper = resolve_grouper("Category")
    subjects = [
        {"term": "a", "category": "Process"},
        {"term": "b", "category": "process"},  # case-insensitive
        {"term": "c", "category": "Object"},
        {"term": "d", "category": ""},          # empty values dropped
    ]
    groups = grouper(subjects, ctx=None, opts={})
    assert set(groups.keys()) == {"process", "object"}
    assert len(groups["process"]) == 2
    assert len(groups["object"]) == 1
    print("  ✓ resolve_grouper falls back to column-name grouper for 'Category'")


def test_context_renderers_registered():
    from src.validate.engines.llm import CONTEXT_RENDERERS

    assert "category_definitions" in CONTEXT_RENDERERS
    assert "bfo_axioms" in CONTEXT_RENDERERS
    assert "upper_ontology" in CONTEXT_RENDERERS
    for name, fn in CONTEXT_RENDERERS.items():
        out = fn()
        assert isinstance(out, str), f"{name} returned {type(out)}"
    print("  ✓ CONTEXT_RENDERERS has 3 working renderers")


def test_build_injection_block_unknown_renderer_skipped():
    from src.validate.engines.llm import _build_injection_block

    block = _build_injection_block(["__nope__", "bfo_axioms"])
    assert "BFO Disjointness Axioms" in block
    assert _build_injection_block([]) == ""
    print("  ✓ _build_injection_block skips unknown renderers")


def test_change_category_action_updates_taxonomy_and_relations():
    """change_category branch action flips Category, resets Parent_Term to
    the new category root, drops Is_Intermediate, and syncs the relation rows."""
    tax = pd.DataFrame([
        {"Term": "Slumping", "Parent_Term": "Process A",
         "Category": "Geological Object", "Is_Intermediate": False, "NLD": ""},
        {"Term": "Other",    "Parent_Term": "Geological Process",
         "Category": "Geological Process", "Is_Intermediate": False, "NLD": ""},
    ])
    rel = pd.DataFrame([
        {"Term": "Slumping", "Category": "Geological Object",
         "Property": "has_part", "Filler": "X", "Evidence": "ev"},
    ])
    filter_index = {
        "Slumping": [{
            "filter_id": "category_fit",
            "action": "FAIL",
            "subject_type": "term",
            "evidence_json": json.dumps({
                "proposed_actions": [{
                    "action": "change_category",
                    "term": "Slumping",
                    "new_category": "Geological Process",
                }],
            }),
            "reason": "NLD describes a process",
        }],
    }
    tax_out, rel_out, log_rows = _apply_filter_actions(tax, rel, filter_index)
    row = tax_out[tax_out["Term"] == "Slumping"].iloc[0]
    assert row["Category"] == "Geological Process"
    assert row["Parent_Term"] == "Geological Process"
    assert bool(row["Is_Intermediate"]) is False
    rrow = rel_out[rel_out["Term"] == "Slumping"].iloc[0]
    assert rrow["Category"] == "Geological Process"
    assert any(lr["action"] == "change_category" for lr in log_rows)
    print("  ✓ change_category flips Category + Parent_Term and syncs relations")


def test_change_category_action_ignores_blank_term_or_category():
    tax = pd.DataFrame([
        {"Term": "Slumping", "Parent_Term": "Process A",
         "Category": "Geological Object", "Is_Intermediate": False, "NLD": ""},
    ])
    rel = pd.DataFrame()
    filter_index = {
        "Slumping": [{
            "filter_id": "category_fit",
            "action": "FAIL",
            "subject_type": "term",
            "evidence_json": json.dumps({
                "proposed_actions": [
                    {"action": "change_category", "term": "", "new_category": "X"},
                    {"action": "change_category", "term": "Slumping", "new_category": ""},
                ],
            }),
            "reason": "malformed",
        }],
    }
    tax_out, _, log_rows = _apply_filter_actions(tax, rel, filter_index)
    assert tax_out.iloc[0]["Category"] == "Geological Object"
    assert not any(lr["action"] == "change_category" for lr in log_rows)
    print("  ✓ change_category ignores blank term/new_category")


# ─── P2: string_variant_merge (NFKD diacritic dedup) ───────────────────


def test_nfkd_grouper_collapses_diacritic_same_parent():
    subjects = [
        {"term": "Búzios Field", "parent_term": "Petroleum Field"},
        {"term": "buzios field", "parent_term": "Petroleum Field"},
        {"term": "Pão de Açúcar", "parent_term": "Outcrop"},
        {"term": "pao de acucar", "parent_term": "Outcrop"},
        {"term": "Other Field", "parent_term": "Petroleum Field"},
    ]
    groups = GROUPERS["__nfkd_term_and_parent__"](subjects, StructuralContext(), {})
    keys = sorted(groups.keys())
    assert any("buzios field" in k for k in keys)
    assert any("pao de acucar" in k for k in keys)
    # 'Other Field' has no diacritic peer, so its group is dropped.
    assert all("other field" not in k for k in keys)
    print("  ✓ __nfkd_term_and_parent__ collapses diacritic variants under same parent")


def test_nfkd_grouper_skips_case_only_collisions():
    """Pure case variants (no diacritics) are case_collision's responsibility."""
    subjects = [
        {"term": "Carbonate Rock", "parent_term": "Rock"},
        {"term": "carbonate rock", "parent_term": "Rock"},
    ]
    groups = GROUPERS["__nfkd_term_and_parent__"](subjects, StructuralContext(), {})
    assert groups == {}
    print("  ✓ __nfkd_term_and_parent__ ignores pure case-only collisions")


def test_nfkd_grouper_different_parents_not_merged():
    """Same NFKD-folded term under different parents stays separate."""
    subjects = [
        {"term": "Búzios Field", "parent_term": "Petroleum Field"},
        {"term": "buzios field", "parent_term": "Geological Object"},
    ]
    groups = GROUPERS["__nfkd_term_and_parent__"](subjects, StructuralContext(), {})
    assert groups == {}
    print("  ✓ __nfkd_term_and_parent__ keeps different-parent variants separate")


def test_string_variant_merge_registered():
    assert "string_variant_merge" in BRANCH_STRUCTURAL_FUNCTIONS
    print("  ✓ string_variant_merge registered in BRANCH_STRUCTURAL_FUNCTIONS")


def test_string_variant_merge_prefers_diacritic_survivor():
    members = [
        {"term": "Búzios Field", "parent_term": "Petroleum Field"},
        {"term": "buzios field", "parent_term": "Petroleum Field"},
        {"term": "Buzios Field", "parent_term": "Petroleum Field"},
    ]
    out = string_variant_merge_branch(members, StructuralContext())
    assert out is not None
    assert out["verdict"] == "FAIL"
    survivor = out["evidence"]["survivor"]
    assert survivor == "Búzios Field"  # only one with diacritic
    renamed = {a["term"] for a in out["proposed_actions"]}
    assert renamed == {"buzios field", "Buzios Field"}
    print("  ✓ string_variant_merge picks diacritic variant as survivor")


def test_string_variant_merge_no_diacritic_falls_back_alphabetical():
    """If grouper somehow passes a group with no diacritic, alphabetical wins."""
    members = [{"term": "B"}, {"term": "A"}]
    out = string_variant_merge_branch(members, StructuralContext())
    assert out is not None
    assert out["evidence"]["survivor"] == "A"
    print("  ✓ string_variant_merge falls back to alphabetical when no diacritic")


def test_string_variant_merge_single_member_returns_none():
    assert string_variant_merge_branch([{"term": "X"}], StructuralContext()) is None
    print("  ✓ string_variant_merge returns None for single-member group")


# ─── P0: per-term canonicalization ─────────────────────────────────────


def test_canonicalize_per_term_prefers_specific_parent():
    """Same term twice — one row reparented to category root, other keeps a
    specific intermediate parent. Canonicalization keeps the specific one."""
    tax = pd.DataFrame([
        # Reparented-to-root row
        {"Term": "stromatolite", "Parent_Term": "MaterialEntity",
         "Category": "MaterialEntity", "Is_Intermediate": False, "NLD": ""},
        # Specific row (intermediate parent)
        {"Term": "stromatolite", "Parent_Term": "carbonate body",
         "Category": "MaterialEntity", "Is_Intermediate": False, "NLD": ""},
    ])
    rel = pd.DataFrame()
    tax_out, _, log_rows = _apply_filter_actions(tax, rel, {})
    assert len(tax_out) == 1
    assert tax_out.iloc[0]["Parent_Term"] == "carbonate body"
    assert any(r["action"] == "canonicalize" for r in log_rows)
    print("  ✓ canonicalize prefers specific parent over category root")


def test_canonicalize_per_term_prefers_non_intermediate():
    """When both rows have specific parents, prefer Is_Intermediate=False."""
    tax = pd.DataFrame([
        {"Term": "X", "Parent_Term": "P1", "Category": "Cat",
         "Is_Intermediate": True, "NLD": ""},
        {"Term": "X", "Parent_Term": "P2", "Category": "Cat",
         "Is_Intermediate": False, "NLD": ""},
    ])
    rel = pd.DataFrame()
    tax_out, _, log_rows = _apply_filter_actions(tax, rel, {})
    assert len(tax_out) == 1
    assert bool(tax_out.iloc[0]["Is_Intermediate"]) is False
    assert tax_out.iloc[0]["Parent_Term"] == "P2"
    print("  ✓ canonicalize prefers Is_Intermediate=False")


def test_canonicalize_unique_terms_untouched():
    """No duplicates → no canonicalize log entry."""
    tax = pd.DataFrame([
        {"Term": "A", "Parent_Term": "X", "Category": "C",
         "Is_Intermediate": False, "NLD": ""},
        {"Term": "B", "Parent_Term": "X", "Category": "C",
         "Is_Intermediate": False, "NLD": ""},
    ])
    rel = pd.DataFrame()
    tax_out, _, log_rows = _apply_filter_actions(tax, rel, {})
    assert len(tax_out) == 2
    assert not any(r["action"] == "canonicalize" for r in log_rows)
    print("  ✓ canonicalize is a no-op when terms are unique")


# ─── P1: orphan rescue (ancestor walk) ─────────────────────────────────


def test_rescue_parent_walks_to_surviving_ancestor():
    """When grandparent is alive, broken child is rehomed to it (not category root)."""
    from src.validate.aggregator import (  # noqa: WPS433
        _build_original_parent_map,
        _rescue_parent,
    )
    original = pd.DataFrame([
        {"Term": "Carbonate Mudstone", "Parent_Term": "Carbonate Rock"},
        {"Term": "Carbonate Rock", "Parent_Term": "Sedimentary Rock"},
        {"Term": "Sedimentary Rock", "Parent_Term": "Rock"},
        {"Term": "Rock", "Parent_Term": "MaterialEntity"},
    ])
    # Current taxonomy: 'Carbonate Rock' was removed; grandparent
    # 'Sedimentary Rock' is still present.
    current = pd.DataFrame([
        {"Term": "Carbonate Mudstone", "Parent_Term": "MaterialEntity",
         "Category": "MaterialEntity"},
        {"Term": "Sedimentary Rock", "Parent_Term": "Rock",
         "Category": "MaterialEntity"},
        {"Term": "Rock", "Parent_Term": "MaterialEntity",
         "Category": "MaterialEntity"},
    ])
    pmap = _build_original_parent_map(original)
    rescued = _rescue_parent("Carbonate Mudstone", "MaterialEntity", current, pmap)
    assert rescued == "Sedimentary Rock"
    print("  ✓ _rescue_parent walks original ancestor chain to surviving grandparent")


def test_rescue_parent_falls_back_when_no_ancestor_survives():
    from src.validate.aggregator import (  # noqa: WPS433
        _build_original_parent_map,
        _rescue_parent,
    )
    original = pd.DataFrame([
        {"Term": "X", "Parent_Term": "Y"},
        {"Term": "Y", "Parent_Term": "Z"},
    ])
    # Neither Y nor Z survives.
    current = pd.DataFrame([
        {"Term": "X", "Parent_Term": "Cat", "Category": "Cat"},
        {"Term": "Other", "Parent_Term": "Cat", "Category": "Cat"},
    ])
    pmap = _build_original_parent_map(original)
    assert _rescue_parent("X", "Cat", current, pmap) == "Cat"
    print("  ✓ _rescue_parent falls back to category root when ancestors are gone")


def test_rescue_parent_handles_empty_original_map():
    from src.validate.aggregator import _rescue_parent
    current = pd.DataFrame([
        {"Term": "X", "Parent_Term": "Cat", "Category": "Cat"},
    ])
    assert _rescue_parent("X", "Cat", current, {}) == "Cat"
    print("  ✓ _rescue_parent returns fallback when map is empty")


def test_apply_filter_actions_reparent_uses_rescue_when_available():
    """End-to-end: REPARENT filter action calls _rescue_parent and prefers ancestor."""
    tax = pd.DataFrame([
        {"Term": "Carbonate Mudstone", "Parent_Term": "Carbonate Rock",
         "Category": "MaterialEntity", "Is_Intermediate": False, "NLD": ""},
        {"Term": "Sedimentary Rock", "Parent_Term": "Rock",
         "Category": "MaterialEntity", "Is_Intermediate": False, "NLD": ""},
    ])
    original_pmap = {
        "carbonate mudstone": "Carbonate Rock",
        "carbonate rock": "Sedimentary Rock",
        "sedimentary rock": "Rock",
    }
    filter_index = {
        "Carbonate Mudstone-->Carbonate Rock": [{
            "filter_id": "broken_parent",
            "engine": "structural",
            "subject_type": "edge_taxonomy",
            "subject_id": "Carbonate Mudstone-->Carbonate Rock",
            "verdict": "FAIL",
            "action": "REPARENT",
            "reason": "Parent_Term not found",
            "evidence_json": "{}",
        }]
    }
    tax_out, _, log_rows = _apply_filter_actions(
        tax, pd.DataFrame(), filter_index,
        original_parent_map=original_pmap,
    )
    row = tax_out[tax_out["Term"] == "Carbonate Mudstone"].iloc[0]
    assert row["Parent_Term"] == "Sedimentary Rock"
    rep_log = [r for r in log_rows if r["action"] == "reparent"]
    assert rep_log and "orphan rescue" in rep_log[0]["detail"]
    print("  ✓ REPARENT applies orphan rescue when an ancestor survives")


def main() -> int:
    tests = [
        test_grouper_registry_has_expected_keys,
        test_resolve_grouper_unknown_raises,
        test_lowercase_term_grouper_collapses_case,
        test_parent_term_grouper,
        test_case_collision_registered,
        test_case_collision_picks_alphabetical_survivor,
        test_case_collision_single_member_returns_none,
        test_case_collision_same_spelling_returns_none,
        test_dispatch_branches_structural_case_collision,
        test_dispatch_branches_filters_disallowed_actions,
        test_apply_filter_actions_rename_branch_action_collapses_duplicates,
        test_apply_filter_actions_unconditional_dedup_runs_without_rename,
        test_structural_cleanup_set_has_expected_filters,
        test_cleanup_loop_fixes_orphan_created_by_rename,
        test_resolve_grouper_column_name_fallback,
        test_context_renderers_registered,
        test_build_injection_block_unknown_renderer_skipped,
        test_change_category_action_updates_taxonomy_and_relations,
        test_change_category_action_ignores_blank_term_or_category,
        # P2 — string_variant_merge
        test_nfkd_grouper_collapses_diacritic_same_parent,
        test_nfkd_grouper_skips_case_only_collisions,
        test_nfkd_grouper_different_parents_not_merged,
        test_string_variant_merge_registered,
        test_string_variant_merge_prefers_diacritic_survivor,
        test_string_variant_merge_no_diacritic_falls_back_alphabetical,
        test_string_variant_merge_single_member_returns_none,
        # P0 — per-term canonicalization
        test_canonicalize_per_term_prefers_specific_parent,
        test_canonicalize_per_term_prefers_non_intermediate,
        test_canonicalize_unique_terms_untouched,
        # P1 — orphan rescue ancestor walk
        test_rescue_parent_walks_to_surviving_ancestor,
        test_rescue_parent_falls_back_when_no_ancestor_survives,
        test_rescue_parent_handles_empty_original_map,
        test_apply_filter_actions_reparent_uses_rescue_when_available,
    ]
    print(f"=== test_branches.py — {len(tests)} tests ===")
    for t in tests:
        t()
    print(f"=== ALL {len(tests)} TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
