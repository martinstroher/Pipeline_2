"""Unit tests for the aggregator decision policy and action applicators."""

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
    _apply_filter_actions,
    _apply_rule_actions,
    _index_filter_verdicts,
    _index_rule_verdicts,
    _load_aggregator_policy,
    _select_action,
    _summarise,
    run_aggregator,
)


# ─── Fixtures ──────────────────────────────────────────────────────────


def _tax_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Term": "carbonate_rock", "Parent_Term": "rock", "Category": "MaterialEntity", "Is_Intermediate": True, "NLD": "..."},
        {"Term": "stromatolite", "Parent_Term": "carbonate_rock", "Category": "MaterialEntity", "Is_Intermediate": False, "NLD": "..."},
        {"Term": "porosity", "Parent_Term": "MaterialEntity", "Category": "MaterialEntity", "Is_Intermediate": False, "NLD": "..."},
        {"Term": "drilling", "Parent_Term": "Process", "Category": "Process", "Is_Intermediate": False, "NLD": "..."},
    ])


def _rel_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Term": "stromatolite", "Property": "located_in", "Filler": "buzios_field", "Category": "MaterialEntity"},
        {"Term": "drilling", "Property": "has_participant", "Filler": "porosity", "Category": "Process"},
    ])


# ─── Tests ─────────────────────────────────────────────────────────────


def test_load_policy_has_default():
    policy = _load_aggregator_policy()
    assert policy, "Policy must not be empty"
    assert any(c.get("default") == "accept" for c in policy), "Policy must end in default accept"


def test_select_action_default_when_no_verdicts():
    policy = _load_aggregator_policy()
    action, src = _select_action([], policy)
    assert action == "accept"
    assert src is None


def test_select_action_drop_on_deterministic_reject():
    policy = _load_aggregator_policy()
    v = {"engine": "deterministic", "decision_on_fail": "REJECT", "rule_id": "bfo_disjointness"}
    action, src = _select_action([v], policy)
    assert action == "drop_edge"
    assert src is not None


def test_select_action_refine_on_deterministic_refine():
    policy = _load_aggregator_policy()
    v = {"engine": "deterministic", "decision_on_fail": "REFINE", "rule_id": "relation_evidence_refinement"}
    action, src = _select_action([v], policy)
    assert action == "refine_to_subclass"


def test_select_action_llm_requires_two_rejects():
    policy = _load_aggregator_policy()
    one = [{"engine": "llm", "decision_on_fail": "REJECT", "rule_id": "ontoclean_rigidity"}]
    two = one + [{"engine": "llm", "decision_on_fail": "REJECT", "rule_id": "ontoclean_identity"}]
    assert _select_action(one, policy)[0] == "accept"
    assert _select_action(two, policy)[0] == "drop_edge"


def test_apply_rule_actions_drop_edge_taxonomy_reparents():
    tax = _tax_df()
    rel = _rel_df()
    rule_df = pd.DataFrame([{
        "rule_id": "bfo_disjointness",
        "engine": "deterministic",
        "subject_type": "edge_taxonomy",
        "subject_id": "stromatolite-->carbonate_rock",
        "verdict": "FAIL",
        "decision_on_fail": "REJECT",
        "reason": "metatype clash",
        "evidence_json": "{}",
    }])
    policy = _load_aggregator_policy()
    tax_after, rel_after, log_rows = _apply_rule_actions(tax, rel, _index_rule_verdicts(rule_df), policy)
    row = tax_after[tax_after["Term"] == "stromatolite"].iloc[0]
    assert row["Parent_Term"] == "MaterialEntity"
    assert bool(row["Is_Intermediate"]) is False
    assert len(tax_after) == 4  # term preserved
    assert any(r["action"] == "drop_edge" for r in log_rows)


def test_apply_rule_actions_drop_edge_relation_removes_row():
    tax = _tax_df()
    rel = _rel_df()
    rule_df = pd.DataFrame([{
        "rule_id": "property_domain_range",
        "engine": "deterministic",
        "subject_type": "edge_relation",
        "subject_id": "drilling--has_participant-->porosity",
        "verdict": "FAIL",
        "decision_on_fail": "REJECT",
        "reason": "range violation",
        "evidence_json": "{}",
    }])
    policy = _load_aggregator_policy()
    _, rel_after, log_rows = _apply_rule_actions(tax, rel, _index_rule_verdicts(rule_df), policy)
    assert len(rel_after) == 1
    assert rel_after.iloc[0]["Term"] == "stromatolite"
    assert any(r["subject_type"] == "edge_relation" and r["action"] == "drop_edge" for r in log_rows)


def test_apply_rule_actions_refine_updates_category():
    tax = _tax_df()
    rel = _rel_df()
    rule_df = pd.DataFrame([{
        "rule_id": "relation_evidence_refinement",
        "engine": "deterministic",
        "subject_type": "edge_taxonomy",
        "subject_id": "porosity-->MaterialEntity",
        "verdict": "FAIL",
        "decision_on_fail": "REFINE",
        "reason": "evidence suggests Quality",
        "evidence_json": json.dumps({"new_category": "Quality"}),
    }])
    policy = _load_aggregator_policy()
    tax_after, _, log_rows = _apply_rule_actions(tax, rel, _index_rule_verdicts(rule_df), policy)
    row = tax_after[tax_after["Term"] == "porosity"].iloc[0]
    assert row["Category"] == "Quality"
    assert row["Parent_Term"] == "Quality"  # retargeted since old parent was the old root
    assert any(r["action"] == "refine_to_subclass" for r in log_rows)


def test_apply_filter_actions_remove_term_cascades_to_relations():
    tax = _tax_df()
    rel = _rel_df()
    filter_df = pd.DataFrame([{
        "filter_id": "overly_generic",
        "engine": "llm",
        "subject_type": "term",
        "subject_id": "porosity",
        "verdict": "FAIL",
        "action": "REMOVE",
        "reason": "too generic",
        "evidence_json": "{}",
    }])
    tax_after, rel_after, log_rows = _apply_filter_actions(tax, rel, _index_filter_verdicts(filter_df))
    assert "porosity" not in tax_after["Term"].values
    assert "porosity" not in rel_after["Filler"].values
    assert any(r["action"] == "remove" for r in log_rows)


def test_apply_filter_actions_merge_renames_cluster():
    tax = _tax_df()
    rel = _rel_df()
    filter_df = pd.DataFrame([{
        "filter_id": "near_duplicate",
        "engine": "hybrid",
        "subject_type": "term_cluster",
        "subject_id": "cluster_0",
        "verdict": "FAIL",
        "action": "MERGE",
        "reason": "near-duplicates",
        "evidence_json": json.dumps({
            "surviving_term": "carbonate_rock",
            "members": ["carbonate_rock", "stromatolite"],
        }),
    }])
    tax_after, rel_after, log_rows = _apply_filter_actions(tax, rel, _index_filter_verdicts(filter_df))
    assert "stromatolite" not in tax_after["Term"].values
    # Relation that referenced stromatolite is now referencing carbonate_rock
    assert "carbonate_rock" in rel_after["Term"].values
    assert "stromatolite" not in rel_after["Term"].values
    assert any(r["action"] == "merge" for r in log_rows)


def test_summarise_counts_per_rule_and_filter():
    rule_df = pd.DataFrame([
        {"rule_id": "bfo_disjointness", "engine": "deterministic", "verdict": "PASS"},
        {"rule_id": "bfo_disjointness", "engine": "deterministic", "verdict": "FAIL"},
        {"rule_id": "bfo_disjointness", "engine": "deterministic", "verdict": "FAIL"},
    ])
    filter_df = pd.DataFrame([
        {"filter_id": "self_loop", "engine": "structural", "verdict": "PASS"},
        {"filter_id": "self_loop", "engine": "structural", "verdict": "FAIL"},
    ])
    summary = _summarise(rule_df, filter_df)
    bfo = summary[summary["id"] == "bfo_disjointness"].iloc[0]
    assert bfo["n_total"] == 3 and bfo["n_pass"] == 1 and bfo["n_fail"] == 2
    sl = summary[summary["id"] == "self_loop"].iloc[0]
    assert sl["n_fail"] == 1


def test_run_aggregator_end_to_end_writes_all_outputs():
    with tempfile.TemporaryDirectory() as tmpd:
        tax_csv = os.path.join(tmpd, "construct_taxonomy.csv")
        rel_csv = os.path.join(tmpd, "construct_relations.csv")
        _tax_df().to_csv(tax_csv, index=False, encoding="utf-8-sig")
        _rel_df().to_csv(rel_csv, index=False, encoding="utf-8-sig")

        rule_csv = os.path.join(tmpd, "validate_rule_verdicts.csv")
        pd.DataFrame([{
            "rule_id": "bfo_disjointness",
            "engine": "deterministic",
            "subject_type": "edge_taxonomy",
            "subject_id": "stromatolite-->carbonate_rock",
            "verdict": "FAIL",
            "decision_on_fail": "REJECT",
            "reason": "...",
            "evidence_json": "{}",
        }]).to_csv(rule_csv, index=False, encoding="utf-8-sig")

        filter_csv = os.path.join(tmpd, "validate_filter_verdicts.csv")
        pd.DataFrame(columns=[
            "filter_id", "engine", "subject_type", "subject_id",
            "verdict", "action", "reason", "evidence_json",
        ]).to_csv(filter_csv, index=False, encoding="utf-8-sig")

        tax_out, rel_out, log_out, stats_out = run_aggregator(
            tax_csv, rule_csv, filter_csv, tmpd, relations_csv=rel_csv
        )
        for p in (tax_out, rel_out, log_out, stats_out):
            assert p and os.path.exists(p)

        log_df = pd.read_csv(log_out, encoding="utf-8-sig")
        assert (log_df["action"] == "drop_edge").any()


# ─── Test runner ───────────────────────────────────────────────────────


def main() -> int:
    print("=== test_aggregator.py ===")
    tests = [
        test_load_policy_has_default,
        test_select_action_default_when_no_verdicts,
        test_select_action_drop_on_deterministic_reject,
        test_select_action_refine_on_deterministic_refine,
        test_select_action_llm_requires_two_rejects,
        test_apply_rule_actions_drop_edge_taxonomy_reparents,
        test_apply_rule_actions_drop_edge_relation_removes_row,
        test_apply_rule_actions_refine_updates_category,
        test_apply_filter_actions_remove_term_cascades_to_relations,
        test_apply_filter_actions_merge_renames_cluster,
        test_summarise_counts_per_rule_and_filter,
        test_run_aggregator_end_to_end_writes_all_outputs,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"=== {failed}/{len(tests)} FAILED ===")
        return 1
    print("=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
