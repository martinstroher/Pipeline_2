"""End-to-end smoke test of rule_applier against test_output construct CSVs.

Restricts to deterministic rules only (VALIDATION_RULES_ACTIVE) so no
LLM calls happen. Asserts the verdicts CSV is produced with the expected
schema and at least one verdict row.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def test_rule_applier_runs_deterministic_only():
    tax_csv = _REPO / "test" / "output_test" / "construct_taxonomy.csv"
    rel_csv = _REPO / "test" / "output_test" / "construct_relations.csv"
    assert tax_csv.exists(), f"Missing fixture: {tax_csv}"

    os.environ["VALIDATION_RULES_ACTIVE"] = (
        "bfo_disjointness,property_domain_range,"
        "relation_evidence_refinement,cq_coverage_gate"
    )
    try:
        from src.validate.rule_applier import run_rule_applier

        with tempfile.TemporaryDirectory() as tmpd:
            out_path = run_rule_applier(
                str(tax_csv),
                str(rel_csv) if rel_csv.exists() else None,
                tmpd,
            )
            assert Path(out_path).exists(), f"Output not created: {out_path}"
            df = pd.read_csv(out_path, encoding="utf-8-sig")

            expected_cols = {
                "rule_id", "engine", "subject_type", "subject_id",
                "verdict", "decision_on_fail", "reason", "evidence_json",
            }
            assert set(df.columns) == expected_cols, (
                f"Schema mismatch: missing {expected_cols - set(df.columns)}, "
                f"extra {set(df.columns) - expected_cols}"
            )
            assert len(df) > 0, "No verdicts produced"

            # All rule_ids must be from the active deterministic subset
            active = {
                "bfo_disjointness", "property_domain_range",
                "relation_evidence_refinement", "cq_coverage_gate",
            }
            seen = set(df["rule_id"].unique())
            assert seen <= active, f"Unexpected rules ran: {seen - active}"

            # Every row must have a valid verdict
            valid_verdicts = {"PASS", "FAIL", "ABSTAIN"}
            assert set(df["verdict"].unique()) <= valid_verdicts

            # Engine column must be 'deterministic' for all rows (no LLM ran)
            assert (df["engine"] == "deterministic").all(), "Non-deterministic verdict slipped in"

            print(f"  ✓ rule_applier produced {len(df)} verdicts across {len(seen)} rules")
            print(f"    rule_ids: {sorted(seen)}")
            print(f"    verdict counts: {df['verdict'].value_counts().to_dict()}")
    finally:
        os.environ.pop("VALIDATION_RULES_ACTIVE", None)


def main() -> int:
    print("=== test_rule_applier.py ===")
    test_rule_applier_runs_deterministic_only()
    print("=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
