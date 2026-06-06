"""End-to-end smoke test of run_validate against test fixtures.

Restricts to deterministic rules + structural filters (no LLM, no
embedding model) so the test stays cheap and offline.
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


def test_run_validate_e2e_structural_only():
    tax_csv = _REPO / "test" / "output_test" / "construct_taxonomy.csv"
    rel_csv = _REPO / "test" / "output_test" / "construct_relations.csv"
    assert tax_csv.exists() and rel_csv.exists()

    os.environ["VALIDATION_RULES_ACTIVE"] = (
        "bfo_disjointness,property_domain_range,"
        "relation_evidence_refinement,cq_coverage_gate"
    )
    os.environ["VALIDATION_FILTERS_ACTIVE"] = (
        "single_child_intermediate,orphan_intermediate,self_loop,broken_parent"
    )
    try:
        from src.validate.run import run_validate

        with tempfile.TemporaryDirectory() as tmpd:
            # Copy inputs into tmpd so validate writes alongside them
            new_tax = Path(tmpd) / "construct_taxonomy.csv"
            new_rel = Path(tmpd) / "construct_relations.csv"
            new_tax.write_bytes(tax_csv.read_bytes())
            new_rel.write_bytes(rel_csv.read_bytes())

            final_tax, final_rel = run_validate(
                str(new_tax),
                tmpd,
                relations_csv=str(new_rel),
            )
            assert final_tax and Path(final_tax).exists()
            assert final_rel and Path(final_rel).exists()

            tax_df = pd.read_csv(final_tax, encoding="utf-8-sig")
            rel_df = pd.read_csv(final_rel, encoding="utf-8-sig")
            assert len(tax_df) > 0
            assert len(rel_df) > 0

            for name in (
                "validate_rule_verdicts.csv",
                "validate_filter_verdicts.csv",
                "validate_taxonomy.csv",
                "validate_relations.csv",
                "validate_log.csv",
                "validate_per_condition_stats.csv",
            ):
                assert (Path(tmpd) / name).exists(), f"missing {name}"

            log_df = pd.read_csv(Path(tmpd) / "validate_log.csv", encoding="utf-8-sig")
            stats_df = pd.read_csv(
                Path(tmpd) / "validate_per_condition_stats.csv", encoding="utf-8-sig"
            )
            print(f"  ✓ taxonomy {len(tax_df)} rows, relations {len(rel_df)} rows")
            print(f"  ✓ {len(log_df)} aggregator actions, {len(stats_df)} stats rows")
    finally:
        os.environ.pop("VALIDATION_RULES_ACTIVE", None)
        os.environ.pop("VALIDATION_FILTERS_ACTIVE", None)


def main() -> int:
    print("=== test_run_validate.py ===")
    test_run_validate_e2e_structural_only()
    print("=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
