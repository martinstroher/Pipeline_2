"""End-to-end smoke test of domain_filters against test fixtures.

Restricts to the 4 structural filters via VALIDATION_FILTERS_ACTIVE so
no LLM calls and no embedding model loads happen.
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


def test_domain_filters_structural_only():
    tax_csv = _REPO / "test" / "output_test" / "construct_taxonomy.csv"
    assert tax_csv.exists()

    os.environ["VALIDATION_FILTERS_ACTIVE"] = (
        "single_child_intermediate,orphan_intermediate,self_loop,broken_parent"
    )
    try:
        from src.validate.domain_filters import run_domain_filters

        with tempfile.TemporaryDirectory() as tmpd:
            out_path = run_domain_filters(str(tax_csv), tmpd)
            assert Path(out_path).exists()
            df = pd.read_csv(out_path, encoding="utf-8-sig")
            expected = {
                "filter_id", "engine", "subject_type", "subject_id",
                "verdict", "action", "reason", "evidence_json",
            }
            assert set(df.columns) == expected, f"Schema: {set(df.columns) ^ expected}"
            assert (df["engine"] == "structural").all()
            seen = set(df["filter_id"].unique())
            assert seen <= {
                "single_child_intermediate", "orphan_intermediate",
                "self_loop", "broken_parent",
            }, f"Unexpected filters ran: {seen}"
            assert set(df["verdict"].unique()) <= {"PASS", "FAIL", "ABSTAIN"}
            print(f"  ✓ domain_filters produced {len(df)} verdicts across {len(seen)} filters")
            print(f"    verdict counts: {df['verdict'].value_counts().to_dict()}")
    finally:
        os.environ.pop("VALIDATION_FILTERS_ACTIVE", None)


def main() -> int:
    print("=== test_domain_filters.py ===")
    test_domain_filters_structural_only()
    print("=== ALL TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
