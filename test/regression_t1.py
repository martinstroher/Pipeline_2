"""Regression test for the deterministic tail of the pipeline.

Re-runs Steps 6d → 7 → 7b on the canonical T1 inputs in a throwaway temp
directory, then compares against test/fixtures/t1_baseline.json. No LLM
calls. Costs zero.

Tolerances (intentionally loose so the refactor has room to make
internal changes that don't alter semantics):
  - 6d action counts within ±5% of baseline (or ±2 absolute, whichever is larger)
  - 7b class count within ±10% of baseline
  - 7b individuals + upper_iris_referenced within ±10%
  - HermiT skipped (no Java requirement during refactor); syntax + structure must PASS
  - CSV columns must be a superset of the baseline column set (no column removals;
    additions are flagged but not fatal)

Exit code 0 = pass, non-zero = fail. Prints a one-line summary per check.
"""
import json
import os
import shutil
import sys
import tempfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASELINE_DIR = os.path.join(ROOT, "output", "refined", "t1")
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "t1_baseline.json")

_FAILED: list[str] = []


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def _fail(msg: str) -> None:
    _FAILED.append(msg)
    print(f"[FAIL] {msg}")


def _within(actual: int, expected: int, pct: float, abs_floor: int = 2) -> bool:
    tol = max(abs_floor, int(round(expected * pct)))
    return abs(actual - expected) <= tol


def main() -> int:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    with tempfile.TemporaryDirectory(prefix="regression_t1_") as tmp:
        # Stage the 6c inputs the regression step needs.
        for name in ("6c_taxonomy_cleaned.csv", "6c_relations_cleaned.csv"):
            shutil.copy(os.path.join(BASELINE_DIR, name), os.path.join(tmp, name))

        tax_in = os.path.join(tmp, "6c_taxonomy_cleaned.csv")
        rel_in = os.path.join(tmp, "6c_relations_cleaned.csv")

        # ── Step 6d ─────────────────────────────────────────────────
        from src.modules.validate.relation_reclassifier import run_relation_reclassification
        tax_6d = run_relation_reclassification(tax_in, rel_in)
        log_6d = os.path.join(tmp, "6d_reclassification_log.csv")

        log_df = pd.read_csv(log_6d, encoding="utf-8-sig")
        counts = log_df["Action"].value_counts().to_dict()
        for action, expected in baseline["6d_action_counts"].items():
            actual = int(counts.get(action, 0))
            if _within(actual, int(expected), pct=0.05, abs_floor=2):
                _ok(f"6d action {action}: {actual} (baseline {expected})")
            else:
                _fail(f"6d action {action}: {actual} (baseline {expected}, tol ±5%/±2)")

        # CSV column parity (must be superset of baseline)
        for fname in ("6d_taxonomy_reclassified.csv", "6d_reclassification_log.csv"):
            actual_cols = set(pd.read_csv(os.path.join(tmp, fname), encoding="utf-8-sig", nrows=0).columns)
            expected_cols = set(baseline["files"][fname]["columns"])
            missing = expected_cols - actual_cols
            if not missing:
                _ok(f"{fname} columns: {len(actual_cols)} (baseline {len(expected_cols)})")
            else:
                _fail(f"{fname} missing columns: {sorted(missing)}")

        # ── Step 7 ──────────────────────────────────────────────────
        from src.modules.emit.owl_exporter import run_owl_export
        ttl_path = os.path.join(tmp, "7_ontology.ttl")
        run_owl_export(tax_6d, output_path=ttl_path, relations_csv=rel_in)
        if not os.path.exists(ttl_path):
            _fail("Step 7 did not produce 7_ontology.ttl")
            return _exit()
        _ok(f"Step 7 wrote {os.path.basename(ttl_path)} ({os.path.getsize(ttl_path)} bytes)")

        # ── Step 7b ─────────────────────────────────────────────────
        from src.modules.emit.verifier import run_ontology_verification
        report = run_ontology_verification(
            ttl_path,
            output_path=os.path.join(tmp, "7b_verification_report.json"),
            skip_oops=True,
            skip_reasoner=True,
        )

        b7 = baseline["7b_summary"]
        if report["layers"]["syntax"]["status"] == b7["syntax_status"]:
            _ok(f"7b syntax: {report['layers']['syntax']['status']}")
        else:
            _fail(f"7b syntax: {report['layers']['syntax']['status']} (baseline {b7['syntax_status']})")

        struct = report["layers"]["structure"]
        if struct["status"] == b7["structure_status"]:
            _ok(f"7b structure: {struct['status']}")
        else:
            _fail(f"7b structure: {struct['status']} (baseline {b7['structure_status']})")

        for k, pct in (("classes", 0.10), ("individuals", 0.10), ("upper_iris_referenced", 0.10)):
            actual = int(struct.get(k, 0))
            expected = int(b7[k])
            if _within(actual, expected, pct=pct, abs_floor=3):
                _ok(f"7b {k}: {actual} (baseline {expected})")
            else:
                _fail(f"7b {k}: {actual} (baseline {expected}, tol ±{int(pct*100)}%)")

    return _exit()


def _exit() -> int:
    if _FAILED:
        print(f"\n=== REGRESSION FAILED — {len(_FAILED)} check(s) ===")
        return 1
    print("\n=== REGRESSION PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
