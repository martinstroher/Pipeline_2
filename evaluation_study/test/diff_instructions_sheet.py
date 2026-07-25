"""Compare build_instructions_sheet() output against the locked baseline.
PASS only if every row is byte-equal. Run after Phase 4 externalization.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation_study.expert_eval_generator import build_instructions_sheet

fixture_path = Path(__file__).parent / "fixtures" / "instructions_baseline.json"
baseline = json.loads(fixture_path.read_text(encoding="utf-8"))
current = build_instructions_sheet()

if len(baseline) != len(current):
    print(f"FAIL: row count differs (baseline={len(baseline)}, current={len(current)})")
    sys.exit(1)

mismatches = []
for i, (b, c) in enumerate(zip(baseline, current)):
    if b != c:
        mismatches.append((i, b, c))

if mismatches:
    print(f"FAIL: {len(mismatches)} row(s) differ")
    for i, b, c in mismatches[:5]:
        print(f"  row {i}:")
        print(f"    baseline: {b}")
        print(f"    current : {c}")
    sys.exit(1)

print(f"=== INSTRUCTIONS MATCH BASELINE ({len(current)} rows) ===")
