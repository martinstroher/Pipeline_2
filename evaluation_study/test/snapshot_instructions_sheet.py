"""Snapshot the build_instructions_sheet() output for regression-locking the
Phase 4 externalization. Writes test/fixtures/instructions_baseline.json.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation_study.expert_eval_generator import build_instructions_sheet

rows = build_instructions_sheet()
fixture_path = Path(__file__).parent / "fixtures" / "instructions_baseline.json"
fixture_path.parent.mkdir(exist_ok=True)
fixture_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {len(rows)} rows to {fixture_path}")
