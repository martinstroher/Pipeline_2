"""Verify rendered prompts match the Phase-0 baseline snapshot.

Loads each prompt file via the current prompt_loader, compares against
test/fixtures/prompts_baseline.json. Exits 0 on byte-equal match.
Used during Phase 2 to catch any unintended drift in prompt text after
the templating refactor.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.prompt_loader import load_prompt, _PROMPTS_DIR  # noqa: E402

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "prompts_baseline.json")


def main() -> int:
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline: dict[str, dict[str, str]] = json.load(f)

    current: dict[str, dict[str, str]] = {}
    for name in sorted(os.listdir(_PROMPTS_DIR)):
        if not name.endswith(".txt"):
            continue
        system, template = load_prompt(name)
        current[name] = {"system": system, "template": template}

    failed: list[str] = []
    for name in sorted(set(baseline) | set(current)):
        if name not in current:
            failed.append(f"{name}: MISSING in current")
            continue
        if name not in baseline:
            print(f"[NEW]  {name}: present in current, absent in baseline (informational)")
            continue
        for part in ("system", "template"):
            if baseline[name][part] != current[name][part]:
                failed.append(f"{name}:{part}: text drift")
            else:
                print(f"[OK]   {name}:{part}")

    if failed:
        print(f"\n=== PROMPT DIFF DETECTED — {len(failed)} mismatch(es) ===")
        for msg in failed:
            print(f"[FAIL] {msg}")
        return 1
    print("\n=== PROMPTS MATCH BASELINE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
