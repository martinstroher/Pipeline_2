"""Snapshot all prompt files' (system_instruction, prompt_template) pairs.

Writes test/fixtures/prompts_baseline.json. Phase 2 (templating refactor)
must reproduce these strings byte-for-byte after going through the new
domain-profile + template machinery, or document any diff.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.prompt_loader import load_prompt, prompt_files  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "prompts_baseline.json")


def main() -> int:
    snapshot: dict[str, dict[str, str]] = {}
    for name, _path in prompt_files():
        system, template = load_prompt(name)
        snapshot[name] = {"system": system, "template": template}

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Wrote {OUT_PATH} ({len(snapshot)} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
