"""Regression checks for the standalone Condition-D raw-context prompt."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation_study.prompt_loader import load_prompt


def main() -> int:
    system, template = load_prompt("ablation_categorization_rag.txt")
    assert system.strip()
    assert "{categories_block}" in template
    assert "{json_batch}" in template
    assert '"context"' in template
    assert '"nld"' not in template
    assert "<<" not in system + template
    print("=== STUDY PROMPT TEST PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
