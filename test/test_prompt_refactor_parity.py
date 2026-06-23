"""
Parity test — verifies the prompt loader + prompt_blocks.yaml refactor
preserves semantic intent and does not introduce unresolved markers.

Three checks:
1. Loader produces no unresolved ``<<...>>`` markers in any prompt.
2. Every refactored prompt diffs by < 25 changed lines vs. its pre-refactor
   snapshot in ``test/fixtures/prompts_pre_refactor/``.
3. Runtime placeholders ({batch_size}, {chunk_text}, {category}, etc.)
   that existed in the pre-refactor snapshot still exist after loading.

Run:
    python test/test_prompt_refactor_parity.py
"""

import difflib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.prompt_loader import load_prompt

SNAPSHOT = REPO_ROOT / "test" / "fixtures" / "prompts_pre_refactor"
SEP = "[PROMPT_TEMPLATE]"
DRIFT_THRESHOLD = 25
_BLOCK_MARKER = re.compile(r"<<[a-zA-Z_][a-zA-Z0-9_]*>>")
_RUNTIME_PLACEHOLDER = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")

_FAILED: list[str] = []


def _split(text: str) -> tuple[str, str]:
    if SEP not in text:
        return "", text.strip()
    sys_part, body = text.split(SEP, 1)
    return sys_part.replace("[SYSTEM_INSTRUCTION]", "").strip(), body.strip()


def _changed_lines(a: str, b: str) -> int:
    diff = difflib.unified_diff(a.splitlines(), b.splitlines(), n=0, lineterm="")
    return sum(1 for l in diff if l and l[0] in "+-" and not l.startswith(("+++", "---")))


def _check_no_markers(name: str, sys_part: str, body: str) -> None:
    merged = sys_part + "\n" + body
    stragglers = _BLOCK_MARKER.findall(merged)
    if stragglers:
        _FAILED.append(f"[FAIL] {name}: unresolved markers {sorted(set(stragglers))}")
    else:
        print(f"[OK]   {name}: no unresolved markers")


def _check_drift(name: str, orig_sys: str, orig_body: str, new_sys: str, new_body: str) -> None:
    s = _changed_lines(orig_sys, new_sys)
    b = _changed_lines(orig_body, new_body)
    total = s + b
    if total > DRIFT_THRESHOLD:
        _FAILED.append(
            f"[FAIL] {name}: {total} changed lines vs. snapshot "
            f"(sys={s}, body={b}, threshold={DRIFT_THRESHOLD})"
        )
    else:
        print(f"[OK]   {name}: {total} changed lines vs. snapshot")


def _check_runtime_placeholders(name: str, original: str, refactored: str) -> None:
    orig = set(_RUNTIME_PLACEHOLDER.findall(original))
    new = set(_RUNTIME_PLACEHOLDER.findall(refactored))
    # Allow new placeholders to appear (unlikely) but original ones must survive.
    missing = orig - new
    if missing:
        _FAILED.append(
            f"[FAIL] {name}: runtime placeholders dropped after refactor: {sorted(missing)}"
        )
    else:
        kept = sorted(orig) or "(none)"
        print(f"[OK]   {name}: runtime placeholders preserved {kept}")


def main() -> int:
    snapshots = sorted(SNAPSHOT.glob("*.txt"))
    print(f"Checking {len(snapshots)} prompts against pre-refactor snapshots\n")
    for snap in snapshots:
        original = snap.read_text(encoding="utf-8")
        orig_sys, orig_body = _split(original)
        try:
            new_sys, new_body = load_prompt(snap.name)
        except Exception as e:
            _FAILED.append(f"[FAIL] {snap.name}: load_prompt raised {e!r}")
            continue
        _check_no_markers(snap.name, new_sys, new_body)
        _check_drift(snap.name, orig_sys, orig_body, new_sys, new_body)
        _check_runtime_placeholders(snap.name, original, new_sys + "\n" + new_body)

    print()
    if _FAILED:
        print(f"=== PARITY FAILED ({len(_FAILED)} failure(s)) ===")
        for f in _FAILED:
            print(f)
        return 1
    print(f"=== PARITY PASSED ({len(snapshots) * 3} checks) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
