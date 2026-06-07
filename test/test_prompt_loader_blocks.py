"""
Unit tests for ``src.utils.prompt_loader`` block-substitution behaviour.

Covers:
- nested-key flattening (``domain.role`` -> ``<<domain_role>>``)
- nested marker resolution (block values can reference other blocks)
- cycle detection (raises)
- missing-block detection (raises KeyError with the offending name)
- runtime ``{placeholder}`` strings survive ``<<...>>`` substitution

Run:
    python test/test_prompt_loader_blocks.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.prompt_loader import (
    _flatten,
    _interpolate_blocks,
    _resolve_inner_markers,
)


_FAILED: list[str] = []


def _expect(label: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"[OK]   {label}")
    else:
        msg = f"[FAIL] {label}"
        if detail:
            msg += f" :: {detail}"
        _FAILED.append(msg)
        print(msg)


def test_flatten() -> None:
    out = _flatten({"domain": {"role": "geo", "scope": "x"}, "foo": "bar"})
    _expect(
        "flatten produces flat keys",
        out == {"domain_role": "geo", "domain_scope": "x", "foo": "bar"},
        detail=str(out),
    )


def test_resolve_inner_markers_nested() -> None:
    blocks = {
        "role": "geoscientist",
        "persona": "You are a senior <<role>> and engineer",
    }
    out = _resolve_inner_markers(blocks)
    _expect(
        "nested marker resolves",
        out["persona"] == "You are a senior geoscientist and engineer",
        detail=str(out),
    )


def test_resolve_inner_markers_cycle() -> None:
    blocks = {"a": "x <<b>>", "b": "y <<a>>"}
    try:
        _resolve_inner_markers(blocks)
        _expect("cycle raises", False, detail="expected RuntimeError")
    except RuntimeError as e:
        _expect("cycle raises", "cycle" in str(e).lower(), detail=str(e))


def test_resolve_inner_markers_missing() -> None:
    blocks = {"a": "uses <<ghost>>"}
    try:
        _resolve_inner_markers(blocks)
        _expect("missing inner block raises", False, detail="expected KeyError")
    except KeyError as e:
        _expect("missing inner block raises", "ghost" in str(e), detail=str(e))


def test_interpolate_blocks_basic() -> None:
    text = "Hello <<who>>, value is <<n>>."
    out = _interpolate_blocks(text, {"who": "world", "n": "42"}, "demo.txt")
    _expect("basic substitution", out == "Hello world, value is 42.", detail=out)


def test_interpolate_blocks_missing_raises() -> None:
    try:
        _interpolate_blocks("Unknown <<ghost>>", {}, "demo.txt")
        _expect("missing prompt block raises", False, detail="expected KeyError")
    except KeyError as e:
        _expect(
            "missing prompt block raises",
            "ghost" in str(e) and "demo.txt" in str(e),
            detail=str(e),
        )


def test_runtime_placeholders_survive() -> None:
    # str.format-style {placeholders} must pass through untouched.
    text = "Batch: <<persona>>. Process {batch_size} items: {json_batch}"
    out = _interpolate_blocks(text, {"persona": "Expert"}, "demo.txt")
    _expect(
        "runtime placeholders untouched",
        "{batch_size}" in out and "{json_batch}" in out and "Expert" in out,
        detail=out,
    )


def test_double_brace_json_examples_survive() -> None:
    # JSON output examples in prompts use {{...}} to escape str.format braces.
    # <<...>> substitution should not touch them either.
    text = 'Example: {{"term": "x"}} and <<n>> items.'
    out = _interpolate_blocks(text, {"n": "3"}, "demo.txt")
    _expect(
        "double-brace JSON examples untouched",
        '{{"term": "x"}}' in out and "3 items" in out,
        detail=out,
    )


def main() -> int:
    test_flatten()
    test_resolve_inner_markers_nested()
    test_resolve_inner_markers_cycle()
    test_resolve_inner_markers_missing()
    test_interpolate_blocks_basic()
    test_interpolate_blocks_missing_raises()
    test_runtime_placeholders_survive()
    test_double_brace_json_examples_survive()

    print()
    if _FAILED:
        print(f"=== LOADER TESTS FAILED ({len(_FAILED)}) ===")
        return 1
    print("=== LOADER TESTS PASSED (8/8) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
