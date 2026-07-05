"""Unit tests for src.validate.engines.llm.

Mocks `llm_client.generate` so no actual LLM calls happen. Exercises
the cache path, single + batch dispatch, and verdict mapping.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _reset_cache_dir() -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="llm_engine_test_"))
    os.environ["VALIDATION_CACHE_DIR"] = str(tmpdir)
    # Reload module to pick up new env
    import importlib
    import src.validate.engines.llm as mod
    importlib.reload(mod)
    return tmpdir


def _restore_cache_default(orig: str | None) -> None:
    if orig is None:
        os.environ.pop("VALIDATION_CACHE_DIR", None)
    else:
        os.environ["VALIDATION_CACHE_DIR"] = orig


def test_evaluate_single_pass_caches_and_short_circuits():
    orig = os.environ.get("VALIDATION_CACHE_DIR")
    tmpdir = _reset_cache_dir()
    try:
        import src.validate.engines.llm as mod
        from src.validate.engines.structural import Verdict

        call_count = {"n": 0}

        def fake_generate(prompt, **kw):
            call_count["n"] += 1
            return json.dumps({"verdict": "ACCEPT", "reason": "all good"})

        with patch("src.validate.engines.llm.generate", side_effect=fake_generate):
            v1 = mod.evaluate(
                "rule_ontoclean_rigidity.txt",
                {"batch_json": json.dumps({"term": "A", "parent": "B"})},
                rule_id="ontoclean_rigidity",
                subject_type="edge_taxonomy",
                subject_id="A-->B",
                model="fake-model",
            )
            v2 = mod.evaluate(
                "rule_ontoclean_rigidity.txt",
                {"batch_json": json.dumps({"term": "A", "parent": "B"})},
                rule_id="ontoclean_rigidity",
                subject_type="edge_taxonomy",
                subject_id="A-->B",
                model="fake-model",
            )

        assert isinstance(v1, Verdict)
        assert v1.verdict == "PASS"
        assert v2.verdict == "PASS"
        assert call_count["n"] == 1, f"Cache miss: {call_count['n']} calls"
        # Cache file written
        files = list(tmpdir.glob("*.json"))
        assert len(files) == 1, f"Expected 1 cache file, got {len(files)}"
        print("  ✓ single evaluate caches and short-circuits")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _restore_cache_default(orig)


def test_evaluate_reject_verdict_mapping():
    orig = os.environ.get("VALIDATION_CACHE_DIR")
    tmpdir = _reset_cache_dir()
    try:
        import src.validate.engines.llm as mod

        def fake_generate(prompt, **kw):
            return json.dumps({"verdict": "REJECT", "reason": "rigidity violation"})

        with patch("src.validate.engines.llm.generate", side_effect=fake_generate):
            v = mod.evaluate(
                "rule_ontoclean_rigidity.txt",
                {"batch_json": "{}"},
                rule_id="ontoclean_rigidity",
                subject_type="edge_taxonomy",
                subject_id="X-->Y",
                model="fake-model",
            )
        assert v.verdict == "FAIL", f"Expected FAIL, got {v.verdict}"
        print("  ✓ REJECT maps to FAIL")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _restore_cache_default(orig)


def test_evaluate_unknown_verdict_abstains():
    orig = os.environ.get("VALIDATION_CACHE_DIR")
    tmpdir = _reset_cache_dir()
    try:
        import src.validate.engines.llm as mod

        with patch(
            "src.validate.engines.llm.generate",
            side_effect=lambda p, **kw: json.dumps({"verdict": "MAYBE", "reason": "unsure"}),
        ):
            v = mod.evaluate(
                "rule_ontoclean_rigidity.txt",
                {"batch_json": "{}"},
                rule_id="ontoclean_rigidity",
                subject_type="edge_taxonomy",
                subject_id="X-->Y",
                model="fake-model",
            )
        assert v.verdict == "ABSTAIN", f"Expected ABSTAIN, got {v.verdict}"
        print("  ✓ unknown verdict maps to ABSTAIN")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _restore_cache_default(orig)


def test_evaluate_parse_error_returns_abstain():
    orig = os.environ.get("VALIDATION_CACHE_DIR")
    tmpdir = _reset_cache_dir()
    try:
        import src.validate.engines.llm as mod

        with patch(
            "src.validate.engines.llm.generate",
            side_effect=lambda p, **kw: "<<not json>>",
        ):
            v = mod.evaluate(
                "rule_ontoclean_rigidity.txt",
                {"batch_json": "{}"},
                rule_id="ontoclean_rigidity",
                subject_type="edge_taxonomy",
                subject_id="X-->Y",
                model="fake-model",
            )
        assert v.verdict == "ABSTAIN"
        # No cache should have been written for the failed parse
        files = list(tmpdir.glob("*.json"))
        assert len(files) == 0, f"Cache should not have been written, got {len(files)} files"
        print("  ✓ parse error returns ABSTAIN and skips cache")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _restore_cache_default(orig)


def test_evaluate_call_failure_returns_abstain():
    orig = os.environ.get("VALIDATION_CACHE_DIR")
    tmpdir = _reset_cache_dir()
    try:
        import src.validate.engines.llm as mod

        def boom(p, **kw):
            raise RuntimeError("network down")

        with patch("src.validate.engines.llm.generate", side_effect=boom):
            v = mod.evaluate(
                "rule_ontoclean_rigidity.txt",
                {"batch_json": "{}"},
                rule_id="ontoclean_rigidity",
                subject_type="edge_taxonomy",
                subject_id="X-->Y",
                model="fake-model",
            )
        assert v.verdict == "ABSTAIN"
        print("  ✓ call failure returns ABSTAIN")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _restore_cache_default(orig)


def test_evaluate_batch_maps_per_subject():
    orig = os.environ.get("VALIDATION_CACHE_DIR")
    tmpdir = _reset_cache_dir()
    try:
        import src.validate.engines.llm as mod

        responses = [
            {"term": "t1", "verdict": "KEEP", "reason": "fine"},
            {"term": "t2", "verdict": "REMOVE", "reason": "measurement"},
            {"term": "t3", "verdict": "MAYBE", "reason": "unsure"},
        ]
        with patch(
            "src.validate.engines.llm.generate",
            side_effect=lambda p, **kw: json.dumps(responses),
        ):
            verdicts = mod.evaluate_batch(
                "filter_measurement_property.txt",
                {"batch_json": json.dumps([{"term": "t1"}, {"term": "t2"}, {"term": "t3"}])},
                rule_id="measurement_property",
                subject_type="term",
                subject_ids=["t1", "t2", "t3"],
                model="fake-model",
            )
        assert len(verdicts) == 3
        assert verdicts[0].verdict == "PASS"
        assert verdicts[1].verdict == "FAIL"
        assert verdicts[2].verdict == "ABSTAIN"
        print("  ✓ batch evaluate maps verdicts and abstains on unknowns")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _restore_cache_default(orig)


def test_evaluate_batch_length_mismatch_abstains_missing():
    orig = os.environ.get("VALIDATION_CACHE_DIR")
    tmpdir = _reset_cache_dir()
    try:
        import src.validate.engines.llm as mod

        with patch(
            "src.validate.engines.llm.generate",
            side_effect=lambda p, **kw: json.dumps(
                [{"term": "t1", "verdict": "KEEP", "reason": "fine"}]
            ),
        ):
            verdicts = mod.evaluate_batch(
                "filter_measurement_property.txt",
                {"batch_json": json.dumps([{"term": "t1"}, {"term": "t2"}])},
                rule_id="measurement_property",
                subject_type="term",
                subject_ids=["t1", "t2"],
                model="fake-model",
            )
        assert verdicts[0].verdict == "PASS"
        assert verdicts[1].verdict == "ABSTAIN"
        print("  ✓ batch evaluate abstains on missing entries")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        _restore_cache_default(orig)


def main() -> int:
    tests = [
        test_evaluate_single_pass_caches_and_short_circuits,
        test_evaluate_reject_verdict_mapping,
        test_evaluate_unknown_verdict_abstains,
        test_evaluate_parse_error_returns_abstain,
        test_evaluate_call_failure_returns_abstain,
        test_evaluate_batch_maps_per_subject,
        test_evaluate_batch_length_mismatch_abstains_missing,
    ]
    print(f"=== test_llm_engine.py — {len(tests)} tests ===")
    for t in tests:
        t()
    print(f"=== ALL {len(tests)} TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
