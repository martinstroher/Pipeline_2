"""LLM validation engine — runs prompt-driven rules with a disk cache.

A single entry point, `evaluate(prompt_filename, payload, *, item_id, ...)`,
loads the prompt via `prompt_loader.load_prompt()`, formats the template
with the caller-supplied placeholders, dispatches to
`gemini_client.generate()` with `response_mime_type=application/json`,
parses the response, and returns one or more `Verdict` objects.

Outputs are cached on disk under `.cache/validation/<key>.json`, keyed by
SHA-256 over `(prompt_text, payload_canonical_json, model)`. Re-runs that
hit the cache make zero LLM calls. Cache writes are atomic.

Schema expected from the model:
  * Rule prompts (rigidity/identity/unity/dependence): single object
        {{verdict, child_<m>, parent_<m>, reason}}
    Returned as one Verdict.
  * Filter prompts (per-term batch / per-cluster batch): list of objects
        [{{term, verdict, reason}}, ...]  or
        [{{cluster_id, verdict, surviving_term, merged_terms, reason}}, ...]
    Returned as a list of Verdicts (caller maps subject_ids).

All caller-facing exceptions are wrapped — a parse error or network error
becomes an ABSTAIN verdict so the dispatcher can move on.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.utils import log
from src.utils.gemini_client import generate
from src.utils.prompt_loader import load_prompt

from src.validate.engines.structural import Verdict


_CACHE_DIR = Path(os.environ.get("VALIDATION_CACHE_DIR", ".cache/validation"))


def _cache_key(prompt_text: str, payload_json: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(prompt_text.encode("utf-8"))
    h.update(b"\x1e")
    h.update(payload_json.encode("utf-8"))
    h.update(b"\x1e")
    h.update(model.encode("utf-8"))
    return h.hexdigest()


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.json"


def _cache_read(key: str) -> dict | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_write(key: str, payload: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_path(key)
    # Atomic write — tempfile in same dir + rename
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", dir=str(_CACHE_DIR), suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_name, p)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _canonical_json(obj: Any) -> str:
    """Stable JSON serialisation for cache keying."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def evaluate(
    prompt_filename: str,
    payload: dict[str, Any],
    *,
    rule_id: str,
    subject_type: str,
    subject_id: str,
    model: str | None = None,
    temperature: float | None = None,
    use_cache: bool = True,
) -> Verdict:
    """Run a single LLM rule call and return one Verdict.

    `payload` keys must match the `{placeholder}` names in the prompt template
    (typically `{edge_json}` for OntoClean rules). The function injects them
    via `str.format(**payload)`.

    The model response is expected to be a single JSON object with at least
    a `verdict` field whose value is one of {ACCEPT, REJECT, ABSTAIN}.
    """
    model = model or os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    system_instruction, prompt_template = load_prompt(prompt_filename)
    payload_json = _canonical_json(payload)
    prompt_text = prompt_template.format(**payload)
    key = _cache_key(prompt_text + system_instruction, payload_json, model)

    raw: dict | None = _cache_read(key) if use_cache else None
    cache_hit = raw is not None
    if raw is None:
        try:
            response_text = generate(
                prompt_text,
                model=model,
                system_instruction=system_instruction,
                temperature=temperature,
                response_mime_type="application/json",
            )
        except Exception as e:
            log.warn(f"LLM rule {rule_id} on {subject_id} failed: {e}")
            return Verdict(
                rule_id=rule_id,
                subject_type=subject_type,
                subject_id=subject_id,
                verdict="ABSTAIN",
                reason=f"LLM call failed: {e}",
            )
        try:
            raw = json.loads(response_text)
        except Exception as e:
            log.warn(f"LLM rule {rule_id} on {subject_id} JSON parse failed: {e}")
            return Verdict(
                rule_id=rule_id,
                subject_type=subject_type,
                subject_id=subject_id,
                verdict="ABSTAIN",
                reason=f"Could not parse JSON response: {e}",
                evidence={"raw_response": response_text[:500]},
            )
        if use_cache:
            try:
                _cache_write(key, raw)
            except Exception as e:
                log.warn(f"LLM cache write failed for {key[:12]}…: {e}")

    return _verdict_from_dict(
        raw, rule_id=rule_id, subject_type=subject_type, subject_id=subject_id, cache_hit=cache_hit
    )


def evaluate_batch(
    prompt_filename: str,
    payload: dict[str, Any],
    *,
    rule_id: str,
    subject_type: str,
    subject_ids: list[str],
    model: str | None = None,
    temperature: float | None = None,
    use_cache: bool = True,
) -> list[Verdict]:
    """Run a batched LLM rule (filter prompts) and return one Verdict per subject.

    The model is expected to return a JSON array with one object per input,
    in the same order as `subject_ids`. If the length mismatches, the missing
    entries become ABSTAIN verdicts.
    """
    model = model or os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    system_instruction, prompt_template = load_prompt(prompt_filename)
    payload_json = _canonical_json(payload)
    prompt_text = prompt_template.format(**payload)
    key = _cache_key(prompt_text + system_instruction, payload_json, model)

    raw_list: list[dict] | None = None
    cached = _cache_read(key) if use_cache else None
    if cached is not None and isinstance(cached, list):
        raw_list = cached
    elif cached is not None and isinstance(cached, dict) and "items" in cached:
        raw_list = cached["items"]

    if raw_list is None:
        try:
            response_text = generate(
                prompt_text,
                model=model,
                system_instruction=system_instruction,
                temperature=temperature,
                response_mime_type="application/json",
            )
        except Exception as e:
            log.warn(f"LLM batch rule {rule_id} failed: {e}")
            return [
                Verdict(rule_id, subject_type, sid, "ABSTAIN", f"LLM call failed: {e}")
                for sid in subject_ids
            ]
        try:
            parsed = json.loads(response_text)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected list, got {type(parsed).__name__}")
            raw_list = parsed
        except Exception as e:
            log.warn(f"LLM batch rule {rule_id} JSON parse failed: {e}")
            return [
                Verdict(
                    rule_id,
                    subject_type,
                    sid,
                    "ABSTAIN",
                    f"Could not parse JSON response: {e}",
                    {"raw_response": response_text[:500]},
                )
                for sid in subject_ids
            ]
        if use_cache:
            try:
                _cache_write(key, {"items": raw_list})
            except Exception as e:
                log.warn(f"LLM cache write failed for {key[:12]}…: {e}")

    verdicts: list[Verdict] = []
    for i, sid in enumerate(subject_ids):
        if i >= len(raw_list):
            verdicts.append(
                Verdict(rule_id, subject_type, sid, "ABSTAIN", "Response shorter than batch.")
            )
            continue
        verdicts.append(
            _verdict_from_dict(
                raw_list[i],
                rule_id=rule_id,
                subject_type=subject_type,
                subject_id=sid,
                cache_hit=cached is not None,
            )
        )
    return verdicts


def _verdict_from_dict(
    raw: dict, *, rule_id: str, subject_type: str, subject_id: str, cache_hit: bool
) -> Verdict:
    """Map a raw LLM response dict to a Verdict."""
    if not isinstance(raw, dict):
        return Verdict(
            rule_id, subject_type, subject_id, "ABSTAIN", "Response item not an object.",
            {"raw": raw, "cache_hit": cache_hit},
        )
    label = str(raw.get("verdict", "")).strip().upper()
    # Map prompt-vocabulary → engine-vocabulary
    if label in ("ACCEPT", "KEEP", "KEEP_SEPARATE", "PASS"):
        v = "PASS"
    elif label in ("REJECT", "REMOVE", "MERGE", "FAIL"):
        v = "FAIL"
    else:
        v = "ABSTAIN"
    reason = str(raw.get("reason", "")).strip() or "(no reason given)"
    # Strip control fields, keep the rest as evidence
    evidence = {k: v_ for k, v_ in raw.items() if k not in {"verdict", "reason"}}
    evidence["cache_hit"] = cache_hit
    return Verdict(
        rule_id=rule_id,
        subject_type=subject_type,
        subject_id=subject_id,
        verdict=v,
        reason=reason,
        evidence=evidence,
    )
