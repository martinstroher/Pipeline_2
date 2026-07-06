"""
Shared LLM client — Azure AI Foundry (Azure OpenAI, GPT-5.x) via the OpenAI SDK v1 API.

Configuration via environment variables:
  AZURE_OPENAI_API_KEY   — API key for the Foundry resource
  AZURE_OPENAI_ENDPOINT  — resource host, e.g. https://<resource>.openai.azure.com
                           (the client appends /openai/v1/)
  LLM_GENERATION_MODEL   — Azure *deployment* name (e.g. gpt-5.4), NOT the bare model id
  LLM_REASONING_EFFORT   — none|minimal|low|medium|high  (default: high)
  LLM_MAX_OUTPUT_TOKENS  — max_completion_tokens ceiling (default: 32000; covers reasoning + visible)
  LLM_SEED               — best-effort determinism seed (default: 42)

Notes on GPT-5.x reasoning models:
  - `temperature` is NOT supported and is IGNORED (accepted only for backward compatibility).
  - `max_completion_tokens` bounds reasoning + visible tokens; if too low the visible
    content can come back empty (finish_reason == "length"). Keep it generous.
  - Determinism is best-effort (pinned deployment version + seed); not bit-reproducible.
  - JSON: we do NOT force ``response_format={"type": "json_object"}``. That mode makes
    the model collapse "array of objects" prompts into a single object (dropping items).
    Instead we rely on the prompts (which all instruct "return valid JSON") and parse
    tolerantly — stripping any markdown code fences and, for array call sites, unwrapping
    a single wrapped array via :func:`parse_json_array`.
"""

import json
import os
import threading
import time

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from src.utils import log

_client = None

# Retry configuration
_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", 5))
_RETRY_BASE_DELAY = float(os.environ.get("LLM_RETRY_BASE_DELAY", 2.0))
_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}

# Per-call usage logging (token counts incl. reasoning tokens) for cost tracking.
_USAGE_LOG = os.environ.get("LLM_USAGE_LOG", "output/usage_log.csv")
_usage_lock = threading.Lock()


def get_client() -> OpenAI:
    """Return a singleton OpenAI client pointed at the Azure Foundry v1 endpoint."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set.")
    if not endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT is not set "
            "(e.g. https://<resource>.openai.azure.com)."
        )

    base_url = f"{endpoint.rstrip('/')}/openai/v1/"
    _client = OpenAI(api_key=api_key, base_url=base_url)
    log.info(f"LLM client: Azure OpenAI (Foundry) v1 @ {endpoint.rstrip('/')}")
    return _client


def _log_usage(model: str, usage, call: str = "generate") -> None:
    """Append token usage (prompt / reasoning / completion) to the usage log."""
    if usage is None:
        return
    try:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None) if details else None
        os.makedirs(os.path.dirname(_USAGE_LOG) or ".", exist_ok=True)
        is_new = not os.path.exists(_USAGE_LOG)
        with _usage_lock:
            with open(_USAGE_LOG, "a", encoding="utf-8-sig") as fh:
                if is_new:
                    fh.write("timestamp,call,model,prompt_tokens,reasoning_tokens,completion_tokens\n")
                fh.write(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{call},{model},"
                    f"{prompt},{reasoning},{completion}\n"
                )
    except Exception as e:  # never let logging break a run
        log.warn(f"Usage log write failed: {e}")


def generate(
    prompt: str,
    *,
    model: str | None = None,
    system_instruction: str | None = None,
    temperature: float | None = None,
    response_mime_type: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """One-shot chat completion via Azure OpenAI. Returns the message content string.

    Preserves the previous Gemini wrapper's signature so call sites are unchanged.
    `temperature` is accepted for backward compatibility but IGNORED — GPT-5.x
    reasoning models reject it. Reads model/effort/limits from env when not given.
    """
    client = get_client()
    model = model or os.environ.get("LLM_GENERATION_MODEL", "gpt-5.4")
    effort = reasoning_effort or os.environ.get("LLM_REASONING_EFFORT", "high")
    max_out = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", 32000))
    seed = int(os.environ.get("LLM_SEED", 42))

    messages: list[dict] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_out,
        "reasoning_effort": effort,
        "seed": seed,
    }
    # NOTE: we intentionally do NOT set response_format={"type": "json_object"}.
    # On gpt-5.x that mode collapses "array of objects" prompts into a single
    # object (silently dropping list items). The prompts already instruct the
    # model to return valid JSON; callers parse tolerantly (fence-stripped here,
    # array-unwrapped via parse_json_array).
    want_json = response_mime_type == "application/json"

    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            _log_usage(model, getattr(resp, "usage", None))
            choice = resp.choices[0]
            content = choice.message.content
            if not content or getattr(choice, "finish_reason", None) == "length":
                log.warn(
                    f"Empty/truncated completion (finish_reason="
                    f"{getattr(choice, 'finish_reason', None)}). Raise "
                    f"LLM_MAX_OUTPUT_TOKENS (current {max_out}) or lower reasoning_effort."
                )
            content = content or ""
            return _strip_code_fences(content) if want_json else content
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last_error = e
            if attempt == _MAX_RETRIES:
                raise
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warn(
                f"API error (attempt {attempt}/{_MAX_RETRIES}): {e}. "
                f"Retrying in {delay:.0f}s..."
            )
            time.sleep(delay)
        except APIError as e:
            last_error = e
            status = getattr(e, "status_code", None)
            if status in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                log.warn(
                    f"API error {status} (attempt {attempt}/{_MAX_RETRIES}). "
                    f"Retrying in {delay:.0f}s..."
                )
                time.sleep(delay)
                continue
            raise

    raise last_error  # unreachable, but satisfies type checker


# Common keys models use when they wrap an array inside an object.
_ARRAY_WRAPPER_KEYS = (
    "result", "results", "items", "data", "output", "outputs",
    "terms", "entities", "list", "array", "values", "response",
)


def _strip_code_fences(text: str) -> str:
    """Strip a surrounding markdown code fence (```json ... ``` or ``` ... ```).

    Reasoning models usually return bare JSON, but occasionally wrap it in a
    fenced block; this makes downstream ``json.loads`` robust to that.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    newline = s.find("\n")
    s = s[newline + 1:] if newline != -1 else s[3:]
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3]
    return s.strip()


def parse_json_array(text: str) -> list:
    """Parse an LLM JSON response that is logically an array.

    Azure OpenAI's ``response_format={"type": "json_object"}`` forces a JSON
    *object* at the top level, so prompts that ask for a bare array come back
    wrapped, e.g. ``{"result": [...]}`` (the wrapper key is non-deterministic).
    This unwraps that single list so array-consuming call sites stay simple.

    Raises ``ValueError`` if no array can be recovered.
    """
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in _ARRAY_WRAPPER_KEYS:
            if isinstance(data.get(key), list):
                return data[key]
        list_values = [v for v in data.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return list_values[0]
    raise ValueError(
        f"Expected a JSON array (or an object wrapping exactly one array); "
        f"got {type(data).__name__}."
    )
