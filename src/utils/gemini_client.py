"""
Shared Gemini client — supports Vertex AI (API key or ADC) and AI Studio.

Configuration via environment variables:
  GEMINI_API_KEY       — API key
  VERTEX_AI=true       — Route through Vertex AI endpoint
  GCP_PROJECT          — GCP project ID (required for ADC, ignored with API key)
  GCP_LOCATION         — GCP region (required for ADC, ignored with API key)

Auth priority:
  1. VERTEX_AI + API key  → Vertex AI express mode (api_key only, no project/location)
  2. VERTEX_AI + ADC      → Vertex AI with project/location from env or ADC
  3. API key alone         → AI Studio (generativelanguage.googleapis.com)
"""

import os
import time
from google import genai
from google.genai import types
from src.utils import log

_client = None

# Retry configuration
_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", 5))
_RETRY_BASE_DELAY = float(os.environ.get("LLM_RETRY_BASE_DELAY", 2.0))
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def get_client() -> genai.Client:
    """Return a singleton Gemini client, configured from environment."""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("GEMINI_API_KEY")
    use_vertex = os.environ.get("VERTEX_AI", "").lower() in ("true", "1", "yes")

    if use_vertex and api_key:
        # Vertex AI express mode: api_key routes through aiplatform.googleapis.com
        # project/location must NOT be passed (SDK treats them as mutually exclusive)
        _client = genai.Client(vertexai=True, api_key=api_key)
        log.info("Gemini client: Vertex AI + API key")
    elif use_vertex:
        # Vertex AI with ADC
        project = os.environ.get("GCP_PROJECT")
        location = os.environ.get("GCP_LOCATION", "us-central1")
        if not project:
            raise RuntimeError("VERTEX_AI=true but GCP_PROJECT is not set.")
        _client = genai.Client(vertexai=True, project=project, location=location)
        log.info(f"Gemini client: Vertex AI ADC ({project}/{location})")
    elif api_key:
        # AI Studio mode
        _client = genai.Client(api_key=api_key)
        log.info("Gemini client: AI Studio API key")
    else:
        raise RuntimeError(
            "No Gemini auth configured. Set GEMINI_API_KEY or VERTEX_AI=true with ADC."
        )

    return _client


def generate(
    prompt: str,
    *,
    model: str | None = None,
    system_instruction: str | None = None,
    temperature: float | None = None,
    response_mime_type: str | None = None,
) -> str:
    """
    One-shot generate_content wrapper. Returns response.text.
    Reads model/temperature defaults from env if not provided.
    """
    client = get_client()
    model = model or os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    if temperature is None:
        temperature = float(os.environ.get("LLM_GENERATION_TEMPERATURE", 0.0))

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        response_mime_type=response_mime_type,
    )

    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception as e:
            last_error = e
            error_str = str(e)
            # Check if retryable (rate limit or server error)
            is_retryable = any(str(code) in error_str for code in _RETRYABLE_STATUS_CODES)
            if not is_retryable or attempt == _MAX_RETRIES:
                raise
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warn(f"API error (attempt {attempt}/{_MAX_RETRIES}): {error_str}. "
                      f"Retrying in {delay:.0f}s...")
            time.sleep(delay)

    raise last_error  # unreachable, but satisfies type checker
