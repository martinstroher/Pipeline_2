"""
Shared Gemini client — supports both AI Studio (API key) and Vertex AI.

Configuration via environment variables:
  GEMINI_API_KEY       — API key (AI Studio mode, default)
  VERTEX_AI=true       — Enable Vertex AI mode (uses ADC for auth)
  GCP_PROJECT          — GCP project ID (required for Vertex AI)
  GCP_LOCATION         — GCP region (default: us-central1)
"""

import os
from google import genai
from google.genai import types

_client = None


def get_client() -> genai.Client:
    """Return a singleton Gemini client, configured from environment."""
    global _client
    if _client is not None:
        return _client

    use_vertex = os.environ.get("VERTEX_AI", "").lower() in ("true", "1", "yes")

    if use_vertex:
        project = os.environ.get("GCP_PROJECT")
        location = os.environ.get("GCP_LOCATION", "us-central1")
        if not project:
            raise RuntimeError("VERTEX_AI=true but GCP_PROJECT is not set.")
        _client = genai.Client(vertexai=True, project=project, location=location)
        print(f"Gemini client configured (Vertex AI: {project} / {location})")
    else:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        _client = genai.Client(api_key=api_key)
        print("Gemini client configured (AI Studio API key)")

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

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text
