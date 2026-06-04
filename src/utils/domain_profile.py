"""Domain Profile Loader — single source of truth for domain-specific text
(personas, examples, evaluation labels) used by prompt templates.

Reads `domains/presalt/domain_profile.yaml` (or path from env var
`DOMAIN_PROFILE_PATH`) and exposes typed accessors. Mirrors the
`ontology_config.py` pattern (frozen dataclass + lru_cache singleton).

Env vars:
  DOMAIN_PROFILE_PATH — profile file path
                       (default: domains/presalt/domain_profile.yaml)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

_DEFAULT_PATH = Path("domains") / "presalt" / "domain_profile.yaml"


@dataclass(frozen=True)
class DomainProfile:
    name: str
    short_name: str
    personas: dict[str, str] = field(default_factory=dict)

    def persona(self, prompt_key: str) -> str:
        """Return persona text for a given prompt key (filename without .txt).

        Raises KeyError if missing — fail fast so prompts that reference
        `<<persona>>` cannot silently render with an empty string.
        """
        if prompt_key not in self.personas:
            raise KeyError(
                f"Domain profile has no persona for '{prompt_key}'. "
                f"Add it under 'personas:' in the profile YAML."
            )
        return self.personas[prompt_key]


def _resolve_path() -> Path:
    override = os.environ.get("DOMAIN_PROFILE_PATH")
    if override:
        return Path(override)
    return _DEFAULT_PATH


@lru_cache(maxsize=1)
def get_profile() -> DomainProfile:
    path = _resolve_path()
    if not path.exists():
        raise RuntimeError(
            f"Domain profile not found at '{path}'. "
            f"Set DOMAIN_PROFILE_PATH env var or create the file."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return DomainProfile(
        name=data.get("name", ""),
        short_name=data.get("short_name", ""),
        personas=dict(data.get("personas", {})),
    )
