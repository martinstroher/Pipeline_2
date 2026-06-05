"""Utility to load LLM prompts from the prompts/ directory.

Prompts can contain `<<key>>` placeholders that the loader interpolates
from the active domain profile (see `src/utils/domain_profile.py`). This
keeps domain-specific text (personas, examples, evaluation labels) out
of the prompt files so the same pipeline can be retargeted to another
scientific domain by swapping the profile YAML.

Domain placeholders use `<<…>>` (rather than `{…}`) so they do not
collide with the runtime `{var}` placeholders that callers later resolve
via `str.format(...)`.
"""

import os
import re
from pathlib import Path

from src.utils.domain_profile import get_profile
from src.utils.ontology_config import get_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STUDIES_PROMPTS_DIR = _REPO_ROOT / "studies" / "prompts"
_SEPARATOR = "[PROMPT_TEMPLATE]"
_PLACEHOLDER_RE = re.compile(r"<<\s*(\w+)\s*>>")


def _prompt_roots() -> list[Path]:
    """Active prompt-search roots, in priority order.

    1. <active-domain>/prompts/   — production pipeline prompts (per-domain)
    2. studies/prompts/           — cross-domain study prompts (ablation, etc.)
    """
    domain_dir = get_config()._source_path.parent
    return [domain_dir / "prompts", _STUDIES_PROMPTS_DIR]


def prompt_files() -> list[tuple[str, Path]]:
    """Return [(filename, full_path), …] across all prompt roots, dedup'd by name."""
    seen: dict[str, Path] = {}
    for root in _prompt_roots():
        if not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            if entry.is_file() and entry.suffix == ".txt" and entry.name not in seen:
                seen[entry.name] = entry
    return list(seen.items())


def _interpolate(text: str, filename: str) -> str:
    """Replace `<<key>>` placeholders with values from the domain profile.

    Currently supports:
      <<persona>>     → profile.persona(<filename without .txt>)
      <<short_name>>  → profile.short_name
      <<name>>        → profile.name
    """
    if "<<" not in text:
        return text
    profile = get_profile()
    stem = os.path.splitext(filename)[0]

    def repl(match: re.Match) -> str:
        key = match.group(1)
        if key == "persona":
            return profile.persona(stem)
        if key == "short_name":
            return profile.short_name
        if key == "name":
            return profile.name
        raise KeyError(
            f"Prompt '{filename}' references unknown placeholder <<{key}>>. "
            f"Supported: persona, short_name, name."
        )

    return _PLACEHOLDER_RE.sub(repl, text)


def load_prompt(filename: str) -> tuple[str, str]:
    """Load system instruction and prompt template from a prompt file.

    Returns (system_instruction, prompt_template) with domain placeholders
    already interpolated. Resolves the file across the active prompt roots
    (active domain first, then `studies/prompts/`).
    """
    path: Path | None = None
    for root in _prompt_roots():
        candidate = root / filename
        if candidate.exists():
            path = candidate
            break
    if path is None:
        roots = [str(r) for r in _prompt_roots()]
        raise FileNotFoundError(
            f"Prompt '{filename}' not found in any prompt root: {roots}"
        )
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if _SEPARATOR not in content:
        raise ValueError(f"Prompt file {filename} missing {_SEPARATOR} separator")

    system_part, prompt_part = content.split(_SEPARATOR, 1)
    system_part = system_part.replace("[SYSTEM_INSTRUCTION]", "").strip()
    prompt_part = prompt_part.strip()

    system_part = _interpolate(system_part, filename)
    prompt_part = _interpolate(prompt_part, filename)

    return system_part, prompt_part
