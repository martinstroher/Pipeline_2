"""Utility to load LLM prompts from per-domain and studies prompt roots.

Prompts are end-to-end artifacts authored per domain — they ship with their
persona, examples, and constraints inline. Runtime data is injected by the
caller via `str.format(**vars)` on the returned template strings; load-time
domain interpolation (`<<persona>>` etc.) was removed in Phase 6.5.

Resolution order: active-domain prompts first, then cross-domain studies
prompts. The active domain is determined by the directory containing
`ontology_config.yaml` (see `src/utils/ontology_config.py`).
"""

from pathlib import Path

from src.utils.ontology_config import get_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STUDIES_PROMPTS_DIR = _REPO_ROOT / "studies" / "prompts"
_SEPARATOR = "[PROMPT_TEMPLATE]"


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


def load_prompt(filename: str) -> tuple[str, str]:
    """Load system instruction and prompt template from a prompt file.

    Returns (system_instruction, prompt_template). Resolves the file across
    the active prompt roots (active domain first, then `studies/prompts/`).
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

    return system_part, prompt_part
