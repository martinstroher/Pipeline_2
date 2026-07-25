"""Load LLM prompts with per-domain block substitution.

Two layers of templating coexist in every prompt file:

* ``<<key>>`` markers are resolved at LOAD time from
  ``<active-domain>/prompt_blocks.yaml`` (this module). Used to inject the
  domain identity, personas, worked examples, and any other text that must
  change when retargeting the pipeline to a new domain.
* ``{placeholder}`` strings are resolved at CALL time by the caller via
  ``str.format(**vars)``. Used for batch payloads, category names, etc.
  ``{{escaped braces}}`` are JSON examples — they survive both passes.

The active domain is the directory containing ``ontology_config.yaml``
(see ``src/utils/ontology_config.py``). ``prompt_blocks.yaml`` is
discovered in the same directory.

Block resolution rules:
* Nested-dict keys are flattened with ``_`` (e.g. ``domain.role`` →
  ``<<domain_role>>``).
* Block values may themselves contain ``<<other_block>>`` markers; these
  are resolved recursively at load time.
* Cycles raise ``RuntimeError``. Missing inner markers raise ``KeyError``
  naming the offending block. Missing outer markers in a prompt raise
  ``KeyError`` naming the block and the prompt filename.
"""

from functools import lru_cache
from pathlib import Path
import re

import yaml

from src.utils.ontology_config import get_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEPARATOR = "[PROMPT_TEMPLATE]"
_BLOCK_MARKER = re.compile(r"<<([a-zA-Z_][a-zA-Z0-9_]*)>>")


def _prompt_roots() -> list[Path]:
    """Active prompt-search roots, in priority order.

    1. <active-domain>/prompts/   — production pipeline prompts (per-domain)
    """
    domain_dir = get_config()._source_path.parent
    return [domain_dir / "prompts"]


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


def _flatten(data: dict, prefix: str = "") -> dict[str, str]:
    """Flatten a nested dict into a single-level dict with `_`-joined keys."""
    out: dict[str, str] = {}
    for key, value in data.items():
        flat_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, flat_key))
        else:
            out[flat_key] = str(value)
    return out


def _resolve_inner_markers(blocks: dict[str, str]) -> dict[str, str]:
    """Resolve ``<<key>>`` markers that appear inside block values.

    Memoised DFS with on-stack cycle detection. Raises ``RuntimeError`` on
    cycles and ``KeyError`` on missing references.
    """
    resolved: dict[str, str] = {}
    in_progress: set[str] = set()

    def resolve(name: str) -> str:
        if name in resolved:
            return resolved[name]
        if name not in blocks:
            raise KeyError(name)
        if name in in_progress:
            raise RuntimeError(f"cycle detected in prompt blocks: {name}")
        in_progress.add(name)
        value = _BLOCK_MARKER.sub(lambda m: resolve(m.group(1)), blocks[name])
        in_progress.discard(name)
        resolved[name] = value
        return value

    for key in blocks:
        resolve(key)
    return resolved


def _interpolate_blocks(text: str, blocks: dict[str, str], filename: str) -> str:
    """Replace ``<<key>>`` markers in ``text`` with values from ``blocks``.

    Runtime ``{placeholders}`` and ``{{escaped braces}}`` are left untouched.
    Raises ``KeyError`` naming both the missing block and the prompt file.
    """
    def sub(match: re.Match) -> str:
        name = match.group(1)
        if name not in blocks:
            raise KeyError(
                f"prompt block '<<{name}>>' (used in {filename}) "
                "not defined in prompt_blocks.yaml"
            )
        return blocks[name]
    return _BLOCK_MARKER.sub(sub, text)


@lru_cache(maxsize=1)
def _load_blocks_for_active_domain() -> dict[str, str]:
    """Load and fully-resolve ``prompt_blocks.yaml`` for the active domain.

    Cached for the process lifetime. Returns an empty dict if the file does
    not exist — prompts that contain no ``<<key>>`` markers still load.
    """
    blocks_path = get_config()._source_path.parent / "prompt_blocks.yaml"
    if not blocks_path.exists():
        return {}
    with open(blocks_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"prompt_blocks.yaml must be a mapping at the root: {blocks_path}"
        )
    return _resolve_inner_markers(_flatten(raw))


def load_prompt(filename: str) -> tuple[str, str]:
    """Load system instruction and prompt template from a prompt file.

    Resolves the file from the active domain prompt root and substitutes
    ``<<key>>`` markers from the
    active domain's ``prompt_blocks.yaml``, and returns the system part
    and template part separately. Runtime ``{placeholders}`` are not
    touched — callers inject those via ``str.format``.
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

    blocks = _load_blocks_for_active_domain()
    content = _interpolate_blocks(content, blocks, filename)

    system_part, prompt_part = content.split(_SEPARATOR, 1)
    system_part = system_part.replace("[SYSTEM_INSTRUCTION]", "").strip()
    prompt_part = prompt_part.strip()

    return system_part, prompt_part
