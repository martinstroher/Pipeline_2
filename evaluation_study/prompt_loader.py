"""Load production prompts plus study-only prompt variants.

Production prompts remain owned by the active pipeline domain. The standalone
study adds its own prompt directory without making the production prompt loader
aware of study files.
"""

from src.utils.prompt_loader import (
    _interpolate_blocks,
    _load_blocks_for_active_domain,
    load_prompt as load_production_prompt,
)

from evaluation_study.paths import STUDY_PROMPTS

_SEPARATOR = "[PROMPT_TEMPLATE]"


def load_prompt(filename: str) -> tuple[str, str]:
    """Load a study prompt when present, otherwise load the production prompt."""
    path = STUDY_PROMPTS / filename
    if not path.exists():
        return load_production_prompt(filename)

    content = path.read_text(encoding="utf-8")
    if _SEPARATOR not in content:
        raise ValueError(f"Prompt file {path} missing {_SEPARATOR} separator")
    content = _interpolate_blocks(
        content,
        _load_blocks_for_active_domain(),
        filename,
    )
    system_part, prompt_part = content.split(_SEPARATOR, 1)
    return (
        system_part.replace("[SYSTEM_INSTRUCTION]", "").strip(),
        prompt_part.strip(),
    )
