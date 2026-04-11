"""Utility to load LLM prompts from the prompts/ directory."""

import os

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "prompts")
_SEPARATOR = "[PROMPT_TEMPLATE]"


def load_prompt(filename: str) -> tuple[str, str]:
    """Load system instruction and prompt template from a prompt file.

    Returns (system_instruction, prompt_template).
    """
    path = os.path.join(_PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if _SEPARATOR not in content:
        raise ValueError(f"Prompt file {filename} missing {_SEPARATOR} separator")

    system_part, prompt_part = content.split(_SEPARATOR, 1)
    system_part = system_part.replace("[SYSTEM_INSTRUCTION]", "").strip()
    prompt_part = prompt_part.strip()

    return system_part, prompt_part
