"""Study Configuration Loader — single source of truth for study-level
text (currently the expert-evaluation workbook instructions sheet).

Reads `evaluation_study/config/expert_eval.yaml` (or `STUDY_CONFIG_PATH`)
and exposes typed accessors. Mirrors the `ontology_config.py` pattern
(frozen dataclass + lru_cache singleton).

Env vars:
    STUDY_CONFIG_PATH — optional path to another study configuration
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from evaluation_study.paths import STUDY_CONFIG

_DEFAULT_PATH = STUDY_CONFIG


@dataclass(frozen=True)
class StudyConfig:
    instruction_rows: tuple[tuple[str, str], ...] = field(default_factory=tuple)


def _resolve_path() -> Path:
    override = os.environ.get("STUDY_CONFIG_PATH")
    if override:
        return Path(override)
    return _DEFAULT_PATH


@lru_cache(maxsize=1)
def get_study_config() -> StudyConfig:
    path = _resolve_path()
    if not path.exists():
        raise RuntimeError(
            f"Study config not found at '{path}'. "
            f"Set STUDY_CONFIG_PATH env var or create the file."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    sheet = data.get("instructions_sheet") or {}
    raw_rows = sheet.get("rows") or []
    instruction_rows = tuple(
        (str(row[0]), str(row[1])) for row in raw_rows if len(row) >= 2
    )
    if not instruction_rows:
        raise RuntimeError(
            f"Study config '{path}' has no instructions_sheet.rows entries."
        )
    return StudyConfig(instruction_rows=instruction_rows)
