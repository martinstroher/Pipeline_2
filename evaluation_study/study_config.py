"""Study Configuration Loader — single source of truth for study-level
text (currently the expert-evaluation workbook instructions sheet).

Reads `evaluation_study/config/expert_eval.yaml` (or `STUDY_CONFIG_PATH`)
and exposes typed accessors. Mirrors the `ontology_config.py` pattern
(frozen dataclass + lru_cache singleton).

Env vars:
    STUDY_CONFIG_PATH — optional path to another study configuration
    DISPLAY_TEXT_CONFIG_PATH — optional path to another display registry
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from evaluation_study.paths import DISPLAY_TEXT_CONFIG, STUDY_CONFIG

_DEFAULT_PATH = STUDY_CONFIG


@dataclass(frozen=True)
class StudyConfig:
    instruction_rows: tuple[tuple[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class DisplayEntry:
    display: str
    definition: str = ""
    source: str = ""
    example: str = ""
    counterexample: str = ""


@dataclass(frozen=True)
class DisplayRegistry:
    categories: dict[str, DisplayEntry] = field(default_factory=dict)
    properties: dict[str, str] = field(default_factory=dict)
    defined_class_features: dict[str, str] = field(default_factory=dict)
    ambiguous_terms: tuple[str, ...] = field(default_factory=tuple)
    term_glosses: dict[str, str] = field(default_factory=dict)


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


@lru_cache(maxsize=1)
def get_display_registry() -> DisplayRegistry:
    path = Path(os.environ.get("DISPLAY_TEXT_CONFIG_PATH", DISPLAY_TEXT_CONFIG))
    if not path.exists():
        raise RuntimeError(
            f"Display-text config not found at '{path}'. "
            "Set DISPLAY_TEXT_CONFIG_PATH or create the file."
        )
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    categories = {
        str(key).casefold(): DisplayEntry(
            display=str(value.get("display") or key),
            definition=str(value.get("definition") or ""),
            source=str(value.get("source") or ""),
            example=str(value.get("example") or ""),
            counterexample=str(value.get("counterexample") or ""),
        )
        for key, value in (data.get("categories") or {}).items()
    }
    return DisplayRegistry(
        categories=categories,
        properties={
            str(key).casefold(): str(value)
            for key, value in (data.get("properties") or {}).items()
        },
        defined_class_features={
            str(key).casefold(): str(value)
            for key, value in (data.get("defined_class_features") or {}).items()
        },
        ambiguous_terms=tuple(
            str(value).casefold() for value in (data.get("ambiguous_terms") or [])
        ),
        term_glosses={
            str(key).casefold(): str(value)
            for key, value in (data.get("term_glosses") or {}).items()
        },
    )


def display_label(value: object, namespace: str) -> str:
    """Return a geologist-facing label, with a readable fallback."""
    text = str(value).strip()
    key = text.casefold()
    registry = get_display_registry()
    if namespace == "category" and key in registry.categories:
        return registry.categories[key].display
    if namespace == "property" and key in registry.properties:
        return registry.properties[key]
    if namespace == "defined_class" and key in registry.defined_class_features:
        return registry.defined_class_features[key]
    words = text.replace("_", " ")
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", words).strip()
