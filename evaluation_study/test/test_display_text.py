"""Focused checks for the geologist-facing display registry."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation_study.study_config import display_label, get_display_registry
from evaluation_study.expert_eval_workbook import (
    _defined_class_sentence,
    _relation_statement,
    build_term_glosses,
)
import pandas as pd


def main() -> int:
    registry = get_display_registry()
    assert display_label("quality", "category") == "Dependent property"
    assert display_label("has_disposition", "property") == "has the capacity to"
    assert display_label("GeologicalProcess", "category") == "Geological Process"
    assert display_label("source rock", "defined_class") == (
        "has the capacity to generate hydrocarbons"
    )
    assert "mound" in registry.ambiguous_terms
    assert registry.term_glosses["reservoir"].startswith("A subsurface rock body")
    assert _defined_class_sentence(
        "source rock",
        "Sedimentary Rock",
        "has_disposition",
        "HydrocarbonGenerationDisposition",
    ) == (
        "A source rock is a sedimentary rock that has the capacity to generate "
        "hydrocarbons."
    )
    assert _relation_statement(
        "dolomudstone",
        "constituted_by",
        "calcite",
        "generic",
    ) == "Generally, dolomudstone is constituted by calcite."
    glosses = build_term_glosses(pd.DataFrame([
        {"Term": "mound", "NLD": "Mound is a carbonate buildup. More detail."},
        {"Term": "other", "NLD": "Other is not ambiguous."},
    ]))
    assert glosses["mound"] == "Mound is a carbonate buildup."
    assert "other" not in glosses
    print("=== DISPLAY TEXT TEST PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())