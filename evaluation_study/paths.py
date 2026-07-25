"""Stable paths shared by the standalone thesis evaluation study."""

from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = STUDY_ROOT.parent

STUDY_OUTPUT = STUDY_ROOT / "output"
ABLATION_OUTPUT = STUDY_OUTPUT / "ablation"
REHEARSAL_OUTPUT = STUDY_OUTPUT / "ablation_rehearsal"

STUDY_INPUTS = STUDY_ROOT / "inputs"
FROZEN_A_INPUTS = STUDY_INPUTS / "frozen_a"
APPROVED_ONTOLOGY_DIR = STUDY_INPUTS / "approved_run"

FILTERED_TERMS = FROZEN_A_INPUTS / "extract_filtered.csv"
FROZEN_A_NLD = FROZEN_A_INPUTS / "define_nld.csv"
FROZEN_A_CATEGORIES = FROZEN_A_INPUTS / "classify_categories.csv"

PIPELINE_OUTPUT = REPO_ROOT / "output"
APPROVED_ONTOLOGY = PIPELINE_OUTPUT / "final" / "presalt_ontology.ttl"

STUDY_CONFIG = STUDY_ROOT / "config" / "expert_eval.yaml"
STUDY_PROMPTS = STUDY_ROOT / "prompts"
