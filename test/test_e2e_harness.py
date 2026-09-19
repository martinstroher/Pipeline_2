"""Offline regression tests for the live end-to-end harness validators."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))

from run_e2e_test import validate_csv_artifacts  # noqa: E402


def test_csv_artifact_validation_returns_one_combined_boolean(tmp_path: Path):
    valid = tmp_path / "valid.csv"
    valid.write_text("Term,Category\nalpha,material entity\n", encoding="utf-8")
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("Term\nbeta\n", encoding="utf-8")

    assert validate_csv_artifacts({
        "valid": (str(valid), ["Term", "Category"], 1),
    }) is True
    assert validate_csv_artifacts({
        "valid": (str(valid), ["Term", "Category"], 1),
        "invalid": (str(invalid), ["Term", "Category"], 1),
    }) is False
