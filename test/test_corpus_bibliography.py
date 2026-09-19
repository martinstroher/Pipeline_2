"""Contract checks for the public corpus title/author bibliography."""

from pathlib import Path

from src.utils.csv_io import read_csv

ROOT = Path(__file__).resolve().parents[1]
BIBLIOGRAPHY = ROOT / "evaluation_study/corpus_bibliography.csv"


def test_corpus_bibliography_is_a_clean_unique_title_author_list():
    frame = read_csv(BIBLIOGRAPHY, dtype=str).fillna("")
    assert list(frame.columns) == ["Title", "Authors"]
    assert len(frame) == 80
    assert frame["Title"].str.strip().ne("").all()
    assert frame["Authors"].str.strip().ne("").all()
    assert not frame.duplicated(["Title", "Authors"]).any()


def test_corpus_bibliography_contains_no_doi_or_source_file_audit_fields():
    frame = read_csv(BIBLIOGRAPHY, dtype=str).fillna("")
    forbidden = {
        "Source_Filename",
        "DOI",
        "DOI_URL",
        "Verification_Source_URL",
        "Confidence",
        "Notes",
    }
    assert not forbidden & set(frame.columns)
