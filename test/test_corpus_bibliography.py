"""Contract checks for the public 82-source bibliography and DOI inventory."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
import re

from src.utils.csv_io import read_csv

ROOT = Path(__file__).resolve().parents[1]
BIBLIOGRAPHY = ROOT / "evaluation_study/corpus_bibliography.csv"
DOI_LIST = ROOT / "evaluation_study/corpus_dois.txt"
SOURCE_SET_SHA256 = "0efca783a21de92ceea9151da43d1eeb8eef0201eed78520afa7e8cc82c5be6e"
DOI_PATTERN = re.compile(r"^10\.[0-9]{4,9}/\S+$")


def test_corpus_bibliography_accounts_for_every_recovered_source():
    frame = read_csv(BIBLIOGRAPHY, dtype=str).fillna("")
    assert list(frame.columns) == [
        "Source_Filename",
        "Title",
        "Authors",
        "Year",
        "Venue",
        "DOI",
        "DOI_URL",
        "Verification_Source_URL",
        "Confidence",
        "Notes",
    ]
    assert len(frame) == 82
    assert frame["Source_Filename"].is_unique
    source_set = "\n".join(sorted(frame["Source_Filename"])).encode()
    assert sha256(source_set).hexdigest() == SOURCE_SET_SHA256
    assert frame["Title"].str.strip().ne("").all()
    assert frame["Authors"].str.strip().ne("").all()
    assert frame["Year"].str.fullmatch(r"\d{4}").all()
    assert frame["Venue"].str.strip().ne("").all()
    assert frame["Verification_Source_URL"].str.startswith(("http://", "https://")).all()
    assert set(frame["Confidence"]) <= {"HIGH", "MEDIUM", "LOW"}
    assert frame["Notes"].str.strip().ne("").all()


def test_confirmed_dois_are_normalized_and_unresolved_sources_are_explicit():
    frame = read_csv(BIBLIOGRAPHY, dtype=str).fillna("")
    confirmed = frame[frame["DOI"] != "NO_CONFIRMED_DOI"]
    unresolved = frame[frame["DOI"] == "NO_CONFIRMED_DOI"]

    assert len(confirmed) == 78
    assert confirmed["DOI"].map(lambda value: bool(DOI_PATTERN.fullmatch(value))).all()
    assert confirmed["DOI"].eq(confirmed["DOI"].str.lower()).all()
    assert confirmed.apply(
        lambda row: row["DOI_URL"] == f"https://doi.org/{row['DOI']}", axis=1
    ).all()
    assert len(set(confirmed["DOI"])) == 76
    duplicates = sorted(
        doi for doi, count in Counter(confirmed["DOI"]).items() if count > 1
    )
    assert duplicates == ["10.1190/int-2018-0004.1", "10.22564/rbgf.v39i3.2110"]

    assert len(unresolved) == 4
    assert unresolved["DOI_URL"].eq("").all()
    assert unresolved["Confidence"].eq("LOW").all()


def test_plain_text_inventory_matches_the_bibliography():
    frame = read_csv(BIBLIOGRAPHY, dtype=str).fillna("")
    expected_dois = list(dict.fromkeys(
        frame.loc[frame["DOI"] != "NO_CONFIRMED_DOI", "DOI"]
    ))
    lines = DOI_LIST.read_text(encoding="utf-8").splitlines()
    listed_dois = [line for line in lines if DOI_PATTERN.fullmatch(line)]
    assert listed_dois == expected_dois
    assert "# Source-document rows: 82" in lines
    assert "# Unique confirmed DOIs: 76" in lines
    assert "# NO CONFIRMED DOI" in lines
