"""Build the T1 baseline manifest used by regression_t1.py.

Snapshots SHA256, line counts, CSV column sets, 6d action counts, and 7b
structural summary from output/refined/t1/. Run once to (re)create
test/fixtures/t1_baseline.json. The regression test re-runs the deterministic
tail of the pipeline (Steps 6d → 7 → 7b) in a temp dir and compares against
this manifest.
"""
import hashlib
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T1_DIR = os.path.join(ROOT, "output", "refined", "t1")
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "t1_baseline.json")

CSV_FILES = [
    "6c_taxonomy_cleaned.csv",
    "6c_relations_cleaned.csv",
    "6d_taxonomy_reclassified.csv",
    "6d_reclassification_log.csv",
]
TTL_FILE = "7_ontology.ttl"
REPORT_FILE = "7b_verification_report.json"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_meta(path: str) -> dict:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return {
        "sha256": _sha256(path),
        "rows": len(df),
        "columns": list(df.columns),
    }


def build() -> dict:
    if not os.path.isdir(T1_DIR):
        raise SystemExit(f"Baseline dir not found: {T1_DIR}")

    manifest: dict = {"source_dir": "output/refined/t1", "files": {}}

    for name in CSV_FILES:
        path = os.path.join(T1_DIR, name)
        if not os.path.exists(path):
            raise SystemExit(f"Missing baseline CSV: {path}")
        manifest["files"][name] = _csv_meta(path)

    ttl_path = os.path.join(T1_DIR, TTL_FILE)
    manifest["files"][TTL_FILE] = {
        "sha256": _sha256(ttl_path),
        "bytes": os.path.getsize(ttl_path),
    }

    # 6d action counts (regression target)
    log_df = pd.read_csv(os.path.join(T1_DIR, "6d_reclassification_log.csv"), encoding="utf-8-sig")
    manifest["6d_action_counts"] = log_df["Action"].value_counts().to_dict()

    # 7b verification summary (regression target)
    with open(os.path.join(T1_DIR, REPORT_FILE), "r", encoding="utf-8") as f:
        report = json.load(f)
    structure = report["layers"]["structure"]
    manifest["7b_summary"] = {
        "syntax_status": report["layers"]["syntax"]["status"],
        "structure_status": structure["status"],
        "classes": structure["classes"],
        "individuals": structure["individuals"],
        "triples": structure["triples"],
        "upper_iris_referenced": structure["upper_iris_referenced"],
    }
    return manifest


if __name__ == "__main__":
    manifest = build()
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"Wrote {MANIFEST_PATH}")
    print(json.dumps(manifest["6d_action_counts"], indent=2))
    print(json.dumps(manifest["7b_summary"], indent=2))
    sys.exit(0)
