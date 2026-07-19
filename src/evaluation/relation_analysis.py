"""
Relation Extraction Analysis — Descriptive statistics and precision sample.

Produces two outputs:
  1. relation_descriptive_stats.txt — summary statistics for the thesis
  2. relation_precision_sample.csv — 100 random accepted relations for expert review
     (columns: Term, Property, Filler, Evidence, Correct, Notes)

Usage:
  python pipeline.py --relation-analysis [construct_relations.csv]
  # or directly:
  python -m src.evaluation.relation_analysis output/construct_relations.csv
"""

import os
import sys

import pandas as pd

from src.utils.csv_io import read_csv, write_csv

SEED = 42
SAMPLE_SIZE = 100


def load_relations(path: str) -> pd.DataFrame:
    return read_csv(path)


def descriptive_stats(df: pd.DataFrame) -> str:
    """Compute and format descriptive statistics."""
    total = len(df)
    accepted_mask = df["Validation_Status"].astype(str).str.upper() == "ACCEPTED"
    accepted = df[accepted_mask]
    rejected = df[~accepted_mask]
    n_accepted = len(accepted)
    n_rejected = len(rejected)
    unique_terms_with_relations = accepted["Term"].nunique()
    total_terms = df["Term"].nunique()

    lines = [
        "=" * 60,
        "RELATION EXTRACTION — DESCRIPTIVE STATISTICS",
        "=" * 60,
        "",
        f"Total extracted relations:     {total}",
        f"Accepted (post-validation):    {n_accepted} ({n_accepted/total*100:.1f}%)",
        f"Rejected (post-validation):    {n_rejected} ({n_rejected/total*100:.1f}%)",
        f"Terms with ≥1 accepted rel:    {unique_terms_with_relations} / {total_terms}",
        f"Mean relations per term:        {n_accepted/total_terms:.2f}",
        "",
        "--- Property Distribution (accepted) ---",
    ]

    prop_counts = accepted["Property"].value_counts()
    for prop, count in prop_counts.items():
        lines.append(f"  {prop:<30s} {count:>4d}  ({count/n_accepted*100:.1f}%)")

    lines.append("")
    lines.append("--- Confidence Distribution (accepted) ---")
    if "Confidence" in accepted.columns:
        conf_counts = accepted["Confidence"].value_counts().sort_index(ascending=False)
        for conf, count in conf_counts.items():
            lines.append(f"  {conf:<10}  {count:>4d}  ({count/n_accepted*100:.1f}%)")

    lines.append("")
    lines.append("--- Filler Source (accepted) ---")
    if "Filler_Source" in accepted.columns:
        src_counts = accepted["Filler_Source"].value_counts()
        for src, count in src_counts.items():
            lines.append(f"  {src:<20s} {count:>4d}  ({count/n_accepted*100:.1f}%)")

    lines.append("")
    lines.append("--- Rejection Reasons ---")
    if n_rejected > 0:
        reason_counts = rejected["Validation_Reason"].value_counts()
        for reason, count in reason_counts.items():
            lines.append(f"  {reason:<50s} {count:>4d}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def generate_precision_sample(df: pd.DataFrame, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Sample n accepted relations for expert review."""
    accepted = df[
        df["Validation_Status"].astype(str).str.upper() == "ACCEPTED"
    ].copy()

    if len(accepted) <= n:
        sample = accepted
    else:
        sample = accepted.sample(n=n, random_state=SEED)

    sample = sample.sort_values("Term").reset_index(drop=True)

    review = pd.DataFrame({
        "Row_ID": range(1, len(sample) + 1),
        "Term": sample["Term"].values,
        "Property": sample["Property"].values,
        "Filler": sample["Filler"].values,
        "Evidence": sample["Evidence"].values,
        "Correct": "",        # Expert fills: Yes / No
        "Notes": "",          # Expert fills: optional comment
    })
    return review


def run_relation_analysis(relations_csv: str | None = None):
    """Main entry point."""
    if relations_csv is None:
        relations_csv = os.environ.get(
            "RELATIONS_OUTPUT",
            "output/construct_relations.csv",
        )

    if not os.path.exists(relations_csv):
        print(f"ERROR: {relations_csv} not found. Run Step 6b first.")
        sys.exit(1)

    output_dir = os.path.dirname(relations_csv)
    df = load_relations(relations_csv)

    # 1. Descriptive stats
    stats_text = descriptive_stats(df)
    print(stats_text)

    stats_path = os.path.join(output_dir, "relation_descriptive_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(stats_text)
    print(f"\nSaved: {stats_path}")

    # 2. Precision sample
    sample = generate_precision_sample(df, SAMPLE_SIZE)
    sample_path = os.path.join(output_dir, "relation_precision_sample.csv")
    write_csv(sample, sample_path)
    print(f"Saved: {sample_path}  ({len(sample)} relations for expert review)")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_relation_analysis(csv_path)
