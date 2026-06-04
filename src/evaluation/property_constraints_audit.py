"""
Property Constraints Audit — emits a CSV of every relation declared in
ontology_config.yaml with its provenance, domain, range, inverse, and IRI.

Usage:
    python -m src.evaluation.property_constraints_audit
    python -m src.evaluation.property_constraints_audit --output custom_path.csv

The audit is a read-only summary; it does NOT modify the YAML or the loader.
Active vs. inactive status is computed against `provenance_tiers_active` (the
default YAML value — env override via RELATION_PROVENANCE_TIERS is honoured).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from src.utils.csv_io import write_csv

from src.utils.ontology_config import get_config


def _build_rows() -> list[dict]:
    cfg = get_config()
    active_tiers = cfg.active_provenance_tiers()
    rows = []
    for name in sorted(cfg.all_relations().keys()):
        pc = cfg.all_relations()[name]
        rows.append({
            "Relation": name,
            "IRI": pc.iri,
            "Provenance": pc.provenance,
            "Active": pc.provenance in active_tiers,
            "Inverse": pc.inverse or "",
            "Domain_Metatypes": "|".join(sorted(pc.domain)),
            "Range_Metatypes": "|".join(sorted(pc.range)),
            "Notes": pc.notes,
        })
    return rows


def main(output_path: str = "output/property_constraints_audit.csv") -> int:
    rows = _build_rows()
    df = pd.DataFrame(rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(df, out)

    cfg = get_config()
    active_tiers = sorted(cfg.active_provenance_tiers())
    print(f"Wrote {len(df)} relation entries to {out}")
    print(f"Active provenance tiers: {active_tiers}")
    print("Provenance distribution:")
    for tier, count in df["Provenance"].value_counts().items():
        active_marker = "(active)" if tier in active_tiers else "(inactive)"
        print(f"  {tier}: {count} {active_marker}")
    active_count = int(df["Active"].sum())
    print(f"Total active: {active_count} / {len(df)}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="output/property_constraints_audit.csv",
        help="Output CSV path (default: output/property_constraints_audit.csv)",
    )
    args = parser.parse_args()
    raise SystemExit(main(args.output))
