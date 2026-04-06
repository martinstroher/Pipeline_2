"""
Generate 3 synthetic expert workbooks with realistic random responses
for testing the Layer 2 statistical analysis pipeline.

Biases:
  - Relevance: mostly 4-5 (terms are pre-filtered geological concepts)
  - NLD Quality: Condition A slightly better than B (~0.5 Likert point advantage)
  - Category: ~80% Yes, ~12% Partial, ~8% No (skewed realistic)
  - Taxonomy: ~75% Yes, ~15% Partial, ~10% No
  - Inter-rater agreement: moderate (experts agree ~70% of the time)
"""

import os
import random
import shutil
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

SEED = 42
N_EXPERTS = 3
SOURCE = "output/ablation/expert_evaluation.xlsx"
OUTPUT_DIR = "output/ablation/test_experts"


def fill_workbook(source_path: str, expert_id: int, seed: int) -> str:
    """Fill a copy of the expert workbook with synthetic responses."""
    rng = random.Random(seed + expert_id * 1000)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dest = os.path.join(OUTPUT_DIR, f"expert_{expert_id}.xlsx")
    shutil.copy2(source_path, dest)

    wb = load_workbook(dest)

    # --- Term Relevance ---
    ws = wb["Term_Relevance"]
    for row in range(2, ws.max_row + 1):
        term = ws.cell(row=row, column=1).value
        if term:
            # Bias toward 4-5 (these are filtered geological terms)
            ws.cell(row=row, column=2).value = rng.choices(
                [1, 2, 3, 4, 5], weights=[1, 3, 10, 35, 51]
            )[0]

    # --- NLD Quality ---
    ws = wb["NLD_Quality"]
    for row in range(2, ws.max_row + 1):
        term = ws.cell(row=row, column=2).value
        if not term:
            continue

        # Both definitions get reasonable scores, but add noise per expert
        base_q1 = rng.choices([1, 2, 3, 4, 5], weights=[1, 3, 15, 45, 36])[0]
        base_q2 = rng.choices([1, 2, 3, 4, 5], weights=[2, 6, 22, 42, 28])[0]

        ws.cell(row=row, column=5).value = base_q1  # Quality_1
        ws.cell(row=row, column=6).value = base_q2  # Quality_2

        # Preference: slightly favor the higher-scored one
        if base_q1 > base_q2:
            pref = rng.choices(["1", "2", "Tie"], weights=[65, 15, 20])[0]
        elif base_q2 > base_q1:
            pref = rng.choices(["1", "2", "Tie"], weights=[15, 65, 20])[0]
        else:
            pref = rng.choices(["1", "2", "Tie"], weights=[20, 20, 60])[0]
        ws.cell(row=row, column=7).value = pref

    # --- Category Correct ---
    ws = wb["Category_Correct"]
    for row in range(2, ws.max_row + 1):
        term = ws.cell(row=row, column=2).value
        if not term:
            continue
        ws.cell(row=row, column=6).value = rng.choices(
            ["Yes", "No", "Partial"], weights=[78, 8, 14]
        )[0]

    # --- Taxonomy Correct ---
    if "Taxonomy_Correct" in wb.sheetnames:
        ws = wb["Taxonomy_Correct"]
        for row in range(2, ws.max_row + 1):
            term = ws.cell(row=row, column=2).value
            if not term:
                continue
            ws.cell(row=row, column=5).value = rng.choices(
                ["Yes", "No", "Partial"], weights=[73, 10, 17]
            )[0]

    wb.save(dest)
    print(f"  Expert {expert_id} -> {dest}")
    return dest


def main():
    print("Generating synthetic expert workbooks...")
    paths = []
    for i in range(1, N_EXPERTS + 1):
        p = fill_workbook(SOURCE, i, SEED)
        paths.append(p)
    print(f"\nGenerated {len(paths)} workbooks in {OUTPUT_DIR}/")
    print(f"Run analysis with:")
    print(f"  py -m src.evaluation.expert_eval_analyzer {' '.join(paths)} --key output/ablation/blinding_key_42.csv")


if __name__ == "__main__":
    main()
