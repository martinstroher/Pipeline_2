"""
Expert Evaluation Spreadsheet Generator (Layer 2).

Generates a blinded, randomized spreadsheet for expert evaluation:
  - 50-80 terms (stratified by category and Context_Used flag)
  - For each term: NLD-A vs NLD-B (random order), Category-A vs Category-B (random order)
  - Expert tasks: NLD preference, NLD quality (1-5), category correctness

Output: Excel file ready to send to domain experts.
"""

import os
import random

import pandas as pd

OUTPUT_DIR = os.environ.get("ABLATION_OUTPUT_DIR", "output/ablation")


def generate_expert_spreadsheet(
    n_terms: int = 60,
    seed: int = 42,
    output_path: str | None = None,
) -> str:
    """
    Generate a blinded expert evaluation spreadsheet.

    Args:
        n_terms: Target number of terms to sample.
        seed: Random seed for reproducibility.
        output_path: Output Excel path (default: output/ablation/expert_evaluation.xlsx).

    Returns:
        Path to the generated file.
    """
    random.seed(seed)

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, "expert_evaluation.xlsx")

    # Load condition A and B NLD outputs
    nld_a = pd.read_csv(os.path.join(OUTPUT_DIR, "nld_A.csv"), encoding="utf-8-sig")
    nld_b = pd.read_csv(os.path.join(OUTPUT_DIR, "nld_B.csv"), encoding="utf-8-sig")
    cat_a = pd.read_csv(os.path.join(OUTPUT_DIR, "cat_A.csv"), encoding="utf-8-sig")
    cat_b = pd.read_csv(os.path.join(OUTPUT_DIR, "cat_B.csv"), encoding="utf-8-sig")

    # Merge A and B data per term
    merged = nld_a[["Term", "NLD", "Context_Used"]].rename(
        columns={"NLD": "NLD_A", "Context_Used": "Context_Used_A"}
    )
    merged = merged.merge(
        nld_b[["Term", "NLD"]].rename(columns={"NLD": "NLD_B"}),
        on="Term",
        how="inner",
    )
    merged = merged.merge(
        cat_a[["Term", "Category"]].rename(columns={"Category": "Cat_A"}),
        on="Term",
        how="left",
    )
    merged = merged.merge(
        cat_b[["Term", "Category"]].rename(columns={"Category": "Cat_B"}),
        on="Term",
        how="left",
    )

    # Filter out error rows
    merged = merged[~merged["NLD_A"].str.startswith("ERROR", na=False)]
    merged = merged[~merged["NLD_B"].str.startswith("ERROR", na=False)]

    # Stratified sampling: balance by category and Context_Used
    merged["Context_Used_A"] = merged["Context_Used_A"].astype(str)
    strata_col = merged["Cat_A"].fillna("UNKNOWN") + "_" + merged["Context_Used_A"]
    merged["strata"] = strata_col

    # Sample proportionally from each stratum
    sampled = merged.groupby("strata", group_keys=False).apply(
        lambda x: x.sample(
            n=max(1, round(len(x) / len(merged) * n_terms)),
            random_state=seed,
        )
    )
    # Trim or pad to target
    if len(sampled) > n_terms:
        sampled = sampled.sample(n=n_terms, random_state=seed)

    # Randomize order (blind: which is A vs B)
    rows = []
    for _, r in sampled.iterrows():
        # Random assignment: Definition_1 / Definition_2
        if random.random() < 0.5:
            d1, d2 = r["NLD_A"], r["NLD_B"]
            c1, c2 = r["Cat_A"], r["Cat_B"]
            order = "A_first"
        else:
            d1, d2 = r["NLD_B"], r["NLD_A"]
            c1, c2 = r["Cat_B"], r["Cat_A"]
            order = "B_first"

        rows.append({
            "Term": r["Term"],
            "Definition_1": d1,
            "Definition_2": d2,
            "Category_1": c1,
            "Category_2": c2,
            # Hidden blinding key (in separate sheet)
            "_order": order,
            "_Context_Used": r["Context_Used_A"],
            "_Cat_A": r["Cat_A"],
        })

    eval_df = pd.DataFrame(rows)

    # Expert sheet (what they see)
    expert_sheet = eval_df[["Term", "Definition_1", "Definition_2", "Category_1", "Category_2"]].copy()
    expert_sheet["NLD_Preference (1/2/Tie)"] = ""
    expert_sheet["NLD_1_Quality (1-5)"] = ""
    expert_sheet["NLD_2_Quality (1-5)"] = ""
    expert_sheet["Category_Correct (1/2/Both/Neither)"] = ""
    expert_sheet["Notes"] = ""

    # Blinding key (hidden from experts, for analysis)
    key_sheet = eval_df[["Term", "_order", "_Context_Used", "_Cat_A"]]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        expert_sheet.to_excel(writer, sheet_name="Evaluation", index=False)
        key_sheet.to_excel(writer, sheet_name="Blinding_Key", index=False)

    print(f"Expert evaluation spreadsheet: {output_path}")
    print(f"  {len(expert_sheet)} terms sampled (stratified by category + Context_Used)")
    print(f"  Sheets: 'Evaluation' (send to experts), 'Blinding_Key' (keep for analysis)")
    return output_path


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    generate_expert_spreadsheet()
