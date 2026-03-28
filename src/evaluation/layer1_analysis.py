"""
Layer 1 Analysis — Automated cross-condition comparison for the ablation study.

All analyses are fully automated (no expert effort) and run on all terms.
Measures WHETHER conditions produce different outputs and HOW they diverge.

Analyses:
  1. Cross-condition agreement matrix (6 pairwise agreement rates)
  2. Cochran's Q test (omnibus: do conditions differ at all?)
  3. Chi-squared test on category frequency distributions
  4. Category migration analysis (stable / RAG-sensitive / NLD-sensitive / compression-sensitive)
  5. NOT_CLASSIFIED rate comparison
  6. Context_Used subgroup analysis (within Condition A vs B)
"""

import os
import itertools

import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = os.environ.get("ABLATION_OUTPUT_DIR", "output/ablation")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")


def load_merged_results(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(OUTPUT_DIR, "ablation_merged.csv")
    return pd.read_csv(path, encoding="utf-8-sig")


def _pivot_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot merged df to wide format: one row per term, one column per condition."""
    return df.pivot_table(
        index="Term", columns="Condition", values="Category", aggfunc="first"
    ).reset_index()


# ---------------------------------------------------------------------------
# 1. Cross-condition agreement matrix
# ---------------------------------------------------------------------------

def agreement_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise % agreement between conditions."""
    wide = _pivot_categories(df)
    conditions = sorted([c for c in wide.columns if c != "Term"])
    results = []
    for c1, c2 in itertools.combinations(conditions, 2):
        agree = (wide[c1] == wide[c2]).sum()
        total = len(wide)
        results.append({
            "Pair": f"{c1} vs {c2}",
            "Agreement": agree,
            "Total": total,
            "Agreement_Rate": round(agree / total, 4) if total else 0,
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 2. Cochran's Q test
# ---------------------------------------------------------------------------

def cochrans_q_test(df: pd.DataFrame, reference_category: str | None = None) -> dict:
    """
    Cochran's Q test per category (or a specific one).
    Tests whether the proportion of terms assigned to a category differs across conditions.
    """
    wide = _pivot_categories(df)
    conditions = sorted([c for c in wide.columns if c != "Term"])

    if reference_category:
        categories = [reference_category]
    else:
        all_cats = set()
        for c in conditions:
            all_cats.update(wide[c].unique())
        categories = sorted(all_cats)

    results = []
    for cat in categories:
        # Binary matrix: 1 if term assigned to this category, 0 otherwise
        binary = pd.DataFrame()
        binary["Term"] = wide["Term"]
        for c in conditions:
            binary[c] = (wide[c] == cat).astype(int)

        # Cochran's Q requires at least 3 groups
        if len(conditions) < 3:
            continue

        k = len(conditions)
        n = len(binary)
        col_sums = binary[conditions].sum(axis=0)
        row_sums = binary[conditions].sum(axis=1)

        T = col_sums.sum()
        if T == 0 or T == n * k:
            # All same — Q = 0
            results.append({"Category": cat, "Q": 0.0, "p_value": 1.0, "df": k - 1})
            continue

        numerator = (k - 1) * (k * (col_sums ** 2).sum() - T ** 2)
        denominator = k * T - (row_sums ** 2).sum()

        if denominator == 0:
            results.append({"Category": cat, "Q": 0.0, "p_value": 1.0, "df": k - 1})
            continue

        Q = numerator / denominator
        p_value = 1 - stats.chi2.cdf(Q, df=k - 1)
        results.append({"Category": cat, "Q": round(Q, 4), "p_value": round(p_value, 6), "df": k - 1})

    return results


# ---------------------------------------------------------------------------
# 3. Chi-squared on category frequency distributions
# ---------------------------------------------------------------------------

def chi_squared_category_distributions(df: pd.DataFrame) -> dict:
    """Chi-squared test comparing category frequency distributions across conditions."""
    conditions = sorted(df["Condition"].unique())
    # Build contingency table: rows = categories, columns = conditions
    contingency = pd.crosstab(df["Category"], df["Condition"])

    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {
        "chi2": round(chi2, 4),
        "p_value": round(p, 6),
        "dof": dof,
        "contingency_table": contingency,
    }


# ---------------------------------------------------------------------------
# 4. Category migration analysis
# ---------------------------------------------------------------------------

def category_migration(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each term by its sensitivity pattern."""
    wide = _pivot_categories(df)
    conditions = sorted([c for c in wide.columns if c != "Term"])

    rows = []
    for _, row in wide.iterrows():
        term = row["Term"]
        cats = {c: row[c] for c in conditions if c in row.index}

        a_val = cats.get("A", None)
        b_val = cats.get("B", None)
        c_val = cats.get("C", None)
        d_val = cats.get("D", None)

        all_same = len(set(cats.values())) == 1

        if all_same:
            pattern = "Stable"
        elif a_val != b_val and a_val == c_val:
            pattern = "RAG-sensitive"
        elif a_val != c_val and a_val == b_val:
            pattern = "NLD-sensitive"
        elif a_val != d_val and a_val == b_val:
            pattern = "Compression-sensitive"
        else:
            pattern = "Multi-sensitive"

        rows.append({
            "Term": term,
            "Cat_A": a_val,
            "Cat_B": b_val,
            "Cat_C": c_val,
            "Cat_D": d_val,
            "Pattern": pattern,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. NOT_CLASSIFIED rate comparison
# ---------------------------------------------------------------------------

def not_classified_rates(df: pd.DataFrame) -> pd.DataFrame:
    """NOT_CLASSIFIED rate per condition."""
    rows = []
    for cond in sorted(df["Condition"].unique()):
        df_c = df[df["Condition"] == cond]
        total = len(df_c)
        nc = len(df_c[df_c["Category"] == "NOT_CLASSIFIED"])
        rows.append({
            "Condition": cond,
            "Total": total,
            "NOT_CLASSIFIED": nc,
            "Rate": round(nc / total, 4) if total else 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Context_Used subgroup analysis
# ---------------------------------------------------------------------------

def context_used_subgroup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Within Condition A, split by Context_Used.
    Compare agreement with Condition B for each subgroup.
    If Context_Used=True terms diverge more from B, RAG retrieval is the active ingredient.
    """
    wide = _pivot_categories(df)
    conditions = sorted([c for c in wide.columns if c != "Term"])

    if "A" not in conditions or "B" not in conditions:
        return pd.DataFrame()

    # Get Context_Used flag from condition A's data
    df_a = df[df["Condition"] == "A"][["Term", "Context_Used"]].drop_duplicates()
    merged = wide.merge(df_a, on="Term", how="left")

    rows = []
    for ctx_val in [True, False, "True", "False", "Error"]:
        sub = merged[merged["Context_Used"].astype(str) == str(ctx_val)]
        if len(sub) == 0:
            continue
        agree = (sub["A"] == sub["B"]).sum()
        total = len(sub)
        rows.append({
            "Context_Used": ctx_val,
            "N_terms": total,
            "A_B_Agreement": agree,
            "A_B_Agreement_Rate": round(agree / total, 4) if total else 0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run all analyses
# ---------------------------------------------------------------------------

def run_layer1_analysis(merged_path: str | None = None) -> dict:
    """Run all Layer 1 analyses and save results."""
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    df = load_merged_results(merged_path)
    print(f"\nLayer 1 Analysis: {len(df)} rows, conditions: {sorted(df['Condition'].unique())}")

    results = {}

    # 1. Agreement matrix
    print("\n1. Cross-condition agreement matrix:")
    am = agreement_matrix(df)
    am.to_csv(os.path.join(ANALYSIS_DIR, "agreement_matrix.csv"), index=False)
    print(am.to_string(index=False))
    results["agreement_matrix"] = am

    # 2. Cochran's Q
    print("\n2. Cochran's Q test (per category):")
    cq = cochrans_q_test(df)
    cq_df = pd.DataFrame(cq)
    cq_df.to_csv(os.path.join(ANALYSIS_DIR, "cochrans_q.csv"), index=False)
    significant = cq_df[cq_df["p_value"] < 0.05]
    print(f"  {len(significant)}/{len(cq_df)} categories show significant differences (p < 0.05)")
    results["cochrans_q"] = cq_df

    # 3. Chi-squared
    print("\n3. Chi-squared on category distributions:")
    cs = chi_squared_category_distributions(df)
    print(f"  chi2 = {cs['chi2']}, p = {cs['p_value']}, dof = {cs['dof']}")
    cs["contingency_table"].to_csv(os.path.join(ANALYSIS_DIR, "contingency_table.csv"))
    results["chi_squared"] = cs

    # 4. Category migration
    print("\n4. Category migration analysis:")
    cm = category_migration(df)
    cm.to_csv(os.path.join(ANALYSIS_DIR, "category_migration.csv"), index=False)
    pattern_counts = cm["Pattern"].value_counts()
    for pat, count in pattern_counts.items():
        print(f"  {pat}: {count} terms ({count/len(cm)*100:.1f}%)")
    results["migration"] = cm

    # 5. NOT_CLASSIFIED rates
    print("\n5. NOT_CLASSIFIED rates:")
    nc = not_classified_rates(df)
    nc.to_csv(os.path.join(ANALYSIS_DIR, "not_classified_rates.csv"), index=False)
    print(nc.to_string(index=False))
    results["not_classified"] = nc

    # 6. Context_Used subgroup
    print("\n6. Context_Used subgroup analysis (A vs B):")
    cu = context_used_subgroup(df)
    if not cu.empty:
        cu.to_csv(os.path.join(ANALYSIS_DIR, "context_used_subgroup.csv"), index=False)
        print(cu.to_string(index=False))
    else:
        print("  (Skipped — requires conditions A and B)")
    results["context_used"] = cu

    print(f"\nAll Layer 1 results saved to {ANALYSIS_DIR}/")
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_layer1_analysis()
