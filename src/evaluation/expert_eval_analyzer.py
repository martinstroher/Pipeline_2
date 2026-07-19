"""Public entry point for modular Layer 2 expert analysis.

The executable path delegates to ``src.evaluation.expert_eval_analysis``.
Legacy analysis helpers remain available for historical workbook compatibility.

Statistical tests:
  Term Relevance:    Descriptive stats, ICC (inter-annotator agreement)
  NLD Quality:       Wilcoxon signed-rank (A vs B), sign test on preference, weighted kappa
  Category Correct:  Friedman test (4 conditions), post-hoc Wilcoxon, Fleiss' kappa
  Cross-layer:       Spearman correlations (NLD quality <-> category correctness)
"""

import os
import json

import numpy as np
import pandas as pd

from src.utils.csv_io import read_csv
from scipy import stats
from sklearn.metrics import cohen_kappa_score

OUTPUT_DIR = os.environ.get("ABLATION_OUTPUT_DIR", "output/ablation")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")


# ---------------------------------------------------------------------------
# Data Loading & Unblinding
# ---------------------------------------------------------------------------

def load_expert_responses(workbook_paths: list[str]) -> dict:
    """Load expert workbooks into a structured dict.

    Returns:
        {
            "expert_1": {
                "relevance": DataFrame,
                "nld_quality": DataFrame,
                "category": DataFrame,
            },
            ...
        }
    """
    experts = {}
    for i, path in enumerate(workbook_paths):
        expert_id = f"expert_{i + 1}"
        xl = pd.ExcelFile(path, engine="openpyxl")
        expert_data = {
            "relevance": pd.read_excel(xl, sheet_name="Term_Relevance"),
            "nld_quality": pd.read_excel(xl, sheet_name="NLD_Quality"),
            "category": pd.read_excel(xl, sheet_name="Category_Correct"),
        }
        if "Taxonomy_Correct" in xl.sheet_names:
            expert_data["taxonomy"] = pd.read_excel(xl, sheet_name="Taxonomy_Correct")
        experts[expert_id] = expert_data
    return experts


def unblind_nld(nld_df: pd.DataFrame, key_df: pd.DataFrame) -> pd.DataFrame:
    """Map blinded NLD quality ratings back to conditions A and B.

    Returns DataFrame with columns: Term, Quality_A, Quality_B, Preference_A
    """
    nld_key = key_df[key_df["Sheet"] == "NLD_Quality"].set_index("Row_ID")

    rows = []
    for _, row in nld_df.iterrows():
        row_id = row["Row_ID"]
        if row_id not in nld_key.index:
            continue

        key = nld_key.loc[row_id]
        order = key["Order"]

        q1 = row.get("Quality_1 (1-5)", np.nan)
        q2 = row.get("Quality_2 (1-5)", np.nan)
        pref = row.get("Preference (1/2/Tie)", "")

        if order == "A_first":
            qa, qb = q1, q2
            pref_a = 1 if str(pref) == "1" else (-1 if str(pref) == "2" else 0)
        else:
            qa, qb = q2, q1
            pref_a = 1 if str(pref) == "2" else (-1 if str(pref) == "1" else 0)

        rows.append({
            "Term": key["Term"],
            "Quality_A": qa,
            "Quality_B": qb,
            "Preference_A": pref_a,  # 1=prefers A, -1=prefers B, 0=tie
            "Context_Used_A": key.get("Context_Used_A", ""),
        })

    return pd.DataFrame(rows)


def unblind_category(
    cat_df: pd.DataFrame, key_df: pd.DataFrame
) -> pd.DataFrame:
    """Map deduplicated category judgments back to per-condition correctness.

    Returns DataFrame with columns: Term, Condition, Category, Correct_Score
    """
    cat_key = key_df[key_df["Sheet"] == "Category_Correct"].set_index("Row_ID")

    rows = []
    for _, row in cat_df.iterrows():
        row_id = row["Row_ID"]
        if row_id not in cat_key.index:
            continue

        key = cat_key.loc[row_id]
        correct_raw = str(row.get("Correct (Yes/No/Partial)", "")).strip()

        # Map to numeric score
        if correct_raw.lower() == "yes":
            score = 1.0
        elif correct_raw.lower() == "partial":
            score = 0.5
        else:
            score = 0.0

        # Propagate to all conditions that share this (term, category) pair
        conditions = str(key["Conditions"]).split(",")
        for cond in conditions:
            rows.append({
                "Term": key["Term"],
                "Condition": cond.strip(),
                "Category": key["Assigned_Category"],
                "Correct_Score": score,
                "Correct_Raw": correct_raw,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Term Relevance Analysis
# ---------------------------------------------------------------------------

def analyze_term_relevance(experts: dict) -> dict:
    """Descriptive stats and inter-annotator agreement for term relevance."""
    results = {}

    # Collect all ratings into a matrix (terms x experts)
    all_terms = None
    expert_ratings = {}

    for expert_id, data in experts.items():
        df = data["relevance"]
        ratings = df.set_index("Term")["Relevance (1-5)"].dropna()
        expert_ratings[expert_id] = ratings
        if all_terms is None:
            all_terms = set(ratings.index)
        else:
            all_terms &= set(ratings.index)

    common_terms = sorted(all_terms) if all_terms else []
    n_experts = len(expert_ratings)

    if not common_terms:
        return {"error": "No common rated terms found across experts"}

    # Build matrix
    matrix = np.array([
        [expert_ratings[eid][term] for eid in sorted(expert_ratings)]
        for term in common_terms
    ])

    # Descriptive stats (pooled across experts)
    all_scores = matrix.flatten()
    results["descriptive"] = {
        "n_terms": len(common_terms),
        "n_experts": n_experts,
        "mean": round(float(np.mean(all_scores)), 3),
        "median": float(np.median(all_scores)),
        "std": round(float(np.std(all_scores)), 3),
        "min": float(np.min(all_scores)),
        "max": float(np.max(all_scores)),
    }

    # Distribution
    for val in range(1, 6):
        results["descriptive"][f"count_{val}"] = int(np.sum(all_scores == val))

    # ICC (two-way random, absolute agreement) via ANOVA decomposition
    if n_experts >= 2:
        results["icc"] = _compute_icc(matrix)

    return results


def _compute_icc(matrix: np.ndarray) -> dict:
    """Compute ICC(2,1) — two-way random, single measures, absolute agreement."""
    n, k = matrix.shape  # n subjects, k raters
    grand_mean = matrix.mean()

    # Sum of squares
    ss_total = np.sum((matrix - grand_mean) ** 2)
    ss_rows = k * np.sum((matrix.mean(axis=1) - grand_mean) ** 2)
    ss_cols = n * np.sum((matrix.mean(axis=0) - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    # Mean squares
    ms_rows = ss_rows / (n - 1) if n > 1 else 0
    ms_cols = ss_cols / (k - 1) if k > 1 else 0
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0

    # ICC(2,1)
    denom = ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error)
    icc_val = (ms_rows - ms_error) / denom if denom != 0 else 0

    return {
        "icc_2_1": round(float(icc_val), 4),
        "interpretation": (
            "excellent" if icc_val > 0.9 else
            "good" if icc_val > 0.75 else
            "moderate" if icc_val > 0.5 else
            "poor"
        ),
    }


# ---------------------------------------------------------------------------
# NLD Quality Analysis (A vs B)
# ---------------------------------------------------------------------------

def analyze_nld_quality(experts: dict, key_df: pd.DataFrame) -> dict:
    """Wilcoxon signed-rank, sign test, weighted kappa for NLD quality."""
    results = {}

    # Unblind all experts
    all_unblinded = []
    for expert_id, data in experts.items():
        unblinded = unblind_nld(data["nld_quality"], key_df)
        unblinded["Expert"] = expert_id
        all_unblinded.append(unblinded)

    if not all_unblinded:
        return {"error": "No NLD quality data found"}

    merged = pd.concat(all_unblinded, ignore_index=True)
    merged = merged.dropna(subset=["Quality_A", "Quality_B"])

    qa = merged["Quality_A"].values.astype(float)
    qb = merged["Quality_B"].values.astype(float)

    # Wilcoxon signed-rank test
    diff = qa - qb
    nonzero = diff[diff != 0]
    if len(nonzero) > 0:
        w_stat, w_p = stats.wilcoxon(nonzero)
        z_approx = (w_stat - len(nonzero) * (len(nonzero) + 1) / 4) / \
                   np.sqrt(len(nonzero) * (len(nonzero) + 1) * (2 * len(nonzero) + 1) / 24)
        effect_r = abs(z_approx) / np.sqrt(len(nonzero))
    else:
        w_stat, w_p, effect_r = 0, 1.0, 0.0

    results["wilcoxon"] = {
        "W": float(w_stat),
        "p_value": round(float(w_p), 6),
        "effect_size_r": round(float(effect_r), 4),
        "n_pairs": len(nonzero),
        "mean_diff_A_minus_B": round(float(np.mean(diff)), 4),
    }

    # Sign test on preference
    pref = merged["Preference_A"].values
    n_prefer_a = int(np.sum(pref == 1))
    n_prefer_b = int(np.sum(pref == -1))
    n_tie = int(np.sum(pref == 0))
    n_decisive = n_prefer_a + n_prefer_b

    if n_decisive > 0:
        sign_result = stats.binomtest(n_prefer_a, n_decisive, p=0.5)
        sign_p = sign_result.pvalue
    else:
        sign_p = 1.0

    results["sign_test"] = {
        "prefer_A": n_prefer_a,
        "prefer_B": n_prefer_b,
        "tie": n_tie,
        "p_value": round(float(sign_p), 6),
    }

    # Inter-annotator agreement (weighted kappa) — pairwise
    expert_ids = sorted(experts.keys())
    if len(expert_ids) >= 2:
        kappas = []
        for i in range(len(expert_ids)):
            for j in range(i + 1, len(expert_ids)):
                e1_data = merged[merged["Expert"] == expert_ids[i]]
                e2_data = merged[merged["Expert"] == expert_ids[j]]
                # Align on terms
                common = pd.merge(
                    e1_data[["Term", "Quality_A"]].rename(columns={"Quality_A": "r1"}),
                    e2_data[["Term", "Quality_A"]].rename(columns={"Quality_A": "r2"}),
                    on="Term",
                )
                if len(common) > 0:
                    k = cohen_kappa_score(
                        common["r1"].astype(int),
                        common["r2"].astype(int),
                        weights="quadratic",
                    )
                    kappas.append({
                        "pair": f"{expert_ids[i]} vs {expert_ids[j]}",
                        "kappa": round(float(k), 4),
                    })
        results["inter_annotator_kappa"] = kappas

    # Subgroup analysis by Context_Used
    for ctx_val in [True, False, "True", "False"]:
        sub = merged[merged["Context_Used_A"].astype(str) == str(ctx_val)]
        if len(sub) > 5:
            sub_diff = sub["Quality_A"].values - sub["Quality_B"].values
            sub_nonzero = sub_diff[sub_diff != 0]
            if len(sub_nonzero) > 0:
                _, sub_p = stats.wilcoxon(sub_nonzero)
                results[f"subgroup_context_{ctx_val}"] = {
                    "n": len(sub),
                    "mean_diff": round(float(np.mean(sub_diff)), 4),
                    "p_value": round(float(sub_p), 6),
                }

    return results


# ---------------------------------------------------------------------------
# Category Correctness Analysis (4 conditions)
# ---------------------------------------------------------------------------

def analyze_category_correctness(experts: dict, key_df: pd.DataFrame) -> dict:
    """Friedman test, post-hoc Wilcoxon, Fleiss' kappa for category correctness."""
    results = {}

    # Unblind all experts
    all_unblinded = []
    for expert_id, data in experts.items():
        unblinded = unblind_category(data["category"], key_df)
        unblinded["Expert"] = expert_id
        all_unblinded.append(unblinded)

    if not all_unblinded:
        return {"error": "No category correctness data found"}

    merged = pd.concat(all_unblinded, ignore_index=True)

    # Average across experts per (Term, Condition)
    avg_scores = merged.groupby(["Term", "Condition"])["Correct_Score"].mean().reset_index()

    # Pivot to wide: one row per term, one column per condition
    wide = avg_scores.pivot_table(
        index="Term", columns="Condition", values="Correct_Score"
    ).dropna()

    conditions = sorted([c for c in wide.columns])

    # Proportion correct per condition
    prop_correct = {}
    for cond in conditions:
        scores = wide[cond].values
        prop_correct[cond] = {
            "mean_score": round(float(np.mean(scores)), 4),
            "proportion_correct": round(float(np.mean(scores >= 0.9)), 4),  # Yes=1.0
            "proportion_partial_or_correct": round(float(np.mean(scores >= 0.4)), 4),
            "n": len(scores),
        }
    results["proportion_correct"] = prop_correct

    # Friedman test (requires all 4 conditions)
    if len(conditions) >= 3 and len(wide) >= 5:
        cols = [wide[c].values for c in conditions]
        f_stat, f_p = stats.friedmanchisquare(*cols)
        # Kendall's W effect size
        n = len(wide)
        k = len(conditions)
        w = f_stat / (n * (k - 1))
        results["friedman"] = {
            "chi2": round(float(f_stat), 4),
            "p_value": round(float(f_p), 6),
            "kendalls_w": round(float(w), 4),
            "df": k - 1,
        }

    # Post-hoc Wilcoxon pairwise — only if Friedman omnibus is significant
    posthoc = []
    friedman_significant = results.get("friedman", {}).get("p_value", 1.0) < 0.05

    if friedman_significant:
        comparisons = [("A", "B"), ("A", "C"), ("A", "D")]
        bonferroni_alpha = 0.05 / len(comparisons)

        for c1, c2 in comparisons:
            if c1 not in wide.columns or c2 not in wide.columns:
                continue
            diff = wide[c1].values - wide[c2].values
            nonzero = diff[diff != 0]
            if len(nonzero) > 0:
                w_stat, w_p = stats.wilcoxon(nonzero)
            else:
                w_stat, w_p = 0, 1.0

            posthoc.append({
                "comparison": f"{c1} vs {c2}",
                "W": float(w_stat),
                "p_value": round(float(w_p), 6),
                "significant_bonferroni": w_p < bonferroni_alpha,
                "mean_diff": round(float(np.mean(diff)), 4),
            })
    else:
        posthoc = [{"note": "Omnibus Friedman test not significant (p >= 0.05); post-hoc tests not performed."}]

    results["posthoc_wilcoxon"] = posthoc

    # Fleiss' kappa (inter-annotator agreement on category correctness)
    if len(experts) >= 2:
        results["fleiss_kappa"] = _compute_fleiss_kappa(merged)

    return results


def _compute_fleiss_kappa(merged: pd.DataFrame) -> dict:
    """Compute Fleiss' kappa on category correctness judgments (Yes/Partial/No)."""
    # Group by (Term, Condition, Category) — each unique combination is a "subject"
    # Each expert provides one rating per subject
    subjects = merged.groupby(["Term", "Condition"]).agg(
        ratings=("Correct_Raw", list)
    ).reset_index()

    categories = ["yes", "partial", "no"]
    rating_matrix = []

    for _, row in subjects.iterrows():
        raw_ratings = [str(r).strip().lower() for r in row["ratings"]]
        counts = [raw_ratings.count(c) for c in categories]
        if sum(counts) > 0:
            rating_matrix.append(counts)

    if not rating_matrix:
        return {"kappa": 0.0, "interpretation": "no data"}

    matrix = np.array(rating_matrix)
    n, k_cats = matrix.shape
    N = matrix.sum(axis=1)

    if N[0] < 2:
        return {"kappa": 0.0, "interpretation": "insufficient raters"}

    n_raters = int(N[0])

    # Fleiss' kappa formula
    p_j = matrix.sum(axis=0) / (n * n_raters)
    P_i = (np.sum(matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)
    P_e = np.sum(p_j ** 2)

    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) != 0 else 0

    interpretation = (
        "almost perfect" if kappa > 0.8 else
        "substantial" if kappa > 0.6 else
        "moderate" if kappa > 0.4 else
        "fair" if kappa > 0.2 else
        "slight"
    )

    return {"kappa": round(float(kappa), 4), "interpretation": interpretation}


# ---------------------------------------------------------------------------
# Taxonomy Correctness Analysis
# ---------------------------------------------------------------------------

def analyze_taxonomy_correctness(experts: dict, key_df: pd.DataFrame) -> dict:
    """Descriptive stats and inter-annotator agreement for taxonomy IS-A pairs."""
    results = {}

    # Collect all taxonomy judgments
    all_judgments = []
    for expert_id, data in experts.items():
        if "taxonomy" not in data:
            continue
        tax_df = data["taxonomy"]
        for _, row in tax_df.iterrows():
            correct_raw = str(row.get("Correct (Yes/No/Partial)", "")).strip().lower()
            if correct_raw in ("yes", "no", "partial"):
                if correct_raw == "yes":
                    score = 1.0
                elif correct_raw == "partial":
                    score = 0.5
                else:
                    score = 0.0
                all_judgments.append({
                    "Row_ID": row.get("Row_ID", ""),
                    "Term": row.get("Term", ""),
                    "Parent_Term": row.get("Parent_Term", ""),
                    "Expert": expert_id,
                    "Correct_Raw": correct_raw,
                    "Correct_Score": score,
                })

    if not all_judgments:
        return {"error": "No taxonomy judgments found"}

    merged = pd.DataFrame(all_judgments)

    # Overall accuracy
    overall_scores = merged["Correct_Score"].values
    results["descriptive"] = {
        "n_pairs_evaluated": len(merged["Row_ID"].unique()),
        "n_experts": len(merged["Expert"].unique()),
        "n_judgments": len(merged),
        "mean_score": round(float(np.mean(overall_scores)), 4),
        "proportion_correct": round(float(np.mean(overall_scores >= 0.9)), 4),
        "proportion_partial_or_correct": round(float(np.mean(overall_scores >= 0.4)), 4),
        "proportion_incorrect": round(float(np.mean(overall_scores < 0.1)), 4),
    }

    # Per-pair agreement across experts (Fleiss' kappa on Yes/Partial/No)
    expert_ids = sorted(merged["Expert"].unique())
    if len(expert_ids) >= 2:
        subjects = merged.groupby("Row_ID").agg(
            ratings=("Correct_Raw", list)
        ).reset_index()

        categories = ["yes", "partial", "no"]
        rating_matrix = []
        for _, row in subjects.iterrows():
            raw = [str(r).strip().lower() for r in row["ratings"]]
            counts = [raw.count(c) for c in categories]
            if sum(counts) >= 2:
                rating_matrix.append(counts)

        if rating_matrix:
            matrix = np.array(rating_matrix)
            n_subj, _ = matrix.shape
            N_per = matrix.sum(axis=1)
            n_raters = int(N_per[0]) if len(N_per) > 0 else 0

            if n_raters >= 2:
                p_j = matrix.sum(axis=0) / (n_subj * n_raters)
                P_i = (np.sum(matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
                P_bar = np.mean(P_i)
                P_e = np.sum(p_j ** 2)
                kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) != 0 else 0

                interpretation = (
                    "almost perfect" if kappa > 0.8 else
                    "substantial" if kappa > 0.6 else
                    "moderate" if kappa > 0.4 else
                    "fair" if kappa > 0.2 else
                    "slight"
                )
                results["fleiss_kappa"] = {
                    "kappa": round(float(kappa), 4),
                    "interpretation": interpretation,
                }

    # Stratify by taxonomy key metadata (if available)
    tax_key = key_df[key_df["Sheet"] == "Taxonomy_Correct"]
    if not tax_key.empty and "Category" in tax_key.columns:
        key_map = tax_key.set_index("Row_ID")["Category"].to_dict()
        merged["Category"] = merged["Row_ID"].map(key_map)

        per_cat = merged.groupby("Category")["Correct_Score"].agg(
            ["mean", "count"]
        ).reset_index()
        per_cat.columns = ["Category", "Mean_Score", "N_Judgments"]
        per_cat["Mean_Score"] = per_cat["Mean_Score"].round(4)
        results["per_category"] = per_cat.to_dict("records")

    return results


# ---------------------------------------------------------------------------
# Cross-Layer Analysis
# ---------------------------------------------------------------------------

def analyze_cross_layer(
    experts: dict, key_df: pd.DataFrame
) -> dict:
    """Spearman correlations between NLD quality and category correctness."""
    results = {}

    # Collect per-term averages for NLD quality (condition A only)
    nld_quality_by_term = {}
    for expert_id, data in experts.items():
        unblinded = unblind_nld(data["nld_quality"], key_df)
        for _, row in unblinded.iterrows():
            term = row["Term"]
            qa = row["Quality_A"]
            if pd.notna(qa):
                nld_quality_by_term.setdefault(term, []).append(float(qa))

    nld_means = {t: np.mean(v) for t, v in nld_quality_by_term.items()}

    # Collect per-term averages for category correctness (condition A only)
    cat_scores_by_term = {}
    for expert_id, data in experts.items():
        unblinded = unblind_category(data["category"], key_df)
        cond_a = unblinded[unblinded["Condition"] == "A"]
        for _, row in cond_a.iterrows():
            term = row["Term"]
            cat_scores_by_term.setdefault(term, []).append(row["Correct_Score"])

    cat_means = {t: np.mean(v) for t, v in cat_scores_by_term.items()}

    # Correlate NLD quality with category correctness
    common_terms = sorted(set(nld_means.keys()) & set(cat_means.keys()))
    if len(common_terms) >= 10:
        nld_vals = [nld_means[t] for t in common_terms]
        cat_vals = [cat_means[t] for t in common_terms]
        rho, p = stats.spearmanr(nld_vals, cat_vals)
        results["nld_vs_category"] = {
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(p), 6),
            "n_terms": len(common_terms),
        }

    # Collect per-term relevance averages
    relevance_by_term = {}
    for expert_id, data in experts.items():
        rel_df = data["relevance"]
        for _, row in rel_df.iterrows():
            term = row["Term"]
            score = row.get("Relevance (1-5)", np.nan)
            if pd.notna(score):
                relevance_by_term.setdefault(term, []).append(float(score))

    rel_means = {t: np.mean(v) for t, v in relevance_by_term.items()}

    # Correlate relevance with NLD quality
    common_nld_rel = sorted(set(nld_means.keys()) & set(rel_means.keys()))
    if len(common_nld_rel) >= 10:
        nld_vals = [nld_means[t] for t in common_nld_rel]
        rel_vals = [rel_means[t] for t in common_nld_rel]
        rho, p = stats.spearmanr(rel_vals, nld_vals)
        results["relevance_vs_nld"] = {
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(p), 6),
            "n_terms": len(common_nld_rel),
        }

    return results


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def run_layer2_analysis(
    workbook_paths: list[str],
    key_path: str,
    output_dir: str | None = None,
) -> dict:
    """Run the modular, item-aggregated Layer 2 analysis.

    Args:
        workbook_paths: List of paths to completed expert workbooks.
        key_path: Path to the blinding key CSV.
        output_dir: Output directory for results.

    Returns:
        Dict with all analysis results.
    """
    if output_dir is None:
        output_dir = ANALYSIS_DIR
    from src.evaluation.expert_eval_analysis import run_modular_analysis

    return run_modular_analysis(
        workbook_paths=workbook_paths,
        key_path=key_path,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Analyze expert evaluation results")
    parser.add_argument(
        "workbooks", nargs="+",
        help="Paths to completed expert workbooks (.xlsx)",
    )
    parser.add_argument(
        "--key", required=True,
        help="Path to blinding key CSV",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for analysis results",
    )
    args = parser.parse_args()
    run_layer2_analysis(args.workbooks, args.key, args.output_dir)
