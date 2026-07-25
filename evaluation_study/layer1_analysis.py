"""
Layer 1 Analysis — Automated cross-condition comparison for the ablation study.

All analyses are fully automated (no expert effort) and run on all terms.
Measures WHETHER conditions produce different outputs and HOW they diverge.

These analyses measure sensitivity and agreement, not classification accuracy.
Condition A is an experimental anchor, not a gold standard.
"""

import os
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from src.utils.csv_io import read_csv, write_csv
from src.utils.ontology_config import get_config
from evaluation_study.paths import ABLATION_OUTPUT

OUTPUT_DIR = os.environ.get("ABLATION_OUTPUT_DIR", str(ABLATION_OUTPUT))
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "analysis")
REQUIRED_CONDITIONS = ("A", "B", "C", "D")


def load_merged_results(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(OUTPUT_DIR, "ablation_merged.csv")
    return read_csv(path)


def _valid_categories() -> set[str]:
    cfg = get_config()
    return {
        label
        for ontology_key in cfg.waterfall_ontologies()
        for label in cfg.categories_for(ontology_key)
    } | {"NOT_CLASSIFIED"}


def validate_paired_results(
    df: pd.DataFrame,
    expected_term_count: int | None = None,
) -> None:
    """Require a complete, error-free A/B/C/D matrix before paired analysis."""
    required_columns = {"Term", "Condition", "Category"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Merged ablation results missing columns: {sorted(missing_columns)}")

    if df[list(required_columns)].isna().any().any():
        raise ValueError("Merged ablation results contain missing Term, Condition, or Category values")

    observed_conditions = set(df["Condition"].astype(str))
    if observed_conditions != set(REQUIRED_CONDITIONS):
        missing = sorted(set(REQUIRED_CONDITIONS) - observed_conditions)
        extra = sorted(observed_conditions - set(REQUIRED_CONDITIONS))
        raise ValueError(f"Expected conditions A/B/C/D (missing={missing}, extra={extra})")

    duplicate_mask = df.duplicated(["Term", "Condition"], keep=False)
    if duplicate_mask.any():
        duplicates = df.loc[duplicate_mask, ["Term", "Condition"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate term-condition rows: {duplicates}")

    invalid = sorted(set(df["Category"].astype(str)) - _valid_categories())
    errors = df["Category"].astype(str).str.startswith("ERROR")
    if invalid or errors.any():
        raise ValueError(
            f"Invalid category matrix (errors={int(errors.sum())}, invalid={invalid[:10]})"
        )

    term_sets = {
        condition: set(df.loc[df["Condition"] == condition, "Term"].astype(str))
        for condition in REQUIRED_CONDITIONS
    }
    anchor_terms = term_sets["A"]
    mismatches = {
        condition: {
            "missing": sorted(anchor_terms - terms)[:10],
            "extra": sorted(terms - anchor_terms)[:10],
        }
        for condition, terms in term_sets.items()
        if terms != anchor_terms
    }
    if mismatches:
        raise ValueError(f"Condition term sets are not paired: {mismatches}")
    if expected_term_count is not None and len(anchor_terms) != expected_term_count:
        raise ValueError(
            f"Expected {expected_term_count} paired terms; found {len(anchor_terms)}"
        )


def _pivot_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot merged df to wide format: one row per term, one column per condition."""
    validate_paired_results(df)
    return (
        df.assign(
            Term=df["Term"].astype(str),
            Condition=df["Condition"].astype(str),
            Category=df["Category"].astype(str),
        )
        .pivot(index="Term", columns="Condition", values="Category")
        .reset_index()
    )


def _category_to_tier() -> dict[str, str]:
    cfg = get_config()
    mapping = {"NOT_CLASSIFIED": "NOT_CLASSIFIED"}
    for ontology_key in cfg.waterfall_ontologies():
        tier = cfg.ontologies[ontology_key].eval_tier.upper()
        for category in cfg.categories_for(ontology_key):
            mapping[category] = tier
    return mapping


def _pivot_tiers(df: pd.DataFrame) -> pd.DataFrame:
    if "Tier" in df.columns:
        validate_paired_results(df)
        if df["Tier"].isna().any():
            raise ValueError("Frozen Tier column contains missing values")
        return (
            df.assign(
                Term=df["Term"].astype(str),
                Condition=df["Condition"].astype(str),
                Tier=df["Tier"].astype(str),
            )
            .pivot(index="Term", columns="Condition", values="Tier")
            .reset_index()
        )
    wide = _pivot_categories(df)
    mapping = _category_to_tier()
    for condition in REQUIRED_CONDITIONS:
        wide[condition] = wide[condition].map(mapping)
        if wide[condition].isna().any():
            raise ValueError(f"Could not map all Condition {condition} categories to tiers")
    return wide


# ---------------------------------------------------------------------------
# 1. Cross-condition agreement matrix
# ---------------------------------------------------------------------------

def agreement_matrix(df: pd.DataFrame, level: str = "exact") -> pd.DataFrame:
    """Pairwise exact or ontology-tier agreement and Cohen's kappa."""
    if level not in {"exact", "tier"}:
        raise ValueError("level must be 'exact' or 'tier'")
    wide = _pivot_categories(df) if level == "exact" else _pivot_tiers(df)
    results = []
    for c1, c2 in itertools.combinations(REQUIRED_CONDITIONS, 2):
        agree = (wide[c1] == wide[c2]).sum()
        total = len(wide)
        labels = set(wide[c1]) | set(wide[c2])
        kappa = (
            float(cohen_kappa_score(wide[c1], wide[c2]))
            if len(labels) > 1
            else float("nan")
        )
        results.append({
            "Pair": f"{c1} vs {c2}",
            "Level": level,
            "Agreement": agree,
            "Total": total,
            "Agreement_Rate": round(agree / total, 4) if total else 0,
            "Cohens_Kappa": round(kappa, 4) if np.isfinite(kappa) else np.nan,
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 2. Cochran's Q test
# ---------------------------------------------------------------------------

def _agreement_indicators(df: pd.DataFrame) -> pd.DataFrame:
    wide = _pivot_categories(df)
    return pd.DataFrame({
        "Term": wide["Term"],
        "A_B_Agreement": (wide["A"] == wide["B"]).astype(int),
        "A_C_Agreement": (wide["A"] == wide["C"]).astype(int),
        "A_D_Agreement": (wide["A"] == wide["D"]).astype(int),
    })


def cochrans_q_test(df: pd.DataFrame) -> dict:
    """Test whether the three A-anchored exact-agreement rates are equal.

    The test has k=3 paired binary indicators and therefore df=k-1=2. It
    compares sensitivity rates; it is not a test of category correctness.
    """
    indicators = _agreement_indicators(df)
    columns = ["A_B_Agreement", "A_C_Agreement", "A_D_Agreement"]
    matrix = indicators[columns].to_numpy(dtype=float)
    n_terms, n_measures = matrix.shape
    column_sums = matrix.sum(axis=0)
    row_sums = matrix.sum(axis=1)
    total = column_sums.sum()
    denominator = n_measures * total - np.square(row_sums).sum()
    if denominator == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        numerator = (n_measures - 1) * (
            n_measures * np.square(column_sums).sum() - total ** 2
        )
        statistic = float(numerator / denominator)
        p_value = float(stats.chi2.sf(statistic, df=n_measures - 1))
    rates = column_sums / n_terms
    return {
        "Test": "Cochran Q on A-anchored exact agreement",
        "Q": round(statistic, 6),
        "df": n_measures - 1,
        "p_value": p_value,
        "N_terms": n_terms,
        "A_B_Agreement_Rate": round(float(rates[0]), 6),
        "A_C_Agreement_Rate": round(float(rates[1]), 6),
        "A_D_Agreement_Rate": round(float(rates[2]), 6),
        "Agreement_Range": round(float(rates.max() - rates.min()), 6),
    }


# ---------------------------------------------------------------------------
# 3. Category migration analysis
# ---------------------------------------------------------------------------

def sensitivity_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Emit independent sensitivity flags; one term may be sensitive on several axes."""
    wide = _pivot_categories(df)
    tiers = _pivot_tiers(df)
    result = pd.DataFrame({
        "Term": wide["Term"],
        "Cat_A": wide["A"],
        "Cat_B": wide["B"],
        "Cat_C": wide["C"],
        "Cat_D": wide["D"],
        "Tier_A": tiers["A"],
        "Tier_B": tiers["B"],
        "Tier_C": tiers["C"],
        "Tier_D": tiers["D"],
    })
    result["RAG_Sensitive"] = result["Cat_A"] != result["Cat_B"]
    result["NLD_Sensitive"] = result["Cat_A"] != result["Cat_C"]
    result["Structuring_Sensitive"] = result["Cat_A"] != result["Cat_D"]
    result["RAG_Tier_Sensitive"] = result["Tier_A"] != result["Tier_B"]
    result["NLD_Tier_Sensitive"] = result["Tier_A"] != result["Tier_C"]
    result["Structuring_Tier_Sensitive"] = result["Tier_A"] != result["Tier_D"]
    exact_columns = ["RAG_Sensitive", "NLD_Sensitive", "Structuring_Sensitive"]
    result["Sensitivity_Count"] = result[exact_columns].sum(axis=1)
    result["Exact_Stable"] = result["Sensitivity_Count"] == 0
    return result


def category_migration(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for the independent sensitivity report."""
    return sensitivity_flags(df)


def confusion_matrices(df: pd.DataFrame, level: str = "exact") -> pd.DataFrame:
    """Return observed A-vs-comparator confusion cells in long form."""
    if level not in {"exact", "tier"}:
        raise ValueError("level must be 'exact' or 'tier'")
    wide = _pivot_categories(df) if level == "exact" else _pivot_tiers(df)
    rows = []
    for comparator in ("B", "C", "D"):
        counts = (
            wide.groupby(["A", comparator], dropna=False)
            .size()
            .reset_index(name="Count")
        )
        for _, cell in counts.iterrows():
            rows.append({
                "Pair": f"A vs {comparator}",
                "Level": level,
                "Anchor_Value": cell["A"],
                "Comparator_Value": cell[comparator],
                "Count": int(cell["Count"]),
            })
    return pd.DataFrame(rows)


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down family-wise error correction."""
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[index]))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()


def mcnemar_posthoc(
    df: pd.DataFrame,
    omnibus_p_value: float,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Exact McNemar contrasts between A-anchored agreement indicators.

    Each contrast is the exact binomial form of McNemar's paired test on the
    discordant indicator pairs. Holm correction controls the three-test family.
    """
    indicators = _agreement_indicators(df)
    columns = ["A_B_Agreement", "A_C_Agreement", "A_D_Agreement"]
    rows = []
    raw_p_values = []
    for first, second in itertools.combinations(columns, 2):
        first_only = int(((indicators[first] == 1) & (indicators[second] == 0)).sum())
        second_only = int(((indicators[first] == 0) & (indicators[second] == 1)).sum())
        discordant = first_only + second_only
        p_value = (
            float(stats.binomtest(first_only, discordant, p=0.5).pvalue)
            if discordant
            else 1.0
        )
        raw_p_values.append(p_value)
        rows.append({
            "Comparison": f"{first} vs {second}",
            "First_Only": first_only,
            "Second_Only": second_only,
            "Discordant": discordant,
            "Discordant_Fraction": round(discordant / len(indicators), 6),
            "First_Rate": round(float(indicators[first].mean()), 6),
            "Second_Rate": round(float(indicators[second].mean()), 6),
            "Rate_Difference": round(
                float(indicators[first].mean() - indicators[second].mean()),
                6,
            ),
            "p_value": p_value,
        })
    adjusted = _holm_adjust(raw_p_values)
    gate_passed = omnibus_p_value < alpha
    for row, adjusted_p in zip(rows, adjusted):
        row["p_value_holm"] = adjusted_p
        row["Omnibus_Gate_Passed"] = gate_passed
        row["Significant"] = bool(gate_passed and adjusted_p < alpha)
    return pd.DataFrame(rows)


def _stuart_maxwell_test(first: pd.Series, second: pd.Series) -> dict:
    """Stuart-Maxwell marginal-homogeneity test for paired nominal outcomes."""
    labels = sorted(set(first.astype(str)) | set(second.astype(str)))
    table = pd.crosstab(first.astype(str), second.astype(str)).reindex(
        index=labels,
        columns=labels,
        fill_value=0,
    )
    matrix = table.to_numpy(dtype=float)
    row_totals = matrix.sum(axis=1)
    column_totals = matrix.sum(axis=0)
    differences = row_totals - column_totals
    covariance = np.zeros_like(matrix)
    for row_index in range(len(labels)):
        covariance[row_index, row_index] = (
            row_totals[row_index]
            + column_totals[row_index]
            - 2 * matrix[row_index, row_index]
        )
        for column_index in range(len(labels)):
            if row_index != column_index:
                covariance[row_index, column_index] = -(
                    matrix[row_index, column_index]
                    + matrix[column_index, row_index]
                )
    expected_degrees_freedom = max(len(labels) - 1, 0)
    degrees_freedom = int(np.linalg.matrix_rank(covariance))
    if degrees_freedom == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = float(differences @ np.linalg.pinv(covariance) @ differences)
        p_value = float(stats.chi2.sf(statistic, degrees_freedom))
    marginal_l1 = float(
        0.5 * np.abs(row_totals / row_totals.sum() - column_totals / column_totals.sum()).sum()
    )
    return {
        "Statistic": statistic,
        "df": degrees_freedom,
        "Levels": len(labels),
        "Expected_df": expected_degrees_freedom,
        "Rank_Deficient": degrees_freedom < expected_degrees_freedom,
        "p_value": p_value,
        "Marginal_L1_Distance": marginal_l1,
    }


def stuart_maxwell_tests(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Compare A with B/C/D at the ontology-tier level, Holm-corrected."""
    tiers = _pivot_tiers(df)
    rows = []
    p_values = []
    for comparator in ("B", "C", "D"):
        result = _stuart_maxwell_test(tiers["A"], tiers[comparator])
        p_values.append(result["p_value"])
        rows.append({"Pair": f"A vs {comparator}", **result})
    for row, adjusted_p in zip(rows, _holm_adjust(p_values)):
        row["p_value_holm"] = adjusted_p
        row["Significant"] = bool(adjusted_p < alpha)
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
    Describe A/B agreement by Condition A's Context_Used flag.

    This is a sensitivity summary only; divergence does not imply improvement.
    """
    wide = _pivot_categories(df)

    # Get Context_Used flag from condition A's data
    df_a = df[df["Condition"] == "A"].copy()
    if "RAG_Context_Used" in df_a.columns:
        context_used = df_a["RAG_Context_Used"]
        if "Context_Used" in df_a.columns:
            context_used = context_used.combine_first(df_a["Context_Used"])
    elif "Context_Used" in df_a.columns:
        context_used = df_a["Context_Used"]
    else:
        return pd.DataFrame()
    df_a = pd.DataFrame({
        "Term": df_a["Term"].astype(str),
        "Context_Used": context_used,
    }).drop_duplicates()
    merged = wide.merge(df_a, on="Term", how="left")

    rows = []
    normalized = merged["Context_Used"].astype(str).str.strip().str.lower()
    for ctx_val in ("true", "false", "error"):
        sub = merged[normalized == ctx_val]
        if len(sub) == 0:
            continue
        agree = (sub["A"] == sub["B"]).sum()
        total = len(sub)
        rows.append({
            "Context_Used": ctx_val.title(),
            "N_terms": total,
            "A_B_Agreement": agree,
            "A_B_Agreement_Rate": round(agree / total, 4) if total else 0,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run all analyses
# ---------------------------------------------------------------------------

def run_layer1_analysis(
    merged_path: str | None = None,
    expected_term_count: int | None = None,
) -> dict:
    """Run all automated sensitivity analyses and save their outputs."""
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    df = load_merged_results(merged_path)
    if expected_term_count is None:
        expected_term_count = int(os.environ.get("ABLATION_EXPECTED_TERM_COUNT", 407))
    validate_paired_results(df, expected_term_count=expected_term_count)
    print(f"\nLayer 1 Analysis: {len(df)} rows, conditions: {sorted(df['Condition'].unique())}")
    print("  Interpretation: sensitivity/agreement only; Condition A is not ground truth.")

    results = {}

    print("\n1. Pairwise exact and ontology-tier agreement:")
    exact_agreement = agreement_matrix(df, level="exact")
    tier_agreement = agreement_matrix(df, level="tier")
    write_csv(exact_agreement, os.path.join(ANALYSIS_DIR, "exact_agreement.csv"))
    write_csv(tier_agreement, os.path.join(ANALYSIS_DIR, "tier_agreement.csv"))
    print(exact_agreement.to_string(index=False))
    results["exact_agreement"] = exact_agreement
    results["tier_agreement"] = tier_agreement

    print("\n2. Independent sensitivity flags:")
    sensitivities = sensitivity_flags(df)
    write_csv(sensitivities, os.path.join(ANALYSIS_DIR, "sensitivity_flags.csv"))
    for column in ("RAG_Sensitive", "NLD_Sensitive", "Structuring_Sensitive"):
        count = int(sensitivities[column].sum())
        print(f"  {column}: {count}/{len(sensitivities)} ({count / len(sensitivities):.1%})")
    results["sensitivity_flags"] = sensitivities

    print("\n3. Category and tier confusion matrices:")
    category_confusion = confusion_matrices(df, level="exact")
    tier_confusion = confusion_matrices(df, level="tier")
    write_csv(category_confusion, os.path.join(ANALYSIS_DIR, "category_confusion.csv"))
    write_csv(tier_confusion, os.path.join(ANALYSIS_DIR, "tier_confusion.csv"))
    results["category_confusion"] = category_confusion
    results["tier_confusion"] = tier_confusion

    print("\n4. Global Cochran Q and Holm-corrected McNemar post-hoc tests:")
    cochran = cochrans_q_test(df)
    cochran_df = pd.DataFrame([cochran])
    write_csv(cochran_df, os.path.join(ANALYSIS_DIR, "cochrans_q_agreement.csv"))
    mcnemar = mcnemar_posthoc(df, omnibus_p_value=float(cochran["p_value"]))
    write_csv(mcnemar, os.path.join(ANALYSIS_DIR, "mcnemar_posthoc.csv"))
    print(f"  Q({cochran['df']}) = {cochran['Q']:.4f}, p = {cochran['p_value']:.6g}")
    results["cochrans_q"] = cochran_df
    results["mcnemar_posthoc"] = mcnemar

    print("\n5. Holm-corrected Stuart-Maxwell tier tests:")
    stuart_maxwell = stuart_maxwell_tests(df)
    write_csv(stuart_maxwell, os.path.join(ANALYSIS_DIR, "stuart_maxwell_tier.csv"))
    print(stuart_maxwell.to_string(index=False))
    results["stuart_maxwell"] = stuart_maxwell

    print("\n6. NOT_CLASSIFIED rates:")
    nc = not_classified_rates(df)
    write_csv(nc, os.path.join(ANALYSIS_DIR, "not_classified_rates.csv"))
    print(nc.to_string(index=False))
    results["not_classified"] = nc

    print("\n7. Descriptive Context_Used subgroup (A vs B):")
    cu = context_used_subgroup(df)
    if not cu.empty:
        write_csv(cu, os.path.join(ANALYSIS_DIR, "context_used_subgroup.csv"))
        print(cu.to_string(index=False))
    else:
        print("  (Skipped: Condition A has no Context_Used field.)")
    results["context_used"] = cu

    print(f"\nAll Layer 1 results saved to {ANALYSIS_DIR}/")
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_layer1_analysis()
