"""Statistical analysis for modular expert-evaluation workbooks.

Expert ratings are aggregated to the sampled item before inferential tests.
The final ontology task families remain separate; no composite ontology score
is computed.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from src.utils.csv_io import read_csv, write_csv


REQUIRED_SHEETS = (
    "Representation",
    "Category_Correct",
    "Taxonomy",
    "Defined_Classes",
    "Relations",
    "Individuals",
    "Critic_Decisions",
)
CORRECTNESS_CHOICES = ("yes", "partial", "no", "unsure")
BINARY_CHOICES = ("yes", "no", "unsure")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_manifest(key_path: str) -> str:
    """Verify that workbook source artifacts have not changed since sampling."""
    key = Path(key_path).resolve()
    candidates = [
        key.parent.parent / "expert_evaluation_manifest.json",
        key.parent / "expert_evaluation_manifest.json",
    ]
    manifest_path = next((path for path in candidates if path.exists()), None)
    if manifest_path is None:
        raise FileNotFoundError(
            "Expert evaluation manifest not found beside the private key. "
            "Do not analyze workbooks without their source-hash manifest."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_paths = manifest.get("source_paths") or {}
    source_hashes = manifest.get("source_sha256") or {}
    if set(source_paths) != set(source_hashes) or not source_paths:
        raise ValueError("Expert evaluation manifest has incomplete source path/hash mappings")
    mismatches = []
    for name, path in source_paths.items():
        if not os.path.exists(path):
            mismatches.append(f"{name}: missing {path}")
            continue
        observed = _sha256_file(path)
        if observed != source_hashes[name]:
            mismatches.append(f"{name}: expected {source_hashes[name]}, observed {observed}")
    if mismatches:
        raise ValueError(
            "Expert evaluation source artifacts changed after workbook generation: "
            + "; ".join(mismatches[:10])
        )
    return str(manifest_path)


def _expert_id(path: str, fallback_index: int) -> str:
    match = re.search(r"expert_evaluation_(\d+)\.xlsx$", os.path.basename(path))
    return f"expert_{match.group(1)}" if match else f"expert_{fallback_index}"


def load_completed_workbooks(workbook_paths: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    """Load modular sheets and reject missing or duplicate stable row IDs."""
    experts: dict[str, dict[str, pd.DataFrame]] = {}
    for index, path in enumerate(workbook_paths, 1):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expert workbook not found: {path}")
        expert_id = _expert_id(path, index)
        if expert_id in experts:
            raise ValueError(f"Duplicate expert ID inferred from workbook paths: {expert_id}")
        excel = pd.ExcelFile(path, engine="openpyxl")
        missing = sorted(set(REQUIRED_SHEETS) - set(excel.sheet_names))
        if missing:
            raise ValueError(f"{path} missing sheets: {missing}")
        sheets = {
            sheet: pd.read_excel(excel, sheet_name=sheet)
            for sheet in REQUIRED_SHEETS
        }
        for sheet, frame in sheets.items():
            if "Row_ID" not in frame.columns:
                raise ValueError(f"{path}:{sheet} missing Row_ID")
            if frame["Row_ID"].isna().any() or frame["Row_ID"].astype(str).duplicated().any():
                raise ValueError(f"{path}:{sheet} has missing or duplicate Row_ID values")
        experts[expert_id] = sheets
    return experts


def _key_rows(key: pd.DataFrame, expert_id: str, sheet: str) -> pd.DataFrame:
    required = {"Expert_ID", "Sheet", "Row_ID"}
    missing = required - set(key.columns)
    if missing:
        raise ValueError(f"Blinding key missing columns: {sorted(missing)}")
    rows = key[(key["Expert_ID"] == expert_id) & (key["Sheet"] == sheet)].copy()
    if rows.empty:
        raise ValueError(f"Blinding key has no rows for {expert_id}:{sheet}")
    if rows["Row_ID"].astype(str).duplicated().any():
        raise ValueError(f"Blinding key has duplicate rows for {expert_id}:{sheet}")
    return rows


def _validate_id_match(
    response: pd.DataFrame,
    key: pd.DataFrame,
    expert_id: str,
    sheet: str,
) -> None:
    response_ids = set(response["Row_ID"].astype(str))
    key_ids = set(key["Row_ID"].astype(str))
    if response_ids != key_ids:
        raise ValueError(
            f"{expert_id}:{sheet} Row_ID mismatch "
            f"(missing={sorted(key_ids - response_ids)[:10]}, "
            f"extra={sorted(response_ids - key_ids)[:10]})"
        )


def _choice(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _validated_choice(value: object, allowed: tuple[str, ...], label: str) -> str:
    choice = _choice(value).lower()
    if choice not in allowed:
        raise ValueError(f"{label}: expected one of {allowed}, got {value!r}")
    return choice


def _numeric_rating(value: object, label: str) -> float:
    if _choice(value).lower() == "unsure":
        return np.nan
    try:
        rating = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label}: missing or non-numeric rating {value!r}") from None
    if not 1 <= rating <= 5:
        raise ValueError(f"{label}: rating must be between 1 and 5, got {rating}")
    return rating


def unblind_representation(
    experts: dict[str, dict[str, pd.DataFrame]],
    key: pd.DataFrame,
) -> pd.DataFrame:
    """Unblind A/B quality and aggregate-ready preference values."""
    rows = []
    for expert_id, sheets in experts.items():
        response = sheets["Representation"]
        key_part = _key_rows(key, expert_id, "Representation")
        _validate_id_match(response, key_part, expert_id, "Representation")
        key_map = key_part.set_index("Row_ID")
        for item in response.to_dict("records"):
            row_id = item["Row_ID"]
            hidden = key_map.loc[row_id]
            first_condition = str(hidden["Definition_1_Condition"])
            second_condition = str(hidden["Definition_2_Condition"])
            if {first_condition, second_condition} != {"A", "B"}:
                raise ValueError(f"{expert_id}:{row_id} invalid definition condition order")
            quality_first = _numeric_rating(item.get("Quality_1 (1-5/Unsure)"), f"{expert_id}:{row_id}:Quality_1")
            quality_second = _numeric_rating(item.get("Quality_2 (1-5/Unsure)"), f"{expert_id}:{row_id}:Quality_2")
            quality = {first_condition: quality_first, second_condition: quality_second}
            preference = _validated_choice(
                item.get("Preference (1/2/Tie/Unsure)"),
                ("1", "2", "tie", "unsure"),
                f"{expert_id}:{row_id}:Preference",
            )
            if preference == "unsure":
                preference_a = np.nan
            elif preference == "tie":
                preference_a = 0
            else:
                preferred_condition = first_condition if preference == "1" else second_condition
                preference_a = 1 if preferred_condition == "A" else -1
            rows.append({
                "Row_ID": row_id,
                "Term": hidden["Term"],
                "Expert": expert_id,
                "Relevance": _numeric_rating(
                    item.get("Relevance (1-5/Unsure)"),
                    f"{expert_id}:{row_id}:Relevance",
                ),
                "Quality_A": quality["A"],
                "Quality_B": quality["B"],
                "Preference_A": preference_a,
                "Context_Used_A": hidden.get("Context_Used_A", ""),
                "Tier_A": hidden.get("Tier_A", ""),
                "Frequency_Band": hidden.get("Frequency_Band", ""),
                "Final_Fate": hidden.get("Final_Fate", ""),
            })
    return pd.DataFrame(rows)


def unblind_categories(
    experts: dict[str, dict[str, pd.DataFrame]],
    key: pd.DataFrame,
) -> pd.DataFrame:
    """Propagate each deduplicated assignment judgment to its conditions."""
    rows = []
    for expert_id, sheets in experts.items():
        response = sheets["Category_Correct"]
        key_part = _key_rows(key, expert_id, "Category_Correct")
        _validate_id_match(response, key_part, expert_id, "Category_Correct")
        key_map = key_part.set_index("Row_ID")
        for item in response.to_dict("records"):
            row_id = item["Row_ID"]
            hidden = key_map.loc[row_id]
            raw = _validated_choice(
                item.get("Correct (Yes/Partial/No/Unsure)"),
                CORRECTNESS_CHOICES,
                f"{expert_id}:{row_id}:Category correctness",
            )
            score = {"yes": 1.0, "partial": 0.5, "no": 0.0, "unsure": np.nan}[raw]
            conditions = [value.strip() for value in str(hidden["Conditions"]).split(",")]
            if not conditions or any(condition not in {"A", "B", "C", "D"} for condition in conditions):
                raise ValueError(f"{expert_id}:{row_id} has invalid condition mapping {conditions}")
            for condition in conditions:
                rows.append({
                    "Assignment_ID": row_id,
                    "Term": hidden["Term"],
                    "Assigned_Category": hidden["Assigned_Category"],
                    "Condition": condition,
                    "Expert": expert_id,
                    "Correct_Raw": raw,
                    "Correct_Score": score,
                    "Tier": hidden.get("Tier", ""),
                    "Final_Fate": hidden.get("Final_Fate", ""),
                })
    result = pd.DataFrame(rows)
    duplicates = result.duplicated(["Term", "Condition", "Expert"], keep=False)
    if duplicates.any():
        raise ValueError("Category unblinding produced duplicate term-condition-expert rows")
    return result


def collect_final_sheet(
    experts: dict[str, dict[str, pd.DataFrame]],
    key: pd.DataFrame,
    sheet: str,
) -> pd.DataFrame:
    """Join one final-ontology response sheet to its hidden sample metadata."""
    frames = []
    for expert_id, sheets in experts.items():
        response = sheets[sheet].copy()
        key_part = _key_rows(key, expert_id, sheet)
        _validate_id_match(response, key_part, expert_id, sheet)
        hidden_columns = [column for column in key_part.columns if column not in response.columns or column == "Row_ID"]
        joined = response.merge(
            key_part[hidden_columns],
            on="Row_ID",
            how="left",
            validate="one_to_one",
        )
        joined["Expert"] = expert_id
        frames.append(joined)
    return pd.concat(frames, ignore_index=True)


def _icc_2_1(matrix: np.ndarray) -> dict:
    """ICC(2,1): two-way random, absolute agreement, single rater."""
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return {"icc_2_1": None, "interpretation": "insufficient data"}
    subjects, raters = matrix.shape
    grand_mean = matrix.mean()
    ss_total = np.square(matrix - grand_mean).sum()
    ss_subjects = raters * np.square(matrix.mean(axis=1) - grand_mean).sum()
    ss_raters = subjects * np.square(matrix.mean(axis=0) - grand_mean).sum()
    ss_error = ss_total - ss_subjects - ss_raters
    ms_subjects = ss_subjects / (subjects - 1)
    ms_raters = ss_raters / (raters - 1)
    ms_error = ss_error / ((subjects - 1) * (raters - 1))
    denominator = (
        ms_subjects
        + (raters - 1) * ms_error
        + (raters / subjects) * (ms_raters - ms_error)
    )
    value = (
        (ms_subjects - ms_error) / denominator
        if abs(denominator) >= 1e-12
        else np.nan
    )
    interpretation = (
        "excellent" if value >= 0.90 else
        "good" if value >= 0.75 else
        "moderate" if value >= 0.50 else
        "poor"
    ) if np.isfinite(value) else "undefined"
    return {
        "icc_2_1": round(float(value), 4) if np.isfinite(value) else None,
        "interpretation": interpretation,
        "n_items": subjects,
        "n_raters": raters,
    }


def _rating_matrix(frame: pd.DataFrame, value_column: str) -> np.ndarray:
    wide = frame.pivot(index="Term", columns="Expert", values=value_column).dropna()
    return wide.to_numpy(dtype=float)


def _pairwise_weighted_kappa(frame: pd.DataFrame, value_column: str) -> list[dict]:
    wide = frame.pivot(index="Term", columns="Expert", values=value_column)
    rows = []
    for first, second in combinations(sorted(wide.columns), 2):
        pair = wide[[first, second]].dropna()
        if pair.empty:
            continue
        labels = set(pair[first].astype(int)) | set(pair[second].astype(int))
        value = (
            cohen_kappa_score(
                pair[first].astype(int),
                pair[second].astype(int),
                weights="quadratic",
            )
            if len(labels) > 1
            else np.nan
        )
        rows.append({
            "pair": f"{first} vs {second}",
            "quadratic_weighted_kappa": round(float(value), 4) if np.isfinite(value) else None,
            "n_items": len(pair),
        })
    return rows


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[np.isfinite(differences) & (differences != 0)]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted.tolist()


def _bootstrap_mean_ci(
    values: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if iterations < 1:
        raise ValueError(f"Bootstrap iterations must be positive; got {iterations}")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None, None
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return round(float(lower), 4), round(float(upper), 4)


def _fleiss_kappa(
    frame: pd.DataFrame,
    item_column: str,
    raw_column: str,
    categories: tuple[str, ...],
) -> dict:
    rating_lists = frame.groupby(item_column)[raw_column].apply(list)
    counts = np.array([
        [ratings.count(category) for category in categories]
        for ratings in rating_lists
    ], dtype=float)
    if len(counts) == 0:
        return {"kappa": None, "interpretation": "no data"}
    raters = counts.sum(axis=1)
    if np.any(raters != raters[0]) or raters[0] < 2:
        return {"kappa": None, "interpretation": "unequal or insufficient raters"}
    n_raters = int(raters[0])
    category_rates = counts.sum(axis=0) / counts.sum()
    observed = (np.square(counts).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    observed_mean = float(observed.mean())
    expected = float(np.square(category_rates).sum())
    value = (observed_mean - expected) / (1 - expected) if expected < 1 else np.nan
    interpretation = (
        "almost perfect" if value >= 0.81 else
        "substantial" if value >= 0.61 else
        "moderate" if value >= 0.41 else
        "fair" if value >= 0.21 else
        "slight"
    ) if np.isfinite(value) else "undefined"
    return {
        "kappa": round(float(value), 4) if np.isfinite(value) else None,
        "interpretation": interpretation,
        "n_items": len(counts),
        "n_raters": n_raters,
        "category_marginals": {
            category: round(float(rate), 4)
            for category, rate in zip(categories, category_rates)
        },
        "prevalence_warning": bool(category_rates.max() >= 0.70),
    }


def _summarize_judgments(
    frame: pd.DataFrame,
    item_column: str,
    raw_column: str,
    allowed: tuple[str, ...],
    bootstrap_iterations: int,
    seed: int,
    partial: bool,
) -> dict:
    work = frame[[item_column, "Expert", raw_column]].copy()
    work["Raw"] = [
        _validated_choice(value, allowed, f"{item_column}:{raw_column}")
        for value in work[raw_column]
    ]
    if work.duplicated([item_column, "Expert"]).any():
        raise ValueError(f"Duplicate expert judgments for {item_column}:{raw_column}")
    score_map = {"yes": 1.0, "no": 0.0, "unsure": np.nan}
    if partial:
        score_map["partial"] = 0.5
    work["Score"] = work["Raw"].map(score_map)
    work["Yes"] = work["Raw"].eq("yes").astype(float)
    work["Decisive"] = ~work["Raw"].eq("unsure")
    item_rows = []
    for item, group in work.groupby(item_column):
        decisive = group[group["Decisive"]]
        item_rows.append({
            item_column: item,
            "Mean_Score": float(decisive["Score"].mean()) if not decisive.empty else np.nan,
            "Yes_Proportion": float(decisive["Yes"].mean()) if not decisive.empty else np.nan,
            "Decisive_Ratings": int(len(decisive)),
            "Unsure_Ratings": int((~group["Decisive"]).sum()),
        })
    items = pd.DataFrame(item_rows)
    lower, upper = _bootstrap_mean_ci(
        items["Yes_Proportion"].to_numpy(dtype=float),
        bootstrap_iterations,
        seed,
    )
    score_lower, score_upper = _bootstrap_mean_ci(
        items["Mean_Score"].to_numpy(dtype=float),
        bootstrap_iterations,
        seed + 1,
    )
    mean_score = items["Mean_Score"].mean()
    proportion_yes = items["Yes_Proportion"].mean()
    distribution = work["Raw"].value_counts().reindex(allowed, fill_value=0).to_dict()
    decisive_ratings = int(work["Decisive"].sum())
    unsure_ratings = int((~work["Decisive"]).sum())
    all_unsure_items = int(items["Decisive_Ratings"].eq(0).sum())
    return {
        "n_items": int(len(items)),
        "n_ratings": int(len(work)),
        "decisive_ratings": decisive_ratings,
        "unsure_ratings": unsure_ratings,
        "unsure_rate": round(unsure_ratings / len(work), 4) if len(work) else None,
        "all_unsure_items": all_unsure_items,
        "mean_score": round(float(mean_score), 4) if pd.notna(mean_score) else None,
        "mean_score_ci_95": [score_lower, score_upper],
        "proportion_yes": (
            round(float(proportion_yes), 4) if pd.notna(proportion_yes) else None
        ),
        "proportion_yes_ci_95": [lower, upper],
        "proportion_yes_denominator": "decisive non-Unsure ratings, aggregated by item",
        "rating_distribution": {key: int(value) for key, value in distribution.items()},
        "fleiss_kappa": _fleiss_kappa(work, item_column, "Raw", allowed),
    }


def analyze_representation(
    representation: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> dict:
    """Analyze relevance and A/B NLD quality using terms as independent units."""
    item_means = representation.groupby("Term").agg(
        Relevance=("Relevance", "mean"),
        Quality_A=("Quality_A", "mean"),
        Quality_B=("Quality_B", "mean"),
        Preference_Sum=("Preference_A", lambda values: values.sum(min_count=1)),
    ).reset_index()
    quality_items = item_means.dropna(subset=["Quality_A", "Quality_B"])
    differences = (
        quality_items["Quality_A"] - quality_items["Quality_B"]
    ).to_numpy(dtype=float)
    nonzero = differences[differences != 0]
    if len(nonzero):
        wilcoxon = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
        statistic, p_value = float(wilcoxon.statistic), float(wilcoxon.pvalue)
    else:
        statistic, p_value = 0.0, 1.0

    preference_items = item_means.dropna(subset=["Preference_Sum"])
    item_preference = np.sign(
        preference_items["Preference_Sum"].to_numpy(dtype=float)
    )
    prefer_a = int((item_preference > 0).sum())
    prefer_b = int((item_preference < 0).sum())
    ties = int((item_preference == 0).sum())
    unsure_preferences = len(item_means) - len(preference_items)
    decisive = prefer_a + prefer_b
    preference_p = float(stats.binomtest(prefer_a, decisive, p=0.5).pvalue) if decisive else 1.0

    quality_results = {
        "unit_of_analysis": "term-level mean across experts",
        "n_terms": len(quality_items),
        "n_terms_sampled": len(item_means),
        "n_terms_excluded_all_unsure": len(item_means) - len(quality_items),
        "quality_A_mean": (
            round(float(quality_items["Quality_A"].mean()), 4)
            if len(quality_items)
            else None
        ),
        "quality_B_mean": (
            round(float(quality_items["Quality_B"].mean()), 4)
            if len(quality_items)
            else None
        ),
        "mean_difference_A_minus_B": (
            round(float(differences.mean()), 4) if len(differences) else None
        ),
        "wilcoxon": {
            "W": statistic,
            "p_value": p_value,
            "n_nonzero_pairs": int(len(nonzero)),
            "rank_biserial": round(float(_rank_biserial(differences)), 4),
        },
        "preference_sign_test": {
            "prefer_A": prefer_a,
            "prefer_B": prefer_b,
            "ties": ties,
            "unsure": unsure_preferences,
            "p_value": preference_p,
            "unit_of_analysis": "term-level majority preference",
        },
        "agreement": {
            "icc_A": _icc_2_1(_rating_matrix(representation, "Quality_A")),
            "icc_B": _icc_2_1(_rating_matrix(representation, "Quality_B")),
            "weighted_kappa_A": _pairwise_weighted_kappa(representation, "Quality_A"),
            "weighted_kappa_B": _pairwise_weighted_kappa(representation, "Quality_B"),
        },
    }
    relevance_items = item_means.dropna(subset=["Relevance"])
    relevance_lower, relevance_upper = _bootstrap_mean_ci(
        relevance_items["Relevance"].to_numpy(dtype=float),
        bootstrap_iterations,
        seed,
    )
    relevance = {
        "n_terms": len(relevance_items),
        "n_terms_sampled": len(item_means),
        "n_terms_excluded_all_unsure": len(item_means) - len(relevance_items),
        "mean": (
            round(float(relevance_items["Relevance"].mean()), 4)
            if len(relevance_items)
            else None
        ),
        "median": (
            round(float(relevance_items["Relevance"].median()), 4)
            if len(relevance_items)
            else None
        ),
        "mean_ci_95": [relevance_lower, relevance_upper],
        "icc": _icc_2_1(_rating_matrix(representation, "Relevance")),
    }
    fate_counts = (
        representation.drop_duplicates("Term")["Final_Fate"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return {
        "relevance": relevance,
        "nld_quality": quality_results,
        "sample_final_fates": {key: int(value) for key, value in fate_counts.items()},
    }


def analyze_categories(
    category: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> dict:
    """Analyze four-condition correctness after term-level expert aggregation."""
    conditions = ("A", "B", "C", "D")
    summaries = {}
    for offset, condition in enumerate(conditions):
        subset = category[category["Condition"] == condition].copy()
        subset["Item"] = subset["Term"].astype(str) + "|" + condition
        summaries[condition] = _summarize_judgments(
            subset,
            item_column="Item",
            raw_column="Correct_Raw",
            allowed=CORRECTNESS_CHOICES,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed + offset * 10,
            partial=True,
        )

    item_scores = category.groupby(["Term", "Condition"])["Correct_Score"].mean().reset_index()
    wide_all = item_scores.pivot(index="Term", columns="Condition", values="Correct_Score")
    wide_all = wide_all.reindex(columns=list(conditions))
    wide = wide_all.dropna()
    excluded_terms = len(wide_all) - len(wide)
    if len(wide) < 2:
        statistic, p_value, kendalls_w = 0.0, 1.0, 0.0
        omnibus = {
            "performed": False,
            "reason": "Fewer than two terms had decisive ratings in all four conditions",
            "chi2": statistic,
            "df": len(conditions) - 1,
            "p_value": p_value,
            "kendalls_w": kendalls_w,
            "n_complete_terms": len(wide),
            "n_total_terms": len(wide_all),
            "n_excluded_incomplete_or_unsure": excluded_terms,
            "unit_of_analysis": "term-condition mean across experts",
        }
    else:
        no_within_term_differences = wide.eq(wide["A"], axis=0).all().all()
        try:
            if no_within_term_differences:
                statistic, p_value = 0.0, 1.0
            else:
                friedman = stats.friedmanchisquare(*(wide[condition] for condition in conditions))
                statistic, p_value = float(friedman.statistic), float(friedman.pvalue)
        except ValueError:
            statistic, p_value = 0.0, 1.0
        if not np.isfinite(statistic) or not np.isfinite(p_value):
            statistic, p_value = 0.0, 1.0
        kendalls_w = statistic / (len(wide) * (len(conditions) - 1))
        omnibus = {
            "performed": True,
            "chi2": round(statistic, 6),
            "df": len(conditions) - 1,
            "p_value": p_value,
            "kendalls_w": round(float(kendalls_w), 4),
            "n_complete_terms": len(wide),
            "n_total_terms": len(wide_all),
            "n_excluded_incomplete_or_unsure": excluded_terms,
            "unit_of_analysis": "term-condition mean across experts",
        }

    posthoc = []
    if p_value < 0.05:
        raw_p_values = []
        for comparator in ("B", "C", "D"):
            differences = (wide["A"] - wide[comparator]).to_numpy(dtype=float)
            nonzero = differences[differences != 0]
            if len(nonzero):
                test = stats.wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided")
                test_statistic, test_p = float(test.statistic), float(test.pvalue)
            else:
                test_statistic, test_p = 0.0, 1.0
            raw_p_values.append(test_p)
            posthoc.append({
                "comparison": f"A vs {comparator}",
                "W": test_statistic,
                "p_value": test_p,
                "mean_difference": round(float(differences.mean()), 4),
                "rank_biserial": round(float(_rank_biserial(differences)), 4),
                "n_nonzero_pairs": int(len(nonzero)),
                "nonzero_fraction": round(len(nonzero) / len(wide), 4),
                "sparse_contrast_warning": bool(
                    len(nonzero) < 10 or len(nonzero) / len(wide) < 0.10
                ),
            })
        for row, adjusted in zip(posthoc, _holm_adjust(raw_p_values)):
            row["p_value_holm"] = float(adjusted)
            row["significant"] = bool(adjusted < 0.05)
    else:
        posthoc = [{
            "performed": False,
            "reason": "Friedman omnibus test was not significant at alpha=0.05",
        }]

    assignment_ratings = category.drop_duplicates(["Assignment_ID", "Expert"])
    agreement = _fleiss_kappa(
        assignment_ratings,
        item_column="Assignment_ID",
        raw_column="Correct_Raw",
        categories=CORRECTNESS_CHOICES,
    )
    return {
        "correctness_by_condition": summaries,
        "friedman": omnibus,
        "posthoc_wilcoxon_holm": posthoc,
        "assignment_agreement": agreement,
    }


def _final_outcome(
    frame: pd.DataFrame,
    raw_column: str,
    allowed: tuple[str, ...],
    bootstrap_iterations: int,
    seed: int,
) -> dict:
    return _summarize_judgments(
        frame,
        item_column="Row_ID",
        raw_column=raw_column,
        allowed=allowed,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
        partial="partial" in allowed,
    )


def analyze_final_ontology(
    final_frames: dict[str, pd.DataFrame],
    bootstrap_iterations: int,
    seed: int,
) -> dict:
    """Analyze each final-ontology task independently."""
    taxonomy = final_frames["Taxonomy"]
    defined = final_frames["Defined_Classes"]
    relations = final_frames["Relations"]
    individuals = final_frames["Individuals"]
    decisions = final_frames["Critic_Decisions"]

    results = {
        "taxonomy": {
            "relationship_correctness": _final_outcome(
                taxonomy,
                "Relationship_Correct (Yes/Partial/No/Unsure)",
                CORRECTNESS_CHOICES,
                bootstrap_iterations,
                seed + 10,
            ),
            "useful_presalt_distinction": _final_outcome(
                taxonomy,
                "Useful_PreSalt_Distinction (Yes/No/Unsure)",
                BINARY_CHOICES,
                bootstrap_iterations,
                seed + 20,
            ),
        },
        "defined_classes": {
            "definition_correctness": _final_outcome(
                defined,
                "Definition_Correct (Yes/Partial/No/Unsure)",
                CORRECTNESS_CHOICES,
                bootstrap_iterations,
                seed + 30,
            ),
            "presalt_scope_support": _final_outcome(
                defined,
                "Broadly_True_in_PreSalt (Yes/No/Unsure)",
                BINARY_CHOICES,
                bootstrap_iterations,
                seed + 40,
            ),
        },
        "relations": {
            "statement_correctness": _final_outcome(
                relations,
                "Statement_Correct (Yes/Partial/No/Unsure)",
                CORRECTNESS_CHOICES,
                bootstrap_iterations,
                seed + 50,
            ),
            "general_scope_support": _final_outcome(
                relations,
                "Generally_True (Yes/No/Unsure)",
                BINARY_CHOICES,
                bootstrap_iterations,
                seed + 60,
            ),
        },
        "individuals": {
            "named_entity_correctness": _final_outcome(
                individuals,
                "Specific_Named_Entity (Yes/No/Unsure)",
                BINARY_CHOICES,
                bootstrap_iterations,
                seed + 70,
            ),
            "type_correctness": _final_outcome(
                individuals,
                "Type_Correct (Yes/Partial/No/Unsure)",
                CORRECTNESS_CHOICES,
                bootstrap_iterations,
                seed + 80,
            ),
        },
        "critic_decisions": {},
    }
    for offset, decision_type in enumerate(sorted(decisions["Decision_Type"].dropna().unique())):
        subset = decisions[decisions["Decision_Type"] == decision_type]
        summary = _final_outcome(
            subset,
            "Agree (Yes/Partial/No/Unsure)",
            CORRECTNESS_CHOICES,
            bootstrap_iterations,
            seed + 90 + offset * 10,
        )
        treatments = (
            subset["Preferred_Treatment"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        treatments = treatments[treatments != ""]
        summary["preferred_treatment_distribution"] = {
            key: int(value)
            for key, value in treatments.value_counts().sort_index().to_dict().items()
        }
        results["critic_decisions"][str(decision_type)] = summary
    return results


def run_modular_analysis(
    workbook_paths: list[str],
    key_path: str,
    output_dir: str,
    bootstrap_iterations: int | None = None,
    seed: int = 42,
) -> dict:
    """Run the complete Layer 2 analysis and save unblinded audit tables."""
    if len(workbook_paths) < 2:
        raise ValueError("Layer 2 agreement analysis requires at least two expert workbooks")
    if not os.path.exists(key_path):
        raise FileNotFoundError(f"Blinding key not found: {key_path}")
    os.makedirs(output_dir, exist_ok=True)
    if bootstrap_iterations is None:
        bootstrap_iterations = int(os.environ.get("EXPERT_BOOTSTRAP_ITERATIONS", 5000))
    if bootstrap_iterations < 1:
        raise ValueError(
            f"EXPERT_BOOTSTRAP_ITERATIONS must be positive; got {bootstrap_iterations}"
        )

    experts = load_completed_workbooks(workbook_paths)
    source_manifest_path = _validate_source_manifest(key_path)
    key = read_csv(key_path)
    if set(experts) != set(key["Expert_ID"].dropna().astype(str).unique()):
        raise ValueError(
            f"Workbook/key expert mismatch: workbooks={sorted(experts)}, "
            f"key={sorted(key['Expert_ID'].dropna().astype(str).unique())}"
        )

    representation = unblind_representation(experts, key)
    categories = unblind_categories(experts, key)
    final_frames = {
        sheet: collect_final_sheet(experts, key, sheet)
        for sheet in ("Taxonomy", "Defined_Classes", "Relations", "Individuals", "Critic_Decisions")
    }

    write_csv(representation, os.path.join(output_dir, "layer2_representation_unblinded.csv"))
    write_csv(categories, os.path.join(output_dir, "layer2_categories_unblinded.csv"))
    for sheet, frame in final_frames.items():
        filename = f"layer2_{sheet.lower()}_unblinded.csv"
        write_csv(frame, os.path.join(output_dir, filename))

    results = {
        "analysis_design": {
            "n_experts": len(experts),
            "bootstrap_iterations": bootstrap_iterations,
            "seed": seed,
            "source_manifest": source_manifest_path,
            "source_hashes_validated": True,
            "inference_unit": "sampled item after averaging expert ratings",
            "final_ontology_composite_score": False,
        },
        "representation": analyze_representation(
            representation,
            bootstrap_iterations,
            seed,
        ),
        "category_correctness": analyze_categories(
            categories,
            bootstrap_iterations,
            seed + 100,
        ),
        "final_ontology": analyze_final_ontology(
            final_frames,
            bootstrap_iterations,
            seed + 200,
        ),
    }
    results_path = os.path.join(output_dir, "layer2_results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, allow_nan=False)

    print(f"\nLayer 2 analysis: {len(experts)} experts")
    nld = results["representation"]["nld_quality"]
    print(
        f"  NLD A-B: W={nld['wilcoxon']['W']}, "
        f"p={nld['wilcoxon']['p_value']}, "
        f"rank-biserial={nld['wilcoxon']['rank_biserial']}"
    )
    category = results["category_correctness"]["friedman"]
    print(
        f"  Categories: chi2({category['df']})={category['chi2']}, "
        f"p={category['p_value']}, W={category['kendalls_w']}"
    )
    print("  Final ontology tasks reported separately (no composite score).")
    print(f"  Results: {results_path}")
    return results