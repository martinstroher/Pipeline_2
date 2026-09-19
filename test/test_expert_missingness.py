"""Regression tests for matched expert-term handling of Unsure responses."""

import numpy as np
import pandas as pd

from evaluation_study.expert_eval_analysis import (
    analyze_categories,
    analyze_representation,
)


def test_nld_contrast_uses_the_same_experts_for_a_and_b():
    rows = []
    for index in range(12):
        term = f"term-{index:02d}"
        rows.extend([
            {
                "Term": term,
                "Expert": "generous",
                "Relevance": 5.0,
                "Quality_A": 5.0,
                "Quality_B": np.nan,
                "Preference_A": np.nan,
                "Final_Fate": "KEEP",
            },
            {
                "Term": term,
                "Expert": "strict",
                "Relevance": 3.0,
                "Quality_A": 1.0,
                "Quality_B": 1.0,
                "Preference_A": 0.0,
                "Final_Fate": "KEEP",
            },
            {
                "Term": term,
                "Expert": "neutral",
                "Relevance": 3.0,
                "Quality_A": 3.0,
                "Quality_B": 3.0,
                "Preference_A": 0.0,
                "Final_Fate": "KEEP",
            },
        ])

    result = analyze_representation(
        pd.DataFrame(rows),
        bootstrap_iterations=50,
        seed=42,
    )["nld_quality"]

    assert result["quality_A_mean"] == 2.0
    assert result["quality_B_mean"] == 2.0
    assert result["mean_difference_A_minus_B"] == 0.0
    assert result["wilcoxon"]["p_value"] == 1.0
    assert result["n_matched_expert_term_ratings"] == 24


def test_category_omnibus_uses_complete_expert_condition_sets():
    rows = []
    for index in range(12):
        term = f"term-{index:02d}"
        for condition in ("A", "B", "C", "D"):
            rows.extend([
                {
                    "Term": term,
                    "Condition": condition,
                    "Expert": "strict",
                    "Correct_Raw": "no",
                    "Correct_Score": 0.0,
                    "Assignment_ID": f"{term}|{condition}",
                },
                {
                    "Term": term,
                    "Condition": condition,
                    "Expert": "neutral",
                    "Correct_Raw": "partial",
                    "Correct_Score": 0.5,
                    "Assignment_ID": f"{term}|{condition}",
                },
            ])
        rows.append({
            "Term": term,
            "Condition": "A",
            "Expert": "generous",
            "Correct_Raw": "yes",
            "Correct_Score": 1.0,
            "Assignment_ID": f"{term}|A",
        })
        for condition in ("B", "C", "D"):
            rows.append({
                "Term": term,
                "Condition": condition,
                "Expert": "generous",
                "Correct_Raw": "unsure",
                "Correct_Score": np.nan,
                "Assignment_ID": f"{term}|{condition}",
            })

    result = analyze_categories(
        pd.DataFrame(rows),
        bootstrap_iterations=50,
        seed=42,
    )["friedman"]

    assert result["p_value"] == 1.0
    assert result["kendalls_w"] == 0.0
    assert result["n_complete_terms"] == 12
    assert result["n_complete_expert_term_sets"] == 24
