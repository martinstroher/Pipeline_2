"""Append-mode CSV checkpointing for resumable pipeline steps.

Centralises the load/append pattern used across `nld_generator`,
`relation_extractor`, `cq_scorer`, `ablation_study`, and `category_assigner`.

Usage:
    cp = Checkpoint("output/4_nld.csv", key_column="Term")
    completed, rows = cp.load()
    for term in pending:
        row = _process(term)
        rows.append(row)
        cp.append(row, is_first=(len(rows) == 1))
"""
from __future__ import annotations

import os

import pandas as pd


class Checkpoint:
    def __init__(self, path: str, key_column: str = "Term"):
        self.path = path
        self.key_column = key_column

    def load(self) -> tuple[set, list[dict]]:
        if not os.path.exists(self.path):
            return set(), []
        try:
            df = pd.read_csv(self.path, encoding="utf-8-sig")
            return set(df[self.key_column].unique()), df.to_dict("records")
        except Exception:
            return set(), []

    def append(self, row: dict, is_first: bool | None = None) -> None:
        if is_first is None:
            is_first = not os.path.exists(self.path)
        pd.DataFrame([row]).to_csv(
            self.path, mode="a", header=is_first, index=False, encoding="utf-8-sig"
        )

    def append_batch(self, rows: list[dict], is_first: bool | None = None) -> None:
        if not rows:
            return
        if is_first is None:
            is_first = not os.path.exists(self.path)
        pd.DataFrame(rows).to_csv(
            self.path, mode="a", header=is_first, index=False, encoding="utf-8-sig"
        )
