"""Thin pandas CSV wrappers with project defaults.

All CSV reads/writes in the pipeline are BOM-safe (`utf-8-sig`) and writes
omit the pandas index by default. These helpers eliminate the per-callsite
repetition without changing semantics. Pass any extra pandas kwargs
through; explicit kwargs always win over the defaults.
"""
from __future__ import annotations

import pandas as pd


def read_csv(path: str, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("encoding", "utf-8-sig")
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
    kwargs.setdefault("encoding", "utf-8-sig")
    kwargs.setdefault("index", False)
    df.to_csv(path, **kwargs)
