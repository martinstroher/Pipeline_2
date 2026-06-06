"""Aggregator — combines rule + filter verdicts into final taxonomy and relations.

Reads `validate_rule_verdicts.csv` and `validate_filter_verdicts.csv`,
applies the aggregator policy declared in `src/validate/rules.yaml`,
and emits the cleaned outputs the OWL exporter consumes.

Decision policy (rules.yaml `aggregator.policy`, top-to-bottom; first
match wins, applied per subject):
  1. ≥1 deterministic rule FAIL with decision_on_fail=REJECT → drop_edge
  2. ≥1 deterministic rule FAIL with decision_on_fail=RECLASSIFY
                                                → reparent_to_category_root
  3. ≥1 deterministic rule FAIL with decision_on_fail=REFINE
                                                → refine_to_subclass
  4. ≥2 LLM rule FAIL with decision_on_fail=REJECT → drop_edge
  5. default: accept

Action semantics:
  drop_edge
    - on edge_taxonomy: reparent child to its Category root (term preserved)
    - on edge_relation: remove the relation row
    - on term:          remove the term + any relations referencing it
  reparent_to_category_root
    - on edge_taxonomy: Parent_Term ← Category root
  refine_to_subclass
    - on term: Category ← evidence_json["new_category"];
               Parent_Term ← new category if currently equals old root

Domain-filter actions are applied AFTER rule actions, using the
filter's declared `action` (REMOVE | MERGE | REPARENT). MERGE on a
near_duplicate cluster collapses all non-surviving terms to the
surviving term across both taxonomy and relations.

Outputs (in `output_dir`):
  - validate_taxonomy.csv          final taxonomy
  - validate_relations.csv         final relations
  - validate_log.csv               audit trail
  - validate_per_condition_stats.csv summary counters
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils import log
from src.utils.csv_io import read_csv, write_csv


_DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "rules.yaml"


# ─── Policy loader ─────────────────────────────────────────────────────


def _load_aggregator_policy() -> list[dict]:
    path = Path(os.environ.get("VALIDATION_RULES_PATH", _DEFAULT_RULES_PATH))
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return list(cfg.get("aggregator", {}).get("policy", []))


# ─── Verdict indexing ──────────────────────────────────────────────────


def _index_rule_verdicts(rule_df: pd.DataFrame) -> dict[str, list[dict]]:
    """Group rule verdicts by subject_id, keeping only FAIL rows."""
    by_subject: dict[str, list[dict]] = {}
    if rule_df.empty:
        return by_subject
    fails = rule_df[rule_df["verdict"] == "FAIL"]
    for _, r in fails.iterrows():
        sid = str(r["subject_id"])
        by_subject.setdefault(sid, []).append(r.to_dict())
    return by_subject


def _index_filter_verdicts(filter_df: pd.DataFrame) -> dict[str, list[dict]]:
    by_subject: dict[str, list[dict]] = {}
    if filter_df.empty:
        return by_subject
    fails = filter_df[filter_df["verdict"] == "FAIL"]
    for _, r in fails.iterrows():
        sid = str(r["subject_id"])
        by_subject.setdefault(sid, []).append(r.to_dict())
    return by_subject


# ─── Subject IDs ───────────────────────────────────────────────────────


def _edge_tax_id(row: pd.Series) -> str:
    return f"{str(row.get('Term', '')).strip()}-->{str(row.get('Parent_Term', '')).strip()}"


def _edge_rel_id(row: pd.Series) -> str:
    return (
        f"{str(row.get('Term', '')).strip()}--"
        f"{str(row.get('Property', '')).strip()}-->"
        f"{str(row.get('Filler', '')).strip()}"
    )


# ─── Policy matching ───────────────────────────────────────────────────


def _select_action(verdicts: list[dict], policy: list[dict]) -> tuple[str, dict | None]:
    """Walk policy clauses; return (action, triggering_verdict).

    If `triggering_verdict` is None and action is "accept", no action
    is taken. Otherwise the verdict whose evidence drove the action is
    returned so the caller can read evidence_json["new_category"] etc.
    """
    if not verdicts:
        return "accept", None
    for clause in policy:
        if clause.get("default") == "accept":
            return "accept", None
        cond = clause.get("if") or {}
        eng = cond.get("engine")
        status = (cond.get("status") or "").upper()
        min_count = int(cond.get("min_count", 1))
        matching = [
            v for v in verdicts
            if (eng is None or v.get("engine") == eng)
            and str(v.get("decision_on_fail", "")).upper() == status
        ]
        if len(matching) >= min_count:
            return clause["action"], matching[0]
    return "accept", None


# ─── Action applicators ────────────────────────────────────────────────


def _parse_evidence(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _apply_rule_actions(
    tax_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    rule_index: dict[str, list[dict]],
    policy: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Apply rule-level actions to taxonomy + relations. Returns (tax, rel, log_rows)."""
    log_rows: list[dict] = []
    tax = tax_df.copy().reset_index(drop=True)
    rel = rel_df.copy().reset_index(drop=True) if not rel_df.empty else rel_df

    # 1. Taxonomy edges: subject_id is "term-->parent"
    tax_keep_mask = pd.Series([True] * len(tax))
    for idx, row in tax.iterrows():
        sid = _edge_tax_id(row)
        action, src = _select_action(rule_index.get(sid, []), policy)
        if action == "accept" or src is None:
            continue
        ev = _parse_evidence(src.get("evidence_json"))
        cat = str(row.get("Category", "")).strip()
        if action == "drop_edge":
            # Preserve term, sever bad parent link.
            tax.at[idx, "Parent_Term"] = cat
            tax.at[idx, "Is_Intermediate"] = False
            log_rows.append({
                "action": "drop_edge",
                "subject_type": "edge_taxonomy",
                "subject_id": sid,
                "rule_id": src.get("rule_id"),
                "detail": f"Reparented to category root '{cat}'.",
            })
        elif action == "reparent_to_category_root":
            tax.at[idx, "Parent_Term"] = cat
            tax.at[idx, "Is_Intermediate"] = False
            log_rows.append({
                "action": "reparent_to_category_root",
                "subject_type": "edge_taxonomy",
                "subject_id": sid,
                "rule_id": src.get("rule_id"),
                "detail": f"Reparented to category root '{cat}'.",
            })
        elif action == "refine_to_subclass":
            new_cat = ev.get("new_category")
            if new_cat:
                tax.at[idx, "Category"] = new_cat
                # If the row was attached directly to the old category root, retarget the parent too.
                if str(row.get("Parent_Term", "")).strip().lower() == cat.lower():
                    tax.at[idx, "Parent_Term"] = new_cat
                log_rows.append({
                    "action": "refine_to_subclass",
                    "subject_type": "term",
                    "subject_id": str(row.get("Term", "")).strip(),
                    "rule_id": src.get("rule_id"),
                    "detail": f"Category {cat} → {new_cat}.",
                })

    # 2. Relation edges: subject_id is "term--prop-->filler"
    if not rel.empty:
        rel_keep_mask = pd.Series([True] * len(rel))
        for idx, row in rel.iterrows():
            sid = _edge_rel_id(row)
            action, src = _select_action(rule_index.get(sid, []), policy)
            if action == "drop_edge":
                rel_keep_mask.iloc[idx] = False
                log_rows.append({
                    "action": "drop_edge",
                    "subject_type": "edge_relation",
                    "subject_id": sid,
                    "rule_id": (src or {}).get("rule_id", ""),
                    "detail": "Relation removed.",
                })
        rel = rel[rel_keep_mask].reset_index(drop=True)

    # 3. Term-level rule actions (cq_coverage_gate, term-targeted relation_evidence_refinement)
    drop_terms: set[str] = set()
    for sid, verdicts in rule_index.items():
        if "-->" in sid or "--" in sid and "-->" in sid:
            continue  # already handled as edges
        action, src = _select_action(verdicts, policy)
        if action == "drop_edge":
            drop_terms.add(sid.lower())
            log_rows.append({
                "action": "drop_term",
                "subject_type": "term",
                "subject_id": sid,
                "rule_id": (src or {}).get("rule_id", ""),
                "detail": "Term + dependent relations removed.",
            })
        elif action == "refine_to_subclass":
            ev = _parse_evidence((src or {}).get("evidence_json"))
            new_cat = ev.get("new_category")
            if new_cat:
                mask = tax["Term"].str.lower() == sid.lower()
                old_cats = tax.loc[mask, "Category"].unique().tolist()
                tax.loc[mask, "Category"] = new_cat
                log_rows.append({
                    "action": "refine_to_subclass",
                    "subject_type": "term",
                    "subject_id": sid,
                    "rule_id": (src or {}).get("rule_id", ""),
                    "detail": f"Category {old_cats} → {new_cat}.",
                })

    if drop_terms:
        tax = tax[~tax["Term"].str.lower().isin(drop_terms)].reset_index(drop=True)
        if not rel.empty:
            keep = ~(
                rel["Term"].str.lower().isin(drop_terms)
                | rel["Filler"].str.lower().isin(drop_terms)
            )
            rel = rel[keep].reset_index(drop=True)

    return tax, rel, log_rows


def _apply_filter_actions(
    tax_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    filter_index: dict[str, list[dict]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Apply filter-level actions to taxonomy + relations."""
    log_rows: list[dict] = []
    tax = tax_df.copy().reset_index(drop=True)
    rel = rel_df.copy().reset_index(drop=True) if not rel_df.empty else rel_df

    drop_terms: set[str] = set()
    rename_map: dict[str, str] = {}

    # Process clusters first (MERGE) — they may target multiple terms.
    for sid, verdicts in filter_index.items():
        for v in verdicts:
            action = str(v.get("action", "")).upper()
            ev = _parse_evidence(v.get("evidence_json"))
            if action == "MERGE":
                survivor = ev.get("surviving_term")
                members = ev.get("members") or ev.get("merged_terms") or []
                if survivor and members:
                    for m in members:
                        if m.lower() != survivor.lower():
                            rename_map[m.lower()] = survivor
                    log_rows.append({
                        "action": "merge",
                        "subject_type": "term_cluster",
                        "subject_id": sid,
                        "rule_id": v.get("filter_id"),
                        "detail": f"Merged {members} → '{survivor}'.",
                    })
            elif action == "REMOVE":
                drop_terms.add(str(sid).lower())
                log_rows.append({
                    "action": "remove",
                    "subject_type": v.get("subject_type", "term"),
                    "subject_id": sid,
                    "rule_id": v.get("filter_id"),
                    "detail": v.get("reason", "Removed by domain filter."),
                })
            elif action == "REPARENT":
                # subject_id is "term-->parent"; reparent to Category root.
                term = sid.split("-->", 1)[0]
                mask = tax["Term"].str.lower() == term.lower()
                if mask.any():
                    new_parent = tax.loc[mask, "Category"].iloc[0]
                    tax.loc[mask, "Parent_Term"] = new_parent
                    tax.loc[mask, "Is_Intermediate"] = False
                    log_rows.append({
                        "action": "reparent",
                        "subject_type": "edge_taxonomy",
                        "subject_id": sid,
                        "rule_id": v.get("filter_id"),
                        "detail": f"Reparented to category root '{new_parent}'.",
                    })

    # Apply rename_map across taxonomy + relations
    if rename_map:
        def _rename(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            return rename_map.get(value.lower(), value)
        for col in ("Term", "Parent_Term"):
            if col in tax.columns:
                tax[col] = tax[col].map(_rename)
        if not rel.empty:
            for col in ("Term", "Filler"):
                if col in rel.columns:
                    rel[col] = rel[col].map(_rename)
        # Deduplicate identical rows that may result from the merge
        tax = tax.drop_duplicates().reset_index(drop=True)
        if not rel.empty:
            rel = rel.drop_duplicates().reset_index(drop=True)

    # Apply drop_terms (after rename, in case a renamed term is also removed)
    if drop_terms:
        tax = tax[~tax["Term"].str.lower().isin(drop_terms)].reset_index(drop=True)
        if not rel.empty:
            keep = ~(
                rel["Term"].str.lower().isin(drop_terms)
                | rel["Filler"].str.lower().isin(drop_terms)
            )
            rel = rel[keep].reset_index(drop=True)

    return tax, rel, log_rows


# ─── Per-rule stats ────────────────────────────────────────────────────


def _summarise(rule_df: pd.DataFrame, filter_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if not rule_df.empty:
        for rule_id, g in rule_df.groupby("rule_id"):
            counts = g["verdict"].value_counts().to_dict()
            rows.append({
                "kind": "rule",
                "id": rule_id,
                "engine": g["engine"].iloc[0] if len(g) else "",
                "n_total": len(g),
                "n_pass": int(counts.get("PASS", 0)),
                "n_fail": int(counts.get("FAIL", 0)),
                "n_abstain": int(counts.get("ABSTAIN", 0)),
            })
    if not filter_df.empty:
        for filter_id, g in filter_df.groupby("filter_id"):
            counts = g["verdict"].value_counts().to_dict()
            rows.append({
                "kind": "filter",
                "id": filter_id,
                "engine": g["engine"].iloc[0] if len(g) else "",
                "n_total": len(g),
                "n_pass": int(counts.get("PASS", 0)),
                "n_fail": int(counts.get("FAIL", 0)),
                "n_abstain": int(counts.get("ABSTAIN", 0)),
            })
    return pd.DataFrame(rows, columns=[
        "kind", "id", "engine", "n_total", "n_pass", "n_fail", "n_abstain",
    ])


# ─── Main entry point ──────────────────────────────────────────────────


def run_aggregator(
    taxonomy_csv: str,
    rule_verdicts_csv: str,
    filter_verdicts_csv: str,
    output_dir: str,
    *,
    relations_csv: str | None = None,
) -> tuple[str, str | None, str, str]:
    """Apply aggregator policy + filter actions; write cleaned outputs.

    Returns (taxonomy_path, relations_path_or_None, log_path, stats_path).
    """
    policy = _load_aggregator_policy()
    if not policy:
        log.warn("Aggregator: empty policy in YAML; nothing will be applied.")

    tax_df = read_csv(taxonomy_csv)
    rel_df = read_csv(relations_csv) if relations_csv and os.path.exists(relations_csv) else pd.DataFrame()
    rule_df = read_csv(rule_verdicts_csv) if os.path.exists(rule_verdicts_csv) else pd.DataFrame()
    filter_df = read_csv(filter_verdicts_csv) if os.path.exists(filter_verdicts_csv) else pd.DataFrame()

    rule_index = _index_rule_verdicts(rule_df)
    filter_index = _index_filter_verdicts(filter_df)

    tax_after_rules, rel_after_rules, rule_log = _apply_rule_actions(
        tax_df, rel_df, rule_index, policy
    )
    tax_final, rel_final, filter_log = _apply_filter_actions(
        tax_after_rules, rel_after_rules, filter_index
    )

    tax_out = os.path.join(output_dir, "validate_taxonomy.csv")
    write_csv(tax_final, tax_out)

    rel_out: str | None = None
    if not rel_df.empty:
        rel_out = os.path.join(output_dir, "validate_relations.csv")
        write_csv(rel_final, rel_out)

    log_out = os.path.join(output_dir, "validate_log.csv")
    write_csv(
        pd.DataFrame(
            rule_log + filter_log,
            columns=["action", "subject_type", "subject_id", "rule_id", "detail"],
        ),
        log_out,
    )

    stats_out = os.path.join(output_dir, "validate_per_condition_stats.csv")
    write_csv(_summarise(rule_df, filter_df), stats_out)

    log.success(
        f"Aggregator: taxonomy {len(tax_df)}→{len(tax_final)}, "
        f"relations {len(rel_df)}→{len(rel_final)}, "
        f"{len(rule_log) + len(filter_log)} actions"
    )
    return tax_out, rel_out, log_out, stats_out
