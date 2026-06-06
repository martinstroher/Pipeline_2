"""Rule applier — dispatches every rule in `src/validate/rules.yaml`.

Reads the YAML config, iterates the active rule set, dispatches each rule
to the appropriate engine (structural or LLM), and writes one row per
(rule_id, subject_id) to `validate_rule_verdicts.csv`.

The aggregator (Phase 2.6) reads this CSV plus
`validate_filter_verdicts.csv` (from `domain_filters.py`) to compute
the final cleaned taxonomy and relations.

Active rule selection:
  - VALIDATION_RULES_PATH overrides the default `src/validate/rules.yaml`.
  - VALIDATION_RULES_ACTIVE is a comma-separated subset of rule ids; if
    set, only those rules run.

Output schema (validate_rule_verdicts.csv):
  rule_id, engine, subject_type, subject_id, verdict, reason,
  decision_on_fail, evidence_json
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from src.utils import log
from src.utils.csv_io import read_csv, write_csv
from src.validate.engines import llm as llm_engine
from src.validate.engines.structural import (
    RULE_FUNCTIONS,
    StructuralContext,
    Verdict,
)


_DEFAULT_RULES_PATH = Path(__file__).resolve().parents[1] / "validate" / "rules.yaml"


# ─── Config loading ────────────────────────────────────────────────────


def _load_rules() -> tuple[list[dict], dict]:
    path = Path(os.environ.get("VALIDATION_RULES_PATH", _DEFAULT_RULES_PATH))
    if not path.exists():
        raise RuntimeError(f"Validation rules YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rules = list(cfg.get("rules", []))
    active = os.environ.get("VALIDATION_RULES_ACTIVE", "").strip()
    if active:
        wanted = {r.strip() for r in active.split(",") if r.strip()}
        rules = [r for r in rules if r["id"] in wanted]
        log.info(f"Rule applier: active subset = {sorted(wanted)}")
    return rules, cfg.get("aggregator", {})


# ─── Subject builders ──────────────────────────────────────────────────


def _build_taxonomy_edges(tax_df: pd.DataFrame) -> list[dict]:
    """One dict per (Term, Parent_Term) row in the taxonomy CSV."""
    cat_to_term: dict[str, str] = {}
    edges: list[dict] = []
    # Build a lookup so parent_category can be filled when parent is a Term.
    term_to_cat = {
        str(r["Term"]).strip().lower(): str(r.get("Category", "")).strip()
        for _, r in tax_df.iterrows()
    }
    for _, r in tax_df.iterrows():
        term = str(r.get("Term", "")).strip()
        parent = str(r.get("Parent_Term", "")).strip()
        category = str(r.get("Category", "")).strip()
        parent_cat = term_to_cat.get(parent.lower(), parent)  # parent IS a category if not a Term
        edges.append({
            "term": term,
            "parent_term": parent,
            "relationship_type": str(r.get("Relationship_Type", "")).strip(),
            "is_intermediate": bool(r.get("Is_Intermediate", False)),
            "category": category,
            "parent_category": parent_cat,
            "nld": str(r.get("NLD", "") or ""),
        })
    return edges


def _build_relation_edges(rel_df: pd.DataFrame, tax_df: pd.DataFrame) -> list[dict]:
    """One dict per ACCEPTED row in the relations CSV. Already-rejected
    rows are skipped (they failed extraction's deterministic check).

    Filler_Category is not persisted in construct_relations.csv (it was
    only known at extraction time), so we re-derive it from `tax_df` by
    matching the Filler to a Term. Unmapped fillers get an empty string
    and the per-rule engine handles that case (typically ABSTAIN).
    """
    if rel_df.empty:
        return []
    if "Validation_Status" in rel_df.columns:
        accepted = rel_df[rel_df["Validation_Status"].str.upper() == "ACCEPTED"]
    else:
        accepted = rel_df
    term_to_cat = {
        str(r["Term"]).strip().lower(): str(r.get("Category", "")).strip()
        for _, r in tax_df.iterrows()
    }
    return [
        {
            "term": str(r["Term"]).strip(),
            "property": str(r.get("Property", "")).strip(),
            "filler": str(r.get("Filler", "")).strip(),
            "category": str(r.get("Category", "")).strip(),
            "filler_category": term_to_cat.get(
                str(r.get("Filler", "")).strip().lower(), ""
            ),
            "evidence": str(r.get("Evidence", "")).strip(),
        }
        for _, r in accepted.iterrows()
    ]


def _build_terms(tax_df: pd.DataFrame) -> list[dict]:
    """One dict per unique Term in the taxonomy."""
    seen: dict[str, dict] = {}
    for _, r in tax_df.iterrows():
        term = str(r.get("Term", "")).strip()
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen[key] = {
            "term": term,
            "category": str(r.get("Category", "")).strip(),
            "nld": str(r.get("NLD", "") or ""),
            "is_intermediate": bool(r.get("Is_Intermediate", False)),
        }
    return list(seen.values())


# ─── Exemption check ───────────────────────────────────────────────────


def _is_exempt(rule: dict, subject: dict) -> bool:
    """Apply rule.exemptions to a subject. True ⇒ skip the rule for this subject."""
    ex = rule.get("exemptions") or {}
    # Skip rdf:type edges (named individuals) — protect proper names.
    proper = ex.get("proper_names") or {}
    rel_types = {r.strip().lower() for r in proper.get("relationship_types", [])}
    if rel_types and str(subject.get("relationship_type", "")).strip().lower() in rel_types:
        return True
    preserved = {c.strip().lower() for c in ex.get("preserved_categories", [])}
    if preserved and str(subject.get("category", "")).strip().lower() in preserved:
        return True
    return False


# ─── Dispatch ──────────────────────────────────────────────────────────


def _subject_id(target: str, subject: dict) -> str:
    if target == "edges_in_taxonomy":
        return f"{subject.get('term')}-->{subject.get('parent_term')}"
    if target == "edges_in_relations":
        return f"{subject.get('term')}--{subject.get('property')}-->{subject.get('filler')}"
    return str(subject.get("term", ""))


def _subject_type(target: str) -> str:
    return {
        "edges_in_taxonomy": "edge_taxonomy",
        "edges_in_relations": "edge_relation",
        "terms": "term",
    }.get(target, "term")


def _dispatch_deterministic(
    rule: dict, subjects: list[dict], ctx: StructuralContext
) -> list[Verdict]:
    fn = RULE_FUNCTIONS.get(rule["id"])
    if fn is None:
        log.warn(f"Rule applier: no structural function for '{rule['id']}', skipping.")
        return []
    out: list[Verdict] = []
    for subj in subjects:
        if _is_exempt(rule, subj):
            continue
        out.append(fn(subj, ctx))
    return out


def _dispatch_llm(rule: dict, subjects: list[dict], ctx: StructuralContext) -> list[Verdict]:
    prompt = rule.get("prompt")
    if not prompt:
        log.warn(f"Rule applier: LLM rule '{rule['id']}' has no prompt, skipping.")
        return []
    target = rule["target"]
    subj_type = _subject_type(target)
    # Drop exempt subjects up-front so batching only includes evaluable edges.
    active = [s for s in subjects if not _is_exempt(rule, s)]
    if not active:
        return []

    # Build one payload entry per active subject. OntoClean prompts accept a
    # JSON array under {batch_json} and return a JSON array of verdicts.
    parent_nld_lookup: dict[str, str] = {}
    if ctx.taxonomy_df is not None:
        for _, r in ctx.taxonomy_df.iterrows():
            key = str(r.get("Term", "")).strip().lower()
            if key:
                parent_nld_lookup[key] = str(r.get("NLD", "") or "")

    payloads: list[dict] = []
    subject_ids: list[str] = []
    for subj in active:
        subject_ids.append(_subject_id(target, subj))
        payloads.append({
            "child_term": subj.get("term"),
            "parent_term": subj.get("parent_term"),
            "child_nld": subj.get("nld", ""),
            "parent_nld": parent_nld_lookup.get(
                str(subj.get("parent_term", "")).strip().lower(), ""
            ),
            "category": subj.get("category"),
            "parent_category": subj.get("parent_category"),
        })

    batch_size = int(os.environ.get("VALIDATION_LLM_BATCH_SIZE", "10"))
    concurrency = max(1, int(os.environ.get("VALIDATION_LLM_CONCURRENCY", "6")))

    chunks: list[tuple[list[dict], list[str]]] = []
    for start in range(0, len(payloads), batch_size):
        chunks.append((
            payloads[start:start + batch_size],
            subject_ids[start:start + batch_size],
        ))

    def _call(args: tuple[list[dict], list[str]]) -> list[Verdict]:
        chunk, chunk_ids = args
        return llm_engine.evaluate_batch(
            prompt,
            {"batch_json": json.dumps(chunk, ensure_ascii=False)},
            rule_id=rule["id"],
            subject_type=subj_type,
            subject_ids=chunk_ids,
        )

    out: list[Verdict] = []
    desc = f"  {rule['id']} ({len(chunks)} batches×{batch_size})"
    if concurrency <= 1 or len(chunks) <= 1:
        for ch in tqdm(chunks, desc=desc, unit="batch"):
            out.extend(_call(ch))
    else:
        # Pre-warm gemini client so concurrent first-call doesn't race on init.
        from src.utils.gemini_client import get_client
        get_client()
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for result in tqdm(
                ex.map(_call, chunks),
                total=len(chunks),
                desc=f"{desc} x{concurrency}",
                unit="batch",
            ):
                out.extend(result)
    return out


# ─── Main entry point ──────────────────────────────────────────────────


def run_rule_applier(
    taxonomy_csv: str,
    relations_csv: str | None,
    output_dir: str,
    *,
    cq_covered_terms_csv: str | None = None,
) -> str:
    """Apply every active rule to the taxonomy + relations; write verdicts CSV.

    Returns the path to validate_rule_verdicts.csv.
    """
    rules, _ = _load_rules()
    if not rules:
        log.warn("Rule applier: no active rules; producing empty verdicts CSV.")

    tax_df = read_csv(taxonomy_csv)
    rel_df = read_csv(relations_csv) if relations_csv and os.path.exists(relations_csv) else pd.DataFrame()
    accepted = rel_df  # full df; engines filter by Validation_Status internally
    if not rel_df.empty and "Validation_Status" in rel_df.columns:
        accepted = rel_df[rel_df["Validation_Status"].str.upper() == "ACCEPTED"]

    cq_covered: set[str] | None = None
    if cq_covered_terms_csv and os.path.exists(cq_covered_terms_csv):
        cq_df = read_csv(cq_covered_terms_csv)
        cq_covered = {str(t).strip().lower() for t in cq_df.get("Term", []) if str(t).strip()}

    ctx = StructuralContext(
        taxonomy_df=tax_df,
        relations_df=accepted,
        cq_covered_terms=cq_covered,
    )

    tax_edges = _build_taxonomy_edges(tax_df)
    rel_edges = _build_relation_edges(rel_df, tax_df)
    terms = _build_terms(tax_df)

    rows: list[dict[str, Any]] = []
    for rule in rules:
        target = rule.get("target", "terms")
        engine = rule.get("engine", "deterministic")
        if target == "edges_in_taxonomy":
            subjects = tax_edges
        elif target == "edges_in_relations":
            subjects = rel_edges
        elif target == "terms":
            subjects = terms
        else:
            log.warn(f"Rule applier: unknown target '{target}' for rule {rule['id']}")
            continue

        if engine == "deterministic":
            verdicts = _dispatch_deterministic(rule, subjects, ctx)
        elif engine == "llm":
            verdicts = _dispatch_llm(rule, subjects, ctx)
        else:
            log.warn(f"Rule applier: unknown engine '{engine}' for rule {rule['id']}")
            continue

        decision_on_fail = str(rule.get("decision_on_fail", "REJECT"))
        for v in verdicts:
            rows.append({
                "rule_id": v.rule_id,
                "engine": engine,
                "subject_type": v.subject_type,
                "subject_id": v.subject_id,
                "verdict": v.verdict,
                "decision_on_fail": decision_on_fail,
                "reason": v.reason,
                "evidence_json": json.dumps(v.evidence, ensure_ascii=False),
            })
        log.info(
            f"Rule applier: {rule['id']} ({engine}/{target}) → "
            f"{sum(1 for v in verdicts if v.verdict == 'FAIL')} FAIL, "
            f"{sum(1 for v in verdicts if v.verdict == 'PASS')} PASS, "
            f"{sum(1 for v in verdicts if v.verdict == 'ABSTAIN')} ABSTAIN"
        )

    out_path = os.path.join(output_dir, "validate_rule_verdicts.csv")
    out_df = pd.DataFrame(rows, columns=[
        "rule_id", "engine", "subject_type", "subject_id",
        "verdict", "decision_on_fail", "reason", "evidence_json",
    ])
    write_csv(out_df, out_path)
    log.success(f"Rule applier: wrote {len(out_df)} verdicts to {out_path}")
    return out_path
