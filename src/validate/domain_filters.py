"""Domain-filter applier — dispatches every filter in `domains/<name>/domain_filters.yaml`.

Mirrors `src/validate/rule_applier.py` but loads the domain-specific
filter set instead of the published rule set. Adds the hybrid engine
path used exclusively by the `near_duplicate` filter (embedding-based
clustering + LLM adjudication on clusters).

Output: `validate_filter_verdicts.csv` with the same schema as
`validate_rule_verdicts.csv` plus an `action` column carrying the
filter-specific decision (REMOVE | MERGE | REPARENT) declared in YAML.
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
from src.validate.engines import hybrid as hybrid_engine
from src.validate.engines import llm as llm_engine
from src.validate.engines.structural import (
    RULE_FUNCTIONS,
    StructuralContext,
    Verdict,
)
from src.validate.rule_applier import (
    _build_taxonomy_edges,
    _build_terms,
    _is_exempt,
)


_DEFAULT_FILTERS_PATH = Path("domains") / "presalt" / "domain_filters.yaml"


# ─── Config loading ────────────────────────────────────────────────────


def _load_filters() -> tuple[list[dict], dict]:
    """Resolve and load `domain_filters.yaml` from the active domain.

    Precedence:
      1. DOMAIN_FILTERS_PATH env var (full path).
      2. <active-domain>/domain_filters.yaml (active domain = parent of
         the ontology_config.yaml the loader is using).
      3. Default `domains/presalt/domain_filters.yaml` (repo root).
    """
    env_path = os.environ.get("DOMAIN_FILTERS_PATH")
    if env_path:
        path = Path(env_path)
    else:
        try:
            from src.utils.ontology_config import get_config
            domain_dir = get_config()._source_path.parent
            candidate = domain_dir / "domain_filters.yaml"
            path = candidate if candidate.exists() else Path.cwd() / _DEFAULT_FILTERS_PATH
        except Exception:
            path = Path.cwd() / _DEFAULT_FILTERS_PATH
    if not path.exists():
        raise RuntimeError(f"Domain filters YAML not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    filters = list(cfg.get("filters", []))
    active = os.environ.get("VALIDATION_FILTERS_ACTIVE", "").strip()
    if active:
        wanted = {r.strip() for r in active.split(",") if r.strip()}
        filters = [r for r in filters if r["id"] in wanted]
        log.info(f"Domain filters: active subset = {sorted(wanted)}")
    return filters, cfg.get("exemptions", {})


# ─── Subject builders shared with rule_applier ─────────────────────────
# (taxonomy_edges and terms reused via import; relations subject is not
# used by any current filter, so no _build_relation_edges here.)


# ─── Filter-level exemption check ──────────────────────────────────────


def _filter_is_exempt(global_ex: dict, subject: dict) -> bool:
    """Apply the global filter exemptions block to a subject."""
    proper = global_ex.get("proper_names") or {}
    rel_types = {r.strip().lower() for r in proper.get("relationship_types", [])}
    if rel_types and str(subject.get("relationship_type", "")).strip().lower() in rel_types:
        return True
    preserved = {c.strip().lower() for c in global_ex.get("preserved_categories", [])}
    if preserved and str(subject.get("term", "")).strip().lower() in preserved:
        return True
    if preserved and str(subject.get("category", "")).strip().lower() in preserved:
        return True
    return False


# ─── Dispatch helpers ──────────────────────────────────────────────────


def _subject_id(target: str, subject: dict) -> str:
    if target == "taxonomy":
        return f"{subject.get('term')}-->{subject.get('parent_term')}"
    return str(subject.get("term", ""))


def _subject_type(target: str) -> str:
    return {"taxonomy": "edge_taxonomy", "terms": "term", "relations": "edge_relation"}.get(
        target, "term"
    )


def _dispatch_structural(
    flt: dict, subjects: list[dict], ctx: StructuralContext, global_ex: dict
) -> list[Verdict]:
    fn = RULE_FUNCTIONS.get(flt["id"])
    if fn is None:
        log.warn(f"Domain filters: no structural function for '{flt['id']}', skipping.")
        return []
    return [fn(s, ctx) for s in subjects if not _filter_is_exempt(global_ex, s)]


def _dispatch_llm_filter(
    flt: dict, subjects: list[dict], ctx: StructuralContext, global_ex: dict
) -> list[Verdict]:
    """LLM filters operate on terms. Batched JSON payload."""
    prompt = flt.get("prompt")
    if not prompt:
        log.warn(f"Domain filters: LLM filter '{flt['id']}' has no prompt, skipping.")
        return []
    active_subjects = [s for s in subjects if not _filter_is_exempt(global_ex, s)]
    if not active_subjects:
        return []
    # Build a child-count map once for filters that care about it (overly_generic).
    children_count: dict[str, int] = {}
    if ctx.taxonomy_df is not None:
        for _, r in ctx.taxonomy_df.iterrows():
            p = str(r.get("Parent_Term", "")).strip().lower()
            if p:
                children_count[p] = children_count.get(p, 0) + 1

    batch = [
        {
            "term": s["term"],
            "nld": s.get("nld", ""),
            "category": s.get("category", ""),
            "child_count": children_count.get(s["term"].lower(), 0),
        }
        for s in active_subjects
    ]
    subject_ids = [s["term"] for s in active_subjects]

    return llm_engine.evaluate_batch(
        prompt,
        {"batch_json": json.dumps(batch, ensure_ascii=False)},
        rule_id=flt["id"],
        subject_type=_subject_type(flt.get("target", "terms")),
        subject_ids=subject_ids,
    )


def _dispatch_hybrid(
    flt: dict, subjects: list[dict], ctx: StructuralContext, global_ex: dict
) -> list[Verdict]:
    """Hybrid filter (near_duplicate): embedding-cluster terms, then LLM adjudication.

    Heavy: loads BGE-M3 weights. Callers can skip this filter via
    `VALIDATION_FILTERS_ACTIVE` when running cheap smoke tests.
    """
    prompt = flt.get("prompt")
    if not prompt:
        log.warn(f"Domain filters: hybrid filter '{flt['id']}' has no prompt, skipping.")
        return []
    params = flt.get("params") or {}
    model_name = params.get("embedding_model", "BAAI/bge-m3")
    threshold = float(params.get("similarity_threshold", 0.92))
    max_size = int(params.get("max_cluster_size", 4))

    active_subjects = [s for s in subjects if not _filter_is_exempt(global_ex, s)]
    if len(active_subjects) < 2:
        return []
    term_names = [s["term"] for s in active_subjects]

    log.info(
        f"Hybrid filter '{flt['id']}': embedding {len(term_names)} terms "
        f"({model_name}, sim≥{threshold}, max_cluster={max_size})"
    )
    embeddings = hybrid_engine.embed_terms(term_names, model_name=model_name)
    clusters = hybrid_engine.cluster_terms_by_similarity(
        term_names, embeddings, threshold=threshold, max_cluster_size=max_size
    )
    log.info(f"Hybrid filter '{flt['id']}': {len(clusters)} candidate clusters above threshold")
    if not clusters:
        return []

    # Build a per-cluster payload for LLM adjudication.
    by_term = {s["term"]: s for s in active_subjects}
    batch = [
        {
            "cluster_id": c.cluster_id,
            "terms": [
                {
                    "term": m,
                    "nld": by_term[m].get("nld", ""),
                    "category": by_term[m].get("category", ""),
                }
                for m in c.members
            ],
        }
        for c in clusters
    ]
    cluster_ids = [c.cluster_id for c in clusters]
    verdicts = llm_engine.evaluate_batch(
        prompt,
        {"batch_json": json.dumps(batch, ensure_ascii=False)},
        rule_id=flt["id"],
        subject_type="term_cluster",
        subject_ids=cluster_ids,
    )

    # Enrich each cluster verdict's evidence with the member list so the
    # aggregator can resolve cluster_id → member terms when applying MERGE.
    members_by_cid = {c.cluster_id: list(c.members) for c in clusters}
    enriched: list[Verdict] = []
    for v in verdicts:
        new_evidence = dict(v.evidence or {})
        new_evidence["members"] = members_by_cid.get(v.subject_id, [])
        enriched.append(
            Verdict(
                rule_id=v.rule_id,
                subject_type=v.subject_type,
                subject_id=v.subject_id,
                verdict=v.verdict,
                reason=v.reason,
                evidence=new_evidence,
            )
        )
    return enriched


# ─── Main entry point ──────────────────────────────────────────────────


def run_domain_filters(
    taxonomy_csv: str,
    output_dir: str,
    *,
    relations_csv: str | None = None,
) -> str:
    """Apply every active domain filter; write `validate_filter_verdicts.csv`."""
    filters, exemptions = _load_filters()
    if not filters:
        log.warn("Domain filters: no active filters; producing empty verdicts CSV.")

    tax_df = read_csv(taxonomy_csv)
    rel_df = read_csv(relations_csv) if relations_csv and os.path.exists(relations_csv) else pd.DataFrame()

    ctx = StructuralContext(taxonomy_df=tax_df, relations_df=rel_df)

    tax_edges = _build_taxonomy_edges(tax_df)
    terms = _build_terms(tax_df)

    rows: list[dict[str, Any]] = []
    for flt in filters:
        target = flt.get("target", "terms")
        engine = flt.get("engine", "structural")
        if target == "taxonomy":
            subjects = tax_edges
        elif target == "terms":
            subjects = terms
        elif target == "relations":
            log.warn(f"Domain filter '{flt['id']}' targets relations but none built; skipping.")
            continue
        else:
            log.warn(f"Domain filters: unknown target '{target}' for '{flt['id']}'")
            continue

        if engine == "structural":
            verdicts = _dispatch_structural(flt, subjects, ctx, exemptions)
        elif engine == "llm":
            verdicts = _dispatch_llm_filter(flt, subjects, ctx, exemptions)
        elif engine == "hybrid":
            verdicts = _dispatch_hybrid(flt, subjects, ctx, exemptions)
        else:
            log.warn(f"Domain filters: unknown engine '{engine}' for '{flt['id']}'")
            continue

        action = str(flt.get("action", "REMOVE"))
        for v in verdicts:
            rows.append({
                "filter_id": v.rule_id,
                "engine": engine,
                "subject_type": v.subject_type,
                "subject_id": v.subject_id,
                "verdict": v.verdict,
                "action": action,
                "reason": v.reason,
                "evidence_json": json.dumps(v.evidence, ensure_ascii=False),
            })
        log.info(
            f"Domain filter '{flt['id']}' ({engine}/{target}) → "
            f"{sum(1 for v in verdicts if v.verdict == 'FAIL')} FAIL, "
            f"{sum(1 for v in verdicts if v.verdict == 'PASS')} PASS, "
            f"{sum(1 for v in verdicts if v.verdict == 'ABSTAIN')} ABSTAIN"
        )

    out_path = os.path.join(output_dir, "validate_filter_verdicts.csv")
    out_df = pd.DataFrame(rows, columns=[
        "filter_id", "engine", "subject_type", "subject_id",
        "verdict", "action", "reason", "evidence_json",
    ])
    write_csv(out_df, out_path)
    log.success(f"Domain filters: wrote {len(out_df)} verdicts to {out_path}")
    return out_path
