"""Ontology critic — the `validate` verb.

Per category, focused LLM calls run in sequence so each is informed by the
previous:

  1. **Taxonomy critic (Stage 1, per-term, CHUNKED).** Judges the IS-A rows
     (KEEP / REPARENT / DROP_AS_MIXIN / DROP_AS_REDUNDANT / CONVERT_TO_INSTANCE)
     in small chunks (``CRITIC_TAXONOMY_CHUNK_SIZE`` terms, default 5) so each
     call reasons about only a handful of terms. It sees the chunk's terms +
     NLDs, the chunk terms' own/ancestor relations as read-only context, the
     parents' NLDs (to judge vacuous restatement), and the allowed target-class
     list. It also runs an OntoClean parent–child edge check
     (rigidity / dependence drive REPARENT; identity is advisory-only) from the
     parent NLDs, and buckets the
     differentia by BFO category (Quality / Disposition / Role / Site / Process
     / TemporalRegion). Output carries a per-row probe trace (probe1/2/3 + the
     OntoClean signs rigidity/identity/dependence) and, on every
     DROP_AS_MIXIN, a `carried_by` note (logged for audit, not re-emitted). DROP_AS_REDUNDANT here is
     **parent-collapse only** (a term that vacuously restates its parent).
  2. **Class-worthiness critic (Stage 1b, chunked).** Uses frequency/document
      coverage, CQ evidence, NLDs, siblings, and relations to assign one
      authoritative fate: primitive, defined, demote-to-property, or drop.
  3. **Dedup critic (Stage 2, cross-term, ONE call over survivors).** Sees all
     surviving terms of the category together and makes the two decisions that
     need a global view: sibling near-synonym redundancy and weak intermediates
     (using a Python-computed `child_count`). Only emits DROP_AS_REDUNDANT, each
     citing the `survivor` it collapses into. A mutual-drop guard then un-drops
     any term whose cited survivor was itself dropped (never lose a concept).
  4. **Facet/frame critic (Stage 2b).** Finds missing subsumptions, singleton
      intermediates, same/mixed-axis frames, corpus-verifiable completion
      candidates, and disjointness diagnostics.
  5. **Relation correctness + scope critics.** First judges property/filler
      correctness (KEEP / DROP / FIX), then independently classifies scope as
      generic, corpus-context, or individual-fact.

Categories run in parallel on the worker pool; the calls above are sequential
within a category. A completeness guard re-asks the model for any input ids it
forgot in stages 1 and 3, so large categories are not silently under-reviewed.

I/O contract:
    run_critic(taxonomy_csv, output_dir, *, relations_csv=None)
        -> (final_taxonomy_path, final_relations_path_or_None)

Outputs written to `output_dir`:
    validate_taxonomy.csv            — cleaned taxonomy
    validate_relations.csv           — cleaned relations (only if relations_csv given)
    validate_edits.csv               — full audit log (taxonomy rows carry the probe trace)
    validate_instances.csv           — terms converted to NamedIndividuals (Term, Target_Class, …)
    validate_minted_properties.csv   — newly invented ObjectProperties (provenance=critic_minted)
    validate_defined_classes.csv     — generalized owl:equivalentClass handoff
    validate_evidence_bundle.csv     — frequency/CQ evidence joined to taxonomy rows
    validate_demotions.csv           — useful property-like distinctions removed as named classes
    validate_facet_frames.csv        — same/mixed-axis frame diagnostics
    validate_frame_completion.csv    — corpus-attested completion proposals and decisions
    validate_disjointness.csv        — candidate same-axis disjointness sets
    validate_lateral_coherence_summary.json — run metrics
    validate_responses_archive/{ts}.jsonl — raw LLM responses, tagged by call

Safety guards:
    - Default temperature 0 (deterministic).
    - Completeness guard: any input id the model omits is re-asked once; rows
      still missing fall back to implicit KEEP.
    - After taxonomy edits, any relation whose Filler was DROPped is dropped
      (phantom-filler cleanup, logged as DROP/phantom).
    - Every decision is recorded in validate_edits.csv; nothing is silently
      undone — the audit log is the safety net.
"""

from __future__ import annotations

import json
import os
import threading
import glob
import re
import unicodedata
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils import log
from src.utils.csv_io import read_csv, write_csv
from src.utils.llm_client import get_client, generate
from src.utils.ontology_config import get_config
from src.utils.prompt_loader import load_prompt
from src.utils.rag_setup import get_embedding_model
from src.utils.relation_validator import (
    PROPERTY_CONSTRAINTS,
    get_metatypes,
    normalize_property,
    validate_relation,
)


_TAXONOMY_VERDICTS = {
    "KEEP", "REPARENT", "KEEP_AS_BEARER", "KEEP_AS_DEFINED",
    "DROP_AS_MIXIN", "DROP_AS_REDUNDANT", "DROP_AS_OVER_SPECIFIC",
    "DEMOTE_TO_PROPERTY", "CONVERT_TO_INSTANCE",
}
_RELATION_VERDICTS = {"KEEP", "DROP", "FIX"}
_RELATION_SCOPES = {"generic", "corpus_context", "individual_fact"}

_DROP_TAX_VERDICTS = {
    "DROP_AS_MIXIN", "DROP_AS_REDUNDANT", "DROP_AS_OVER_SPECIFIC",
    "DEMOTE_TO_PROPERTY", "CONVERT_TO_INSTANCE",
}

# KEEP_AS_BEARER: companion object-property → the BFO parent the minted filler
# class is declared under. Keeps a material bearer under its genus and carries
# the realizable/quality off the IS-A edge onto a companion axiom.
_BEARER_PROPERTIES = {
    "has_role": "role",
    "has_function": "function",
    "has_disposition": "disposition",
    "has_quality": "quality",
}

# Deterministic Aristotelian fallback NLD for a minted filler class, keyed by
# its BFO genus. Used only when the critic does not supply a usable `filler_nld`
# so every minted class always carries an `rdfs:comment` (verification requires
# it). `{bearer}` is the material entity the realizable/quality inheres in.
_BEARER_NLD_TEMPLATES = {
    "role": "A role that inheres in a {bearer}.",
    "function": "A function that inheres in a {bearer}.",
    "disposition": "A disposition that inheres in a {bearer}.",
    "quality": "A quality that inheres in a {bearer}.",
}

# BFO realizable genera whose KEEP_AS_BEARER carry makes the bearer a DEFINED
# class (`bearer ≡ genus ⊓ <property> some <minted role>`) rather than a
# primitive kind — the OntoClean fix for a role/function/disposition fused into
# a rigid class name. Quality carries stay primitive (a quality-bearing entity
# is not a role mixin).
_DEFINING_BEARER_PARENTS = {"role", "function", "disposition"}


# ─── Payload builders ─────────────────────────────────────────────────────

def _term_to_category(tax: pd.DataFrame) -> dict[str, str]:
    return {
        str(t).strip().lower(): c
        for t, c in zip(tax["Term"].tolist(), tax["Category"].tolist())
    }


def _term_to_parent(tax: pd.DataFrame) -> dict[str, str]:
    return {
        str(t).strip().lower(): str(p).strip()
        for t, p in zip(tax["Term"].tolist(), tax["Parent_Term"].tolist())
    }


def _ancestor_chain(term: str, parents: dict[str, str], valid_categories: set[str], max_depth: int = 4) -> list[str]:
    """Walk up Parent_Term until we hit a category root or run out."""
    out: list[str] = []
    seen: set[str] = set()
    cur = parents.get(term.strip().lower())
    while cur and cur not in valid_categories and cur.lower() not in seen and len(out) < max_depth:
        seen.add(cur.lower())
        out.append(cur)
        cur = parents.get(cur.strip().lower())
    return out


def _build_validation_evidence(
    tax: pd.DataFrame,
    output_dir: str,
) -> tuple[dict[str, dict], pd.DataFrame]:
    """Join frequency and CQ evidence for class-worthiness decisions.

    Missing evidence stays explicit (zero/empty) so builder-minted intermediates
    are not mistaken for corpus-attested terms.
    """
    evidence: dict[str, dict] = {}
    frequency_path = os.environ.get("FILTERED_TERMS_OUTPUT", "output/extract_filtered.csv")
    cq_path = os.path.join(output_dir, "5b_cq_matrix.csv")
    docs_dir = os.environ.get("DOCS_DIR", "inputs")
    document_count = len(glob.glob(os.path.join(docs_dir, "*.md")))

    if os.path.exists(frequency_path):
        freq_df = read_csv(frequency_path)
        for _, row in freq_df.iterrows():
            term = str(row.get("Readable_Term", row.get("Term", ""))).strip().lower()
            if not term:
                continue
            frequency = int(row.get("Frequency", 0) or 0)
            evidence.setdefault(term, {}).update({
                "frequency": frequency,
                "document_coverage": round(frequency / document_count, 4) if document_count else 0.0,
            })

    if os.path.exists(cq_path):
        cq_df = read_csv(cq_path)
        cq_cols = [c for c in cq_df.columns if re.fullmatch(r"CQ\d+", str(c))]
        for _, row in cq_df.iterrows():
            term = str(row.get("Term", "")).strip().lower()
            if not term:
                continue
            matched = [
                col for col in cq_cols
                if str(row.get(col, "")).strip().lower() in {"1", "true", "yes"}
                or row.get(col, 0) == 1
            ]
            evidence.setdefault(term, {}).update({
                "cq_count": int(row.get("CQ_Count", len(matched)) or 0),
                "matched_cqs": matched,
                "cq_reasoning": str(row.get("Reasoning", ""))[:300],
            })

    rows: list[dict] = []
    for _, row in tax.iterrows():
        term = str(row["Term"]).strip()
        ev = evidence.get(term.lower(), {})
        entry = {
            "Term": term,
            "Frequency": int(ev.get("frequency", 0)),
            "Document_Coverage": float(ev.get("document_coverage", 0.0)),
            "CQ_Count": int(ev.get("cq_count", 0)),
            "Matched_CQs": "|".join(ev.get("matched_cqs", [])),
            "CQ_Reasoning": ev.get("cq_reasoning", ""),
            "Is_Intermediate": bool(row.get("Is_Intermediate", False)),
        }
        rows.append(entry)
        evidence.setdefault(term.lower(), {}).update({
            "frequency": entry["Frequency"],
            "document_coverage": entry["Document_Coverage"],
            "cq_count": entry["CQ_Count"],
            "matched_cqs": ev.get("matched_cqs", []),
            "cq_reasoning": entry["CQ_Reasoning"],
        })
    return evidence, pd.DataFrame(rows)


def _build_taxonomy_payload(
    rows: pd.DataFrame,
) -> list[dict]:
    out = []
    for _, r in rows.iterrows():
        out.append({
            "id": int(r["_critic_id"]),
            "term": r["Term"],
            "parent_term": r["Parent_Term"],
            "category": r.get("Category", ""),
            "is_intermediate": bool(r.get("Is_Intermediate", False)),
            "nld": str(r.get("NLD", ""))[:400],
        })
    return out


def _build_worthiness_payload(
    rows: list[pd.Series],
    taxonomy_edits: dict[int, dict],
    evidence: dict[str, dict],
) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        rid = int(row["_critic_id"])
        edit = taxonomy_edits.get(rid, {})
        corrected_parent = str(edit.get("new_parent", "") or row["Parent_Term"]).strip()
        ev = evidence.get(str(row["Term"]).strip().lower(), {})
        out.append({
            "id": rid,
            "term": row["Term"],
            "corrected_parent": corrected_parent,
            "category": row.get("Category", ""),
            "is_intermediate": bool(row.get("Is_Intermediate", False)),
            "nld": str(row.get("NLD", ""))[:500],
            "taxonomy_action": str(edit.get("action", "KEEP") or "KEEP").upper(),
            "frequency": int(ev.get("frequency", 0)),
            "document_coverage": float(ev.get("document_coverage", 0.0)),
            "cq_count": int(ev.get("cq_count", 0)),
            "matched_cqs": ev.get("matched_cqs", []),
            "cq_reasoning": str(ev.get("cq_reasoning", ""))[:300],
        })
    return out


def _build_worthiness_sibling_context(
    rows: list[pd.Series],
    taxonomy_edits: dict[int, dict],
    evidence: dict[str, dict],
) -> list[dict]:
    corrected_parents = {
        int(row["_critic_id"]): str(
            taxonomy_edits.get(int(row["_critic_id"]), {}).get("new_parent", "")
            or row["Parent_Term"]
        ).strip()
        for row in rows
    }
    child_counts: dict[str, int] = {}
    for parent in corrected_parents.values():
        if parent:
            child_counts[parent.lower()] = child_counts.get(parent.lower(), 0) + 1
    return [
        {
            "term": row["Term"],
            "parent": corrected_parents[int(row["_critic_id"])],
            "child_count": child_counts.get(str(row["Term"]).strip().lower(), 0),
            "frequency": int(evidence.get(str(row["Term"]).strip().lower(), {}).get("frequency", 0)),
            "cq_count": int(evidence.get(str(row["Term"]).strip().lower(), {}).get("cq_count", 0)),
            "matched_cqs": evidence.get(str(row["Term"]).strip().lower(), {}).get("matched_cqs", []),
            "nld": str(row.get("NLD", ""))[:180],
        }
        for row in rows
    ]


def _label_tokens(label: str) -> list[str]:
    return [tok for tok in str(label).strip().lower().replace("-", " ").split() if tok]


def _build_weak_taxonomy_observations(
    rows: pd.DataFrame,
    all_terms: set[str],
) -> list[dict]:
    """Noisy, non-actionable cues for the taxonomy critic.

    These observations deliberately avoid suggested actions/verdicts. They only
    ask the critic to inspect possible lateral-coherence issues more carefully;
    final decisions must be justified from NLDs, parent/sibling context, and
    relation evidence.
    """
    out: list[dict] = []
    all_lower = {term.strip().lower() for term in all_terms if str(term).strip()}
    for _, r in rows.iterrows():
        term = str(r["Term"]).strip()
        parent = str(r.get("Parent_Term", "")).strip()
        tokens = _label_tokens(term)
        term_lower = term.lower()

        if len(tokens) >= 3 or "-" in term:
            out.append({
                "id": int(r["_critic_id"]),
                "term": term,
                "observation_type": "stacked_or_compound_label",
                "observed_pattern": "multi-token or hyphenated label",
                "question": "Check whether this is a reusable kind or an over-specific property/facet combination.",
            })

        candidates = []
        for other in sorted(all_lower, key=len, reverse=True):
            if other == term_lower:
                continue
            if term_lower.endswith(f" {other}") and parent.lower() != other:
                candidates.append(other)
                break
        if candidates:
            out.append({
                "id": int(r["_critic_id"]),
                "term": term,
                "observation_type": "label_contains_existing_class",
                "observed_pattern": f"label ends with existing class label '{candidates[0]}'",
                "question": "Check whether the NLD entails this existing class as a more specific parent.",
            })
    return out


def _build_parent_context(
    chunk: pd.DataFrame,
    cat_term_nld: dict[str, str],
    valid_categories: set[str],
) -> list[dict]:
    """NLDs of the chunk terms' parents (when the parent is itself a category
    term), so the Stage-1 critic can judge vacuous restatement (Check 5) even
    when the parent lives in a different chunk. Category roots and upper-class
    parents are skipped (no NLD / no collapse risk). Deduped."""
    seen: set[str] = set()
    out: list[dict] = []
    for _, r in chunk.iterrows():
        parent = str(r["Parent_Term"]).strip()
        pl = parent.lower()
        if not parent or parent in valid_categories or pl in seen:
            continue
        nld = cat_term_nld.get(pl)
        if not nld:
            continue
        seen.add(pl)
        out.append({"term": parent, "nld": str(nld)[:400]})
    return out


def _build_relation_payload(
    own_rows: pd.DataFrame,
    ancestor_rows: pd.DataFrame,
    context_lookup: dict[str, dict] | None = None,
) -> list[dict]:
    """Combine own + ancestor relation rows. Ancestor rows carry chain_role='ancestor'
    and the critic is instructed not to vote on them."""
    out: list[dict] = []
    context_lookup = context_lookup or {}
    def _entity(term) -> dict:
        key = str(term).strip().lower()
        return context_lookup.get(key, {
            "label": term, "nld": "", "parent": "", "ancestors": [],
            "category": "", "metatypes": [], "is_individual": False,
        })
    for _, r in own_rows.iterrows():
        out.append({
            "id": int(r["_critic_id"]),
            "chain_role": "own",
            "term": r["Term"],
            "property": r["Property"],
            "filler": r["Filler"],
            "evidence": str(r.get("Evidence", ""))[:300],
            "subject_context": _entity(r["Term"]),
            "filler_context": _entity(r["Filler"]),
        })
    for _, r in ancestor_rows.iterrows():
        out.append({
            "id": int(r["_critic_id"]),
            "chain_role": "ancestor",
            "term": r["Term"],
            "property": r["Property"],
            "filler": r["Filler"],
            "evidence": str(r.get("Evidence", ""))[:200],
            "subject_context": _entity(r["Term"]),
            "filler_context": _entity(r["Filler"]),
        })
    return out


def _build_relations_menu() -> list[dict]:
    """Rewrite/FIX property menu for the relation critic, sourced from the active
    ontology config: every relation flagged `critic_menu: true` in
    `ontology_config.yaml`. Domain-agnostic (no hardcoded names) and direction-
    correct (the YAML author picks the subject-anchored direction, which a
    mechanical inverse-dedup cannot do)."""
    cfg = get_config()
    out: list[dict] = []
    for name, pc in cfg.all_relations().items():
        if not pc.critic_menu:
            continue
        out.append({
            "name": pc.name,
            "domain": sorted(pc.domain),
            "range": sorted(pc.range),
            "inverse": pc.inverse,
        })
    return out


def _build_target_classes() -> list[str]:
    """Allowed `target_class` / `mint_parent` values for CONVERT_TO_INSTANCE.

    Sourced live from the active ontology config (BFO + GeoCore + GeoReservoir
    labels). Domain-agnostic: retargeting the pipeline to another domain
    automatically changes this list with zero code change.
    """
    return sorted(get_config().upper_iris().keys())


def _build_previously_minted(minted_csv: str) -> list[dict]:
    if not os.path.exists(minted_csv):
        return []
    try:
        df = read_csv(minted_csv)
    except Exception:
        return []
    out: list[dict] = []
    for _, r in df.iterrows():
        out.append({
            "name": str(r.get("Name", "")),
            "parent_property": str(r.get("ParentProperty", "")),
            "domain": str(r.get("Domain", "")),
            "range": str(r.get("Range", "")),
        })
    return out


def _build_taxonomy_context(rows: pd.DataFrame) -> list[dict]:
    """Lean term→NLD context for the relation critic (no ids — not votable)."""
    return [
        {"term": r["Term"], "nld": str(r.get("NLD", ""))[:300]}
        for _, r in rows.iterrows()
    ]


def _build_entity_context_lookup(
    tax: pd.DataFrame,
    individual_terms: set[str] | None = None,
) -> dict[str, dict]:
    """Rich local/upper class context keyed by normalized label."""
    individual_terms = {str(v).strip().lower() for v in (individual_terms or set())}
    parents = _term_to_parent(tax)
    row_by_term = {
        str(row["Term"]).strip().lower(): row for _, row in tax.iterrows()
    }
    out: dict[str, dict] = {}
    for key, row in row_by_term.items():
        ancestors: list[str] = []
        current = str(row.get("Parent_Term", "") or "").strip()
        seen: set[str] = set()
        while current and current.lower() not in seen and len(ancestors) < 4:
            seen.add(current.lower())
            ancestors.append(current)
            current = parents.get(current.lower(), "")
        category = str(row.get("Category", "") or "").strip()
        out[key] = {
            "label": row["Term"], "nld": str(row.get("NLD", ""))[:500],
            "parent": row.get("Parent_Term", ""), "ancestors": ancestors,
            "category": category, "metatypes": sorted(get_metatypes(category) or []),
            "is_individual": key in individual_terms,
        }

    cfg = get_config()
    for ontology in cfg.ontologies.values():
        for cls in ontology.classes:
            key = cls.label.strip().lower()
            if key in out:
                continue
            out[key] = {
                "label": cls.label,
                "nld": str(cls.llm_definition or "")[:500],
                "parent": "", "ancestors": [], "category": cls.label,
                "metatypes": sorted(cls.metatypes), "is_individual": False,
            }
    return out


def _select_facet_target_context(
    rows: list[pd.Series],
    context_lookup: dict[str, dict],
    max_per_term: int = 5,
) -> list[dict]:
    """Select plausible parent contexts by ancestry and lexical/head similarity."""
    selected: dict[str, dict] = {}
    for row in rows:
        term = str(row["Term"]).strip()
        term_norm = _normalised_label(term)
        parent = str(row.get("Parent_Term", "") or "").strip().lower()
        candidates: list[tuple[float, str]] = []
        for key, context in context_lookup.items():
            if key == term.lower():
                continue
            label_norm = _normalised_label(str(context.get("label", "")))
            if not label_norm:
                continue
            score = SequenceMatcher(None, term_norm, label_norm).ratio()
            if term_norm.endswith(f" {label_norm}"):
                score = max(score, 1.0)
            if key == parent or str(context.get("label", "")) in context_lookup.get(parent, {}).get("ancestors", []):
                score = max(score, 0.99)
            if score >= 0.55:
                candidates.append((score, key))
        for _, key in sorted(candidates, reverse=True)[:max_per_term]:
            selected[key] = context_lookup[key]
        if parent in context_lookup:
            selected[parent] = context_lookup[parent]
    return list(selected.values())


def _child_counts_among_survivors(survivor_rows: list[pd.Series]) -> dict[str, int]:
    """How many surviving rows name each (lower-cased) term as their parent.
    Computed in Python so the dedup critic's weak-intermediate check is
    deterministic rather than asking the model to count."""
    counts: dict[str, int] = {}
    for r in survivor_rows:
        parent = str(r["Parent_Term"]).strip().lower()
        if parent:
            counts[parent] = counts.get(parent, 0) + 1
    return counts


def _build_dedup_payload(
    survivor_rows: list[pd.Series],
    child_counts: dict[str, int],
) -> list[dict]:
    """Compact cross-term view for the Stage-2 dedup critic: every survivor with
    a short NLD, its parent, intermediate flag, and surviving child count."""
    out: list[dict] = []
    for r in survivor_rows:
        term = str(r["Term"]).strip()
        out.append({
            "id": int(r["_critic_id"]),
            "term": r["Term"],
            "parent": r["Parent_Term"],
            "is_intermediate": bool(r.get("Is_Intermediate", False)),
            "child_count": int(child_counts.get(term.lower(), 0)),
            "nld": str(r.get("NLD", ""))[:400],
        })
    return out


def _normalised_label(label: str) -> str:
    text = _normalise_search_text(label).replace("-", " ")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _build_cross_category_candidates(
    tax: pd.DataFrame,
    edits_by_id: dict[int, dict],
    top_k: int,
) -> list[dict]:
    """Union of each term's top-k cross-category NLD neighbors; never edit directly."""
    survivors = []
    for _, row in tax.iterrows():
        rid = int(row["_critic_id"])
        edit = edits_by_id.get(rid, {})
        if str(edit.get("action", "KEEP") or "KEEP").upper() in _DROP_TAX_VERDICTS:
            continue
        nld = str(row.get("NLD", "") or "").strip()
        if not nld:
            continue
        survivors.append(row)
    if len(survivors) < 2:
        return []

    texts = [str(row.get("NLD", ""))[:2000] for row in survivors]
    vectors = np.asarray(get_embedding_model().embed_documents(texts), dtype=float)
    similarity = vectors @ vectors.T  # BGE-M3 embeddings are normalized.
    labels = [_normalised_label(str(row["Term"])) for row in survivors]
    seen: set[tuple[int, int]] = set()
    candidates: list[dict] = []
    for i, row_a in enumerate(survivors):
        ranked = np.argsort(-similarity[i])
        nearest: list[int] = []
        for j in ranked:
            if i == j:
                continue
            row_b = survivors[int(j)]
            if str(row_a.get("Category", "")) == str(row_b.get("Category", "")):
                continue
            nearest.append(int(j))
            if len(nearest) >= top_k:
                break
        for j in nearest:
            row_b = survivors[j]
            pair_key = tuple(sorted((int(row_a["_critic_id"]), int(row_b["_critic_id"]))))
            if pair_key in seen:
                continue
            label_sim = SequenceMatcher(None, labels[i], labels[j]).ratio()
            nld_sim = float(similarity[i, j])
            same_head = bool(labels[i] and labels[j] and labels[i].split()[-1] == labels[j].split()[-1])
            seen.add(pair_key)
            candidates.append({
                "pair_id": len(candidates),
                "term_a": {
                    "id": int(row_a["_critic_id"]), "label": row_a["Term"],
                    "category": row_a.get("Category", ""), "parent": row_a.get("Parent_Term", ""),
                    "nld": str(row_a.get("NLD", ""))[:500],
                },
                "term_b": {
                    "id": int(row_b["_critic_id"]), "label": row_b["Term"],
                    "category": row_b.get("Category", ""), "parent": row_b.get("Parent_Term", ""),
                    "nld": str(row_b.get("NLD", ""))[:500],
                },
                "weak_similarity_signals": {
                    "nld_cosine": round(nld_sim, 4),
                    "label_similarity": round(label_sim, 4),
                    "same_head_token": same_head,
                },
            })
    return candidates


def _build_taxonomy_decisions(
    tax_edits_local: dict[int, dict],
    id_to_term: dict[int, str],
) -> list[dict]:
    """Compact handoff: the non-KEEP taxonomy decisions the relation critic must
    align to. KEEP is the default and omitted to keep the block small."""
    out: list[dict] = []
    for rid, edit in tax_edits_local.items():
        action = (edit.get("action") or "KEEP").upper()
        if action == "KEEP":
            continue
        term = id_to_term.get(rid)
        if not term:
            continue
        entry = {"term": term, "action": action}
        if action == "REPARENT" and (edit.get("new_parent") or "").strip():
            entry["new_parent"] = edit["new_parent"].strip()
        out.append(entry)
    return out


def _probe_cols(edit: dict | None) -> dict:
    """Extract the Option-E probe trace + OntoClean signs from a taxonomy edit
    for the audit log."""
    if not isinstance(edit, dict):
        return {"probe1_genus_ok": "", "probe2_bucket": "", "probe3_rewrite": "",
                "rigidity": "", "identity": "", "dependence": "", "carried_by": "",
                "drop_basis": ""}
    cb = edit.get("carried_by")
    return {
        "probe1_genus_ok": edit.get("probe1_genus_ok", ""),
        "probe2_bucket": str(edit.get("probe2_bucket", "") or ""),
        "probe3_rewrite": str(edit.get("probe3_rewrite", "") or "")[:200],
        "rigidity": str(edit.get("rigidity", "") or ""),
        "identity": str(edit.get("identity", "") or ""),
        "dependence": str(edit.get("dependence", "") or ""),
        "carried_by": json.dumps(cb, ensure_ascii=False) if isinstance(cb, dict) else "",
        "proposed_fate": str(edit.get("proposed_fate", "") or ""),
        "class_fate": str(edit.get("class_fate", "") or ""),
        "drop_basis": str(edit.get("drop_basis", "") or ""),
        "centrality": str(edit.get("centrality", "") or ""),
        "cross_axis": edit.get("cross_axis", ""),
        "over_specificity_reason": str(edit.get("over_specificity_reason", "") or "")[:300],
        "placement_rationale": str(edit.get("placement_rationale", "") or "")[:300],
        "needs_review": edit.get("needs_review", ""),
        "confidence": edit.get("confidence", ""),
    }


# ─── LLM calls + archive ──────────────────────────────────────────────────

def _archive(record: dict, archive_path: str, archive_lock: threading.Lock) -> None:
    with archive_lock:
        with open(archive_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _call_taxonomy_critic(
    category: str,
    terms_payload: list[dict],
    relations_context: list[dict],
    parent_context: list[dict],
    weak_observations: list[dict],
    target_classes: list[str],
    system_instruction: str,
    prompt_template: str,
    model: str,
    temperature: float,
    archive_path: str,
    archive_lock: threading.Lock,
) -> list[dict]:
    """Call 1: judge taxonomy rows. Returns the `edits` list."""
    prompt = prompt_template.format(
        category=category,
        terms_json=json.dumps(terms_payload, indent=2),
        relations_context_json=json.dumps(relations_context, indent=2) if relations_context else "[]",
        parent_context_json=json.dumps(parent_context, indent=2) if parent_context else "[]",
        weak_observations_json=json.dumps(weak_observations, indent=2) if weak_observations else "[]",
        target_classes_json=json.dumps(target_classes, indent=2),
    )
    raw_text = ""
    edits: list[dict] = []
    try:
        raw_text = generate(
            prompt, model=model, system_instruction=system_instruction,
            temperature=temperature, response_mime_type="application/json",
        )
        data = json.loads(raw_text)
        edits = data.get("edits", []) if isinstance(data, dict) else []
        if not isinstance(edits, list):
            edits = []
    except Exception as e:
        tqdm.write(f"  [{category}] taxonomy critic failed ({e}); defaulting to KEEP")
        edits = []
    _archive({
        "timestamp": datetime.now(timezone.utc).isoformat(), "call": "taxonomy",
        "category": category, "model": model, "n_terms": len(terms_payload),
        "n_edits": len(edits), "weak_observations": weak_observations,
        "response_text": raw_text,
    }, archive_path, archive_lock)
    return edits


def _call_class_worthiness_critic(
    category: str,
    candidates_payload: list[dict],
    sibling_context: list[dict],
    relations_context: list[dict],
    weak_observations: list[dict],
    existing_classes: list[str],
    properties: list[dict],
    system_instruction: str,
    prompt_template: str,
    model: str,
    temperature: float,
    archive_path: str,
    archive_lock: threading.Lock,
) -> list[dict]:
    """Focused class-worthiness judgment over taxonomy survivors."""
    prompt = prompt_template.format(
        category=category,
        candidates_json=json.dumps(candidates_payload, indent=2),
        sibling_context_json=json.dumps(sibling_context, indent=2),
        relations_context_json=json.dumps(relations_context, indent=2) if relations_context else "[]",
        weak_observations_json=json.dumps(weak_observations, indent=2) if weak_observations else "[]",
        existing_classes_json=json.dumps(existing_classes, indent=2),
        properties_json=json.dumps(properties, indent=2),
    )
    raw_text = ""
    decisions: list[dict] = []
    try:
        raw_text = generate(
            prompt, model=model, system_instruction=system_instruction,
            temperature=temperature, response_mime_type="application/json",
        )
        data = json.loads(raw_text)
        if isinstance(data, dict) and isinstance(data.get("decisions"), list):
            decisions = data["decisions"]
    except Exception as e:
        tqdm.write(f"  [{category}] class-worthiness critic failed ({e}); keeping survivors")
    _archive({
        "timestamp": datetime.now(timezone.utc).isoformat(), "call": "class_worthiness",
        "category": category, "model": model, "n_candidates": len(candidates_payload),
        "n_decisions": len(decisions), "weak_observations": weak_observations,
        "response_text": raw_text,
    }, archive_path, archive_lock)
    return decisions


def _merge_worthiness_decision(
    existing: dict,
    decision: dict,
    allow_defined_classes: bool,
    conservative_drop: bool = True,
    min_confidence_apply: float = 0.70,
    needs_review_below: float = 0.85,
) -> dict:
    """Merge one authoritative worthiness fate into an existing taxonomy edit."""
    merged = dict(existing or {"action": "KEEP"})
    fate = str(decision.get("fate", "KEEP_PRIMITIVE") or "KEEP_PRIMITIVE").upper()
    fate_label = {
        "KEEP_PRIMITIVE": "primitive",
        "KEEP_DEFINED": "defined",
        "DEMOTE_TO_PROPERTY": "demote",
        "DROP_CLASS": "drop",
    }.get(fate, "primitive")
    try:
        confidence = float(decision.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    requires_threshold = fate == "KEEP_DEFINED" or (
        conservative_drop and fate in {"DROP_CLASS", "DEMOTE_TO_PROPERTY"}
    )
    apply_fate = not (requires_threshold and confidence < min_confidence_apply)
    needs_review = bool(decision.get("needs_review", False)) or confidence < needs_review_below
    merged.update({
        "proposed_fate": fate_label,
        "class_fate": fate_label if apply_fate else "primitive",
        "centrality": decision.get("centrality", "medium"),
        "cross_axis": decision.get("cross_axis", False),
        "placement_rationale": decision.get("placement_rationale", ""),
        "needs_review": needs_review,
        "confidence": confidence,
        "over_specificity_reason": decision.get("reason", "") if fate in {"DROP_CLASS", "DEMOTE_TO_PROPERTY"} else "",
    })
    prior_action = str(merged.get("action", "KEEP") or "KEEP").upper()
    retained_class_fate = "defined" if prior_action in {"KEEP_AS_BEARER", "KEEP_AS_DEFINED"} else "primitive"
    if prior_action == "CONVERT_TO_INSTANCE":
        return merged
    if prior_action == "KEEP_AS_BEARER" and fate in {"KEEP_PRIMITIVE", "KEEP_DEFINED"}:
        merged["class_fate"] = "defined"
        return merged
    if not apply_fate:
        merged["reason"] = (
            f"class-worthiness fate {fate} not applied: confidence {confidence:.2f} "
            f"below {min_confidence_apply:.2f}; kept for review"
        )
        return merged
    if fate == "DROP_CLASS":
        drop_basis = str(decision.get("drop_basis", "") or "").upper()
        if drop_basis in {"NO_MARGINAL_VALUE", "NARROW_EXTENSION_DETAIL", "STACKED_CONTEXT"}:
            merged["action"] = "DROP_AS_OVER_SPECIFIC"
            merged["drop_basis"] = drop_basis
            merged["reason"] = decision.get("reason", "class-worthiness critic: drop class")
        else:
            merged["action"] = prior_action
            merged["class_fate"] = retained_class_fate
            merged["needs_review"] = True
            merged["reason"] = "DROP_CLASS rejected: missing/invalid core-exclusion basis; kept for reconciliation/review"
    elif fate == "DEMOTE_TO_PROPERTY":
        demoted_as = decision.get("demoted_as") if isinstance(decision.get("demoted_as"), dict) else {}
        required = all(str(demoted_as.get(k, "")).strip() for k in ("base_class", "property", "filler"))
        if required:
            merged["action"] = "DEMOTE_TO_PROPERTY"
            merged["demoted_as"] = demoted_as
            merged["reason"] = decision.get("reason", "class-worthiness critic: demote to property")
        else:
            merged["class_fate"] = retained_class_fate
            merged["needs_review"] = True
            merged["reason"] = "DEMOTE_TO_PROPERTY rejected: incomplete property representation"
    elif fate == "KEEP_DEFINED":
        if not allow_defined_classes:
            merged["class_fate"] = retained_class_fate
            merged["needs_review"] = True
            merged["reason"] = "KEEP_DEFINED rejected: defined classes are disabled"
        elif prior_action != "KEEP_AS_BEARER":
            defined_by = decision.get("defined_by") if isinstance(decision.get("defined_by"), dict) else {}
            required = all(str(defined_by.get(k, "")).strip() for k in ("base_class", "property", "filler"))
            if required:
                merged["action"] = "KEEP_AS_DEFINED"
                merged["defined_by"] = defined_by
                merged["reason"] = decision.get("reason", "class-worthiness critic: keep defined")
            else:
                merged["class_fate"] = retained_class_fate
                merged["needs_review"] = True
                merged["reason"] = "KEEP_DEFINED rejected: incomplete class definition"
    return merged


def _call_relation_critic(
    category: str,
    relations_payload: list[dict],
    relations_menu: list[dict],
    previously_minted: list[dict],
    taxonomy_context: list[dict],
    taxonomy_decisions: list[dict],
    system_instruction: str,
    prompt_template: str,
    model: str,
    temperature: float,
    archive_path: str,
    archive_lock: threading.Lock,
) -> list[dict]:
    """Call 2: judge relation rows. Returns the `edits` list."""
    prompt = prompt_template.format(
        category=category,
        relations_json=json.dumps(relations_payload, indent=2) if relations_payload else "[]",
        relations_menu_json=json.dumps(relations_menu, indent=2),
        previously_minted_json=json.dumps(previously_minted, indent=2) if previously_minted else "[]",
        taxonomy_context_json=json.dumps(taxonomy_context, indent=2) if taxonomy_context else "[]",
        taxonomy_decisions_json=json.dumps(taxonomy_decisions, indent=2) if taxonomy_decisions else "[]",
    )
    raw_text = ""
    edits: list[dict] = []
    try:
        raw_text = generate(
            prompt, model=model, system_instruction=system_instruction,
            temperature=temperature, response_mime_type="application/json",
        )
        data = json.loads(raw_text)
        if isinstance(data, dict) and isinstance(data.get("edits"), list):
            edits = data["edits"]
    except Exception as e:
        tqdm.write(f"  [{category}] relation critic failed ({e}); defaulting to KEEP")
        edits = []
    _archive({
        "timestamp": datetime.now(timezone.utc).isoformat(), "call": "relation",
        "category": category, "model": model, "n_relations": len(relations_payload),
        "n_edits": len(edits), "response_text": raw_text,
    }, archive_path, archive_lock)
    return edits


def _call_relation_scope_critic(
    category: str,
    relations_payload: list[dict],
    taxonomy_context: list[dict],
    taxonomy_decisions: list[dict],
    system_instruction: str,
    prompt_template: str,
    model: str,
    temperature: float,
    archive_path: str,
    archive_lock: threading.Lock,
) -> list[dict]:
    prompt = prompt_template.format(
        category=category,
        relations_json=json.dumps(relations_payload, indent=2),
        taxonomy_context_json=json.dumps(taxonomy_context, indent=2),
        taxonomy_decisions_json=json.dumps(taxonomy_decisions, indent=2),
    )
    raw_text = ""
    decisions: list[dict] = []
    try:
        raw_text = generate(
            prompt, model=model, system_instruction=system_instruction,
            temperature=temperature, response_mime_type="application/json",
        )
        data = json.loads(raw_text)
        if isinstance(data, dict) and isinstance(data.get("decisions"), list):
            decisions = data["decisions"]
    except Exception as e:
        tqdm.write(f"  [{category}] relation-scope critic failed ({e}); defaulting to generic")
    _archive({
        "timestamp": datetime.now(timezone.utc).isoformat(), "call": "relation_scope",
        "category": category, "model": model, "n_relations": len(relations_payload),
        "n_decisions": len(decisions), "response_text": raw_text,
    }, archive_path, archive_lock)
    return decisions


def _scope_payload_after_relation_edits(
    relations_payload: list[dict],
    edits_by_id: dict[int, dict],
) -> list[dict]:
    out: list[dict] = []
    for row in relations_payload:
        if row.get("chain_role") == "ancestor":
            continue
        rid = row.get("id")
        edit = edits_by_id.get(rid, {}) if isinstance(rid, int) else {}
        if str(edit.get("action", "KEEP") or "KEEP").upper() == "DROP":
            continue
        corrected = dict(row)
        if edit.get("new_property"):
            corrected["property"] = edit["new_property"]
        if edit.get("new_filler"):
            corrected["filler"] = edit["new_filler"]
        corrected["correctness_reason"] = edit.get("reason", "")
        out.append(corrected)
    return out


def _call_dedup_critic(
    category: str,
    survivors_payload: list[dict],
    cross_candidates: list[dict],
    system_instruction: str,
    prompt_template: str,
    model: str,
    temperature: float,
    archive_path: str,
    archive_lock: threading.Lock,
) -> tuple[list[dict], list[dict]]:
    """Stage 2 reconciliation for local survivors or shortlisted global pairs."""
    prompt = prompt_template.format(
        category=category,
        survivors_json=json.dumps(survivors_payload, indent=2),
        cross_candidates_json=json.dumps(cross_candidates, indent=2),
    )
    raw_text = ""
    edits: list[dict] = []
    reconciliations: list[dict] = []
    try:
        raw_text = generate(
            prompt, model=model, system_instruction=system_instruction,
            temperature=temperature, response_mime_type="application/json",
        )
        data = json.loads(raw_text)
        if isinstance(data, dict) and isinstance(data.get("edits"), list):
            edits = data["edits"]
        if isinstance(data, dict) and isinstance(data.get("reconciliations"), list):
            reconciliations = data["reconciliations"]
    except Exception as e:
        tqdm.write(f"  [{category}] dedup critic failed ({e}); keeping all survivors")
        edits = []
    _archive({
        "timestamp": datetime.now(timezone.utc).isoformat(), "call": "dedup",
        "category": category, "model": model, "n_survivors": len(survivors_payload),
        "n_cross_candidates": len(cross_candidates), "n_edits": len(edits),
        "n_reconciliations": len(reconciliations), "response_text": raw_text,
    }, archive_path, archive_lock)
    return edits, reconciliations


def _collect_global_reconciliation_decisions(
    candidates: list[dict],
    batch_size: int,
    max_workers: int,
    invoke_batch,
) -> list[dict]:
    """Run independent reconciliation batches concurrently, preserving order."""
    batches = [
        candidates[start:start + batch_size]
        for start in range(0, len(candidates), batch_size)
    ]
    if not batches:
        return []

    batch_results: list[list[dict] | None] = [None] * len(batches)
    workers = min(max(1, max_workers), len(batches))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(invoke_batch, candidate_batch): batch_index
            for batch_index, candidate_batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_index = futures[future]
            try:
                batch_results[batch_index] = future.result()
            except Exception as exc:
                log.warn(
                    f"Global reconciliation batch {batch_index + 1}/{len(batches)} "
                    f"failed: {exc}"
                )
                batch_results[batch_index] = []

    decisions: list[dict] = []
    for candidate_batch, result in zip(batches, batch_results):
        decisions_batch = list(result or [])
        returned_ids = {
            decision.get("pair_id") for decision in decisions_batch
            if isinstance(decision, dict)
        }
        for candidate in candidate_batch:
            if candidate["pair_id"] not in returned_ids:
                decisions_batch.append({
                    "pair_id": candidate["pair_id"],
                    "decision": "NEEDS_REVIEW", "confidence": 0.0,
                    "needs_review": True,
                    "reason": "reconciliation critic omitted candidate pair",
                })
        decisions.extend(decisions_batch)
    return decisions


def _call_facet_critic(
    category: str,
    survivors_payload: list[dict],
    existing_classes: list[str],
    target_context: list[dict],
    max_candidates: int,
    system_instruction: str,
    prompt_template: str,
    model: str,
    temperature: float,
    archive_path: str,
    archive_lock: threading.Lock,
) -> dict:
    prompt = prompt_template.format(
        category=category,
        survivors_json=json.dumps(survivors_payload, indent=2),
        existing_classes_json=json.dumps(existing_classes, indent=2),
        target_context_json=json.dumps(target_context, indent=2),
        max_candidates=max_candidates,
    )
    raw_text = ""
    data: dict = {}
    try:
        raw_text = generate(
            prompt, model=model, system_instruction=system_instruction,
            temperature=temperature, response_mime_type="application/json",
        )
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            data = parsed
    except Exception as e:
        tqdm.write(f"  [{category}] facet-frame critic failed ({e}); no frame edits")
    _archive({
        "timestamp": datetime.now(timezone.utc).isoformat(), "call": "facet_frames",
        "category": category, "model": model, "n_survivors": len(survivors_payload),
        "response_text": raw_text,
    }, archive_path, archive_lock)
    return data


def _call_frame_completion_verifier(
    candidates: list[dict],
    system_instruction: str,
    prompt_template: str,
    model: str,
    temperature: float,
    archive_path: str,
    archive_lock: threading.Lock,
) -> list[dict]:
    if not candidates:
        return []
    prompt = prompt_template.format(candidates_json=json.dumps(candidates, indent=2))
    raw_text = ""
    decisions: list[dict] = []
    try:
        raw_text = generate(
            prompt, model=model, system_instruction=system_instruction,
            temperature=temperature, response_mime_type="application/json",
        )
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("decisions"), list):
            decisions = parsed["decisions"]
    except Exception as e:
        log.warn(f"Frame-completion evidence verifier failed ({e}); no candidates added")
    _archive({
        "timestamp": datetime.now(timezone.utc).isoformat(), "call": "frame_completion",
        "category": "all", "model": model, "n_candidates": len(candidates),
        "n_decisions": len(decisions), "response_text": raw_text,
    }, archive_path, archive_lock)
    return decisions


def _generate_completion_nlds(
    evidence_payload: list[dict],
    decisions: list[dict],
) -> list[dict]:
    """Use the standard NLD workflow for accepted, evidence-backed new terms."""
    from src.modules.define.nld_generator import generate_nld

    evidence_by_id = {
        int(item["candidate_id"]): item for item in evidence_payload
        if isinstance(item.get("candidate_id"), int)
    }
    out: list[dict] = []
    for decision in decisions:
        enriched = dict(decision)
        if not bool(decision.get("accept", False)):
            out.append(enriched)
            continue
        candidate_id = decision.get("candidate_id")
        evidence = evidence_by_id.get(candidate_id) if isinstance(candidate_id, int) else None
        if not evidence:
            enriched.update({"accept": False, "nld": "", "reason": "accepted candidate lacked evidence payload"})
            out.append(enriched)
            continue
        context_parts = [
            f"[{item.get('source', 'Unknown')}]\n{item.get('excerpt', '')}"
            for item in evidence.get("evidence", [])
        ]
        context = "\n\n".join(context_parts)
        try:
            nld_json, _ = generate_nld(str(evidence.get("label", "")), context)
            parsed = json.loads(nld_json)
            nld = str(parsed.get("Definition", "") or "").strip() if isinstance(parsed, dict) else ""
        except Exception as e:
            nld = ""
            enriched["reason"] = f"standard NLD generation failed: {e}"
        if not nld:
            enriched.update({"accept": False, "nld": ""})
        else:
            enriched["nld"] = nld
        out.append(enriched)
    return out


def _normalise_search_text(text: str) -> str:
    normalised = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(ch for ch in normalised if not unicodedata.combining(ch))


def _attest_completion_candidates(
    candidates: list[dict],
    min_documents: int,
) -> tuple[list[dict], list[dict]]:
    """Find proposed labels/search terms in distinct corpus documents."""
    docs_dir = os.environ.get("DOCS_DIR", "inputs")
    documents: list[tuple[str, str, str]] = []
    for path in glob.glob(os.path.join(docs_dir, "*.md")):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                raw = fh.read()
        except OSError:
            continue
        documents.append((os.path.basename(path), raw, _normalise_search_text(raw)))

    evidence_payload: list[dict] = []
    audit_rows: list[dict] = []
    seen_labels: set[str] = set()
    for candidate_id, candidate in enumerate(candidates, start=1):
        label = str(candidate.get("label", "") or "").strip()
        if not label or label.lower() in seen_labels:
            continue
        seen_labels.add(label.lower())
        search_terms = candidate.get("search_terms") if isinstance(candidate.get("search_terms"), list) else []
        search_terms = [label, *[str(v).strip() for v in search_terms if str(v).strip()]]
        norm_terms = list(dict.fromkeys(_normalise_search_text(v) for v in search_terms if v))
        matches: list[dict] = []
        for filename, raw, normalised in documents:
            positions: list[tuple[str, int]] = []
            for term in norm_terms:
                found = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalised)
                if found:
                    positions.append((term, found.start()))
            if not positions:
                continue
            term, pos = min(positions, key=lambda item: item[1])
            start = max(0, pos - 180)
            end = min(len(raw), pos + len(term) + 320)
            matches.append({"source": filename, "excerpt": raw[start:end].replace("\n", " ")[:600]})
        status = "ATTESTED" if len(matches) >= min_documents else "INSUFFICIENT_EVIDENCE"
        audit_rows.append({
            "Candidate_ID": candidate_id, "Candidate": label,
            "Parent_Term": candidate.get("parent", ""), "Category": candidate.get("category", ""),
            "Search_Terms": "|".join(search_terms), "Document_Count": len(matches),
            "Evidence_Documents": "|".join(m["source"] for m in matches[:10]),
            "Status": status, "Proposed_Reason": candidate.get("reason", ""),
            "Confidence": candidate.get("confidence", ""), "NLD": "", "Decision_Reason": "",
            "Relation_Extraction_Pending": False,
        })
        if status == "ATTESTED":
            evidence_payload.append({
                "candidate_id": candidate_id, "label": label,
                "parent": candidate.get("parent", ""), "category": candidate.get("category", ""),
                "proposed_reason": candidate.get("reason", ""),
                "document_count": len(matches), "evidence": matches[:5],
            })
    return evidence_payload, audit_rows


def _apply_frame_completion(
    cleaned_tax: pd.DataFrame,
    audit_rows: list[dict],
    decisions: list[dict],
    valid_categories: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_by_id = {
        int(d["candidate_id"]): d for d in decisions
        if isinstance(d, dict) and isinstance(d.get("candidate_id"), int)
    }
    existing_lower = {str(t).strip().lower() for t in cleaned_tax["Term"].astype(str)}
    parent_to_category = {
        str(t).strip().lower(): str(c).strip()
        for t, c in zip(cleaned_tax["Term"].astype(str), cleaned_tax["Category"].astype(str))
    }
    category_lower = {str(c).lower(): str(c) for c in valid_categories}
    upper_lower = {str(c).lower(): str(c) for c in get_config().upper_iris()}
    appended: list[dict] = []
    next_id = int(cleaned_tax["_critic_id"].max()) + 1 if "_critic_id" in cleaned_tax.columns and len(cleaned_tax) else 0
    for row in audit_rows:
        decision = decision_by_id.get(int(row["Candidate_ID"]))
        if row["Status"] != "ATTESTED" or not decision or not bool(decision.get("accept", False)):
            if decision:
                row["Status"] = "REJECTED_BY_CRITIC"
                row["Decision_Reason"] = decision.get("reason", "")
            continue
        label = str(row["Candidate"]).strip()
        parent = str(row["Parent_Term"]).strip()
        nld = str(decision.get("nld", "") or "").strip()
        parent_key = parent.lower()
        if label.lower() in existing_lower or not nld or (
            parent_key not in existing_lower and parent_key not in category_lower and parent_key not in upper_lower
        ):
            row["Status"] = "REJECTED_STRUCTURAL"
            row["Decision_Reason"] = "candidate exists, lacks grounded NLD, or parent is not surviving"
            continue
        category = parent_to_category.get(
            parent_key,
            category_lower.get(parent_key, upper_lower.get(parent_key, row.get("Category", ""))),
        )
        new_row = {col: "" for col in cleaned_tax.columns}
        new_row.update({
            "_critic_id": next_id, "Term": label, "Parent_Term": parent,
            "Relationship_Type": "rdfs:subClassOf", "Category": category,
            "Is_Intermediate": False, "NLD": nld, "FALLBACK": False,
        })
        next_id += 1
        appended.append(new_row)
        existing_lower.add(label.lower())
        row["Status"] = "ADDED"
        row["NLD"] = nld
        row["Decision_Reason"] = decision.get("reason", "")
        row["Relation_Extraction_Pending"] = True
    if appended:
        cleaned_tax = pd.concat([cleaned_tax, pd.DataFrame(appended)], ignore_index=True)
    return cleaned_tax, pd.DataFrame(audit_rows)


def _ask_complete(
    invoke,
    full_payload: list[dict],
    expected_ids: set[int],
    category: str,
    kind: str,
) -> list[dict]:
    """Call `invoke(payload)` and re-ask once for any expected id the model
    omitted. `invoke` returns an edits list. Rows still missing after the retry
    fall back to implicit KEEP downstream."""
    edits = invoke(full_payload)
    got = {e["id"] for e in edits if isinstance(e, dict) and isinstance(e.get("id"), int)}
    missing = expected_ids - got
    if missing:
        log.warn(f"  [{category}] {kind}: {len(missing)}/{len(expected_ids)} ids omitted — re-asking")
        subset = [p for p in full_payload if p.get("id") in missing]
        if subset:
            more = invoke(subset)
            edits = edits + [e for e in more if isinstance(e, dict)]
            got = {e["id"] for e in edits if isinstance(e, dict) and isinstance(e.get("id"), int)}
            still = expected_ids - got
            if still:
                log.warn(f"  [{category}] {kind}: {len(still)} ids still missing after retry (implicit KEEP)")
    return edits


def _mutual_drop_guard(
    tax_edits_local: dict[int, dict],
    id_to_term: dict[int, str],
    id_to_parent: dict[int, str],
) -> None:
    """Never lose a concept to a dangling redundancy drop.

    A DROP_AS_REDUNDANT survives only if the term it collapses into (the
    `survivor` field, or, for Stage-1 parent-collapse, the row's parent) is
    itself kept. If the cited survivor was *also* dropped — e.g. two near-
    synonyms each pointing at the other, or a parent that got dropped — the
    drop is reverted to KEEP so at least one representative of the concept
    remains. Mutates `tax_edits_local` in place. Single-pass over a snapshot of
    the dropped set: conservative (may keep an extra near-duplicate in long
    chains) but it can never delete the last survivor of a concept.
    """
    dropped_terms = {
        id_to_term[rid].strip().lower()
        for rid, e in tax_edits_local.items()
        if rid in id_to_term and (e.get("action", "") or "").upper() in _DROP_TAX_VERDICTS
    }
    for rid, e in list(tax_edits_local.items()):
        if (e.get("action", "") or "").upper() != "DROP_AS_REDUNDANT":
            continue
        survivor = (e.get("survivor") or "").strip() or id_to_parent.get(rid, "")
        if survivor and survivor.strip().lower() not in dropped_terms:
            continue  # survivor is alive — the drop is safe
        reverted = dict(e)
        note = "survivor also dropped" if survivor else "no survivor cited"
        prior = e.get("_pre_dedup_edit") if isinstance(e.get("_pre_dedup_edit"), dict) else None
        if prior:
            reverted = dict(prior)
        else:
            reverted["action"] = "KEEP"
        reverted["reason"] = f"(dedup reverted — {note}) {reverted.get('reason', '')}".strip()
        tax_edits_local[rid] = reverted
        log.detail(f"Mutual-drop guard: kept '{id_to_term.get(rid, rid)}' ({note})")


def _apply_cross_category_reconciliations(
    candidates: list[dict],
    decisions: list[dict],
    tax: pd.DataFrame,
    edits_by_id: dict[int, dict],
    min_confidence: float,
) -> tuple[dict[str, str], list[dict]]:
    """Apply at most one unambiguous global mutation per term."""
    candidate_by_id = {int(c["pair_id"]): c for c in candidates}
    row_by_id = {int(row["_critic_id"]): row for _, row in tax.iterrows()}
    aliases: dict[str, str] = {}
    audit: list[dict] = []
    prepared: list[tuple[dict, dict | None]] = []
    for decision in decisions:
        if not isinstance(decision, dict) or not isinstance(decision.get("pair_id"), int):
            continue
        pair = candidate_by_id.get(int(decision["pair_id"]))
        if not pair:
            continue
        kind = str(decision.get("decision", "NEEDS_REVIEW") or "NEEDS_REVIEW").upper()
        confidence = _as_float(decision.get("confidence", 0.0))
        apply = confidence >= min_confidence and kind not in {"DISTINCT", "NEEDS_REVIEW"}
        record = {
            "Pair_ID": decision["pair_id"],
            "Term_A": pair["term_a"]["label"], "Category_A": pair["term_a"]["category"],
            "Term_B": pair["term_b"]["label"], "Category_B": pair["term_b"]["category"],
            "Decision": kind, "Confidence": confidence,
            "Needs_Review": bool(decision.get("needs_review", False)) or confidence < min_confidence,
            "Reason": decision.get("reason", ""), "Applied": False,
        }
        if not apply:
            prepared.append((record, None))
            continue

        mutation: dict | None = None
        if kind == "SAME_KIND":
            survivor_id = decision.get("survivor_id")
            duplicate_id = decision.get("duplicate_id")
            if not isinstance(survivor_id, int) or not isinstance(duplicate_id, int):
                record["Needs_Review"] = True
                record["Reason"] = f"{record['Reason']} (invalid survivor/duplicate ids)".strip()
                prepared.append((record, None))
                continue
            survivor = row_by_id.get(survivor_id)
            duplicate = row_by_id.get(duplicate_id)
            if survivor is None or duplicate is None or survivor_id == duplicate_id:
                record["Needs_Review"] = True
                record["Reason"] = f"{record['Reason']} (invalid SAME_KIND endpoints)".strip()
                prepared.append((record, None))
                continue
            record.update({"Survivor": survivor["Term"], "Duplicate": duplicate["Term"]})
            mutation = {
                "target_id": duplicate_id,
                "signature": ("SAME_KIND", survivor_id),
                "kind": "SAME_KIND", "survivor": survivor, "duplicate": duplicate,
                "reason": str(decision.get("reason", "cross-category co-extension")),
            }
        elif kind in {"A_SUBCLASS_OF_B", "B_SUBCLASS_OF_A"}:
            child_key, parent_key = (
                ("term_a", "term_b") if kind == "A_SUBCLASS_OF_B" else ("term_b", "term_a")
            )
            child_id = int(pair[child_key]["id"])
            parent_id = int(pair[parent_key]["id"])
            child = row_by_id.get(child_id)
            parent = row_by_id.get(parent_id)
            if child is None or parent is None:
                record["Needs_Review"] = True
                record["Reason"] = f"{record['Reason']} (invalid subsumption endpoints)".strip()
                prepared.append((record, None))
                continue
            record.update({"Child": child["Term"], "New_Parent": parent["Term"]})
            mutation = {
                "target_id": child_id,
                "signature": ("REPARENT", parent_id),
                "kind": "REPARENT", "child": child, "parent": parent,
                "reason": str(decision.get("reason", "cross-category subsumption")),
            }
        prepared.append((record, mutation))

    mutation_groups: dict[int, list[int]] = {}
    for index, (_, mutation) in enumerate(prepared):
        if mutation is not None:
            mutation_groups.setdefault(int(mutation["target_id"]), []).append(index)

    winners: set[int] = set()
    for indices in mutation_groups.values():
        signatures = {prepared[index][1]["signature"] for index in indices}
        if len(signatures) > 1:
            for index in indices:
                record = prepared[index][0]
                record["Needs_Review"] = True
                record["Reason"] = f"{record['Reason']} (conflicting reconciliation targets; not applied)".strip()
            continue
        winners.add(max(indices, key=lambda index: (prepared[index][0]["Confidence"], -index)))
        for index in indices:
            if index in winners:
                continue
            record = prepared[index][0]
            record["Needs_Review"] = True
            record["Reason"] = f"{record['Reason']} (duplicate mutation proposal; not applied)".strip()

    for index, (record, mutation) in enumerate(prepared):
        if mutation is None or index not in winners:
            audit.append(record)
            continue
        target_id = int(mutation["target_id"])
        current = dict(edits_by_id.get(target_id, {"action": "KEEP"}))
        if str(current.get("action", "KEEP")).upper() in _DROP_TAX_VERDICTS:
            record["Needs_Review"] = True
            record["Reason"] = f"{record['Reason']} (target does not survive core selection)".strip()
            audit.append(record)
            continue
        if mutation["kind"] == "SAME_KIND":
            survivor = mutation["survivor"]
            duplicate = mutation["duplicate"]
            current.update({
                "action": "DROP_AS_REDUNDANT", "survivor": str(survivor["Term"]),
                "reason": mutation["reason"],
            })
            edits_by_id[target_id] = current
            aliases[str(duplicate["Term"]).strip().lower()] = str(survivor["Term"]).strip()
            record["Applied"] = True
        else:
            parent = mutation["parent"]
            if str(current.get("action", "KEEP")).upper() not in {"KEEP_AS_BEARER", "KEEP_AS_DEFINED"}:
                current["action"] = "REPARENT"
            current.update({
                "new_parent": str(parent["Term"]),
                "new_category": str(parent.get("Category", "")),
                "reason": mutation["reason"],
            })
            edits_by_id[target_id] = current
            record["Applied"] = True
        audit.append(record)
    return aliases, audit


def _finalize_reconciliation_audit(
    audit: list[dict],
    final_taxonomy: pd.DataFrame,
    valid_aliases: dict[str, str],
) -> None:
    """Mark Applied only when the materialized final taxonomy reflects it."""
    final_terms = {
        str(term).strip().lower() for term in final_taxonomy["Term"].astype(str)
    }
    final_parents = {
        str(term).strip().lower(): str(parent).strip().lower()
        for term, parent in zip(
            final_taxonomy["Term"].astype(str),
            final_taxonomy["Parent_Term"].astype(str),
        )
    }
    for record in audit:
        if not bool(record.get("Applied")):
            continue
        duplicate = str(record.get("Duplicate", "") or "").strip().lower()
        if duplicate:
            survivor = str(record.get("Survivor", "") or "").strip().lower()
            reflected = (
                duplicate not in final_terms
                and survivor in final_terms
                and valid_aliases.get(duplicate, "").strip().lower() == survivor
            )
        else:
            child = str(record.get("Child", "") or "").strip().lower()
            reflected = final_parents.get(child) == str(
                record.get("New_Parent", "") or ""
            ).strip().lower()
        if reflected:
            continue
        record["Applied"] = False
        record["Needs_Review"] = True
        record["Reason"] = f"{record.get('Reason', '')} (not reflected in final taxonomy)".strip()


def _redirect_relation_aliases(rel: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    if rel.empty or not aliases:
        return rel
    rel = rel.copy()
    for column in ("Term", "Filler"):
        rel[column] = rel[column].apply(
            lambda value: aliases.get(str(value).strip().lower(), value)
        )
    return rel


def _guard_reparent_cycles(
    tax: pd.DataFrame,
    edits_by_id: dict[int, dict],
) -> list[str]:
    """Revert critic reparents that would introduce a local taxonomy cycle."""
    row_by_id = {int(row["_critic_id"]): row for _, row in tax.iterrows()}
    id_by_term = {str(row["Term"]).strip().lower(): rid for rid, row in row_by_id.items()}
    reverted: list[str] = []
    while True:
        parent_by_id: dict[int, int] = {}
        for rid, row in row_by_id.items():
            edit = edits_by_id.get(rid, {})
            if str(edit.get("action", "KEEP") or "KEEP").upper() in _DROP_TAX_VERDICTS:
                continue
            parent = str(edit.get("new_parent", "") or row.get("Parent_Term", "")).strip().lower()
            if parent in id_by_term:
                parent_by_id[rid] = id_by_term[parent]
        cycle: list[int] | None = None
        for start in parent_by_id:
            path: list[int] = []
            positions: dict[int, int] = {}
            current = start
            while current in parent_by_id:
                if current in positions:
                    cycle = path[positions[current]:]
                    break
                positions[current] = len(path)
                path.append(current)
                current = parent_by_id[current]
            if cycle:
                break
        if not cycle:
            return reverted
        edited_cycle = [
            rid for rid in cycle
            if str(edits_by_id.get(rid, {}).get("action", "")).upper() == "REPARENT"
            or bool(str(edits_by_id.get(rid, {}).get("new_parent", "")).strip())
        ]
        if not edited_cycle:
            return reverted
        rid = edited_cycle[-1]
        edit = dict(edits_by_id.get(rid, {}))
        if str(edit.get("action", "")).upper() == "KEEP_AS_BEARER":
            edit.pop("new_parent", None)
        else:
            edit["action"] = "KEEP"
            edit.pop("new_parent", None)
            edit.pop("new_category", None)
        term = str(row_by_id[rid]["Term"])
        edit["reason"] = f"(reparent reverted — would create taxonomy cycle) {edit.get('reason', '')}".strip()
        edits_by_id[rid] = edit
        reverted.append(term)


# ─── Edit appliers ────────────────────────────────────────────────────────

def _nearest_surviving_parent(
    start_parent: str,
    tax: pd.DataFrame,
    edits_by_id: dict[int, dict],
    surviving_lower: set[str],
    upper_lower: set[str],
    valid_categories: set[str],
    excluded_lower: set[str] | None = None,
) -> str | None:
    """Follow edit survivors/original ancestors to the nearest live parent."""
    excluded_lower = excluded_lower or set()
    row_by_term = {
        str(row["Term"]).strip().lower(): row
        for _, row in tax.iterrows()
    }
    category_lower = {str(c).strip().lower(): str(c).strip() for c in valid_categories}
    seen: set[str] = set()
    current = str(start_parent).strip()
    while current and current.lower() not in seen:
        key = current.lower()
        seen.add(key)
        if (key in surviving_lower or key in upper_lower) and key not in excluded_lower:
            return current
        if key in category_lower and key not in excluded_lower:
            return category_lower[key]
        row = row_by_term.get(key)
        if row is None:
            return None
        edit = edits_by_id.get(int(row["_critic_id"]), {})
        action = str(edit.get("action", "KEEP") or "KEEP").upper()
        if action == "DROP_AS_REDUNDANT":
            survivor = str(edit.get("survivor", "") or "").strip()
            if survivor and survivor.lower() not in excluded_lower:
                current = survivor
                continue
        if action in {"REPARENT", "KEEP_AS_BEARER"}:
            new_parent = str(edit.get("new_parent", "") or "").strip()
            if new_parent:
                current = new_parent
                continue
        current = str(row.get("Parent_Term", "") or "").strip()
    return None

def _apply_taxonomy_edits(
    tax: pd.DataFrame,
    edits_by_id: dict[int, dict],
    valid_categories: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict], list[dict], list[dict]]:
    """Apply 6-verdict taxonomy edits.

    Returns:
        cleaned_tax_df — surviving taxonomy rows (drops applied, parents fixed)
        instances_df   — rows converted to NamedIndividuals
        log_rows       — full audit-log rows
        bearer_records — KEEP_AS_BEARER carries (bearer kept under its material
                         genus; the realizable/quality becomes a companion axiom)
        defined_records — KEEP_AS_DEFINED definitions (term stays, definition
                  emitted later as owl:equivalentClass)
        demotion_records — property-like distinctions removed as named classes
                   and preserved for audit (not universal base axioms)
    """
    log_rows: list[dict] = []
    keep_mask = [True] * len(tax)
    new_parents: dict[int, str] = {}
    new_categories: dict[int, str] = {}
    instance_records: list[dict] = []
    bearer_records: list[dict] = []
    defined_records: list[dict] = []
    demotion_records: list[dict] = []
    # A KEEP_AS_BEARER companion filler is only ever a freshly-minted
    # realizable/quality class. Reject any carry whose filler collides with an
    # existing class: minting `<filler> ⊑ role / quality / …` would retype it.
    # Upper-ontology classes are off-limits (we never alter a published
    # ontology), and reusing an existing domain term can force a continuant into
    # `quality`/`role` and make the ontology inconsistent.
    upper_lower = {k.strip().lower() for k in get_config().upper_iris()}
    existing_term_lower = {str(t).strip().lower() for t in tax["Term"].astype(str)}

    for idx, row in tax.iterrows():
        rid = int(row["_critic_id"])
        cat = row["Category"]
        edit = edits_by_id.get(rid)
        probes = _probe_cols(edit)
        if edit is None:
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "KEEP",
                             "reason": "(implicit — not mentioned by critic)", **probes})
            continue
        action = (edit.get("action") or "KEEP").upper()
        if action not in _TAXONOMY_VERDICTS:
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "KEEP",
                             "reason": f"(unknown verdict {action!r} — kept)", **probes})
            continue
        reason = edit.get("reason", "")
        if action == "KEEP":
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "KEEP", "reason": reason, **probes})
        elif action == "REPARENT":
            new_parent = (edit.get("new_parent") or "").strip()
            if not new_parent:
                log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                                 "term": row["Term"], "action": "KEEP",
                                 "reason": f"(REPARENT rejected — no new_parent) {reason}", **probes})
                continue
            new_parents[rid] = new_parent
            if str(edit.get("new_category", "") or "").strip():
                new_categories[rid] = str(edit["new_category"]).strip()
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "REPARENT",
                             "reason": f"parent → {new_parent}: {reason}", **probes})
        elif action == "KEEP_AS_BEARER":
            # Keep the material bearer; carry the realizable / quality off the
            # IS-A edge as a companion axiom (run_critic mints the filler class
            # and the `<bearer> <property> some <filler>` row). `new_parent` is
            # OPTIONAL — the bearer stays under its current (already-material)
            # parent unless the critic names a better genus, and a genus is only
            # ever REUSED (an existing class), never minted here.
            cb = edit.get("carried_by") if isinstance(edit.get("carried_by"), dict) else {}
            prop = str(cb.get("property", "") or "").strip()
            filler = str(cb.get("filler", "") or "").strip()
            filler_nld = str(cb.get("filler_nld", "") or "").strip()
            if prop not in _BEARER_PROPERTIES or not filler:
                # The carry itself is incomplete — reject to KEEP so the term is
                # never lost.
                log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                                 "term": row["Term"], "action": "KEEP",
                                 "reason": f"(KEEP_AS_BEARER rejected — incomplete carry) {reason}", **probes})
                continue
            filler_key = filler.strip().lower()
            if filler_key in upper_lower or filler_key in existing_term_lower:
                # The filler names an existing class — minting it under a BFO
                # realizable/quality parent would retype that class (forbidden
                # for upper ontologies, unsound for domain continuants). Reject
                # to KEEP so neither the bearer nor the named class is mutated.
                where = "upper-ontology" if filler_key in upper_lower else "existing domain"
                log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                                 "term": row["Term"], "action": "KEEP",
                                 "reason": f"(KEEP_AS_BEARER rejected — filler '{filler}' is an {where} class; would retype it) {reason}", **probes})
                continue
            new_parent = (edit.get("new_parent") or "").strip()
            # Move the bearer only when a non-realizable genus is named (the
            # orphan pass reuses it if it exists, else falls back). Absent or a
            # realizable branch → keep the current material parent untouched.
            if new_parent and new_parent.strip().lower() not in _BEARER_PROPERTIES.values():
                new_parents[rid] = new_parent
            bearer_records.append({
                "bearer": str(row["Term"]).strip(),
                "bearer_category": cat,
                "property": prop,
                "filler": filler,
                "filler_nld": filler_nld,
                "filler_parent": _BEARER_PROPERTIES[prop],
            })
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "KEEP_AS_BEARER",
                             "reason": f"bearer kept under '{new_parent or row['Parent_Term']}'; {prop} some {filler}: {reason}", **probes})
        elif action == "KEEP_AS_DEFINED":
            definition = edit.get("defined_by") if isinstance(edit.get("defined_by"), dict) else {}
            base = str(definition.get("base_class", "") or definition.get("genus", "") or "").strip()
            prop = str(definition.get("property", "") or "").strip()
            filler = str(definition.get("filler", "") or "").strip()
            def_reason = str(definition.get("rationale", "") or reason).strip()
            if not (base and prop and filler):
                log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                                 "term": row["Term"], "action": "KEEP",
                                 "reason": f"(KEEP_AS_DEFINED rejected — incomplete definition) {reason}", **probes})
                continue
            prop_pc = PROPERTY_CONSTRAINTS.get(prop)
            base_key = base.lower()
            filler_key = filler.lower()
            base_known = base in valid_categories or base_key in upper_lower or base_key in existing_term_lower
            filler_known = filler in valid_categories or filler_key in upper_lower or filler_key in existing_term_lower
            if prop_pc is None or not base_known or not filler_known:
                missing = []
                if prop_pc is None:
                    missing.append(f"property '{prop}'")
                if not base_known:
                    missing.append(f"base '{base}'")
                if not filler_known:
                    missing.append(f"filler '{filler}'")
                log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                                 "term": row["Term"], "action": "KEEP",
                                 "reason": f"(KEEP_AS_DEFINED rejected — unknown {', '.join(missing)}) {reason}", **probes})
                continue
            new_parent = (edit.get("new_parent") or "").strip()
            if new_parent:
                new_parents[rid] = new_parent
            defined_records.append({
                "bearer": str(row["Term"]).strip(),
                "genus": base,
                "property": prop,
                "filler": filler,
                "definition_type": "cross_axis",
                "rationale": def_reason,
            })
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "KEEP_AS_DEFINED",
                             "reason": f"defined as {base} + {prop} some {filler}: {reason}", **probes})
        elif action == "DEMOTE_TO_PROPERTY":
            demoted = edit.get("demoted_as") if isinstance(edit.get("demoted_as"), dict) else {}
            base = str(demoted.get("base_class", "") or "").strip()
            prop = str(demoted.get("property", "") or "").strip()
            filler = str(demoted.get("filler", "") or "").strip()
            base_known = (
                base.lower() in existing_term_lower
                or base.lower() in upper_lower
                or base in valid_categories
            )
            if not (base and prop and filler) or not base_known or prop not in PROPERTY_CONSTRAINTS:
                log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                                 "term": row["Term"], "action": "KEEP",
                                 "reason": f"(DEMOTE_TO_PROPERTY rejected — incomplete/unknown base or property) {reason}", **probes})
                continue
            keep_mask[idx] = False
            demotion_records.append({
                "Term": row["Term"], "Base_Class": base, "Property": prop,
                "Filler": filler, "Rationale": demoted.get("rationale", reason),
                "Original_Parent": row["Parent_Term"], "Original_Category": cat,
            })
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "DEMOTE_TO_PROPERTY",
                             "reason": f"demoted to {base} + {prop} {filler}: {reason}", **probes})
        elif action in _DROP_TAX_VERDICTS:
            keep_mask[idx] = False
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": action, "reason": reason, **probes})
            if action == "CONVERT_TO_INSTANCE":
                instance_records.append({
                    "Term": row["Term"],
                    "Target_Class": (edit.get("target_class") or "").strip() or "owl:NamedIndividual",
                    "Mint_Parent": (edit.get("mint_parent") or "").strip(),
                    "Original_Category": cat,
                    "Original_Parent": row["Parent_Term"],
                    "Reason": reason,
                })

    cleaned = tax.loc[keep_mask].copy()
    if new_parents:
        cleaned["Parent_Term"] = cleaned.apply(
            lambda r: new_parents.get(int(r["_critic_id"]), r["Parent_Term"]),
            axis=1,
        )
    if new_categories:
        cleaned["Category"] = cleaned.apply(
            lambda r: new_categories.get(int(r["_critic_id"]), r["Category"]),
            axis=1,
        )

    # Orphan re-parenting: collapse through a dropped parent's explicit
    # survivor/nearest surviving ancestor; Category is the last resort only.
    surviving_lower = {str(t).strip().lower() for t in cleaned["Term"].astype(str)}
    for idx, row in cleaned.iterrows():
        parent = str(row["Parent_Term"]).strip()
        term_key = str(row["Term"]).strip().lower()
        parent_key = parent.lower()
        if not parent:
            continue
        if parent_key == term_key and term_key in upper_lower:
            log.detail(
                f"Upper-class identity: '{row['Term']}' reuses '{parent}'; "
                "self-parent suppressed"
            )
            cleaned.at[idx, "Parent_Term"] = ""
            continue
        if parent in valid_categories:
            continue
        if parent_key in surviving_lower or parent_key in upper_lower:
            continue
        replacement = _nearest_surviving_parent(
            parent, tax, edits_by_id, surviving_lower, upper_lower, valid_categories,
            excluded_lower={term_key},
        )
        fallback = str(row["Category"]).strip()
        if not replacement and fallback.lower() != term_key:
            replacement = fallback
        replacement = replacement or ""
        log.detail(f"Orphan re-parent: '{row['Term']}' → '{replacement}' (was '{parent}', dropped)")
        cleaned.at[idx, "Parent_Term"] = replacement

    instances_df = pd.DataFrame(instance_records) if instance_records else pd.DataFrame(
        columns=["Term", "Target_Class", "Mint_Parent", "Original_Category", "Original_Parent", "Reason"]
    )
    return cleaned, instances_df, log_rows, bearer_records, defined_records, demotion_records


def _materialize_bearer_carries(
    bearer_records: list[dict],
    tax_cols: list[str] | None,
    rel_cols: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn KEEP_AS_BEARER carries into the rows emit consumes:

    * a minted **filler class** row `<filler> ⊑ <bfo parent>` (role / function /
      disposition / quality), deduplicated by filler name, and
    * a **companion relation** row `<bearer> <property> some <filler>`.

    Column order is aligned to the existing taxonomy / relation outputs so the
    appended rows merge cleanly. The bearer itself is already kept under its
    material genus by `_apply_taxonomy_edits` (via `new_parents`)."""
    if not bearer_records:
        return pd.DataFrame(), pd.DataFrame()
    filler_rows: dict[str, dict] = {}
    rel_rows: list[dict] = []
    for rec in bearer_records:
        filler = rec["filler"].strip()
        prop = rec["property"].strip()
        fp = rec["filler_parent"]
        key = filler.lower()
        if key and key not in filler_rows:
            # Prefer the critic's grounded NLD; fall back to the deterministic
            # Aristotelian template so every minted filler carries a comment.
            llm_nld = str(rec.get("filler_nld", "") or "").strip()
            nld = llm_nld if (llm_nld and not llm_nld.upper().startswith("ERROR")) else \
                _BEARER_NLD_TEMPLATES.get(fp, "A {genus} that inheres in a {bearer}.").format(
                    genus=fp, bearer=rec["bearer"])
            filler_rows[key] = {
                "Term": filler, "Parent_Term": fp, "Relationship_Type": "subClassOf",
                "Category": fp, "Is_Intermediate": True, "NLD": nld, "FALLBACK": False,
            }
        pc = PROPERTY_CONSTRAINTS.get(prop)
        rel_rows.append({
            "Term": rec["bearer"], "Category": rec.get("bearer_category", ""),
            "Property": prop, "Property_IRI": pc.iri if pc else "",
            "Filler": filler, "Filler_Source": "critic_bearer",
            "Confidence": 1.0, "Evidence": f"KEEP_AS_BEARER companion ({prop} some {filler})",
            "Validation_Status": "ACCEPTED", "Validation_Reason": "critic KEEP_AS_BEARER",
            "Relation_Scope": "generic",
            "Scope_Reason": "critic KEEP_AS_BEARER companion is definitional",
            "Scope_Confidence": 1.0,
            "Scope_Needs_Review": False,
        })
    filler_df = pd.DataFrame(list(filler_rows.values()))
    rel_df = pd.DataFrame(rel_rows)
    if tax_cols and not filler_df.empty:
        filler_df = filler_df.reindex(columns=tax_cols)
    if rel_cols and not rel_df.empty:
        rel_df = rel_df.reindex(columns=rel_cols)
    return filler_df, rel_df


def _build_defined_classes(
    bearer_records: list[dict],
    explicit_defined_records: list[dict],
    cleaned_tax: pd.DataFrame,
) -> pd.DataFrame:
    """Rows the emitter turns into `owl:equivalentClass` definitions.

    A KEEP_AS_BEARER bearer that carries a *realizable* (role / function /
    disposition) is defined as `genus ⊓ (<property> some <minted role>)` instead
    of being asserted as a primitive rigid kind — the OntoClean fix for a role
    fused into a class name. Quality carries stay primitive. The genus is the
    bearer's final taxonomy parent (after any REPARENT). KEEP_AS_DEFINED records
    use the same output schema for general cross-axis definitions."""
    cols = [
        "Bearer", "Genus", "Property", "Property_IRI", "Filler",
        "Definition_Type", "Definition_Rationale",
    ]
    if not bearer_records and not explicit_defined_records:
        return pd.DataFrame(columns=cols)
    genus_by_term = {
        str(t).strip().lower(): str(p).strip()
        for t, p in zip(cleaned_tax["Term"].astype(str), cleaned_tax["Parent_Term"].astype(str))
    }
    rows: list[dict] = []
    valid_targets = {
        str(term).strip().lower() for term in cleaned_tax["Term"].astype(str)
    } | {str(label).strip().lower() for label in get_config().upper_iris()}
    for rec in bearer_records:
        if rec.get("filler_parent") not in _DEFINING_BEARER_PARENTS:
            continue
        genus = genus_by_term.get(str(rec["bearer"]).strip().lower(), "")
        if not genus:
            continue
        prop = str(rec["property"]).strip()
        pc = PROPERTY_CONSTRAINTS.get(prop)
        rows.append({
            "Bearer": rec["bearer"], "Genus": genus,
            "Property": prop, "Property_IRI": pc.iri if pc else "",
            "Filler": rec["filler"],
            "Definition_Type": "bearer_realizable",
            "Definition_Rationale": "KEEP_AS_BEARER realizable carry",
        })
    for rec in explicit_defined_records:
        prop = str(rec.get("property", "")).strip()
        pc = PROPERTY_CONSTRAINTS.get(prop)
        bearer = str(rec.get("bearer", "")).strip()
        genus = str(rec.get("genus", "")).strip()
        filler = str(rec.get("filler", "")).strip()
        if (
            not pc
            or bearer.lower() not in valid_targets
            or genus.lower() not in valid_targets
            or filler.lower() not in valid_targets
        ):
            log.warn(
                f"Skipped defined class '{bearer}': base/filler/property no longer survives"
            )
            continue
        rows.append({
            "Bearer": bearer,
            "Genus": genus,
            "Property": prop,
            "Property_IRI": pc.iri,
            "Filler": filler,
            "Definition_Type": rec.get("definition_type", "cross_axis"),
            "Definition_Rationale": rec.get("rationale", ""),
        })
    return pd.DataFrame(rows, columns=cols)


def _relation_scope(edit: dict | None) -> str:
    if not isinstance(edit, dict):
        return "generic"
    scope = str(edit.get("relation_scope") or edit.get("scope") or "generic").strip().lower()
    return scope if scope in _RELATION_SCOPES else "generic"


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _apply_relation_edits(
    rel: pd.DataFrame,
    edits_by_id: dict[int, dict],
    minted_collector: list[dict],
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply KEEP/DROP/FIX to relation rows. FIX with a `mint` block appends
    a record to `minted_collector` for downstream persistence."""
    log_rows: list[dict] = []
    keep_mask = [True] * len(rel)
    updates: dict[int, dict[str, object]] = {}
    ts = datetime.now(timezone.utc).isoformat()
    cfg = get_config()
    project_ns = cfg.project_namespace()
    if "Relation_Scope" not in rel.columns:
        rel = rel.copy()
        rel["Relation_Scope"] = "generic"
    for col, default in (
        ("Scope_Reason", ""), ("Scope_Confidence", ""), ("Scope_Needs_Review", False),
    ):
        if col not in rel.columns:
            rel[col] = pd.Series([default] * len(rel), index=rel.index, dtype="object")
        else:
            rel[col] = rel[col].astype("object")

    for idx, row in rel.iterrows():
        rid = int(row["_critic_id"])
        cat = row.get("Category", "")
        edit = edits_by_id.get(rid)
        if edit is None:
            log_rows.append({"id": rid, "kind": "relation", "category": cat,
                             "term": row["Term"], "action": "KEEP",
                             "relation_scope": "generic",
                             "reason": "(implicit — not mentioned by critic)"})
            continue
        action = (edit.get("action") or "KEEP").upper()
        if action not in _RELATION_VERDICTS:
            log_rows.append({"id": rid, "kind": "relation", "category": cat,
                             "term": row["Term"], "action": "KEEP",
                             "relation_scope": "generic",
                             "reason": f"(unknown verdict {action!r} — kept)"})
            continue
        reason = edit.get("reason", "")
        scope = _relation_scope(edit)
        scope_reason = str(edit.get("scope_reason", "") or "")
        scope_confidence = edit.get("scope_confidence", "")
        scope_needs_review = edit.get("scope_needs_review", False)
        scope_patch = {
            "Relation_Scope": scope,
            "Scope_Reason": scope_reason,
            "Scope_Confidence": scope_confidence,
            "Scope_Needs_Review": scope_needs_review,
        }
        if action == "DROP":
            keep_mask[idx] = False
            log_rows.append({"id": rid, "kind": "relation", "category": cat,
                             "term": row["Term"], "action": "DROP",
                             "relation_scope": scope, "scope_reason": scope_reason,
                             "scope_confidence": scope_confidence,
                             "scope_needs_review": scope_needs_review, "reason": reason})
        elif action == "FIX":
            patch: dict[str, object] = {}
            np_ = (edit.get("new_property") or "").strip()
            nf_ = (edit.get("new_filler") or "").strip()
            mint = edit.get("mint") or {}
            if np_ and isinstance(mint, dict) and mint:
                # Mint a new property. Caller persists collector to CSV after the run.
                name = np_
                iri = f"{project_ns}{name}"
                minted_collector.append({
                    "Name": name,
                    "IRI": iri,
                    "ParentProperty": str(mint.get("parent_property", "")).strip(),
                    "Domain": str(mint.get("domain", "")).strip(),
                    "Range": str(mint.get("range", "")).strip(),
                    "Justification": str(mint.get("justification", "")).strip(),
                    "Timestamp": ts,
                })
            if np_:
                patch["Property"] = np_
            if nf_:
                patch["Filler"] = nf_
            if patch:
                patch.update(scope_patch)
                updates[rid] = patch
                bits = ", ".join(f"{k}→{v}" for k, v in patch.items())
                tag = " [MINT]" if mint else ""
                log_rows.append({"id": rid, "kind": "relation", "category": cat,
                                 "term": row["Term"], "action": "FIX",
                                 "relation_scope": scope,
                                 "scope_reason": scope_reason,
                                 "scope_confidence": scope_confidence,
                                 "scope_needs_review": scope_needs_review,
                                 "reason": f"{bits}{tag}: {reason}"})
            else:
                updates[rid] = scope_patch
                log_rows.append({"id": rid, "kind": "relation", "category": cat,
                                 "term": row["Term"], "action": "KEEP",
                                 "relation_scope": scope,
                                 "scope_reason": scope_reason,
                                 "scope_confidence": scope_confidence,
                                 "scope_needs_review": scope_needs_review,
                                 "reason": f"(FIX rejected — no new field) {reason}"})
        else:  # KEEP
            updates[rid] = scope_patch
            log_rows.append({"id": rid, "kind": "relation", "category": cat,
                             "term": row["Term"], "action": "KEEP",
                             "relation_scope": scope, "scope_reason": scope_reason,
                             "scope_confidence": scope_confidence,
                             "scope_needs_review": scope_needs_review, "reason": reason})

    cleaned = rel.loc[keep_mask].copy()
    if updates:
        for rid, patch in updates.items():
            mask = cleaned["_critic_id"] == rid
            for col, val in patch.items():
                cleaned.loc[mask, col] = val
    return cleaned, log_rows


def _normalize_and_revalidate_relations(
    cleaned_rel: pd.DataFrame,
    term_to_cat: dict[str, str],
) -> tuple[pd.DataFrame, list[dict], int]:
    """Finalise relations after the critic's edits, in one pass:

    1. **Re-normalise mereology** — specialization (`has_part` →
       `has_continuant_part`, …) is a pure function of the subject/filler
       metatypes but only runs at extraction. A critic FIX can genericise a
       property or swap the filler, so every row is genericised then
       re-specialised against its current subject/filler categories.
       `Property_IRI` is refreshed to match.
    2. **Re-validate domain/range** — a FIX can also make a relation
       BFO-invalid (wrong property for the metatypes, or a forbidden
       continuant↔occurrent parthood). Such rows are dropped and logged.
       This is the same `validate_relation` check the extractor runs — applied
       again because the critic is a second LLM mutation of the relations.

    Both checks are skipped for a row whose subject or filler category is
    unresolvable (external/upper-class filler): we cannot judge it, so we leave
    it exactly as the extractor accepted it (never drop on uncertainty).
    Non-mereological, still-valid properties pass through untouched.
    """
    if cleaned_rel.empty or "Property" not in cleaned_rel.columns:
        return cleaned_rel, [], 0
    n_norm = 0
    drop_log: list[dict] = []
    keep_idx: list = []
    for idx, row in cleaned_rel.iterrows():
        prop = str(row["Property"]).strip()
        subj_cat = str(row.get("Category", "")).strip()
        filler = str(row.get("Filler", "")).strip()
        filler_cat = term_to_cat.get(filler.lower(), "")

        # 1. Re-specialise mereology.
        new_prop = normalize_property(prop, subj_cat, filler_cat)
        if new_prop != prop:
            cleaned_rel.at[idx, "Property"] = new_prop
            pc = PROPERTY_CONSTRAINTS.get(new_prop)
            if pc is not None and "Property_IRI" in cleaned_rel.columns:
                cleaned_rel.at[idx, "Property_IRI"] = pc.iri
            prop = new_prop
            n_norm += 1

        # 2. Re-validate domain/range — only when both categories resolve.
        if get_metatypes(subj_cat) is not None and filler_cat and get_metatypes(filler_cat) is not None:
            ok, reason = validate_relation(subj_cat, prop, filler_cat)
            if not ok:
                drop_log.append({
                    "id": int(row.get("_critic_id", -1)), "kind": "relation",
                    "category": subj_cat, "term": row.get("Term", ""),
                    "action": "DROP", "reason": f"(post-critic re-validation — {reason})",
                })
                continue  # drop this row
        keep_idx.append(idx)

    cleaned_rel = cleaned_rel.loc[keep_idx].copy()
    return cleaned_rel, drop_log, n_norm


# ─── Orchestrator ─────────────────────────────────────────────────────────

def run_critic(
    taxonomy_csv: str,
    output_dir: str,
    *,
    relations_csv: str | None = None,
) -> tuple[str, str | None]:
    from dotenv import load_dotenv
    load_dotenv()
    get_client()

    os.makedirs(output_dir, exist_ok=True)
    tax_out = os.path.join(output_dir, "validate_taxonomy.csv")
    rel_out = os.path.join(output_dir, "validate_relations.csv") if relations_csv else None
    edits_out = os.path.join(output_dir, "validate_edits.csv")
    instances_out = os.path.join(output_dir, "validate_instances.csv")
    minted_out = os.path.join(output_dir, "validate_minted_properties.csv")
    defined_out = os.path.join(output_dir, "validate_defined_classes.csv")
    class_fates_out = os.path.join(output_dir, "validate_class_fates.csv")
    facet_frames_out = os.path.join(output_dir, "validate_facet_frames.csv")
    disjointness_out = os.path.join(output_dir, "validate_disjointness.csv")
    evidence_out = os.path.join(output_dir, "validate_evidence_bundle.csv")
    demotions_out = os.path.join(output_dir, "validate_demotions.csv")
    frame_completion_out = os.path.join(output_dir, "validate_frame_completion.csv")
    lateral_summary_out = os.path.join(output_dir, "validate_lateral_coherence_summary.json")
    subsumption_out = os.path.join(output_dir, "validate_subsumption_hints.csv")
    reconciliation_out = os.path.join(output_dir, "validate_term_reconciliation.csv")
    completion_relations_out = os.path.join(output_dir, "validate_frame_completion_relations.csv")

    archive_dir = os.path.join(output_dir, "validate_responses_archive")
    os.makedirs(archive_dir, exist_ok=True)
    archive_path = os.path.join(
        archive_dir,
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".jsonl",
    )
    archive_lock = threading.Lock()

    tax_system, tax_template = load_prompt("critic_taxonomy.txt")
    worth_system, worth_template = load_prompt("critic_class_worthiness.txt")
    dedup_system, dedup_template = load_prompt("critic_taxonomy_dedup.txt")
    facet_system, facet_template = load_prompt("critic_facet_frames.txt")
    completion_system, completion_template = load_prompt("critic_frame_completion.txt")
    rel_system, rel_template = load_prompt("critic_relations.txt")
    scope_system, scope_template = load_prompt("critic_relation_scope.txt")
    model = os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    temperature = float(os.environ.get("LLM_GENERATION_TEMPERATURE", 0))

    log.banner("validate", "Validate (taxonomy → worthiness → dedup → facets → relations/scope)")

    tax = read_csv(taxonomy_csv).reset_index(drop=True)
    tax["_critic_id"] = range(len(tax))
    log.info(f"Loaded {len(tax)} taxonomy rows from {taxonomy_csv}")
    term_evidence, evidence_df = _build_validation_evidence(tax, output_dir)
    write_csv(evidence_df, evidence_out)
    log.info(f"Validation evidence bundle: {len(evidence_df)} rows → {evidence_out}")

    rel: pd.DataFrame | None = None
    if relations_csv and os.path.exists(relations_csv):
        rel = read_csv(relations_csv).reset_index(drop=True)
        if "Validation_Status" in rel.columns:
            n_before = len(rel)
            rel = rel[rel["Validation_Status"].fillna("ACCEPTED") == "ACCEPTED"].copy()
            log.info(f"Loaded {len(rel)} ACCEPTED relations (skipped {n_before - len(rel)} REJECTED)")
        else:
            log.info(f"Loaded {len(rel)} relations from {relations_csv}")
        rel = rel.reset_index(drop=True)
        rel["_critic_id"] = range(len(rel))
        if "Category" not in rel.columns or rel["Category"].isna().all():
            t2c = _term_to_category(tax)
            rel["Category"] = rel["Term"].astype(str).str.strip().str.lower().map(t2c).fillna("")

    valid_categories = set(tax["Category"].astype(str).unique())
    categories = sorted(valid_categories)
    relations_menu = _build_relations_menu()
    target_classes = _build_target_classes()
    previously_minted = _build_previously_minted(minted_out)
    parent_lookup = _term_to_parent(tax)
    lateral_cfg = get_config().lateral_coherence()
    weak_observations_enabled = lateral_cfg.enabled and lateral_cfg.hints_enabled
    all_term_labels = {str(t).strip() for t in tax["Term"].astype(str) if str(t).strip()}
    entity_context_lookup = _build_entity_context_lookup(tax)
    if weak_observations_enabled:
        log.info("Lateral-coherence weak observations: enabled (LATERAL_HINTS_ENABLED override supported)")
    else:
        log.info("Lateral-coherence weak observations: disabled")

    all_tax_edits: dict[int, dict] = {}
    all_rel_edits: dict[int, dict] = {}
    all_facet_frames: list[dict] = []
    all_disjointness: list[dict] = []
    all_completion_candidates: list[dict] = []
    all_subsumption_hints: list[dict] = []
    edits_lock = threading.Lock()
    max_workers = int(os.environ.get("MAX_CONCURRENT_CRITIC", 5))

    def _critique_category(cat: str) -> None:
        tax_group = tax[tax["Category"] == cat]
        if len(tax_group) == 0:
            return

        # Category-level relations (used by the relation critic + cheap-skip).
        own_rel = pd.DataFrame()
        ancestor_rel = pd.DataFrame()
        if rel is not None:
            own_rel = rel[rel["Term"].astype(str).str.strip().str.lower().isin(
                {str(t).strip().lower() for t in tax_group["Term"]}
            )]
            # Ancestor relations: walk parents of every term in the group, collect rels.
            ancestor_terms: set[str] = set()
            for term in tax_group["Term"].astype(str):
                for anc in _ancestor_chain(term, parent_lookup, valid_categories):
                    ancestor_terms.add(anc.strip().lower())
            if ancestor_terms:
                ancestor_rel = rel[rel["Term"].astype(str).str.strip().str.lower().isin(ancestor_terms)]

        worth_properties = list(relations_menu)
        worth_property_names = {str(item.get("name", "")) for item in worth_properties}
        constraints = get_config().property_constraints()
        if not own_rel.empty and "Property" in own_rel.columns:
            for prop_name in own_rel["Property"].dropna().astype(str).unique():
                if prop_name in worth_property_names or prop_name not in constraints:
                    continue
                pc = constraints[prop_name]
                worth_properties.append({
                    "name": prop_name, "domain": sorted(pc.domain),
                    "range": sorted(pc.range), "inverse": pc.inverse,
                })
                worth_property_names.add(prop_name)

        id_to_term = {int(r["_critic_id"]): str(r["Term"]) for _, r in tax_group.iterrows()}
        id_to_parent = {int(r["_critic_id"]): str(r["Parent_Term"]).strip() for _, r in tax_group.iterrows()}
        cat_term_nld = {
            str(r["Term"]).strip().lower(): str(r.get("NLD", ""))
            for _, r in tax_group.iterrows()
        }

        # ── Stage 1: per-term taxonomy critic, CHUNKED ──
        tax_edits_local: dict[int, dict] = {}
        chunk_size = max(1, int(os.environ.get("CRITIC_TAXONOMY_CHUNK_SIZE", 5)))
        for start in range(0, len(tax_group), chunk_size):
            chunk = tax_group.iloc[start:start + chunk_size]
            chunk_terms_lower = {str(t).strip().lower() for t in chunk["Term"]}

            # Per-chunk relation context (only the chunk terms' own + ancestors).
            chunk_own_rel = pd.DataFrame()
            chunk_anc_rel = pd.DataFrame()
            if rel is not None:
                chunk_own_rel = rel[rel["Term"].astype(str).str.strip().str.lower().isin(chunk_terms_lower)]
                chunk_anc: set[str] = set()
                for term in chunk["Term"].astype(str):
                    for anc in _ancestor_chain(term, parent_lookup, valid_categories):
                        chunk_anc.add(anc.strip().lower())
                if chunk_anc:
                    chunk_anc_rel = rel[rel["Term"].astype(str).str.strip().str.lower().isin(chunk_anc)]

            rel_context = _build_relation_payload(chunk_own_rel, chunk_anc_rel, entity_context_lookup)
            parent_context = _build_parent_context(chunk, cat_term_nld, valid_categories)
            tax_payload = _build_taxonomy_payload(chunk)
            weak_observations = _build_weak_taxonomy_observations(chunk, all_term_labels) \
                if weak_observations_enabled else []

            def _invoke_tax(
                payload: list[dict],
                _rc=rel_context,
                _pc=parent_context,
                _wo=weak_observations,
            ) -> list[dict]:
                return _call_taxonomy_critic(
                    cat, payload, _rc, _pc, _wo, target_classes,
                    tax_system, tax_template, model, temperature,
                    archive_path, archive_lock,
                )

            chunk_edits = _ask_complete(
                _invoke_tax, tax_payload, {p["id"] for p in tax_payload}, cat, "taxonomy",
            )
            for e in chunk_edits:
                if isinstance(e, dict) and isinstance(e.get("id"), int):
                    tax_edits_local[e["id"]] = e

        # ── Stage 1b: focused class-worthiness over taxonomy survivors ──
        def _is_dropped(rid: int) -> bool:
            e = tax_edits_local.get(rid)
            return bool(e) and (e.get("action", "") or "").upper() in _DROP_TAX_VERDICTS

        survivor_rows = [r for _, r in tax_group.iterrows() if not _is_dropped(int(r["_critic_id"]))]
        if lateral_cfg.enabled and lateral_cfg.class_worthiness_enabled and survivor_rows:
            worth_chunk_size = max(1, int(os.environ.get("CRITIC_CLASS_WORTHINESS_CHUNK_SIZE", 5)))
            sibling_context = _build_worthiness_sibling_context(
                survivor_rows, tax_edits_local, term_evidence,
            )
            for start in range(0, len(survivor_rows), worth_chunk_size):
                worth_rows = survivor_rows[start:start + worth_chunk_size]
                worth_payload = _build_worthiness_payload(worth_rows, tax_edits_local, term_evidence)
                worth_df = pd.DataFrame(worth_rows)
                worth_terms = {str(row["Term"]).strip().lower() for row in worth_rows}
                worth_own_rel = own_rel[
                    own_rel["Term"].astype(str).str.strip().str.lower().isin(worth_terms)
                ] if not own_rel.empty else pd.DataFrame()
                worth_rel_context = _build_relation_payload(worth_own_rel, pd.DataFrame(), entity_context_lookup)
                worth_observations = _build_weak_taxonomy_observations(worth_df, all_term_labels) \
                    if weak_observations_enabled else []

                def _invoke_worth(
                    payload: list[dict],
                    _siblings=sibling_context,
                    _relations=worth_rel_context,
                    _observations=worth_observations,
                ) -> list[dict]:
                    return _call_class_worthiness_critic(
                        cat, payload, _siblings, _relations, _observations,
                        sorted(all_term_labels | set(target_classes)), worth_properties,
                        worth_system, worth_template, model, temperature,
                        archive_path, archive_lock,
                    )

                worth_decisions = _ask_complete(
                    _invoke_worth, worth_payload, {p["id"] for p in worth_payload},
                    cat, "class-worthiness",
                )
                for decision in worth_decisions:
                    if not (isinstance(decision, dict) and isinstance(decision.get("id"), int)):
                        continue
                    rid = decision["id"]
                    tax_edits_local[rid] = _merge_worthiness_decision(
                        tax_edits_local.get(rid, {"action": "KEEP"}),
                        decision,
                        lateral_cfg.allow_defined_classes,
                        lateral_cfg.conservative_drop,
                        lateral_cfg.min_confidence_apply,
                        lateral_cfg.needs_review_below,
                    )

        # ── Stage 2: cross-term dedup over class-worthy survivors ──
        survivor_rows = [r for _, r in tax_group.iterrows() if not _is_dropped(int(r["_critic_id"]))]
        if len(survivor_rows) >= 2:
            child_counts = _child_counts_among_survivors(survivor_rows)
            dedup_payload = _build_dedup_payload(survivor_rows, child_counts)
            dedup_edits, _ = _call_dedup_critic(
                cat, dedup_payload, [], dedup_system, dedup_template,
                model, temperature, archive_path, archive_lock,
            )
            for e in dedup_edits:
                if not (isinstance(e, dict) and isinstance(e.get("id"), int)):
                    continue
                if (e.get("action", "") or "").upper() == "DROP_AS_REDUNDANT":
                    tax_edits_local[e["id"]] = {
                        **e,
                        "_pre_dedup_edit": dict(tax_edits_local.get(e["id"], {"action": "KEEP"})),
                    }

        # ── Mutual-drop guard: never lose a concept to a dangling redundancy ──
        _mutual_drop_guard(tax_edits_local, id_to_term, id_to_parent)

        with edits_lock:
            all_tax_edits.update(tax_edits_local)

    workers = min(max_workers, len(categories)) if categories else 1
    log.info(f"Running critic on {len(categories)} categories with {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_critique_category, cat): cat for cat in categories}
        with tqdm(total=len(categories), desc="Critic per category", unit="cat") as pbar:
            for future in as_completed(futures):
                cat = futures[future]
                try:
                    future.result()
                    pbar.set_postfix_str(cat[:30])
                except Exception as e:
                    tqdm.write(f"  [error] Category '{cat}': {e}")
                pbar.update(1)

    # Global reconciliation uses the same semantic critic as local dedup, but
    # only over BGE/label-shortlisted pairs from different categories.
    reconciliation_aliases: dict[str, str] = {}
    reconciliation_audit: list[dict] = []
    if lateral_cfg.enabled and lateral_cfg.reconciliation_enabled:
        try:
            cross_candidates = _build_cross_category_candidates(
                tax, all_tax_edits, lateral_cfg.reconciliation_top_k,
            )
            batch_size = 20
            batch_count = (len(cross_candidates) + batch_size - 1) // batch_size
            reconciliation_workers = min(max(1, max_workers), batch_count) if batch_count else 1
            log.info(
                f"Global reconciliation: {len(cross_candidates)} pairs in "
                f"{batch_count} batches with {reconciliation_workers} workers"
            )

            def _invoke_global_batch(candidate_batch: list[dict]) -> list[dict]:
                _, decisions_batch = _call_dedup_critic(
                    "GLOBAL", [], candidate_batch,
                    dedup_system, dedup_template, model, temperature,
                    archive_path, archive_lock,
                )
                return decisions_batch

            reconciliation_decisions = _collect_global_reconciliation_decisions(
                cross_candidates, batch_size, max_workers, _invoke_global_batch,
            )
            reconciliation_aliases, reconciliation_audit = _apply_cross_category_reconciliations(
                cross_candidates, reconciliation_decisions, tax, all_tax_edits,
                lateral_cfg.min_confidence_apply,
            )
        except Exception as e:
            log.warn(f"Cross-category reconciliation skipped after local failure: {e}")
    reconciliation_cols = [
        "Pair_ID", "Term_A", "Category_A", "Term_B", "Category_B", "Decision",
        "Confidence", "Needs_Review", "Reason", "Applied", "Survivor", "Duplicate",
        "Child", "New_Parent",
    ]
    # Facet/subsumption auditing runs after global reconciliation so sibling
    # frames see the effective category and no longer reason over duplicates.
    if lateral_cfg.enabled and lateral_cfg.frame_audit_enabled:
        effective_groups: dict[str, list[pd.Series]] = {}
        for _, row in tax.iterrows():
            rid = int(row["_critic_id"])
            edit = all_tax_edits.get(rid, {})
            if str(edit.get("action", "KEEP") or "KEEP").upper() in _DROP_TAX_VERDICTS:
                continue
            effective_category = str(edit.get("new_category", "") or row["Category"])
            effective_groups.setdefault(effective_category, []).append(row)
        surviving_labels = {
            str(row["Term"]).strip() for rows in effective_groups.values() for row in rows
        }

        def _audit_facet_category(cat: str, rows: list[pd.Series]) -> None:
            facet_payload = _build_worthiness_payload(rows, all_tax_edits, term_evidence)
            facet_target_context = _select_facet_target_context(rows, entity_context_lookup)
            facet_result = _call_facet_critic(
                cat, facet_payload, sorted(surviving_labels | set(target_classes)),
                facet_target_context, lateral_cfg.frame_completion_max_candidates,
                facet_system, facet_template, model, temperature,
                archive_path, archive_lock,
            )
            rows_by_id = {int(row["_critic_id"]): row for row in rows}
            valid_targets_lower = {
                label.lower() for label in surviving_labels | set(target_classes)
            }
            local_hints: list[dict] = []
            local_frames: list[dict] = []
            local_disjointness: list[dict] = []
            local_completion: list[dict] = []

            with edits_lock:
                for reparent in facet_result.get("reparents", []) if isinstance(facet_result.get("reparents"), list) else []:
                    if not (isinstance(reparent, dict) and isinstance(reparent.get("id"), int)):
                        continue
                    rid = reparent["id"]
                    new_parent = str(reparent.get("new_parent", "") or "").strip()
                    if rid not in rows_by_id or new_parent.lower() not in valid_targets_lower:
                        continue
                    confidence = _as_float(reparent.get("confidence", 0.0))
                    apply_reparent = confidence >= lateral_cfg.min_confidence_apply
                    local_hints.append({
                        "category": cat, "id": rid, "term": rows_by_id[rid]["Term"],
                        "action": "REPARENT", "old_parent": rows_by_id[rid]["Parent_Term"],
                        "new_parent": new_parent, "confidence": confidence,
                        "needs_review": bool(reparent.get("needs_review", False)) or not apply_reparent,
                        "reason": reparent.get("reason", ""), "applied": apply_reparent,
                    })
                    if not apply_reparent:
                        continue
                    current = dict(all_tax_edits.get(rid, {"action": "KEEP"}))
                    if str(current.get("action", "KEEP")).upper() in {"KEEP_AS_BEARER", "KEEP_AS_DEFINED"}:
                        current["new_parent"] = new_parent
                    else:
                        current.update({"action": "REPARENT", "new_parent": new_parent})
                    current["reason"] = str(reparent.get("reason", "facet/subsumption audit"))
                    all_tax_edits[rid] = current

                for collapse in facet_result.get("collapses", []) if isinstance(facet_result.get("collapses"), list) else []:
                    if not (isinstance(collapse, dict) and isinstance(collapse.get("id"), int)):
                        continue
                    rid = collapse["id"]
                    survivor = str(collapse.get("survivor", "") or "").strip()
                    row = rows_by_id.get(rid)
                    if row is None or not bool(row.get("Is_Intermediate", False)) or survivor.lower() not in valid_targets_lower:
                        continue
                    confidence = _as_float(collapse.get("confidence", 0.0))
                    apply_collapse = confidence >= lateral_cfg.min_confidence_apply
                    local_hints.append({
                        "category": cat, "id": rid, "term": row["Term"],
                        "action": "COLLAPSE_SINGLETON", "old_parent": row["Parent_Term"],
                        "new_parent": survivor, "confidence": confidence,
                        "needs_review": bool(collapse.get("needs_review", False)) or not apply_collapse,
                        "reason": collapse.get("reason", ""), "applied": apply_collapse,
                    })
                    if apply_collapse:
                        all_tax_edits[rid] = {
                            **all_tax_edits.get(rid, {}), "action": "DROP_AS_REDUNDANT",
                            "survivor": survivor,
                            "reason": str(collapse.get("reason", "singleton intermediate collapsed")),
                        }

                id_to_term = {int(row["_critic_id"]): str(row["Term"]) for row in rows}
                id_to_parent = {int(row["_critic_id"]): str(row["Parent_Term"]) for row in rows}
                _mutual_drop_guard(all_tax_edits, id_to_term, id_to_parent)

            for frame in facet_result.get("facet_frames", []) if isinstance(facet_result.get("facet_frames"), list) else []:
                if isinstance(frame, dict):
                    local_frames.append({"category": cat, **frame})
            for disjoint in facet_result.get("disjointness", []) if isinstance(facet_result.get("disjointness"), list) else []:
                if isinstance(disjoint, dict):
                    local_disjointness.append({"category": cat, **disjoint})
            if lateral_cfg.frame_completion_enabled:
                raw_candidates = facet_result.get("completion_candidates", []) \
                    if isinstance(facet_result.get("completion_candidates"), list) else []
                for candidate in raw_candidates[:lateral_cfg.frame_completion_max_candidates]:
                    if isinstance(candidate, dict):
                        local_completion.append({"category": cat, **candidate})
            with edits_lock:
                all_subsumption_hints.extend(local_hints)
                all_facet_frames.extend(local_frames)
                all_disjointness.extend(local_disjointness)
                all_completion_candidates.extend(local_completion)

        facet_workers = min(max_workers, len(effective_groups)) if effective_groups else 1
        with ThreadPoolExecutor(max_workers=facet_workers) as pool:
            facet_futures = {
                pool.submit(_audit_facet_category, cat, rows): cat
                for cat, rows in effective_groups.items()
            }
            with tqdm(total=len(effective_groups), desc="Facet audit per category", unit="cat") as pbar:
                for future in as_completed(facet_futures):
                    cat = facet_futures[future]
                    try:
                        future.result()
                        pbar.set_postfix_str(cat[:30])
                    except Exception as e:
                        tqdm.write(f"  [error] Facet category '{cat}': {e}")
                    pbar.update(1)

    global_id_to_term = {int(row["_critic_id"]): str(row["Term"]) for _, row in tax.iterrows()}
    global_id_to_parent = {int(row["_critic_id"]): str(row["Parent_Term"]) for _, row in tax.iterrows()}
    _mutual_drop_guard(all_tax_edits, global_id_to_term, global_id_to_parent)
    cycle_reverts = _guard_reparent_cycles(tax, all_tax_edits)
    if cycle_reverts:
        log.warn(f"Reverted {len(cycle_reverts)} reparent(s) that would create taxonomy cycles")
    valid_aliases: dict[str, str] = {}
    for duplicate, survivor in reconciliation_aliases.items():
        duplicate_id = next(
            (rid for rid, term in global_id_to_term.items() if term.strip().lower() == duplicate), None
        )
        edit = all_tax_edits.get(duplicate_id, {}) if duplicate_id is not None else {}
        if (
            str(edit.get("action", "")).upper() == "DROP_AS_REDUNDANT"
            and str(edit.get("survivor", "")).strip().lower() == survivor.strip().lower()
        ):
            valid_aliases[duplicate] = survivor
    if rel is not None and valid_aliases:
        rel = _redirect_relation_aliases(rel, valid_aliases)

    cleaned_tax, instances_df, tax_log, bearer_records, explicit_defined_records, demotion_records = _apply_taxonomy_edits(
        tax, all_tax_edits, valid_categories
    )
    _finalize_reconciliation_audit(
        reconciliation_audit, cleaned_tax, valid_aliases,
    )
    reconciliation_df = pd.DataFrame(reconciliation_audit).reindex(columns=reconciliation_cols)
    write_csv(reconciliation_df, reconciliation_out)
    if reconciliation_audit:
        applied_count = sum(bool(row.get("Applied")) for row in reconciliation_audit)
        log.success(
            f"Term reconciliation: {applied_count} applied / {len(reconciliation_audit)} candidates "
            f"→ {reconciliation_out}"
        )

    completion_cols = [
        "Candidate_ID", "Candidate", "Parent_Term", "Category", "Search_Terms",
        "Document_Count", "Evidence_Documents", "Status", "Proposed_Reason",
        "Confidence", "NLD", "Decision_Reason", "Relation_Extraction_Pending",
    ]
    completion_df = pd.DataFrame(columns=completion_cols)
    if lateral_cfg.enabled and lateral_cfg.frame_completion_enabled and all_completion_candidates:
        existing_labels = {str(t).strip().lower() for t in tax["Term"].astype(str)}
        proposed = [
            c for c in all_completion_candidates
            if str(c.get("label", "")).strip().lower() not in existing_labels
            and str(c.get("centrality", "") or "").strip().lower() == "high"
            and _as_float(c.get("confidence", 0.0)) >= lateral_cfg.min_confidence_apply
        ]
        evidence_payload, completion_audit = _attest_completion_candidates(
            proposed, lateral_cfg.frame_completion_min_documents,
        )
        if lateral_cfg.frame_completion_auto_add:
            completion_decisions: list[dict] = []
            for start in range(0, len(evidence_payload), 5):
                completion_decisions.extend(_call_frame_completion_verifier(
                    evidence_payload[start:start + 5], completion_system, completion_template,
                    model, temperature, archive_path, archive_lock,
                ))
            completion_decisions = _generate_completion_nlds(evidence_payload, completion_decisions)
            cleaned_tax, completion_df = _apply_frame_completion(
                cleaned_tax, completion_audit, completion_decisions, valid_categories,
            )
        else:
            for row in completion_audit:
                if row["Status"] == "ATTESTED":
                    row["Decision_Reason"] = "diagnostic only; automatic frame completion disabled"
            completion_df = pd.DataFrame(completion_audit)
    completion_df = completion_df.reindex(columns=completion_cols)
    write_csv(completion_df, frame_completion_out)
    if len(completion_df):
        added_count = int((completion_df["Status"] == "ADDED").sum())
        mode = "auto-add" if lateral_cfg.frame_completion_auto_add else "diagnostic-only"
        log.success(
            f"Frame completion ({mode}): {added_count} added / "
            f"{len(completion_df)} candidates → {frame_completion_out}"
        )

    # Newly completed terms receive standard Step-6b extraction before the
    # relation critics run, so they participate in the same validation path.
    targeted_rel_df = pd.DataFrame(columns=[
        "Term", "Category", "Property", "Property_IRI", "Filler",
        "Filler_Source", "Confidence", "Evidence", "Validation_Status",
        "Validation_Reason",
    ])
    if rel is not None and not completion_df.empty:
        added_terms = completion_df[completion_df["Status"] == "ADDED"]["Candidate"].astype(str).tolist()
        if added_terms:
            from src.modules.construct.relation_extractor import extract_relations_for_terms
            added_rows = cleaned_tax[cleaned_tax["Term"].astype(str).isin(added_terms)]
            term_rows = [
                {"term": row["Term"], "nld": row["NLD"], "category": row["Category"]}
                for _, row in added_rows.iterrows()
            ]
            targeted_rel_df = extract_relations_for_terms(
                term_rows,
                cleaned_tax[["Term", "Category"]],
                rel.drop(columns=["_critic_id"], errors="ignore").to_dict("records"),
            )
            completion_df.loc[
                completion_df["Candidate"].astype(str).isin(added_terms),
                "Relation_Extraction_Pending",
            ] = False
            write_csv(completion_df, frame_completion_out)
    write_csv(targeted_rel_df, completion_relations_out)
    if rel is not None and not targeted_rel_df.empty:
        accepted_new = targeted_rel_df[targeted_rel_df["Validation_Status"] == "ACCEPTED"].copy()
        rel = pd.concat([
            rel.drop(columns=["_critic_id"], errors="ignore"), accepted_new,
        ], ignore_index=True, sort=False)
    if rel is not None:
        rel = rel.reset_index(drop=True)
        rel["_critic_id"] = range(len(rel))

    # Relation correctness/scope runs only after taxonomy and frame completion
    # are stable, with rich context for subjects and cross-category fillers.
    if rel is not None and len(rel):
        individual_terms = set(instances_df["Term"].astype(str)) if not instances_df.empty else set()
        individual_lower = {term.strip().lower() for term in individual_terms}
        for _, row in rel.iterrows():
            if str(row["Term"]).strip().lower() not in individual_lower:
                continue
            rid = int(row["_critic_id"])
            all_rel_edits[rid] = {
                "id": rid, "action": "KEEP",
                "reason": "accepted relation on converted named individual",
                "relation_scope": "individual_fact",
                "scope_reason": "subject was converted to owl:NamedIndividual",
                "scope_confidence": 1.0,
                "scope_needs_review": False,
            }
        relation_context_lookup = _build_entity_context_lookup(cleaned_tax, individual_terms)
        final_parent_lookup = _term_to_parent(cleaned_tax)
        final_categories = sorted(set(cleaned_tax["Category"].astype(str)))
        id_to_term_global = {int(row["_critic_id"]): str(row["Term"]) for _, row in tax.iterrows()}
        decisions_all = _build_taxonomy_decisions(all_tax_edits, id_to_term_global)

        def _critique_relation_category(cat: str) -> None:
            tax_group = cleaned_tax[cleaned_tax["Category"] == cat]
            subject_terms = {str(term).strip().lower() for term in tax_group["Term"].astype(str)}
            own_rel = rel[rel["Term"].astype(str).str.strip().str.lower().isin(subject_terms)]
            own_rel = own_rel[
                ~own_rel["Term"].astype(str).str.strip().str.lower().isin(individual_lower)
            ]
            if own_rel.empty:
                return
            ancestor_terms: set[str] = set()
            for term in tax_group["Term"].astype(str):
                for ancestor in _ancestor_chain(term, final_parent_lookup, set(final_categories)):
                    ancestor_terms.add(ancestor.strip().lower())
            ancestor_rel = rel[
                rel["Term"].astype(str).str.strip().str.lower().isin(ancestor_terms)
            ] if ancestor_terms else pd.DataFrame()
            rel_payload = _build_relation_payload(own_rel, ancestor_rel, relation_context_lookup)
            involved_keys = {
                str(row.get(field, "")).strip().lower()
                for row in rel_payload for field in ("term", "filler")
                if str(row.get(field, "")).strip()
            }
            tax_context = [
                relation_context_lookup[key] for key in involved_keys if key in relation_context_lookup
            ]
            relevant_decisions = [
                decision for decision in decisions_all
                if str(decision.get("term", "")).strip().lower() in subject_terms
            ]
            own_ids = {int(row["_critic_id"]) for _, row in own_rel.iterrows()}

            def _invoke_rel(payload: list[dict]) -> list[dict]:
                return _call_relation_critic(
                    cat, payload, relations_menu, previously_minted,
                    tax_context, relevant_decisions,
                    rel_system, rel_template, model, temperature,
                    archive_path, archive_lock,
                )

            rel_edits = _ask_complete(_invoke_rel, rel_payload, own_ids, cat, "relation")
            rel_edits_local = {
                edit["id"]: edit for edit in rel_edits
                if isinstance(edit, dict) and isinstance(edit.get("id"), int)
            }
            if lateral_cfg.enabled and lateral_cfg.relation_scope_enabled:
                scope_payload = _scope_payload_after_relation_edits(rel_payload, rel_edits_local)
                scope_ids = {int(row["id"]) for row in scope_payload if isinstance(row.get("id"), int)}

                def _invoke_scope(payload: list[dict]) -> list[dict]:
                    return _call_relation_scope_critic(
                        cat, payload, tax_context, relevant_decisions,
                        scope_system, scope_template, model, temperature,
                        archive_path, archive_lock,
                    )

                scope_decisions = _ask_complete(
                    _invoke_scope, scope_payload, scope_ids, cat, "relation-scope",
                ) if scope_payload else []
                scoped_ids: set[int] = set()
                for decision in scope_decisions:
                    if not (isinstance(decision, dict) and isinstance(decision.get("id"), int)):
                        continue
                    rid = decision["id"]
                    scoped_ids.add(rid)
                    current = dict(rel_edits_local.get(
                        rid, {"id": rid, "action": "KEEP", "reason": "scope-only KEEP"},
                    ))
                    current.update({
                        "relation_scope": _relation_scope(decision),
                        "scope_reason": decision.get("reason", ""),
                        "scope_confidence": decision.get("confidence", ""),
                        "scope_needs_review": decision.get("needs_review", False),
                    })
                    rel_edits_local[rid] = current
                for rid in scope_ids - scoped_ids:
                    current = dict(rel_edits_local.get(
                        rid, {"id": rid, "action": "KEEP", "reason": "scope fallback KEEP"},
                    ))
                    current.update({
                        "relation_scope": "corpus_context",
                        "scope_reason": "scope critic omitted row after retry; conservative non-generic fallback",
                        "scope_confidence": 0.0,
                        "scope_needs_review": True,
                    })
                    rel_edits_local[rid] = current
            with edits_lock:
                all_rel_edits.update(rel_edits_local)

        relation_workers = min(max_workers, len(final_categories)) if final_categories else 1
        with ThreadPoolExecutor(max_workers=relation_workers) as pool:
            relation_futures = {
                pool.submit(_critique_relation_category, cat): cat for cat in final_categories
            }
            with tqdm(total=len(final_categories), desc="Relation critic per category", unit="cat") as pbar:
                for future in as_completed(relation_futures):
                    cat = relation_futures[future]
                    try:
                        future.result()
                        pbar.set_postfix_str(cat[:30])
                    except Exception as e:
                        tqdm.write(f"  [error] Relation category '{cat}': {e}")
                    pbar.update(1)

    removed_taxonomy_terms = {
        str(entry["term"]).strip().lower()
        for entry in tax_log
        if entry["action"] in {
            "DROP_AS_MIXIN", "DROP_AS_REDUNDANT", "DROP_AS_OVER_SPECIFIC",
            "DEMOTE_TO_PROPERTY",
        }
    }

    # KEEP_AS_BEARER carries → minted filler classes (taxonomy) + companion
    # relations. Built once; appended to each output below.
    tax_cols = [c for c in cleaned_tax.columns if c != "_critic_id"]
    rel_cols = [c for c in rel.columns if c != "_critic_id"] if rel is not None else None
    if rel_cols is not None:
        for col in ("Relation_Scope", "Scope_Reason", "Scope_Confidence", "Scope_Needs_Review"):
            if col not in rel_cols:
                rel_cols.append(col)
    bearer_filler_df, bearer_rel_df = _materialize_bearer_carries(
        bearer_records, tax_cols, rel_cols
    )

    # Realizable KEEP_AS_BEARER carries → defined-class handoff for the emitter.
    defined_df = _build_defined_classes(bearer_records, explicit_defined_records, cleaned_tax)
    write_csv(defined_df, defined_out)
    if not defined_df.empty:
        log.success(f"Defined bearer classes: {len(defined_df)} rows → {defined_out}")

    cleaned_tax_out = cleaned_tax.drop(columns=["_critic_id"], errors="ignore")
    if not bearer_filler_df.empty:
        cleaned_tax_out = pd.concat([cleaned_tax_out, bearer_filler_df], ignore_index=True)
    write_csv(cleaned_tax_out, tax_out)
    _bearer_note = f", +{len(bearer_filler_df)} bearer-role fillers" if not bearer_filler_df.empty else ""
    log.success(f"Cleaned taxonomy: {len(cleaned_tax_out)} rows (was {len(tax)}{_bearer_note}) → {tax_out}")

    write_csv(instances_df, instances_out)
    if not instances_df.empty:
        log.success(f"Converted to instances: {len(instances_df)} rows → {instances_out}")
    demotion_cols = [
        "Term", "Base_Class", "Property", "Filler", "Rationale",
        "Original_Parent", "Original_Category",
    ]
    demotion_df = pd.DataFrame(demotion_records, columns=demotion_cols)
    write_csv(demotion_df, demotions_out)
    if demotion_records:
        log.success(f"Demoted class distinctions: {len(demotion_records)} rows → {demotions_out}")

    rel_log: list[dict] = []
    minted_collector: list[dict] = []
    if rel is not None and rel_out:
        cleaned_rel, rel_log = _apply_relation_edits(rel, all_rel_edits, minted_collector)
        if removed_taxonomy_terms:
            phantom_filler_mask = cleaned_rel["Filler"].astype(str).str.strip().str.lower().isin(removed_taxonomy_terms)
            phantom_subject_mask = cleaned_rel["Term"].astype(str).str.strip().str.lower().isin(removed_taxonomy_terms)
            phantom_mask = phantom_filler_mask | phantom_subject_mask
            phantom_rows = cleaned_rel[phantom_mask]
            for _, r in phantom_rows.iterrows():
                endpoint = "subject" if str(r["Term"]).strip().lower() in removed_taxonomy_terms else "filler"
                missing_term = r["Term"] if endpoint == "subject" else r["Filler"]
                rel_log.append({
                    "id": int(r.get("_critic_id", -1)),
                    "kind": "relation",
                    "category": r.get("Category", ""),
                    "term": r["Term"],
                    "action": "DROP",
                    "reason": f"(phantom-{endpoint} cleanup — {endpoint} '{missing_term}' removed from taxonomy)",
                })
            cleaned_rel = cleaned_rel[~phantom_mask]
        # Finalise: re-specialise mereology + re-validate domain/range after the critic.
        cleaned_rel, reval_drops, n_norm = _normalize_and_revalidate_relations(
            cleaned_rel, _term_to_category(tax)
        )
        if n_norm:
            log.detail(f"Re-normalised {n_norm} mereological propert{'y' if n_norm == 1 else 'ies'} post-critic")
        if reval_drops:
            rel_log.extend(reval_drops)
            log.warn(f"Post-critic re-validation dropped {len(reval_drops)} BFO-invalid relation(s)")
        cleaned_rel = cleaned_rel.drop(columns=["_critic_id"], errors="ignore")
        if not bearer_rel_df.empty:
            cleaned_rel = pd.concat([cleaned_rel, bearer_rel_df], ignore_index=True)
        write_csv(cleaned_rel, rel_out)
        _comp_note = f", +{len(bearer_rel_df)} bearer companions" if not bearer_rel_df.empty else ""
        log.success(f"Cleaned relations: {len(cleaned_rel)} rows (was {len(rel)}{_comp_note}) → {rel_out}")

    if minted_collector:
        # Merge with any existing minted CSV (preserve prior runs' mints).
        existing_minted_df = read_csv(minted_out) if os.path.exists(minted_out) else pd.DataFrame()
        new_minted_df = pd.DataFrame(minted_collector)
        merged = pd.concat([existing_minted_df, new_minted_df], ignore_index=True)
        # Deduplicate on IRI (latest wins).
        if "IRI" in merged.columns:
            merged = merged.drop_duplicates(subset=["IRI"], keep="last")
        write_csv(merged, minted_out)
        log.success(f"Minted properties: +{len(new_minted_df)} (total {len(merged)}) → {minted_out}")

    facet_cols = ["category", "parent", "axis", "members", "frame_type", "reason"]
    facet_df = pd.DataFrame(all_facet_frames).reindex(columns=facet_cols)
    write_csv(facet_df, facet_frames_out)
    if all_facet_frames:
        log.success(f"Facet frames: {len(all_facet_frames)} rows → {facet_frames_out}")
    disjoint_cols = ["category", "parent", "members", "confidence", "needs_review", "reason"]
    disjoint_df = pd.DataFrame(all_disjointness).reindex(columns=disjoint_cols)
    write_csv(disjoint_df, disjointness_out)
    if all_disjointness:
        log.success(f"Disjointness diagnostics: {len(all_disjointness)} rows → {disjointness_out}")
    subsumption_cols = [
        "category", "id", "term", "action", "old_parent", "new_parent",
        "confidence", "needs_review", "reason", "applied",
    ]
    subsumption_df = pd.DataFrame(all_subsumption_hints).reindex(columns=subsumption_cols)
    write_csv(subsumption_df, subsumption_out)
    if all_subsumption_hints:
        log.success(f"Subsumption/frame edits: {len(all_subsumption_hints)} rows → {subsumption_out}")

    write_csv(pd.DataFrame(tax_log + rel_log), edits_out)
    fate_cols = [
        "id", "term", "category", "action", "proposed_fate", "class_fate", "drop_basis", "centrality",
        "cross_axis", "placement_rationale", "over_specificity_reason",
        "needs_review", "confidence", "reason",
    ]
    fate_df = pd.DataFrame(tax_log)
    if not fate_df.empty:
        fate_df = fate_df.reindex(columns=fate_cols)
        write_csv(fate_df, class_fates_out)
        log.success(f"Class fates: {len(fate_df)} rows → {class_fates_out}")
    by_action: dict[str, int] = {}
    for entry in tax_log + rel_log:
        by_action[entry["action"]] = by_action.get(entry["action"], 0) + 1
    log.success(
        f"Audit log: {len(tax_log) + len(rel_log)} decisions "
        f"({', '.join(f'{a}={n}' for a, n in sorted(by_action.items()))}) → {edits_out}"
    )
    scope_counts: dict[str, int] = {}
    if rel is not None and rel_out and "Relation_Scope" in cleaned_rel.columns:
        for scope in cleaned_rel["Relation_Scope"].fillna("generic").astype(str):
            scope = scope.strip().lower() or "generic"
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
    fate_counts: dict[str, int] = {}
    review_count = 0
    for entry in tax_log:
        fate = str(entry.get("class_fate", "") or "unspecified")
        fate_counts[fate] = fate_counts.get(fate, 0) + 1
        if str(entry.get("needs_review", "")).lower() == "true":
            review_count += 1
    completion_counts = completion_df["Status"].value_counts().to_dict() if not completion_df.empty else {}
    summary = {
        "taxonomy_input_rows": len(tax),
        "taxonomy_output_rows": len(cleaned_tax_out),
        "relation_input_rows": len(rel) if rel is not None else 0,
        "relation_output_rows": len(cleaned_rel) if rel is not None and rel_out else 0,
        "action_counts": by_action,
        "class_fate_counts": fate_counts,
        "needs_review_count": review_count,
        "defined_class_count": len(defined_df),
        "demotion_count": len(demotion_records),
        "relation_scope_counts": scope_counts,
        "facet_frame_count": len(all_facet_frames),
        "subsumption_edit_count": len(all_subsumption_hints),
        "reconciliation_candidate_count": len(reconciliation_audit),
        "reconciliation_applied_count": sum(bool(row.get("Applied")) for row in reconciliation_audit),
        "disjointness_candidate_count": len(all_disjointness),
        "frame_completion_counts": completion_counts,
        "frame_completion_auto_add": lateral_cfg.frame_completion_auto_add,
        "frame_completion_relation_rows": len(targeted_rel_df),
        "weak_observations_enabled": weak_observations_enabled,
    }
    with open(lateral_summary_out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    log.success(f"Lateral-coherence summary → {lateral_summary_out}")
    log.detail(f"Raw responses archived → {archive_path}")

    return tax_out, rel_out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Ontology critic — taxonomy + relation passes (validate verb)")
    p.add_argument("taxonomy_csv")
    p.add_argument("output_dir")
    p.add_argument("--relations", default=None)
    args = p.parse_args()
    run_critic(args.taxonomy_csv, args.output_dir, relations_csv=args.relations)
