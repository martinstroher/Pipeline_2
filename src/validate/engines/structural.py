"""Deterministic validation engine — pure-Python per-rule evaluators.

This module exposes a name-keyed registry (`RULE_FUNCTIONS`) of pure
functions that each consume one subject (an edge or a term) plus a small
context and return a `Verdict`. The dispatcher in
`src/validate/rule_applier.py` (and the dispatcher for engineering
filters) calls these functions by id; the rule IDs match those in
`src/validate/rules.yaml` and `domains/<name>/domain_filters.yaml`.

No LLM calls happen here. Use `engines.llm.evaluate()` for LLM rules and
`engines.hybrid` helpers for embedding-based pre-clustering.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from src.utils.ontology_config import get_config
from src.utils.relation_validator import (
    PROPERTY_CONSTRAINTS,
    get_metatypes,
    validate_relation,
)


# ─── Verdict schema ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Verdict:
    """Outcome of evaluating one rule against one subject."""

    rule_id: str
    subject_type: str  # "term" | "edge_taxonomy" | "edge_relation"
    subject_id: str  # term name or "<term>--><parent>" or "<term>--<property>--><filler>"
    verdict: str  # "PASS" | "FAIL" | "ABSTAIN"
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "evidence": self.evidence,
        }


# ─── Context the engine receives ───────────────────────────────────────


@dataclass
class StructuralContext:
    """Bag of pre-loaded artifacts the structural engine reads.

    Built once per validation run; passed to every rule function. Callers
    populate only the fields a given rule needs (others stay `None`).
    """

    taxonomy_df: pd.DataFrame | None = None  # construct_taxonomy.csv
    relations_df: pd.DataFrame | None = None  # construct_relations.csv (ACCEPTED filtered)
    cq_covered_terms: set[str] | None = None  # lower-cased term names with ≥1 active CQ
    cfg: Any = None  # OntologyConfig; defaults to get_config()

    def __post_init__(self):
        if self.cfg is None:
            self.cfg = get_config()


# ─── Cached config helpers ─────────────────────────────────────────────

_CFG = get_config()
_DISJOINT_METATYPE_PAIRS = _CFG.disjoint_metatype_pairs()
_GENERIC_METATYPES = _CFG.non_distinguishing_metatypes()
_CATEGORY_TO_METATYPES = _CFG.category_to_metatypes()
_PROPERTY_LOOKUP = {k.lower(): k for k in PROPERTY_CONSTRAINTS}


def _edge_tax_id(term: str, parent: str) -> str:
    return f"{term}-->{parent}"


def _edge_rel_id(term: str, prop: str, filler: str) -> str:
    return f"{term}--{prop}-->{filler}"


# ─── Rule: bfo_disjointness ────────────────────────────────────────────


def bfo_disjointness(edge: dict, ctx: StructuralContext) -> Verdict:
    """REJECT a taxonomy edge whose child + parent metatypes straddle a BFO disjoint pair.

    `edge` must carry: term, parent_term, category (of child), parent_category.
    Each side contributes its category's metatype set; if the UNION contains
    both members of any declared disjoint pair, the edge is incoherent.
    """
    term = str(edge.get("term", "")).strip()
    parent = str(edge.get("parent_term", "")).strip()
    child_cat = str(edge.get("category", "")).strip()
    parent_cat = str(edge.get("parent_category", "")).strip()
    sid = _edge_tax_id(term, parent)

    child_meta = get_metatypes(child_cat) or frozenset()
    parent_meta = get_metatypes(parent_cat) or frozenset()
    union = child_meta | parent_meta

    if not union:
        return Verdict(
            rule_id="bfo_disjointness",
            subject_type="edge_taxonomy",
            subject_id=sid,
            verdict="ABSTAIN",
            reason="Neither side resolves to a known upper-ontology category.",
            evidence={"child_category": child_cat, "parent_category": parent_cat},
        )

    for pair in _DISJOINT_METATYPE_PAIRS:
        if pair <= union:
            a, b = sorted(pair)
            return Verdict(
                rule_id="bfo_disjointness",
                subject_type="edge_taxonomy",
                subject_id=sid,
                verdict="FAIL",
                reason=f"Disjoint metatype pair {{{a}, {b}}} present across child+parent.",
                evidence={
                    "child_metatypes": sorted(child_meta),
                    "parent_metatypes": sorted(parent_meta),
                    "disjoint_pair": [a, b],
                },
            )

    return Verdict(
        rule_id="bfo_disjointness",
        subject_type="edge_taxonomy",
        subject_id=sid,
        verdict="PASS",
        reason="No disjoint pair violated.",
    )


# ─── Rule: property_domain_range ───────────────────────────────────────


def property_domain_range(edge: dict, ctx: StructuralContext) -> Verdict:
    """REJECT a relation whose subject/filler categories violate the property's domain/range."""
    term = str(edge.get("term", "")).strip()
    prop_raw = str(edge.get("property", "")).strip()
    filler = str(edge.get("filler", "")).strip()
    subj_cat = str(edge.get("category", "")).strip()
    obj_cat = str(edge.get("filler_category", "")).strip()
    sid = _edge_rel_id(term, prop_raw, filler)

    prop_name = _PROPERTY_LOOKUP.get(prop_raw.lower())
    if prop_name is None:
        return Verdict(
            rule_id="property_domain_range",
            subject_type="edge_relation",
            subject_id=sid,
            verdict="ABSTAIN",
            reason=f"Property '{prop_raw}' not in active constraint set.",
            evidence={"property": prop_raw},
        )

    ok, reason = validate_relation(subj_cat, prop_name, obj_cat)
    return Verdict(
        rule_id="property_domain_range",
        subject_type="edge_relation",
        subject_id=sid,
        verdict="PASS" if ok else "FAIL",
        reason=reason,
        evidence={
            "property": prop_name,
            "subject_category": subj_cat,
            "object_category": obj_cat,
        },
    )


# ─── Rule: relation_evidence_refinement ────────────────────────────────


def _distinguishing(metaset: frozenset[str]) -> frozenset[str]:
    return metaset - _GENERIC_METATYPES


def _is_strict_subclass(candidate: frozenset[str], current: frozenset[str]) -> bool:
    return current < candidate


def _refined_target_category(
    implied: frozenset[str], current_cat: str
) -> tuple[str | None, str]:
    """Return (new_category_or_None, reason). Strict-subclass refinement only."""
    if not implied:
        return None, "Evidence intersection is empty (contradiction)."
    current_meta = get_metatypes(current_cat)
    if current_meta is None:
        return None, f"Current category '{current_cat}' not in upper ontology."

    candidates: list[tuple[str, frozenset[str], int]] = []
    for cat_name, cat_meta in _CATEGORY_TO_METATYPES.items():
        if cat_name.lower() == current_cat.lower():
            continue
        overlap = cat_meta & implied
        if not overlap:
            continue
        if not _is_strict_subclass(cat_meta, current_meta):
            continue
        added = (cat_meta - current_meta) - _GENERIC_METATYPES
        if added and not added.issubset(implied):
            continue
        candidates.append((cat_name, cat_meta, len(_distinguishing(overlap))))
    if not candidates:
        return None, "No strict-subclass candidate consistent with evidence."
    candidates.sort(key=lambda c: (-c[2], len(c[1]), -len(c[0])))
    return candidates[0][0], f"Refined to strict subclass {candidates[0][0]}."


def relation_evidence_refinement(term_row: dict, ctx: StructuralContext) -> Verdict:
    """Refine a term's category to a strict-subclass when ≥2 accepted relations agree."""
    term = str(term_row.get("term", "")).strip()
    current_cat = str(term_row.get("category", "")).strip()
    min_evidence = 2

    if ctx.relations_df is None or ctx.relations_df.empty:
        return Verdict(
            rule_id="relation_evidence_refinement",
            subject_type="term",
            subject_id=term,
            verdict="ABSTAIN",
            reason="No relations available.",
        )

    df = ctx.relations_df
    tl = term.lower()
    as_subj = df[df["Term"].str.lower() == tl]
    as_fill = df[df["Filler"].str.lower() == tl]

    evidence: list[frozenset[str]] = []
    for _, r in as_subj.iterrows():
        pn = _PROPERTY_LOOKUP.get(str(r["Property"]).lower())
        if pn:
            evidence.append(PROPERTY_CONSTRAINTS[pn].domain)
    for _, r in as_fill.iterrows():
        pn = _PROPERTY_LOOKUP.get(str(r["Property"]).lower())
        if pn:
            evidence.append(PROPERTY_CONSTRAINTS[pn].range)

    if len(evidence) < min_evidence:
        return Verdict(
            rule_id="relation_evidence_refinement",
            subject_type="term",
            subject_id=term,
            verdict="PASS",
            reason=f"Insufficient evidence ({len(evidence)} < {min_evidence}).",
            evidence={"evidence_count": len(evidence)},
        )

    implied = evidence[0]
    for e in evidence[1:]:
        implied = implied & e

    new_cat, reason = _refined_target_category(implied, current_cat)
    if new_cat is None:
        return Verdict(
            rule_id="relation_evidence_refinement",
            subject_type="term",
            subject_id=term,
            verdict="PASS",
            reason=reason,
            evidence={"implied_metatypes": sorted(implied), "evidence_count": len(evidence)},
        )
    return Verdict(
        rule_id="relation_evidence_refinement",
        subject_type="term",
        subject_id=term,
        verdict="FAIL",
        reason=reason,
        evidence={
            "current_category": current_cat,
            "new_category": new_cat,
            "implied_metatypes": sorted(implied),
            "evidence_count": len(evidence),
        },
    )


# ─── Rule: cq_coverage_gate ────────────────────────────────────────────


def cq_coverage_gate(term_row: dict, ctx: StructuralContext) -> Verdict:
    """Demote a term from class-status when no active CQ touches it."""
    term = str(term_row.get("term", "")).strip()
    if ctx.cq_covered_terms is None:
        return Verdict(
            rule_id="cq_coverage_gate",
            subject_type="term",
            subject_id=term,
            verdict="ABSTAIN",
            reason="No CQ coverage data supplied.",
        )
    if term.lower() in ctx.cq_covered_terms:
        return Verdict(
            rule_id="cq_coverage_gate",
            subject_type="term",
            subject_id=term,
            verdict="PASS",
            reason="Term covered by ≥1 active competency question.",
        )
    return Verdict(
        rule_id="cq_coverage_gate",
        subject_type="term",
        subject_id=term,
        verdict="FAIL",
        reason="No active CQ references this term.",
    )


# ─── Domain structural filters ─────────────────────────────────────────


def _index_children(taxonomy_df: pd.DataFrame) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for _, r in taxonomy_df.iterrows():
        parent = str(r.get("Parent_Term", "")).strip().lower()
        term = str(r.get("Term", "")).strip()
        if parent and term:
            children[parent].append(term)
    return children


def single_child_intermediate(term_row: dict, ctx: StructuralContext) -> Verdict:
    """REMOVE an intermediate node whose only purpose is to wrap one child."""
    term = str(term_row.get("term", "")).strip()
    if ctx.taxonomy_df is None:
        return Verdict("single_child_intermediate", "term", term, "ABSTAIN", "No taxonomy supplied.")
    is_inter = bool(term_row.get("is_intermediate", False))
    if not is_inter:
        return Verdict("single_child_intermediate", "term", term, "PASS", "Not an intermediate node.")
    children = _index_children(ctx.taxonomy_df).get(term.lower(), [])
    if len(children) == 1:
        return Verdict(
            "single_child_intermediate",
            "term",
            term,
            "FAIL",
            f"Intermediate node has exactly one child ({children[0]}).",
            {"children": children},
        )
    return Verdict(
        "single_child_intermediate",
        "term",
        term,
        "PASS",
        f"Intermediate node has {len(children)} children.",
    )


def orphan_intermediate(term_row: dict, ctx: StructuralContext) -> Verdict:
    """REMOVE an intermediate node with no children at all."""
    term = str(term_row.get("term", "")).strip()
    if ctx.taxonomy_df is None:
        return Verdict("orphan_intermediate", "term", term, "ABSTAIN", "No taxonomy supplied.")
    is_inter = bool(term_row.get("is_intermediate", False))
    if not is_inter:
        return Verdict("orphan_intermediate", "term", term, "PASS", "Not an intermediate node.")
    children = _index_children(ctx.taxonomy_df).get(term.lower(), [])
    if not children:
        return Verdict(
            "orphan_intermediate",
            "term",
            term,
            "FAIL",
            "Intermediate node has no remaining children.",
        )
    return Verdict("orphan_intermediate", "term", term, "PASS", f"{len(children)} children present.")


def self_loop(edge: dict, ctx: StructuralContext) -> Verdict:
    """REPARENT a row whose Term equals its Parent_Term."""
    term = str(edge.get("term", "")).strip()
    parent = str(edge.get("parent_term", "")).strip()
    sid = _edge_tax_id(term, parent)
    if term.lower() == parent.lower():
        return Verdict(
            "self_loop",
            "edge_taxonomy",
            sid,
            "FAIL",
            "Term equals Parent_Term (self-loop).",
        )
    return Verdict("self_loop", "edge_taxonomy", sid, "PASS", "Distinct term and parent.")


def broken_parent(edge: dict, ctx: StructuralContext) -> Verdict:
    """REPARENT a row whose Parent_Term is not present as a Term in the taxonomy."""
    term = str(edge.get("term", "")).strip()
    parent = str(edge.get("parent_term", "")).strip()
    sid = _edge_tax_id(term, parent)
    if ctx.taxonomy_df is None:
        return Verdict("broken_parent", "edge_taxonomy", sid, "ABSTAIN", "No taxonomy supplied.")
    # Upper-ontology category roots are valid parents even without a Term row;
    # caller supplies known_categories via context if needed. Here we accept
    # any parent that appears either as a Term or as a Category value.
    terms = set(ctx.taxonomy_df["Term"].astype(str).str.lower())
    cats = set(ctx.taxonomy_df.get("Category", pd.Series(dtype=str)).astype(str).str.lower())
    valid = terms | cats
    if parent.lower() in valid or not parent:
        return Verdict("broken_parent", "edge_taxonomy", sid, "PASS", "Parent resolves.")
    return Verdict(
        "broken_parent",
        "edge_taxonomy",
        sid,
        "FAIL",
        f"Parent_Term '{parent}' not found in taxonomy.",
    )


# ─── Registry ──────────────────────────────────────────────────────────


RULE_FUNCTIONS: dict[str, Callable[[dict, StructuralContext], Verdict]] = {
    # Cross-domain rules (from src/validate/rules.yaml)
    "bfo_disjointness": bfo_disjointness,
    "property_domain_range": property_domain_range,
    "relation_evidence_refinement": relation_evidence_refinement,
    "cq_coverage_gate": cq_coverage_gate,
    # Domain structural filters (from domains/<name>/domain_filters.yaml)
    "single_child_intermediate": single_child_intermediate,
    "orphan_intermediate": orphan_intermediate,
    "self_loop": self_loop,
    "broken_parent": broken_parent,
}
