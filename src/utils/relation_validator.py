"""
Relation Validator — BFO domain/range constraint checking for extracted relations.

Validates (subject_category, property, object_category) triples against
BFO 2020 / RO domain–range axioms before they are added to the OWL ontology.

Reference specifications:
  - BFO 2020: https://basic-formal-ontology.org/bfo-2020.html
  - OBO Relation Ontology: https://oborelations.github.io/
  - GeoCore (Abel et al. 2015)
  - GeoReservoir (Abel et al.)

Design:
  1. Map each PreSaltOntoLearn category → BFO metatype set
  2. Map each property → allowed domain metatypes × range metatypes
  3. validate_relation() checks membership
  4. Additional heuristic filters beyond domain/range
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.ontology_config import PropertyConstraint, get_config

# ────────────────────────────────────────────────────────────────────────
# §1  BFO Metatypes — the coarse-grained ontological kinds
# ────────────────────────────────────────────────────────────────────────

class BFOMeta:
    """String constants for BFO 2020 metatype buckets."""
    # Top
    CONTINUANT = "Continuant"
    OCCURRENT = "Occurrent"
    # Continuant subtypes
    INDEPENDENT_CONTINUANT = "IndependentContinuant"
    MATERIAL_ENTITY = "MaterialEntity"
    IMMATERIAL_ENTITY = "ImmaterialEntity"
    OBJECT = "Object"
    SITE = "Site"
    SDC = "SpecificallyDependentContinuant"  # quality, role, disposition
    QUALITY = "Quality"
    RELATIONAL_QUALITY = "RelationalQuality"
    DISPOSITION = "Disposition"
    GDC = "GenericallyDependentContinuant"
    SPATIAL_REGION = "SpatialRegion"
    # Occurrent subtypes
    PROCESS = "Process"
    PROCESS_BOUNDARY = "ProcessBoundary"
    HISTORY = "History"
    TEMPORAL_REGION = "TemporalRegion"
    TEMPORAL_INSTANT = "TemporalInstant"
    SPATIOTEMPORAL_REGION = "SpatiotemporalRegion"


# ────────────────────────────────────────────────────────────────────────
# §2  Category → BFO Metatype Mapping
# ────────────────────────────────────────────────────────────────────────
# Sourced from `ontology_config.yaml`: each category's `metatypes` field
# is interned as a frozenset. The set ordering (most specific → general)
# is preserved by the YAML author; the validator only checks membership.
#
# To add/remove a category or its metatypes, edit the YAML — DO NOT
# hardcode entries here.
_CATEGORY_TO_METATYPES: dict[str, frozenset[str]] = get_config().category_to_metatypes()

# Case-insensitive lookup
_CATEGORY_LOOKUP: dict[str, frozenset[str]] = {
    k.lower(): v for k, v in _CATEGORY_TO_METATYPES.items()
}


def get_metatypes(category: str) -> frozenset[str] | None:
    """Return BFO metatype set for a category, or None if unknown."""
    return _CATEGORY_LOOKUP.get(category.strip().lower())


# ------------------------------------------------------------------------
# §2b  Mereological property specialization (single source of truth)
# ------------------------------------------------------------------------
# Generic mereology (`has_part` / `part_of`) is split into disjoint BFO
# branches (continuant vs occurrent) that must not be mixed. The split is a
# pure function of the subject's and filler's metatypes, declared in
# `property_specializations:`. Both the extraction step (LLM emits generic)
# and the validate step (re-normalise after the critic) call these helpers so
# the logic lives in exactly one place.

# Reverse map {specialized_name -> generic_parent}, built once from config.
_SPECIALIZED_TO_GENERIC: dict[str, str] = {
    rule.specialize_to: spec.generic
    for spec in get_config().property_specializations()
    for rule in spec.rules
    if rule.specialize_to
}


def genericize_property(property_name: str) -> str:
    """Map a specialized parthood property back to its generic parent
    (`has_continuant_part` -> `has_part`); return unchanged if not specialized."""
    return _SPECIALIZED_TO_GENERIC.get(property_name, property_name)


def specialize_property(
    property_name: str,
    subject_cat: str,
    filler_cat: str | None,
) -> str:
    """Apply `property_specializations:` rules to a generic property.

    The first rule whose `when:` predicates match the subject/filler metatypes
    wins. Returns `property_name` unchanged if it has no rules, if no rule
    matches (e.g. a forbidden mixed continuant/occurrent pair — left generic so
    the validator rejects it), or if the filler's metatypes are unresolvable.
    """
    subj_meta = get_metatypes(subject_cat) or frozenset()
    filler_meta = get_metatypes(filler_cat) if filler_cat else frozenset()
    if not filler_meta:
        return property_name
    for spec in get_config().property_specializations():
        if spec.generic != property_name:
            continue
        for rule in spec.rules:
            if rule.matches(subj_meta, filler_meta):
                return rule.specialize_to
        break
    return property_name


def normalize_property(
    property_name: str,
    subject_cat: str,
    filler_cat: str | None,
) -> str:
    """Idempotent mereology normaliser: genericize any specialized parthood,
    then re-specialize for the current subject/filler categories.

    Safe to call on any property at any pipeline stage. Non-mereological
    properties (no specialization rules) pass through unchanged.
    """
    return specialize_property(genericize_property(property_name), subject_cat, filler_cat)



# ------------------------------------------------------------------------
# §3  Property Domain/Range Constraints (loaded from ontology_config.yaml)
# ------------------------------------------------------------------------
# All 71 constraints are defined declaratively in ontology_config.yaml under
# the relations: section, with per-entry provenance tagging (owl_axiom /
# ro_release / bfo_shape_axiom / critic_minted). Entries are filtered at
# load time against the active provenance tier set
# (env var RELATION_PROVENANCE_TIERS, default: all four tiers active).
#
# To add or modify constraints, edit ontology_config.yaml — DO NOT hardcode
# entries here. The PropertyConstraint dataclass lives in ontology_config.

PROPERTY_CONSTRAINTS: dict[str, PropertyConstraint] = get_config().property_constraints()

# Internal metatype groupings used by validate_relation() cross-category check.
# Sourced from the same metatype_groups: block in YAML to stay consistent.
_CONTINUANT: frozenset[str] = get_config().metatype_groups["CONTINUANT"]
_OCCURRENT: frozenset[str] = get_config().metatype_groups["OCCURRENT"]



# ────────────────────────────────────────────────────────────────────────
# §4  Core Validation Function
# ────────────────────────────────────────────────────────────────────────

def validate_relation(
    subject_cat: str,
    property_name: str,
    object_cat: str,
) -> tuple[bool, str]:
    """
    Check whether (subject_category, property, object_category) is valid
    per BFO 2020 domain/range axioms.

    Returns:
        (is_valid, reason) — reason explains the verdict.
    """
    # 1. Property must be known
    prop = PROPERTY_CONSTRAINTS.get(property_name)
    if prop is None:
        return False, f"Unknown property '{property_name}'."

    # 2. Categories must be mapped
    subj_meta = get_metatypes(subject_cat)
    if subj_meta is None:
        return False, f"Unknown subject category '{subject_cat}'."

    obj_meta = get_metatypes(object_cat)
    if obj_meta is None:
        return False, f"Unknown object category '{object_cat}'."

    # 3. Domain check: subject metatypes ∩ property domain ≠ ∅
    if not (subj_meta & prop.domain):
        return False, (
            f"Domain violation: '{subject_cat}' has metatypes "
            f"{sorted(subj_meta)} which do not intersect "
            f"allowed domain {sorted(prop.domain)} for '{property_name}'."
        )

    # 4. Range check: object metatypes ∩ property range ≠ ∅
    if not (obj_meta & prop.range):
        return False, (
            f"Range violation: '{object_cat}' has metatypes "
            f"{sorted(obj_meta)} which do not intersect "
            f"allowed range {sorted(prop.range)} for '{property_name}'."
        )

    # 5. Special cross-category check for has_part / part_of
    #    BFO forbids Continuant has_part Occurrent and vice versa.
    if property_name in ("has_part", "part_of"):
        subj_is_cont = bool(subj_meta & _CONTINUANT)
        subj_is_occ = bool(subj_meta & _OCCURRENT)
        obj_is_cont = bool(obj_meta & _CONTINUANT)
        obj_is_occ = bool(obj_meta & _OCCURRENT)

        # If subject is purely continuant and object is purely occurrent (or vice versa)
        if subj_is_cont and not subj_is_occ and obj_is_occ and not obj_is_cont:
            return False, (
                f"Cross-category mereology: '{subject_cat}' (Continuant) "
                f"cannot have '{property_name}' relation with "
                f"'{object_cat}' (Occurrent). BFO forbids this."
            )
        if subj_is_occ and not subj_is_cont and obj_is_cont and not obj_is_occ:
            return False, (
                f"Cross-category mereology: '{subject_cat}' (Occurrent) "
                f"cannot have '{property_name}' relation with "
                f"'{object_cat}' (Continuant). BFO forbids this."
            )

    return True, "Valid per BFO domain/range constraints."


# ────────────────────────────────────────────────────────────────────────
# §5  Heuristic Filters (beyond domain/range)
# ────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Full validation result for a single extracted relation."""
    is_valid: bool
    domain_range_ok: bool
    reason: str
    warnings: list[str]


def validate_relation_full(
    subject_term: str,
    subject_cat: str,
    property_name: str,
    object_term: str,
    object_cat: str,
    confidence: float,
    evidence: str,
    nld_text: str = "",
    existing_relations: set[tuple[str, str, str]] | None = None,
    taxonomy_parents: dict[str, str] | None = None,
    confidence_threshold: float = 0.7,
) -> ValidationResult:
    """
    Full validation: domain/range + heuristic checks.

    Args:
        subject_term: The subject term string.
        subject_cat: Category of the subject term.
        property_name: The BFO/RO property name.
        object_term: The object/filler term string.
        object_cat: Category of the object term.
        confidence: LLM-assigned confidence score (0.0–1.0).
        evidence: The NLD fragment cited as evidence.
        nld_text: The full NLD text (for evidence verification).
        existing_relations: Set of (subj, prop, obj) already accepted.
        taxonomy_parents: Dict mapping term → parent_term from taxonomy.
        confidence_threshold: Minimum confidence to accept (default 0.7).

    Returns:
        ValidationResult with is_valid, reason, and warnings.
    """
    warnings: list[str] = []

    # ── Check 1: Domain/range ──
    dr_valid, dr_reason = validate_relation(subject_cat, property_name, object_cat)
    if not dr_valid:
        return ValidationResult(
            is_valid=False,
            domain_range_ok=False,
            reason=dr_reason,
            warnings=warnings,
        )

    # ── Check 2: Self-relation ──
    if subject_term.strip().lower() == object_term.strip().lower():
        return ValidationResult(
            is_valid=False,
            domain_range_ok=True,
            reason=f"Self-relation: '{subject_term}' cannot have "
                   f"'{property_name}' relation with itself.",
            warnings=warnings,
        )

    # ── Check 3: Confidence threshold ──
    if confidence < confidence_threshold:
        return ValidationResult(
            is_valid=False,
            domain_range_ok=True,
            reason=f"Confidence {confidence:.2f} below threshold "
                   f"{confidence_threshold:.2f}.",
            warnings=warnings,
        )

    # ── Check 4: Inverse redundancy ──
    if existing_relations is not None:
        prop_constraint = PROPERTY_CONSTRAINTS.get(property_name)
        if prop_constraint and prop_constraint.inverse:
            inverse_triple = (object_term, prop_constraint.inverse, subject_term)
            if inverse_triple in existing_relations:
                return ValidationResult(
                    is_valid=False,
                    domain_range_ok=True,
                    reason=f"Inverse redundancy: '{object_term} "
                           f"{prop_constraint.inverse} {subject_term}' "
                           f"already exists.",
                    warnings=warnings,
                )
        # Exact duplicate
        forward_triple = (subject_term, property_name, object_term)
        if forward_triple in existing_relations:
            return ValidationResult(
                is_valid=False,
                domain_range_ok=True,
                reason=f"Duplicate: relation already exists.",
                warnings=warnings,
            )

    # ── Check 5: Taxonomy redundancy (has_part overlapping subClassOf) ──
    if taxonomy_parents is not None and property_name in ("has_part", "part_of"):
        if property_name == "has_part":
            child, parent = object_term, subject_term
        else:
            child, parent = subject_term, object_term
        # If the "part" is already a subClassOf the "whole", this is
        # suspicious — parthood and subsumption are different relations
        # but LLMs frequently confuse them
        if taxonomy_parents.get(child, "").strip().lower() == parent.strip().lower():
            warnings.append(
                f"Taxonomy overlap: '{child}' is already rdfs:subClassOf "
                f"'{parent}' in the taxonomy. Verify this is genuinely "
                f"parthood, not subsumption."
            )

    # ── Check 6: Evidence quality ──
    if nld_text and evidence:
        # Normalize whitespace for fuzzy match
        evidence_norm = " ".join(evidence.lower().split())
        nld_norm = " ".join(nld_text.lower().split())
        if evidence_norm not in nld_norm:
            # Check for partial overlap (at least 60% of evidence words in NLD)
            ev_words = set(evidence_norm.split())
            nld_words = set(nld_norm.split())
            overlap = len(ev_words & nld_words) / max(len(ev_words), 1)
            if overlap < 0.6:
                warnings.append(
                    f"Evidence mismatch: quoted evidence not found in NLD "
                    f"(word overlap {overlap:.0%}). Possible hallucination."
                )
            else:
                warnings.append(
                    f"Evidence is paraphrased (not exact quote from NLD). "
                    f"Word overlap: {overlap:.0%}."
                )

    # ── Check 7: GDC characteristic_of warning ──
    # Geological Structure, Facies, etc. are GDCs in GeoCore.
    # BFO: inheres_in (BFO_0000197) is for SDCs only; GDCs use
    # generically_depends_on or is_concretized_as.
    # characteristic_of (RO_0000052) is broader and accepts GDC domain,
    # but we still warn when a GDC uses it.
    if property_name in ("inheres_in", "characteristic_of"):
        subj_meta = get_metatypes(subject_cat)
        if subj_meta and BFOMeta.GDC in subj_meta and BFOMeta.SDC not in subj_meta:
            warnings.append(
                f"BFO strictness: '{subject_cat}' is a GDC, not an SDC. "
                f"BFO 2020 reserves 'inheres_in' for SDCs. Consider "
                f"'generically_depends_on' or 'is_concretized_as' for GDCs."
            )

    return ValidationResult(
        is_valid=True,
        domain_range_ok=True,
        reason="Valid.",
        warnings=warnings,
    )


# ────────────────────────────────────────────────────────────────────────
# §6  Batch Validation
# ────────────────────────────────────────────────────────────────────────

@dataclass
class BatchValidationReport:
    """Summary of batch validation results."""
    total: int
    accepted: int
    rejected_domain_range: int
    rejected_self_relation: int
    rejected_confidence: int
    rejected_inverse_redundancy: int
    rejected_duplicate: int
    warnings_count: int
    details: list[dict]


def validate_batch(
    relations: list[dict],
    taxonomy_parents: dict[str, str] | None = None,
    confidence_threshold: float = 0.7,
) -> BatchValidationReport:
    """
    Validate a batch of extracted relations.

    Each relation dict must have keys:
        term, category, property, filler, filler_category,
        confidence, evidence

    Optional: nld (the full NLD text for evidence checking)

    Returns:
        BatchValidationReport with per-relation details.
    """
    accepted_relations: set[tuple[str, str, str]] = set()
    details: list[dict] = []

    counters = {
        "domain_range": 0,
        "self_relation": 0,
        "confidence": 0,
        "inverse_redundancy": 0,
        "duplicate": 0,
    }

    for rel in relations:
        result = validate_relation_full(
            subject_term=rel["term"],
            subject_cat=rel["category"],
            property_name=rel["property"],
            object_term=rel["filler"],
            object_cat=rel.get("filler_category", ""),
            confidence=rel.get("confidence", 0.0),
            evidence=rel.get("evidence", ""),
            nld_text=rel.get("nld", ""),
            existing_relations=accepted_relations,
            taxonomy_parents=taxonomy_parents,
            confidence_threshold=confidence_threshold,
        )

        detail = {
            "term": rel["term"],
            "property": rel["property"],
            "filler": rel["filler"],
            "is_valid": result.is_valid,
            "reason": result.reason,
            "warnings": result.warnings,
        }
        details.append(detail)

        if result.is_valid:
            accepted_relations.add(
                (rel["term"], rel["property"], rel["filler"])
            )
        else:
            # Classify rejection reason
            reason = result.reason.lower()
            if not result.domain_range_ok:
                counters["domain_range"] += 1
            elif "self-relation" in reason:
                counters["self_relation"] += 1
            elif "confidence" in reason:
                counters["confidence"] += 1
            elif "inverse" in reason:
                counters["inverse_redundancy"] += 1
            elif "duplicate" in reason:
                counters["duplicate"] += 1

    accepted = sum(1 for d in details if d["is_valid"])
    warnings_count = sum(len(d["warnings"]) for d in details)

    return BatchValidationReport(
        total=len(relations),
        accepted=accepted,
        rejected_domain_range=counters["domain_range"],
        rejected_self_relation=counters["self_relation"],
        rejected_confidence=counters["confidence"],
        rejected_inverse_redundancy=counters["inverse_redundancy"],
        rejected_duplicate=counters["duplicate"],
        warnings_count=warnings_count,
        details=details,
    )


# ────────────────────────────────────────────────────────────────────────
# §7  Disjointness Recommendations
# ────────────────────────────────────────────────────────────────────────

# Safe disjointness axioms: sibling classes under the same parent where
# instances should NEVER overlap.  These follow directly from BFO 2020
# and GeoCore definitions.
#
# Risk/reward analysis:
#   REWARD — Disjointness catches classification errors early.  A reasoner
#            will flag inconsistency if a term is typed under two disjoint
#            classes (e.g., something is both a Process and a Quality).
#   RISK  — If the extraction or categorization makes a mistake and assigns
#            an entity to the wrong class, disjointness axioms turn a
#            silent error into a reasoning failure.  For a thesis pipeline
#            where perfect precision is not guaranteed, overly aggressive
#            disjointness is dangerous.
#
# Recommendation: Add only the TOP-LEVEL BFO disjointness axioms.  These
# are part of the BFO specification itself and are always safe.  Domain-
# level disjointness (e.g., Rock ⊥ Earth Fluid) should be imported from
# GeoCore/GeoReservoir rather than asserted in the generated ontology.

SAFE_DISJOINT_PAIRS: list[tuple[str, str]] = [
    # BFO 2020 top-level disjointness (from the specification)
    ("Continuant", "Occurrent"),
    # Within Continuant
    ("IndependentContinuant", "SpecificallyDependentContinuant"),
    ("IndependentContinuant", "GenericallyDependentContinuant"),
    ("SpecificallyDependentContinuant", "GenericallyDependentContinuant"),
    # Within IndependentContinuant
    ("MaterialEntity", "ImmaterialEntity"),
    # Within Occurrent
    ("Process", "TemporalRegion"),
    ("Process", "SpatialRegion"),
]

# Domain-level disjointness that is PROBABLY safe but carries risk:
RISKY_DISJOINT_PAIRS: list[tuple[str, str]] = [
    # GeoCore siblings under MaterialEntity — probably safe but a
    # mis-categorized term could trigger inconsistency
    ("Rock", "Earth Fluid"),
    ("Sedimentary Rock", "Earth Fluid"),
    # Process vs Quality (already covered by BFO top-level but
    # explicitly useful for geo categories)
    ("Geological Process", "quality"),
    ("Geological Process", "Geological Age"),
    # Object vs Quality
    ("Geological Object", "quality"),
    ("Geological Object", "Geological Structure"),  # Object vs GDC
]


# ────────────────────────────────────────────────────────────────────────
# §8  False Positive Estimation
# ────────────────────────────────────────────────────────────────────────

EXPECTED_FILTER_RATES = """
## Estimated False Positive Filter Rates

Based on typical LLM ontology extraction error patterns:

| Filter                  | Est. catch rate | Notes                                         |
|-------------------------|-----------------|-----------------------------------------------|
| Domain/range violations | 15–25%          | Most common: process↔quality confusion,       |
|                         |                 | has_participant with wrong domain, inheres_in  |
|                         |                 | applied to non-SDC categories.                 |
| Self-relations          |  2–5%           | LLM sometimes extracts "X has_part X" when     |
|                         |                 | the NLD says "X is composed of X-type material"|
| Low confidence (<0.7)   |  5–10%          | Already controlled by the prompt's 0.6 floor;  |
|                         |                 | raising to 0.7 catches borderline extractions  |
| Inverse redundancy      |  5–8%           | LLM often extracts both directions of a pair   |
| Duplicate relations     |  2–3%           | Batch overlap across categories                |
| Evidence hallucination  |  3–5%           | Evidence string not in NLD — fabricated         |
|-------------------------|-----------------|-----------------------------------------------|
| **Combined automated**  | **30–45%**      | Of all raw extractions are caught              |
| **Remaining for review**| **5–15%**       | Semantically wrong but pass all filters —      |
|                         |                 | typically incorrect filler resolution or        |
|                         |                 | over-extraction from vague NLD language         |

Key insight: Domain/range checking is the SINGLE MOST EFFECTIVE automated
filter.  It catches the majority of cross-category errors that LLMs make
because LLMs are weak at tracking BFO's Continuant/Occurrent partition.

What gets through:
  - Correct metatype but wrong specific relation (e.g., has_part vs derives_from
    between two MaterialEntities)
  - Correct relation type but wrong filler (e.g., "generated_by diagenesis"
    when the NLD says "formed by precipitation")
  - Over-extraction from dispositional language that the conservatism rules
    should have blocked
  - Genuine edge cases (is Geological Structure a GDC or SDC?)

These residual errors require human/expert review.
"""
