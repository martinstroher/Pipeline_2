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
# Each PreSaltOntoLearn category maps to a SET of applicable BFO metatypes,
# ordered from most specific to most general.  The validator checks whether
# the metatype set for a category intersects the allowed domain/range set
# of the property.

_CATEGORY_TO_METATYPES: dict[str, frozenset[str]] = {
    # ── BFO direct categories ──
    "process": frozenset({
        BFOMeta.PROCESS, BFOMeta.OCCURRENT,
    }),
    "quality": frozenset({
        BFOMeta.QUALITY, BFOMeta.SDC, BFOMeta.CONTINUANT,
    }),
    "object": frozenset({
        BFOMeta.OBJECT, BFOMeta.MATERIAL_ENTITY,
        BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.CONTINUANT,
    }),
    "site": frozenset({
        BFOMeta.SITE, BFOMeta.IMMATERIAL_ENTITY,
        BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.CONTINUANT,
    }),
    "material entity": frozenset({
        BFOMeta.MATERIAL_ENTITY, BFOMeta.INDEPENDENT_CONTINUANT,
        BFOMeta.CONTINUANT,
    }),
    "independent continuant": frozenset({
        BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.CONTINUANT,
    }),
    "specifically dependent continuant": frozenset({
        BFOMeta.SDC, BFOMeta.CONTINUANT,
    }),
    "generically dependent continuant": frozenset({
        BFOMeta.GDC, BFOMeta.CONTINUANT,
    }),
    "relational quality": frozenset({
        BFOMeta.RELATIONAL_QUALITY, BFOMeta.QUALITY,
        BFOMeta.SDC, BFOMeta.CONTINUANT,
    }),
    "fiat object part": frozenset({
        BFOMeta.MATERIAL_ENTITY, BFOMeta.INDEPENDENT_CONTINUANT,
        BFOMeta.CONTINUANT,
    }),
    "object aggregate": frozenset({
        BFOMeta.MATERIAL_ENTITY, BFOMeta.INDEPENDENT_CONTINUANT,
        BFOMeta.CONTINUANT,
    }),
    "immaterial entity": frozenset({
        BFOMeta.IMMATERIAL_ENTITY, BFOMeta.INDEPENDENT_CONTINUANT,
        BFOMeta.CONTINUANT,
    }),
    "continuant fiat boundary": frozenset({
        BFOMeta.IMMATERIAL_ENTITY, BFOMeta.INDEPENDENT_CONTINUANT,
        BFOMeta.CONTINUANT,
    }),
    "fiat surface": frozenset({
        BFOMeta.IMMATERIAL_ENTITY, BFOMeta.INDEPENDENT_CONTINUANT,
        BFOMeta.CONTINUANT,
    }),
    "spatial region": frozenset({
        BFOMeta.SPATIAL_REGION, BFOMeta.CONTINUANT,
    }),
    "process boundary": frozenset({
        BFOMeta.PROCESS, BFOMeta.OCCURRENT,
    }),
    "one-dimensional temporal region": frozenset({
        BFOMeta.TEMPORAL_REGION, BFOMeta.OCCURRENT,
    }),
    "temporal region": frozenset({
        BFOMeta.TEMPORAL_REGION, BFOMeta.OCCURRENT,
    }),
    "spatiotemporal region": frozenset({
        BFOMeta.OCCURRENT,
    }),

    # ── GeoCore categories  ──
    "Geological Object": frozenset({
        BFOMeta.OBJECT, BFOMeta.MATERIAL_ENTITY,
        BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.CONTINUANT,
    }),
    "Geological Process": frozenset({
        BFOMeta.PROCESS, BFOMeta.OCCURRENT,
    }),
    "Earth Material": frozenset({
        BFOMeta.MATERIAL_ENTITY, BFOMeta.INDEPENDENT_CONTINUANT,
        BFOMeta.CONTINUANT,
    }),
    "Rock": frozenset({
        BFOMeta.MATERIAL_ENTITY, BFOMeta.INDEPENDENT_CONTINUANT,
        BFOMeta.CONTINUANT,
    }),
    "Earth Fluid": frozenset({
        BFOMeta.MATERIAL_ENTITY, BFOMeta.INDEPENDENT_CONTINUANT,
        BFOMeta.CONTINUANT,
    }),
    "Geological Structure": frozenset({
        # GeoCore: "Generically Dependent Continuant which describes
        # the internal arrangement of a Geological Object"
        BFOMeta.GDC, BFOMeta.CONTINUANT,
    }),
    "Geological Boundary": frozenset({
        # GeoCore: "Continuant Fiat-Boundary"
        BFOMeta.IMMATERIAL_ENTITY, BFOMeta.INDEPENDENT_CONTINUANT,
        BFOMeta.CONTINUANT,
    }),
    "Geological Contact": frozenset({
        # GeoCore: "Relational Quality"
        BFOMeta.RELATIONAL_QUALITY, BFOMeta.QUALITY,
        BFOMeta.SDC, BFOMeta.CONTINUANT,
    }),
    "Geological Time Interval": frozenset({
        BFOMeta.TEMPORAL_REGION, BFOMeta.OCCURRENT,
    }),
    "Geological Age": frozenset({
        # GeoCore: "Quality of Geological Object related to a Geological
        # Time Interval"
        BFOMeta.QUALITY, BFOMeta.SDC, BFOMeta.CONTINUANT,
    }),

    # ── GeoReservoir categories ──
    "Sedimentary Rock": frozenset({
        BFOMeta.MATERIAL_ENTITY, BFOMeta.INDEPENDENT_CONTINUANT,
        BFOMeta.CONTINUANT,
    }),
    "Sediment": frozenset({
        BFOMeta.MATERIAL_ENTITY, BFOMeta.INDEPENDENT_CONTINUANT,
        BFOMeta.CONTINUANT,
    }),
    "Depositional Unit": frozenset({
        BFOMeta.OBJECT, BFOMeta.MATERIAL_ENTITY,
        BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.CONTINUANT,
    }),
    "Channel Unit": frozenset({
        BFOMeta.OBJECT, BFOMeta.MATERIAL_ENTITY,
        BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.CONTINUANT,
    }),
    "Lobe Unit": frozenset({
        BFOMeta.OBJECT, BFOMeta.MATERIAL_ENTITY,
        BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.CONTINUANT,
    }),
    "Levee Unit": frozenset({
        BFOMeta.OBJECT, BFOMeta.MATERIAL_ENTITY,
        BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.CONTINUANT,
    }),
    "Mound Unit": frozenset({
        BFOMeta.OBJECT, BFOMeta.MATERIAL_ENTITY,
        BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.CONTINUANT,
    }),
    "Depositional System": frozenset({
        # GeoReservoir: "Object Aggregate"
        BFOMeta.MATERIAL_ENTITY, BFOMeta.INDEPENDENT_CONTINUANT,
        BFOMeta.CONTINUANT,
    }),
    "Sedimentary Facies": frozenset({
        # GeoReservoir: visual pattern of properties → GDC-like
        BFOMeta.GDC, BFOMeta.CONTINUANT,
    }),
    "Facies Association": frozenset({
        BFOMeta.GDC, BFOMeta.CONTINUANT,
    }),
    "Facies": frozenset({
        BFOMeta.GDC, BFOMeta.CONTINUANT,
    }),
    "Sedimentary Structure": frozenset({
        BFOMeta.GDC, BFOMeta.CONTINUANT,
    }),
    "Fossil": frozenset({
        BFOMeta.OBJECT, BFOMeta.MATERIAL_ENTITY,
        BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.CONTINUANT,
    }),
    "Geometry": frozenset({
        BFOMeta.QUALITY, BFOMeta.SDC, BFOMeta.CONTINUANT,
    }),
    "Dimension": frozenset({
        BFOMeta.QUALITY, BFOMeta.SDC, BFOMeta.CONTINUANT,
    }),
    "Sinuosity": frozenset({
        BFOMeta.QUALITY, BFOMeta.SDC, BFOMeta.CONTINUANT,
    }),
    "Lithology": frozenset({
        # Lithology = quality describing rock composition
        BFOMeta.QUALITY, BFOMeta.SDC, BFOMeta.CONTINUANT,
    }),
    "Sedimentary Environment": frozenset({
        # Depositional environment = site/setting where processes occur
        BFOMeta.SITE, BFOMeta.IMMATERIAL_ENTITY,
        BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.CONTINUANT,
    }),
    "Formation": frozenset({
        # Formal lithostratigraphic unit — a geological object
        BFOMeta.OBJECT, BFOMeta.MATERIAL_ENTITY,
        BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.CONTINUANT,
    }),
    "Stratigraphic Unit": frozenset({
        BFOMeta.OBJECT, BFOMeta.MATERIAL_ENTITY,
        BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.CONTINUANT,
    }),
    "Sedimentary Geological Object": frozenset({
        BFOMeta.OBJECT, BFOMeta.MATERIAL_ENTITY,
        BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.CONTINUANT,
    }),
    "Channel Surface": frozenset({
        BFOMeta.IMMATERIAL_ENTITY, BFOMeta.INDEPENDENT_CONTINUANT,
        BFOMeta.CONTINUANT,
    }),
}

# Case-insensitive lookup
_CATEGORY_LOOKUP: dict[str, frozenset[str]] = {
    k.lower(): v for k, v in _CATEGORY_TO_METATYPES.items()
}


def get_metatypes(category: str) -> frozenset[str] | None:
    """Return BFO metatype set for a category, or None if unknown."""
    return _CATEGORY_LOOKUP.get(category.strip().lower())


# ────────────────────────────────────────────────────────────────────────
# §3  Property Domain/Range Constraints (BFO 2020 + RO)
# ────────────────────────────────────────────────────────────────────────
# Each property maps to (domain_metatypes, range_metatypes) where the
# extracted subject's metatypes must intersect domain_metatypes and the
# extracted object's metatypes must intersect range_metatypes.
#
# BFO 2020 reference:
#   has_part / part_of        — BFO_0000051/50  Continuant→Continuant OR Occurrent→Occurrent
#                               (but not cross-category)
#   has_participant           — BFO_0000057     Process → Continuant
#   participates_in           — BFO_0000056     Continuant → Process
#   inheres_in                — RO_0000052      SDC → IndependentContinuant
#   has_quality               — RO_0000086      IndependentContinuant → Quality (subtype of SDC)
#   occurs_in                 — BFO_0000066     Process → IndependentContinuant (specifically
#                               material entity or site per BFO 2020 §8.11)
#   derives_from              — RO_0001000      MaterialEntity → MaterialEntity
#   generated_by              — (GeoCore custom) Continuant → Process
#   preceded_by               — BFO_0000062     Occurrent → Occurrent
#   precedes                  — BFO_0000063     Occurrent → Occurrent

@dataclass(frozen=True)
class PropertyConstraint:
    """Domain/range constraint for a single object property."""
    domain: frozenset[str]   # allowed BFO metatypes for subject
    range: frozenset[str]    # allowed BFO metatypes for object
    iri: str                 # canonical IRI fragment
    inverse: str | None      # inverse property name (for redundancy check)
    notes: str = ""


# Helper sets
_CONTINUANT = frozenset({
    BFOMeta.CONTINUANT, BFOMeta.INDEPENDENT_CONTINUANT,
    BFOMeta.MATERIAL_ENTITY, BFOMeta.OBJECT, BFOMeta.SITE,
    BFOMeta.IMMATERIAL_ENTITY, BFOMeta.SDC, BFOMeta.QUALITY,
    BFOMeta.RELATIONAL_QUALITY, BFOMeta.DISPOSITION,
    BFOMeta.GDC, BFOMeta.SPATIAL_REGION,
})
_OCCURRENT = frozenset({
    BFOMeta.OCCURRENT, BFOMeta.PROCESS, BFOMeta.PROCESS_BOUNDARY,
    BFOMeta.HISTORY, BFOMeta.TEMPORAL_REGION, BFOMeta.TEMPORAL_INSTANT,
    BFOMeta.SPATIOTEMPORAL_REGION,
})
_ALL = _CONTINUANT | _OCCURRENT
_MATERIAL = frozenset({
    BFOMeta.MATERIAL_ENTITY, BFOMeta.OBJECT,
})
_INDEPENDENT_CONTINUANT = frozenset({
    BFOMeta.INDEPENDENT_CONTINUANT, BFOMeta.MATERIAL_ENTITY,
    BFOMeta.OBJECT, BFOMeta.SITE, BFOMeta.IMMATERIAL_ENTITY,
})
_IMMATERIAL = frozenset({
    BFOMeta.IMMATERIAL_ENTITY, BFOMeta.SITE,
})
_PROCESS = frozenset({
    BFOMeta.PROCESS, BFOMeta.PROCESS_BOUNDARY,
})
_PROCESS_STRICT = frozenset({
    BFOMeta.PROCESS,
})
_SDC = frozenset({
    BFOMeta.SDC, BFOMeta.QUALITY, BFOMeta.RELATIONAL_QUALITY,
    BFOMeta.DISPOSITION,
})
_GDC = frozenset({
    BFOMeta.GDC,
})
_QUALITY = frozenset({
    BFOMeta.QUALITY, BFOMeta.RELATIONAL_QUALITY,
})
_DISPOSITION = frozenset({
    BFOMeta.DISPOSITION, BFOMeta.SDC,
})
_TEMPORAL = frozenset({
    BFOMeta.TEMPORAL_REGION, BFOMeta.TEMPORAL_INSTANT,
})
_TEMPORAL_INSTANT = frozenset({
    BFOMeta.TEMPORAL_INSTANT,
})
_SPATIOTEMPORAL = frozenset({
    BFOMeta.SPATIOTEMPORAL_REGION,
})
_HISTORY = frozenset({
    BFOMeta.HISTORY,
})
_SPATIAL = frozenset({
    BFOMeta.SPATIAL_REGION,
})
# Site or material entity (for occurs_in range)
_SITE_OR_MATERIAL = frozenset({
    BFOMeta.SITE, BFOMeta.IMMATERIAL_ENTITY,
    BFOMeta.MATERIAL_ENTITY, BFOMeta.OBJECT,
    BFOMeta.INDEPENDENT_CONTINUANT,
})


PROPERTY_CONSTRAINTS: dict[str, PropertyConstraint] = {

    # ════════════════════════════════════════════════════════════════════
    # RO-Core (Relation Ontology)  —  from ro-core.owl
    # ════════════════════════════════════════════════════════════════════

    # ── Mereological ──
    "part_of": PropertyConstraint(
        domain=_CONTINUANT | _OCCURRENT,
        range=_CONTINUANT | _OCCURRENT,
        iri="http://purl.obolibrary.org/obo/BFO_0000050",
        inverse="has_part",
        notes="Cross-category (Continuant↔Occurrent) forbidden; checked separately. Transitive.",
    ),
    "has_part": PropertyConstraint(
        domain=_CONTINUANT | _OCCURRENT,
        range=_CONTINUANT | _OCCURRENT,
        iri="http://purl.obolibrary.org/obo/BFO_0000051",
        inverse="part_of",
        notes="Cross-category (Continuant↔Occurrent) forbidden; checked separately. Transitive.",
    ),

    # ── Realization (RO alias: "realized_in") ──
    "realized_in": PropertyConstraint(
        domain=_SDC,
        range=_PROCESS,
        iri="http://purl.obolibrary.org/obo/BFO_0000054",
        inverse="realizes",
        notes="RO alias for has_realization. Same IRI.",
    ),

    # ── Contains process (RO-only, inverse of occurs_in) ──
    "contains_process": PropertyConstraint(
        domain=_INDEPENDENT_CONTINUANT,
        range=_PROCESS,
        iri="http://purl.obolibrary.org/obo/BFO_0000067",
        inverse="occurs_in",
        notes="RO-only IRI (BFO 2020 uses 'environs' BFO_0000183 instead).",
    ),

    # ── Characteristic / Inherence (RO namespace — no BFO 2020 equivalent) ──
    "characteristic_of": PropertyConstraint(
        domain=_SDC | _GDC,
        range=_INDEPENDENT_CONTINUANT,
        iri="http://purl.obolibrary.org/obo/RO_0000052",
        inverse="has_characteristic",
        notes="Broad: SDC or GDC → IC. Functional. Parent of quality_of, role_of, etc.",
    ),
    "has_characteristic": PropertyConstraint(
        domain=_ALL,
        range=_SDC,
        iri="http://purl.obolibrary.org/obo/RO_0000053",
        inverse="characteristic_of",
        notes="Inverse of characteristic_of. Range is SDC (BFO_0000020).",
    ),

    # ── Concretization (RO alias: "is_concretized_as") ──
    "is_concretized_as": PropertyConstraint(
        domain=_GDC,
        range=_SDC | _PROCESS,
        iri="http://purl.obolibrary.org/obo/RO_0000058",
        inverse="concretizes",
        notes="RO alias for is_concretized_by. Same IRI.",
    ),

    # ── Function / Quality / Role / Disposition (specific characteristic relations) ──
    "function_of": PropertyConstraint(
        domain=_SDC,
        range=_INDEPENDENT_CONTINUANT,
        iri="http://purl.obolibrary.org/obo/RO_0000079",
        inverse="has_function",
        notes="SubPropertyOf characteristic_of. Domain is BFO:Function (⊂ SDC).",
    ),
    "quality_of": PropertyConstraint(
        domain=_QUALITY,
        range=_INDEPENDENT_CONTINUANT,
        iri="http://purl.obolibrary.org/obo/RO_0000080",
        inverse="has_quality",
        notes="SubPropertyOf characteristic_of.",
    ),
    "role_of": PropertyConstraint(
        domain=_SDC,
        range=_INDEPENDENT_CONTINUANT,
        iri="http://purl.obolibrary.org/obo/RO_0000081",
        inverse="has_role",
        notes="SubPropertyOf characteristic_of. Domain is BFO:Role (⊂ SDC).",
    ),
    "has_function": PropertyConstraint(
        domain=_INDEPENDENT_CONTINUANT,
        range=_SDC,
        iri="http://purl.obolibrary.org/obo/RO_0000085",
        inverse="function_of",
        notes="SubPropertyOf has_characteristic. Range is BFO:Function (⊂ SDC).",
    ),
    "has_quality": PropertyConstraint(
        domain=_INDEPENDENT_CONTINUANT,
        range=_QUALITY,
        iri="http://purl.obolibrary.org/obo/RO_0000086",
        inverse="quality_of",
    ),
    "has_role": PropertyConstraint(
        domain=_INDEPENDENT_CONTINUANT,
        range=_SDC,
        iri="http://purl.obolibrary.org/obo/RO_0000087",
        inverse="role_of",
        notes="SubPropertyOf has_characteristic. Range is BFO:Role (⊂ SDC).",
    ),
    "has_disposition": PropertyConstraint(
        domain=_INDEPENDENT_CONTINUANT,
        range=_SDC,
        iri="http://purl.obolibrary.org/obo/RO_0000091",
        inverse="disposition_of",
        notes="SubPropertyOf has_characteristic. Range is BFO:Disposition (⊂ SDC).",
    ),
    "disposition_of": PropertyConstraint(
        domain=_SDC,
        range=_INDEPENDENT_CONTINUANT,
        iri="http://purl.obolibrary.org/obo/RO_0000092",
        inverse="has_disposition",
        notes="SubPropertyOf characteristic_of.",
    ),

    # ── Derivation ──
    "derives_from": PropertyConstraint(
        domain=_MATERIAL,
        range=_MATERIAL,
        iri="http://purl.obolibrary.org/obo/RO_0001000",
        inverse="derives_into",
        notes="Material entity → material entity. Tracks material transformation lineage.",
    ),
    "derives_into": PropertyConstraint(
        domain=_MATERIAL,
        range=_MATERIAL,
        iri="http://purl.obolibrary.org/obo/RO_0001001",
        inverse="derives_from",
        notes="Material entity → material entity. Inverse of derives_from.",
    ),

    # ── Boundary (RO-only) ──
    "2d_boundary_of": PropertyConstraint(
        domain=_IMMATERIAL,
        range=_MATERIAL,
        iri="http://purl.obolibrary.org/obo/RO_0002000",
        inverse="has_2d_boundary",
        notes="2D immaterial entity (boundary) → material entity.",
    ),
    "has_2d_boundary": PropertyConstraint(
        domain=_MATERIAL,
        range=_IMMATERIAL,
        iri="http://purl.obolibrary.org/obo/RO_0002002",
        inverse="2d_boundary_of",
        notes="Material entity → 2D immaterial entity (boundary).",
    ),

    # ── Member (RO collection-level) ──
    "member_of": PropertyConstraint(
        domain=_CONTINUANT | _OCCURRENT,
        range=_CONTINUANT | _OCCURRENT,
        iri="http://purl.obolibrary.org/obo/RO_0002350",
        inverse="has_member",
        notes="SubPropertyOf part_of. Mereological: item → collection.",
    ),
    "has_member": PropertyConstraint(
        domain=_CONTINUANT | _OCCURRENT,
        range=_CONTINUANT | _OCCURRENT,
        iri="http://purl.obolibrary.org/obo/RO_0002351",
        inverse="member_of",
        notes="SubPropertyOf has_part. Mereological: collection → item.",
    ),

    # ════════════════════════════════════════════════════════════════════
    # BFO 2020 namespace  —  from bfo-core.owl + geocore/geores declarations
    # ════════════════════════════════════════════════════════════════════

    # ── Realization (BFO 2020 labels: "has realization" / "realizes") ──
    "has_realization": PropertyConstraint(
        domain=_SDC,
        range=_PROCESS,
        iri="http://purl.obolibrary.org/obo/BFO_0000054",
        inverse="realizes",
        notes="Domain is BFO:RealizableEntity (⊂ SDC). Range is Process.",
    ),
    "realizes": PropertyConstraint(
        domain=_PROCESS,
        range=_SDC,
        iri="http://purl.obolibrary.org/obo/BFO_0000055",
        inverse="has_realization",
    ),

    # ── Participation (BFO 2020 IRI) ──
    "participates_in": PropertyConstraint(
        domain=_CONTINUANT,
        range=_PROCESS,
        iri="http://purl.obolibrary.org/obo/BFO_0000056",
        inverse="has_participant",
    ),
    "has_participant": PropertyConstraint(
        domain=_PROCESS,
        range=_CONTINUANT,
        iri="http://purl.obolibrary.org/obo/BFO_0000057",
        inverse="participates_in",
    ),

    # ── Concretization (BFO 2020) ──
    "is_concretized_by": PropertyConstraint(
        domain=_GDC,
        range=_SDC | _PROCESS,
        iri="http://purl.obolibrary.org/obo/BFO_0000058",
        inverse="concretizes",
    ),
    "concretizes": PropertyConstraint(
        domain=_SDC | _PROCESS,
        range=_GDC,
        iri="http://purl.obolibrary.org/obo/BFO_0000059",
        inverse="is_concretized_by",
    ),

    # ── Temporal ordering (Transitive) ──
    "preceded_by": PropertyConstraint(
        domain=_OCCURRENT,
        range=_OCCURRENT,
        iri="http://purl.obolibrary.org/obo/BFO_0000062",
        inverse="precedes",
        notes="Transitive.",
    ),
    "precedes": PropertyConstraint(
        domain=_OCCURRENT,
        range=_OCCURRENT,
        iri="http://purl.obolibrary.org/obo/BFO_0000063",
        inverse="preceded_by",
        notes="Transitive.",
    ),

    # ── Spatial / Occurrence ──
    "occurs_in": PropertyConstraint(
        domain=_PROCESS,
        range=_SITE_OR_MATERIAL,
        iri="http://purl.obolibrary.org/obo/BFO_0000066",
        inverse="environs",
    ),
    "environs": PropertyConstraint(
        domain=_SITE_OR_MATERIAL,
        range=_PROCESS,
        iri="http://purl.obolibrary.org/obo/BFO_0000183",
        inverse="occurs_in",
    ),

    # ── Generic dependence ──
    "generically_depends_on": PropertyConstraint(
        domain=_GDC,
        range=_INDEPENDENT_CONTINUANT,
        iri="http://purl.obolibrary.org/obo/BFO_0000084",
        inverse="is_carrier_of",
    ),
    "is_carrier_of": PropertyConstraint(
        domain=_INDEPENDENT_CONTINUANT,
        range=_GDC,
        iri="http://purl.obolibrary.org/obo/BFO_0000101",
        inverse="generically_depends_on",
    ),

    # ── Exists at ──
    "exists_at": PropertyConstraint(
        domain=_ALL,
        range=_TEMPORAL,
        iri="http://purl.obolibrary.org/obo/BFO_0000108",
        inverse=None,
        notes="Any particular → TemporalRegion.",
    ),

    # ── Member part (BFO 2020 mereology) ──
    "has_member_part": PropertyConstraint(
        domain=_MATERIAL,
        range=_MATERIAL,
        iri="http://purl.obolibrary.org/obo/BFO_0000115",
        inverse="member_part_of",
        notes="SubPropertyOf has_continuant_part. Domain/Range: MaterialEntity.",
    ),
    "member_part_of": PropertyConstraint(
        domain=_MATERIAL,
        range=_MATERIAL,
        iri="http://purl.obolibrary.org/obo/BFO_0000129",
        inverse="has_member_part",
        notes="SubPropertyOf continuant_part_of. Domain/Range: MaterialEntity.",
    ),

    # ── Occurrent parthood (Transitive) ──
    "has_occurrent_part": PropertyConstraint(
        domain=_OCCURRENT,
        range=_OCCURRENT,
        iri="http://purl.obolibrary.org/obo/BFO_0000117",
        inverse="occurrent_part_of",
        notes="Transitive.",
    ),
    "occurrent_part_of": PropertyConstraint(
        domain=_OCCURRENT,
        range=_OCCURRENT,
        iri="http://purl.obolibrary.org/obo/BFO_0000132",
        inverse="has_occurrent_part",
        notes="Transitive.",
    ),

    # ── Temporal parthood (Transitive, subPropOf occurrent part) ──
    "has_temporal_part": PropertyConstraint(
        domain=_OCCURRENT,
        range=_OCCURRENT,
        iri="http://purl.obolibrary.org/obo/BFO_0000121",
        inverse="temporal_part_of",
        notes="Transitive. SubPropertyOf has_occurrent_part.",
    ),
    "temporal_part_of": PropertyConstraint(
        domain=_OCCURRENT,
        range=_OCCURRENT,
        iri="http://purl.obolibrary.org/obo/BFO_0000139",
        inverse="has_temporal_part",
        notes="Transitive. SubPropertyOf occurrent_part_of.",
    ),

    # ── Location (BFO 2020 IRI) ──
    "location_of": PropertyConstraint(
        domain=_INDEPENDENT_CONTINUANT,
        range=_INDEPENDENT_CONTINUANT,
        iri="http://purl.obolibrary.org/obo/BFO_0000124",
        inverse="located_in",
        notes="BFO 2020 IRI. Also exists as RO_0001015.",
    ),
    "located_in": PropertyConstraint(
        domain=_INDEPENDENT_CONTINUANT,
        range=_INDEPENDENT_CONTINUANT,
        iri="http://purl.obolibrary.org/obo/BFO_0000171",
        inverse="location_of",
        notes="BFO 2020 IRI. Also exists as RO_0001025.",
    ),

    # ── Material basis ──
    "material_basis_of": PropertyConstraint(
        domain=_MATERIAL,
        range=_DISPOSITION,
        iri="http://purl.obolibrary.org/obo/BFO_0000127",
        inverse="has_material_basis",
    ),
    "has_material_basis": PropertyConstraint(
        domain=_DISPOSITION,
        range=_MATERIAL,
        iri="http://purl.obolibrary.org/obo/BFO_0000218",
        inverse="material_basis_of",
    ),

    # ── Continuant parthood ──
    "continuant_part_of": PropertyConstraint(
        domain=_CONTINUANT,
        range=_CONTINUANT,
        iri="http://purl.obolibrary.org/obo/BFO_0000176",
        inverse="has_continuant_part",
    ),
    "has_continuant_part": PropertyConstraint(
        domain=_CONTINUANT,
        range=_CONTINUANT,
        iri="http://purl.obolibrary.org/obo/BFO_0000178",
        inverse="continuant_part_of",
    ),

    # ── History ──
    "history_of": PropertyConstraint(
        domain=_HISTORY,
        range=_MATERIAL,
        iri="http://purl.obolibrary.org/obo/BFO_0000184",
        inverse="has_history",
        notes="Functional.",
    ),
    "has_history": PropertyConstraint(
        domain=_MATERIAL,
        range=_HISTORY,
        iri="http://purl.obolibrary.org/obo/BFO_0000185",
        inverse="history_of",
    ),

    # ── Specific dependence ──
    "specifically_depended_on_by": PropertyConstraint(
        domain=_INDEPENDENT_CONTINUANT | _SDC,
        range=_SDC,
        iri="http://purl.obolibrary.org/obo/BFO_0000194",
        inverse="specifically_depends_on",
    ),
    "specifically_depends_on": PropertyConstraint(
        domain=_SDC,
        range=_INDEPENDENT_CONTINUANT | _SDC,
        iri="http://purl.obolibrary.org/obo/BFO_0000195",
        inverse="specifically_depended_on_by",
    ),

    # ── Bearer / Inherence ──
    "bearer_of": PropertyConstraint(
        domain=_INDEPENDENT_CONTINUANT,
        range=_SDC,
        iri="http://purl.obolibrary.org/obo/BFO_0000196",
        inverse="inheres_in",
        notes="SubPropertyOf specifically_depended_on_by.",
    ),
    "inheres_in": PropertyConstraint(
        domain=_SDC,
        range=_INDEPENDENT_CONTINUANT,
        iri="http://purl.obolibrary.org/obo/BFO_0000197",
        inverse="bearer_of",
        notes="SubPropertyOf specifically_depends_on.",
    ),

    # ── Temporal occupation (Functional) ──
    "occupies_temporal_region": PropertyConstraint(
        domain=_PROCESS | _OCCURRENT,
        range=_TEMPORAL,
        iri="http://purl.obolibrary.org/obo/BFO_0000199",
        inverse=None,
        notes="Functional.",
    ),

    # ── Spatiotemporal occupation (Functional) ──
    "occupies_spatiotemporal_region": PropertyConstraint(
        domain=_PROCESS | _OCCURRENT,
        range=_SPATIOTEMPORAL,
        iri="http://purl.obolibrary.org/obo/BFO_0000200",
        inverse=None,
        notes="Functional.",
    ),

    # ── Spatial occupation ──
    "occupies_spatial_region": PropertyConstraint(
        domain=_INDEPENDENT_CONTINUANT,
        range=_SPATIAL,
        iri="http://purl.obolibrary.org/obo/BFO_0000210",
        inverse=None,
    ),

    # ── Projection ──
    "temporally_projects_onto": PropertyConstraint(
        domain=_SPATIOTEMPORAL,
        range=_TEMPORAL,
        iri="http://purl.obolibrary.org/obo/BFO_0000153",
        inverse=None,
        notes="Functional.",
    ),
    "spatially_projects_onto": PropertyConstraint(
        domain=_SPATIOTEMPORAL,
        range=_SPATIAL,
        iri="http://purl.obolibrary.org/obo/BFO_0000216",
        inverse=None,
    ),

    # ── Temporal instants ──
    "first_instant_of": PropertyConstraint(
        domain=_TEMPORAL_INSTANT,
        range=_TEMPORAL,
        iri="http://purl.obolibrary.org/obo/BFO_0000221",
        inverse="has_first_instant",
    ),
    "has_first_instant": PropertyConstraint(
        domain=_TEMPORAL,
        range=_TEMPORAL_INSTANT,
        iri="http://purl.obolibrary.org/obo/BFO_0000222",
        inverse="first_instant_of",
    ),
    "last_instant_of": PropertyConstraint(
        domain=_TEMPORAL_INSTANT,
        range=_TEMPORAL,
        iri="http://purl.obolibrary.org/obo/BFO_0000223",
        inverse="has_last_instant",
    ),
    "has_last_instant": PropertyConstraint(
        domain=_TEMPORAL,
        range=_TEMPORAL_INSTANT,
        iri="http://purl.obolibrary.org/obo/BFO_0000224",
        inverse="last_instant_of",
    ),

    # ════════════════════════════════════════════════════════════════════
    # GeoCore  —  from geocore-full.owl
    # ════════════════════════════════════════════════════════════════════

    "generated_by": PropertyConstraint(
        domain=_INDEPENDENT_CONTINUANT,
        range=_PROCESS,
        iri="https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000013",
        inverse=None,
        notes="SubPropertyOf participates_in. Domain is Earth Material ∪ Geological Object.",
    ),
    "generated_in": PropertyConstraint(
        domain=_INDEPENDENT_CONTINUANT,
        range=_TEMPORAL,
        iri="https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000014",
        inverse=None,
        notes="The Geological Time Interval in which the object was generated.",
    ),
    "has_age": PropertyConstraint(
        domain=_INDEPENDENT_CONTINUANT,
        range=_QUALITY,
        iri="https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000015",
        inverse="age_of",
        notes="SubPropertyOf bearer_of. Geological Object → Geological Age (Quality).",
    ),
    "age_of": PropertyConstraint(
        domain=_QUALITY,
        range=_INDEPENDENT_CONTINUANT,
        iri="https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000016",
        inverse="has_age",
        notes="SubPropertyOf inheres_in. Inverse of has_age.",
    ),
    "constituted_by": PropertyConstraint(
        domain=_MATERIAL,
        range=_MATERIAL,
        iri="https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000017",
        inverse=None,
        notes="The relation between some material entity and the material that it is made of.",
    ),

    # ════════════════════════════════════════════════════════════════════
    # GeoReservoir  —  from geores-full.owl
    # ════════════════════════════════════════════════════════════════════

    "has_contact_position": PropertyConstraint(
        domain=_QUALITY,
        range=_SDC,
        iri="https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000044",
        inverse=None,
        notes="Domain is Geological Contact (RelationalQuality). Range is Contact Position.",
    ),
    "has_contact_type": PropertyConstraint(
        domain=_QUALITY,
        range=_SDC,
        iri="https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000045",
        inverse=None,
        notes="Domain is Geological Contact (RelationalQuality). Range is Contact Type.",
    ),
    "has_geometry_type": PropertyConstraint(
        domain=_QUALITY,
        range=_QUALITY,
        iri="https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000046",
        inverse=None,
        notes="Domain is Geometry (Quality). Range is Geometry Type (Quality).",
    ),
    "has_base_geometry": PropertyConstraint(
        domain=_QUALITY,
        range=_QUALITY,
        iri="https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000047",
        inverse=None,
        notes="SubPropertyOf has_geometry_type.",
    ),
    "has_top_geometry": PropertyConstraint(
        domain=_QUALITY,
        range=_QUALITY,
        iri="https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000048",
        inverse=None,
        notes="SubPropertyOf has_geometry_type.",
    ),

    # ════════════════════════════════════════════════════════════════════
    # BFO properties not in the 3 OWL files but valid BFO 2020 relations
    # ════════════════════════════════════════════════════════════════════

    # ── Temporal ordering ──
    "preceded_by": PropertyConstraint(
        domain=_OCCURRENT,
        range=_OCCURRENT,
        iri="http://purl.obolibrary.org/obo/BFO_0000062",
        inverse="precedes",
    ),
    "precedes": PropertyConstraint(
        domain=_OCCURRENT,
        range=_OCCURRENT,
        iri="http://purl.obolibrary.org/obo/BFO_0000063",
        inverse="preceded_by",
    ),
}


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
