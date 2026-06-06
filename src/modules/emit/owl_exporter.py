"""
OWL Exporter — Step 7 of the PreSaltOntoLearn pipeline.

Converts taxonomy CSV to a valid OWL ontology in Turtle format using rdflib.

Features:
  - Maps categories to published BFO/GeoCore/GeoReservoir IRIs
  - owl:Class + rdfs:subClassOf for classes
  - owl:NamedIndividual + rdf:type for individuals
  - rdfs:comment for NLDs, rdfs:label for readable names
  - Loadable in Protege
"""

import os
import re
from collections import defaultdict

import pandas as pd

from src.utils.csv_io import read_csv, write_csv
from rdflib import BNode, Graph, Namespace, Literal, URIRef, RDF, RDFS, OWL, XSD

from src.utils import log
from src.utils.ontology_config import get_config

_CFG = get_config()

# Namespaces — sourced from ontology_config.yaml. Edit the YAML to change.
ONTO_NS = Namespace(_CFG.project_namespace())
BFO_NS = Namespace(_CFG.namespace_for("bfo"))
GEOCORE_NS = Namespace(_CFG.namespace_for("geocore"))
GEORESERVOIR_NS = Namespace(_CFG.namespace_for("georeservoir"))

# Import the IRI mapping from taxonomy_builder
from src.modules.construct.taxonomy_builder import UPPER_IRIS

# Build case-insensitive lookup for UPPER_IRIS
_UPPER_IRIS_LOWER = {k.lower(): v for k, v in UPPER_IRIS.items()}

# Set of IRI values for quick "is upper-level?" checks
_UPPER_IRI_VALUES = set(UPPER_IRIS.values())

# BFO disjointness pairs — used for conflict detection and axiom generation.
# Sourced from ontology_config.yaml; expand to full IRI tuples for downstream use.
_BFO_DISJOINT = _CFG.bfo_disjoint_pairs()


def _is_upper_iri(iri: URIRef) -> bool:
    """Return True if the IRI belongs to an upper-level ontology (BFO/GeoCore/GeoReservoir)."""
    return str(iri) in _UPPER_IRI_VALUES


def _local_name(iri_str: str) -> str:
    """Extract the local name (fragment or last path segment) from an IRI string."""
    if "#" in iri_str:
        return iri_str.split("#")[-1]
    if "/" in iri_str:
        return iri_str.split("/")[-1]
    return iri_str


def _term_to_iri(term: str) -> URIRef:
    """Convert a term string to a valid OWL IRI in the ontology namespace.

    If the term matches a known upper-level concept (BFO, GeoCore, GeoReservoir),
    returns the published IRI.  Otherwise mints a local presalt: IRI.
    """
    term_lower = term.strip().lower()

    if term_lower in _UPPER_IRIS_LOWER:
        return URIRef(_UPPER_IRIS_LOWER[term_lower])

    return _mint_presalt_iri(term)


def _mint_presalt_iri(term: str) -> URIRef:
    """Generate a local presalt: CamelCase IRI from a term string."""
    local = re.sub(r"[^a-zA-Z0-9]", "_", term.strip().lower())
    local = re.sub(r"_+", "_", local).strip("_")
    parts = local.split("_")
    camel = "".join(p.capitalize() for p in parts if p)
    return ONTO_NS[camel]


# ── Upper-ontology backbone ────────────────────────────────────────────
# Caches the subClassOf chain from published OWL files so that every
# GeoCore/GeoReservoir class referenced in the graph gets its parent
# links up to BFO.

_UPPER_PARENT_MAP: dict[str, str] | None = None
_UPPER_LABEL_FROM_OWL: dict[str, str] | None = None


def _load_upper_parent_map() -> dict[str, str]:
    """Parse reference OWL files and return {child_IRI: parent_IRI} for named classes.

    Also populates _UPPER_LABEL_FROM_OWL with rdfs:label from the same files.
    """
    global _UPPER_PARENT_MAP, _UPPER_LABEL_FROM_OWL
    if _UPPER_PARENT_MAP is not None:
        return _UPPER_PARENT_MAP

    _UPPER_PARENT_MAP = {}
    _UPPER_LABEL_FROM_OWL = {}
    for fpath in _CFG.owl_class_paths():
        fpath = str(fpath)
        try:
            ref_g = Graph()
            ref_g.parse(fpath)
            for s, _, o in ref_g.triples((None, RDFS.subClassOf, None)):
                s_str, o_str = str(s), str(o)
                if s_str.startswith("http") and o_str.startswith("http"):
                    _UPPER_PARENT_MAP[s_str] = o_str
            for s, _, o in ref_g.triples((None, RDFS.label, None)):
                s_str = str(s)
                if s_str.startswith("http"):
                    _UPPER_LABEL_FROM_OWL[s_str] = str(o)
        except Exception:
            pass

    return _UPPER_PARENT_MAP


def _add_upper_backbone(g: Graph, referenced_upper_iris: set[str]) -> int:
    """Add rdfs:subClassOf chain for every referenced upper-level IRI.

    Walks up the published hierarchy (GeoCore → BFO) and adds class
    declarations, labels, and subClassOf triples for each intermediate.
    Returns the number of backbone triples added.
    """
    parent_map = _load_upper_parent_map()
    # Published OWL labels are canonical and win over UPPER_IRIS friendly names.
    # UPPER_IRIS is only the fallback for IRIs whose OWL has no rdfs:label.
    label_map: dict[str, str] = {v: k for k, v in UPPER_IRIS.items()}
    label_map.update(_UPPER_LABEL_FROM_OWL or {})
    added = 0

    # For each referenced upper IRI, walk up its chain
    to_process = set(referenced_upper_iris)
    processed = set()

    while to_process:
        iri_str = to_process.pop()
        if iri_str in processed:
            continue
        processed.add(iri_str)

        # Add label for this IRI if we know it and it's missing
        if iri_str in label_map:
            iri_uri = URIRef(iri_str)
            if not list(g.objects(iri_uri, RDFS.label)):
                g.add((iri_uri, RDF.type, OWL.Class))
                g.add((iri_uri, RDFS.label, Literal(label_map[iri_str], lang="en")))

        parent_str = parent_map.get(iri_str)
        if not parent_str:
            continue

        child_uri = URIRef(iri_str)
        parent_uri = URIRef(parent_str)

        # Add subClassOf if not already present
        if (child_uri, RDFS.subClassOf, parent_uri) not in g:
            g.add((child_uri, RDF.type, OWL.Class))
            g.add((child_uri, RDFS.subClassOf, parent_uri))
            added += 1

        # Continue walking up
        to_process.add(parent_str)

    return added


def _detect_and_repair_disjointness(g: Graph, df: pd.DataFrame) -> list[dict]:
    """Detect and repair disjointness conflicts in the class hierarchy.

    Walks rdfs:subClassOf chains to find presalt classes that inherit from
    both sides of a BFO disjoint pair (e.g., MaterialEntity AND
    ImmaterialEntity).  Repairs by removing the parent edge that conflicts
    with the term's assigned Category from the taxonomy.

    Must be called AFTER upper-ontology backbone triples have been added.
    """
    upper_parent_map = _load_upper_parent_map()

    def _build_direct_parents():
        dp: dict[str, set[str]] = defaultdict(set)
        for s, _, o in g.triples((None, RDFS.subClassOf, None)):
            s_str, o_str = str(s), str(o)
            if s_str.startswith("http") and o_str.startswith("http"):
                dp[s_str].add(o_str)
        return dp

    def _ancestors(cls_iri: str, dp: dict[str, set[str]]) -> set[str]:
        visited: set[str] = set()
        queue = list(dp.get(cls_iri, set()))
        while queue:
            curr = queue.pop()
            if curr in visited:
                continue
            visited.add(curr)
            queue.extend(dp.get(curr, set()))
        return visited

    # Build term IRI → Category map from taxonomy
    category_of: dict[str, str] = {}
    for _, row in df.iterrows():
        term_iri_str = str(_term_to_iri(str(row["Term"])))
        cat = row.get("Category", "")
        if cat and not pd.isna(cat):
            category_of[term_iri_str] = str(cat)

    repairs: list[dict] = []
    onto_prefix = str(ONTO_NS)

    for iteration in range(5):  # safety: max 5 repair passes
        dp = _build_direct_parents()
        # Check ALL classes, not just presalt: — upper IRIs can also
        # acquire taxonomy-derived edges that cross disjoint boundaries.
        all_classes = [
            str(s) for s in g.subjects(RDF.type, OWL.Class)
        ]

        edges_to_remove: list[tuple[str, str, str, str]] = []

        for iri_a, iri_b in _BFO_DISJOINT:
            for cls_str in all_classes:
                anc = _ancestors(cls_str, dp)
                if iri_a not in anc or iri_b not in anc:
                    continue

                # Conflict: class inherits from both sides of a disjoint pair.
                # Determine which side is "correct" via Category or backbone.
                intended: set[str] = set()
                is_upper = cls_str in _UPPER_IRI_VALUES

                if is_upper:
                    # For upper-ontology IRIs, the backbone (published
                    # hierarchy) is authoritative.  Walk up the backbone
                    # to see which disjoint side it belongs to.
                    current: str | None = cls_str
                    while current:
                        intended.add(current)
                        current = upper_parent_map.get(current)
                else:
                    cat = category_of.get(cls_str, "")
                    if cat:
                        cat_upper = _UPPER_IRIS_LOWER.get(cat.lower(), "")
                        if cat_upper:
                            current = cat_upper
                            while current:
                                intended.add(current)
                                current = upper_parent_map.get(current)

                if iri_a in intended and iri_b not in intended:
                    keep_side, remove_side = iri_a, iri_b
                elif iri_b in intended and iri_a not in intended:
                    keep_side, remove_side = iri_b, iri_a
                else:
                    log.warn(
                        f"  Disjointness conflict unresolved: {_local_name(cls_str)} "
                        f"— category '{cat}' doesn't disambiguate"
                    )
                    continue

                for parent_str in list(dp.get(cls_str, [])):
                    p_anc = _ancestors(parent_str, dp) | {parent_str}
                    if remove_side in p_anc and keep_side not in p_anc:
                        edges_to_remove.append(
                            (cls_str, parent_str, keep_side, remove_side)
                        )

        if not edges_to_remove:
            break

        for cls_str, parent_str, keep_side, remove_side in edges_to_remove:
            g.remove((URIRef(cls_str), RDFS.subClassOf, URIRef(parent_str)))
            cls_label = _local_name(cls_str)
            parent_label = _local_name(parent_str)
            repairs.append({
                "class": cls_label,
                "removed_parent": parent_label,
                "kept_side": _local_name(keep_side),
                "removed_side": _local_name(remove_side),
                "iteration": iteration + 1,
            })
            log.warn(
                f"  Disjointness repair: {cls_label} ⊏ {parent_label} removed "
                f"(conflicted with {_local_name(keep_side)})"
            )

    return repairs


def run_owl_export(
    taxonomy_csv: str,
    nld_csv: str | None = None,
    output_path: str | None = None,
    relations_csv: str | None = None,
):
    """
    Export taxonomy to OWL Turtle format.

    Args:
        taxonomy_csv: Path to taxonomy CSV (output of taxonomy_builder)
        nld_csv: Optional path to NLD CSV (for adding definitions as rdfs:comment)
        output_path: Output .ttl path (default: derived from input)
    """
    if output_path is None:
        base = os.path.splitext(taxonomy_csv)[0]
        output_path = base.replace("6_taxonomy", "7_ontology") + ".ttl"

    df = read_csv(taxonomy_csv)
    log.info(f"OWL Export: {len(df)} taxonomy entries from {taxonomy_csv}")

    # Load NLDs — primary source: NLD column in taxonomy CSV (added by taxonomy_builder)
    nld_map = {}
    if "NLD" in df.columns:
        for _, row in df.iterrows():
            nld_val = row.get("NLD", "")
            if nld_val and not pd.isna(nld_val):
                nld_map[str(row["Term"])] = str(nld_val)

    # Optional separate NLD CSV (fallback / override for backward compatibility)
    if nld_csv and os.path.exists(nld_csv):
        nld_df = read_csv(nld_csv)
        for _, row in nld_df.iterrows():
            nld_map[row["Term"]] = row.get("NLD", "")

    # Build graph
    g = Graph()
    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind(_CFG.project_prefix(), ONTO_NS)
    g.bind(_CFG.prefix_for("bfo"), BFO_NS)
    g.bind(_CFG.prefix_for("geocore"), GEOCORE_NS)
    g.bind(_CFG.prefix_for("georeservoir"), GEORESERVOIR_NS)

    # Ontology declaration — metadata sourced from ontology_config.yaml
    onto_uri = ONTO_NS[_CFG.project_name()]
    g.add((onto_uri, RDF.type, OWL.Ontology))
    project_label = f"{_CFG.project_name()}: {_CFG.project_description()}".strip(": ").strip()
    g.add((onto_uri, RDFS.label, Literal(project_label)))
    long_desc = _CFG.project_long_description()
    if long_desc:
        g.add((onto_uri, RDFS.comment, Literal(long_desc)))
    g.add((onto_uri, OWL.versionInfo, Literal(_CFG.project_version())))

    # Import declarations — only for ontologies with import_iri set in YAML
    for onto_key in _CFG.ontologies.keys():
        import_iri = _CFG.import_iri_for(onto_key)
        if import_iri:
            g.add((onto_uri, OWL.imports, URIRef(import_iri)))

    # Track individual IRIs (rdf:type entities) to handle differently in relations
    _individual_iris: set[str] = set()

    # Build set of all terms for parent-existence validation
    _taxonomy_terms: set[str] = set(df["Term"].dropna().astype(str).str.strip())

    # Process taxonomy entries
    for _, row in df.iterrows():
        term = row["Term"]
        parent = row["Parent_Term"]
        rel_type = row.get("Relationship_Type", "rdfs:subClassOf")
        is_intermediate = row.get("Is_Intermediate", False)

        term_iri = _term_to_iri(str(term))
        parent_iri = None
        has_parent = parent and not (isinstance(parent, float) and pd.isna(parent)) and str(parent).strip()

        if rel_type == "rdf:type":
            _individual_iris.add(str(term_iri))

        if has_parent:
            parent_str = str(parent).strip()
            # Validate parent exists in taxonomy or upper-ontology before minting IRI
            if parent_str not in _taxonomy_terms and parent_str.lower() not in _UPPER_IRIS_LOWER:
                log.warn(f"  Phantom parent '{parent_str}' for '{term}' — treating as root class")
                has_parent = False
                parent_iri = None

        if has_parent:
            parent_iri = _term_to_iri(str(parent))

            # Never create triples between two upper-level entities.
            # We only create triples where at least one side is a presalt: entity.
            if _is_upper_iri(term_iri) and _is_upper_iri(parent_iri):
                log.detail(
                    f"Skipped upper→upper triple: '{term}' → '{parent}'"
                )
                continue

            if rel_type == "rdf:type":
                # Named individual
                g.add((term_iri, RDF.type, OWL.NamedIndividual))
                g.add((term_iri, RDF.type, parent_iri))
            elif term_iri != parent_iri:
                # Class — guard against self-referential subClassOf
                g.add((term_iri, RDF.type, OWL.Class))
                g.add((term_iri, RDFS.subClassOf, parent_iri))
            else:
                # Self-reference detected (case collision) — declare as class only
                g.add((term_iri, RDF.type, OWL.Class))
        else:
            # Root node — try to anchor to upper-ontology via Category
            g.add((term_iri, RDF.type, OWL.Class))
            category = str(row.get("Category", "")).strip()
            if category and not _is_upper_iri(term_iri):
                upper_iri_str = UPPER_IRIS.get(category) or _UPPER_IRIS_LOWER.get(category.lower())
                if upper_iri_str:
                    g.add((term_iri, RDFS.subClassOf, URIRef(upper_iri_str)))

        # Label
        g.add((term_iri, RDFS.label, Literal(term, lang="en")))

        # NLD as comment
        nld = nld_map.get(term, "")
        if nld and not str(nld).startswith("ERROR"):
            g.add((term_iri, RDFS.comment, Literal(str(nld), lang="en")))

        # Ensure parent class is also declared (but not if parent is an individual)
        if has_parent and parent_iri is not None and parent not in UPPER_IRIS and not is_intermediate:
            if str(parent_iri) not in _individual_iris:
                g.add((parent_iri, RDF.type, OWL.Class))

    # ── Upper-ontology backbone: add subClassOf chains + labels ──
    # Collect all upper-level IRIs referenced in subClassOf and rdf:type triples
    referenced_uppers = set()
    for _, _, o in g.triples((None, RDFS.subClassOf, None)):
        o_str = str(o)
        if o_str in _UPPER_IRI_VALUES:
            referenced_uppers.add(o_str)
    for _, _, o in g.triples((None, RDF.type, None)):
        o_str = str(o)
        if o_str in _UPPER_IRI_VALUES:
            referenced_uppers.add(o_str)

    n_backbone = _add_upper_backbone(g, referenced_uppers)
    if n_backbone:
        log.detail(f"Added {n_backbone} upper-ontology backbone triples (GeoCore/GeoReservoir → BFO)")

    # ── Relation restrictions (Step 6b) ──
    n_restrictions = 0
    if relations_csv and os.path.exists(relations_csv):
        rel_df = read_csv(relations_csv)
        accepted = rel_df[rel_df["Validation_Status"] == "ACCEPTED"]
        log.info(f"Adding {len(accepted)} relation restrictions from {relations_csv}")

        # Declare used object properties
        declared_props = set()
        _skipped_phantom_fillers: list[tuple[str, str, str]] = []
        for _, rel in accepted.iterrows():
            prop_iri_str = rel.get("Property_IRI", "")
            prop_name = rel.get("Property", "")
            if prop_iri_str and prop_iri_str not in declared_props:
                prop_uri = URIRef(prop_iri_str)
                g.add((prop_uri, RDF.type, OWL.ObjectProperty))
                g.add((prop_uri, RDFS.label, Literal(prop_name.replace("_", " "), lang="en")))
                declared_props.add(prop_iri_str)

        # Add existential restrictions: Class ⊑ ∃property.Filler
        for _, rel in accepted.iterrows():
            term_str = str(rel["Term"]).strip()
            filler_str = str(rel["Filler"]).strip()

            # Skip if subject is not a known taxonomy term or upper-ontology IRI
            # — avoids creating restrictions for terms only in relations CSV
            if term_str not in _taxonomy_terms and term_str.lower() not in _UPPER_IRIS_LOWER:
                continue

            term_iri = _term_to_iri(term_str)
            filler_iri = _term_to_iri(filler_str)
            prop_iri_str = rel.get("Property_IRI", "")
            if not prop_iri_str:
                continue

            prop_uri = URIRef(prop_iri_str)

            # Never create restrictions between two upper-level entities
            if _is_upper_iri(term_iri) and _is_upper_iri(filler_iri):
                continue

            # Skip relations where subject is an individual (factual, not ontological)
            if str(term_iri) in _individual_iris:
                continue

            # Declare filler as a class only if it's a known taxonomy term or
            # upper-ontology IRI — avoids minting phantom orphan classes.
            filler_is_individual = str(filler_iri) in _individual_iris
            filler_is_known = (
                filler_str in _taxonomy_terms
                or filler_str.lower() in _UPPER_IRIS_LOWER
            )

            # Skip entire restriction if filler is unknown — referencing an
            # unknown IRI in owl:someValuesFrom would create a phantom class
            # under owl:Thing in Protégé with no label, comment, or parent.
            if not filler_is_individual and not filler_is_known:
                _skipped_phantom_fillers.append((term_str, str(rel.get("Property", "")), filler_str))
                continue

            if not filler_is_individual and filler_is_known:
                g.add((filler_iri, RDF.type, OWL.Class))

            # Restriction: use owl:hasValue for individual fillers,
            # owl:someValuesFrom for class fillers (OWL 2 compliance)
            restriction = BNode()
            g.add((restriction, RDF.type, OWL.Restriction))
            g.add((restriction, OWL.onProperty, prop_uri))
            if filler_is_individual:
                g.add((restriction, OWL.hasValue, filler_iri))
            else:
                g.add((restriction, OWL.someValuesFrom, filler_iri))
            g.add((term_iri, RDFS.subClassOf, restriction))
            n_restrictions += 1

        if _skipped_phantom_fillers:
            log.warn(
                f"Skipped {len(_skipped_phantom_fillers)} relation restriction(s) "
                f"with phantom fillers (filler not in taxonomy and not an upper-ontology class). "
                f"These would have appeared as orphan classes under owl:Thing in Protégé. "
                f"Upstream cause is usually the ontology critic removing a term without stripping "
                f"the relations that reference it as a filler."
            )
            for term_str, prop, filler_str in _skipped_phantom_fillers[:10]:
                log.detail(f"  {term_str} --[{prop}]--> {filler_str}  (phantom)")
            if len(_skipped_phantom_fillers) > 10:
                log.detail(f"  ... and {len(_skipped_phantom_fillers) - 10} more")

    # ── Second backbone pass: pick up upper IRIs referenced in restrictions ──
    extra_uppers = set()
    for _, _, o in g.triples((None, OWL.someValuesFrom, None)):
        o_str = str(o)
        if o_str in _UPPER_IRI_VALUES:
            extra_uppers.add(o_str)
    n_extra = _add_upper_backbone(g, extra_uppers)
    if n_extra:
        log.detail(f"Added {n_extra} extra backbone triples for restriction fillers")

    # ── Disjointness conflict detection & repair ──
    repairs = _detect_and_repair_disjointness(g, df)
    if repairs:
        log.info(f"Disjointness repairs: {len(repairs)} conflicting edges removed")

    # ── BFO disjointness axioms ──
    for iri_a, iri_b in _BFO_DISJOINT:
        g.add((URIRef(iri_a), OWL.disjointWith, URIRef(iri_b)))
    log.detail(f"Added {len(_BFO_DISJOINT)} BFO disjointness axioms")

    # Serialize
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    g.serialize(destination=output_path, format="turtle")

    # Stats
    n_classes = len(list(g.subjects(RDF.type, OWL.Class)))
    n_individuals = len(list(g.subjects(RDF.type, OWL.NamedIndividual)))
    n_triples = len(g)

    log.success(f"OWL ontology exported: {output_path}")
    log.detail(f"Classes: {n_classes}, Individuals: {n_individuals}, Triples: {n_triples}")
    if n_restrictions > 0:
        log.detail(f"Existential restrictions: {n_restrictions}")
    log.detail(f"Format: Turtle (.ttl) — open in Protege to verify")

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("taxonomy_csv", help="Path to taxonomy CSV")
    parser.add_argument("--nld", default=None, help="Path to NLD CSV for adding definitions")
    parser.add_argument("--output", default=None, help="Output .ttl path")
    parser.add_argument("--relations", default=None, help="Path to 6b_relations.csv")
    args = parser.parse_args()
    run_owl_export(args.taxonomy_csv, args.nld, args.output, args.relations)
