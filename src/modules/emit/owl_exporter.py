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
import ast
import shutil
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


# Minor words kept lower-case in Title Case labels (matches GeoCore's
# "Body of Rock" style).
_LABEL_MINOR_WORDS = {
    "of", "the", "a", "an", "and", "or", "in", "on", "for", "to",
    "with", "by", "from", "as", "at", "per",
}


def _cap_token(token: str) -> str:
    """Capitalise the first letter of a token, preserving acronyms / existing
    capitals (`CO2` stays `CO2`, `rock` -> `Rock`)."""
    if not token:
        return token
    if token[0].isupper() or any(c.isupper() for c in token[1:]):
        return token
    return token[:1].upper() + token[1:]


def _title_case(label: str) -> str:
    """Title-case a label in the GeoCore/GeoReservoir house style
    (`Body of Rock`, `Pre-Salt Sequence`): capitalise each word and each
    hyphen-separated part, keep minor words lower-case (except the first), and
    leave acronyms / already-capitalised tokens untouched. Used only for class
    and individual labels — never for object properties."""
    label = (label or "").strip()
    if not label:
        return label
    out: list[str] = []
    for wi, word in enumerate(label.split()):
        parts = word.split("-")
        cased: list[str] = []
        for pi, part in enumerate(parts):
            if not part:
                cased.append(part)
                continue
            is_first = wi == 0 and pi == 0
            if part.lower() in _LABEL_MINOR_WORDS and not is_first:
                cased.append(part.lower())
            else:
                cased.append(_cap_token(part))
        out.append("-".join(cased))
    return " ".join(out)


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
                    # Category cannot disambiguate — it resolves to a *third*
                    # branch disjoint with both conflict sides. This is the
                    # case-collision signature: two taxonomy rows that normalise
                    # to one IRI (an LLM-invented intermediate genus + a real
                    # term whose critic reparent points at a bare BFO metatype
                    # root) have their parents silently unioned across a disjoint
                    # boundary. Fallback: if the class is a DIRECT subclass of
                    # exactly one of the two disjoint roots while reaching the
                    # other side only transitively (via a substantive published
                    # genus), the direct bare-root edge is the artifact — drop it
                    # and keep the genus. (This branch only runs where the repair
                    # previously gave up, so a currently-satisfiable class is
                    # never altered.)
                    direct_parents = dp.get(cls_str, set())
                    a_direct = iri_a in direct_parents
                    b_direct = iri_b in direct_parents
                    if a_direct != b_direct:
                        remove_side = iri_a if a_direct else iri_b
                        keep_side = iri_b if a_direct else iri_a
                        log.warn(
                            f"  Disjointness fallback: {_local_name(cls_str)} "
                            f"— dropping direct upper-root parent "
                            f"{_local_name(remove_side)}; kept genus toward "
                            f"{_local_name(keep_side)} (category did not disambiguate)"
                        )
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


# ── Critic-driven emission helpers (added for the `validate` verb) ────────
#
# All IRIs are pulled from ontology_config at call time via cfg.upper_iris()
# and cfg.all_relations() — no constants. If the config drops a relevant
# class/property, the helper logs and skips rather than emitting bad RDF.

def _emit_minted_properties(g: Graph, minted_csv: str) -> int:
    """Declare each critic-minted ObjectProperty with subPropertyOf, domain, range.

    Reads `validate_minted_properties.csv` (columns:
    Name, IRI, ParentProperty, Domain, Range, Justification, Timestamp).
    """
    if not minted_csv or not os.path.exists(minted_csv):
        return 0
    df = read_csv(minted_csv)
    if df.empty:
        return 0
    relations = _CFG.all_relations()
    upper = {k.lower(): v for k, v in _CFG.upper_iris().items()}
    n = 0
    for _, r in df.iterrows():
        iri_str = str(r.get("IRI", "")).strip()
        name = str(r.get("Name", "")).strip()
        if not iri_str or not name:
            continue
        prop_uri = URIRef(iri_str)
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))
        g.add((prop_uri, RDFS.label, Literal(name.replace("_", " "), lang="en")))
        justification = str(r.get("Justification", "")).strip()
        if justification:
            g.add((prop_uri, RDFS.comment, Literal(f"[critic_minted] {justification}", lang="en")))
        parent = str(r.get("ParentProperty", "")).strip()
        if parent and parent in relations:
            g.add((prop_uri, RDFS.subPropertyOf, URIRef(relations[parent].iri)))
        # Domain/Range can be a metatype label or an upper-class label.
        for col, pred in (("Domain", RDFS.domain), ("Range", RDFS.range)):
            label = str(r.get(col, "")).strip()
            if not label:
                continue
            upper_iri = upper.get(label.lower())
            if upper_iri:
                g.add((prop_uri, pred, URIRef(upper_iri)))
        n += 1
    return n


def _emit_named_individuals(g: Graph, instances_csv: str) -> int:
    """Emit `<term> a owl:NamedIndividual, <target_class>` for each row in
    `validate_instances.csv` (columns: Term, Target_Class, Mint_Parent, …).

    Target_Class is matched against `cfg.upper_iris()` (case-insensitive,
    stripping a `bfo:`/`geocore:`/`georeservoir:`/`presalt:` prefix and
    de-CamelCasing if present). If it does not resolve but the row carries a
    `Mint_Parent` that does, a new domain class `<project>:<Target_Class>` is
    minted as a subclass of that parent and the individual is typed under it.
    If neither resolves, the row is skipped with a warning.
    """
    if not instances_csv or not os.path.exists(instances_csv):
        return 0
    df = read_csv(instances_csv)
    if df.empty:
        return 0
    upper = {k.lower(): v for k, v in _CFG.upper_iris().items()}

    def _resolve_upper(label: str) -> str | None:
        """Resolve a possibly-prefixed / CamelCase label to an upper-class IRI."""
        if not label:
            return None
        core = label.split(":", 1)[-1] if ":" in label else label
        candidates = [core, core.lower(),
                      re.sub(r"(?<!^)([A-Z])", r" \1", core).strip().lower()]
        return next((upper[c] for c in candidates if c in upper), None)

    n = 0
    for _, r in df.iterrows():
        term = str(r.get("Term", "")).strip()
        target = str(r.get("Target_Class", "")).strip()
        if not term or not target:
            continue
        resolved = _resolve_upper(target)
        type_iri = URIRef(resolved) if resolved else None
        if type_iri is None:
            # Fallback: critic proposed a new domain kind. Mint
            # <project>:<target> as a subclass of the resolved mint_parent.
            mint_parent_iri = _resolve_upper(str(r.get("Mint_Parent", "")).strip())
            if mint_parent_iri:
                target_label = target.split(":", 1)[-1] if ":" in target else target
                type_iri = _mint_presalt_iri(target_label)
                g.add((type_iri, RDF.type, OWL.Class))
                g.add((type_iri, RDFS.label, Literal(_title_case(target_label), lang="en")))
                g.add((type_iri, RDFS.subClassOf, URIRef(mint_parent_iri)))
                g.add((type_iri, RDFS.comment,
                       Literal("[critic CONVERT_TO_INSTANCE minted target class]", lang="en")))
        if type_iri is None:
            log.warn(f"  validate_instances: cannot resolve target class '{target}' for '{term}' "
                     f"(no usable Mint_Parent) — skipping")
            continue
        term_iri = _term_to_iri(term)
        g.add((term_iri, RDF.type, OWL.NamedIndividual))
        g.add((term_iri, RDF.type, type_iri))
        g.add((term_iri, RDFS.label, Literal(_title_case(term), lang="en")))
        reason = str(r.get("Reason", "")).strip()
        if reason:
            g.add((term_iri, RDFS.comment, Literal(f"[critic CONVERT_TO_INSTANCE] {reason}", lang="en")))
        n += 1
    return n


def _emit_defined_bearer_classes(g: Graph, defined_csv: str) -> int:
    """Emit `bearer owl:equivalentClass (genus ⊓ <prop> some <role>)` for each
    KEEP_AS_BEARER realizable bearer in ``defined_csv``.

    This makes a role-fused term (e.g. "carbonate reservoir") a *defined* class —
    "a carbonate rock that plays the reservoir role" — rather than a primitive
    rigid kind (the OntoClean mixin fix). The asserted `bearer ⊑ genus` is kept
    (browsable hierarchy + verifier anchoring); the loose role restriction is
    skipped in the relation loop so the role lives only inside this definition.
    """
    if not defined_csv or not os.path.exists(defined_csv):
        return 0
    from rdflib.collection import Collection

    df = read_csv(defined_csv)
    n = 0
    for _, r in df.iterrows():
        bearer = str(r["Bearer"]).strip()
        genus = str(r["Genus"]).strip()
        filler = str(r["Filler"]).strip()
        prop_iri = str(r.get("Property_IRI", "")).strip()
        if not (bearer and genus and filler and prop_iri):
            continue
        bearer_iri = _term_to_iri(bearer)
        genus_iri = _term_to_iri(genus)
        role_iri = _term_to_iri(filler)
        prop_uri = URIRef(prop_iri)

        restriction = BNode()
        g.add((restriction, RDF.type, OWL.Restriction))
        g.add((restriction, OWL.onProperty, prop_uri))
        g.add((restriction, OWL.someValuesFrom, role_iri))

        members = BNode()
        Collection(g, members, [genus_iri, restriction])
        defn = BNode()
        g.add((defn, RDF.type, OWL.Class))
        g.add((defn, OWL.intersectionOf, members))
        g.add((bearer_iri, OWL.equivalentClass, defn))
        n += 1
    return n


def _parse_members(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    return [part.strip() for part in re.split(r"[|,]", text) if part.strip()]


def _emit_domain_disjointness(g: Graph, disjointness_csv: str) -> int:
    """Emit high-confidence domain disjointness diagnostics as OWL axioms.

    Disabled by default in config because domain disjointness should be trusted
    only with reasoner validation enabled.
    """
    if not disjointness_csv or not os.path.exists(disjointness_csv):
        return 0
    from rdflib.collection import Collection

    df = read_csv(disjointness_csv)
    n = 0
    for _, row in df.iterrows():
        members = _parse_members(row.get("members", ""))
        if len(members) < 2:
            continue
        iris = [_term_to_iri(member) for member in members]
        if len(iris) == 2:
            g.add((iris[0], OWL.disjointWith, iris[1]))
        else:
            members_node = BNode()
            Collection(g, members_node, iris)
            ax = BNode()
            g.add((ax, RDF.type, OWL.AllDisjointClasses))
            g.add((ax, OWL.members, members_node))
        n += 1
    return n


def _emit_companion_axioms(g: Graph) -> int:
    """Emit BFO companion restrictions for presalt classes that descend from
    Quality or Role but lack the canonical inheres_in / realized_in restriction.

    Quality descendants → `inheres_in some IndependentContinuant`
    Role descendants    → `realized_in some Process`

    All IRIs are resolved at call time from ontology_config. If any required
    class/property is absent from config, this function is a no-op for that
    side.
    """
    upper = {k.lower(): v for k, v in _CFG.upper_iris().items()}
    relations = _CFG.all_relations()
    quality_iri = upper.get("quality")
    role_iri = upper.get("role")
    inheres_in_pc = relations.get("inheres_in")
    realized_in_pc = relations.get("realized_in")
    independent_iri = upper.get("independent continuant")
    process_iri = upper.get("process")

    def _has_restriction_with_prop(cls: URIRef, prop_iri: str) -> bool:
        for _, _, obj in g.triples((cls, RDFS.subClassOf, None)):
            if isinstance(obj, BNode):
                for _, _, p in g.triples((obj, OWL.onProperty, None)):
                    if str(p) == prop_iri:
                        return True
        return False

    def _ancestors_contain(cls: URIRef, target_iri: str, max_depth: int = 12) -> bool:
        seen: set[str] = set()
        stack: list[URIRef] = [cls]
        depth = 0
        while stack and depth < max_depth:
            depth += 1
            nxt: list[URIRef] = []
            for c in stack:
                for _, _, parent in g.triples((c, RDFS.subClassOf, None)):
                    if not isinstance(parent, URIRef):
                        continue
                    p_str = str(parent)
                    if p_str == target_iri:
                        return True
                    if p_str in seen:
                        continue
                    seen.add(p_str)
                    nxt.append(parent)
            stack = nxt
        return False

    # Collect presalt classes (declared in our namespace).
    presalt_ns_str = str(ONTO_NS)
    presalt_classes: set[URIRef] = {
        s for s in g.subjects(RDF.type, OWL.Class) if isinstance(s, URIRef) and str(s).startswith(presalt_ns_str)
    }

    n_emitted = 0
    pairs: list[tuple[str | None, "PropertyConstraint | None", str | None, str]] = [
        (quality_iri, inheres_in_pc, independent_iri, "inheres_in"),
        (role_iri, realized_in_pc, process_iri, "realized_in"),
    ]
    for ancestor_iri, prop_pc, filler_iri, prop_name in pairs:
        if not ancestor_iri or prop_pc is None or not filler_iri:
            log.detail(f"  companion-axiom skip: missing config for {prop_name} pair")
            continue
        prop_iri_str = prop_pc.iri
        for cls in presalt_classes:
            if not _ancestors_contain(cls, ancestor_iri):
                continue
            if _has_restriction_with_prop(cls, prop_iri_str):
                continue
            bn = BNode()
            g.add((bn, RDF.type, OWL.Restriction))
            g.add((bn, OWL.onProperty, URIRef(prop_iri_str)))
            g.add((bn, OWL.someValuesFrom, URIRef(filler_iri)))
            g.add((cls, RDFS.subClassOf, bn))
            n_emitted += 1
    return n_emitted


def _label_used_properties(g: Graph) -> int:
    """Declare + label every object property used via owl:onProperty that has no
    rdfs:label yet, so Protege shows a readable name (e.g. BFO_0000054 ->
    'realized in') even when the BFO/RO imports are not resolved. Labels come
    from the reference OWL files first, then the config relation names. Property
    labels stay lower-case (BFO/RO convention)."""
    _load_upper_parent_map()  # populates _UPPER_LABEL_FROM_OWL
    ref_labels = dict(_UPPER_LABEL_FROM_OWL or {})
    cfg_labels = {pc.iri: name.replace("_", " ") for name, pc in _CFG.all_relations().items()}
    n = 0
    seen: set[str] = set()
    for _, _, prop in g.triples((None, OWL.onProperty, None)):
        if not isinstance(prop, URIRef):
            continue
        pstr = str(prop)
        if pstr in seen:
            continue
        seen.add(pstr)
        if list(g.objects(prop, RDFS.label)):
            continue  # already labelled (e.g. relation-restriction properties)
        label = ref_labels.get(pstr) or cfg_labels.get(pstr) or _local_name(pstr)
        g.add((prop, RDF.type, OWL.ObjectProperty))
        g.add((prop, RDFS.label, Literal(label, lang="en")))
        n += 1
    return n


def run_owl_export(
    taxonomy_csv: str,
    nld_csv: str | None = None,
    output_path: str | None = None,
    relations_csv: str | None = None,
    minted_csv: str | None = None,
    instances_csv: str | None = None,
    defined_csv: str | None = None,
    disjointness_csv: str | None = None,
):
    """
    Export taxonomy to OWL Turtle format.

    Args:
        taxonomy_csv: Path to taxonomy CSV (output of taxonomy_builder)
        nld_csv: Optional path to NLD CSV (for adding definitions as rdfs:comment)
        output_path: Output .ttl path (default: derived from input)
        relations_csv: Optional path to relations CSV (existential restrictions)
        minted_csv: Optional path to validate_minted_properties.csv (critic-minted
                    object properties). Auto-derived from taxonomy_csv's dir if None.
        instances_csv: Optional path to validate_instances.csv (CONVERT_TO_INSTANCE
                       rows). Auto-derived from taxonomy_csv's dir if None.
        defined_csv: Optional path to validate_defined_classes.csv (KEEP_AS_BEARER
                     realizable bearers emitted as owl:equivalentClass definitions).
                     Auto-derived from taxonomy_csv's dir if None.
        disjointness_csv: Optional path to validate_disjointness.csv. Emitted only
                  when `lateral_coherence.disjointness.enabled` is true.
    """
    if output_path is None:
        base = os.path.splitext(taxonomy_csv)[0]
        output_path = base.replace("construct_taxonomy", "emit_ontology") + ".ttl"

    # Auto-derive critic-output paths from the taxonomy CSV's directory if not passed.
    tax_dir = os.path.dirname(taxonomy_csv) or "."
    if minted_csv is None:
        candidate = os.path.join(tax_dir, "validate_minted_properties.csv")
        if os.path.exists(candidate):
            minted_csv = candidate
    if instances_csv is None:
        candidate = os.path.join(tax_dir, "validate_instances.csv")
        if os.path.exists(candidate):
            instances_csv = candidate
    if defined_csv is None:
        candidate = os.path.join(tax_dir, "validate_defined_classes.csv")
        if os.path.exists(candidate):
            defined_csv = candidate
    if disjointness_csv is None:
        candidate = os.path.join(tax_dir, "validate_disjointness.csv")
        if os.path.exists(candidate):
            disjointness_csv = candidate

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
        g.add((term_iri, RDFS.label, Literal(_title_case(str(term)), lang="en")))

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
    # Role restrictions that are folded into a defined-class definition
    # (owl:equivalentClass, emitted later) must NOT also be emitted here as loose
    # subClassOf restrictions — that would double-encode the role.
    _defined_skip: set[tuple[str, str, str]] = set()
    if defined_csv and os.path.exists(defined_csv):
        _dc = read_csv(defined_csv)
        for _, _r in _dc.iterrows():
            _defined_skip.add((
                str(_r["Bearer"]).strip().lower(),
                str(_r.get("Property_IRI", "")).strip(),
                str(_r["Filler"]).strip().lower(),
            ))
    if relations_csv and os.path.exists(relations_csv):
        rel_df = read_csv(relations_csv)
        accepted = rel_df[rel_df["Validation_Status"] == "ACCEPTED"]
        if _CFG.lateral_coherence().emit_only_generic_relations and "Relation_Scope" in accepted.columns:
            n_before_scope = len(accepted)
            scopes = accepted["Relation_Scope"].fillna("generic").astype(str).str.strip().str.lower()
            accepted = accepted[scopes.isin(["", "generic"])]
            n_contextual = n_before_scope - len(accepted)
            if n_contextual:
                log.detail(f"Skipped {n_contextual} non-generic relation(s) from OWL class restrictions")
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

            # Skip a role restriction that is folded into this bearer's
            # owl:equivalentClass definition (avoids double-encoding).
            if (term_str.lower(), prop_iri_str, filler_str.lower()) in _defined_skip:
                continue

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

    # ── Critic-driven additions: minted properties, instances, companion axioms ──
    if minted_csv:
        n_minted = _emit_minted_properties(g, minted_csv)
        if n_minted:
            log.detail(f"Emitted {n_minted} critic-minted object properties from {minted_csv}")
    if instances_csv:
        n_inst = _emit_named_individuals(g, instances_csv)
        if n_inst:
            log.detail(f"Emitted {n_inst} CONVERT_TO_INSTANCE individuals from {instances_csv}")
    n_companion = _emit_companion_axioms(g)
    if n_companion:
        log.detail(f"Emitted {n_companion} BFO companion-axiom restrictions (Quality/Role descendants)")
    if defined_csv:
        n_defined = _emit_defined_bearer_classes(g, defined_csv)
        if n_defined:
            log.detail(f"Emitted {n_defined} defined bearer classes (equivalentClass genus ⊓ role) from {defined_csv}")
    lateral_cfg = _CFG.lateral_coherence()
    if disjointness_csv and lateral_cfg.emit_disjointness:
        java_available = bool(os.environ.get("JAVA_EXE") or shutil.which("java"))
        if lateral_cfg.require_reasoner_for_disjointness and not java_available:
            log.warn("Domain disjointness emission skipped: Java/HermiT is required by config but no Java executable was found")
        else:
            n_domain_disjoint = _emit_domain_disjointness(g, disjointness_csv)
            if n_domain_disjoint:
                log.detail(f"Emitted {n_domain_disjoint} domain disjointness axiom(s) from {disjointness_csv}")

    # ── Third backbone pass: link any upper classes introduced by minted
    #    target classes / instances / companion axioms up to BFO. ──
    post_uppers = set()
    for pred in (RDF.type, RDFS.subClassOf):
        for _, _, o in g.triples((None, pred, None)):
            o_str = str(o)
            if o_str in _UPPER_IRI_VALUES:
                post_uppers.add(o_str)
    n_post = _add_upper_backbone(g, post_uppers)
    if n_post:
        log.detail(f"Added {n_post} backbone triples for critic-introduced upper classes")

    # ── BFO disjointness axioms ──
    for iri_a, iri_b in _BFO_DISJOINT:
        g.add((URIRef(iri_a), OWL.disjointWith, URIRef(iri_b)))
    log.detail(f"Added {len(_BFO_DISJOINT)} BFO disjointness axioms")

    # ── Label every object property used in a restriction, so Protege shows a
    #    readable name even without resolving the BFO/RO imports ──
    n_prop_labels = _label_used_properties(g)
    if n_prop_labels:
        log.detail(f"Labelled {n_prop_labels} object propert{'y' if n_prop_labels == 1 else 'ies'} used in restrictions")

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
    parser.add_argument("--relations", default=None, help="Path to construct_relations.csv")
    parser.add_argument("--minted", default=None, help="Path to validate_minted_properties.csv")
    parser.add_argument("--instances", default=None, help="Path to validate_instances.csv")
    args = parser.parse_args()
    run_owl_export(
        args.taxonomy_csv, args.nld, args.output, args.relations,
        minted_csv=args.minted, instances_csv=args.instances,
    )
