"""Side-by-side comparison of two PreSaltOntoLearn ontology TTLs.

Usage:
    python scripts/compare_ontologies.py <old.ttl> <new.ttl>

Computes structural, semantic, anchoring, and reasoning metrics for each
ontology and prints a comparison table. Designed to evaluate Run A
(pre-refactor) vs Run B (post-refactor) outputs.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

BFO_PREFIX = "http://purl.obolibrary.org/obo/BFO_"
GEO_PREFIX = "https://www.inf.ufrgs.br/bdi/ontologies/"
PRESALT_PREFIX = "https://w3id.org/presalt-onto#"


def load(ttl_path: str) -> Graph:
    g = Graph()
    g.parse(ttl_path, format="turtle")
    return g


def is_presalt(u) -> bool:
    return str(u).startswith(PRESALT_PREFIX)


def is_upper(u) -> bool:
    s = str(u)
    return s.startswith(BFO_PREFIX) or s.startswith(GEO_PREFIX)


def upper_bucket(u) -> str:
    s = str(u)
    if s.startswith(BFO_PREFIX):
        return "BFO"
    if s.startswith(GEO_PREFIX):
        if "GEORES_" in s:
            return "GeoReservoir"
        if "GEOCORE_" in s:
            return "GeoCore"
    return "other"


def analyse(ttl_path: str) -> dict:
    g = load(ttl_path)

    classes = set(g.subjects(RDF.type, OWL.Class))
    individuals = set(g.subjects(RDF.type, OWL.NamedIndividual))
    object_props = set(g.subjects(RDF.type, OWL.ObjectProperty))
    data_props = set(g.subjects(RDF.type, OWL.DatatypeProperty))
    annotation_props = set(g.subjects(RDF.type, OWL.AnnotationProperty))

    presalt_classes = {c for c in classes if is_presalt(c)}
    presalt_individuals = {i for i in individuals if is_presalt(i)}
    upper_classes_referenced = {c for c in classes if is_upper(c)}

    # subClassOf edges (presalt subject only)
    sub_edges = [
        (s, o) for s, _, o in g.triples((None, RDFS.subClassOf, None))
        if isinstance(s, URIRef) and isinstance(o, URIRef)
    ]
    presalt_sub_edges = [(s, o) for s, o in sub_edges if is_presalt(s)]

    # rdf:type (instance-of) edges (presalt individual only)
    type_edges_individuals = [
        (s, o) for s, _, o in g.triples((None, RDF.type, None))
        if isinstance(s, URIRef) and isinstance(o, URIRef)
        and is_presalt(s) and o not in (OWL.NamedIndividual, OWL.Class,
                                          OWL.ObjectProperty, OWL.DatatypeProperty,
                                          OWL.AnnotationProperty, OWL.Ontology)
    ]

    # Anchoring: how many presalt entities reach an upper IRI via subClassOf or rdf:type?
    anchored_via_subclass = {s for s, o in presalt_sub_edges if is_upper(o)}
    anchored_via_type = {s for s, o in type_edges_individuals if is_upper(o)}
    anchored = anchored_via_subclass | anchored_via_type

    upper_targets = Counter()
    for _, o in presalt_sub_edges:
        if is_upper(o):
            upper_targets[upper_bucket(o)] += 1
    for _, o in type_edges_individuals:
        if is_upper(o):
            upper_targets[upper_bucket(o)] += 1

    # Distinct upper IRIs touched
    distinct_upper_iris = {str(o) for _, o in presalt_sub_edges if is_upper(o)} | \
                          {str(o) for _, o in type_edges_individuals if is_upper(o)}
    distinct_upper_by_bucket = defaultdict(set)
    for iri in distinct_upper_iris:
        distinct_upper_by_bucket[upper_bucket(URIRef(iri))].add(iri)

    # Labels / NLD coverage
    labelled = {e for e in (presalt_classes | presalt_individuals)
                if list(g.objects(e, RDFS.label))}
    with_comment = {e for e in (presalt_classes | presalt_individuals)
                    if list(g.objects(e, RDFS.comment))}

    # Self-referential subclass
    self_refs = [str(s) for s, o in sub_edges if s == o]

    # Build a parent map of presalt-only subclassing (for depth/orphan analysis)
    parent_map = defaultdict(set)
    for s, o in presalt_sub_edges:
        parent_map[s].add(o)

    # Orphans: presalt classes with no rdfs:subClassOf at all
    orphans = [c for c in presalt_classes if not parent_map.get(c)]

    # Depth: longest chain from a presalt class up to either an upper IRI
    # or another presalt class with no parent. Compute via memoized DFS.
    depth_cache: dict[URIRef, int] = {}

    def depth(node: URIRef, seen: set[URIRef] | None = None) -> int:
        if node in depth_cache:
            return depth_cache[node]
        seen = seen or set()
        if node in seen:
            return 0  # cycle guard
        parents = parent_map.get(node, set())
        if not parents:
            depth_cache[node] = 0
            return 0
        best = 0
        for p in parents:
            if is_upper(p):
                best = max(best, 1)
            elif is_presalt(p):
                best = max(best, 1 + depth(p, seen | {node}))
        depth_cache[node] = best
        return best

    depths = [depth(c) for c in presalt_classes]
    depth_dist = Counter(depths)

    # Branching: per upper-IRI, how many direct presalt children?
    upper_children = Counter()
    for s, o in presalt_sub_edges:
        if is_upper(o):
            upper_children[str(o)] += 1
    for s, o in type_edges_individuals:
        if is_upper(o):
            upper_children[str(o)] += 1

    # Object-property usage in OWL restrictions (DL axioms): every
    # owl:Restriction blank node with owl:onProperty ?p contributes one
    # use of ?p, attributed to the presalt class that has the restriction
    # as a subClassOf object.
    op_uses = Counter()
    restriction_count = 0
    restriction_kinds = Counter()
    for r in g.subjects(RDF.type, OWL.Restriction):
        prop = next(g.objects(r, OWL.onProperty), None)
        if prop is None:
            continue
        # Owner: any presalt class that has this restriction as subClassOf object
        owners = [s for s in g.subjects(RDFS.subClassOf, r) if is_presalt(s)]
        if not owners:
            continue
        restriction_count += len(owners)
        op_uses[str(prop)] += len(owners)
        if list(g.objects(r, OWL.someValuesFrom)):
            restriction_kinds["someValuesFrom"] += len(owners)
        elif list(g.objects(r, OWL.allValuesFrom)):
            restriction_kinds["allValuesFrom"] += len(owners)
        elif list(g.objects(r, OWL.hasValue)):
            restriction_kinds["hasValue"] += len(owners)
        else:
            restriction_kinds["other"] += len(owners)

    # Transitive anchoring: a presalt entity is "transitively anchored" if
    # following its subClassOf / rdf:type chain eventually hits an upper IRI.
    def reaches_upper(node: URIRef, seen: set[URIRef] | None = None) -> bool:
        seen = seen or set()
        if node in seen:
            return False
        seen.add(node)
        for o in g.objects(node, RDFS.subClassOf):
            if isinstance(o, URIRef):
                if is_upper(o):
                    return True
                if is_presalt(o) and reaches_upper(o, seen):
                    return True
        for o in g.objects(node, RDF.type):
            if isinstance(o, URIRef) and is_upper(o):
                return True
        return False

    transitively_anchored = {e for e in (presalt_classes | presalt_individuals)
                              if reaches_upper(e)}

    return {
        "path": ttl_path,
        "triples": len(g),
        "classes_total": len(classes),
        "classes_presalt": len(presalt_classes),
        "classes_upper_referenced": len(upper_classes_referenced),
        "individuals_total": len(individuals),
        "individuals_presalt": len(presalt_individuals),
        "object_properties": len(object_props),
        "data_properties": len(data_props),
        "annotation_properties": len(annotation_props),
        "subclass_edges_total": len(sub_edges),
        "subclass_edges_presalt_subject": len(presalt_sub_edges),
        "type_edges_presalt_individuals": len(type_edges_individuals),
        "anchored_entities": len(anchored),
        "anchoring_rate": (
            len(anchored) / (len(presalt_classes) + len(presalt_individuals))
            if (presalt_classes or presalt_individuals) else 0.0
        ),
        "transitively_anchored": len(transitively_anchored),
        "transitive_anchoring_rate": (
            len(transitively_anchored) / (len(presalt_classes) + len(presalt_individuals))
            if (presalt_classes or presalt_individuals) else 0.0
        ),
        "upper_targets_by_tier": dict(upper_targets),
        "distinct_upper_iris": len(distinct_upper_iris),
        "distinct_upper_by_tier": {k: len(v) for k, v in distinct_upper_by_bucket.items()},
        "label_coverage": len(labelled) / max(1, len(presalt_classes | presalt_individuals)),
        "comment_coverage": len(with_comment) / max(1, len(presalt_classes | presalt_individuals)),
        "self_referential_subclass": len(self_refs),
        "orphan_presalt_classes": len(orphans),
        "depth_max": max(depths) if depths else 0,
        "depth_mean": mean(depths) if depths else 0,
        "depth_median": median(depths) if depths else 0,
        "depth_distribution": dict(sorted(depth_dist.items())),
        "object_property_relations": restriction_count,
        "distinct_object_properties_used": len(op_uses),
        "restriction_kinds": dict(restriction_kinds),
        "top_10_properties": op_uses.most_common(10),
        "branching_top10_upper_anchors": upper_children.most_common(10),
    }


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_num(x):
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def render_table(a: dict, b: dict, label_a: str, label_b: str) -> None:
    rows = [
        ("Triples (total)",                          a["triples"], b["triples"]),
        ("Classes (total)",                          a["classes_total"], b["classes_total"]),
        ("  ├ presalt classes",                      a["classes_presalt"], b["classes_presalt"]),
        ("  └ upper IRIs declared as classes",       a["classes_upper_referenced"], b["classes_upper_referenced"]),
        ("Individuals (total)",                      a["individuals_total"], b["individuals_total"]),
        ("  └ presalt individuals",                  a["individuals_presalt"], b["individuals_presalt"]),
        ("Object properties",                        a["object_properties"], b["object_properties"]),
        ("Data properties",                          a["data_properties"], b["data_properties"]),
        ("Annotation properties",                    a["annotation_properties"], b["annotation_properties"]),
        ("",                                          "", ""),
        ("subClassOf edges (all)",                   a["subclass_edges_total"], b["subclass_edges_total"]),
        ("subClassOf edges (presalt subject)",       a["subclass_edges_presalt_subject"], b["subclass_edges_presalt_subject"]),
        ("rdf:type edges (presalt individuals)",     a["type_edges_presalt_individuals"], b["type_edges_presalt_individuals"]),
        ("",                                          "", ""),
        ("Entities anchored to upper ontology",      a["anchored_entities"], b["anchored_entities"]),
        ("Direct anchoring rate",                    fmt_pct(a["anchoring_rate"]), fmt_pct(b["anchoring_rate"])),
        ("Transitively anchored entities",           a["transitively_anchored"], b["transitively_anchored"]),
        ("Transitive anchoring rate",                fmt_pct(a["transitive_anchoring_rate"]), fmt_pct(b["transitive_anchoring_rate"])),
        ("Distinct upper IRIs touched",              a["distinct_upper_iris"], b["distinct_upper_iris"]),
        ("  ├ BFO",                                  a["distinct_upper_by_tier"].get("BFO", 0), b["distinct_upper_by_tier"].get("BFO", 0)),
        ("  ├ GeoCore",                              a["distinct_upper_by_tier"].get("GeoCore", 0), b["distinct_upper_by_tier"].get("GeoCore", 0)),
        ("  └ GeoReservoir",                         a["distinct_upper_by_tier"].get("GeoReservoir", 0), b["distinct_upper_by_tier"].get("GeoReservoir", 0)),
        ("Anchor edges by tier (sum)",               "", ""),
        ("  ├ BFO edges",                            a["upper_targets_by_tier"].get("BFO", 0), b["upper_targets_by_tier"].get("BFO", 0)),
        ("  ├ GeoCore edges",                        a["upper_targets_by_tier"].get("GeoCore", 0), b["upper_targets_by_tier"].get("GeoCore", 0)),
        ("  └ GeoReservoir edges",                   a["upper_targets_by_tier"].get("GeoReservoir", 0), b["upper_targets_by_tier"].get("GeoReservoir", 0)),
        ("",                                          "", ""),
        ("Label coverage (presalt entities)",        fmt_pct(a["label_coverage"]), fmt_pct(b["label_coverage"])),
        ("NLD/comment coverage (presalt entities)",  fmt_pct(a["comment_coverage"]), fmt_pct(b["comment_coverage"])),
        ("",                                          "", ""),
        ("Self-referential subClassOf (BUG)",        a["self_referential_subclass"], b["self_referential_subclass"]),
        ("Orphan presalt classes (no parent at all)",a["orphan_presalt_classes"], b["orphan_presalt_classes"]),
        ("",                                          "", ""),
        ("Taxonomy max depth (from presalt class)",  a["depth_max"], b["depth_max"]),
        ("Taxonomy mean depth",                      fmt_num(a["depth_mean"]), fmt_num(b["depth_mean"])),
        ("Taxonomy median depth",                    fmt_num(a["depth_median"]), fmt_num(b["depth_median"])),
        ("",                                          "", ""),
        ("Object-property relation triples (non-tax)", a["object_property_relations"], b["object_property_relations"]),
        ("Distinct object properties actually used", a["distinct_object_properties_used"], b["distinct_object_properties_used"]),
        ("  ├ someValuesFrom restrictions",          a["restriction_kinds"].get("someValuesFrom", 0), b["restriction_kinds"].get("someValuesFrom", 0)),
        ("  ├ allValuesFrom restrictions",           a["restriction_kinds"].get("allValuesFrom", 0), b["restriction_kinds"].get("allValuesFrom", 0)),
        ("  └ hasValue restrictions",                a["restriction_kinds"].get("hasValue", 0), b["restriction_kinds"].get("hasValue", 0)),
    ]

    col1 = max(len(r[0]) for r in rows) + 2
    col2 = max(len(label_a), max(len(str(r[1])) for r in rows)) + 2
    col3 = max(len(label_b), max(len(str(r[2])) for r in rows)) + 2

    print()
    print(f"{'Metric'.ljust(col1)}{label_a.ljust(col2)}{label_b.ljust(col3)}")
    print("-" * (col1 + col2 + col3))
    for name, va, vb in rows:
        print(f"{name.ljust(col1)}{str(va).ljust(col2)}{str(vb).ljust(col3)}")

    print()
    print(f"Depth distribution {label_a}: {a['depth_distribution']}")
    print(f"Depth distribution {label_b}: {b['depth_distribution']}")
    print()
    print(f"Top 10 object properties — {label_a}:")
    for prop, n in a["top_10_properties"]:
        print(f"  {n:5d}  {prop}")
    print()
    print(f"Top 10 object properties — {label_b}:")
    for prop, n in b["top_10_properties"]:
        print(f"  {n:5d}  {prop}")
    print()
    print(f"Top 10 upper-IRI anchors by direct children — {label_a}:")
    for iri, n in a["branching_top10_upper_anchors"]:
        print(f"  {n:5d}  {iri}")
    print()
    print(f"Top 10 upper-IRI anchors by direct children — {label_b}:")
    for iri, n in b["branching_top10_upper_anchors"]:
        print(f"  {n:5d}  {iri}")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_ontologies.py <old.ttl> <new.ttl>")
        sys.exit(1)

    old_path = sys.argv[1]
    new_path = sys.argv[2]
    for p in (old_path, new_path):
        if not Path(p).is_file():
            print(f"Not found: {p}")
            sys.exit(2)

    print(f"Loading old: {old_path}")
    a = analyse(old_path)
    print(f"Loading new: {new_path}")
    b = analyse(new_path)

    render_table(a, b, "RUN A (old)", "RUN B (new)")

    # Reasoner check on both
    print()
    print("=" * 70)
    print("HermiT reasoner check (via src.modules.emit.verifier)")
    print("=" * 70)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.modules.emit.verifier import _verify_hermit  # noqa
    for label, path in (("RUN A (old)", old_path), ("RUN B (new)", new_path)):
        print(f"\n{label} → {path}")
        result = _verify_hermit(path)
        print(f"  status:      {result['status']}")
        print(f"  consistent:  {result['consistent']}")
        if result["unsatisfiable_classes"]:
            print(f"  unsatisfiable: {len(result['unsatisfiable_classes'])} classes")
            for c in result["unsatisfiable_classes"][:10]:
                print(f"     - {c}")
        if result["errors"]:
            for err in result["errors"]:
                print(f"  error: {err}")


if __name__ == "__main__":
    main()
