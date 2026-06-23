"""Quick structural diagnosis of a candidate inconsistent ontology."""
import sys
from collections import Counter
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

ttl = sys.argv[1] if len(sys.argv) > 1 else "output/archive/run_a_apr2026_ontology.ttl"
g = Graph()
g.parse(ttl, format="turtle")

print(f"=== {ttl} ===")
print(f"Triples: {len(g)}")

print("\nowl:imports targets:")
for s, _, o in g.triples((None, OWL.imports, None)):
    print(f"  {s} -> {o}")

ontologies = list(g.subjects(RDF.type, OWL.Ontology))
print(f"\nowl:Ontology declarations: {[str(o) for o in ontologies]}")

# Multi-typed individuals
inds = list(g.subjects(RDF.type, OWL.NamedIndividual))
print(f"\nIndividuals: {len(inds)}")
multi = []
for i in inds:
    types = [str(t) for t in g.objects(i, RDF.type)
             if isinstance(t, URIRef) and t not in (OWL.NamedIndividual, OWL.Class, OWL.Ontology)]
    if len(types) > 1:
        multi.append((str(i), types))

print(f"Individuals with >1 non-meta type: {len(multi)}")
for iri, types in multi[:20]:
    print(f"  {iri}")
    for t in types:
        print(f"      a  {t}")

# Multi-typed classes (subclass of multiple parents in upper ontology)
classes = list(g.subjects(RDF.type, OWL.Class))
print(f"\nClasses: {len(classes)}")
multi_parent = []
for c in classes:
    parents = [str(p) for p in g.objects(c, RDFS.subClassOf) if isinstance(p, URIRef)]
    if len(parents) > 1:
        multi_parent.append((str(c), parents))
print(f"Classes with >1 direct parent: {len(multi_parent)}")
for iri, parents in multi_parent[:10]:
    print(f"  {iri}")
    for p in parents:
        print(f"      subClassOf  {p}")

# Disjointness declarations
disjoint = list(g.triples((None, OWL.disjointWith, None)))
print(f"\nowl:disjointWith axioms: {len(disjoint)}")
adw = list(g.subjects(RDF.type, OWL.AllDisjointClasses))
print(f"owl:AllDisjointClasses: {len(adw)}")

# Type distribution among all upper IRIs used
upper_type_pairs = Counter()
for i in inds:
    types = [str(t) for t in g.objects(i, RDF.type)
             if isinstance(t, URIRef) and t not in (OWL.NamedIndividual,)]
    for t in types:
        if "GEOCORE_" in t or "GEORES_" in t or "BFO_" in t:
            upper_type_pairs[t] += 1
print("\nTop upper types used on individuals:")
for t, n in upper_type_pairs.most_common(10):
    print(f"  {n:4d}  {t}")
