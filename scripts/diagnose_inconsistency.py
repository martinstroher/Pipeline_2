"""Try to identify the contradiction in an inconsistent OWL ontology."""
import sys
import tempfile
from pathlib import Path
from rdflib import Graph

if len(sys.argv) != 2:
    print("Usage: python scripts/diagnose_inconsistency.py <ttl>")
    sys.exit(1)

ttl = sys.argv[1]
g = Graph()
g.parse(ttl, format="turtle")

tmp = tempfile.NamedTemporaryFile(suffix=".nt", delete=False)
g.serialize(tmp, format="ntriples")
tmp.close()

import owlready2
world = owlready2.World()
onto = world.get_ontology("http://test/").load(fileobj=open(tmp.name, "rb"), format="ntriples")

print(f"Loaded ontology: {len(g)} triples")
print()

try:
    with onto:
        owlready2.sync_reasoner_hermit(world, infer_property_values=False, debug=2)
    print("Reasoner finished without raising.")
    inc = list(world.inconsistent_classes())
    print(f"Inconsistent classes: {len(inc)}")
    for c in inc[:20]:
        print(f"  - {c.iri}")
except owlready2.OwlReadyInconsistentOntologyError as e:
    print("OwlReadyInconsistentOntologyError raised:")
    print(str(e)[:5000])
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")
