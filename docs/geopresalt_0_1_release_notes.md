# GeoPreSalt 0.1 research-release notes

## What this release represents

GeoPreSalt 0.1 is the frozen ontology produced and evaluated in the research
study. It is retained with known limitations, not presented as a logically
coherent or independently validated ontology for general application use.
The pipeline maintenance changes improve installation, configuration checks,
and new-domain setup without changing this evaluated artifact.

The artifact is `output/final/presalt_ontology.ttl`, with SHA-256:

```text
5b1c950609b5a8b2cd2e2ade78877f769f950c891eb26b84bba166956bd195bb
```

The ten frozen input files and their hashes are recorded in
`evaluation_study/inputs/manifest.json`.

## Known logical limitations

The read-only assessment on 16 September 2026 used the saved BFO, GeoCore, and
GeoReservoir files with HermiT 1.3.8.1099 and Java 21.0.12.1. The combined
ontology was consistent, but these **13 named classes were unsatisfiable**:

- BasementHigh
- Chalcedony
- Graben
- HalfGraben
- InSituFacies
- Micropore
- MuddySpherulitestone
- PassiveMarginBasin
- ReworkedCarbonate
- ReworkFacies
- Shrubstone
- SilicaCement
- Vug

An unsatisfiable class cannot have an instance while satisfying all the
included axioms. Overall ontology consistency does not establish that every
class is usable. A complete RO-inclusive satisfiability result is not
established; the narrower result must not be described as a full-import pass.
These findings are carried forward from that assessment, not from a new
reasoner run under the dependency lock in this maintenance change.

Existing expert judgments concern sampled material and do not establish the
correctness of every class or relation. Query results and structural checks
do not establish adequate domain coverage or suitability for reservoir-data
integration.

## Preservation and current validation

The committed offline regression can be run from the repository root:

```bash
python test/test_shared_relation_regeneration.py -v
```

It verifies all frozen input hashes, calls the exporter with the approved
taxonomy, relations, instances, and defined-class CSVs, and compares the result
with the frozen artifact by RDF graph isomorphism and a zero-added/zero-removed
triple diff. Both graphs contain 1,819 triples. The original Turtle file is
never rewritten; regenerated blank-node identifiers can change its byte-level
serialization without changing the graph.

Network connections are disabled during that export and syntax/structure
verification. OOPS! and HermiT are skipped. Passing this regression preserves
the known defects as well as the evaluated content; it is not a repair or a
new scientific evaluation.

The maintenance environment is checked locally on macOS Apple Silicon.
Linux dependency resolution is available, but Linux execution, live-model
runs, and Java reasoning under this lock remain unverified. Model deployment
versions and retrieved-model snapshots must be recorded separately for a
new study run; the Python lock does not make LLM results bit-reproducible.

GeoPreSalt 0.2 repairs and the guided proposal/approval workflow are separate
work and are not included in this release preparation.
