# Advisor Review — T1 Ontology Analysis

Date: May 2026 | Reviewer: Orientadora | Ontology: output/refined/t1/7_ontology.ttl

## Executive Summary

Core architectural gap: pipeline correctly extracts BFO-typed relations (located_in → spatial container,
has_continuant_part → object, has_quality → quality) but NEVER feeds them back to refine classification.
Pipeline is strictly one-way: Categorize → Taxonomy → Relations → OWL.

Advisor conclusion: "If we add relation-based reclassification, we'd have ~100% correct classifications."

## 15 Issues Identified

### CRITICAL
- **Issue 6: Camboriú Formation** — contradictory relations: `basalt located_in Camboriú` (spatial) AND
  `tholeiitic basalt continuant_part_of camboriú` (object). Cannot be both.

### CLASSIFICATION ERRORS (relations know the right answer)
- **Issue 1: Accommodation Space** — Should be spatial region, NLD says "volume" misled LLM
- **Issue 2: Alkaline Lake** — Has located_in associations → spatial container, classified as Object
- **Issue 4: Basin / Issue 7: Campos Basin** — located_in filler proves spatial container, classified as Object
- **Issue 8: Carbonate Fabric** — Quality (SDC), wrongly participates in processes (only bearer can)
- **Issue 12: Structural Block** — Category says "Geological Object" but Parent is "Geological Structure" (contradictory)
- **Issue 14: Pore Space** — Category says "site" but Parent is "spatial region" (should be site)
- **Issue 15: Long tail** — Many terms with correct relations but imprecise/generic classification

### CORRECT BUT NOTED
- **Issue 3: Basement** — Correctly Object, but by accident (categorizer doesn't use relations)
- **Issue 5: Cabiúnas Formation** — Depositional Unit is valid but could be more precise
- **Issue 9: Composition/Dip/Density** — Correctly quality, not explicit in taxonomy
- **Issue 10: Diagenesis/Displacement** — Correctly process, not formally explicit
- **Issue 11: Geological Structures** — Many appear as has_continuant_part fillers (→Object) but classified as Structure (→GDC)
- **Issue 13: Cave/Channel under Pore** — Domain-correct (macroporosity) but Channel is duplicated (depositional vs dissolution)

## Root Cause: One-Way Pipeline
Relations extracted in Step 6b contain implicit BFO metatype knowledge via domain/range constraints
(defined in relation_validator.py), but this is used only for accept/reject, never for reclassification.

## Proposed Fix: Relation-Based Reclassification Step (Step 6d)
Implemented in `src/modules/relation_reclassifier.py`. Fully dynamic — reads PROPERTY_CONSTRAINTS
domain/range from relation_validator.py and reverses the validation logic:
- For each term, collects all accepted relations where it appears as subject or filler
- Subject position → accumulates the property's domain metatype constraints
- Filler position → accumulates the property's range metatype constraints
- Intersects all evidence → finds the most specific compatible category
- If intersection is empty → flags as CONTRADICTION for human review
- If current category is less specific than evidence → reclassifies
- After reclassification, checks Parent_Term compatibility and reparents if needed

No LLM. No hardcoded inference rules. Adding new properties or categories to
relation_validator.py automatically updates the reclassifier's behaviour.

## What's Correct
- Continuant Fiat Boundary tree: all correct
- Site tree: all correct
- Geological Structures: all correct EXCEPT Hydrothermal Vent (should be Object) and Structural Block (Object wrongly under Structure)
- Overall: advisor is optimistic — relations already contain the signal for near-perfect classification
