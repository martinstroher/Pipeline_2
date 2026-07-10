# PreSaltOntoLearn — Project Overview

> **Audience:** Anyone encountering this project for the first time — supervisors, collaborators, committee members, PhD peers.
> **Reading time:** ~5 minutes.

---

## What problem does this solve?

Building a formal ontology manually is a major bottleneck in scientific knowledge management. For a domain like **Brazilian Pre-Salt petroleum geology**, domain experts would need to:

1. Read hundreds of scientific papers to identify relevant terminology
2. Write precise, machine-readable definitions for each term
3. Decide which of three upper ontology frameworks each term belongs to
4. Arrange all terms into a logical hierarchy (taxonomy)
5. Export the result as an OWL file that ontology tools can load

This process takes months and requires rare combinations of domain expertise and ontology engineering knowledge. **PreSaltOntoLearn automates all five tasks** using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), starting from a folder of PDFs and ending with a Protégé-compatible OWL ontology.

---

## The domain

**Brazilian Pre-Salt carbonate reservoirs** — a set of giant petroleum provinces discovered in the Santos, Campos, and Espírito Santo basins beneath a thick evaporite layer. The reservoirs are predominantly lacustrine carbonates (coquinas, microbial carbonates, spherulites), with complex diagenetic histories that strongly affect porosity and permeability.

The domain is **underrepresented in general LLM training data**, which is why retrieval from the actual literature is critical: terms like *coquina*, *stromatolite*, *vugy porosity*, and *Barra Velha Formation* have precise technical meanings that a general-purpose LLM cannot reliably reproduce from memory alone.

---

## Competency questions — what the ontology must answer

The ontology scope is governed by 10 competency questions (CQs), grounded in the GeoCore/GeoReservoir upper-ontology vocabulary. These define what the ontology commits to representing and guide iterative refinement to remove out-of-scope terms. The full list is maintained in `resources/competency_questions.txt`.

| ID | Question |
|---|---|
| CQ1 | What are the geological objects of the Pre-Salt domain? |
| CQ2 | What earth materials constitute the Pre-Salt geological objects? |
| CQ3 | What geological structures occur in the Pre-Salt geological objects? |
| CQ4 | What post-depositional processes and structures occur in the Pre-Salt geological objects? |
| CQ5 | What are the spatial relations and arrangements in the Pre-Salt domain? |
| CQ6 | What are the dimensions and positions of the Pre-Salt geological objects? |
| CQ7 | What types of boundaries exist and where are they in the Pre-Salt geological objects? |
| CQ8 | What deposits and stratigraphic units are associated with Pre-Salt reservoirs? |
| CQ9 | What are the petrophysical characteristics of Pre-Salt reservoirs? |
| CQ10 | What geological age is associated with the Pre-Salt reservoir intervals? |

---

## How it works — the 7-step pipeline

```
PDFs  →  [Step 0]  →  Markdown files
                   →  [Step R]  RAG index (ChromaDB + BM25)
                       ↕
          [Step 1]  →  Raw term lists (per paper)
          [Step 2]  →  Deduplicated frequency table
          [Step 3]  →  Filtered term list (≥7 papers)
          [Step 4]  →  Terms + Natural Language Definitions (NLDs)
          [Step 5]  →  Terms + NLDs + ontology category
          [Step 6]  →  Taxonomy (parent-child hierarchy)
          [Step 6b] →  Relations (property axioms from NLDs)
          [Step 6c] →  Cleaned taxonomy + relations (LLM quality review)
          [Step 6d] →  Reclassified taxonomy (relation-based BFO metatype correction)
          [Step 7]  →  OWL/Turtle file  (.ttl)
```

| Step | What happens (plain language) |
|------|-------------------------------|
| **0 — Ingest** | PDF papers are converted to plain-text Markdown. |
| **R — RAG Index** | All Markdown documents are chunked (1024-character windows) and indexed in a vector database (ChromaDB) and a keyword index (BM25). This index is queried by later steps to retrieve relevant passages. |
| **1 — Extract** | An LLM reads each paper and outputs a deduplicated list of geological terms found in that text. |
| **2 — Aggregate** | All term lists are merged, spelling variants are lemmatised (e.g. “dolomites” → “dolomite”), and the number of papers mentioning each term is counted. Since extraction deduplicates per paper, Frequency = document frequency. |
| **3 — Filter** | Terms appearing in fewer than N papers are discarded (configurable via `MINIMUM_FREQUENCY_FILTER`, default 5). For the 82-paper corpus (gpt-5.4 extraction), freq≥5 (~6% of papers, 407 terms) was selected after distribution analysis: freq≥7 gave 265 terms (narrower concepts dropped) and freq≥10 gave 157; terms entering at freq 5–6 are legitimate domain concepts, with generic noise only at freq≤3. *(Model-change note, removable: the earlier freq≥7 was set on the Gemini extraction — 614 terms — but gpt-5.4 extracts ~43% as many, so the threshold was re-derived and lowered to keep coverage and expert-eval headroom.)* Downstream steps provide additional quality filtering (NOT_CLASSIFIED removal, cycle detection). |
| **4 — Define (NLD)** | For each term, the system retrieves the 5 most relevant passages from the corpus and asks the LLM to write a **Natural Language Definition (NLD)** in strict Aristotelian form: *"X is a Y that Z"* — where Y is the proximate genus and Z is the differentiating characteristic. |
| **5 — Classify** | Each term + its NLD is fed to the LLM, which classifies it into the most specific applicable upper ontology namespace using a waterfall: GeoReservoir → GeoCore → BFO. The BFO realizables (role, disposition, function) are deliberately kept out of the menu — a term that bears a role is classified as its material bearer, and the realizable is added later by the validate-step critic. |
| **5b — Refine** | Cleans encoding/synonym duplicates, then scores each term against the 10 competency questions (CQs) in parallel. SPECIALIZATION pairs are passed to Step 6 as parent-child hints. Terms that don't contribute to any CQ (CQ_Count < 1) are dropped; the filtered categorized CSV at `output/refined/classify_categories.csv` becomes the input to Step 6. Runs unconditionally as part of the `classify` verb. |
| **6 — Taxonomy** | Terms within each namespace are arranged into a parent-child hierarchy by an LLM that uses the genus Y from each NLD to propose intermediate class names. Cycle detection prevents circular hierarchies (A→B→A) by re-parenting cyclic terms to the category root. |
| **6b — Extract Relations** | For each term's NLD, an LLM extracts ontological relations (has_part, derives_from, occurs_in, etc.) from 16 BFO/RO properties. Relations are validated against domain/range constraints and encoded as OWL restrictions in the final ontology. |
| **validate — Critic** | Three focused LLM calls per category, run in sequence. **Stage 1 (taxonomy, per-term)** runs in small chunks (`CRITIC_TAXONOMY_CHUNK_SIZE` terms, default 5) so each call reasons about only a handful of terms; it sees the chunk's terms + NLDs, those terms' relations and parents' NLDs as read-only context, and the allowed target-class list, emitting one verdict per row (`KEEP / REPARENT / DROP_AS_MIXIN / DROP_AS_REDUNDANT / CONVERT_TO_INSTANCE`) and a per-row probe trace (`probe1_genus_ok`, `probe2_bucket`, `probe3_rewrite`) plus OntoClean signs (`rigidity`, `identity`, `dependence`) from a parent–child edge check that REPARENTs anti-rigid/dependent mis-parentings (identity is recorded but advisory-only). Here `DROP_AS_REDUNDANT` is **parent-collapse only** (a term that vacuously restates its parent). **Stage 2 (dedup)** is one call over all of the category's survivors together — the two decisions that need a global view: sibling near-synonym redundancy and weak intermediates (using a Python-computed `child_count`); it only emits `DROP_AS_REDUNDANT`, each citing the `survivor` it collapses into, and a mutual-drop guard then un-drops any term whose cited survivor was itself dropped (never lose a concept). **Stage 3 (relations)** sees the relation rows, the critic menu (relations flagged `critic_menu: true` in config — 21 for Pre-Salt; mereology offered only as generic `has_part`/`part_of`), previously minted properties, the taxonomy NLDs as context, and a handoff of the merged stage 1–2 decisions; it emits one verdict per relation (`KEEP / DROP / FIX`, where `FIX` may mint a property). After the edits, a deterministic pass re-specialises mereology (continuant vs occurrent parthood) so it stays correct regardless of any FIX. A completeness guard re-asks the model for any input row it forgets in stages 1 and 3, so large categories are not silently under-reviewed. REPARENT is evidence-based (no hard citation floor). CONVERT_TO_INSTANCE rows are recorded in `validate_instances.csv` for emission as `owl:NamedIndividual`; minted properties accumulate in `validate_minted_properties.csv` (provenance `critic_minted`). A full audit log — taxonomy rows carrying the probe trace — is written to `validate_edits.csv`. |
| **7 — Export** | The taxonomy is serialised as an OWL/Turtle file with `rdfs:subClassOf` links, `rdfs:label`, `rdfs:comment` (the NLD), and `owl:imports` for the BFO upper ontology. Critic-minted properties and CONVERT_TO_INSTANCE individuals are emitted from the validate-step outputs, and BFO Quality/Role descendants receive companion `inheres_in some IndependentContinuant` / `realized_in some Process` restrictions. Role-fused bearers from the critic (e.g. "Carbonate Reservoir") are emitted as **defined classes** — `bearer ≡ genus ⊓ (has_role some MintedRole)` via `owl:equivalentClass` — so a term that names a kind playing a role is no longer asserted as a primitive rigid kind (the OntoClean mixin fix). Case-insensitive IRI matching, self-reference guards, and upper→upper triple suppression prevent duplicate/invalid OWL triples. Disjointness conflict detection automatically resolves presalt: classes that inherit from both sides of BFO disjoint pairs, using the term's Category to determine which parent to keep. |
| **7b — Verify** | The exported ontology is verified post-hoc: RDFLib syntax parsing, structural analysis (orphan classes, missing labels/comments, self-references, upper-ontology anchoring), optionally OOPS! pitfall scanning via REST API, and optionally HermiT reasoner consistency checking (requires Java and owlready2). Results are saved as a JSON report. |

---

## Key design choices — why these matter

### Why Natural Language Definitions (NLDs)?

Lopes Junior (2024)¹ proved across 110 domain ontologies and 12 scientific domains that Aristotelian NLDs consistently outperform other textual representations — including term-only, raw definition, and example sentences — for classifying domain entities into upper ontology concepts, achieving >90% macro-F1 on BFO.

The Aristotelian form *"X is a Y that Z"* makes the **genus** (Y) explicit. This matters because:
- It resolves polysemy: "fault" as a geological structure vs. as a human error becomes unambiguous.
- It generates intermediate taxonomy node names automatically: if several terms share genus "diagenetic process", that string becomes a node in the hierarchy.
- It is the form already used in formal ontology definitions (OBO Foundry, GeoCore).

**Example:** `"Dolomitization is a diagenetic process that replaces original calcium carbonate minerals with dolomite through Mg-rich fluid migration."` → genus = *diagenetic process*.

### Why RAG?

The LLM's parametric knowledge of Pre-Salt geology is limited. Retrieving text passages from the actual corpus grounds term definitions in **what the authors wrote**, reducing hallucination and producing definitions that are citable and peer-reviewed in origin.

The system uses **hybrid retrieval**: dense semantic search (BGE-M3 embeddings in ChromaDB) combined with keyword search (BM25), fused with Reciprocal Rank Fusion, then reranked by a cross-encoder. This ensures both semantic and lexical relevance.

### Why three upper ontology levels?

| Namespace | Scope | Example |
|-----------|-------|---------|
| **GeoReservoir** | Petroleum-geology-specific (reservoir properties, fluids, traps) | Porosity Type, Reservoir Seal, Hydrocarbon Column |
| **GeoCore** | General geological science (rocks, structures, processes, environments) | Sedimentary Rock, Geological Structure, Diagenetic Process |
| **BFO** *(Basic Formal Ontology)* | Abstract universals shared across all scientific domains | material entity, process, quality, continuant |

The **waterfall priority** (GeoReservoir first, then GeoCore, then BFO) ensures that each term is classified into the most specific applicable namespace rather than being trivially dumped into BFO. BFO provides interoperability with thousands of OBO Foundry ontologies in biology, medicine, and chemistry.

### Classes vs. individuals

Not every term is a "type of thing" (OWL class). Some terms are **named entities** — specific instances of a class. The pipeline automatically distinguishes:

- `owl:Class` (generic type, e.g., *Carbonate Ramp*, *Normal Fault*) → linked with `rdfs:subClassOf`
- `owl:NamedIndividual` (specific named entity, e.g., *Lula Field*, *Santos Basin*, *Aptian*) → linked with `rdf:type`

---

## Evaluation framework

The pipeline's design choices are validated by an **ablation study** — running the same term set through four variations:

| Condition | What it does | What comparison shows |
|-----------|-------------|-----------------------|
| **A — Full** | RAG retrieval + Aristotelian NLD + classification | baseline |
| **B — NoRAG** | NLD generated from LLM knowledge only (no retrieval) | A vs B → RAG contribution |
| **C — NoNLD** | Classification from the bare term string only | A vs C → NLD contribution |
| **D — RawRAG** | Raw retrieved passages as context (no structured NLD) | A vs D → structuring benefit |

This directly replicates the core comparison from Lopes Junior (2024): the thesis showed NLD > definiendum for upper-ontology classification. We replicate this on a new domain and a new architecture (RAG-augmented LLM instead of a supervised classifier trained on OBO Foundry).

### Two evaluation layers

- **Layer 1 — Automated:** Cross-condition agreement matrices, Cochran's Q significance test, NOT_CLASSIFIED rates, category migration analysis. Runs on all terms with no expert effort.
- **Layer 2 — Expert-in-the-loop:** A blinded 5-sheet Excel workbook is generated for 3 domain experts (geologists) to evaluate 200 terms:
  - **Sheet 2 — Term Relevance** (1-5 Likert, condition-independent)
  - **Sheet 3 — NLD Quality** (blinded A-vs-B comparison, 1-5 + preference)
  - **Sheet 4 — Category Correctness** (stratified by ontology tier: GeoReservoir categories get full binary validation with descriptions; GeoCore/BFO categories get simplified evaluation — see note below)
  - **Sheet 5 — Taxonomy Correctness** (~80 parent-child IS-A pairs: "Is X a type of Y?"). Pairs are stratified by category (~75 % involving selected terms, ~25 % intermediate-node edges for depth coverage), shuffled and blinded.
  - Analysed with Wilcoxon signed-rank (with Friedman omnibus gate for post-hoc), ICC, and Fleiss' kappa.

**Stratified expert evaluation by ontology tier:** Following NeOn methodology and OntoClean best practices, the category evaluation is stratified. Geologists validate GeoReservoir assignments with full confidence (their domain). GeoCore/BFO assignments receive simplified evaluation, and formal ontological alignment is validated separately by the thesis author. This separates domain plausibility (expert task) from formal correctness (engineering task).

The winning ablation condition is then used to build the final taxonomy and OWL export.

---

## What the output looks like

The final file `output/6d_taxonomy_reclassified.ttl` (OWL Turtle format) contains:

```turtle
# Example excerpt
geo:DolomitizationProcess a owl:Class ;
    rdfs:label "Dolomitization" ;
    rdfs:comment "Dolomitization is a diagenetic process that replaces original calcium
                  carbonate minerals with dolomite through the substitution of calcium
                  ions by magnesium ions from Mg-rich fluids." ;
    rdfs:subClassOf geo:DiageneticProcess .

geo:LulaField a owl:NamedIndividual, geo:OilField ;
    rdfs:label "Lula Field" .
```

The file can be opened in **Protégé** for inspection, visualisation, and reasoning.

---

## Related files in this repository

| File | Purpose |
|------|---------|
| `README.md` | Technical quick-start and CLI reference |
| `docs/CONTEXT.md` | Full architecture reference with Mermaid diagram and per-module descriptions |
| `docs/OVERVIEW.md` | This file — plain-language project summary |
| `pipeline.py` | Main entry point (`python pipeline.py --help`) |

---

## Retargeting to another scientific domain

The pipeline architecture is domain-agnostic. The Pre-Salt-specific knowledge lives under `domains/presalt/`; copy the folder, rewrite its contents, and the same 7-step pipeline runs on biomedicine, materials science, palaeoclimate, etc.

| File / folder | Holds |
|------|-------|
| `domains/<name>/ontology_config.yaml` | Upper ontologies and their classes (BFO, GeoCore, GeoReservoir for Pre-Salt → e.g. BFO + ChEBI + OBI for biomedicine), the categorization waterfall, 71 relation property constraints with provenance, BFO disjoint pairs, Step 6d behaviour |
| `domains/<name>/prompts/` | 10 production prompts (term extraction, NLD generation, categorization, taxonomy, critic, relations, CQ scoring …) authored end-to-end with personas inlined and calibration examples specific to the domain |
| `domains/<name>/resources/` | Reference OWL files for the upper ontologies (loaded by `owl_exporter.py` for the upper backbone) |
| `domains/<name>/competency_questions.txt` | CQs used by Step 5b (mandatory) |
| `studies/expert_eval.yaml` | Cross-domain workbook prose: 49 instruction-sheet rows. Edit only if your evaluation Likert anchors or calibration examples differ. |

No Python code needs to change to retarget. Point the loader at the new YAML via `ONTOLOGY_CONFIG_PATH=domains/<name>/ontology_config.yaml`; the prompt loader picks up `domains/<name>/prompts/` from the same parent folder automatically. The expert workbook generator (`src/evaluation/expert_eval_generator.py`) reads `instructions_sheet.rows` from `studies/expert_eval.yaml` and renders Sheet 1 from it verbatim.

For a new domain you will also need to:
- Provide your own scientific PDFs in `inputs/`
- Author or curate competency questions in `domains/<name>/competency_questions.txt`
- Provide reference OWL files for your upper ontologies in `domains/<name>/resources/`

See [domains/README.md](../domains/README.md) for the per-prompt runtime-placeholder contract and a step-by-step retargeting walkthrough.

This separation between **what the pipeline does** (Python code) and **what the domain is** (two YAML files) is the architectural contribution that distinguishes PreSaltOntoLearn from one-off ontology learning experiments. The same code that produced the Pre-Salt ontology should produce a working ontology for any other scientific domain with the same documentary inputs.

---

¹ Lopes Junior, A.G. (2024). *Automatic Classification of Domain Entities into Top-Level Ontology Concepts Using Natural Language Definitions.* PhD thesis, PPGC/UFRGS.
