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

## How it works — the 7-step pipeline

```
PDFs  →  [Step 0]  →  Markdown files
                   →  [Step R]  RAG index (ChromaDB + BM25)
                       ↕
          [Step 1]  →  Raw term lists (per paper)
          [Step 2]  →  Deduplicated frequency table
          [Step 3]  →  Filtered term list (≥3 papers)
          [Step 4]  →  Terms + Natural Language Definitions (NLDs)
          [Step 5]  →  Terms + NLDs + ontology category
          [Step 6]  →  Taxonomy (parent-child hierarchy)
          [Step 6b] →  Relations (property axioms from NLDs)
          [Step 7]  →  OWL/Turtle file  (.ttl)
```

| Step | What happens (plain language) |
|------|-------------------------------|
| **0 — Ingest** | PDF papers are converted to plain-text Markdown. |
| **R — RAG Index** | All Markdown documents are chunked (1024-character windows) and indexed in a vector database (ChromaDB) and a keyword index (BM25). This index is queried by later steps to retrieve relevant passages. |
| **1 — Extract** | An LLM reads each paper and outputs a list of geological terms found in that text. |
| **2 — Aggregate** | All term lists are merged, spelling variants are lemmatised (e.g. "dolomites" → "dolomite"), and the number of papers mentioning each term is counted. |
| **3 — Filter** | Terms appearing in fewer than 3 papers are discarded. Appearing in 3 of 80 papers = 3.75% cross-document consensus, a threshold consistent with established terminology extraction methodology (Frantzi et al. C-value; Kageura & Umino). |
| **4 — Define (NLD)** | For each term, the system retrieves the 5 most relevant passages from the corpus and asks the LLM to write a **Natural Language Definition (NLD)** in strict Aristotelian form: *"X is a Y that Z"* — where Y is the proximate genus and Z is the differentiating characteristic. |
| **5 — Classify** | Each term + its NLD is fed to the LLM, which classifies it into the most specific applicable upper ontology namespace using a waterfall: GeoReservoir → GeoCore → BFO. |
| **6 — Taxonomy** | Terms within each namespace are arranged into a parent-child hierarchy by an LLM that uses the genus Y from each NLD to propose intermediate class names. Cycle detection prevents circular hierarchies (A→B→A) by re-parenting cyclic terms to the category root. |
| **6b — Extract Relations** | For each term's NLD, an LLM extracts ontological relations (has_part, derives_from, occurs_in, etc.) from 16 BFO/RO properties. Relations are validated against domain/range constraints and encoded as OWL restrictions in the final ontology. |
| **7 — Export** | The taxonomy is serialised as an OWL/Turtle file with `rdfs:subClassOf` links, `rdfs:label`, `rdfs:comment` (the NLD), and `owl:imports` for the BFO upper ontology. Case-insensitive IRI matching, self-reference guards, and upper→upper triple suppression (never emits triples between two published upper-level IRIs) prevent duplicate/invalid OWL triples. |
| **7b — Verify** | The exported ontology is verified post-hoc: RDFLib syntax parsing, structural analysis (orphan classes, missing labels/comments, self-references, upper-ontology anchoring), and optionally OOPS! pitfall scanning via REST API. Results are saved as a JSON report. |

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

The final file `output/7_ontology.ttl` (OWL Turtle format) contains:

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

¹ Lopes Junior, A.G. (2024). *Automatic Classification of Domain Entities into Top-Level Ontology Concepts Using Natural Language Definitions.* PhD thesis, PPGC/UFRGS.
