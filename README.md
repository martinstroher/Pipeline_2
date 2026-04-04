# PreSaltOntoLearn — Geological Ontology Learning Pipeline

An LLM-driven ontology learning pipeline for Brazilian Pre-Salt petroleum geology. Processes scientific PDFs through a 7-step pipeline — extraction, aggregation, filtering, RAG-grounded NLD generation, ontology categorization, taxonomy building, and OWL export — producing a Protege-compatible `.ttl` ontology anchored to BFO, GeoCore, and GeoReservoir upper ontologies.

---

## Quick Start

### Prerequisites
- Python 3.10+
- Google Gemini API key (AI Studio) **or** Vertex AI credentials

### Installation
```bash
git clone <repo-url>
cd Pipeline_2
pip install -r requirements.txt
```

### Configuration
```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY (or VERTEX_AI=true + GCP_PROJECT + GCP_LOCATION)
```

### Run
Place PDF files in `inputs/`, then:
```bash
python pipeline.py
```
The full pipeline runs Steps 0-7 and writes a Turtle OWL file (`output/7_ontology.ttl`).

---

## Pipeline Steps

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 0 | `pdf_processor.py` | `inputs/*.pdf` | `inputs/*.md` |
| R | `rag_setup.py` | `inputs/*.md` | ChromaDB index + BM25 (in-memory) |
| 1 | `term_extractor.py` | `inputs/*.md` | `output/1_raw_llm_extraction.json` |
| 2 | `term_aggregator.py` | Step 1 JSON | `output/2_aggregated_counts.csv` |
| 3 | `term_filter.py` | Step 2 CSV | `output/3_filtered_top_terms.csv` |
| 4 | `nld_generator.py` | Step 3 + RAG | `output/4_nld_generated_definitions.csv` |
| 5 | `term_categorizer.py` | Step 4 CSV | `output/5_categorized_ontology.csv` |
| 6 | `taxonomy_builder.py` | Step 5 CSV | `output/6_taxonomy.csv` |
| 7 | `owl_exporter.py` | Step 6 CSV | `output/7_ontology.ttl` |

**Step R (RAG setup)** runs once after Step 0 and provides retrieval context to Steps 4 and 5. ChromaDB is cached to disk (`chroma_db_1024/`) on first run; subsequent runs load from cache. BM25 is always rebuilt in-memory.

---

## CLI Flags

| Flag | Description |
|------|-------------|
| *(none)* | Run the full standard pipeline (Steps 0-7) |
| `--skip-pdf` | Skip PDF→Markdown (Step 0); use existing `.md` files |
| `--skip-extraction` | Skip extraction/aggregation/filter (Steps 1-3); use existing filtered terms |
| `--ablation` | Run 4-condition ablation study instead of the standard pipeline |
| `--conditions A,B,C,D` | Select ablation conditions to run (default: all four) |
| `--analysis` | Run Layer 1 automated analysis on ablation output |
| `--expert-eval` | Generate expert evaluation spreadsheet (5-sheet Excel) from ablation output |
| `--layer2-analysis W1.xlsx W2.xlsx` | Run Layer 2 statistical analysis on completed expert workbooks |
| `--layer2-key KEY.csv` | Blinding key CSV (required with `--layer2-analysis`) |
| `--taxonomy CSV` | Build taxonomy from a specific categorized CSV (ablation post-processing) |
| `--owl CSV` | Export OWL from a specific taxonomy CSV (ablation post-processing) |
| `--verify TTL` | Verify an OWL .ttl file (syntax + structure + optional OOPS! pitfalls) |
| `--skip-oops` | Skip OOPS! API call during verification (offline mode) |

---

## Ablation Workflow

The thesis deliverable uses an ablation study to justify the RAG+NLD design choices:

```
# 1. Run all 4 conditions
python pipeline.py --ablation

# 2. Run Layer 1 automated analysis
python pipeline.py --analysis

# 3. Generate expert evaluation spreadsheet
python pipeline.py --expert-eval

# 4. After expert review, run Layer 2 statistical analysis
python pipeline.py --layer2-analysis expert1.xlsx expert2.xlsx --layer2-key key.csv

# 5. Build taxonomy + OWL on the winning condition
python pipeline.py --taxonomy output/ablation/cat_A.csv
python pipeline.py --owl output/ablation/6_taxonomy_A.csv
```

**Ablation conditions and what each comparison tests:**
- **A** (Full): RAG context + NLD-informed categorization ← baseline (the full system)
- **B** (NoRAG): NLD generated from LLM knowledge only, no retrieval ← **A vs B isolates RAG contribution**
- **C** (NoNLD): Categorization from term string alone, no NLD ← **A vs C isolates NLD contribution**
- **D** (RawRAG): Raw retrieved chunks passed directly, no structured NLD ← **A vs D isolates NLD structuring**

---

## Running Tests

```bash
python test/run_e2e_test.py
```

Self-contained test that:
1. Uses a pre-written Markdown document (no PDF conversion needed)
2. Runs Steps 0-7 in an isolated `test/output_test/` directory
3. Validates extraction counts, NLD coverage, categorization, taxonomy columns, and OWL parse

---

## Project Structure

```
pipeline.py               # Main orchestrator + CLI
src/
  modules/
    term_extractor.py     # Step 1: LLM-based term extraction (Gemini 2.5 Pro)
    term_aggregator.py    # Step 2: Frequency aggregation + spaCy lemmatization
    term_filter.py        # Step 3: Frequency threshold filter
    nld_generator.py      # Step 4: RAG-grounded NLD generation (Gemini 2.5 Pro)
    term_categorizer.py   # Step 5: Waterfall categorization (GeoReservoir→GeoCore→BFO)
    taxonomy_builder.py   # Step 6: Group-based LLM hierarchy builder, UPPER_IRIS anchoring
    owl_exporter.py       # Step 7: rdflib Turtle export, Protege-compatible
    ontology_verifier.py  # Step 7b: Post-export verification (syntax, structure, OOPS!)
  utils/
    rag_setup.py          # RAG infrastructure: BGE-M3 dense + BM25 sparse + BGE-Reranker RRF
    pdf_processor.py      # PDF→Markdown conversion (pymupdf4llm)
    log.py                # ANSI colour logging helpers
    gemini_client.py      # Gemini API wrapper (AI Studio + Vertex AI express mode)
  evaluation/
    ablation_study.py     # 4-condition ablation runner with checkpointing and encoding-safe I/O
    layer1_analysis.py    # Automated analysis: agreement, Cochran's Q, migration, NOT_CLASSIFIED
    expert_eval_generator.py  # 5-sheet Excel generator for expert review (200 terms, stratified tiers)
    expert_eval_analyzer.py   # Layer 2 statistics: Wilcoxon (Friedman-gated), ICC, kappa, taxonomy
inputs/                   # Source PDFs (and generated .md files)
output/                   # Step outputs (1_raw → 7_ontology.ttl)
  ablation/               # Ablation condition outputs (cat_A.csv … cat_D.csv)
resources/                # Upper ontology definition text files (BFO, GeoCore, GeoReservoir)
test/                     # E2E test runner + isolated config
```
