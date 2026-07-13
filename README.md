# PreSaltOntoLearn — Geological Ontology Learning Pipeline

An LLM-driven ontology learning pipeline for Brazilian Pre-Salt petroleum geology. Processes scientific PDFs through an automated pipeline — extraction, aggregation, filtering, RAG-grounded NLD generation, ontology categorization, CQ-driven refinement, taxonomy building, relation extraction, LLM-driven ontology construction/validation, and OWL export — producing a Protege-compatible `.ttl` ontology anchored to BFO, GeoCore, and GeoReservoir upper ontologies.

---

## Quick Start

### Prerequisites
- Python 3.10+
- Azure AI Foundry (Azure OpenAI) resource with a `gpt-5.4` deployment + API key

### Installation
```bash
git clone <repo-url>
cd Pipeline_2
pip install -r requirements.txt
```

### Configuration
```bash
cp .env.example .env
# Edit .env and set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
# (and LLM_GENERATION_MODEL / LLM_EXTRACTION_MODEL = your gpt-5.4 deployment name)
```

**Optional environment variables:**
| Variable | Purpose |
|---|---|
| `JAVA_EXE` | Path to Java executable for HermiT reasoner (default: `java` on PATH) |
| `OOPS_URL` | OOPS! REST API endpoint for pitfall scanning (e.g. `http://localhost:8080/OOPS/rest`) |
| `EXTRACTION_WORKERS` | Number of parallel workers for Step 1 extraction (default: `5`) |
| `MINIMUM_FREQUENCY_FILTER` | Minimum document frequency for Step 3 filtering — number of distinct papers a term must appear in (default: `5`; ~6% of an 82-paper corpus after the gpt-5.4 extraction migration) |
| `ONTOLOGY_CONFIG_PATH` | Path to the ontology YAML (default: `domains/presalt/ontology_config.yaml`) — single source of truth for upper ontologies, relations, and the critic's class budget. Swap to retarget the pipeline to another domain. |
| `STUDY_CONFIG_PATH` | Path to the expert-evaluation workbook config (default: `studies/expert_eval.yaml`) — instructions sheet rendered as Sheet 1 of the expert workbook |
| `RELATION_PROVENANCE_TIERS` | Comma-separated subset of `{owl_axiom, bfo_shape_axiom, ro_release, critic_minted}` controlling which property constraints are active (default: all four) |
| `LATERAL_HINTS_ENABLED` | Overrides `lateral_coherence.hints.enabled` in `ontology_config.yaml` (`true`/`false`). Weak observations are auxiliary context for the taxonomy critic only; they never directly edit the ontology. |
| `LATERAL_CLASS_WORTHINESS_ENABLED` | Enable/disable the existing core-selection critic (`KEEP_PRIMITIVE`, `KEEP_DEFINED`, demote, or exclude). |
| `LATERAL_FRAME_COMPLETION_ENABLED` | Enable/disable corpus-attested missing-frame diagnostics. |
| `LATERAL_FRAME_COMPLETION_AUTO_ADD` | Add attested frame candidates automatically; defaults to `false` so completion remains diagnostic. |
| `LATERAL_RELATION_SCOPE_ENABLED` | Enable/disable the focused generic/context/individual relation-scope critic. |
| `MAX_CONCURRENT_CRITIC` | Maximum concurrent validate-step LLM calls for category work and global reconciliation batches (default: `5`) |
| `CRITIC_TAXONOMY_CHUNK_SIZE` | Terms per chunk in the validate-step Stage-1 taxonomy critic (default: `5`); smaller chunks keep each LLM call focused at the cost of more calls |

### Run
Place PDF files in `inputs/`, then:
```bash
python pipeline.py
```
The full pipeline runs Steps 0-7 and writes a Turtle OWL file (`output/6d_taxonomy_reclassified.ttl`).

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
| 5 | `category_assigner.py` | Step 4 CSV | `output/classify_categories.csv` |
| 5b | `cq_scorer.py` | Step 5 CSV | `output/refined/classify_categories.csv` |
| 6 | `taxonomy_builder.py` | Step 5b CSV | `output/refined/construct_taxonomy.csv` |
| 6b | `relation_extractor.py` | Step 5b CSV | `output/refined/construct_relations.csv` |
| validate | `validate/critic.py` | Steps 6 + 6b CSVs | lean core taxonomy/relations + evidence, realizable-bearer definitions, coherent-frame retention, class-fate/demotion, conflict-safe top-3 BGE NLD reconciliation, single-axis subsumption/facet diagnostics, diagnostic frame completion, disjointness, and summary artifacts under `output/refined/` |
| 7 | `owl_exporter.py` | validate CSVs | `output/refined/emit_ontology.ttl` |
| 7b | `emit/verifier.py` | OWL file | `output/refined/emit_verification.json` |

**Step R (RAG setup)** runs once after Step 0 and provides retrieval context to Steps 4 and 5. ChromaDB is cached to disk (`chroma_db_1024/`) on first run; subsequent runs load from cache. BM25 is always rebuilt in-memory.

---

## CLI Flags

| Flag | Description |
|------|-------------|
| *(none)* | Run the full standard pipeline (Steps 0-7) |
| `--skip-pdf` | Skip PDF→Markdown (Step 0); use existing `.md` files |
| `--skip-extraction` | Skip extraction/aggregation/filter (Steps 1-3); use existing filtered terms |
| `--validate TAXONOMY_CSV` | Run only the validate-step critic from an existing taxonomy CSV |
| `--validate-relations RELATIONS_CSV` | Relations CSV to validate with `--validate` |
| `--validate-emit` | After `--validate`, also export OWL and run verification |
| `--ablation` | Run 4-condition ablation study instead of the standard pipeline |
| `--conditions A,B,C,D` | Select ablation conditions to run (default: all four) |
| `--analysis` | Run Layer 1 automated analysis on ablation output |
| `--expert-eval` | Generate expert evaluation spreadsheet (5-sheet Excel) from ablation output |
| `--layer2-analysis W1.xlsx W2.xlsx` | Run Layer 2 statistical analysis on completed expert workbooks |
| `--layer2-key KEY.csv` | Blinding key CSV (required with `--layer2-analysis`) |

To rerun only the production validate/export/verify tail from existing Step 6/6b outputs:

```bash
python pipeline.py \
  --validate output/refined/construct_taxonomy.csv \
  --validate-relations output/refined/construct_relations.csv \
  --validate-emit
```
| `--taxonomy CSV` | Build taxonomy from a specific categorized CSV (ablation post-processing) |
| `--owl CSV` | Export OWL from a specific taxonomy CSV (ablation post-processing) |
| `--verify TTL` | Verify an OWL .ttl file (syntax + structure + optional OOPS! pitfalls) |
| `--relations CSV` | Extract relations from a categorized CSV (standalone Step 6b) |
| `--skip-relations` | Skip Step 6b relation extraction in the standard pipeline |
| `--skip-oops` | Skip OOPS! API call during verification (offline mode) |
| `--skip-reasoner` | Skip HermiT reasoner consistency check during verification |

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
pipeline.py               # Main orchestrator + CLI (thin: helpers for parser, dispatch, cleanup, stop-check)
domains/                  # Per-domain config + assets. Each subfolder is a complete retargetable bundle.
  README.md               # Author guide: layout, activation, per-prompt runtime-placeholder contract
  presalt/
    ontology_config.yaml  # Single source of truth: waterfall, upper-ontology metadata, BFO disjoint pairs, 71 relation property constraints
    prompts/              # 14 production prompts (including focused validate stages)
    resources/            # Upper-ontology OWL files: bfo-core.owl, geocore-full.owl, geores-full.owl, ro-core.owl
    competency_questions.txt  # CQs evaluated by Step 5b
studies/                  # Cross-domain study artifacts (not Pre-Salt-specific)
  prompts/                # 2 ablation-only prompts (ablation_categorization_{nld,rag}.txt)
  expert_eval.yaml        # Expert-evaluation workbook instructions sheet (49 rows)
src/
  modules/
    extract/
      term_extractor.py     # Step 1: LLM-based term extraction (gpt-5.4 via Azure Foundry)
      term_aggregator.py    # Step 2: Frequency aggregation + spaCy lemmatization
      term_filter.py        # Step 3: Frequency threshold filter
    define/
      nld_generator.py      # Step 4: RAG-grounded NLD generation (gpt-5.4 via Azure Foundry)
    classify/
      category_assigner.py  # Step 5: N-tier waterfall categorization
      cq_scorer.py          # Step 5b: CQ-driven refinement (mandatory): cleanup + scoring + filter at CQ>=1
    construct/
      taxonomy_builder.py   # Step 6: Group-based LLM hierarchy builder, UPPER_IRIS anchoring
      relation_extractor.py # Step 6b: LLM relation extraction + BFO domain/range validation
    validate/
      critic.py             # validate: focused taxonomy, worthiness, dedup, facet/frame, relation correctness/scope stages
    emit/
      owl_exporter.py       # Step 7: rdflib Turtle export, OWL restrictions, upper backbone
      verifier.py           # Step 7b: Post-export verification (syntax, structure, OOPS!, HermiT)
  utils/
    ontology_config.py    # ontology_config.yaml loader: frozen dataclass + lru_cache singleton, env overrides; exposes waterfall_ontologies() + categorization_block()
    study_config.py       # expert_eval.yaml loader: instructions sheet rows for the expert workbook
    csv_io.py             # read_csv / write_csv wrappers (utf-8-sig by default; codifies the BOM-safe contract)
    checkpoint.py         # Resumable I/O: Checkpoint(path, key_column='Term') with load / append / append_batch
    rag_setup.py          # RAG infrastructure: BGE-M3 dense + BM25 sparse + BGE-Reranker RRF
    pdf_processor.py      # PDF→Markdown conversion (pymupdf4llm)
    log.py                # ANSI colour logging helpers
    llm_client.py         # Azure OpenAI (Foundry) wrapper — gpt-5.4, Chat Completions, reasoning_effort
    prompt_loader.py      # Loads prompts verbatim across [<active-domain>/prompts/, studies/prompts/] in priority order
    relation_validator.py # BFO domain/range validation for extracted relations (sources constraints from ontology_config.yaml)
  evaluation/
    ablation_study.py     # 4-condition ablation runner with checkpointing and encoding-safe I/O
    layer1_analysis.py    # Automated analysis: agreement, Cochran's Q, migration, NOT_CLASSIFIED
    expert_eval_generator.py  # 5-sheet Excel generator for expert review (200 terms, stratified tiers; instructions sourced from studies/expert_eval.yaml)
    expert_eval_analyzer.py   # Layer 2 statistics: Wilcoxon (Friedman-gated), ICC, kappa, taxonomy
    property_constraints_audit.py  # Writes output/property_constraints_audit.csv (relation provenance + active/inactive)
inputs/                   # Source PDFs (and generated .md files)
output/                   # Step outputs (1_raw → 6d_taxonomy_reclassified.ttl)
  ablation/               # Ablation condition outputs (cat_A.csv … cat_D.csv)
  property_constraints_audit.csv  # Generated by python -m src.evaluation.property_constraints_audit
test/                     # Validation suite (run in this order before any production run)
  test_ontology_config_parity.py  # 26 checks: YAML produces identical literals + waterfall order + categorization_block headers
  diff_instructions_sheet.py      # 49 workbook rows must stay byte-equal to fixtures/instructions_baseline.json
  regression_t1.py                # Deterministic 6d→7→7b regression vs fixtures/t1_baseline.json
  run_e2e_test.py                 # End-to-end smoke test (Steps 0-7 with real LLM calls; ~$0.10-0.50)
```
