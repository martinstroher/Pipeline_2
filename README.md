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
The canonical approved ontology is archived at `output/final/presalt_ontology.ttl`. New pipeline runs write transient artifacts under `output/`; promote a result to `output/final/` only after review.

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

To rerun only the production validate/export/verify tail from existing Step 6/6b outputs:

```bash
python pipeline.py \
  --validate output/refined/construct_taxonomy.csv \
  --validate-relations output/refined/construct_relations.csv \
  --validate-emit
```
| `--taxonomy CSV` | Build taxonomy from a specific categorized CSV |
| `--owl CSV` | Export OWL from a specific taxonomy CSV |
| `--verify TTL` | Verify an OWL .ttl file (syntax + structure + optional OOPS! pitfalls) |
| `--relations CSV` | Extract relations from a categorized CSV (standalone Step 6b) |
| `--skip-relations` | Skip Step 6b relation extraction in the standard pipeline |
| `--skip-oops` | Skip OOPS! API call during verification (offline mode) |
| `--skip-reasoner` | Skip HermiT reasoner consistency check during verification |

---

## Thesis Evaluation Study

The ablation, statistics, expert workbooks, and rehearsal are isolated from the production pipeline under [`evaluation_study/`](evaluation_study/README.md). They read frozen pipeline artifacts and write only to `evaluation_study/output/`. The study uses separate 100-term Representation and 60-term disagreement-enriched Category samples, separates semantic preservation from lean-core appropriateness, and reports disagreement contrasts, consensus, raw agreement, Fleiss kappa, and Gwet sensitivity measures.

---

## Running Tests

```bash
python test/test_prompt_refactor_parity.py
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
evaluation_study/         # Standalone ablation, statistics, workbooks, rehearsal, tests, and outputs
  README.md               # Study commands, inputs, layout, and safety boundary
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
    csv_io.py             # read_csv / write_csv wrappers (utf-8-sig by default; codifies the BOM-safe contract)
    checkpoint.py         # Resumable I/O: Checkpoint(path, key_column='Term') with load / append / append_batch
    rag_setup.py          # RAG infrastructure: BGE-M3 dense + BM25 sparse + BGE-Reranker RRF
    pdf_processor.py      # PDF→Markdown conversion (pymupdf4llm)
    log.py                # ANSI colour logging helpers
    llm_client.py         # Azure OpenAI (Foundry) wrapper — gpt-5.4, Chat Completions, reasoning_effort
    prompt_loader.py      # Loads active-domain production prompts
    relation_validator.py # BFO domain/range validation for extracted relations (sources constraints from ontology_config.yaml)
  evaluation/
    property_constraints_audit.py  # Writes output/property_constraints_audit.csv (relation provenance + active/inactive)
inputs/                   # Source PDFs (and generated .md files)
output/
  final/                  # Canonical approved ontology + verification receipt
  property_constraints_audit.csv  # Generated by python -m src.evaluation.property_constraints_audit
test/                     # Validation suite (run in this order before any production run)
  test_ontology_config_parity.py  # 26 checks: YAML produces identical literals + waterfall order + categorization_block headers
  regression_t1.py                # Deterministic 6d→7→7b regression vs fixtures/t1_baseline.json
  run_e2e_test.py                 # End-to-end smoke test (Steps 0-7 with real LLM calls; ~$0.10-0.50)
```
