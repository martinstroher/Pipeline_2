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
| `ABLATION_OUTPUT_DIR` | Directory for condition artifacts, manifests, workbooks, and analyses (default: `output/ablation`) |
| `ABLATION_FROZEN_A_NLD` | Frozen production Condition-A NLD artifact (default: `CONSOLIDATED_LLM_RESULTS_WITH_NLDS`, normally `output/define_nld.csv`) |
| `ABLATION_FROZEN_A_CATEGORY` | Frozen production Condition-A categorization (default: `CATEGORIZED_LLM_TERMS`, normally `output/classify_categories.csv`) |
| `ABLATION_EXPECTED_TERM_COUNT` | Required paired term count for the study (default: `407`) |
| `EXPERT_ONTOLOGY_DIR` | Approved validate-artifact directory used by final-ontology modules (default: `output/refined`) |
| `EXPERT_BOOTSTRAP_ITERATIONS` | Clustered bootstrap replicates for Layer 2 confidence intervals (default: `5000`) |
| `EXPERT_STRICT_APPROVED_POPULATIONS` | Require the approved final-artifact population counts before workbook generation (default: `true`) |
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
| `--expert-eval` | Generate three independently shuffled 8-sheet expert workbooks plus a separate blinding key |
| `--layer2-analysis W1.xlsx W2.xlsx W3.xlsx` | Run item-aggregated Layer 2 statistics on completed expert workbooks |
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

The thesis deliverable uses a controlled 407-term ablation study to measure how representation choices change upper-ontology assignments. These automated comparisons measure **sensitivity and agreement, not accuracy**; Condition A is an experimental anchor, not a gold standard.

```
# 1. Run all 4 conditions. A is copied from frozen production artifacts;
#    B, C, and D run sequentially with three workers inside each condition.
#    A complete run also launches Layer 1 and workbook generation.
python pipeline.py --ablation

# 2. Run Layer 1 automated analysis
python pipeline.py --analysis

# 3. Regenerate the three expert workbooks without rerunning paid conditions
python pipeline.py --expert-eval

# 4. After expert review, run Layer 2 statistical analysis
python pipeline.py \
  --layer2-analysis output/ablation/expert_workbooks/expert_evaluation_1.xlsx output/ablation/expert_workbooks/expert_evaluation_2.xlsx output/ablation/expert_workbooks/expert_evaluation_3.xlsx \
  --layer2-key output/ablation/private/blinding_key_42.csv
```

**Ablation conditions and what each comparison tests:**
- **A** (Full): frozen production RAG context + NLD-informed categorization; copied byte-for-byte, never regenerated
- **B** (NoRAG): NLD generated from LLM knowledge only, no retrieval ← **A vs B isolates RAG contribution**
- **C** (NoNLD): Categorization from term string alone, no NLD ← **A vs C isolates NLD contribution**
- **D** (RawRAG): the exact five stored Condition-A chunks passed directly, with no new retrieval or structured NLD ← **A vs D isolates NLD structuring**

A/B/C use the production categorization prompt. D uses the equivalent prompt with only the representation field changed from `nld` to `context`. The runner requires high reasoning effort and seed 42, rejects incomplete/error/invalid-category outputs, and records prompt, config, term-set, source, and artifact SHA-256 hashes in `experiment_manifest.json`. Partial `--conditions` runs stop before Layer 1 and workbook generation.

Layer 1 reports exact and ontology-tier agreement, Cohen's kappa, independent RAG/NLD/structuring sensitivity flags, category/tier confusion matrices, global Cochran's Q over A-anchored agreement, gated Holm-corrected McNemar tests, Holm-corrected Stuart-Maxwell tier tests, and descriptive `NOT_CLASSIFIED`/`Context_Used` summaries.

Layer 2 samples 100 terms proportionally by Condition-A tier and corpus-frequency band. Each workbook contains `Instructions`, `Representation`, `Category_Correct`, `Taxonomy`, `Defined_Classes`, `Relations`, `Individuals`, and `Critic_Decisions`. The final modules sample 40/185 taxonomy links, all 13 definitions, 25/125 general relations, 15/58 named entities, and 40/116 exclusion/demotion decisions. Inference averages experts per sampled item first; final task families are never collapsed into one score.

Only files under `output/ablation/expert_workbooks/` are distributable. The unblinding key is written separately under `output/ablation/private/`; never send that directory to experts.
Layer 2 refuses to run if any ablation, ontology, prompt/config, or sampling source no longer matches `expert_evaluation_manifest.json`.

### Offline process rehearsal

Before any paid run, exercise the complete workflow without Azure or human ratings:

```bash
python -m src.evaluation.offline_rehearsal --overwrite
```

This writes explicitly synthetic artifacts to `output/ablation_rehearsal/`, including mock-filled workbooks, both analysis layers, a hash manifest, and `OFFLINE_REHEARSAL_FINDINGS.md`. The B proxy is derived from frozen A and the category/mock ratings are deterministic, so these outputs validate mechanics only and must never be reported as study evidence.

---

## Running Tests

```bash
python test/test_evaluation_study.py
python test/test_offline_rehearsal.py
python test/test_prompt_refactor_parity.py
python test/diff_instructions_sheet.py
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
  prompts/                # D-only raw-context categorization prompt; A/B/C use the production prompt
  expert_eval.yaml        # Modular expert-evaluation workbook instructions (66 rows)
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
    ablation_study.py     # Frozen-A, manifest-backed 4-condition runner; sequential conditions, 3 workers within each
    layer1_analysis.py    # Automated sensitivity/agreement analysis with paired-matrix validation
    expert_eval_generator.py  # Public generator entry point
    expert_eval_workbook.py   # 100-term stratified sample + 8-sheet, 3-expert workbook engine
    expert_eval_analyzer.py   # Public Layer 2 analyzer entry point
    expert_eval_analysis.py   # Item-aggregated inference, bootstrap CIs, agreement, separate final-task results
    offline_rehearsal.py      # Zero-Azure synthetic A/B/C/D + mock workbook + Layer 1/2 preflight
    property_constraints_audit.py  # Writes output/property_constraints_audit.csv (relation provenance + active/inactive)
inputs/                   # Source PDFs (and generated .md files)
output/                   # Step outputs (1_raw → 6d_taxonomy_reclassified.ttl)
  ablation/               # Conditions, manifests, Layer 1 outputs, workbooks, blinding key, Layer 2 outputs
  property_constraints_audit.csv  # Generated by python -m src.evaluation.property_constraints_audit
test/                     # Validation suite (run in this order before any production run)
  test_ontology_config_parity.py  # 26 checks: YAML produces identical literals + waterfall order + categorization_block headers
  test_evaluation_study.py        # Ablation, Layer 1, sampling/fate, item-level inference, relation-status regressions
  test_offline_rehearsal.py       # Full deterministic zero-Azure workflow and workbook/key isolation regression
  diff_instructions_sheet.py      # 66 workbook rows must stay byte-equal to fixtures/instructions_baseline.json
  regression_t1.py                # Deterministic 6d→7→7b regression vs fixtures/t1_baseline.json
  run_e2e_test.py                 # End-to-end smoke test (Steps 0-7 with real LLM calls; ~$0.10-0.50)
```
