# PreSaltOntoLearn — Geological Ontology Learning Pipeline

An LLM-driven ontology learning pipeline for Brazilian Pre-Salt petroleum geology. Processes scientific PDFs through an automated pipeline — extraction, aggregation, filtering, RAG-grounded NLD generation, ontology categorization, CQ-driven refinement, taxonomy building, relation extraction, LLM-driven ontology construction/validation, and OWL export — producing a Protege-compatible `.ttl` ontology anchored to BFO, GeoCore, and GeoReservoir upper ontologies.

---

## Research release: GeoPreSalt 0.1

This maintenance work retains the **frozen, evaluated GeoPreSalt 0.1** as a
research artifact. It does not repair or claim logical coherence of that
ontology. Reasoning with the saved BFO, GeoCore, and GeoReservoir files found
13 unsatisfiable named classes; a complete RO-inclusive result is not
established. These limitations are accepted for the research release and are
documented in [the release notes](docs/geopresalt_0_1_release_notes.md).

The committed offline regression checks every frozen-input hash and reproduces
the same 1,819-triple RDF graph without rewriting the approved Turtle file:

```bash
python test/test_shared_relation_regeneration.py -v
```

Graph equality is not byte equality: RDF blank-node identifiers may differ
between regenerated serializations. The original artifact's SHA-256 remains
unchanged. No GeoPreSalt 0.2 repair or guided human-correction workflow is
included in this maintenance change.

## Quick Start

### Prerequisites
- Target environment: CPython 3.12.14 on macOS Apple Silicon or Linux x86-64 (Ubuntu 24.04 baseline)
- Python with `venv` support to bootstrap the pinned `uv` installer
- Azure AI Foundry (Azure OpenAI) resource with a `gpt-5.4` deployment + API key

### Installation
```bash
git clone https://github.com/martinstroher/Pipeline_2.git
cd Pipeline_2
python3 -m venv .venv-tools
.venv-tools/bin/python -m pip install uv==0.12.12
.venv-tools/bin/uv sync --locked
source .venv/bin/activate
```

`uv` creates the project-local `.venv` and downloads the requested Python
version if it is not installed. `.python-version` pins Python; `uv.lock` pins
Python packages and their source hashes for both target platforms.
`pyproject.toml` is the dependency source of truth, not a publishable Python
package or an ontology release declaration. Windows, Intel Macs, ARM Linux,
and GPU/CUDA environments are outside this initial profile. Linux uses the
CPU build of PyTorch; the pipeline's embedding model already runs on CPU.

The spaCy `en_core_web_sm` model is included at version 3.8.0; do not run an
unpinned model download after installation. BGE-M3 and BGE-Reranker weights
are still downloaded on first use, and their upstream revisions are not
pinned by this environment lock. Record those snapshots and the Azure
deployment version for a study run. This environment is not a reconstruction
of the historical GeoPreSalt 0.1 environment.

Local PDF/retrieval initialization disables ONNX Runtime diagnostic telemetry
before creating inference sessions. This avoids its background telemetry worker,
which caused a macOS shutdown crash during validation; it does not change model
selection or inference outputs.

**Optional groups** (run from the repository root):
```bash
.venv-tools/bin/uv sync --locked --group test
# Include both tests and the Python reasoner bridge:
.venv-tools/bin/uv sync --locked --group test --group reasoner
```
The `test` group adds pytest and mock HTTP transport support; `reasoner` adds
`owlready2`. Java and OOPS! remain separate services/tools. Sync is exact:
include every optional group you want to retain. The original evaluation and
statistics dependencies remain in the base environment.

For an existing **Python 3.12.14** virtual environment, generated pip
requirements are also available:
```bash
python -m pip install --require-hashes -r requirements.txt
python -m pip install --require-hashes -r requirements.txt -r requirements-test.txt -r requirements-reasoner.txt
```
Prefer `uv sync --locked`: pip exports cannot preserve per-package index
selection or uv's build constraints. Hashes restrict pip to the locked source
artifacts, but building `owlready2` from source can still need a compiler and
build tooling. Use a clean environment, not a global installation.

The local offline checks pass on macOS Apple Silicon. The Linux installation
plan resolves for Ubuntu 24.04 x86-64, but execution on Linux and Java/HermiT
checks remain to be run; a dependency plan is not a runtime test.

### Configuration
```bash
cp .env.example .env
# Edit .env and set AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
# (and LLM_GENERATION_MODEL / LLM_EXTRACTION_MODEL = your gpt-5.4 deployment name)
```

Do not overwrite an existing `.env`. Keep the input/output settings from the
example: several modules require them even before a pipeline step starts.
The endpoint is the resource host, without `/openai/v1/`; the client appends
that suffix. Gemini/Vertex credentials are not used by the current client.
The example selects `gpt-5.4` for both model variables; **there is no model
fallback**. Before loading pipeline modules, converting PDFs, building
retrieval indexes, or running `--fresh` cleanup, startup rejects missing, empty,
or whitespace-only model settings with an error naming every missing variable.
The full pipeline requires both names. Extraction-only runs require
`LLM_EXTRACTION_MODEL`; runs that skip extraction and the standalone taxonomy,
relation, and validation commands require `LLM_GENERATION_MODEL`.
Help, standalone OWL export/verification, and runs stopping before extraction
do not require model settings. The check is local: it does not confirm that
the named deployments exist in Azure.

The shared client defaults to reasoning effort `high`, a 32,000-token
completion ceiling, and seed `42`. Temperature settings are ignored by this
client; repeatability of live model calls is best-effort, not exact.

**Optional environment variables:**
| Variable | Purpose |
|---|---|
| `JAVA_EXE` | Path to Java executable for HermiT reasoner (default: `java` on PATH) |
| `OOPS_URL` | OOPS! REST API endpoint. Unset/empty disables scanning; `.env.example` selects the public service. The full ontology is sent to the configured endpoint. |
| `CHROMA_DB_DIR` | Retrieval-cache prefix (default: `chroma_db`; the chunk size is appended). Use a separate prefix for each corpus. |
| `LLM_USAGE_LOG` | Token-usage CSV (default: `output/usage_log.csv`). Override when isolating a run. |
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

### Optional ontology verification

HermiT needs the optional `reasoner` dependency group and a working Java
runtime. Check `java -version`; if Java is not on PATH, set `JAVA_EXE` to its
executable. The Python package does not install Java. No Java version has yet
been exercised against this locked environment; a completed reasoning check
is still required before relying on that layer.

OOPS! can run locally with Docker:
```bash
docker run --rm -p 127.0.0.1:8080:8080 mpovedavillalon/oops:v1
```
Leave the service running and set `OOPS_URL=http://localhost:8080/OOPS/rest`
in `.env`, replacing the public URL. Use `--skip-oops` to avoid sending the
ontology to any OOPS! endpoint, and `--skip-reasoner` to omit HermiT.
These flags do not disable LLM calls in other pipeline steps. Missing or
failed optional checks must be read from the verification report: an overall
`PASS` can coexist with skipped layers and is not proof of domain correctness.

### Run
Place PDF files in `inputs/`, then:
```bash
python pipeline.py
```
The standard pipeline makes live, billable LLM calls. With `.env.example`
paths, its final export is `output/refined/validate_taxonomy.ttl`, alongside
`emit_verification.json`. The export name follows the input taxonomy name.
The frozen evaluated GeoPreSalt 0.1 is archived at
`output/final/presalt_ontology.ttl`; preserve it and the frozen inputs under
`evaluation_study/inputs/`. Keep new runs in separate directories.
See [SETUP.md](SETUP.md) for isolated smoke-run paths and checked new-domain
creation. New starters still require subject-specific editing and review.

### Create a domain

```bash
python scripts/new_domain.py your_domain
# Edit the generated ontology_config.yaml and prompt_blocks.yaml.
python -m src.utils.domain_validation domains/your_domain
```

Generation checks a staged folder before reporting success. The starter has
all text blocks for the current production prompts, current question-scoring
examples, BFO categories, and 61 shared BFO/RO relation entries. Questions live
in the prompt blocks; obsolete filter/question files are not copied.
The property menu derives from the active configuration, and export/verification
do not require geology ontologies. These are structural checks, not a live
end-to-end run or approval of the new domain's meaning.

---

## Pipeline Steps

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 0 | `pdf_processor.py` | `inputs/*.pdf` | `inputs/*.md` |
| R | `rag_setup.py` | `inputs/*.md` | ChromaDB index + BM25 (in-memory) |
| 1 | `term_extractor.py` | `inputs/*.md` | `output/extract_raw.json` (CSV content, legacy extension) |
| 2 | `term_aggregator.py` | Step 1 extraction CSV | `output/extract_aggregated.csv` |
| 3 | `term_filter.py` | Step 2 CSV | `output/extract_filtered.csv` |
| 4 | `nld_generator.py` | Step 3 + RAG | `output/define_nld.csv` |
| 5 | `category_assigner.py` | Step 4 CSV | `output/classify_categories.csv` |
| 5b | `cq_scorer.py` | Step 5 CSV | `output/refined/classify_categories.csv` |
| 6 | `taxonomy_builder.py` | Step 5b CSV | `output/refined/construct_taxonomy.csv` |
| 6b | `relation_extractor.py` | Step 5b CSV | `output/refined/construct_relations.csv` |
| validate | `validate/critic.py` | Steps 6 + 6b CSVs | lean core taxonomy/relations + evidence, realizable-bearer definitions, coherent-frame retention, class-fate/demotion, conflict-safe top-3 BGE NLD reconciliation, single-axis subsumption/facet diagnostics, diagnostic frame completion, disjointness, and summary artifacts under `output/refined/` |
| 7 | `owl_exporter.py` | validate CSVs | `output/refined/validate_taxonomy.ttl` |
| 7b | `emit/verifier.py` | OWL file | `output/refined/emit_verification.json` |

**Step R (RAG setup)** runs once after Step 0 and provides retrieval context to Steps 4 and 5. ChromaDB is cached to disk (`chroma_db_1024/`) on first run; subsequent runs load from cache. BM25 is always rebuilt in-memory.

---

## CLI Flags

| Flag | Description |
|------|-------------|
| *(none)* | Run the full standard pipeline (Steps 0-7) |
| `--skip-pdf` | Skip PDF→Markdown (Step 0); use existing `.md` files |
| `--skip-extraction` | Skip extraction/aggregation/filter (Steps 1-3); use existing filtered terms |
| `--stop-after PHASE` | Stop after `extract`, `define`, `classify`, `construct`, `validate`, or `emit`. `classify` includes mandatory CQ refinement. Earlier steps still run unless explicitly skipped. |
| `--fresh` | Destructive cleanup of selected output files, root `chroma_db_*` directories, and Markdown inputs in `DOCS_DIR`. Not an isolated checkpoint reset; avoid for preserved or shared data. |
| `--validate TAXONOMY_CSV` | Run only the validate-step critic from an existing taxonomy CSV |
| `--validate-relations RELATIONS_CSV` | Relations CSV to validate with `--validate` |
| `--validate-emit` | After `--validate`, also export OWL and run verification |
| `--taxonomy CSV` | Build taxonomy from a specific categorized CSV |
| `--owl CSV` | Export OWL from a specific taxonomy CSV |
| `--verify TTL` | Verify an OWL .ttl file (syntax + structure + optional OOPS! pitfalls) |
| `--relations CSV` | Extract relations from a categorized CSV (standalone Step 6b) |
| `--skip-relations` | Skip Step 6b relation extraction in the standard pipeline |
| `--skip-oops` | Skip OOPS! API call during verification (offline mode) |
| `--skip-reasoner` | Skip HermiT reasoner consistency check during verification |

To rerun only the production validate/export/verify tail from existing Step 6/6b outputs:

```bash
python pipeline.py \
  --validate output/refined/construct_taxonomy.csv \
  --validate-relations output/refined/construct_relations.csv \
  --validate-emit
```

This invokes the LLM critic and writes beside the supplied taxonomy; use
copies in a new run directory when preserving earlier results.

---

## Thesis Evaluation Study

The ablation, statistics, expert workbooks, and rehearsal are isolated from the production pipeline under [`evaluation_study/`](evaluation_study/README.md). They read frozen pipeline artifacts and write only to `evaluation_study/output/`.

---

## Running Tests

Local checks that do not make LLM calls:
```bash
.venv-tools/bin/uv sync --locked --group test
source .venv/bin/activate
python -m pytest -q test/test_new_domain.py test/test_model_configuration.py test/test_runtime_environment.py test/test_lateral_coherence_helpers.py
python -m unittest discover -s test -p 'test_shared_relation*.py' -v
python test/test_prompt_loader_blocks.py
python test/test_ontology_config_parity.py
python test/test_prompt_refactor_parity.py
```

These cover specific contracts, not full release readiness. Runtime smoke
tests deny network connections, use synthetic documents and temporary outputs,
and replace retrieval models with local test doubles. They check imports,
the pinned spaCy model, the Azure request shape through a mock transport, PDF
conversion, retrieval-cache round trips, repeatable graph export, and offline
verification. They do not test real BGE weights, Azure calls, Java, or OOPS!.
Model-configuration tests also check startup failures before dispatch/cleanup,
direct stage calls, reachable-step requirements, and credential-free CLI help
and offline verification.
The separate prompt checks cover block substitution and seven saved snapshots.
Legacy tests under `test/validate/` reference the removed `src.validate` engine;
do not interpret this focused command as a passing full-suite run.

The end-to-end smoke test is **live and billable**, not an offline unit test:
```bash
python test/run_e2e_test.py
```

It requires Azure credentials and `test/.env-test`, generates a Markdown
document, clears earlier `test/output_test/` results and test retrieval caches,
then runs the pipeline and checks its outputs. Review its configuration before
running; it is not a read-only check.

### Updating dependencies

Edit `pyproject.toml`, not the generated requirements. Then deliberately
refresh the lock and all three pip exports with the pinned installer:
```bash
.venv-tools/bin/uv lock
.venv-tools/bin/uv export --locked --no-default-groups --no-emit-project --emit-index-url --output-file requirements.txt
.venv-tools/bin/uv export --locked --only-group test --no-emit-project --output-file requirements-test.txt
.venv-tools/bin/uv export --locked --only-group reasoner --no-emit-project --output-file requirements-reasoner.txt
```
`uv lock` retains existing compatible pins; use `--upgrade-package NAME` only
for an intended upgrade. Re-run the local checks and compare any affected
outputs before accepting the new lock. Do not run live-model smoke tests or
overwrite frozen outputs as part of dependency maintenance.

---

## Project Structure

```
pipeline.py               # Main orchestrator + CLI (thin: helpers for parser, dispatch, cleanup, stop-check)
docs/geopresalt_0_1_release_notes.md # Research-release scope, known logical failures and preservation proof
pyproject.toml            # Runtime dependencies + optional test/reasoner groups; macOS/Linux targets
uv.lock                   # Resolved versions, platform markers, and package hashes
.python-version           # Supported Python patch version
requirements.txt          # Generated, hash-pinned base-environment export for pip
requirements-test.txt     # Generated optional test dependencies
requirements-reasoner.txt # Generated optional Python reasoner dependency (Java is separate)
domains/                  # Pre-Salt bundle + checked new-domain scaffold
  _shared/
    bfo_ro_relations.yaml # 61 generic entries shared by Pre-Salt and new domains
  _template/             # Two editable YAML files + README; neutral examples
  README.md               # Author guide: layout, activation, per-prompt runtime-placeholder contract
  presalt/
    ontology_config.yaml  # Upper-ontology metadata + 10 geology relations; inherits 61 generic entries
    prompts/              # 14 production prompts (including focused validate stages)
    resources/            # Upper-ontology OWL files: bfo-core.owl, geocore-full.owl, geores-full.owl, ro-core.owl
    competency_questions.txt  # Human-readable CQ list; keep aligned with prompt_blocks.yaml used by Step 5b
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
    domain_validation.py  # Offline files/blocks/runtime-fields/example-schema checks for domains
    onnx_runtime.py       # Disables native inference telemetry before PDF/retrieval initialization
    relation_validator.py # BFO domain/range validation for extracted relations (sources constraints from ontology_config.yaml)
  evaluation/
    property_constraints_audit.py  # Writes output/property_constraints_audit.csv (relation provenance + active/inactive)
inputs/                   # Source PDFs (and generated .md files)
output/
  final/                  # Canonical approved ontology + verification receipt
  property_constraints_audit.csv  # Generated by python -m src.evaluation.property_constraints_audit
test/                     # Local contract checks + separate live smoke test
  test_ontology_config_parity.py  # Configuration/consumer parity and waterfall checks
  test_prompt_loader_blocks.py   # Offline prompt-block substitution checks
  test_prompt_refactor_parity.py  # Resolved prompts compared with saved snapshots
  test_runtime_environment.py    # Network-denying dependency smoke tests with temporary outputs
  test_model_configuration.py    # Required deployment settings and fail-fast CLI/stage checks
  test_new_domain.py             # Temporary scaffolds, failure cleanup, relations, export and preservation
  test_shared_relation_config.py # Merged constraint semantics, ordering, overrides and provenance filters
  test_shared_relation_regeneration.py # Frozen hashes and graph-isomorphic 0.1 regeneration
  run_e2e_test.py                 # End-to-end smoke test with real, billable LLM calls
```
