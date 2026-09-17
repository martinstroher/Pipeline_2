# Setting Up the Pipeline for Your Domain

This is the practical walk-through for someone setting up the pipeline
**from scratch** on a new domain (i.e. not Pre-Salt geology). It is
deliberately short — the CLI reference lives in [`README.md`](README.md),
and the pipeline summary is in [`docs/OVERVIEW.md`](docs/OVERVIEW.md).

Pre-Salt is the default domain. All users still need the runtime and Azure
configuration in [`README.md`](README.md); Pre-Salt users can skip scaffolding
and use the isolated run settings below.

The generator checks the starter against the current production prompts before
creating the final domain folder. Its BFO categories, shared BFO/RO relations,
and neutral examples form a structural demonstration, not a scientifically
reviewed domain. Adapt the two configuration files and re-run the offline check
before making model calls.

---

## 0. Prerequisites

| Required | Why |
|---|---|
| Locked CPython 3.12.14 environment on macOS Apple Silicon or Linux x86-64 | Follow the README's `uv sync --locked` instructions; Linux runtime validation is still pending |
| Pinned spaCy language model | `en_core_web_sm` 3.8.0 is included in the locked install; no separate download command |
| Azure AI Foundry (Azure OpenAI) resource, API key, and deployment | Live LLM calls for extraction, definitions, classification/refinement, construction, and validation |
| Local retrieval model weights | BGE-M3 and BGE-Reranker download on first use; retrieval runs before extraction |

| Optional | When you need it |
|---|---|
| `reasoner` dependency group + working Java on PATH (or `JAVA_EXE`) | Step 7b HermiT reasoner; the locked Python bridge does not install or validate Java |
| Docker (with `mpovedavillalon/oops:v1`) | Step 7b OOPS! pitfall scanner (Layer 3 verification) |

The pipeline can export without the optional checks completing. Inspect each
verification layer's status and errors; skipped checks are not evidence of
correctness. See the README for portable Java setup, local OOPS! configuration,
and flags that skip those checks. The example `.env` selects the public OOPS!
service, which receives the full ontology unless disabled or changed to local.

Copy `.env.example` to `.env` only if `.env` does not already exist. Set
`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` (resource host only), and
`LLM_GENERATION_MODEL` / `LLM_EXTRACTION_MODEL` to your Azure deployment names.
The example selects `gpt-5.4` for both model variables. There is no fallback:
startup rejects missing or blank names needed by the selected steps before
conversion, retrieval, or cleanup. Extraction uses `LLM_EXTRACTION_MODEL`;
the other LLM stages use `LLM_GENERATION_MODEL`. Standalone export/verification
and help do not need either. The client always uses Azure; Gemini/Vertex
credentials do not configure it. Keep all required input/output variables
from the example.

---

## 1. Scaffold your domain

```powershell
python scripts/new_domain.py mydomain
```

This creates `domains/mydomain/` from `domains/_template/`, with:

- Three starter files: `README.md`, `ontology_config.yaml`, and `prompt_blocks.yaml`
- The production prompts copied from `domains/presalt/prompts/`;
  their `<<block>>` markers resolve from your `prompt_blocks.yaml`
- `resources/bfo-core.owl` and `resources/ro-core.owl`, the saved upper-ontology resources

The configuration references `domains/_shared/bfo_ro_relations.yaml` rather than
copying its 61 generic relation entries. Keep that shared directory alongside
the domain when relocating it. Missing required source files or failed checks
leave no partial domain folder; success is printed only after validation.

`<mydomain>` must be 2-32 characters, start with a lowercase letter, and
contain only lowercase letters, digits, or underscores. Existing directories
are not overwritten.

---

## 2. Configure the domain

Open `domains/mydomain/` and edit its two configuration files.

### 2a. `prompt_blocks.yaml` (REQUIRED)

Set `domain.name` and `domain.scope`, and keep `domain.upper_stack` and
`domain.waterfall` aligned with `ontology_config.yaml`. Personas derive from
these values. Replace the neutral worked examples with examples from your
subject while preserving their output fields and JSON types.

The relation-property table derives from active constraints flagged
`critic_menu: true` in the ontology configuration. Changing that configuration
updates the table on the next process/load; no separate property list is needed.

### 2b. `ontology_config.yaml` — `project:` section (REQUIRED)

Set `name`, `namespace`, and `prefix` for your ontology. The default
template uses BFO for categorization and RO as a property supplier, with
generic relation constraints inherited from the shared registry. A same-name
local relation or metatype group replaces the whole inherited entry; an
explicit specialization list replaces all inherited rules (`[]` disables
them). To layer in more class ontologies (GeoCore, ChEBI, GO, …) see
[`domains/README.md`](domains/README.md) and the much larger
[`domains/presalt/ontology_config.yaml`](domains/presalt/ontology_config.yaml)
as a reference.

### 2c. Competency questions (REQUIRED for the full pipeline)

Write the questions in `examples.cq_questions` in `prompt_blocks.yaml` and
keep `examples.cq_identifiers` aligned. This is the runtime question list;
new domains have no separate question text file. The current scorer accepts
identifiers `CQ1` through `CQ10`. Its output is a list of matching question
identifiers with a reason, not numeric per-question scores.
CQ refinement is mandatory in the standard pipeline; there is no `--refine`
flag. Customize the demonstration questions before a live run.

The obsolete `domain_filters.yaml` is no longer copied. Production validation
uses the active ontology configuration and `critic_*.txt` prompts.

---

## 3. Activate the domain and isolate the run

Set this line in `.env` for a new domain (Pre-Salt users can leave the default):

```
ONTOLOGY_CONFIG_PATH=domains/mydomain/ontology_config.yaml
```

`prompt_loader` discovers `prompt_blocks.yaml` and `prompts/` from the
same directory automatically.

For a one-document smoke run, replace these settings in `.env`. Use a new
directory name for each experiment; do not point them at frozen outputs.

```dotenv
DOCS_DIR=inputs/mydomain_smoke/
LLM_INPUT_DIR=inputs/mydomain_smoke/
CHROMA_DB_DIR=output/mydomain_smoke/chroma_db
LLM_OUTPUT_FILE=output/mydomain_smoke/extract_raw.json
AGGREGATOR_OUTPUT_FILE=output/mydomain_smoke/extract_aggregated.csv
FILTERED_TERMS_OUTPUT=output/mydomain_smoke/extract_filtered.csv
CONSOLIDATED_LLM_RESULTS_WITH_NLDS=output/mydomain_smoke/define_nld.csv
OUTPUT_FAILURE_FILE=output/mydomain_smoke/define_failures.csv
CATEGORIZED_LLM_TERMS=output/mydomain_smoke/classify_categories.csv
LLM_USAGE_LOG=output/mydomain_smoke/usage_log.csv
MINIMUM_FREQUENCY_FILTER=1
```

The frequency threshold of `1` is for this single-document smoke test only.
The example default is `5` distinct source documents, which would filter out
every term from a one-document corpus. Choose the production threshold
deliberately when moving to the full corpus. Changing only `DOCS_DIR` does not
isolate extraction, outputs, or retrieval caches.

---

## 4. Validate the configuration

Run the same offline checks used by the generator:

```powershell
python -m src.utils.domain_validation domains/mydomain
```

This uses the specified domain directory without changing the active-domain
cache. It checks resource files, all production prompt names, required blocks,
caller placeholders, question identifiers, and example output schemas. It
makes no LLM calls and writes no pipeline outputs. Passing establishes
structural compatibility, not scientific correctness or a successful live run.

---

## 5. First run — single document smoke test

Only after Section 4 passes, put one short PDF or Markdown document in the
configured input directory (here `inputs/mydomain_smoke/`), then:

```powershell
python pipeline.py --stop-after extract
```

This runs PDF conversion, retrieval setup, and extraction/aggregation/filtering.
Extraction makes live, billable LLM calls; definition and classification calls
have not started. Cost depends on the document and Azure deployment.
Inspect `output/mydomain_smoke/extract_filtered.csv` before continuing.

Then run the full pipeline:

```powershell
python pipeline.py --skip-pdf --skip-extraction
```

This reuses the filtered terms but still performs retrieval setup and the
remaining LLM stages, including CQ refinement and the production critic.
With the isolated settings above, the standard final export is
`output/mydomain_smoke/refined/validate_taxonomy.ttl`; its verification report
is `emit_verification.json` in the same directory. Open the Turtle file in
Protégé to inspect. Never overwrite `output/final/` or the frozen evaluation
inputs when trying a new run.

---

## 6. Iterate

Some stages resume completed terms from CSV checkpoints; others regenerate
outputs and make new LLM calls. A prompt or configuration edit does not
invalidate old checkpoints automatically. To evaluate changed behavior, keep
the old results and use a new set of output paths (plus a new retrieval-cache
prefix when the corpus changes).

| You noticed... | Edit... | Re-run... |
|---|---|---|
| Extractor missing key terms | `prompts/term_extraction.txt` + extraction examples | `python pipeline.py --stop-after extract` |
| NLDs wrong tone / scope | `prompt_blocks.yaml` examples + `prompts/nld_generation.txt` | `python pipeline.py --skip-extraction --stop-after define` |
| Wrong categorisation | `prompt_blocks.yaml` (category examples) + `ontology_config.yaml` (waterfall) | `python pipeline.py --skip-extraction --stop-after classify` |
| Junk surviving validation | Active ontology configuration + `prompts/critic_*.txt` | `python pipeline.py --skip-extraction --stop-after validate` |

Commands using `--skip-extraction` require existing filtered terms at the
configured path. Copy any inputs needed for the experiment into its new
directory first. `--stop-after` stops at the named phase; it does not start
there or skip earlier LLM stages. Avoid `--fresh`: it deletes Markdown inputs
and root retrieval caches as well as selected outputs, and is not a
run-specific checkpoint reset.

---

## Where things live

```
domains/
  _template/       ← scaffold source (never edit at runtime)
  _shared/         ← generic BFO/RO relation constraints and specialization rules
  presalt/         ← reference domain (also useful as an example)
  mydomain/        ← your domain
    ontology_config.yaml
    prompt_blocks.yaml
    prompts/       ← current production prompts (auto-copied)
    resources/     ← bfo-core.owl and ro-core.owl (auto-copied) + any extra OWL files
scripts/
  new_domain.py    ← the scaffold script
```

For the per-module summary and the evaluation framework, see
[`docs/OVERVIEW.md`](docs/OVERVIEW.md).
