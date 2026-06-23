# Setting Up the Pipeline for Your Domain

This is the practical walk-through for someone setting up the pipeline
**from scratch** on a new domain (i.e. not Pre-Salt geology). It is
deliberately short — the deep reference lives in [`README.md`](README.md)
and [`docs/CONTEXT.md`](docs/CONTEXT.md).

Pre-Salt users: you do not need this file. `domains/presalt/` is the
default and everything just works.

---

## 0. Prerequisites

| Required | Why |
|---|---|
| Python 3.10+ | Pipeline runtime |
| `pip install -r requirements.txt` | Dependencies |
| Google Gemini API key **or** Vertex AI credentials | LLM calls (Steps 1, 4, 5, 5b, 6b, 6c) |

| Optional | When you need it |
|---|---|
| Java 11+ on PATH (`JAVA_EXE`) | Step 7b HermiT reasoner (Layer 4 verification) |
| Docker (with `mpovedavillalon/oops:v1`) | Step 7b OOPS! pitfall scanner (Layer 3 verification) |

Both optional pieces fail soft — the pipeline still produces a `.ttl`
output without them. Set them up only when you want full verification.

---

## 1. Scaffold your domain

```powershell
python scripts/new_domain.py mydomain
```

This creates `domains/mydomain/` from `domains/_template/`, with:

- 5 user-edit files (`README.md` + the 4 config files you edit below)
- The 23 generic prompts copied from `domains/presalt/prompts/`
  (these are domain-agnostic — `<<block>>` markers resolve from your
  `prompt_blocks.yaml` at load time)
- `resources/bfo-core.owl` (so the BFO-only template works out of the box)

`<mydomain>` must be lowercase, 2-32 chars, letters/digits/underscores only.

---

## 2. Edit four files

Open `domains/mydomain/` and walk through these in order. Time budget:
**~30 minutes for a minimum-viable run, longer for a polished domain.**

### 2a. `prompt_blocks.yaml` — Section A (REQUIRED)

The top of the file has 9 `REPLACE:` markers under `domain:` (role,
scope, name, ontology phrase, subject phrase/plural, adjective,
specialty, waterfall sentence, etc.). Fill them in with your domain's
identity. Sections B (personas) and C (examples) inherit from these
automatically — you can leave them alone for a first run and refine
later.

### 2b. `ontology_config.yaml` — `project:` section (REQUIRED)

Set `name`, `namespace`, and `prefix` for your ontology. The default
template ships with BFO only as the upper ontology, which is enough to
run end-to-end. To layer in more (GeoCore, ChEBI, GO, …) see
[`domains/README.md`](domains/README.md) and the much larger
[`domains/presalt/ontology_config.yaml`](domains/presalt/ontology_config.yaml)
as a reference.

### 2c. `competency_questions.txt` (RECOMMENDED)

Write 5-10 questions your ontology should answer. Only required if you
plan to run `--refine` (CQ-driven refinement); otherwise the file can
stay with TODO defaults.

### 2d. `domain_filters.yaml` (OPTIONAL)

The template enables the 6 cheap structural filters and comments out
the LLM filters. Un-comment LLM filters once your domain stabilises and
you want stronger cleanup. See
[`domains/presalt/domain_filters.yaml`](domains/presalt/domain_filters.yaml)
for the full set with notes.

---

## 3. Activate the domain

Add one line to your `.env`:

```
ONTOLOGY_CONFIG_PATH=domains/mydomain/ontology_config.yaml
```

`prompt_loader` discovers `prompt_blocks.yaml` and `prompts/` from the
same directory automatically.

---

## 4. Validate the configuration

Quick sanity check that your config loads and all prompts resolve:

```powershell
python -c "from src.utils.ontology_config import get_config; print(get_config().project.name, list(get_config().ontologies))"
python -c "from src.utils import prompt_loader; [prompt_loader.load_prompt(n) for n,_ in prompt_loader.prompt_files()]; print('OK')"
```

If either fails, the error message names the missing block or YAML key.
The second command catches typos in `<<block>>` references before you
spend any LLM credits.

---

## 5. First run — single document smoke test

Drop one short source document into `inputs/` (PDF or `.md`), then:

```powershell
python pipeline.py --stop-after extract
```

This runs Steps 0-3 (PDF → terms → aggregate → filter) without any
NLD or categorisation LLM calls. Costs a few cents. Inspect
`output/extract_filtered.csv` — if the terms look sensible, your
extractor prompt is well-tuned to your domain.

Then run the full pipeline:

```powershell
python pipeline.py
```

Outputs land in `output/`. The final ontology is the `.ttl` next to
the last numbered step (`6d_taxonomy_reclassified.ttl` by default).
Open it in Protégé to inspect.

---

## 6. Iterate

The pipeline is designed to be re-run cheaply because every step
checkpoints to a CSV on disk. The `--stop-after` flag takes verb
aliases — `extract`, `define`, `classify`, `construct`, `validate`,
`emit` — each stopping at the last sub-step of that phase.

| You noticed... | Edit... | Re-run... |
|---|---|---|
| Extractor missing key terms | `prompts/extract_terms.txt` | `python pipeline.py --stop-after extract` |
| NLDs wrong tone / scope | `prompt_blocks.yaml` Section C (NLD examples) + `prompts/define_*` | `python pipeline.py --skip-extraction --stop-after define` |
| Wrong categorisation | `prompt_blocks.yaml` (category examples) + `ontology_config.yaml` (waterfall) | `python pipeline.py --skip-extraction --stop-after classify` |
| Junk surviving validation | `domain_filters.yaml` (un-comment LLM filters) | `python pipeline.py --skip-extraction --stop-after validate` |

Earlier checkpointed CSVs are reused automatically as long as their
filenames have not changed, so skipping a phase costs nothing.

---

## Where things live

```
domains/
  _template/       ← scaffold source (never edit at runtime)
  presalt/         ← reference domain (also useful as an example)
  mydomain/        ← your domain
    ontology_config.yaml
    prompt_blocks.yaml
    domain_filters.yaml
    competency_questions.txt
    prompts/       ← 17 generic prompts (auto-copied)
    resources/     ← bfo-core.owl (auto-copied) + any extra OWL files
scripts/
  new_domain.py    ← the scaffold script
```

For the deep architecture reference, see [`docs/CONTEXT.md`](docs/CONTEXT.md).
For the per-module summary and the evaluation framework, see
[`docs/OVERVIEW.md`](docs/OVERVIEW.md).
