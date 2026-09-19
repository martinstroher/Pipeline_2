# PreSaltOntoLearn Evaluation Study

Standalone thesis-study package for the A/B/C/D ablation, automated statistics, expert workbooks, Layer 2 analysis, and zero-Azure rehearsal.

The production ontology pipeline remains at repository root. This package reads frozen pipeline artifacts but writes only under `evaluation_study/output/`.

## Corpus identity

The study source list is published without article files or article passages.
[`corpus_bibliography.csv`](corpus_bibliography.csv) contains the unique
article titles and authors represented in the historical corpus. It identifies
the publications at a human-readable level; it does not grant article
redistribution rights or reconstruct missing document hashes and
per-document extraction records.

## Pipeline Inputs

| Artifact | Default path |
|---|---|
| Filtered 407-term set | `inputs/frozen_a/extract_filtered.csv` |
| Frozen Condition-A terms, NLDs, and context-use flags | `inputs/frozen_a/define_nld.csv` |
| Frozen Condition-A categories | `inputs/frozen_a/classify_categories.csv` |
| Frozen construction artifacts | `inputs/approved_run/` |
| Frozen evaluated ontology | `../output/final/presalt_ontology.ttl` |

The study never modifies these inputs. They are frozen copies, so root pipeline
outputs can be cleaned without breaking study reproducibility.

The public Condition-A CSV excludes the raw `Context` column because it
contained retrieved article passages. Its original full-context SHA-256 and
the public redaction are recorded in `inputs/manifest.json`. Public A/B/C
analysis remains possible. Exact Condition-D replay and the full offline
rehearsal require an authorized private copy of the original CSV:

```bash
ABLATION_FROZEN_A_NLD=/private/define_nld_with_context.csv \
  python -m evaluation_study.cli ablation
python -m evaluation_study.cli rehearsal \
  --a-nld /private/define_nld_with_context.csv --overwrite
```

## Commands

Run from repository root:

Live ablation conditions B/C/D require nonblank `LLM_GENERATION_MODEL` and Azure
credentials. Missing model configuration stops the run before output creation;
the study does not supply a default deployment. Copying frozen condition A,
statistics, workbook generation, and offline rehearsal remain model-free.
For an A-only ablation, the experiment manifest records the configured model
name or `null` if none is configured; it does not invent a model identity.

```bash
python -m evaluation_study.cli ablation
python -m evaluation_study.cli layer1
python -m evaluation_study.cli expert-workbooks
python -m evaluation_study.cli layer2 \
  evaluation_study/output/ablation/expert_workbooks/*.xlsx \
  --key evaluation_study/output/ablation/private/blinding_key_42.csv
# Requires --a-nld with the authorized private full-context CSV:
python -m evaluation_study.cli rehearsal --a-nld /private/define_nld_with_context.csv --overwrite
```

Within this package, only files in `output/ablation/expert_workbooks/` are distributable. Keep `output/ablation/private/` inaccessible to experts.

## Expert workbooks

The default design generates three independently ordered workbooks. Every expert receives every sampled item; only row order and blinded Definition 1/Definition 2 order differ.

Visible sheets use plain geological language:

- `Category_Guide` translates opaque upper-ontology labels and gives examples.
- every expert-facing sheet starts with one short `What to do` banner.
- `Category_Correct` shows the same reviewed, condition-independent `Reference_Definition` for every competing proposal for a term. Real workbook generation refuses missing, pending, duplicate, or empty definitions.
- `Defined_Classes` shows one natural-language definition and one verdict.
- `Relations` samples generic, corpus-context, and individual-fact rows, shows an explicit scope prefix, and asks for one verdict.
- `Taxonomy` separately asks whether the IS-A relation is correct and whether the distinction is useful for the Pre-Salt model.
- `Meaning_Preservation` compares plain-language before/after states for sampled exclusions and demotions without exposing formal critic jargon.
- Meaning Preservation samples only extracted source terms; LLM-created intermediate taxonomy nodes are excluded from domain-expert review.
- Rehearsal `Reference_Definition` values are the first sentence of frozen Condition A and are preview-only. Real workbooks must replace them with the agreed reviewed neutral-definition CSV.
- `Timing` records actual completion time by module.

Formal identifiers and source metadata remain in the private key. The display text is curated in `config/display_text.yaml`; workbook generation makes no LLM call.
Set `EXPERT_REFERENCE_DEFINITIONS` (or pass `--reference-definitions`) to the approved CSV/XLSX. Its hash is recorded in the workbook manifest. Synthetic rehearsal explicitly bypasses the approval gate and uses A-derived preview definitions.

For inferential representation comparisons, `Unsure` is missing data.
A/B contrasts use only expert-term rows containing both ratings. The
four-condition correctness test uses only expert-term rows with decisive
ratings in all four conditions before aggregating to the term level.
Condition-specific descriptive summaries continue to report their available
ratings and Unsure rates.

## Layout

```text
evaluation_study/
  cli.py                    # Standalone command surface
  paths.py                  # Study outputs and frozen pipeline inputs
  config/expert_eval.yaml   # Expert-facing instructions
  config/display_text.yaml  # Plain-language ontology labels and definitions
  prompts/                  # Study-only prompt variants
  test/                     # Study tests and snapshots
  output/                   # Ignored generated study artifacts
```

## Tests

```bash
python evaluation_study/test/test_evaluation_study.py
python evaluation_study/test/test_display_text.py
python evaluation_study/test/test_offline_rehearsal.py
python evaluation_study/test/diff_instructions_sheet.py
```

## Environment Overrides

Study defaults are isolated, but these overrides remain available:

- `STUDY_CONFIG_PATH`
- `DISPLAY_TEXT_CONFIG_PATH`
- `ABLATION_OUTPUT_DIR`
- `ABLATION_FROZEN_A_NLD`
- `ABLATION_FROZEN_A_CATEGORY`
- `ABLATION_EXPECTED_TERM_COUNT`
- `EXPERT_ONTOLOGY_DIR`
- `EXPERT_TERMS_PATH`
- `EXPERT_BOOTSTRAP_ITERATIONS`
- `EXPERT_STRICT_APPROVED_POPULATIONS`

The study still uses the root `.env` for Azure credentials and common model settings when running paid B/C/D conditions.
