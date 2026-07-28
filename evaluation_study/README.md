# PreSaltOntoLearn Evaluation Study

Standalone thesis-study package for the A/B/C/D ablation, automated statistics, expert workbooks, Layer 2 analysis, and zero-Azure rehearsal.

The production ontology pipeline remains at repository root. This package reads frozen pipeline artifacts but writes only under `evaluation_study/output/`.

## Pipeline Inputs

| Artifact | Default path |
|---|---|
| Filtered 407-term set | `inputs/frozen_a/extract_filtered.csv` |
| Frozen Condition-A NLDs and contexts | `inputs/frozen_a/define_nld.csv` |
| Frozen Condition-A categories | `inputs/frozen_a/classify_categories.csv` |
| Approved validate artifacts | `inputs/approved_run/` |
| Final approved ontology | `../output/final/presalt_ontology.ttl` |

The study never modifies these inputs. They are frozen copies, so root pipeline
outputs can be cleaned without breaking study reproducibility.

Categorizer prompts are rendered by replacing only the explicit batch marker;
embedded JSON examples remain literal and are covered by the offline test suite.

## Commands

Run from repository root:

```bash
python -m evaluation_study.cli ablation
python -m evaluation_study.cli layer1
python -m evaluation_study.cli expert-workbooks
python -m evaluation_study.cli layer2 \
  evaluation_study/output/ablation/expert_workbooks/*.xlsx \
  --key evaluation_study/output/ablation/private/blinding_key_42.csv
python -m evaluation_study.cli rehearsal --overwrite
```

Within this package, only files in `output/ablation/expert_workbooks/` are distributable. Keep `output/ablation/private/` inaccessible to experts.

## Expert Workbook vNext

The default design generates three independently ordered workbooks. Every expert receives every sampled item; only row order and blinded Definition 1/Definition 2 order differ. Representation uses a 100-term sample stratified by Condition-A tier and frequency. Category Correct uses a separate deterministic 60-term sample from the union of A-vs-B/C/D proposal disagreements, stratified by disagreement overlap, Condition-A tier, and frequency. Each A contrast has a minimum of 25 sampled terms.

Visible sheets use plain geological language:

- `Category_Guide` translates opaque upper-ontology labels and gives examples.
- every expert-facing sheet starts with one short `What to do` banner.
- `Category_Correct` shows the same reviewed, condition-independent `Reference_Definition` for every competing proposal for a term. The disagreement-enriched sample estimates comparative correctness on disputed terms, not population-wide absolute correctness. Real workbook generation refuses missing, pending, duplicate, or empty definitions.
- `Defined_Classes` shows one natural-language definition and one verdict.
- `Relations` samples generic, corpus-context, and individual-fact rows, shows an explicit scope prefix, and asks for one verdict.
- `Taxonomy` separately asks whether the IS-A relation is correct and whether the distinction is useful for the Pre-Salt model.
- `Meaning_Preservation` compares plain-language before/after states for sampled exclusions and demotions without exposing formal critic jargon. It records semantic preservation and lean-core appropriateness as separate mandatory judgments; Preferred Outcome remains conditional on `Mostly` or `No` semantic preservation.
- Meaning Preservation samples only extracted source terms; LLM-created intermediate taxonomy nodes are excluded from domain-expert review.
- Rehearsal `Reference_Definition` values are the first sentence of frozen Condition A and are preview-only. Real workbooks must replace them with the agreed reviewed neutral-definition CSV.
- `Timing` records actual completion time by module.

Formal identifiers and source metadata remain in the private key. The display text is curated in `config/display_text.yaml`; workbook generation makes no LLM call.
Set `EXPERT_REFERENCE_DEFINITIONS` (or pass `--reference-definitions`) to the approved CSV/XLSX. Its hash is recorded in the workbook manifest. Synthetic rehearsal explicitly bypasses the approval gate and uses A-derived preview definitions.

Layer 2 keeps the term-level paired analysis as primary. It always reports A-vs-B/C/D contrasts on sampled proposal disagreements with bootstrap confidence intervals and Holm correction. Exploratory cross-layer Spearman tests also use Holm correction. Raw pairwise agreement and response marginals are reported beside Fleiss kappa; Gwet AC1 and ordinal AC2 are sensitivity measures. Ties and intermediate responses remain explicit, `Unsure` is reported and excluded only from score-based tests and AC2, and conditional blanks are structurally inapplicable. Conditional issue reasons and preferred outcomes are validated against their exact workbook labels. The analyzer writes `discordant_category_contrasts.csv`, `cross_layer_spearman.csv`, `item_consensus.csv`, and `consensus_summary.csv` beside `layer2_results.json`.

## Layout

```text
evaluation_study/
  cli.py                    # Standalone command surface
  paths.py                  # Study outputs and frozen pipeline inputs
  config/expert_eval.yaml   # Expert-facing instructions
  config/display_text.yaml  # Plain-language ontology labels and definitions
  prompts/                  # Study-only prompt variants
  test/                     # Study tests and snapshots
  docs/                     # Evaluation methods and usability reports
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
