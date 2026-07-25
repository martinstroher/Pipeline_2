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

## Commands

Run from repository root:

```bash
python -m evaluation_study.cli ablation
python -m evaluation_study.cli layer1
python -m evaluation_study.cli expert-workbooks
python -m evaluation_study.cli layer2 \
  evaluation_study/output/ablation/expert_workbooks/expert_evaluation_1.xlsx \
  evaluation_study/output/ablation/expert_workbooks/expert_evaluation_2.xlsx \
  evaluation_study/output/ablation/expert_workbooks/expert_evaluation_3.xlsx \
  --key evaluation_study/output/ablation/private/blinding_key_42.csv
python -m evaluation_study.cli rehearsal --overwrite
```

Within this package, only files in `output/ablation/expert_workbooks/` are distributable. Keep `output/ablation/private/` inaccessible to experts.

## Layout

```text
evaluation_study/
  cli.py                    # Standalone command surface
  paths.py                  # Study outputs and frozen pipeline inputs
  config/expert_eval.yaml   # Expert-facing instructions
  prompts/                  # Study-only prompt variants
  test/                     # Study tests and snapshots
  docs/                     # Evaluation methods and usability reports
  output/                   # Ignored generated study artifacts
```

## Tests

```bash
python evaluation_study/test/test_evaluation_study.py
python evaluation_study/test/test_offline_rehearsal.py
python evaluation_study/test/diff_instructions_sheet.py
```

## Environment Overrides

Study defaults are isolated, but these overrides remain available:

- `STUDY_CONFIG_PATH`
- `ABLATION_OUTPUT_DIR`
- `ABLATION_FROZEN_A_NLD`
- `ABLATION_FROZEN_A_CATEGORY`
- `ABLATION_EXPECTED_TERM_COUNT`
- `EXPERT_ONTOLOGY_DIR`
- `EXPERT_BOOTSTRAP_ITERATIONS`
- `EXPERT_STRICT_APPROVED_POPULATIONS`

The study still uses the root `.env` for Azure credentials and common model settings when running paid B/C/D conditions.
