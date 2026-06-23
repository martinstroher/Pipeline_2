# Authoring a Domain — PreSaltOntoLearn

The PreSaltOntoLearn pipeline is retargetable. The Python code in `src/` does not
contain any domain-specific text (no personas, no calibration examples, no upper-
ontology IRIs, no Likert anchors). Everything that distinguishes Pre-Salt geology
from another scientific domain lives under this `domains/` tree.

To create a new domain, copy `presalt/` to `<your_domain>/`, edit the two artifacts
described below, and point the loaders at the new path via env vars.

## Layout (per domain)

```
domains/<your_domain>/
├── ontology_config.yaml          # Upper-ontology metadata + relation constraints + waterfall
├── prompts/                      # 8 production-pipeline prompts
│   ├── term_extraction.txt
│   ├── nld_generation.txt
│   ├── term_categorization.txt
│   ├── taxonomy_building.txt
│   ├── relation_extraction.txt
│   ├── critic_taxonomy.txt
│   ├── critic_relations.txt
│   ├── cq_scoring.txt
│   └── cq_synonym_triage.txt
├── bfo-core.ttl                  # Upper ontologies imported by Step 7
├── geocore-full.ttl
├── geores-full.ttl
├── ro-core.ttl
└── competency_questions.txt      # CQs evaluated by the CQ pipeline step
```

Two more artifacts sit outside the domain because they describe cross-domain studies:

- `studies/prompts/ablation_categorization_{nld,rag}.txt` — ablation-only prompts
- `studies/expert_eval.yaml` — expert-evaluation workbook instructions sheet

## Activation

```powershell
$env:ONTOLOGY_CONFIG_PATH = "domains/your_domain/ontology_config.yaml"
# Optional: $env:STUDY_CONFIG_PATH = "studies/your_eval.yaml"
```

`ontology_config.py` derives the prompt root from the directory containing the
active `ontology_config.yaml`, so prompts are found automatically.

## Prompt format

Each `.txt` file has two sections separated by `[PROMPT_TEMPLATE]`:

```
[SYSTEM_INSTRUCTION]
You are <persona for this prompt>.
<any standing instructions / constraints / output format>.

[PROMPT_TEMPLATE]
<task body with {runtime_placeholders} that the caller fills via str.format>
```

There is no load-time interpolation. The system instruction and template are read
verbatim. The caller substitutes `{name}` placeholders with `str.format(**vars)` at
call time.

## Runtime placeholders per prompt

| Prompt | Placeholders the caller injects |
|---|---|
| `term_extraction.txt` | `{chunk_text}` |
| `nld_generation.txt` | `{term}`, `{context}` |
| `term_categorization.txt` | `{categories_block}`, `{json_batch}` |
| `taxonomy_building.txt` | `{category}`, `{upper_vocab}`, `{terms_json}` |
| `relation_extraction.txt` | `{known_terms}`, `{json_batch}`, `{batch_size}` |
| `critic_taxonomy.txt` | `{category}`, `{terms_json}`, `{relations_context_json}`, `{target_classes_json}` |
| `critic_relations.txt` | `{category}`, `{relations_json}`, `{relations_menu_json}`, `{previously_minted_json}`, `{taxonomy_context_json}`, `{taxonomy_decisions_json}` |
| `cq_scoring.txt` | `{batch_size}`, `{terms_json}` |
| `cq_synonym_triage.txt` | `{clusters_json}` |
| `ablation_categorization_nld.txt` (studies/) | `{categories_block}`, `{json_batch}` |
| `ablation_categorization_rag.txt` (studies/) | `{categories_block}`, `{json_batch}` |

`{categories_block}` is rendered by `OntologyConfig.categorization_block()` from
the `waterfall:` list in `ontology_config.yaml` — one `### <DisplayName> Categories:`
section per ontology in priority order.

## Expected JSON response

The prompt body itself documents the expected schema (object keys, value types,
batch order). Callers validate:

1. The response parses as JSON.
2. For batch prompts, the response is a JSON array whose length matches the input
   batch size. Mismatches raise `ValueError`.

Keep this contract in mind when rewriting prompts: change the schema and the
caller will reject the response.

## `ontology_config.yaml` contract

See `.github/copilot-instructions.md` for the field-by-field spec. Key rules:

- `waterfall:` lists ontology keys in cascade priority (most-specific first). Each
  key must have at least one class with `metatype:` set.
- Each ontology entry needs `display_name:`, `prefix:`, `iri:`, and a `classes:`
  list. Classes used in categorization need `metatype:` (e.g., `MaterialEntity`).
- `relations:` is the 71-entry property-constraint registry. Each entry needs
  explicit `provenance ∈ {owl_axiom, bfo_shape_axiom, ro_release, critic_minted}`.

## Validation gates

After any change to a prompt, `ontology_config.yaml`, or `studies/expert_eval.yaml`:

```powershell
python test/test_ontology_config_parity.py    # 26 checks; must show "=== PARITY PASSED ==="
python test/diff_instructions_sheet.py        # 49 rows; byte-equal to baseline
python test/regression_t1.py                  # deterministic 6d→7→7b regression
```

For changes that touch live LLM behavior, run the e2e smoke test on one paper
before launching a production run:

```powershell
python test/run_e2e_test.py
```
