# Authoring a Domain — PreSaltOntoLearn

The PreSaltOntoLearn pipeline is retargetable. The Python code in `src/` does not
contain any domain-specific text (no personas, no calibration examples, no upper-
ontology IRIs, no Likert anchors). Everything that distinguishes Pre-Salt geology
from another scientific domain lives under this `domains/` tree.

To create a new domain, run `python scripts/new_domain.py <your_domain>`, edit
the generated domain files, and activate its config with `ONTOLOGY_CONFIG_PATH`.
The template already loads 61 generic BFO/RO relation entries from
[`_shared/bfo_ro_relations.yaml`](_shared/bfo_ro_relations.yaml). Only
discipline-specific relations need to be authored: Pre-Salt adds 10, reducing
its relation-authoring work from 71 entries to 10 (about 86%).

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
│   ├── critic_taxonomy_dedup.txt
│   ├── critic_relations.txt
│   ├── cq_scoring.txt
│   └── cq_synonym_triage.txt
├── resources/                   # Scaffold copies BFO + RO; add other OWL files here
│   ├── bfo-core.owl
│   └── ro-core.owl
└── competency_questions.txt      # CQs evaluated by the CQ pipeline step
```

`domains/_shared/` is shared by domain folders, not copied into each one.
Keep it alongside your domain when moving the bundle, or update the
`relation_defaults` path to its new location.

Two more artifacts sit outside the domain because they describe cross-domain studies:

- `evaluation_study/prompts/ablation_categorization_rag.txt` — standalone Condition-D raw-context prompt
- `evaluation_study/config/expert_eval.yaml` — standalone expert-workbook instructions

## Activation

```powershell
$env:ONTOLOGY_CONFIG_PATH = "domains/your_domain/ontology_config.yaml"
# Study configuration is documented separately in evaluation_study/README.md.
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
| `critic_taxonomy.txt` | `{category}`, `{terms_json}`, `{relations_context_json}`, `{parent_context_json}`, `{target_classes_json}` |
| `critic_taxonomy_dedup.txt` | `{category}`, `{survivors_json}` |
| `critic_relations.txt` | `{category}`, `{relations_json}`, `{relations_menu_json}`, `{previously_minted_json}`, `{taxonomy_context_json}`, `{taxonomy_decisions_json}` |
| `cq_scoring.txt` | `{batch_size}`, `{terms_json}` |
| `cq_synonym_triage.txt` | `{clusters_json}` |

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
  key must have at least one class with `metatypes:` set. RO supplies properties
  only and is not included in this cascade.
- Each ontology entry specifies `owl:`, `namespace:`, `prefix:`, and a `classes:`
  list; `display_name:` names its prompt section. Classes used in categorization
  need `metatypes:` (e.g., `[MaterialEntity]`).
- `relations:` contains local additions to the property-constraint registry.
  Each entry needs `iri`, `domain`, `range`, and
  explicit `provenance ∈ {owl_axiom, bfo_shape_axiom, ro_release, critic_minted}`.
  `inverse`, `notes`, and `critic_menu` are optional.

### Shared BFO/RO relations

Both the template and Pre-Salt use:

```yaml
relation_defaults: "../_shared/bfo_ro_relations.yaml"
relations: {}  # No local additions; all 61 shared entries are still loaded.
```

The fragment declares **44 BFO-IRI and 17 RO-IRI entries**, including existing
aliases. It also supplies their complete metatype groups and the rules that
specialize generic parthood to continuant/occurrent parthood. The template
therefore works without copying constraints or recreating their group names.
The 5 GeoCore and 5 GeoReservoir entries remain only in the Pre-Salt config.

The loader supports one optional defaults file using ordinary `yaml.safe_load`,
not custom YAML tags or recursive includes:

- Paths resolve relative to the domain's config, regardless of the working
  directory. Prompt and OWL-resource paths remain relative to the domain.
- Only `metatype_groups`, `relations`, and `property_specializations` are allowed
  in the fragment. Invalid files and section types raise `RuntimeError`.
- Local groups and relation names are merged after the defaults. A same-name
  entry replaces the **whole** inherited entry, not individual fields. Supply
  all required fields when deliberately overriding a constraint.
- Omit `property_specializations` to inherit the shared rules. A local list
  replaces the entire inherited list; `[]` disables specialization.
- An empty local `relations: {}` or `metatype_groups: {}` retains defaults.
  Configs without `relation_defaults` continue to load as standalone files.
- Use `get_config().all_relations()` or `property_constraints()` to read the
  resolved registry; a raw YAML read sees only the reference and local entries.
  Provenance filtering and `critic_menu` flags keep their existing behavior.

## Validation gates

After any change to a production prompt or `ontology_config.yaml`:

```powershell
python test/test_ontology_config_parity.py    # 26 checks; must show "=== PARITY PASSED ==="
python test/test_prompt_refactor_parity.py    # active prompts resolve without drift or missing placeholders
python -m unittest discover -s test -p "test_shared_relation*.py" -v
```

The new-domain tests check the shared registry, merging, invalid inputs,
cache/reload behavior, and a generated scaffold. The frozen-ontology test runs
the exporter directly from committed approved inputs with network access
disabled, checks input hashes, and requires the same 1,819-triple graph as
GeoPreSalt 0.1. It does not run the live-model construction/critic steps.
See [`output/final/README.md`](../output/final/README.md) for the archived artifact.

For changes that touch live LLM behavior, run the e2e smoke test on one paper
before launching a production run:

```powershell
python test/run_e2e_test.py
```
