# Authoring a domain

From the repository root:

```bash
python scripts/new_domain.py your_domain
```

The generator copies the starter configuration, current production prompts,
and saved BFO/RO resources into a temporary sibling folder. It checks the
complete result before moving it to `domains/your_domain/`. Missing files,
invalid blocks, incompatible examples, and existing target directories cause
an error instead of a success message.

## Layout

```text
domains/
  _shared/bfo_ro_relations.yaml   # Generic relation constraints; retained beside domains
  your_domain/
    README.md
    ontology_config.yaml         # Project metadata, category vocabulary, local relations
    prompt_blocks.yaml           # Identity, personas, questions, worked examples
    prompts/                     # Current production prompt files
    resources/
      bfo-core.owl
      ro-core.owl
```

Edit the two YAML files, then check and activate the domain:

```bash
python -m src.utils.domain_validation domains/your_domain
```

```dotenv
ONTOLOGY_CONFIG_PATH=domains/your_domain/ontology_config.yaml
```

See [SETUP.md](../SETUP.md) for model settings, isolated inputs/outputs, and
live-run costs. The neutral examples demonstrate formats, not scientific
adequacy. Review and replace them for the intended subject before using an LLM.

## Text substitution and questions

Each prompt has `[SYSTEM_INSTRUCTION]` and `[PROMPT_TEMPLATE]` sections.
At load time, `<<name>>` markers resolve from `prompt_blocks.yaml`; nested
keys flatten with underscores (`personas.scope_auditor` becomes
`<<personas_scope_auditor>>`). Block values can reference other blocks.
Missing names and cycles are errors.

At call time, the caller fills `{name}` fields in the prompt body using
`str.format()`. JSON examples in that body need doubled braces. System text
is not formatted again; system-only worked examples can use ordinary JSON.

The runtime question list is `examples.cq_questions`; its identifiers must
match `examples.cq_identifiers` and fall within CQ1 through CQ10. The scorer
returns an array of objects containing `term`, `relevant_cqs`, and `reasoning`.

`<<config_relation_property_table>>` is an opt-in, derived block. It builds
the extraction property table from active configured relations whose
`critic_menu` flag is true. The starter references it from
`examples.relation_property_table`; existing literal tables are unchanged.

## Runtime fields

The checker verifies these interfaces against the copied prompt bodies:

| Prompt | Fields supplied by its caller |
|---|---|
| `term_extraction.txt` | `chunk_text` |
| `nld_generation.txt` | `term`, `context` |
| `term_categorization.txt` | `categories_block`, `json_batch` |
| `taxonomy_building.txt` | `category`, `upper_vocab`, `terms_json` |
| `relation_extraction.txt` | `known_terms`, `json_batch`, `batch_size` |
| `cq_scoring.txt` | `batch_size`, `terms_json` |
| `cq_synonym_triage.txt` | `clusters_json` |
| `critic_taxonomy.txt` | `category`, `terms_json`, `relations_context_json`, `parent_context_json`, `target_classes_json`, `weak_observations_json` |
| `critic_taxonomy_dedup.txt` | `survivors_json`, `cross_candidates_json` |
| `critic_class_worthiness.txt` | `category`, `candidates_json`, `existing_classes_json`, `properties_json`, `relations_context_json`, `sibling_context_json`, `weak_observations_json` |
| `critic_facet_frames.txt` | `category`, `survivors_json`, `existing_classes_json`, `target_context_json`, `max_candidates` |
| `critic_frame_completion.txt` | `candidates_json` |
| `critic_relation_scope.txt` | `category`, `relations_json`, `taxonomy_context_json`, `taxonomy_decisions_json` |
| `critic_relations.txt` | `category`, `relations_json`, `relations_menu_json`, `previously_minted_json`, `taxonomy_context_json`, `taxonomy_decisions_json` |

When adding a production prompt or changing its caller fields, update the
checker contract and its tests in the same patch. Extra custom prompt files
need an explicit contract before this checker accepts them.

## Shared relation configuration

The starter has BFO categorization and a class-free RO property supplier.
Its `relation_defaults: "../_shared/bfo_ro_relations.yaml"` imports 61 generic
BFO/RO entries, their groups, critic-menu flags, and mereology rules.
Pre-Salt and new domains reference the shared registry. Pre-Salt adds 10
geology-specific entries, producing an ordered 71-entry registry.

The shared fragment is safe-loaded YAML, not a custom YAML include tag.
It can contain only `metatype_groups`, `relations`, and
`property_specializations`. Nested imports are rejected. Paths resolve
relative to the domain configuration.

Local groups and relations replace entire same-name entries; other inherited
entries and their order are retained. A local specialization list replaces
the inherited list, including `[]`. Omit the key to inherit specialization.
Provenance-tier filtering applies after merging. New constraints still require
explicit provenance and the existing configuration parity/audit discipline.

Export binds the ontologies declared in the active configuration, and
verification uses its project namespace and upper-ontology prefixes. A
BFO-grounded starter does not need dummy geology ontologies or prefixes.

## Offline checks

```bash
python -m pytest -q test/test_new_domain.py
python -m unittest discover -s test -p 'test_shared_relation*.py' -v
python test/test_ontology_config_parity.py
python test/test_prompt_refactor_parity.py
```

The new-domain tests create isolated temporary folders and exercise generation,
failure cleanup, JSON examples, shared defaults, relation validation, export,
and syntax/structure verification with network access denied. The parity
scripts protect the existing Pre-Salt configuration and saved prompt behavior.
None of these checks establishes geological correctness or exercises a live
model, OOPS!, or Java reasoning.
