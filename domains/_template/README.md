# Domain starter

Create a domain from the repository root:

```bash
python scripts/new_domain.py your_domain
```

The generator stages and checks the complete folder before reporting success.
Existing domains are never intentionally overwritten. Missing assets, unresolved
prompt blocks, incompatible example outputs, and invalid configuration stop
generation without leaving a partial domain folder.

## Edit two files

- **`prompt_blocks.yaml`:** replace `domain.name` and `domain.scope`, then adapt
  the examples and questions to your subject. Personas derive from the domain
  fields. Keep `domain.upper_stack` and `domain.waterfall` aligned with your
  ontology configuration.
- **`ontology_config.yaml`:** set the project name, namespace, and prefix.
  The starter uses BFO categories and inherits 61 generic BFO/RO relation
  entries from `../_shared/bfo_ro_relations.yaml`. Add only domain-specific
  properties locally; a same-name entry replaces the entire inherited entry.

The runtime reads questions from `examples.cq_questions` in
`prompt_blocks.yaml`. Use unique identifiers from CQ1 through CQ10 and keep
`examples.cq_identifiers` aligned. There is no separate question text file
to edit. Scoring outputs contain `term`, `relevant_cqs`, and `reasoning`,
not numerical question scores.

JSON examples inserted in prompt bodies need doubled braces because callers
use `str.format()`. The `examples.cq_scoring` block is inserted in the system
instruction and can use ordinary JSON braces. Preserve the demonstrated output
fields and types when replacing the example terms.

The relation-property table is generated at prompt load time from the active
constraints flagged `critic_menu: true`; do not maintain a second property list
in the prompt text. Shared specialization rules refine generic part relations
according to the configured subject/filler types.

## Check before using the domain

```bash
python -m src.utils.domain_validation domains/your_domain
```

This checks resources, all current production prompts, required text blocks,
caller placeholders, question identifiers, and example JSON shapes. It makes
no model calls and does not validate scientific meaning.

The supplied examples are neutral demonstrations, not a reviewed ontology.
Customize them before a live run. The production critic uses ontology
configuration and critic prompts.
Keep `domains/_shared/` beside your domain when moving it.

Set `ONTOLOGY_CONFIG_PATH=domains/your_domain/ontology_config.yaml` in `.env`.
See [SETUP.md](../../SETUP.md) for required model names, isolated input/output
paths, the single-document frequency threshold, and verification limitations.
