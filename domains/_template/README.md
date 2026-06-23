# Domain Template

This directory is a **scaffold source** for new domains, not a runnable domain
on its own. Materialize it with:

```powershell
python scripts/new_domain.py <your_domain_name>
```

That script copies this folder to `domains/<your_domain_name>/`, plus the
generic prompts and upper-ontology resources from `domains/presalt/`. You
then edit four files and you're done. See `SETUP.md` at the repo root for
the full walk-through.

## What lives here

| File | Required to edit? | Why |
|---|---|---|
| `prompt_blocks.yaml` | **Yes** — sections A & C | Domain identity (Section A) + example blocks (Section C). Personas (Section B) are auto-derived; rarely touched. |
| `ontology_config.yaml` | **Yes** — project metadata | Project name, namespace, prefix. Defaults waterfall to BFO only. Add `geocore`/`georeservoir`/custom ontologies when ready. |
| `competency_questions.txt` | **Yes** | 5-10 questions your ontology should answer. Used by Step 5b (`--refine`). |
| `domain_filters.yaml` | Optional | Engineering cleanups (deduplication, orphan removal). Defaults to 6 cheap structural filters; LLM-based filters opt-in. |

Everything else (`prompts/`, `resources/`) is copied from `domains/presalt/`
by the scaffold script and is **domain-agnostic** — you only touch those if
you want to tune wording or add custom upper ontologies.
