# Critic comparison artifacts — context for the other agent

This folder lets you **re-run, extend, or compare the ontology critic WITHOUT
running the full pipeline from scratch** (no PDF parse, no RAG, no extraction,
no categorisation, no taxonomy build). Everything below starts from two cached
input files.

Generated: 2026-06-25, branch `simplified-critic` (commit with the three-stage
critic: `efe4399 feat(critic): three-stage validate critic, OntoClean probes`).

---

## 1. What the `iteracao_*.ttl` files are (repo root)

All three are the SAME ontology at three critic generations, built from the
**same Jun-7 construct inputs**. Only the critic (and a little emit polish)
differs, so diffs between them isolate critic behaviour.

| File | Built | Critic | presalt classes | individuals | triples |
|---|---|---|---|---|---|
| `iteracao_1.ttl` | Jun 11 | early single-pass | 331 | 26 | 3883 |
| `iteracao_2.ttl` | Jun 11 | older single-pass (baseline) | 352 | 55 | 3778 |
| `iteracao_3.ttl` | Jun 25 | **new 3-stage** (this run) | 300 | 66 | 3207 |

"3-stage" = per category: (1) chunked per-term taxonomy critic with OntoClean
probes, (2) one cross-term dedup pass over survivors, (3) relation critic. See
`src/modules/validate/critic.py` module docstring for the full description.

---

## 2. The reusable inputs (the "don't run from scratch" part)

| File | Rows | What it is |
|---|---|---|
| `construct_taxonomy.csv` | 492 | The IS-A tree + NLDs the critic judges. Columns: `Term, Parent_Term, Relationship_Type, Category, Is_Intermediate, NLD, FALLBACK, Category_IRI, Parent_IRI`. |
| `construct_relations.csv` | 1431 | Object-property triples the critic judges. Columns: `Term, Category, Property, Property_IRI, Filler, Filler_Source, Confidence, Evidence, Validation_Status, Validation_Reason`. (859 are `ACCEPTED`; the critic only sees those.) |

These are the output of pipeline **Step "construct"** (taxonomy_builder +
relation_extractor). They are dated Jun 7 and have NOT changed — `iteracao_2`
and `iteracao_3` were both built from them, which is what makes the comparison
fair. In the live repo they live under `output/refined/` (which is gitignored),
so they are copied here to be shareable.

---

## 3. The new critic's outputs for iteracao_3 (audit trail)

| File | Rows | What it is |
|---|---|---|
| `iter3_validate_taxonomy.csv` | 320 | Cleaned taxonomy after the critic (492 → 320). |
| `iter3_validate_relations.csv` | 445 | Cleaned relations (859 ACCEPTED → 445; 18 dropped by post-critic BFO re-validation). |
| `iter3_validate_instances.csv` | 63 | Terms converted to `owl:NamedIndividual` (proper nouns: formations, fields, ages…). Columns include `Term, Target_Class, Mint_Parent, …`. |
| `iter3_validate_edits.csv` | 1545 | **Full per-decision audit log.** Taxonomy rows carry the probe trace (`probe1_genus_ok, probe2_bucket, probe3_rewrite, carried_by`). This is the file to read to understand WHY the critic did what it did. |

(No `validate_minted_properties.csv` — the critic minted **zero** new
properties this run, because the config-driven `critic_menu` covers the corpus.)

### iteracao_3 verdict tally (from `iter3_validate_edits.csv`)
```
KEEP=856  DROP=414  REPARENT=68  CONVERT_TO_INSTANCE=63
DROP_AS_MIXIN=55  DROP_AS_REDUNDANT=54  FIX=35
```

---

## 4. iteracao_2 → iteracao_3 headline diff

- **Taxonomy is leaner:** 352 → 300 presalt classes. 76 iter2 classes are gone
  (mostly quality/role mixins and near-synonyms the new critic drops, e.g.
  `AlkalineLacustrineEnvironment`, `DeepWater`, `Boundstone`, `Lithotype`,
  `LacustrineCarbonateSystem`), and 24 new ones appear (e.g. `PetroleumSystem`,
  `HydrocarbonAccumulation`, `StructuralHigh/Low`, `Laminite`, `Microbialite`).
- **More individuals:** 55 → 66. The new critic converts 11 more proper nouns to
  instances (e.g. `TupiField`, `GondwanaSupercontinent`, `Alagoas`,
  `PreSaltSuccession`) instead of leaving them as classes.
- **Fewer restrictions/triples:** the relation re-validation + dedup trims noise.

Interpretation: iteracao_3 trades raw size for cleaner BFO discipline — fewer
mixin/synonym classes, more correct individual-vs-class calls.

---

## 5. How to reproduce or extend (the two commands)

From the repo root, with the env configured (`.env` with `GEMINI_API_KEY` +
`VERTEX_AI`), and `$env:PYTHONPATH="."`:

**Step A — run the critic** (this is the ~35-min, paid LLM step):
```powershell
python -m src.modules.validate.critic `
  comparison_artifacts/construct_taxonomy.csv `
  <some_output_dir> `
  --relations comparison_artifacts/construct_relations.csv
```
Writes `validate_taxonomy.csv`, `validate_relations.csv`, `validate_edits.csv`,
`validate_instances.csv` (and `validate_minted_properties.csv` if anything is
minted) into `<some_output_dir>`.

**Step B — emit the TTL** (free, pure Python):
```powershell
python -m src.modules.emit.owl_exporter `
  <some_output_dir>/validate_taxonomy.csv `
  --nld comparison_artifacts/construct_taxonomy.csv `
  --relations <some_output_dir>/validate_relations.csv `
  --instances <some_output_dir>/validate_instances.csv `
  --minted <some_output_dir>/validate_minted_properties.csv `
  --output iteracao_4.ttl
```
(`--minted` is optional if no properties were minted. NLDs come from the
original `construct_taxonomy.csv` via `--nld`.)

To **only re-emit iteracao_3 without re-running the critic**, point Step B at
the `iter3_validate_*.csv` files in this folder.

---

## 6. Caveats for a fair comparison

- iteracao_3 = new critic **+ new emit** (the same commit polished emit labels).
  So a small part of any diff is emit, not critic. For "is the critic better
  end-to-end?" this is fine; for critic-only isolation you'd re-emit iter2's
  validate CSVs with the new emit.
- The critic is `temperature=0` but Gemini is not perfectly deterministic; a
  re-run may differ by a few rows.
- 429 (rate-limit) warnings during the run are auto-retried and harmless.
