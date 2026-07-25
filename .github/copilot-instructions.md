# Copilot Instructions — PreSaltOntoLearn

## Project Identity

- **What:** An LLM-driven ontology learning pipeline for Brazilian Pre-Salt petroleum geology (master's thesis).
- **Core claim:** RAG-augmented Aristotelian NLDs improve upper-ontology classification over term-only, parametric-NLD, and raw-RAG baselines. This replicates and extends Lopes Junior (2024) on a new domain with a new architecture.
- **Pipeline:** 7 steps — PDF → Markdown → RAG Index → Extract → Aggregate → Filter → NLD → Categorize → Taxonomy → Relations → Critic → OWL → Verify.
- **Evaluation:** 4-condition ablation (A/B/C/D) + 2-layer analysis (automated + 3-expert blinded workbook).
- **OOPS! verification** uses a local Docker container (`mpovedavillalon/oops:v1`) to avoid dependence on the remote API. Override with `OOPS_URL` env var.

---

## Documentation Sync Rule

**Every code change that modifies behaviour, adds/removes features, changes defaults, or fixes bugs MUST be accompanied by documentation updates.** This applies to all modules in `src/` and `pipeline.py`.

### Which docs to update

| What changed | Update these files |
|---|---|
| Module behaviour (Steps 0-7) | `docs/CONTEXT.md` (module description), `docs/OVERVIEW.md` (pipeline table) |
| CLI flags or workflow | `README.md` (CLI table, ablation workflow) |
| Evaluation method (ablation, expert eval, analysis) | `docs/OVERVIEW.md` (evaluation section), `docs/CONTEXT.md` (evaluation architecture), `README.md` (project structure) |
| New robustness guard or validation | `docs/CONTEXT.md` (module's bullet list) |
| Environment variables | `README.md` (configuration section), `.env.example` if it exists |
| Project structure (new/renamed files) | `README.md` (project structure tree), `docs/CONTEXT.md` (directory structure) |

### Documentation locations

- **`README.md`** — Technical quick-start, CLI reference, project structure tree. Audience: developers running the code.
- **`docs/OVERVIEW.md`** — Plain-language project summary, pipeline table, evaluation framework. Audience: supervisors, committee members, collaborators.
- **`docs/CONTEXT.md`** — Full architecture reference, per-module descriptions, Mermaid diagram, evaluation architecture. Audience: AI assistants and developers understanding the codebase.

### Style rules

- Keep descriptions factual and concise — no marketing language.
- Use present tense ("detects cycles", not "will detect cycles").
- Module descriptions in `CONTEXT.md` use bullet-point lists under `### module_name.py — Step N: Name` headers.
- `OVERVIEW.md` pipeline table uses `| **N — Name** | Plain-language description |` format.
- When adding robustness guards, add a `**Robustness:**` or `**Robustness guards:**` bullet under the relevant module in `CONTEXT.md`.

---

## Configuration — `ontology_config.yaml`

**`ontology_config.yaml` at the repo root is the single source of truth** for upper-ontology metadata (BFO, GeoCore, GeoReservoir, RO), BFO disjoint pairs, the 71 relation property constraints, and the critic-driven `validate` step's class budget. It is loaded once at import time by `src/utils/ontology_config.py` (frozen dataclass + `lru_cache`-backed singleton).

- **Never hardcode** category lists, IRIs, metatype mappings, prefixes, BFO definitions, property constraints, or disjoint pairs in any module. Source them from `get_config()` instead.
- The deleted `resources/{bfo,geocore,georeservoir}-definitions.txt` files are GONE — never reintroduce them. Use `cfg.llm_definitions_block(ontology_key)` for the formatted text block passed to LLM prompts.
- **Relation property constraints** are in `relations:` (71 entries, each with `provenance ∈ {owl_axiom, bfo_shape_axiom, ro_release, critic_minted}`). `relation_validator.PROPERTY_CONSTRAINTS` is just `get_config().property_constraints()`.
- **Critic menu**: the validate-step relation critic is offered only the relations flagged `critic_menu: true` (21 for Pre-Salt — the corpus-attested properties plus the generic rewrite targets and the realizable carriers `has_role`/`has_function`/`has_disposition`; mereology only in generic `has_part`/`part_of` form). Never hardcode this list in Python; `critic._build_relations_menu()` reads the flag. The direction matters (restrictions are subject-anchored), so pick the subject-side property of each inverse pair.
- **Mereology specialization** (`has_part` → `has_continuant_part`/`has_occurrent_part`) lives in `relation_validator.specialize_property`/`normalize_property` — the single source of truth used by BOTH the extract step and the validate step's post-critic normalisation pass. Never duplicate the rule logic; it is driven by `property_specializations:` in the YAML.
- **Metatype groups** (`CONTINUANT`, `OCCURRENT`, `MATERIAL`, …) in `metatype_groups:` are recursive shorthand for `relations.*.domain/range`. Add new groups there, not in code.
- **Adding a new property constraint** requires: (1) entry in `relations:` with explicit `provenance`, (2) re-running `test/test_ontology_config_parity.py` (24 checks), (3) regenerating the audit CSV via `python -m src.evaluation.property_constraints_audit`.
- **Adding a new upper-ontology class** requires: (1) entry under the right `ontologies.<key>.classes:` list, (2) updating consumers' parity assertions if the count is hardcoded in a test, (3) re-running parity.

### Env overrides (test-friendly switches)
- `ONTOLOGY_CONFIG_PATH` — point loader at a fixture YAML (used by `test/` only)
- `RELATION_PROVENANCE_TIERS` — comma-separated subset of the 4 tiers; restricts which relations are active

### Parity discipline
- After ANY edit to `ontology_config.yaml` or `src/utils/ontology_config.py`, run `python test/test_ontology_config_parity.py`. Must show `=== PARITY PASSED ===` before committing.
- If a check fails, fix the YAML or loader — do not edit the assertion to make it pass.

---

## Code Conventions

### CSV I/O
- All CSV reads MUST use `encoding="utf-8-sig"` for BOM-safe interoperability.
- All CSV writes MUST use `encoding="utf-8-sig"`.
- Use `pandas.read_csv()` / `DataFrame.to_csv()` — never raw `csv` module.
- **Prefer the wrappers** in `src/utils/csv_io.py`: `read_csv(path)` and `write_csv(df, path)`. They default to `encoding="utf-8-sig"` and `index=False` for writes, so call sites cannot drift from the contract. Raw `pd.read_csv` / `df.to_csv` are acceptable only when a non-default option is genuinely needed (and the encoding must still be passed explicitly).

### Environment Variables
- Required variables: access via `os.environ["VAR"]` and validate with `RuntimeError` if missing.
- Optional variables: access via `os.environ.get("VAR", default)` with sensible defaults.
- Naming: `UPPER_SNAKE_CASE`. LLM-related vars start with `LLM_` (e.g., `LLM_GENERATION_MODEL`).

### LLM Interaction
- All generation calls go through `src/utils/gemini_client.generate()` — never call the API directly.
- Always pass `response_mime_type="application/json"` for structured outputs.
- Default temperature: `0.0` (deterministic). Any deviation must be justified and documented.
- Default model: `gemini-2.5-pro` for all LLM tasks (extraction, generation, categorization).
- System instructions always define an expert persona via the `<<persona>>` placeholder, which `prompt_loader` interpolates from the active `domain_profile.yaml`. Never embed the persona literal in the prompt file or in code.
- Batch inputs: `json.dumps(batch_items, indent=2)`. Validate response array length matches input batch size.

### Error Handling
- `RuntimeError` for fatal configuration errors (missing env vars, missing definition files).
- `ValueError` for data validation failures (batch size mismatch, unexpected format).
- JSON parse failures: log warning, store raw response, set `Context_Used = False`, continue processing. Never crash on a single term failure.
- Categorization errors: use `"ERROR_PARSE"`, `"ERROR_INVALID_JSON"`, `"ERROR_GENERAL"` as Category values — these are filtered out by downstream steps.

### Checkpoint / Resume
- Prefer `src/utils/checkpoint.py` `Checkpoint(path, key_column="Term")` for all resumable per-term step outputs. API:
  - `load() → (completed_terms: set, rows: list[dict])` — reads existing CSV, returns the set of already-completed key values plus the loaded rows so the caller can resume.
  - `append(row, is_first=None)` — atomic single-row append; writes the header only when the file is new (`is_first` defaults to `not path.exists()`).
  - `append_batch(rows, is_first=None)` — same contract for a batch.
- The legacy `_load_checkpoint` / `_append_row` pattern (`header=not os.path.exists(path)`) is the contract the wrapper codifies; new code should use the wrapper.
- Resume logic: skip terms already in `completed_terms`, append new results, and write the full DataFrame once at the end so the file is internally consistent if the run is killed.

### Imports
- Always absolute imports from `src/` (e.g., `from src.utils import log`). Never relative imports.
- `src/utils/` for shared infrastructure, `src/modules/` for pipeline steps, `src/evaluation/` for pipeline-internal audits. Thesis-study code belongs under `evaluation_study/`.

### Logging
- Use `from src.utils import log` — never `print()` in production modules.
- Levels: `log.banner(step, title)` for step headers, `log.info()` for status, `log.success()` for completions, `log.warn()` for recoverable issues, `log.error()` for failures, `log.detail()` for verbose info.

### Progress Bars
- Use `tqdm` for any loop processing >10 items. Use `desc=` for context, `set_postfix_str()` for current item.
- Write error messages via `tqdm.write("")` (not `print()`) to avoid corrupting the progress bar.

---

## Ontology & Taxonomy Rules

- UPPER_IRIS lookups must be case-insensitive (use `_UPPER_IRIS_LOWER` dict).
- OWL IRI generation must normalise to lowercase before CamelCase conversion to prevent case-collision duplicates.
- `rdfs:label` for presalt classes and individuals is Title-Cased to the GeoCore/GeoReservoir house style (minor words lower-cased, acronyms preserved); object-property labels stay lower-case. Every property used in a restriction/companion axiom is declared and labelled so it is readable in Protégé without resolving imports.
- Self-referential `rdfs:subClassOf` triples (term IRI = parent IRI) must be detected and suppressed.
- Upper→upper triple suppression: never emit triples where both subject and object resolve to upper-level IRIs. The pipeline only creates triples where at least one side is a `presalt:` entity — it does not alter published upper ontologies.
- Taxonomy outputs must pass cycle detection before being written. Cyclic terms are re-parented to category root.
- Class vs. individual distinction: named entities (fields, basins, formations, time periods) → `rdf:type`; generic types/kinds → `rdfs:subClassOf`.
- Waterfall priority: GeoReservoir → GeoCore → BFO → NOT_CLASSIFIED. Always classify at the most specific level.

---

## Reproducibility Rules

- LLM temperature = 0.0 unless explicitly justified. Document any deviation.
- Random seed = 42 for all sampling, shuffling, and blinding operations.
- RAG configuration is fixed: chunk_size=1024, search_k=20, rerank_k=5 (arbitrarily set — the thesis focus is on NLD/RAG ablation, not RAG hyperparameter optimisation).
- Embedding model: `BAAI/bge-m3` (local, fixed weights — chosen over API models for reproducibility).
- Reranker: `BAAI/bge-reranker-v2-m3` (cross-encoder).
- All model choices and hyperparameters are documented in `docs/thesis_model_selection.md`.
- Never use API-based embeddings (e.g., `text-embedding-004`) — weights may change silently, breaking reproducibility.

---

## Thesis Writing Assistance

When helping write or review thesis text:
- **Terminology:** "PreSaltOntoLearn" is the system name. "Pre-Salt" is always hyphenated. "NLD" = Natural Language Definition. "RAG" = Retrieval-Augmented Generation.
- **Citation anchors:** The core theoretical grounding is Lopes Junior (2024) — NLDs outperform other representations for BFO classification. Always cite when discussing NLD design choice.
- **Ablation framing:** A vs B = RAG contribution, A vs C = NLD contribution, A vs D = structuring benefit. These map to Lopes Junior's Study Case 1 comparisons.
- **Methodology vocabulary:** Use "waterfall priority" for classification cascade, "Aristotelian form" for "X is a Y that Z", "proximate genus" for Y, "differentia" for Z.
- **Upper ontology references:** BFO (Basic Formal Ontology, Smith et al. 2015), GeoCore (Abel et al. 2015), GeoReservoir (Abel et al. — extension of GeoCore for petroleum).
- **Evaluation methodology references:** NeOn methodology (Suárez-Figueroa et al. 2012) for separating domain vs. formal evaluation; OntoClean (Guarino & Welty 2009) for meta-property validation.
- **Statistical reporting:** Always report test statistic, degrees of freedom, p-value, and effect size (e.g., "χ²(3) = 12.4, p = .006, W = 0.31").
- **Figures:** When generating pipeline diagrams, use the Mermaid syntax from `docs/CONTEXT.md` as the canonical source.

---

## Data Flow Contract (CSV Columns Between Steps)

| Step | Output File | Required Columns |
|---|---|---|
| 1 | `1_raw_llm_extraction.json` | `Entity`, `Source_Paper` |
| 2 | `2_aggregated_counts.csv` | `Readable_Term`, `Frequency` |
| 3 | `3_filtered_top_terms.csv` | `Readable_Term`, `Frequency` |
| 4 | `4_nld_generated_definitions.csv` | `Term`, `NLD`, `Context_Used` (bool), `Context` |
| 5 | `5_categorized_ontology.csv` | `Term`, `Category`, `Reasoning`, `NLD`, `RAG_Context_Used` |
| 6 | `6_taxonomy.csv` | `Term`, `Parent_Term`, `Relationship_Type`, `Category`, `Is_Intermediate`, `NLD`, `FALLBACK` |
| 6b | `6b_relations.csv` | `Term`, `Category`, `Property`, `Property_IRI`, `Filler`, `Filler_Source`, `Confidence`, `Evidence`, `Validation_Status`, `Validation_Reason` |
| 6c | `6c_taxonomy_cleaned.csv` | Same columns as Step 6 (cleaned by ontology critic) |
| 6c | `6c_relations_cleaned.csv` | Same columns as Step 6b (cleaned by ontology critic) |
| 6c | `6c_critic_log.csv` | `Action`, `Term`, `Detail`, `Reason` |
| 6d | `6d_taxonomy_reclassified.csv` | Same columns as Step 6 (categories corrected by relation evidence) |
| 6d | `6d_reclassification_log.csv` | `Action`, `Term`, `Old_Category`, `New_Category`, `Detail`, `Evidence_Count` |
| 7 | `7_ontology.ttl` | OWL Turtle format — loadable in Protégé |
| 7b | `7b_verification_report.json` | `layers.syntax.status`, `layers.structure.{classes,individuals,issues}`, `layers.oops_pitfalls`, `overall_status` |

Renaming or removing any of these columns is a **breaking change** that requires updating all downstream consumers.
