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

## Code Conventions

### CSV I/O
- All CSV reads MUST use `encoding="utf-8-sig"` for BOM-safe interoperability.
- All CSV writes MUST use `encoding="utf-8-sig"`.
- Use `pandas.read_csv()` / `DataFrame.to_csv()` — never raw `csv` module.

### Environment Variables
- Required variables: access via `os.environ["VAR"]` and validate with `RuntimeError` if missing.
- Optional variables: access via `os.environ.get("VAR", default)` with sensible defaults.
- Naming: `UPPER_SNAKE_CASE`. LLM-related vars start with `LLM_` (e.g., `LLM_GENERATION_MODEL`).

### LLM Interaction
- All generation calls go through `src/utils/gemini_client.generate()` — never call the API directly.
- Always pass `response_mime_type="application/json"` for structured outputs.
- Default temperature: `0.0` (deterministic). Any deviation must be justified and documented.
- Default model: `gemini-2.5-pro` for all LLM tasks (extraction, generation, categorization).
- System instructions always define an expert persona (e.g., "You are a senior geoscientist and ontology engineer...").
- Batch inputs: `json.dumps(batch_items, indent=2)`. Validate response array length matches input batch size.

### Error Handling
- `RuntimeError` for fatal configuration errors (missing env vars, missing definition files).
- `ValueError` for data validation failures (batch size mismatch, unexpected format).
- JSON parse failures: log warning, store raw response, set `Context_Used = False`, continue processing. Never crash on a single term failure.
- Categorization errors: use `"ERROR_PARSE"`, `"ERROR_INVALID_JSON"`, `"ERROR_GENERAL"` as Category values — these are filtered out by downstream steps.

### Checkpoint / Resume
- Use `_load_checkpoint(path) → (completed_terms: set, rows: list[dict])` pattern.
- Atomic single-row append: `_append_row(path, row, is_first)` with `header=not os.path.exists(path)`.
- Resume logic: skip terms already in `completed_terms`, append new results, overwrite with full DataFrame at end.

### Imports
- Always absolute imports from `src/` (e.g., `from src.utils import log`). Never relative imports.
- `src/utils/` for shared infrastructure, `src/modules/` for pipeline steps, `src/evaluation/` for analysis.

### Logging
- Use `from src.utils import log` — never `print()` in modules (only `print()` in ablation_study.py for progress, which predates the logger).
- Levels: `log.banner(step, title)` for step headers, `log.info()` for status, `log.success()` for completions, `log.warn()` for recoverable issues, `log.error()` for failures, `log.detail()` for verbose info.

### Progress Bars
- Use `tqdm` for any loop processing >10 items. Use `desc=` for context, `set_postfix_str()` for current item.
- Write error messages via `tqdm.write("")` (not `print()`) to avoid corrupting the progress bar.

---

## Ontology & Taxonomy Rules

- UPPER_IRIS lookups must be case-insensitive (use `_UPPER_IRIS_LOWER` dict).
- OWL IRI generation must normalise to lowercase before CamelCase conversion to prevent case-collision duplicates.
- Self-referential `rdfs:subClassOf` triples (term IRI = parent IRI) must be detected and suppressed.
- Upper→upper triple suppression: never emit triples where both subject and object resolve to upper-level IRIs. The pipeline only creates triples where at least one side is a `presalt:` entity — it does not alter published upper ontologies.
- Taxonomy outputs must pass cycle detection before being written. Cyclic terms are re-parented to category root.
- Class vs. individual distinction: named entities (fields, basins, formations, time periods) → `rdf:type`; generic types/kinds → `rdfs:subClassOf`.
- Waterfall priority: GeoReservoir → GeoCore → BFO → NOT_CLASSIFIED. Always classify at the most specific level.

---

## Ablation Study Rules

- All 4 conditions (A/B/C/D) MUST use the **same term set** for fair comparison.
- Condition C must send `{"term": ..., "nld": ""}` (field present but empty string — not omitted).
- Condition D uses `{"term": ..., "context": ...}` (not `"nld"`), and the categorizer prompt switches to a raw-RAG variant via `is_raw_rag=True`.
- Checkpoint files: `output/ablation/nld_{A|B|C|D}.csv` and `cat_{A|B|C|D}.csv`.
- Merged output: `output/ablation/ablation_merged.csv` with a `Condition` column.

---

## Evaluation Rules

### Expert Evaluation (Layer 2)
- **5 sheets** per workbook: Instructions, Term_Relevance, NLD_Quality, Category_Correct, Taxonomy_Correct.
- **200 terms**, seed `42` for all randomised operations (sampling, shuffling, blinding).
- Blinding: experts never see condition labels. Blinding key is a separate CSV (`Row_ID → Condition/Term`).
- Stratified category evaluation by ontology tier: GeoReservoir → full binary validation with description; GeoCore/BFO → simplified 3-way.
- Taxonomy: ~80 parent-child IS-A pairs stratified by category.

### Statistical Tests (Layer 2 Analysis)
- Friedman omnibus test gates all post-hoc pairwise comparisons. Only run Wilcoxon signed-rank if Friedman is significant.
- ICC(2,1) for inter-rater reliability on continuous scales.
- Fleiss' kappa for multi-rater agreement on categorical judgments.
- Always report effect sizes alongside p-values.

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
| 7 | `7_ontology.ttl` | OWL Turtle format — loadable in Protégé |
| 7b | `7b_verification_report.json` | `layers.syntax.status`, `layers.structure.{classes,individuals,issues}`, `layers.oops_pitfalls`, `overall_status` |

Renaming or removing any of these columns is a **breaking change** that requires updating all downstream consumers.
