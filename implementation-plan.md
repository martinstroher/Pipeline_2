# Plan: RAG-Grounded Ontology Learning Pipeline (PreSaltOntoLearn)

## Context

### The Problem
The codebase has 5 blocking bugs that cause 100% NLD generation failure in production. Steps 1-3 outputs exist and are reusable (769 terms). The test pipeline proves E2E works — the production failure is a simple BM25 wiring bug. You need a working pipeline producing a thesis-worthy ontology artifact within 2 weeks.

### Positioning Against NeOn-GPT
NeOn-GPT already does full end-to-end ontology generation with LLMs (multi-domain, verification loops, OWL export). Competing on completeness is a losing game.

Their critical weakness: **no corpus grounding**. For specialized domains like Environmental Microbiology, they captured only **3.7% of gold-standard classes**. Pre-Salt geology is exactly this type of underrepresented domain.

**Thesis contribution: "We developed a RAG-augmented LLM pipeline that generates domain-specific Natural Language Definitions to assist expert ontology construction for underrepresented domains. We demonstrate that RAG grounding produces more accurate definitions than parametric-only approaches (ablation evidence), and deliver a Pre-Salt petroleum geology ontology as a domain artifact."**

### What We're Cutting (and Why)

| Cut | Why | Time Saved |
|-----|-----|------------|
| **Relation extraction** | Thesis claim is about RAG quality, not ontology completeness. Many published OL papers stop at taxonomy. NeOn-GPT already does relations better. Explicit future work. | 3-5 days |
| **RAG grid search re-run** | Existing golden answers were LLM-synthesized (not expert-verified), so RAGAS scores are untrustworthy regardless of JSON bug fix. The 4 completed configs show a spread of only 0.018 — parameters don't matter much. Use literature-justified defaults instead (chunk=1024, K=20, rerank=5 — standard for BGE-M3/cross-encoder setups). Cite BGE-M3 paper + cross-encoder best practices. The ablation study (RAG vs no-RAG) is the real proof, not parameter tuning. | 1 day |
| **Cosmetic bug fixes** | `load_dotenv()` placement, `import json` in loop, `pdf_processor` sys.argv, `run_e2e_test` path — none block production | 0.5 day |
| **Complex OWL axiomatization** | Minimal OWL (classes + subClassOf + NLD annotations) is sufficient. NeOn-GPT's verification loops are their contribution, not ours. | 1-2 days |
| **requirements.txt / .env.example** | Nice-to-have, 5 minutes at the end, not a thesis contribution | — |

### RAG Parameters: Literature-Justified Defaults (No Grid Search)

Instead of optimizing RAG hyperparameters empirically (with questionable ground truth), we adopt fixed, literature-justified defaults and cite accordingly:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Chunk size | 1024 tokens | Standard for dense retrieval with BGE-M3; balances context richness vs noise |
| Chunk overlap | 100 tokens | ~10% overlap prevents context fragmentation at boundaries |
| Search K (initial retrieval) | 20 | Standard top-K for re-ranking pipelines; provides enough candidates for cross-encoder |
| Rerank K (final) | 5 | Cross-encoder re-ranking to top-5 is standard practice (Nogueira et al., 2020) |
| Embedding model | BGE-M3 | Multilingual, supports dense+sparse in single model (~560M params, runs on consumer GPU) |
| Reranker | BGE-Reranker-v2-M3 | Matched to embedding model family; cross-encoder architecture |

**In the thesis:** Describe these as "informed by retrieval-augmented generation best practices" with citations. Acknowledge that per-domain tuning could improve results (future work). The ablation study is what proves RAG matters — not that these specific parameters are optimal.

---

## Plan: 3 Phases, ~7 Working Days

### Phase 1: Fix Blocking Bugs + Production Run (Days 1-2)

**Day 1 — Fix blocking bugs + prep:**

**Bug 1: BM25 not wired through pipeline (DONE — IN CODE)**
- `pipeline.py:15,23` ✅ passes `vector_store, bm25` to `run_nld_generation()`.
- `nld_generator.py:89` ✅ accepts `vector_store=None, bm25_retriever=None` and passes them to `get_relevant_documents()`.
- Lines 124-131 have a fallback `load_vector_store()` with a warning — acceptable safety net for standalone usage.

**Bug 2: ChromaDB destroyed every run (DONE — IN CODE)**
- `rag_setup.py:170` ✅ `setup_rag()` already has `force_rebuild` param. Lines 185-198 handle cache reuse: loads ChromaDB from disk, rebuilds BM25 in-memory.

**Bug 3: 60-second sleep per term (DONE — IN CODE)**
- `nld_generator.py:174` ✅ already defaults to `4` seconds.

**Bug 4: Categorizer error handler crashes (PARTIAL — IN CODE)**
- `term_categorizer.py:143-152` ✅ JSON error handler now uses correct column names.
- **Remaining:** `term_categorizer.py:153-154` generic `except Exception` prints error but silently drops the batch — no results appended. Add error-path append with `ERROR_GENERAL` category (same pattern as JSON handler at lines 144-152).

**Bug 5: Add checkpointing to NLD generator (DONE — IN CODE)**
- `nld_generator.py:110-119` ✅ loads existing results on startup (resume from checkpoint).
- `nld_generator.py:170-172` ✅ atomic append per term.
- Lines 187-188 write final consolidated output at the end.

**Bug 6: Bare `exit()` calls kill the ablation runner**
- `nld_generator.py` (lines 13, 16), `term_categorizer.py` (lines 15, 50, 168) — Replace bare `exit()` with `raise RuntimeError(...)` or `return`. The ablation runner needs to call NLD generation and categorization multiple times in one process; `exit()` would kill everything.
- Move module-level Gemini configuration inside the `run_*()` functions (currently at import time in nld_generator.py lines 7-16).

**Bug 7: Create `.env-test` for E2E test**
- `test/run_e2e_test.py:37` opens `.env-test` which **does not exist** — test crashes with `FileNotFoundError`. Create `test/.env-test` with all required env vars pointing to test paths.
- Also fix `test/run_e2e_test.py:108`: `os.path.getsize(f)` → `os.path.getsize(full_path)`.

**Prep: Add expert ground truth to repo**
- User adds the ~1200-row expert evaluation data (with category labels) to `inputs/` or `resources/`.
- Verify it has columns mappable to: Term, Expert_Category.

After fixes: verify with E2E test on the existing test data (small 9-term test set in `test/`).

**Day 2 — Prompt improvements + Full corpus setup:**

**Prompt fixes (must-do, high leverage):**
- **Add few-shot examples** to both NLD and categorizer prompts. Use 2-3 entries from `inputs/test_dataset.json` (e.g., Grainstone, Coquina, Ostracod) as worked examples. This is the single highest-impact prompt change — 15-25% consistency improvement per few-shot literature.
- **Fix retrieval query** from `"What is the definition of {term}?"` to just `"{term}"`. The question-form adds BM25 noise and biases dense retrieval toward definitional passages only. Term-only queries retrieve contextual passages too.
- **Add `"The definition MUST be written in English."`** to NLD prompt. Portuguese corpus may cause code-switching.
- **Fix NLD system instruction**: pass as `system_instruction=` parameter to `GenerativeModel()`, not as string concatenation. The categorizer already does this correctly (line 104).
- **Add polysemy instruction**: "If a term is polysemous, define the sense most relevant to Pre-Salt petroleum geology."
- **Review and refine** remaining prompts:
  - NLD generation prompt (Aristotelian form, use of RAG context)
  - Categorization prompt (BFO/GeoCore/GeoReservoir mapping, reasoning)
  - Later: taxonomy builder prompt (class vs individual distinction)

**Corpus setup:**
- The ~40 papers are already available. Copy into `inputs/` if not there already.
- Run PDF→Markdown conversion on the full corpus
- Build ChromaDB + BM25 index from full corpus (`force_rebuild=True`)
- Steps 1-3 outputs already exist (769 terms) — no need to re-run.

---

### Phase 2: Ablation + Evaluation + Winner Selection (Days 3-5)

**Day 3 — Run pipeline under 4 conditions (`src/evaluation/ablation_study.py`):**

This is THE central deliverable. The ablation IS the production run — all outputs are candidate ontologies, and the expert-selected winner becomes the final artifact.

#### How It Works

Run Steps 4-5 (NLD generation + categorization) under 4 conditions on the SAME 769 terms:

| Condition | NLD Generation | Categorizer Input | Simulates |
|-----------|---------------|-------------------|-----------|
| A: Full pipeline | RAG context → NLD | Term + NLD | Your approach |
| B: No RAG | `"No additional context available."` → NLD | Term + NLD | NeOn-GPT's parametric approach |
| C: No NLD | Skipped | Term + `"No definition available."` | Naive LLM baseline |
| D: Raw RAG | Skipped | Term + raw RAG chunks | "Why not skip NLDs?" counterargument |

Implementation details:
- Condition B: `generate_nld(term, context="No additional context available.")` — natural text avoids "missing data" cue from empty string
- Condition C: Same categorizer prompt template, NLD field set to `"No definition available."` — avoids rewriting the prompt (keeps it structurally identical, only NLD content changes)
- Condition D: Skip NLD generation. Feed term + `format_docs_for_context(retrieved_docs)` directly to categorizer with a prompt variant that says "classify based on the term and the provided corpus context" instead of "classify based on the NLD"

**Four key comparisons:**
| Comparison | What it proves |
|-----------|---------------|
| A vs B | RAG improves NLD quality → your core contribution |
| A vs C | NLDs improve categorization → addresses Reviewer 1 ("NLD necessity unproven") |
| B vs C | Even parametric NLDs help → isolates the NLD mechanism value |
| A vs D | NLDs compress RAG context better than raw chunks → justifies NLD as architectural choice |

Expected time: ~2h each for A and B (769 terms × ~8s NLD + categorization), ~1h each for C and D (categorization only). ~6h total.
**Note:** Cross-encoder reranking runs on CPU (`rag_setup.py:82` hardcodes `device: cpu`). This adds ~5-10s per term for the reranking step. Realistic total may be ~10-12h. Consider: if you have a GPU available, change the device config to `cuda` for a significant speedup.

#### Two Evaluation Layers (Automated + Expert)

**Layer 1: Automated cross-condition comparison (zero expert effort, all 769 terms)**

No ground truth available for the full 769-term set (the existing expert data covers only ~100 terms from the old pipeline, evaluated against old outputs). Instead, Layer 1 compares conditions against EACH OTHER:

- **Cross-condition agreement matrix**: For each pair of conditions (A/B, A/C, A/D, B/C, B/D, C/D), report % of terms that received the same category. Low agreement between A and B = RAG is changing categorization decisions.
- **Cochran's Q test**: Treats each term as a repeated measure across 4 conditions. Tests whether the proportion of terms in any given category differs significantly across conditions. This is the omnibus "do conditions differ at all?" test.
- **Category migration analysis** (30 min): For each term, track how its category changes across conditions. Report % stable, % RAG-sensitive (changes A↔B), % NLD-sensitive (changes A↔C), % compression-sensitive (changes A↔D). If RAG-sensitive terms disproportionately belong to GeoReservoir categories, that directly supports the thesis claim.
- **NOT_CLASSIFIED rate per condition** (5 min): If Condition A has lower NOT_CLASSIFIED rate, RAG enables more confident classification.
- **Category distribution shift**: Chi-squared test on the category frequency distributions across conditions. Are RAG conditions shifting terms toward more specific categories?
- **Subgroup analysis by `Context_Used` flag** (free, high-value): Within Condition A, split into {Context_Used=true} vs {Context_Used=false}. Compare agreement rates with Condition B for each subgroup. If Context_Used=true terms diverge more from B, that proves the RAG retrieval is the active ingredient.

**Note:** Layer 1 measures WHETHER conditions produce different outputs and HOW they differ. It does NOT measure which is CORRECT — that's Layer 2's job.

**Layer 2: Expert blinded evaluation (~2-3 hours per expert) — THE ACCURACY MEASURE**
Generate a spreadsheet with ~50-80 terms (stratified by category and `Context_Used` flag). For each term, experts see:
- Term name
- NLD-A vs NLD-B (blinded, randomly ordered)
- Category-A vs Category-B (blinded, randomly ordered)
- Expert tasks:
  1. **NLD preference:** Which definition is more accurate? (A / B / tie)
  2. **NLD quality:** Rate each NLD 1-5 for geological accuracy
  3. **Category correctness:** Which categorization is correct? (A / B / both / neither)
- Statistical tests:
  - Wilcoxon signed-rank test on NLD accuracy ratings
  - Sign test on NLD and category preferences
  - **Weighted kappa** (or Krippendorff's alpha) for inter-annotator agreement on ordinal 1-5 ratings (NOT Cohen's kappa — which treats "1 vs 2" the same as "1 vs 5")

**The winner becomes the ontology.** After expert evaluation, the best-performing condition is selected as the final pipeline output. That output feeds directly into taxonomy building (Step 6) and OWL export (Step 7). No separate "final production run" needed.

**Why this approach:**
1. No wasted computation — every run produces a potential deliverable
2. Evaluates BOTH NLDs and categorization (the full causal chain: RAG → better NLDs → better categories)
3. Isolates NLD contribution (A vs C), RAG contribution (A vs B), AND NLD-as-compression (A vs D) independently
4. No LLM-as-judge (avoids circularity)
5. Layer 1 alone is sufficient for the thesis if experts are slow

**Day 4 — Layer 1 analysis + expert evaluation prep:**

- Calculate Layer 1 automated analyses on ablation results (all 769 terms):
  - Cross-condition agreement matrix (6 pairwise agreement rates)
  - Cochran's Q omnibus test (confirm conditions produce different category distributions)
  - Chi-squared tests on category frequency distributions
  - Category migration analysis (stable / RAG-sensitive / NLD-sensitive / compression-sensitive terms)
  - NOT_CLASSIFIED rate comparison across conditions
  - Context_Used subgroup analysis (within Condition A vs Condition B agreement)
- Generate the expert evaluation spreadsheet (Layer 2) — blinded, randomized, includes both NLDs AND categories per term
- Send expert spreadsheet to 2-3 experts
- **Hallucination Catalog** (~2h): Select 10-15 terms where Condition B produced demonstrably incorrect NLDs. For each, present side-by-side: (a) parametric NLD (Condition B), (b) RAG-grounded NLD (Condition A), (c) the retrieved corpus passage that corrected the error, (d) expert judgment. This becomes the most memorable figure in the thesis.

**Day 5 — Expert results + winner selection:**

- Process expert evaluation results (if available) — compute Wilcoxon, sign test, weighted kappa
- Select winner (A, B, C, or D) based on combined Layer 1 + Layer 2 evidence
- Generate thesis-ready tables (ablation comparison, Context_Used subgroup, category migration, hallucination catalog)
- If experts are slow: Layer 1 alone is sufficient to justify selecting Condition A as default winner and proceed

**Day 6 — Taxonomy builder (`src/modules/taxonomy_builder.py`, new Step 6):**

**Important: this runs AFTER winner selection.** Categorization (Step 5) is flat: "Grainstone → ReservoirRock". Taxonomy building arranges terms WITHIN each category into is-a hierarchies: "Grainstone is-a CarbonateRock is-a SedimentaryRock is-a ReservoirRock".

- **Research task (1-2h):** Collect OWL IRIs from published BFO, GeoCore, and GeoReservoir ontologies and hardcode the upper-level hierarchy. The definition files in `resources/` have no IRIs or hierarchical structure — just flat text descriptions.
- Input: winner's categorized output (e.g., Condition A's `5_categorized_ontology.csv`)
- Group terms by BFO/GeoCore/GeoReservoir category
- For each group, prompt Gemini Pro with ALL terms in that group + their NLDs → arrange into is-a hierarchy (NOT one term at a time — needs the full group for tree construction)
- **Class vs individual distinction:** prompt explicitly requires identifying named individuals (e.g., "Cretaceous" → `rdf:type GeologicalTimeInterval`, NOT `subClassOf`)
- Output: `6_taxonomy.csv` with columns (Term, Parent_Term, Relationship_Type, Category)
- Top-level hierarchy (BFO→GeoCore→GeoReservoir) is hardcoded from published ontology IRIs
- **Review taxonomy prompt** before running (class vs individual distinction is critical)

**Note on complexity:** Taxonomy building is fundamentally harder than categorization (group reasoning, transitive consistency, depth calibration). Budget extra time for prompt engineering.

**Day 7 — OWL export (`src/modules/owl_exporter.py`, new Step 7) + documentation:**

Minimal but valid OWL using `rdflib` (~100-120 lines):
- Create ontology with proper namespace declarations
- Map BFO/GeoCore/GeoReservoir categories to published OWL IRIs
- For each term: `owl:Class` + `rdfs:subClassOf` parent + `rdfs:comment` (NLD) + `rdfs:label`
- For individuals: `rdf:type` instead of `rdfs:subClassOf`
- Serialize to Turtle (`.ttl`) format
- Must be loadable in Protégé — take screenshots for thesis

Update `pipeline.py` to orchestrate Steps 6-7. Add `--ablation` CLI flag.

Documentation:
- Create `requirements.txt` and `.env.example` (5 min)
- Update `README.md` with 7-step pipeline description
- Taxonomy statistics (depth, breadth, class/individual counts)

---

### Phase 3: Taxonomy + OWL + Polish (Days 6-7)

Detailed in Days 6-7 above. Key dependency: **taxonomy and OWL run AFTER winner selection** (Day 5). If experts are slow, use Layer 1 results to select Condition A as default winner and proceed.

---

## Critical Files

| File | Action |
|------|--------|
| `pipeline.py` | **Already partially fixed** (RAG wiring done). Add Steps 6-7, add `--ablation` flag |
| `src/utils/rag_setup.py` | Add ChromaDB caching (`force_rebuild` param) |
| `src/modules/nld_generator.py` | Accept `vector_store`/`bm25_retriever` params, fix sleep, add checkpointing, replace `exit()` with exceptions, move Gemini config inside function |
| `src/modules/term_categorizer.py` | Fix error handler columns, fix silent batch drop, replace `exit()` with exceptions, add Condition C alternate prompt |
| `src/modules/taxonomy_builder.py` | **NEW** — ~150 lines, group-based processing, needs IRI research |
| `src/modules/owl_exporter.py` | **NEW** — ~120 lines, rdflib-based, needs IRI mapping |
| `src/evaluation/ablation_study.py` | **NEW** — 4-condition runner + Layer 1 statistical analysis + expert eval spreadsheet generator |
| `test/.env-test` | **NEW** — required for E2E test to run |
| `inputs/expert_ground_truth.csv` | **ALREADY IN REPO** — `Compiled Results Table.xlsx` (2000 rows, 5 evaluators × 4 pipelines × 100 terms). Used for hallucination catalog examples and historical comparison, NOT for automated accuracy. |

---

## Execution Order

```
Day 1: Fix 7 blocking bugs (incl. exit() → exceptions, .env-test) → E2E test passes
Day 2: Review prompts + Full corpus setup → RAG index built, prompts refined
Day 3: Run 4 ablation conditions (RAG+NLD / NLD-only / term-only / raw-RAG) → Four candidate outputs
Day 4: Layer 1 statistical analysis + hallucination catalog + expert eval spreadsheet → Send to experts
Day 5: Process expert results + select winner → Thesis-ready tables
Day 6: Taxonomy builder on winning output → IS-A hierarchies
Day 7: OWL export + documentation → .ttl file loadable in Protégé
```

---

## Evaluation Methodology Summary

The ablation IS the production run. Four conditions produce candidate outputs; the best-performing one becomes the delivered artifact.

**Two independent, complementary layers:**

| Layer | What it measures | Effort | Statistical test |
|-------|-----------------|--------|-----------------|
| **1. Cross-condition comparison** | Do conditions produce different outputs? Where and how do they diverge? | Zero (automated, all 769 terms) | Cochran's Q, chi-squared, agreement matrices |
| **2. Expert pairwise** | Which NLDs and categories are actually CORRECT? | ~2-3h per expert (50-80 terms) | Wilcoxon + sign test + weighted kappa |

**Additional analyses (free, high-value):**
- Context_Used subgroup analysis (proves mechanism)
- Category migration analysis (shows WHERE RAG helps)
- NOT_CLASSIFIED rate comparison
- Category distribution shifts across conditions
- **Hallucination Catalog** (10-15 curated examples — most memorable thesis figure)

**No LLM-as-judge:** Deliberately excluded to avoid circularity — the thesis claims LLMs lack specialized domain knowledge, so using an LLM to judge geological accuracy undermines the argument.

**The winner becomes the ontology:** Experts don't just evaluate — they select the final artifact. This merges evaluation and production into one step.

**Layer roles:** Layer 1 shows conditions produce different outputs at scale (769 terms, automated). Layer 2 shows which output is BETTER (50-80 terms, expert judgment). Together: "RAG changes categorization decisions (Layer 1) and experts confirm the RAG-grounded decisions are more accurate (Layer 2)."

---

## Thesis Contribution (Final)

**Framing:** "We developed a RAG-augmented LLM pipeline that generates domain-specific Natural Language Definitions to assist expert ontology construction for underrepresented domains."

Against NeOn-GPT as state-of-the-art:

1. **RAG-grounded NLD generation** — corpus retrieval provides domain-specific knowledge that parametric LLMs lack, distilled into structured Aristotelian definitions that assist expert ontology construction
2. **NLDs as knowledge transfer mechanism** — RAG context is compressed into human-readable definitions that experts can validate and refine, bridging the gap between automated extraction and expert curation
3. **Quantified evidence** — 4-condition ablation (RAG+NLD / NLD-only / term-only / raw-RAG) × 2 evaluation layers with statistical significance, proving both RAG grounding AND NLD-as-compression add value. Context_Used subgroup analysis proves mechanism.
4. **Cross-lingual** — Portuguese corpus → English upper ontology alignment
5. **Real domain artifact** — OWL ontology for Pre-Salt geology, validated by domain experts

**NOT claiming** (NeOn-GPT's territory): general-purpose framework, verification loops, relation extraction, multi-domain generalizability.

**Key distinction from NeOn-GPT:** They automate the full pipeline but have no corpus grounding (3.7% recall on specialized domains). We provide corpus-grounded building blocks (NLDs) that enable experts to construct higher-quality ontologies for domains where parametric LLM knowledge is insufficient.

---

## Verification Checkpoints

**After Phase 1 (Day 2):** RAG index built on full corpus. All prompts reviewed and refined. E2E test passes.

**After Phase 2 (Day 5):** Four complete ablation outputs. Layer 1 cross-condition comparison computed (agreement matrices, Cochran's Q, category migration). Expert evaluation spreadsheet sent. Hallucination catalog curated. Winner selected (or Condition A as default if experts are slow).

**After Phase 3 (Day 7):** Taxonomy built on winning output. OWL opens in Protégé. Final ontology delivered. Thesis tables generated. `requirements.txt` and `README.md` updated.

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Expert evaluation takes > 4 days | Layer 1 is fully automated and sufficient for thesis |
| Taxonomy builder produces poor hierarchies | Iterative prompt engineering; manual expert review of hierarchy; accept shallow taxonomy as minimum viable |
| API rate limits during production run | Checkpointing (Bug 5 fix) + 4s sleep between calls |
| Ablation shows no RAG advantage | Thesis still delivers a Pre-Salt ontology artifact; reframe NLDs as "expert-assistive" rather than "superior" |
| 40-paper corpus processing is slow | PDF→Markdown is one-time, ChromaDB build is one-time; only NLD generation scales with terms |

---

## Nice to Have (If Time Permits)

### Parametric Recall Experiment (~1-2h)
Give Gemini the same input NeOn-GPT uses: a ~120-word Pre-Salt description + 10-15 keywords. Ask it to generate all domain classes it can. Measure: how many of the 769 expert-grounded terms does the parametric approach capture? This directly quantifies NeOn-GPT's acknowledged weakness on the Pre-Salt domain without running their code. Expected result: significant coverage gap (ICEIS Pipeline 2 already showed 40% recall for parametric-only). **Value:** Strong qualitative argument for RAG necessity. Can run anytime after Day 3.

### Streamlit Demo (~1 day)
Build a lightweight interactive demo (`app.py`, ~100-150 lines) using Streamlit:
- User types a geological term → App shows: RAG retrieval results (top-5 chunks with scores), generated NLD, BFO/GeoCore/GeoReservoir category with reasoning
- Toggle: "With RAG" vs "Without RAG" to see definitions side-by-side (ablation conditions A vs B live)
- Browse the full ontology as a searchable table
- Deploy on Streamlit Cloud (free) → shareable URL for thesis defense, LinkedIn, resume
- **Career ROI:** A live demo URL is worth more than the thesis PDF for job applications. Frame as "Automated Knowledge Graph Construction from Unstructured Documents."
