# Research Brief — Validating an LLM-Generated BFO-Aligned Domain Ontology

> Purpose: give an external research agent enough context to recommend the **best validation approach** for our specific situation. The brief is deliberately short. Detail lives in `docs/CONTEXT.md` and `docs/OVERVIEW.md`.

---

## 1. What we're building

**PreSaltOntoLearn** — a master's thesis pipeline that builds an OWL ontology of Brazilian Pre-Salt petroleum geology **automatically from PDFs**, using Gemini 2.5 Pro (temperature 0.0) + RAG over `BAAI/bge-m3` embeddings.

Pipeline (7 steps): PDF → term extraction → term filtering → Aristotelian NLD generation (RAG-augmented) → upper-ontology categorization → taxonomy + relations → ontology critic → OWL emission → OOPS! verification.

Upper ontologies (waterfall priority):
1. **GeoReservoir** (Abel — petroleum-specific extension)
2. **GeoCore** (Abel 2015 — geoscience core)
3. **BFO** 2020 (Smith et al., ISO/IEC 21838-2)

Output: ~300 `presalt:` classes, ~50 named individuals, anchored to upper ontologies. Currently passes HermiT and OOPS!.

## 2. The thesis claim (the only thing being evaluated)

> *RAG-augmented Aristotelian NLDs improve upper-ontology classification over term-only, parametric-NLD, and raw-RAG baselines.*

Replicates and extends Lopes Junior (2024). Evaluated via 4-condition ablation (A/B/C/D) + a 2-layer analysis (automated metrics + 3-expert blinded workbook of 200 stratified judgements).

**Important: we are not claiming a contribution in upper-ontology design or ontology evaluation methodology.** Any validation layer we add must serve the NLD claim, not compete with it.

## 3. The concrete problem we observed

Comparing two snapshots of our pipeline output:

| Snapshot | presalt classes | Individuals | Notes |
|---|---|---|---|
| **iter1** (May 2026) | 288 | 26 | Sent to advisor. Used a now-deleted OntoClean-style critic. |
| **iter2** (Jun 2026) | 352 | 55 | Current state. After we deleted the validation engine. |

The +64 net delta is dominated by **mixins, qualities, and roles being modelled as IS-A subclasses** of object classes. Concrete examples our advisor flagged:

- `Porosity is-a Rock` — quality treated as kind
- `Aquifer is-a Rock` — role treated as kind
- `OilField is-a Region` — fiat object / role conflation
- `Drift is-a SedimentaryProcess`, `Onlap is-a StratigraphicFeature` — analytical-framework terms treated as real entities
- Several `EarlyDiagenesis`-style temporal-phase terms attached as subClassOf

The root cause is verified through git forensics: a critic prompt that explicitly enforced anti-quality-as-class / anti-framework / anti-overly-specific rules was deleted on commit `5768d56` (Jun 7) and replaced by a much weaker single-pass critic that defaults to KEEP. That critic is the file `domains/presalt/prompts/critic.txt` currently open in the editor.

## 4. What we already considered

A 3-gate proposal:
- **Gate A** — NLD self-classification at Step 4 (prevent mixins before they enter the taxonomy).
- **Gate B** — Per-edge OntoClean-style audit (rigidity / identity / unity / dependence) over taxonomy edges.
- **Gate C** — Orphan-leaf detection at OWL emission time.

Gate B was originally proposed as a literal port of OntoClean (Guarino & Welty 2002). The user then raised the right concern: **isn't OntoClean associated with UFO (Guizzardi), while our ontology is grounded in BFO?**

Our own initial reading (Wikipedia + obofoundry.org + a research subagent) concluded:
- OntoClean (2000–2002) **predates UFO** (~2005) and is explicitly foundational-ontology-neutral.
- UFO **uses** OntoClean meta-properties to define its stereotypes (Kind/Phase/Role/Mixin). OntoClean does not use UFO.
- BFO has overlapping machinery (Continuant/Occurrent disjointness, `SpecificallyDependentContinuant`, `Role`, time-indexed `instance_of`) but does not enforce OntoClean-style per-edge checks.
- OBO Foundry BFO ontologies (GO, ChEBI, OBI) do **not** routinely apply OntoClean. They rely on trained ontologists + HermiT/ELK + OOPS!.
- We are **not** trained ontologists — we are an LLM. So a post-hoc OntoClean-style audit is plausibly more justified for our setting than for hand-built BFO ontologies.

We are unsure whether to:
1. Implement Gate B as adapted-OntoClean (4 meta-property checks, BFO-anchored abstention rules, BFO-native remediation).
2. Skip OntoClean entirely and lean on **OOPS!** pitfall catalogue (Poveda-Villalón) + structural / disjointness checks already present in BFO.
3. Use something else from the BFO / OBO Foundry world we haven't considered (e.g., ROBOT report, OQuaRE metrics, the BFO Conformance Suite, the Relation Ontology constraints, Pellet/HermiT with explicit disjointness axioms, OntOlogy Pitfall Scanner extensions).
4. A hybrid — e.g., Gate A (upstream NLD self-check) + OOPS! at emission, no Gate B.

## 5. Hard constraints

- **Single-pipeline, LLM-driven.** Any validator runs as a step in our pipeline; we can call Gemini, but cannot ask a human ontologist to label things.
- **Reproducible.** Temperature 0.0, seed 42, fixed embedding model, no API-mutable weights.
- **Domain-agnostic refactor in progress.** Prompts live in `domains/<name>/prompts/` with `<<persona>>` / `<<examples>>` placeholders. Any validator we add must work for arbitrary domains, not just Pre-Salt.
- **Thesis hygiene.** The NLD-vs-baselines comparison must remain the headline. The validator is a *quality floor*, not a *contribution*. Its citations and methodology need to be defensible but not so heavy they steal narrative space.
- **Avoid UFO drift.** No `«kind»` / `«phase»` / `«role»` stereotypes. No parallel `presalt:Kind` hierarchy. Remediation must be expressible in BFO + GeoCore + GeoReservoir primitives only.
- **Avoid over-engineering.** The prior validation engine was ~2000 LOC with a DSL + hybrid embedding clustering. It was deleted and we don't want it back. Target: one module, one prompt file per check, JSON-in/JSON-out, deterministic merge.

## 6. The question we want answered

> Given the constraints above, **what is the best validation approach** to catch the mixin / quality-as-class / role-as-class / framework-as-class failures we observed in iter2, **without contradicting BFO's discipline**, **without drifting toward UFO**, and **without overshadowing the thesis's NLD claim**?

Specifically we'd like the agent to opine on:

1. **Is adapted OntoClean (BFO-anchored, no UFO stereotypes) genuinely the right diagnostic for this failure mode, or is there a more BFO-native validator (ROBOT report, OOPS! subset, OQuaRE, RO-based property-constraint check, explicit BFO disjointness axioms) that does the same job with less methodological baggage?**
2. **At which pipeline step should the validator live?** Upstream at NLD time (cheap, prevents the class from being created), at taxonomy time (per-edge check), or at OWL emission time (post-hoc audit)? Or all three?
3. **What is the minimum citable methodology** we can defend in a master's thesis Chapter 4 ("Evaluation") without inflating scope? (e.g., "we use OntoClean's 4 meta-properties as a diagnostic, remediation is BFO-native" — is that defensible, or does it open a hole?)
4. **Concrete prompts/rules.** If OntoClean is the answer, what is the minimum prompt set (rigidity / identity / unity / dependence, or fewer) and what are the BFO-anchored abstention rules that prevent the check from "fighting" BFO's own categories (e.g., never reject -I on `bfo:Quality`, never reject ~R on `bfo:Role`)?
5. **Risks we haven't named.** What could go wrong with the recommended approach that we'd only discover at the defence?

## 7. Inputs the agent can rely on

- The current critic prompt (weak): [domains/presalt/prompts/critic.txt](../domains/presalt/prompts/critic.txt)
- The dormant OntoClean v1 prompts (June 5 build, Pre-Salt examples hardcoded): `studies/prompts/rule_ontoclean_{rigidity,identity,unity,dependence}.txt`
- The full architecture: [docs/CONTEXT.md](CONTEXT.md)
- Pipeline + evaluation summary: [docs/OVERVIEW.md](OVERVIEW.md)
- Ablation/evaluation strategy: [docs/thesis_evaluation_strategy.md](thesis_evaluation_strategy.md)
- Two snapshots for reference: `iteracao_1.ttl` (288 classes, good), `iteracao_2.ttl` (352 classes, bloated).

## 8. What we want back

A short, opinionated recommendation:

1. **Approach** (one of the four numbered options in §4, or a new one).
2. **Justification** in 5–10 lines, citing 2–4 primary sources (papers, standards, or BFO/OBO Foundry docs — not blog posts).
3. **Concrete artefacts** to build (which prompt files, which python module, which CLI flag, which output CSV).
4. **What we should explicitly NOT do** and why.
5. **Defence-ready one-paragraph framing** for the thesis.

Avoid restating our problem. Assume we've read OntoClean, BFO 2020, the OBO Foundry principles, Lopes Junior 2024, and the OOPS! paper. Get to the recommendation.
