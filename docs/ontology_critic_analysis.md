# Ontology Critic — Failure-Mode Analysis and Design Direction

> Working notes captured during the June 2026 critic redesign. Two passes: a forensic audit of `iteracao_2.ttl` and `iteracao_1.ttl` against OntoClean failure patterns, then a brainstorm that converged on a single-pass critic-agent design.

---

## Part 1 — Forensic Audit of iter2 (352 classes)

### Method

Diffed `iteracao_1.ttl` (288 classes, advisor-approved) against `iteracao_2.ttl` (352 classes, advisor-flagged) to find the 144 newly-introduced classes, then classified them by mixin pattern.

### The 144 classes added in iter2, grouped by failure mode

**Qualities masquerading as classes** (~25 — the worst regression)

`Porosity`, `Permeability`, `Salinity`, `Density`, `GrainSize`, `WaterDepth`, `Bathymetry`, `LakeLevel`, `Heterogeneity`, `Mineralogy`, `RockTexture`, `AcousticImpedance`, `FlowCapacity`, `Microporosity`, `PoreSize`, `RockProperty`, `PetrophysicalProperty`, `ReservoirProperty`, `ReservoirQuality`, `PhysicalProperty` + **11 porosity subtypes**: `EffectivePorosity`, `PrimaryPorosity`, `SecondaryPorosity`, `TotalPorosity`, `VuggyPorosity`, `VugularPorosity`, `IntercrystallinePorosity`, `InterparticlePorosity`, `IntraparticlePorosity`, `MoldicPorosity`, `FracturePorosity`.

**Roles masquerading as classes** (~5)

`Aquifer`, `HostRock`, `SourceRock`, `ReservoirRock`, `OilField`.

**Compound mixins — adjective + kind** (~20)

`AlkalineLacustrineEnvironment`, `ShallowMarineEnvironment`, `LowEnergyEnvironment`, `TransitionalEnvironment`, `MudSupportedCarbonateRock`, `GrainSupportedCarbonateRock`, `LacustrineCoquina`, `ReworkedCarbonate`, `SpheruliticCarbonate`, `OrganicRichShale`, `LaminatedMudstone`, `DarkShale`, `MicrocrystallineCalcite`, `MicrocrystallineDolomite`, `MicrocrystallineQuartz`, `MacrocrystallineQuartz`, `FibrousCalcite`, `SaddleDolomite`, `RhombohedralDolomite`, `CarbonateMud`, `CarbonateFabric`, `CarbonateMound`, `PreSaltCarbonate`, `PreSaltSequence`.

**Temporal phases** (~6)

`RiftPhase`, `SagPhase`, `DriftPhase`, `PostRiftSupersequence`, `EarlyDiagenesis`, `UpperRift`.

**Analytical-framework / boundary terms** (~15)

`SequenceBoundary`, `SeismicBoundary`, `SeismicReflector`, `SeismicHorizon`, `Downlap`, `Drift`, `StackingPattern`, `StratigraphicGeometry`, `StratigraphicSurface`, `DepositionalCycle`, `Paragenesis`, `Paleoenvironment`, `Paleotopography`, `MarineSetting`, `TectonicSetting`, `StructuralDomain`, `Lithofacies`, `Lithotype`, `ReservoirFacies`, `DistalFacies`, `ReworkFacies`, `InSituCarbonateFacies`, `PoreType`.

That's ~70–90 classes that should be filtered or re-expressed. The remaining ~50 of the 144 (e.g. `Lake`, `Mollusk`, `Mineral`, `Salt`, `Evaporite`, `Basement`, `MicrobialMat`, `TectonicPlate`, `FaultBlock`, `Diagenesis`, `Subsidence`, `Uplift`) are legitimate rigid kinds or processes — KEEP.

### What the deleted critic actually was

The iter1 ontology was produced by the legacy ~2000 LOC validator (`src/validate/` engine with `rules.yaml`, deleted in commit `5768d56` on Jun 7). That validator had structural disjointness rules and per-edge OntoClean-style prompts (the `rule_ontoclean_{rigidity,identity,unity,dependence}.txt` files still sit dormant in `studies/prompts/`).

The simplified critic that replaced it (commit `6950cc2`, Jun 7) just defaults to KEEP. That's the regression. The current `domains/presalt/prompts/critic.txt` is the weakened version.

### Critic verdict vocabulary (proposed)

| Verdict | When | Action |
|---|---|---|
| **KEEP** | Term is rigid and current parent is in a BFO-compatible branch | Nothing |
| **REPARENT** | Term is rigid but parent is in the wrong BFO branch | New parent from the upper-ontology menu |
| **DROP_AS_MIXIN** | Term is `Kind + Quality/Role` compound with no upper-ontology home | Emit OWL axiom `Kind and has_quality/role some X` on the parent class |
| **DROP_AS_REDUNDANT** | Term is already expressible via an existing class + relation | Just remove |
| **DROP_AS_INSTANCE** | Term is a named individual disguised as a class | Convert to individual |

### Multi-ontology Quality/Role support

The critic does not decide *if* a term is a quality — it decides *where* it should sit, given what's available in the upper ontologies loaded. For `Porosity`, GeoCore likely has `geocore:RockProperty subClassOf bfo:Quality`, so `Porosity` becomes `KEEP-and-REPARENT` under `geocore:RockProperty`, not `DROP`. Same for `Permeability`, `Salinity`, etc.

The critic prompt takes `{candidate_term, NLD, current_parent, available_upper_parents_with_their_BFO_root}` and picks from a config-driven menu. New domain = new YAML, same prompt.

---

## Part 2 — Same Audit Applied to iter1 (288 classes, advisor-approved)

### The honest picture

| Mixin pattern | iter1 count | iter2 count | iter2 gain |
|---|---|---|---|
| Qualities-as-class | ~3 | ~25 | **+22 (regressed badly)** |
| Roles-as-class | ~6 | ~11 | +5 |
| Compound mixins (Adj+Kind) | ~30 | ~50 | +20 |
| Temporal phases | ~11 | ~17 | +6 |
| Framework terms | ~13 | ~28 | +15 |
| Generic weak-identity | ~5 | ~10 | +5 |
| **Total problematic** | **~68** | **~141** | **+73** |
| **Total classes** | **288** | **352** | **+64** |
| **% problematic** | **~24%** | **~40%** |  |

### Three things become clear

1. **iter1 is not a clean baseline.** The advisor approved it as "acceptable", but a quarter of its classes are mixins/qualities/roles/framework-terms. The old validator was filtering noise but missing the same patterns iter2 amplified.
2. **Qualities are the dominant regression.** iter1 had 3 qualities-as-class; iter2 has 25. This is the cleanest, easiest win for the critic. A single pattern rule ("term ending in `-Porosity`, `-Permeability`, `-Density`, `-Size`, `-Depth`, `-Level`, `-Property` → REPARENT under Quality branch") fixes ~20 classes immediately.
3. **Compound mixins are the persistent problem.** Both iterations are bad here (`LacustrineCarbonate`, `OrganicRichShale`, `BioclasticGrainstone`, etc.). This is the hardest filter and where the critic needs the most teeth. The OWL-axiom path (`Carbonate and located_in some LacustrineEnvironment`) is exactly right for this category.

### Win conditions for the new critic

Don't aim for "match iter1 quality". Aim for "iter1 quality on the categories iter1 got right + meaningful improvement on the categories both iterations failed".

| Category | Target |
|---|---|
| Qualities-as-class | Drop from 25 → ~3 (REPARENT under Quality branch) |
| Roles-as-class | Drop from 11 → ~3 (REPARENT under Role branch or DROP_AS_INSTANCE for things like `OilField`) |
| Compound mixins | Drop from 50 → ~15 (DROP_AS_MIXIN with axiom emission) |
| Temporal phases | Drop from 17 → ~5 (REPARENT under Process or DROP) |
| Framework terms | Drop from 28 → ~10 (REPARENT under FiatObjectPart or DROP) |
| **Total problematic** | **141 → ~36 (~10% of total)** |
| **Final class count estimate** | **~250** |

That gives a defensible "after critic" number, and it's *better than iter1*, not just "back to iter1".

---

## Part 3 — Design Pivot: From Per-Edge Critic to Single-Pass Critic Agent

### The insight

Per-edge LLM critic vs. one-shot audit agent that sees the whole ontology and applies fixes directly:

| Per-edge LLM critic | One-shot critic agent |
|---|---|
| 352 LLM calls per run, one per class | 1–5 LLM calls per run (batch by category) |
| Sees one class at a time → can't spot redundancy across classes | Sees whole ontology → can spot redundancy, suggest merges |
| Verdict on each class independently → no global view of bloat | Can say "these 8 porosity subtypes should collapse into 1 class + an axiom" |
| Needs separate remediation prompt | Same call does diagnosis + remediation |
| Hard to defend — sounds like glue | Defensible as "an automated ontology review pass" |

The per-edge approach is what 2018-vintage LLM pipelines did because GPT-3 couldn't hold a whole ontology in context. Gemini 2.5 Pro can hold the whole 352-class TTL + GeoCore + GeoReservoir + BFO + the audit instructions in a single context window with room to spare. **We were designing for a constraint that no longer exists.**

### Shape of the agent

```
INPUT (single call):
  - Full taxonomy from Step 6: CSV of (class, parent, NLD)
  - Upper-ontology menu: every BFO/GeoCore/GeoReservoir class organized by BFO branch
                         (IC | SDC-Quality | SDC-Role | Process | FiatPart)
  - Failure-mode taxonomy: the 6 categories enumerated above with definitions
  - Domain examples: ~70 mixins from iter2 + the right verdict for each (few-shot)

OUTPUT:
  [
    {class: "Porosity", verdict: "REPARENT", new_parent: "geocore:RockProperty", reason: "..."},
    {class: "OrganicRichShale", verdict: "DROP_AS_MIXIN",
     axiom: "Shale and (has_quality some OrganicRichness)", reason: "..."},
    {class: "EarlyDiagenesis", verdict: "DROP_AS_INSTANCE",
     make_individual_of: "geocore:DiageneticProcess", reason: "..."},
    {class: "Rock", verdict: "KEEP"},
    ...
  ]
```

Then a small Python module mechanically applies the JSON to the taxonomy CSV before Step 7 emits OWL.

### Critic prompt vs. critic agent

- **Critic prompt**: produces a recommendations CSV that a human reviews. We've been doing this for 6 months.
- **Critic agent**: produces the recommendations *and* a Python module deterministically applies them, with the audit log providing traceability. No human in the loop unless something looks wrong.

The agent's "agency" here is bounded — it's not running tools, not iterating, not calling subagents. It's one structured LLM call whose output is mechanically applied.

### Sizing (capacity check)

1. **Input context**: 352 classes × (label + NLD + parent) ≈ 60K tokens + upper-ontology menu + examples + instructions ≈ ~80K tokens. Well within Gemini 2.5 Pro's 1M context.
2. **Output size**: 352 verdict JSONs ≈ 30K tokens. Fits comfortably.
3. **Determinism**: T=0.0 + fixed seed + same input → same output.
4. **Defence framing**: "We used Gemini 2.5 Pro to perform an automated ontology review against an OntoClean-derived rigidity criterion and BFO-anchored category constraints" is more defensible than "we hand-coded a per-edge OntoClean simulator", because it admits what we actually did rather than pretending the LLM is a deterministic rule engine.

### Risks

The hidden risk is that the agent might decide a class is bad when it shouldn't be. Mitigation: emit `6c_critic_decisions.csv` with every decision + reason, so 10–20 decisions can be spot-checked before publishing. Same pattern as the expert workbook for Layer 2.

### Diff vs. previous design

| Before | After |
|---|---|
| New prompt `nld_classifier.txt` at Step 4 | **Cut** |
| New prompt `bfo_rigidity_fix.txt` for remediation | **Cut** |
| New module `src/validate/ontoclean_bfo.py` with per-edge JSON loop | **Replaced by** `src/modules/validate/ontology_critic_agent.py` (one LLM call + apply function, ~150 LOC) |
| New Step 6.5 in pipeline | Same step number (6c), totally different prompt + module |
| Defensible as "OntoClean + Seyed–Shapiro" | Defensible as "automated ontology review with OntoClean rigidity criterion + BFO category checks, applied via a single LLM pass over the full taxonomy" |

---

## Open Questions (answered by user — to be filled in next turn)

1. **Scope of the agent's authority**: re-parenting + dropping mixins only, or also *merging* near-duplicates (`Porosity` / `PrimaryPorosity` / `EffectivePorosity` → one class + axioms)?
2. **Stopping point**: one pass and done, or iterate until stable?
3. **Failure mode for ambiguous cases**: default to KEEP, emit `UNCERTAIN` for human review, or DROP with low-confidence flag?

---

## Provenance

- Forensic diff method: regex extraction of `presalt:X a owl:Class` declarations from both TTLs, set diff in PowerShell.
- Failure-mode taxonomy: derived from OntoClean (Guarino & Welty 2002) rigidity criterion, adapted to BFO categories via Seyed & Shapiro (FOIS 2012).
- Methodology hygiene: all remediation expressed in BFO + GeoCore + GeoReservoir primitives only. No UFO stereotypes, no `presalt:Kind` parallel hierarchy, no metric frameworks.
