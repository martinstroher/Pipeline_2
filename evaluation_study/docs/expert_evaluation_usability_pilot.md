# Expert Evaluation Usability Pilot

> **Status:** AI role-play usability evidence only. Five subagents acted as Pre-Salt geology specialists. Their geological judgments are not human-expert evidence and must not be used as thesis results.

> **Implementation update (2026-07-25):** The construct-validity changes recommended below are implemented and pass the zero-Azure rehearsal. Two recommendations were explicitly rejected for the study protocol: Taxonomy usefulness remains a separate judgment, and every expert receives the same sampled items. The original pilot workbooks remain NO-GO evidence; the generated vNext workbooks are ready for a small real-human pilot, not yet for the final thesis study.

## Decision

**NO-GO for distributing the current full workbook unchanged.**

The workbook mechanics work and the sheet purposes were broadly understood. The main risks are construct validity and burden:

1. Formal ontology language can make ratings measure ontology literacy instead of geology.
2. Relation statements do not declare whether they mean possible, typical, or universal truth.
3. Critic review uses an undefined “core vocabulary,” so evaluators apply different private scopes.
4. Ambiguous bare terms lead experts to evaluate different senses.
5. The estimated full burden is 5–12 focused hours, usually 8–10 hours.

## Pilot Design

Five independent personas completed representative partitions of every task type:

| Pilot | Persona |
|---|---|
| Geologist 1 | Carbonate sedimentology and diagenesis |
| Geologist 2 | Structural geology and basin evolution |
| Geologist 3 | Reservoir geology and petrophysics |
| Geologist 4 | Stratigraphy and depositional systems |
| Geologist 5 | Petroleum systems and integrated review |

A common calibration subset was included in all five workbooks; remaining rows were distributed so every source row was evaluated at least once.

### Completion

| Sheet | Source rows | Distinct rows covered | Pilot evaluations including overlap | Notes written |
|---|---:|---:|---:|---:|
| Representation | 100 | 100 | 112 | 97 |
| Category Correct | 151 | 151 | 163 | 101 |
| Taxonomy | 40 | 40 | 52 | 11 |
| Defined Classes | 13 | 13 | 21 | 19 |
| Relations | 25 | 25 | 37 | 29 |
| Individuals | 15 | 15 | 23 | 11 |
| Critic Decisions | 40 | 40 | 52 | 52 |
| **Total** | **384** | **384** | **460 row evaluations** | **320** |

The five workbooks contained **981 required responses**. Independent validation found no blank required responses and no invalid dropdown values.

### Shared-row consistency

The common calibration rows produced 41 comparable field judgments:

- 23/41 were unanimous across all five personas.
- 34/41 had at least 4-of-5 agreement.
- Disagreement clustered around formal ontology meaning, term ambiguity, and local-versus-general scope rather than ordinary geological facts.

## Alignment With Intended Measurements

| Sheet | Intended measurement | What pilot evaluators did | Alignment |
|---|---|---|---|
| Representation | Term relevance, absolute A/B definition quality, blinded preference | Compared geological accuracy, essential meaning, scope, and unsupported detail | **Good** |
| Category Correct | Correctness of the proposed upper/domain category | Judged geological kind, but often translated or guessed BFO terminology | **Partial** |
| Taxonomy | Correct IS-A placement; separate usefulness judgment | Correctly tested “is a type of”; usefulness was always Yes | **Relationship good; usefulness weak** |
| Defined Classes | Correct base kind plus genuinely defining feature | Understood the idea, but role/disposition labels and circular fillers obscured it | **Partial** |
| Relations | Relation correctness and whether it is class-wide | Consistently distinguished local truth from general truth; missing quantifiers caused doubt | **Concept good; interface weak** |
| Individuals | Named particular versus reusable kind; correct type | Clear for fields/basins/formations; unclear for ages/stages and informal intervals | **Mostly good** |
| Critic Decisions | Acceptability of pruning/demotion for a lean core | Evaluated geological information loss, but each persona invented a meaning of “core” | **Partial/unstable** |

## Findings By Task

### `Leave unclassified`

Keep this task. All five personas rejected leaving `oil-water contact` unclassified. Evaluators understood that they only needed to decide whether a term deserves some category, not name the category.

### Taxonomy

Keep IS-A correctness. The pilot suggested removing or redesigning retained-edge usefulness, but the study protocol retains it as a separate judgment.

All 52 usefulness judgments were `Yes`, so it added no discrimination. Retention usefulness is already tested more directly in Critic Decisions. If vocabulary usefulness remains a research question, use a balanced retained/rejected distinction sample rather than retained taxonomy edges only.

### Defined Classes

The two current questions are related but not identical. Shared rows showed the second question can diagnose why an overall definition fails. However, formal phrases such as `has role Regional Seal Role` are circular and ontology-heavy.

Recommended interface:

1. Show one natural-language proposed definition.
2. Ask one verdict: Correct / Partly correct / Incorrect / Unsure.
3. For Partly/Incorrect, collect a bounded reason:
   - base kind is wrong;
   - feature is not defining;
   - too broad;
   - too narrow;
   - wording unclear;
   - other/Unsure.

### Relations

The two intended constructs add value, but the two-question interface causes confusion.

Shared examples:

- `REL-007`: all five judged the statement correct and generally true.
- `REL-006`: statement split Partial/No; all five judged it not generally true.
- `REL-017`: all five judged the statement Partial and not generally true.

Replace the two questions with one explicitly quantified relation sentence and one verdict. Example:

> “In general, dolomudstone is partly composed of calcite.”

Responses:

- Generally true;
- True only in some contexts/instances;
- Relation or direction is partly wrong;
- Incorrect;
- Unsure.

This preserves correctness and scope while removing apparent duplication.

### Critic Decisions

Keep the actual after-state: evaluators need to know what the critic did. The pilot found useful errors:

- all five preferred retaining the information for aggradation;
- four of five preferred keeping alluvial fan separately;
- all five rejected excluding vugular porosity.

Before real use:

1. Define the target core: users, competency questions, geographic/stratigraphic scope, and granularity.
2. Show neutral action and resulting treatment, not persuasive labels such as “too specialized” or “redundant.”
3. Ask one acceptability verdict: Accept / Accept with concern / Reject / Unsure.
4. Require Preferred Treatment only for Reject/Concern.
5. For exclusions claimed as redundant, show the retained parent or where equivalent information survives.

## Cross-Cutting Problems

### Formal ontology language

Difficult labels included `independent continuant`, `specifically dependent continuant`, `fiat object part`, `continuant fiat boundary`, `generically dependent continuant`, `realizable entity`, `role`, `disposition`, `continuant part`, and `site`.

Create geologist-facing display labels and operational definitions with one positive and one negative Pre-Salt example. Preserve formal labels only in the private key or secondary reference.

### Ambiguous term senses

Repeated examples were `mound`, `carbonate`, `lacustrine carbonate`, `reservoir`, `rift`, `micrite`, `seismic facies`, `Neobarremian`, and `pre-salt section`.

Attach a short neutral term gloss or source phrase to fix the intended sense without revealing condition identity.

### Local-to-general leakage

Corpus excerpts often support one field, facies, or “some cases,” while relation sentences look universal.

Label relations as existential, typical, universal, individual fact, or corpus-only before expert judgment.

### Definition detail bias

Longer contextual definitions can appear better due to detail or worse due to unsupported local claims. Keep both quality ratings. State that context counts only when accurate and appropriately scoped. Keep preference only as a declared secondary outcome.

### Undefined core scope

Evaluators disagreed with exclusions because they assumed different boundaries for basement geology, reservoir-scale structures, carbonate classification, pore types, and analytical properties.

State intended users and the ten competency questions before Critic Decisions. Define what remains in extensions.

### Workload

The pilot proposed a balanced incomplete assignment to reduce burden. The study protocol rejects that option: every expert receives every sampled item. Completion time is recorded by module so the real-human pilot can quantify the resulting burden.

## vNext Workbook Plan

| Sheet | Recommended change |
|---|---|
| Representation | Keep relevance and two quality scores. Consider a separate relevance first pass. Keep preference only as a secondary endpoint. |
| Category Correct | Keep one verdict. Add neutral term glosses and geologist-facing category rules/examples. Keep `Leave unclassified`. |
| Taxonomy | Keep relationship correctness and retained-edge usefulness as separate judgments. |
| Defined Classes | Use one natural-language definition verdict plus bounded issue reason. Hide formal relation names in the main display. |
| Relations | Use one quantified relation verdict distinguishing general, contextual, partly wrong, incorrect, and unsure. |
| Individuals | Keep both questions. Explain named age/stage and informal interval conventions. |
| Critic Decisions | Define core scope; show neutral decision and after-state; use one acceptability verdict plus conditional preferred treatment. |

## Phased Plan

### Phase 1 — Construct validity

1. Add geologist-facing category/relation labels, rules, examples, and counterexamples.
2. Add neutral glosses for ambiguous terms.
3. Replace two relation questions with one quantified verdict.
4. Define core-vocabulary users, CQs, scope, granularity, and extension policy.
5. Neutralize critic wording while retaining after-state.
6. Simplify Defined Classes to one verdict plus bounded issue reason.
7. Retain Taxonomy usefulness as a separate protocol outcome.

### Phase 2 — Workload and sampling

1. Give every expert the same sampled items.
2. Preserve independent row order and blinded definition order per expert.
3. Include clear positive, partial, negative, and unsure cases.
4. Record completion time by module and allow experts to pause between modules.

### Phase 3 — Calibration

1. Add a short practice section with feedback.
2. Include one example each for category specificity, time individual, local relation, defined class, and critic decision.
3. Confirm interpretation of Partial, Unsure, scope quantifiers, and core inclusion.

### Phase 4 — Human pilot

1. Recruit one carbonate/reservoir, one structural/basin, and one stratigraphic geologist.
2. Collect actual completion time and think-aloud doubts on a small vNext workbook.
3. Revise before the thesis study.

## Protocol Notes

- Independent inspection confirmed every pilot workbook had the correct visible identity. Geologist 2’s reported identity mismatch was an observation error.
- Geologist 5 accidentally opened another pilot workbook before correcting scope and reconstructed its own workbook after an interrupted save. Its report is useful for recurring themes but receives lower weight for strict protocol conclusions.
- Only findings corroborated by workbook inspection or multiple reports were promoted to this plan.

## Final Recommendation

**NO-GO for the original pilot workbook as the final human study.**

**IMPLEMENTED with protocol decisions noted above. GO for a small real-human pilot.** The next evidence gate is actual completion time and think-aloud feedback from the three planned geology specialties. Only that pilot can justify final-study distribution.
