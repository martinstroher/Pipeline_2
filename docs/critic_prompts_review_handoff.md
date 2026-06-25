# Critic Prompts — Ontology-Engineering Review & Handoff

> Review of the `validate`-step critic prompts in `domains/presalt/prompts/`
> (`critic_taxonomy.txt`, `critic_taxonomy_dedup.txt`, `critic_relations.txt`),
> with a focus on whether the OntoClean-on-BFO approach is sound and correct.
> Written as a handoff for the agent currently editing these files.
>
> **Caveat:** `critic_taxonomy.txt` is mid-edit. This note reflects the file
> state at review time and explicitly flags which findings the in-progress
> edits have *already* addressed, so nothing is redone or reverted by mistake.

---

## TL;DR verdict

The approach **makes sense and is mostly correct.** OntoClean is the right
diagnostic for the documented failure mode (qualities/roles/dispositions/
frameworks modelled as IS-A subclasses), and applying it to a BFO-aligned
ontology is legitimate. Do **not** relitigate the premise. The remaining work
is precision fixes, one of which is a real methodological gap.

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | `DROP_AS_MIXIN` `carried_by` is logged but never emitted as an OWL restriction — the quality/disposition is silently deleted | **High** | **OPEN** |
| 2 | "If unsure, use `-R`/`-I`/`-D`" — the `-D` default is the *violating* child sign under a `+D` parent | Low–Med | **PARTIALLY OPEN** (identity part fixed by the advisory demotion; dependence part remains) |
| 3 | No precedence among the 7 checks when several fire (e.g. Check 1 vs Check 3 on `SourceRock`) | Medium | **OPEN** |
| 4 | "Mixin" used more loosely than strict Guarino–Welty | Low | **OPEN** (terminology/defence) |
| 5 | Unity meta-property dropped (3 of 4 used) | Low | **By design — acknowledge as "OntoClean-derived"** |
| 6 | Risk of OntoClean checks penalising terms correctly under `bfo:Role`/`Quality` | Medium | **MOSTLY FIXED for identity** (advisory + branch guard); rigidity/dependence less explicit |

---

## Theoretical green light (do not relitigate)

- **OntoClean (Guarino & Welty, 2000–2002; 2009) is foundational-ontology-neutral.**
  It predates UFO and is not bound to it. It is a meta-property discipline
  (rigidity / identity / unity / dependence) for auditing subsumption edges in
  *any* taxonomy. Using it on a BFO ontology is standard.
- **It is complementary to BFO, not redundant.** BFO + HermiT/ELK only catch
  *disjointness* (e.g. a class that is both Continuant and Occurrent). They do
  **not** catch `SourceRock is-a Rock` (role-as-kind) or `Porosity is-a Rock`
  (quality-as-kind) — those are metaphysically wrong but logically consistent.
  OntoClean is exactly the instrument that flags them. This is the gap the
  critic fills.
- **Seyed & Shapiro (FOIS 2012, "BFO and OntoClean")** formally map OntoClean
  meta-properties onto BFO categories — the same move this prompt makes. Cite
  it in the thesis as the bridge that licenses OntoClean-on-BFO.
- Net: the premise is correct. Findings below are about *execution*, not the
  decision to use OntoClean.

---

## Open items (action needed)

### 1. (High) `carried_by` is a broken promise — the differentia is deleted, not re-expressed

`critic_taxonomy.txt` says `DROP_AS_MIXIN` "MUST carry `carried_by` … It
records where the meaning lives once the term is removed," and the worked
example claims `OrganicRichShale` → `Shale has_quality some OrganicRichness`
"preserves truth conditions."

It does not, end-to-end:

- `src/modules/validate/critic.py` → `_apply_taxonomy_edits` applies
  `DROP_AS_MIXIN` as `keep_mask[idx] = False` (row removed). `carried_by` is
  written **only to the audit log** via `_probe_cols` (the `carried_by`
  column in `validate_edits.csv`).
- `carried_by` is consumed **nowhere else** in the repo (grep confirms: only
  `critic.py` audit log + the prompt itself).
- `src/modules/emit/owl_exporter.py` *does* emit companion axioms
  (`inheres_in some IndependentContinuant` / `realized_in some Process`) but
  only for **surviving** Quality/Role *classes*. It never reconstitutes a
  dropped mixin's filler (`OrganicRichness`), and the surviving parent
  (`Shale`, an object) gets nothing.

So the `OrganicRichness` content is lost. This contradicts (a) proper
OntoClean remediation (a mixin is *re-expressed*, never deleted) and (b) the
original design in `docs/ontology_critic_analysis.md` ("Emit OWL axiom
`Kind and has_quality/role some X`"). For the thesis this is the kind of claim
an examiner will probe ("show me the triple") — and right now there is none.

**Fix — pick one:**
- **(a) Honour it (preferred).** Thread `carried_by` from `validate_edits.csv`
  into `owl_exporter.py` so the surviving parent gets
  `parent and (has_quality some OrganicRichness)` (mint the filler class +
  the SDC restriction). This makes the methodology true end-to-end.
- **(b) Soften the claim.** If emission is out of scope, change the prompt
  wording to "recorded in the audit log for traceability" and drop "where the
  meaning lives" / "preserves truth conditions." Don't claim preservation the
  pipeline doesn't deliver.

This spans **prompt → `critic.py` → `owl_exporter.py`**; coordinate before
touching emission behaviour.

### 2. (Low–Med, partially open) The "unsure" default sign rule

Output constraints: *"If unsure, use `-R`/`-I`/`-D` and KEEP."* Against
Check 1's own violation conditions:

| Sign | Violation fires when | Default `-x` safe? |
|---|---|---|
| Rigidity | parent `~R` **and** term `+R` | ✅ `-R` never triggers |
| Identity | both `-I` | ✅ now harmless — identity is **advisory**, never REPARENTs |
| Dependence | parent `+D` **and** term `-D` | ❌ `-D` is the *triggering* term-sign under a `+D` parent (`Reservoir`, `Seal`) |

The identity row is fixed by the recent advisory demotion. The **dependence**
row still bites: an *uncertain* term defaulting to `-D` under a `+D` parent
produces a spurious REPARENT — the opposite of "abstain → KEEP."

**Fix:** either default dependence to `+D` when unsure, or (cleaner) decouple
uncertainty from the signs: *"if unsure of a sign, choose KEEP regardless of
the signs."*

### 3. (Medium) No precedence among the seven checks

Several checks can fire on one row with different remedies. Canonical case:
`SourceRock is-a Rock` triggers **Check 1** (anti-rigid/dependent → "REPARENT
onto a rigid/independent genus") *and* **Check 3** (role-as-class → "REPARENT
under `role`"). The schema forces a single `action`, but no order is stated, so
the verdict is left to the model — hurting determinism at T=0 and
defensibility. Also Check 2 (quality DROP) vs Check 4 (mixin) vs Check 7
(framework) overlap.

**Fix:** state an explicit precedence, e.g.
`proper-noun (6) → framework (7) → quality/role/disposition bucket (2/3/4) →
OntoClean edge (1) → redundancy (5)`. For `SourceRock`, Check 3 (role) should
win over Check 1's "find a rigid genus."

### 4. (Low) "Mixin" terminology drift

In Guarino–Welty a *mixin* is specifically an **anti-rigid, identity-non-
supplying property that subsumes identity-supplying ones**. What the prompt
calls `DROP_AS_MIXIN` for `OrganicRichShale` is really a **rigid subkind whose
differentia is a Quality** — BFO hygiene, not a mixin in the strict sense. The
verdict is fine; the *label* invites a challenge at defence. Either cite the
looser usage explicitly in Chapter 4, or rename the verdict (e.g.
`DROP_AS_QUALITY_LADEN`). Low priority — naming only.

---

## Already addressed in the current file (verify, don't redo)

The in-progress edits to `critic_taxonomy.txt` already handle two of the
original findings — keep them:

- **Identity demoted to advisory.** Check 1 now says "record the term's
  `identity` sign, but **never REPARENT on identity alone**," and the output
  constraints echo it. This removes the original logical hazard where
  defaulting to `-I` could trigger a false reject.
- **BFO-branch guard for identity.** Check 1 now states "On Quality / Role /
  Disposition / Process / TemporalRegion branches `-I` is normal and correct —
  do not flag," with a matching worked example
  (`ReservoirQuality is-a GeologicalProperty (quality branch) → KEEP`). This
  was finding #6 (OntoClean fighting BFO's own categories) and is now handled
  for identity. Rigidity/dependence have no equivalent explicit branch guard,
  but the risk there is lower and abstention rule B covers most of it.

---

## What must be preserved (do not "simplify" away)

These are correct and load-bearing:

- **Disposition as a first-class bucket, distinct from Quality and Role**, with
  Function folded into Disposition. This is correct BFO (Function ⊑ Disposition)
  and a genuine improvement over pipelines that collapse everything to Quality.
- **Proper-noun → `CONVERT_TO_INSTANCE`** (class-vs-individual discipline),
  with the `target_class` guidance (geological age → `temporal region`; field/
  basin → `site`; plate/rock body → `object`). BFO-faithful.
- **Reparent targets exist:** `role` (BFO_0000023) and `quality` (BFO_0000019)
  are real classes in `domains/presalt/ontology_config.yaml`, so those verdicts
  are grounded, not dangling.
- **Separation of concerns:** taxonomy critic judges only IS-A; relations are
  read-only context; dedup is a separate stage (`critic_taxonomy_dedup.txt`).
  Don't merge these — it stops the LLM conflating edge types.
- **Probe 3 (counterfactual relational rewrite)** is a sound truth-condition /
  paraphrase test. **Probe 1** correctly anchors "necessary differentia" to the
  *genus* (not the child), so the parent-collapse logic is valid.
- **Conservative abstention ethos** (KEEP over confident-wrong) faithfully
  reproduces OntoClean's stance.

---

## `critic_relations.txt` — quick pass (broadly sound)

Reviewed because it is part of the same suite. No blocking issues:

- KEEP / DROP / FIX with **minting as last resort** and the **role-detour
  pattern** (`has_role some <NewRole>`) preferred over predicate-specific
  properties — this is the BFO-correct way to avoid `produces_hydrocarbons`-
  style properties. Good.
- Mint rules (generic verb name, `subPropertyOf` a menu parent, metatype
  domain/range, justification) are sound. Minted rows are re-validated by
  `validate_relation` in `_normalize_and_revalidate_relations`, so a bad FIX
  can't silently produce a BFO-invalid relation.
- Handoff from taxonomy decisions is coherent (e.g. `SourceRock` REPARENTed
  under `role` → relation becomes `has_role some SourceRockRole`).

**Watch items (not blockers):**
- Minted properties land in the `critic_minted` provenance tier (lowest
  authority, persisted to `validate_minted_properties.csv`). Make sure the
  thesis frames invented vocabulary as auditable, not authoritative.
- The "prefer menu / role-detour over minting" instruction is only as good as
  the menu passed in (`_build_relations_menu`, the `critic_menu: true` flag).
  If a needed generic property isn't flagged, the model is pushed to mint.

---

## References

- Guarino, N. & Welty, C. (2002/2009). *An Overview of OntoClean.* —
  rigidity / identity / unity / dependence meta-properties and the
  subsumption constraints.
- Seyed, A. P. & Shapiro, S. C. (2012, FOIS). *BFO and OntoClean.* — the
  formal bridge that licenses applying OntoClean to BFO categories.
- Arp, R., Smith, B. & Spear, A. (2015). *Building Ontologies with BFO.* —
  Quality / Role / Disposition / Function category discipline; class-vs-
  individual.
- Internal: `docs/research_brief_ontoclean_for_bfo.md`,
  `docs/ontology_critic_analysis.md` (design intent, including the originally
  intended `Kind and has_quality/role some X` emission).
