"""Ontology critic — the `validate` verb.

Per category, three kinds of LLM call run in sequence so each is informed by
the previous:

  1. **Taxonomy critic (Stage 1, per-term, CHUNKED).** Judges the IS-A rows
     (KEEP / REPARENT / DROP_AS_MIXIN / DROP_AS_REDUNDANT / CONVERT_TO_INSTANCE)
     in small chunks (``CRITIC_TAXONOMY_CHUNK_SIZE`` terms, default 5) so each
     call reasons about only a handful of terms. It sees the chunk's terms +
     NLDs, the chunk terms' own/ancestor relations as read-only context, the
     parents' NLDs (to judge vacuous restatement), and the allowed target-class
     list. It also runs an OntoClean parent–child edge check
     (rigidity / dependence drive REPARENT; identity is advisory-only) from the
     parent NLDs, and buckets the
     differentia by BFO category (Quality / Disposition / Role / Site / Process
     / TemporalRegion). Output carries a per-row probe trace (probe1/2/3 + the
     OntoClean signs rigidity/identity/dependence) and, on every
     DROP_AS_MIXIN, a `carried_by` note (logged for audit, not re-emitted). DROP_AS_REDUNDANT here is
     **parent-collapse only** (a term that vacuously restates its parent).
  2. **Dedup critic (Stage 2, cross-term, ONE call over survivors).** Sees all
     surviving terms of the category together and makes the two decisions that
     need a global view: sibling near-synonym redundancy and weak intermediates
     (using a Python-computed `child_count`). Only emits DROP_AS_REDUNDANT, each
     citing the `survivor` it collapses into. A mutual-drop guard then un-drops
     any term whose cited survivor was itself dropped (never lose a concept).
  3. **Relation critic.** Judges the object-property rows (KEEP / DROP / FIX,
     FIX may carry a `mint` block) over the whole category, informed by a
     **handoff** of the merged taxonomy decisions from stages 1–2.

Categories run in parallel on the worker pool; the calls above are sequential
within a category. A completeness guard re-asks the model for any input ids it
forgot in stages 1 and 3, so large categories are not silently under-reviewed.

I/O contract:
    run_critic(taxonomy_csv, output_dir, *, relations_csv=None)
        -> (final_taxonomy_path, final_relations_path_or_None)

Outputs written to `output_dir`:
    validate_taxonomy.csv            — cleaned taxonomy
    validate_relations.csv           — cleaned relations (only if relations_csv given)
    validate_edits.csv               — full audit log (taxonomy rows carry the probe trace)
    validate_instances.csv           — terms converted to NamedIndividuals (Term, Target_Class, …)
    validate_minted_properties.csv   — newly invented ObjectProperties (provenance=critic_minted)
    validate_responses_archive/{ts}.jsonl — raw LLM responses, one line per call (call=taxonomy|dedup|relation)

Safety guards:
    - Default temperature 0 (deterministic).
    - Completeness guard: any input id the model omits is re-asked once; rows
      still missing fall back to implicit KEEP.
    - After taxonomy edits, any relation whose Filler was DROPped is dropped
      (phantom-filler cleanup, logged as DROP/phantom).
    - Every decision is recorded in validate_edits.csv; nothing is silently
      undone — the audit log is the safety net.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

from src.utils import log
from src.utils.csv_io import read_csv, write_csv
from src.utils.llm_client import get_client, generate
from src.utils.ontology_config import get_config
from src.utils.prompt_loader import load_prompt
from src.utils.relation_validator import (
    PROPERTY_CONSTRAINTS,
    get_metatypes,
    normalize_property,
    validate_relation,
)


_TAXONOMY_VERDICTS = {
    "KEEP", "REPARENT", "KEEP_AS_BEARER",
    "DROP_AS_MIXIN", "DROP_AS_REDUNDANT", "CONVERT_TO_INSTANCE",
}
_RELATION_VERDICTS = {"KEEP", "DROP", "FIX"}

_DROP_TAX_VERDICTS = {"DROP_AS_MIXIN", "DROP_AS_REDUNDANT", "CONVERT_TO_INSTANCE"}

# KEEP_AS_BEARER: companion object-property → the BFO parent the minted filler
# class is declared under. Keeps a material bearer under its genus and carries
# the realizable/quality off the IS-A edge onto a companion axiom.
_BEARER_PROPERTIES = {
    "has_role": "role",
    "has_function": "function",
    "has_disposition": "disposition",
    "has_quality": "quality",
}


# ─── Payload builders ─────────────────────────────────────────────────────

def _term_to_category(tax: pd.DataFrame) -> dict[str, str]:
    return {
        str(t).strip().lower(): c
        for t, c in zip(tax["Term"].tolist(), tax["Category"].tolist())
    }


def _term_to_parent(tax: pd.DataFrame) -> dict[str, str]:
    return {
        str(t).strip().lower(): str(p).strip()
        for t, p in zip(tax["Term"].tolist(), tax["Parent_Term"].tolist())
    }


def _ancestor_chain(term: str, parents: dict[str, str], valid_categories: set[str], max_depth: int = 4) -> list[str]:
    """Walk up Parent_Term until we hit a category root or run out."""
    out: list[str] = []
    seen: set[str] = set()
    cur = parents.get(term.strip().lower())
    while cur and cur not in valid_categories and cur.lower() not in seen and len(out) < max_depth:
        seen.add(cur.lower())
        out.append(cur)
        cur = parents.get(cur.strip().lower())
    return out


def _build_taxonomy_payload(rows: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in rows.iterrows():
        out.append({
            "id": int(r["_critic_id"]),
            "term": r["Term"],
            "parent_term": r["Parent_Term"],
            "category": r.get("Category", ""),
            "is_intermediate": bool(r.get("Is_Intermediate", False)),
            "nld": str(r.get("NLD", ""))[:400],
        })
    return out


def _build_parent_context(
    chunk: pd.DataFrame,
    cat_term_nld: dict[str, str],
    valid_categories: set[str],
) -> list[dict]:
    """NLDs of the chunk terms' parents (when the parent is itself a category
    term), so the Stage-1 critic can judge vacuous restatement (Check 5) even
    when the parent lives in a different chunk. Category roots and upper-class
    parents are skipped (no NLD / no collapse risk). Deduped."""
    seen: set[str] = set()
    out: list[dict] = []
    for _, r in chunk.iterrows():
        parent = str(r["Parent_Term"]).strip()
        pl = parent.lower()
        if not parent or parent in valid_categories or pl in seen:
            continue
        nld = cat_term_nld.get(pl)
        if not nld:
            continue
        seen.add(pl)
        out.append({"term": parent, "nld": str(nld)[:400]})
    return out


def _build_relation_payload(
    own_rows: pd.DataFrame,
    ancestor_rows: pd.DataFrame,
) -> list[dict]:
    """Combine own + ancestor relation rows. Ancestor rows carry chain_role='ancestor'
    and the critic is instructed not to vote on them."""
    out: list[dict] = []
    for _, r in own_rows.iterrows():
        out.append({
            "id": int(r["_critic_id"]),
            "chain_role": "own",
            "term": r["Term"],
            "property": r["Property"],
            "filler": r["Filler"],
            "evidence": str(r.get("Evidence", ""))[:300],
        })
    for _, r in ancestor_rows.iterrows():
        out.append({
            "id": int(r["_critic_id"]),
            "chain_role": "ancestor",
            "term": r["Term"],
            "property": r["Property"],
            "filler": r["Filler"],
            "evidence": str(r.get("Evidence", ""))[:200],
        })
    return out


def _build_relations_menu() -> list[dict]:
    """Rewrite/FIX property menu for the relation critic, sourced from the active
    ontology config: every relation flagged `critic_menu: true` in
    `ontology_config.yaml`. Domain-agnostic (no hardcoded names) and direction-
    correct (the YAML author picks the subject-anchored direction, which a
    mechanical inverse-dedup cannot do)."""
    cfg = get_config()
    out: list[dict] = []
    for name, pc in cfg.all_relations().items():
        if not pc.critic_menu:
            continue
        out.append({
            "name": pc.name,
            "domain": sorted(pc.domain),
            "range": sorted(pc.range),
            "inverse": pc.inverse,
        })
    return out


def _build_target_classes() -> list[str]:
    """Allowed `target_class` / `mint_parent` values for CONVERT_TO_INSTANCE.

    Sourced live from the active ontology config (BFO + GeoCore + GeoReservoir
    labels). Domain-agnostic: retargeting the pipeline to another domain
    automatically changes this list with zero code change.
    """
    return sorted(get_config().upper_iris().keys())


def _build_previously_minted(minted_csv: str) -> list[dict]:
    if not os.path.exists(minted_csv):
        return []
    try:
        df = read_csv(minted_csv)
    except Exception:
        return []
    out: list[dict] = []
    for _, r in df.iterrows():
        out.append({
            "name": str(r.get("Name", "")),
            "parent_property": str(r.get("ParentProperty", "")),
            "domain": str(r.get("Domain", "")),
            "range": str(r.get("Range", "")),
        })
    return out


def _build_taxonomy_context(rows: pd.DataFrame) -> list[dict]:
    """Lean term→NLD context for the relation critic (no ids — not votable)."""
    return [
        {"term": r["Term"], "nld": str(r.get("NLD", ""))[:300]}
        for _, r in rows.iterrows()
    ]


def _child_counts_among_survivors(survivor_rows: list[pd.Series]) -> dict[str, int]:
    """How many surviving rows name each (lower-cased) term as their parent.
    Computed in Python so the dedup critic's weak-intermediate check is
    deterministic rather than asking the model to count."""
    counts: dict[str, int] = {}
    for r in survivor_rows:
        parent = str(r["Parent_Term"]).strip().lower()
        if parent:
            counts[parent] = counts.get(parent, 0) + 1
    return counts


def _build_dedup_payload(
    survivor_rows: list[pd.Series],
    child_counts: dict[str, int],
) -> list[dict]:
    """Compact cross-term view for the Stage-2 dedup critic: every survivor with
    a short NLD, its parent, intermediate flag, and surviving child count."""
    out: list[dict] = []
    for r in survivor_rows:
        term = str(r["Term"]).strip()
        out.append({
            "id": int(r["_critic_id"]),
            "term": r["Term"],
            "parent": r["Parent_Term"],
            "is_intermediate": bool(r.get("Is_Intermediate", False)),
            "child_count": int(child_counts.get(term.lower(), 0)),
            "nld": str(r.get("NLD", ""))[:400],
        })
    return out


def _build_taxonomy_decisions(
    tax_edits_local: dict[int, dict],
    id_to_term: dict[int, str],
) -> list[dict]:
    """Compact handoff: the non-KEEP taxonomy decisions the relation critic must
    align to. KEEP is the default and omitted to keep the block small."""
    out: list[dict] = []
    for rid, edit in tax_edits_local.items():
        action = (edit.get("action") or "KEEP").upper()
        if action == "KEEP":
            continue
        term = id_to_term.get(rid)
        if not term:
            continue
        entry = {"term": term, "action": action}
        if action == "REPARENT" and (edit.get("new_parent") or "").strip():
            entry["new_parent"] = edit["new_parent"].strip()
        out.append(entry)
    return out


def _probe_cols(edit: dict | None) -> dict:
    """Extract the Option-E probe trace + OntoClean signs from a taxonomy edit
    for the audit log."""
    if not isinstance(edit, dict):
        return {"probe1_genus_ok": "", "probe2_bucket": "", "probe3_rewrite": "",
                "rigidity": "", "identity": "", "dependence": "", "carried_by": ""}
    cb = edit.get("carried_by")
    return {
        "probe1_genus_ok": edit.get("probe1_genus_ok", ""),
        "probe2_bucket": str(edit.get("probe2_bucket", "") or ""),
        "probe3_rewrite": str(edit.get("probe3_rewrite", "") or "")[:200],
        "rigidity": str(edit.get("rigidity", "") or ""),
        "identity": str(edit.get("identity", "") or ""),
        "dependence": str(edit.get("dependence", "") or ""),
        "carried_by": json.dumps(cb, ensure_ascii=False) if isinstance(cb, dict) else "",
    }


# ─── LLM calls + archive ──────────────────────────────────────────────────

def _archive(record: dict, archive_path: str, archive_lock: threading.Lock) -> None:
    with archive_lock:
        with open(archive_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _call_taxonomy_critic(
    category: str,
    terms_payload: list[dict],
    relations_context: list[dict],
    parent_context: list[dict],
    target_classes: list[str],
    system_instruction: str,
    prompt_template: str,
    model: str,
    temperature: float,
    archive_path: str,
    archive_lock: threading.Lock,
) -> list[dict]:
    """Call 1: judge taxonomy rows. Returns the `edits` list."""
    prompt = prompt_template.format(
        category=category,
        terms_json=json.dumps(terms_payload, indent=2),
        relations_context_json=json.dumps(relations_context, indent=2) if relations_context else "[]",
        parent_context_json=json.dumps(parent_context, indent=2) if parent_context else "[]",
        target_classes_json=json.dumps(target_classes, indent=2),
    )
    raw_text = ""
    edits: list[dict] = []
    try:
        raw_text = generate(
            prompt, model=model, system_instruction=system_instruction,
            temperature=temperature, response_mime_type="application/json",
        )
        data = json.loads(raw_text)
        edits = data.get("edits", []) if isinstance(data, dict) else []
        if not isinstance(edits, list):
            edits = []
    except Exception as e:
        tqdm.write(f"  [{category}] taxonomy critic failed ({e}); defaulting to KEEP")
        edits = []
    _archive({
        "timestamp": datetime.now(timezone.utc).isoformat(), "call": "taxonomy",
        "category": category, "model": model, "n_terms": len(terms_payload),
        "n_edits": len(edits), "response_text": raw_text,
    }, archive_path, archive_lock)
    return edits


def _call_relation_critic(
    category: str,
    relations_payload: list[dict],
    relations_menu: list[dict],
    previously_minted: list[dict],
    taxonomy_context: list[dict],
    taxonomy_decisions: list[dict],
    system_instruction: str,
    prompt_template: str,
    model: str,
    temperature: float,
    archive_path: str,
    archive_lock: threading.Lock,
) -> list[dict]:
    """Call 2: judge relation rows. Returns the `edits` list."""
    prompt = prompt_template.format(
        category=category,
        relations_json=json.dumps(relations_payload, indent=2) if relations_payload else "[]",
        relations_menu_json=json.dumps(relations_menu, indent=2),
        previously_minted_json=json.dumps(previously_minted, indent=2) if previously_minted else "[]",
        taxonomy_context_json=json.dumps(taxonomy_context, indent=2) if taxonomy_context else "[]",
        taxonomy_decisions_json=json.dumps(taxonomy_decisions, indent=2) if taxonomy_decisions else "[]",
    )
    raw_text = ""
    edits: list[dict] = []
    try:
        raw_text = generate(
            prompt, model=model, system_instruction=system_instruction,
            temperature=temperature, response_mime_type="application/json",
        )
        data = json.loads(raw_text)
        if isinstance(data, dict) and isinstance(data.get("edits"), list):
            edits = data["edits"]
    except Exception as e:
        tqdm.write(f"  [{category}] relation critic failed ({e}); defaulting to KEEP")
        edits = []
    _archive({
        "timestamp": datetime.now(timezone.utc).isoformat(), "call": "relation",
        "category": category, "model": model, "n_relations": len(relations_payload),
        "n_edits": len(edits), "response_text": raw_text,
    }, archive_path, archive_lock)
    return edits


def _call_dedup_critic(
    category: str,
    survivors_payload: list[dict],
    system_instruction: str,
    prompt_template: str,
    model: str,
    temperature: float,
    archive_path: str,
    archive_lock: threading.Lock,
) -> list[dict]:
    """Stage 2: cross-term dedup over a category's survivors. The prompt emits
    only DROP_AS_REDUNDANT rows (KEEP is implicit), so there is no completeness
    guard — a sparse or empty response simply means 'keep everything'."""
    prompt = prompt_template.format(
        category=category,
        survivors_json=json.dumps(survivors_payload, indent=2),
    )
    raw_text = ""
    edits: list[dict] = []
    try:
        raw_text = generate(
            prompt, model=model, system_instruction=system_instruction,
            temperature=temperature, response_mime_type="application/json",
        )
        data = json.loads(raw_text)
        if isinstance(data, dict) and isinstance(data.get("edits"), list):
            edits = data["edits"]
    except Exception as e:
        tqdm.write(f"  [{category}] dedup critic failed ({e}); keeping all survivors")
        edits = []
    _archive({
        "timestamp": datetime.now(timezone.utc).isoformat(), "call": "dedup",
        "category": category, "model": model, "n_survivors": len(survivors_payload),
        "n_edits": len(edits), "response_text": raw_text,
    }, archive_path, archive_lock)
    return edits


def _ask_complete(
    invoke,
    full_payload: list[dict],
    expected_ids: set[int],
    category: str,
    kind: str,
) -> list[dict]:
    """Call `invoke(payload)` and re-ask once for any expected id the model
    omitted. `invoke` returns an edits list. Rows still missing after the retry
    fall back to implicit KEEP downstream."""
    edits = invoke(full_payload)
    got = {e["id"] for e in edits if isinstance(e, dict) and isinstance(e.get("id"), int)}
    missing = expected_ids - got
    if missing:
        log.warn(f"  [{category}] {kind}: {len(missing)}/{len(expected_ids)} ids omitted — re-asking")
        subset = [p for p in full_payload if p.get("id") in missing]
        if subset:
            more = invoke(subset)
            edits = edits + [e for e in more if isinstance(e, dict)]
            got = {e["id"] for e in edits if isinstance(e, dict) and isinstance(e.get("id"), int)}
            still = expected_ids - got
            if still:
                log.warn(f"  [{category}] {kind}: {len(still)} ids still missing after retry (implicit KEEP)")
    return edits


def _mutual_drop_guard(
    tax_edits_local: dict[int, dict],
    id_to_term: dict[int, str],
    id_to_parent: dict[int, str],
) -> None:
    """Never lose a concept to a dangling redundancy drop.

    A DROP_AS_REDUNDANT survives only if the term it collapses into (the
    `survivor` field, or, for Stage-1 parent-collapse, the row's parent) is
    itself kept. If the cited survivor was *also* dropped — e.g. two near-
    synonyms each pointing at the other, or a parent that got dropped — the
    drop is reverted to KEEP so at least one representative of the concept
    remains. Mutates `tax_edits_local` in place. Single-pass over a snapshot of
    the dropped set: conservative (may keep an extra near-duplicate in long
    chains) but it can never delete the last survivor of a concept.
    """
    dropped_terms = {
        id_to_term[rid].strip().lower()
        for rid, e in tax_edits_local.items()
        if rid in id_to_term and (e.get("action", "") or "").upper() in _DROP_TAX_VERDICTS
    }
    for rid, e in list(tax_edits_local.items()):
        if (e.get("action", "") or "").upper() != "DROP_AS_REDUNDANT":
            continue
        survivor = (e.get("survivor") or "").strip() or id_to_parent.get(rid, "")
        if survivor and survivor.strip().lower() not in dropped_terms:
            continue  # survivor is alive — the drop is safe
        reverted = dict(e)
        reverted["action"] = "KEEP"
        note = "survivor also dropped" if survivor else "no survivor cited"
        reverted["reason"] = f"(dedup reverted — {note}) {e.get('reason', '')}".strip()
        tax_edits_local[rid] = reverted
        log.detail(f"Mutual-drop guard: kept '{id_to_term.get(rid, rid)}' ({note})")


# ─── Edit appliers ────────────────────────────────────────────────────────

def _apply_taxonomy_edits(
    tax: pd.DataFrame,
    edits_by_id: dict[int, dict],
    valid_categories: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict]]:
    """Apply 6-verdict taxonomy edits.

    Returns:
        cleaned_tax_df — surviving taxonomy rows (drops applied, parents fixed)
        instances_df   — rows converted to NamedIndividuals
        log_rows       — full audit-log rows
        bearer_records — KEEP_AS_BEARER carries (bearer kept under its material
                         genus; the realizable/quality becomes a companion axiom)
    """
    log_rows: list[dict] = []
    keep_mask = [True] * len(tax)
    new_parents: dict[int, str] = {}
    instance_records: list[dict] = []
    bearer_records: list[dict] = []
    # A KEEP_AS_BEARER companion filler is only ever a freshly-minted
    # realizable/quality class. Reject any carry whose filler collides with an
    # existing class: minting `<filler> ⊑ role / quality / …` would retype it.
    # Upper-ontology classes are off-limits (we never alter a published
    # ontology), and reusing an existing domain term can force a continuant into
    # `quality`/`role` and make the ontology inconsistent.
    upper_lower = {k.strip().lower() for k in get_config().upper_iris()}
    existing_term_lower = {str(t).strip().lower() for t in tax["Term"].astype(str)}

    for idx, row in tax.iterrows():
        rid = int(row["_critic_id"])
        cat = row["Category"]
        edit = edits_by_id.get(rid)
        probes = _probe_cols(edit)
        if edit is None:
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "KEEP",
                             "reason": "(implicit — not mentioned by critic)", **probes})
            continue
        action = (edit.get("action") or "KEEP").upper()
        if action not in _TAXONOMY_VERDICTS:
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "KEEP",
                             "reason": f"(unknown verdict {action!r} — kept)", **probes})
            continue
        reason = edit.get("reason", "")
        if action == "KEEP":
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "KEEP", "reason": reason, **probes})
        elif action == "REPARENT":
            new_parent = (edit.get("new_parent") or "").strip()
            if not new_parent:
                log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                                 "term": row["Term"], "action": "KEEP",
                                 "reason": f"(REPARENT rejected — no new_parent) {reason}", **probes})
                continue
            new_parents[rid] = new_parent
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "REPARENT",
                             "reason": f"parent → {new_parent}: {reason}", **probes})
        elif action == "KEEP_AS_BEARER":
            # Keep the material bearer; carry the realizable / quality off the
            # IS-A edge as a companion axiom (run_critic mints the filler class
            # and the `<bearer> <property> some <filler>` row). `new_parent` is
            # OPTIONAL — the bearer stays under its current (already-material)
            # parent unless the critic names a better genus, and a genus is only
            # ever REUSED (an existing class), never minted here.
            cb = edit.get("carried_by") if isinstance(edit.get("carried_by"), dict) else {}
            prop = str(cb.get("property", "") or "").strip()
            filler = str(cb.get("filler", "") or "").strip()
            if prop not in _BEARER_PROPERTIES or not filler:
                # The carry itself is incomplete — reject to KEEP so the term is
                # never lost.
                log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                                 "term": row["Term"], "action": "KEEP",
                                 "reason": f"(KEEP_AS_BEARER rejected — incomplete carry) {reason}", **probes})
                continue
            filler_key = filler.strip().lower()
            if filler_key in upper_lower or filler_key in existing_term_lower:
                # The filler names an existing class — minting it under a BFO
                # realizable/quality parent would retype that class (forbidden
                # for upper ontologies, unsound for domain continuants). Reject
                # to KEEP so neither the bearer nor the named class is mutated.
                where = "upper-ontology" if filler_key in upper_lower else "existing domain"
                log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                                 "term": row["Term"], "action": "KEEP",
                                 "reason": f"(KEEP_AS_BEARER rejected — filler '{filler}' is an {where} class; would retype it) {reason}", **probes})
                continue
            new_parent = (edit.get("new_parent") or "").strip()
            # Move the bearer only when a non-realizable genus is named (the
            # orphan pass reuses it if it exists, else falls back). Absent or a
            # realizable branch → keep the current material parent untouched.
            if new_parent and new_parent.strip().lower() not in _BEARER_PROPERTIES.values():
                new_parents[rid] = new_parent
            bearer_records.append({
                "bearer": str(row["Term"]).strip(),
                "bearer_category": cat,
                "property": prop,
                "filler": filler,
                "filler_parent": _BEARER_PROPERTIES[prop],
            })
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "KEEP_AS_BEARER",
                             "reason": f"bearer kept under '{new_parent or row['Parent_Term']}'; {prop} some {filler}: {reason}", **probes})
        elif action in _DROP_TAX_VERDICTS:
            keep_mask[idx] = False
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": action, "reason": reason, **probes})
            if action == "CONVERT_TO_INSTANCE":
                instance_records.append({
                    "Term": row["Term"],
                    "Target_Class": (edit.get("target_class") or "").strip() or "owl:NamedIndividual",
                    "Mint_Parent": (edit.get("mint_parent") or "").strip(),
                    "Original_Category": cat,
                    "Original_Parent": row["Parent_Term"],
                    "Reason": reason,
                })

    cleaned = tax.loc[keep_mask].copy()
    if new_parents:
        cleaned["Parent_Term"] = cleaned.apply(
            lambda r: new_parents.get(int(r["_critic_id"]), r["Parent_Term"]),
            axis=1,
        )

    # Orphan re-parenting: any child whose parent was DROPped → re-parent to
    # Category. A parent that resolves to an upper-ontology class (e.g. a
    # REPARENT to `role`/`quality`/`object`) is a legitimate target, NOT an
    # orphan — recognising it prevents silently undoing role-reparenting.
    surviving_lower = {str(t).lower() for t in cleaned["Term"].astype(str)}
    for idx, row in cleaned.iterrows():
        parent = str(row["Parent_Term"]).strip()
        if not parent or parent in valid_categories:
            continue
        if parent.lower() in surviving_lower or parent.lower() in upper_lower:
            continue
        log.detail(f"Orphan re-parent: '{row['Term']}' → '{row['Category']}' (was '{parent}', dropped)")
        cleaned.at[idx, "Parent_Term"] = row["Category"]

    instances_df = pd.DataFrame(instance_records) if instance_records else pd.DataFrame(
        columns=["Term", "Target_Class", "Mint_Parent", "Original_Category", "Original_Parent", "Reason"]
    )
    return cleaned, instances_df, log_rows, bearer_records


def _materialize_bearer_carries(
    bearer_records: list[dict],
    tax_cols: list[str] | None,
    rel_cols: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn KEEP_AS_BEARER carries into the rows emit consumes:

    * a minted **filler class** row `<filler> ⊑ <bfo parent>` (role / function /
      disposition / quality), deduplicated by filler name, and
    * a **companion relation** row `<bearer> <property> some <filler>`.

    Column order is aligned to the existing taxonomy / relation outputs so the
    appended rows merge cleanly. The bearer itself is already kept under its
    material genus by `_apply_taxonomy_edits` (via `new_parents`)."""
    if not bearer_records:
        return pd.DataFrame(), pd.DataFrame()
    filler_rows: dict[str, dict] = {}
    rel_rows: list[dict] = []
    for rec in bearer_records:
        filler = rec["filler"].strip()
        prop = rec["property"].strip()
        fp = rec["filler_parent"]
        key = filler.lower()
        if key and key not in filler_rows:
            filler_rows[key] = {
                "Term": filler, "Parent_Term": fp, "Relationship_Type": "subClassOf",
                "Category": fp, "Is_Intermediate": True, "NLD": "", "FALLBACK": False,
            }
        pc = PROPERTY_CONSTRAINTS.get(prop)
        rel_rows.append({
            "Term": rec["bearer"], "Category": rec.get("bearer_category", ""),
            "Property": prop, "Property_IRI": pc.iri if pc else "",
            "Filler": filler, "Filler_Source": "critic_bearer",
            "Confidence": 1.0, "Evidence": f"KEEP_AS_BEARER companion ({prop} some {filler})",
            "Validation_Status": "ACCEPTED", "Validation_Reason": "critic KEEP_AS_BEARER",
        })
    filler_df = pd.DataFrame(list(filler_rows.values()))
    rel_df = pd.DataFrame(rel_rows)
    if tax_cols and not filler_df.empty:
        filler_df = filler_df.reindex(columns=tax_cols)
    if rel_cols and not rel_df.empty:
        rel_df = rel_df.reindex(columns=rel_cols)
    return filler_df, rel_df


def _apply_relation_edits(
    rel: pd.DataFrame,
    edits_by_id: dict[int, dict],
    minted_collector: list[dict],
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply KEEP/DROP/FIX to relation rows. FIX with a `mint` block appends
    a record to `minted_collector` for downstream persistence."""
    log_rows: list[dict] = []
    keep_mask = [True] * len(rel)
    updates: dict[int, dict[str, str]] = {}
    ts = datetime.now(timezone.utc).isoformat()
    cfg = get_config()
    project_ns = cfg.project_namespace()

    for idx, row in rel.iterrows():
        rid = int(row["_critic_id"])
        cat = row.get("Category", "")
        edit = edits_by_id.get(rid)
        if edit is None:
            log_rows.append({"id": rid, "kind": "relation", "category": cat,
                             "term": row["Term"], "action": "KEEP",
                             "reason": "(implicit — not mentioned by critic)"})
            continue
        action = (edit.get("action") or "KEEP").upper()
        if action not in _RELATION_VERDICTS:
            log_rows.append({"id": rid, "kind": "relation", "category": cat,
                             "term": row["Term"], "action": "KEEP",
                             "reason": f"(unknown verdict {action!r} — kept)"})
            continue
        reason = edit.get("reason", "")
        if action == "DROP":
            keep_mask[idx] = False
            log_rows.append({"id": rid, "kind": "relation", "category": cat,
                             "term": row["Term"], "action": "DROP", "reason": reason})
        elif action == "FIX":
            patch: dict[str, str] = {}
            np_ = (edit.get("new_property") or "").strip()
            nf_ = (edit.get("new_filler") or "").strip()
            mint = edit.get("mint") or {}
            if np_ and isinstance(mint, dict) and mint:
                # Mint a new property. Caller persists collector to CSV after the run.
                name = np_
                iri = f"{project_ns}{name}"
                minted_collector.append({
                    "Name": name,
                    "IRI": iri,
                    "ParentProperty": str(mint.get("parent_property", "")).strip(),
                    "Domain": str(mint.get("domain", "")).strip(),
                    "Range": str(mint.get("range", "")).strip(),
                    "Justification": str(mint.get("justification", "")).strip(),
                    "Timestamp": ts,
                })
            if np_:
                patch["Property"] = np_
            if nf_:
                patch["Filler"] = nf_
            if patch:
                updates[rid] = patch
                bits = ", ".join(f"{k}→{v}" for k, v in patch.items())
                tag = " [MINT]" if mint else ""
                log_rows.append({"id": rid, "kind": "relation", "category": cat,
                                 "term": row["Term"], "action": "FIX",
                                 "reason": f"{bits}{tag}: {reason}"})
            else:
                log_rows.append({"id": rid, "kind": "relation", "category": cat,
                                 "term": row["Term"], "action": "KEEP",
                                 "reason": f"(FIX rejected — no new field) {reason}"})
        else:  # KEEP
            log_rows.append({"id": rid, "kind": "relation", "category": cat,
                             "term": row["Term"], "action": "KEEP", "reason": reason})

    cleaned = rel.loc[keep_mask].copy()
    if updates:
        for rid, patch in updates.items():
            mask = cleaned["_critic_id"] == rid
            for col, val in patch.items():
                cleaned.loc[mask, col] = val
    return cleaned, log_rows


def _normalize_and_revalidate_relations(
    cleaned_rel: pd.DataFrame,
    term_to_cat: dict[str, str],
) -> tuple[pd.DataFrame, list[dict], int]:
    """Finalise relations after the critic's edits, in one pass:

    1. **Re-normalise mereology** — specialization (`has_part` →
       `has_continuant_part`, …) is a pure function of the subject/filler
       metatypes but only runs at extraction. A critic FIX can genericise a
       property or swap the filler, so every row is genericised then
       re-specialised against its current subject/filler categories.
       `Property_IRI` is refreshed to match.
    2. **Re-validate domain/range** — a FIX can also make a relation
       BFO-invalid (wrong property for the metatypes, or a forbidden
       continuant↔occurrent parthood). Such rows are dropped and logged.
       This is the same `validate_relation` check the extractor runs — applied
       again because the critic is a second LLM mutation of the relations.

    Both checks are skipped for a row whose subject or filler category is
    unresolvable (external/upper-class filler): we cannot judge it, so we leave
    it exactly as the extractor accepted it (never drop on uncertainty).
    Non-mereological, still-valid properties pass through untouched.
    """
    if cleaned_rel.empty or "Property" not in cleaned_rel.columns:
        return cleaned_rel, [], 0
    n_norm = 0
    drop_log: list[dict] = []
    keep_idx: list = []
    for idx, row in cleaned_rel.iterrows():
        prop = str(row["Property"]).strip()
        subj_cat = str(row.get("Category", "")).strip()
        filler = str(row.get("Filler", "")).strip()
        filler_cat = term_to_cat.get(filler.lower(), "")

        # 1. Re-specialise mereology.
        new_prop = normalize_property(prop, subj_cat, filler_cat)
        if new_prop != prop:
            cleaned_rel.at[idx, "Property"] = new_prop
            pc = PROPERTY_CONSTRAINTS.get(new_prop)
            if pc is not None and "Property_IRI" in cleaned_rel.columns:
                cleaned_rel.at[idx, "Property_IRI"] = pc.iri
            prop = new_prop
            n_norm += 1

        # 2. Re-validate domain/range — only when both categories resolve.
        if get_metatypes(subj_cat) is not None and filler_cat and get_metatypes(filler_cat) is not None:
            ok, reason = validate_relation(subj_cat, prop, filler_cat)
            if not ok:
                drop_log.append({
                    "id": int(row.get("_critic_id", -1)), "kind": "relation",
                    "category": subj_cat, "term": row.get("Term", ""),
                    "action": "DROP", "reason": f"(post-critic re-validation — {reason})",
                })
                continue  # drop this row
        keep_idx.append(idx)

    cleaned_rel = cleaned_rel.loc[keep_idx].copy()
    return cleaned_rel, drop_log, n_norm


# ─── Orchestrator ─────────────────────────────────────────────────────────

def run_critic(
    taxonomy_csv: str,
    output_dir: str,
    *,
    relations_csv: str | None = None,
) -> tuple[str, str | None]:
    from dotenv import load_dotenv
    load_dotenv()
    get_client()

    os.makedirs(output_dir, exist_ok=True)
    tax_out = os.path.join(output_dir, "validate_taxonomy.csv")
    rel_out = os.path.join(output_dir, "validate_relations.csv") if relations_csv else None
    edits_out = os.path.join(output_dir, "validate_edits.csv")
    instances_out = os.path.join(output_dir, "validate_instances.csv")
    minted_out = os.path.join(output_dir, "validate_minted_properties.csv")

    archive_dir = os.path.join(output_dir, "validate_responses_archive")
    os.makedirs(archive_dir, exist_ok=True)
    archive_path = os.path.join(
        archive_dir,
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".jsonl",
    )
    archive_lock = threading.Lock()

    tax_system, tax_template = load_prompt("critic_taxonomy.txt")
    dedup_system, dedup_template = load_prompt("critic_taxonomy_dedup.txt")
    rel_system, rel_template = load_prompt("critic_relations.txt")
    model = os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    temperature = float(os.environ.get("LLM_GENERATION_TEMPERATURE", 0))

    log.banner("validate", "Validate (taxonomy + dedup + relation critic per category)")

    tax = read_csv(taxonomy_csv).reset_index(drop=True)
    tax["_critic_id"] = range(len(tax))
    log.info(f"Loaded {len(tax)} taxonomy rows from {taxonomy_csv}")

    rel: pd.DataFrame | None = None
    if relations_csv and os.path.exists(relations_csv):
        rel = read_csv(relations_csv).reset_index(drop=True)
        if "Validation_Status" in rel.columns:
            n_before = len(rel)
            rel = rel[rel["Validation_Status"].fillna("ACCEPTED") == "ACCEPTED"].copy()
            log.info(f"Loaded {len(rel)} ACCEPTED relations (skipped {n_before - len(rel)} REJECTED)")
        else:
            log.info(f"Loaded {len(rel)} relations from {relations_csv}")
        rel = rel.reset_index(drop=True)
        rel["_critic_id"] = range(len(rel))
        if "Category" not in rel.columns or rel["Category"].isna().all():
            t2c = _term_to_category(tax)
            rel["Category"] = rel["Term"].astype(str).str.strip().str.lower().map(t2c).fillna("")

    valid_categories = set(tax["Category"].astype(str).unique())
    categories = sorted(valid_categories)
    relations_menu = _build_relations_menu()
    target_classes = _build_target_classes()
    previously_minted = _build_previously_minted(minted_out)
    parent_lookup = _term_to_parent(tax)

    all_tax_edits: dict[int, dict] = {}
    all_rel_edits: dict[int, dict] = {}
    edits_lock = threading.Lock()
    max_workers = int(os.environ.get("MAX_CONCURRENT_CRITIC", 5))

    def _critique_category(cat: str) -> None:
        tax_group = tax[tax["Category"] == cat]
        if len(tax_group) == 0:
            return

        # Category-level relations (used by the relation critic + cheap-skip).
        own_rel = pd.DataFrame()
        ancestor_rel = pd.DataFrame()
        if rel is not None:
            own_rel = rel[rel["Term"].astype(str).str.strip().str.lower().isin(
                {str(t).strip().lower() for t in tax_group["Term"]}
            )]
            # Ancestor relations: walk parents of every term in the group, collect rels.
            ancestor_terms: set[str] = set()
            for term in tax_group["Term"].astype(str):
                for anc in _ancestor_chain(term, parent_lookup, valid_categories):
                    ancestor_terms.add(anc.strip().lower())
            if ancestor_terms:
                ancestor_rel = rel[rel["Term"].astype(str).str.strip().str.lower().isin(ancestor_terms)]

        # Skip cheap cases (nothing to judge).
        if len(tax_group) < 2 and len(own_rel) == 0:
            return

        id_to_term = {int(r["_critic_id"]): str(r["Term"]) for _, r in tax_group.iterrows()}
        id_to_parent = {int(r["_critic_id"]): str(r["Parent_Term"]).strip() for _, r in tax_group.iterrows()}
        cat_term_nld = {
            str(r["Term"]).strip().lower(): str(r.get("NLD", ""))
            for _, r in tax_group.iterrows()
        }

        # ── Stage 1: per-term taxonomy critic, CHUNKED ──
        tax_edits_local: dict[int, dict] = {}
        chunk_size = max(1, int(os.environ.get("CRITIC_TAXONOMY_CHUNK_SIZE", 5)))
        for start in range(0, len(tax_group), chunk_size):
            chunk = tax_group.iloc[start:start + chunk_size]
            chunk_terms_lower = {str(t).strip().lower() for t in chunk["Term"]}

            # Per-chunk relation context (only the chunk terms' own + ancestors).
            chunk_own_rel = pd.DataFrame()
            chunk_anc_rel = pd.DataFrame()
            if rel is not None:
                chunk_own_rel = rel[rel["Term"].astype(str).str.strip().str.lower().isin(chunk_terms_lower)]
                chunk_anc: set[str] = set()
                for term in chunk["Term"].astype(str):
                    for anc in _ancestor_chain(term, parent_lookup, valid_categories):
                        chunk_anc.add(anc.strip().lower())
                if chunk_anc:
                    chunk_anc_rel = rel[rel["Term"].astype(str).str.strip().str.lower().isin(chunk_anc)]

            rel_context = _build_relation_payload(chunk_own_rel, chunk_anc_rel)
            parent_context = _build_parent_context(chunk, cat_term_nld, valid_categories)
            tax_payload = _build_taxonomy_payload(chunk)

            def _invoke_tax(payload: list[dict], _rc=rel_context, _pc=parent_context) -> list[dict]:
                return _call_taxonomy_critic(
                    cat, payload, _rc, _pc, target_classes,
                    tax_system, tax_template, model, temperature,
                    archive_path, archive_lock,
                )

            chunk_edits = _ask_complete(
                _invoke_tax, tax_payload, {p["id"] for p in tax_payload}, cat, "taxonomy",
            )
            for e in chunk_edits:
                if isinstance(e, dict) and isinstance(e.get("id"), int):
                    tax_edits_local[e["id"]] = e

        # ── Stage 2: cross-term dedup over survivors ──
        def _is_dropped(rid: int) -> bool:
            e = tax_edits_local.get(rid)
            return bool(e) and (e.get("action", "") or "").upper() in _DROP_TAX_VERDICTS

        survivor_rows = [r for _, r in tax_group.iterrows() if not _is_dropped(int(r["_critic_id"]))]
        if len(survivor_rows) >= 2:
            child_counts = _child_counts_among_survivors(survivor_rows)
            dedup_payload = _build_dedup_payload(survivor_rows, child_counts)
            dedup_edits = _call_dedup_critic(
                cat, dedup_payload, dedup_system, dedup_template,
                model, temperature, archive_path, archive_lock,
            )
            for e in dedup_edits:
                if not (isinstance(e, dict) and isinstance(e.get("id"), int)):
                    continue
                if (e.get("action", "") or "").upper() == "DROP_AS_REDUNDANT":
                    tax_edits_local[e["id"]] = e  # override the Stage-1 KEEP

        # ── Mutual-drop guard: never lose a concept to a dangling redundancy ──
        _mutual_drop_guard(tax_edits_local, id_to_term, id_to_parent)

        # ── Handoff: merged taxonomy decisions inform the relation critic ──
        decisions = _build_taxonomy_decisions(tax_edits_local, id_to_term)

        # ── Call 2: relation critic (only if there are own relations) ──
        rel_edits_local: dict[int, dict] = {}
        if len(own_rel) > 0:
            rel_payload = _build_relation_payload(own_rel, ancestor_rel)
            tax_context = _build_taxonomy_context(tax_group)
            own_ids = {int(r["_critic_id"]) for _, r in own_rel.iterrows()}

            def _invoke_rel(payload: list[dict]) -> list[dict]:
                return _call_relation_critic(
                    cat, payload, relations_menu, previously_minted,
                    tax_context, decisions,
                    rel_system, rel_template, model, temperature,
                    archive_path, archive_lock,
                )

            rel_edits_list = _ask_complete(
                _invoke_rel, rel_payload, own_ids, cat, "relation",
            )
            rel_edits_local = {
                e["id"]: e for e in rel_edits_list
                if isinstance(e, dict) and isinstance(e.get("id"), int)
            }

        with edits_lock:
            all_tax_edits.update(tax_edits_local)
            all_rel_edits.update(rel_edits_local)

    workers = min(max_workers, len(categories)) if categories else 1
    log.info(f"Running critic on {len(categories)} categories with {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_critique_category, cat): cat for cat in categories}
        with tqdm(total=len(categories), desc="Critic per category", unit="cat") as pbar:
            for future in as_completed(futures):
                cat = futures[future]
                try:
                    future.result()
                    pbar.set_postfix_str(cat[:30])
                except Exception as e:
                    tqdm.write(f"  [error] Category '{cat}': {e}")
                pbar.update(1)

    cleaned_tax, instances_df, tax_log, bearer_records = _apply_taxonomy_edits(
        tax, all_tax_edits, valid_categories
    )

    dropped_taxonomy_terms = {
        str(entry["term"]).strip().lower()
        for entry in tax_log
        if entry["action"] in _DROP_TAX_VERDICTS
    }

    # KEEP_AS_BEARER carries → minted filler classes (taxonomy) + companion
    # relations. Built once; appended to each output below.
    tax_cols = [c for c in cleaned_tax.columns if c != "_critic_id"]
    rel_cols = [c for c in rel.columns if c != "_critic_id"] if rel is not None else None
    bearer_filler_df, bearer_rel_df = _materialize_bearer_carries(
        bearer_records, tax_cols, rel_cols
    )

    cleaned_tax_out = cleaned_tax.drop(columns=["_critic_id"], errors="ignore")
    if not bearer_filler_df.empty:
        cleaned_tax_out = pd.concat([cleaned_tax_out, bearer_filler_df], ignore_index=True)
    write_csv(cleaned_tax_out, tax_out)
    _bearer_note = f", +{len(bearer_filler_df)} bearer-role fillers" if not bearer_filler_df.empty else ""
    log.success(f"Cleaned taxonomy: {len(cleaned_tax_out)} rows (was {len(tax)}{_bearer_note}) → {tax_out}")

    if not instances_df.empty:
        write_csv(instances_df, instances_out)
        log.success(f"Converted to instances: {len(instances_df)} rows → {instances_out}")

    rel_log: list[dict] = []
    minted_collector: list[dict] = []
    if rel is not None and rel_out:
        cleaned_rel, rel_log = _apply_relation_edits(rel, all_rel_edits, minted_collector)
        if dropped_taxonomy_terms:
            phantom_mask = cleaned_rel["Filler"].astype(str).str.strip().str.lower().isin(dropped_taxonomy_terms)
            phantom_rows = cleaned_rel[phantom_mask]
            for _, r in phantom_rows.iterrows():
                rel_log.append({
                    "id": int(r.get("_critic_id", -1)),
                    "kind": "relation",
                    "category": r.get("Category", ""),
                    "term": r["Term"],
                    "action": "DROP",
                    "reason": f"(phantom-filler cleanup — filler '{r['Filler']}' dropped from taxonomy)",
                })
            cleaned_rel = cleaned_rel[~phantom_mask]
        # Finalise: re-specialise mereology + re-validate domain/range after the critic.
        cleaned_rel, reval_drops, n_norm = _normalize_and_revalidate_relations(
            cleaned_rel, _term_to_category(tax)
        )
        if n_norm:
            log.detail(f"Re-normalised {n_norm} mereological propert{'y' if n_norm == 1 else 'ies'} post-critic")
        if reval_drops:
            rel_log.extend(reval_drops)
            log.warn(f"Post-critic re-validation dropped {len(reval_drops)} BFO-invalid relation(s)")
        cleaned_rel = cleaned_rel.drop(columns=["_critic_id"], errors="ignore")
        if not bearer_rel_df.empty:
            cleaned_rel = pd.concat([cleaned_rel, bearer_rel_df], ignore_index=True)
        write_csv(cleaned_rel, rel_out)
        _comp_note = f", +{len(bearer_rel_df)} bearer companions" if not bearer_rel_df.empty else ""
        log.success(f"Cleaned relations: {len(cleaned_rel)} rows (was {len(rel)}{_comp_note}) → {rel_out}")

    if minted_collector:
        # Merge with any existing minted CSV (preserve prior runs' mints).
        existing_minted_df = read_csv(minted_out) if os.path.exists(minted_out) else pd.DataFrame()
        new_minted_df = pd.DataFrame(minted_collector)
        merged = pd.concat([existing_minted_df, new_minted_df], ignore_index=True)
        # Deduplicate on IRI (latest wins).
        if "IRI" in merged.columns:
            merged = merged.drop_duplicates(subset=["IRI"], keep="last")
        write_csv(merged, minted_out)
        log.success(f"Minted properties: +{len(new_minted_df)} (total {len(merged)}) → {minted_out}")

    write_csv(pd.DataFrame(tax_log + rel_log), edits_out)
    by_action: dict[str, int] = {}
    for entry in tax_log + rel_log:
        by_action[entry["action"]] = by_action.get(entry["action"], 0) + 1
    log.success(
        f"Audit log: {len(tax_log) + len(rel_log)} decisions "
        f"({', '.join(f'{a}={n}' for a, n in sorted(by_action.items()))}) → {edits_out}"
    )
    log.detail(f"Raw responses archived → {archive_path}")

    return tax_out, rel_out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Ontology critic — taxonomy + relation passes (validate verb)")
    p.add_argument("taxonomy_csv")
    p.add_argument("output_dir")
    p.add_argument("--relations", default=None)
    args = p.parse_args()
    run_critic(args.taxonomy_csv, args.output_dir, relations_csv=args.relations)
