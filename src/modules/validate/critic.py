"""Ontology critic — the `validate` verb.

Two LLM calls per category, run sequentially so the second is informed by the
first:

  1. **Taxonomy critic** judges the IS-A rows (KEEP / REPARENT / DROP_AS_MIXIN /
     DROP_AS_REDUNDANT / CONVERT_TO_INSTANCE). It sees the terms + NLDs, the
     relations as read-only context, and the allowed target-class list. Its
     output carries a per-row probe trace (probe1/2/3) and, on every
     DROP_AS_MIXIN, a `carried_by` relation that should carry the lost meaning.
  2. **Relation critic** judges the object-property rows (KEEP / DROP / FIX,
     FIX may carry a `mint` block). It sees the relations + menu + previously
     minted properties, the taxonomy NLDs as context, and a **handoff** from
     call 1: the taxonomy decisions (so its edits stay consistent) and the
     `carried_by` carrier relations (which it keeps or adds via
     `added_relations`).

Calls are sequential *within* a category but the categories run in parallel on
the worker pool. A completeness guard re-asks the model for any input ids it
forgot, so large categories are not silently under-reviewed.

I/O contract:
    run_critic(taxonomy_csv, output_dir, *, relations_csv=None)
        -> (final_taxonomy_path, final_relations_path_or_None)

Outputs written to `output_dir`:
    validate_taxonomy.csv            — cleaned taxonomy
    validate_relations.csv           — cleaned relations (only if relations_csv given)
    validate_edits.csv               — full audit log (taxonomy rows carry the probe trace)
    validate_instances.csv           — terms converted to NamedIndividuals (Term, Target_Class, …)
    validate_minted_properties.csv   — newly invented ObjectProperties (provenance=critic_minted)
    validate_responses_archive/{ts}.jsonl — raw LLM responses, one line per call (call=taxonomy|relation)

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
from src.utils.gemini_client import get_client, generate
from src.utils.ontology_config import get_config
from src.utils.prompt_loader import load_prompt
from src.utils.relation_validator import PROPERTY_CONSTRAINTS, normalize_property


_TAXONOMY_VERDICTS = {
    "KEEP", "REPARENT", "DROP_AS_MIXIN", "DROP_AS_REDUNDANT", "CONVERT_TO_INSTANCE",
}
_RELATION_VERDICTS = {"KEEP", "DROP", "FIX"}

_DROP_TAX_VERDICTS = {"DROP_AS_MIXIN", "DROP_AS_REDUNDANT", "CONVERT_TO_INSTANCE"}


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
    """Extract the Option-E probe trace from a taxonomy edit for the audit log."""
    if not isinstance(edit, dict):
        return {"probe1_genus_ok": "", "probe2_bucket": "", "probe3_rewrite": "", "carried_by": ""}
    cb = edit.get("carried_by")
    return {
        "probe1_genus_ok": edit.get("probe1_genus_ok", ""),
        "probe2_bucket": str(edit.get("probe2_bucket", "") or ""),
        "probe3_rewrite": str(edit.get("probe3_rewrite", "") or "")[:200],
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


# ─── Edit appliers ────────────────────────────────────────────────────────

def _apply_taxonomy_edits(
    tax: pd.DataFrame,
    edits_by_id: dict[int, dict],
    valid_categories: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Apply 5-verdict taxonomy edits.

    Returns:
        cleaned_tax_df — surviving taxonomy rows (drops applied, parents fixed)
        instances_df   — rows converted to NamedIndividuals
        log_rows       — full audit-log rows
    """
    log_rows: list[dict] = []
    keep_mask = [True] * len(tax)
    new_parents: dict[int, str] = {}
    instance_records: list[dict] = []

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
    upper_lower = {k.lower() for k in get_config().upper_iris()}
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
    return cleaned, instances_df, log_rows


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


def _normalize_relation_mereology(
    cleaned_rel: pd.DataFrame,
    term_to_cat: dict[str, str],
) -> tuple[pd.DataFrame, int]:
    """Re-normalise mereological properties after the critic's edits.

    Specialization (`has_part` → `has_continuant_part`, …) is a pure function of
    the subject's and filler's metatypes, but it only runs at extraction time.
    A critic FIX can genericise a property or swap the filler, leaving a stale
    or over-generic parthood property. This pass genericises then re-specialises
    every row against its current subject/filler categories, keeping mereology
    correct no matter what the critic did. `Property_IRI` is refreshed to match.
    Non-mereological properties pass through untouched.
    """
    if cleaned_rel.empty or "Property" not in cleaned_rel.columns:
        return cleaned_rel, 0
    changed = 0
    for idx, row in cleaned_rel.iterrows():
        prop = str(row["Property"]).strip()
        subj_cat = str(row.get("Category", "")).strip()
        filler_cat = term_to_cat.get(str(row.get("Filler", "")).strip().lower(), "")
        new_prop = normalize_property(prop, subj_cat, filler_cat)
        if new_prop != prop:
            cleaned_rel.at[idx, "Property"] = new_prop
            pc = PROPERTY_CONSTRAINTS.get(new_prop)
            if pc is not None and "Property_IRI" in cleaned_rel.columns:
                cleaned_rel.at[idx, "Property_IRI"] = pc.iri
            changed += 1
    return cleaned_rel, changed


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
    rel_system, rel_template = load_prompt("critic_relations.txt")
    model = os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    temperature = float(os.environ.get("LLM_GENERATION_TEMPERATURE", 0))

    log.banner("validate", "Validate (taxonomy + relation critic per category)")

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

        # Own relations: rows whose Term is in this category.
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

        # ── Call 1: taxonomy critic ──
        tax_payload = _build_taxonomy_payload(tax_group)
        rel_context = _build_relation_payload(own_rel, ancestor_rel)

        def _invoke_tax(payload: list[dict]) -> list[dict]:
            return _call_taxonomy_critic(
                cat, payload, rel_context, target_classes,
                tax_system, tax_template, model, temperature,
                archive_path, archive_lock,
            )

        tax_edits_list = _ask_complete(
            _invoke_tax, tax_payload, {p["id"] for p in tax_payload}, cat, "taxonomy",
        )
        tax_edits_local = {
            e["id"]: e for e in tax_edits_list
            if isinstance(e, dict) and isinstance(e.get("id"), int)
        }

        # ── Handoff: taxonomy decisions inform the relation critic ──
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

    cleaned_tax, instances_df, tax_log = _apply_taxonomy_edits(tax, all_tax_edits, valid_categories)

    dropped_taxonomy_terms = {
        str(entry["term"]).strip().lower()
        for entry in tax_log
        if entry["action"] in _DROP_TAX_VERDICTS
    }

    cleaned_tax_out = cleaned_tax.drop(columns=["_critic_id"], errors="ignore")
    write_csv(cleaned_tax_out, tax_out)
    log.success(f"Cleaned taxonomy: {len(cleaned_tax_out)} rows (was {len(tax)}) → {tax_out}")

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
        # Re-normalise mereology after the critic's edits (genericize → re-specialize).
        cleaned_rel, n_norm = _normalize_relation_mereology(cleaned_rel, _term_to_category(tax))
        if n_norm:
            log.detail(f"Re-normalised {n_norm} mereological propert{'y' if n_norm == 1 else 'ies'} post-critic")
        cleaned_rel = cleaned_rel.drop(columns=["_critic_id"], errors="ignore")
        write_csv(cleaned_rel, rel_out)
        log.success(f"Cleaned relations: {len(cleaned_rel)} rows (was {len(rel)}) → {rel_out}")

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
