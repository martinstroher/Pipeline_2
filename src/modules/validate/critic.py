"""Simplified ontology critic — the `validate` verb.

A single LLM call per category that judges every taxonomy and relation
row as KEEP / DROP / FIX, with a full audit log.

I/O contract:
    run_critic(taxonomy_csv, output_dir, *, relations_csv=None)
        -> (final_taxonomy_path, final_relations_path_or_None)

Outputs written to `output_dir`:
    validate_taxonomy.csv   — cleaned taxonomy
    validate_relations.csv  — cleaned relations (only if relations_csv given)
    validate_edits.csv      — full audit log of every KEEP/DROP/FIX decision

Safety guards:
    - Default temperature 0 (deterministic).
    - If a category's edits would DROP > 50% of its taxonomy rows, the entire
      category's edits are reverted (warn + keep originals). Prevents runaway
      pruning when the LLM misreads a whole category.
    - Any taxonomy child whose parent was DROPped is re-parented to the
      category root so the tree stays connected.
    - Rows the LLM forgets to mention are treated as implicit KEEP.
    - After taxonomy edits are applied, any relation whose Filler was DROPped
      is also dropped (phantom-filler cleanup, logged as DROP/phantom).
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from src.utils import log
from src.utils.csv_io import read_csv, write_csv
from src.utils.gemini_client import get_client, generate
from src.utils.prompt_loader import load_prompt


_DROP_RATIO_SAFETY_LIMIT = 0.5


def _term_to_category(tax: pd.DataFrame) -> dict[str, str]:
    """Lowercase-keyed lookup so relation rows can be grouped by their
    subject term's category."""
    return {
        str(t).strip().lower(): c
        for t, c in zip(tax["Term"].tolist(), tax["Category"].tolist())
    }


def _build_taxonomy_payload(rows: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in rows.iterrows():
        is_intermediate = bool(r.get("Is_Intermediate", False))
        out.append({
            "id": int(r["_critic_id"]),
            "term": r["Term"],
            "parent_term": r["Parent_Term"],
            "relationship_type": r.get("Relationship_Type", "rdfs:subClassOf"),
            "is_intermediate": is_intermediate,
            "nld": str(r.get("NLD", ""))[:400],
        })
    return out


def _build_relation_payload(rows: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in rows.iterrows():
        out.append({
            "id": int(r["_critic_id"]),
            "term": r["Term"],
            "property": r["Property"],
            "filler": r["Filler"],
            "evidence": str(r.get("Evidence", ""))[:300],
        })
    return out


def _call_critic(
    category: str,
    tax_payload: list[dict],
    rel_payload: list[dict],
    system_instruction: str,
    prompt_template: str,
    model: str,
    temperature: float,
) -> list[dict]:
    """One LLM call per category. Returns the `edits` list (possibly empty
    if the LLM produced unparseable output — caller defaults to KEEP)."""
    prompt = prompt_template.format(
        category=category,
        taxonomy_json=json.dumps(tax_payload, indent=2),
        relations_json=json.dumps(rel_payload, indent=2) if rel_payload else "[]",
    )
    try:
        text = generate(
            prompt,
            model=model,
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
        )
        data = json.loads(text)
        edits = data.get("edits", []) if isinstance(data, dict) else []
        if not isinstance(edits, list):
            return []
        return edits
    except Exception as e:
        tqdm.write(f"  [{category}] critic call failed ({e}); defaulting to KEEP for all rows")
        return []


def _apply_taxonomy_edits(
    tax: pd.DataFrame,
    edits_by_id: dict[int, dict],
    valid_categories: set[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply DROP/FIX edits to the taxonomy. Returns (cleaned_df, log_rows).

    Re-parents orphaned children to their category root after drops.
    """
    log_rows: list[dict] = []
    keep_mask = [True] * len(tax)
    new_parents: dict[int, str] = {}  # critic_id → updated Parent_Term

    drops_per_category: dict[str, list[int]] = {}

    for idx, row in tax.iterrows():
        rid = int(row["_critic_id"])
        cat = row["Category"]
        edit = edits_by_id.get(rid)
        if edit is None:
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "KEEP",
                             "reason": "(implicit — not mentioned by critic)"})
            continue
        action = (edit.get("action") or "KEEP").upper()
        reason = edit.get("reason", "")
        if action == "DROP":
            keep_mask[idx] = False
            drops_per_category.setdefault(cat, []).append(rid)
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "DROP", "reason": reason})
        elif action == "FIX":
            new_parent = edit.get("new_parent")
            if new_parent and isinstance(new_parent, str) and new_parent.strip():
                new_parents[rid] = new_parent.strip()
                log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                                 "term": row["Term"], "action": "FIX",
                                 "reason": f"parent → {new_parent}: {reason}"})
            else:
                log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                                 "term": row["Term"], "action": "KEEP",
                                 "reason": f"(FIX rejected — no new_parent) {reason}"})
        else:  # KEEP
            log_rows.append({"id": rid, "kind": "taxonomy", "category": cat,
                             "term": row["Term"], "action": "KEEP", "reason": reason})

    # Safety guard: revert per-category if drops exceed _DROP_RATIO_SAFETY_LIMIT.
    cat_sizes = tax.groupby("Category").size().to_dict()
    reverted_cats: set[str] = set()
    for cat, dropped_ids in drops_per_category.items():
        size = cat_sizes.get(cat, 0)
        if size and len(dropped_ids) / size > _DROP_RATIO_SAFETY_LIMIT:
            reverted_cats.add(cat)
            log.warn(
                f"Category '{cat}': critic proposed dropping "
                f"{len(dropped_ids)}/{size} rows (>{_DROP_RATIO_SAFETY_LIMIT:.0%}) — "
                f"reverting category's taxonomy edits as a safety guard."
            )

    if reverted_cats:
        for idx, row in tax.iterrows():
            if row["Category"] in reverted_cats:
                keep_mask[idx] = True
                new_parents.pop(int(row["_critic_id"]), None)
        for entry in log_rows:
            if entry["kind"] == "taxonomy" and entry["category"] in reverted_cats:
                entry["action"] = "KEEP"
                entry["reason"] = "(safety revert — category drop ratio exceeded threshold)"

    # Build the surviving DataFrame and apply parent fixes.
    cleaned = tax.loc[keep_mask].copy()
    if new_parents:
        cleaned["Parent_Term"] = cleaned.apply(
            lambda r: new_parents.get(int(r["_critic_id"]), r["Parent_Term"]),
            axis=1,
        )

    # Orphan re-parenting: any child whose parent is no longer in `cleaned`
    # AND whose parent is not an upper-ontology category root → re-parent to Category.
    surviving_terms = set(cleaned["Term"].astype(str))
    surviving_terms_lower = {t.lower() for t in surviving_terms}
    for idx, row in cleaned.iterrows():
        parent = str(row["Parent_Term"]).strip()
        if not parent:
            continue
        if parent in valid_categories:
            continue  # parent is the category root — fine
        if parent.lower() not in surviving_terms_lower:
            log.detail(f"Orphan re-parent: '{row['Term']}' → '{row['Category']}' "
                       f"(was '{parent}', dropped)")
            cleaned.at[idx, "Parent_Term"] = row["Category"]

    return cleaned, log_rows


def _apply_relation_edits(
    rel: pd.DataFrame,
    edits_by_id: dict[int, dict],
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply DROP/FIX to relations. Symmetric in spirit to taxonomy edits but
    simpler — no orphan handling needed."""
    log_rows: list[dict] = []
    keep_mask = [True] * len(rel)
    updates: dict[int, dict[str, str]] = {}

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
        reason = edit.get("reason", "")
        if action == "DROP":
            keep_mask[idx] = False
            log_rows.append({"id": rid, "kind": "relation", "category": cat,
                             "term": row["Term"], "action": "DROP", "reason": reason})
        elif action == "FIX":
            patch: dict[str, str] = {}
            np_ = edit.get("new_property")
            nf_ = edit.get("new_filler")
            if isinstance(np_, str) and np_.strip():
                patch["Property"] = np_.strip()
            if isinstance(nf_, str) and nf_.strip():
                patch["Filler"] = nf_.strip()
            if patch:
                updates[rid] = patch
                bits = ", ".join(f"{k}→{v}" for k, v in patch.items())
                log_rows.append({"id": rid, "kind": "relation", "category": cat,
                                 "term": row["Term"], "action": "FIX",
                                 "reason": f"{bits}: {reason}"})
            else:
                log_rows.append({"id": rid, "kind": "relation", "category": cat,
                                 "term": row["Term"], "action": "KEEP",
                                 "reason": f"(FIX rejected — no new field) {reason}"})
        else:
            log_rows.append({"id": rid, "kind": "relation", "category": cat,
                             "term": row["Term"], "action": "KEEP", "reason": reason})

    cleaned = rel.loc[keep_mask].copy()
    if updates:
        for rid, patch in updates.items():
            mask = cleaned["_critic_id"] == rid
            for col, val in patch.items():
                cleaned.loc[mask, col] = val
    return cleaned, log_rows


def run_critic(
    taxonomy_csv: str,
    output_dir: str,
    *,
    relations_csv: str | None = None,
) -> tuple[str, str | None]:
    """Run the simplified single-call-per-category critic."""
    from dotenv import load_dotenv
    load_dotenv()
    get_client()

    os.makedirs(output_dir, exist_ok=True)
    tax_out = os.path.join(output_dir, "validate_taxonomy.csv")
    rel_out = os.path.join(output_dir, "validate_relations.csv") if relations_csv else None
    edits_out = os.path.join(output_dir, "validate_edits.csv")

    system_instruction, prompt_template = load_prompt("critic.txt")
    model = os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    temperature = float(os.environ.get("LLM_GENERATION_TEMPERATURE", 0))

    log.banner("validate", "Validate (single LLM critic per category)")

    tax = read_csv(taxonomy_csv).reset_index(drop=True)
    tax["_critic_id"] = range(len(tax))
    log.info(f"Loaded {len(tax)} taxonomy rows from {taxonomy_csv}")

    rel: pd.DataFrame | None = None
    if relations_csv and os.path.exists(relations_csv):
        rel = read_csv(relations_csv).reset_index(drop=True)
        # Critic operates only on ACCEPTED relations; rejected ones stay rejected.
        if "Validation_Status" in rel.columns:
            n_before = len(rel)
            rel = rel[rel["Validation_Status"].fillna("ACCEPTED") == "ACCEPTED"].copy()
            log.info(f"Loaded {len(rel)} ACCEPTED relations from {relations_csv} (skipped {n_before - len(rel)} REJECTED)")
        else:
            log.info(f"Loaded {len(rel)} relations from {relations_csv}")
        rel = rel.reset_index(drop=True)
        rel["_critic_id"] = range(len(rel))
        # Attach Category for grouping if missing.
        if "Category" not in rel.columns or rel["Category"].isna().all():
            t2c = _term_to_category(tax)
            rel["Category"] = rel["Term"].astype(str).str.strip().str.lower().map(t2c).fillna("")

    valid_categories = set(tax["Category"].astype(str).unique())
    categories = sorted(valid_categories)

    all_tax_edits: dict[int, dict] = {}
    all_rel_edits: dict[int, dict] = {}
    edits_lock = threading.Lock()
    max_workers = int(os.environ.get("MAX_CONCURRENT_CRITIC", 5))

    def _critique_category(cat: str) -> None:
        tax_group = tax[tax["Category"] == cat]
        rel_group = rel[rel["Category"] == cat] if rel is not None else pd.DataFrame()

        # Skip cheap cases — no work for the LLM to do.
        if len(tax_group) < 2 and len(rel_group) == 0:
            return

        tax_payload = _build_taxonomy_payload(tax_group)
        rel_payload = _build_relation_payload(rel_group) if not rel_group.empty else []

        edits = _call_critic(
            cat, tax_payload, rel_payload,
            system_instruction, prompt_template, model, temperature,
        )
        with edits_lock:
            for edit in edits:
                if not isinstance(edit, dict):
                    continue
                rid = edit.get("id")
                kind = edit.get("kind", "taxonomy")
                if not isinstance(rid, int):
                    continue
                if kind == "relation":
                    all_rel_edits[rid] = edit
                else:
                    all_tax_edits[rid] = edit

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

    cleaned_tax, tax_log = _apply_taxonomy_edits(tax, all_tax_edits, valid_categories)

    # Track which terms were DROPped from the taxonomy so we can prune relations
    # that reference them as fillers (otherwise OWL export emits phantom classes).
    dropped_taxonomy_terms = {
        str(entry["term"]).strip().lower()
        for entry in tax_log
        if entry["action"] == "DROP"
    }

    cleaned_tax_out = cleaned_tax.drop(columns=["_critic_id"], errors="ignore")
    write_csv(cleaned_tax_out, tax_out)
    log.success(f"Cleaned taxonomy: {len(cleaned_tax_out)} rows (was {len(tax)}) → {tax_out}")

    rel_log: list[dict] = []
    if rel is not None and rel_out:
        cleaned_rel, rel_log = _apply_relation_edits(rel, all_rel_edits)
        # Phantom-filler cleanup: drop relations whose filler was DROPped from
        # the taxonomy in this same pass. Logged as DROP/phantom in audit log.
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
                    "reason": f"(phantom-filler cleanup — filler '{r['Filler']}' was DROPped from taxonomy)",
                })
            cleaned_rel = cleaned_rel[~phantom_mask]
        cleaned_rel = cleaned_rel.drop(columns=["_critic_id"], errors="ignore")
        write_csv(cleaned_rel, rel_out)
        log.success(f"Cleaned relations: {len(cleaned_rel)} rows (was {len(rel)}) → {rel_out}")

    write_csv(pd.DataFrame(tax_log + rel_log), edits_out)
    n_drops = sum(1 for r in (tax_log + rel_log) if r["action"] == "DROP")
    n_fixes = sum(1 for r in (tax_log + rel_log) if r["action"] == "FIX")
    log.success(f"Audit log: {len(tax_log) + len(rel_log)} decisions "
                f"({n_drops} DROP, {n_fixes} FIX) → {edits_out}")

    return tax_out, rel_out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Simplified single-call-per-category ontology critic")
    p.add_argument("taxonomy_csv")
    p.add_argument("output_dir")
    p.add_argument("--relations", default=None)
    args = p.parse_args()
    run_critic(args.taxonomy_csv, args.output_dir, relations_csv=args.relations)
