"""
Ontology Critic — Step 6c of the PreSaltOntoLearn pipeline.

Post-processing quality pass that uses an LLM to identify and fix:
  - Near-duplicate terms (MERGE)
  - Misplaced siblings (MOVE)
  - Vague/abstract terms (REMOVE)
  - Ambiguous names (RENAME)
  - Cross-category duplicates (CROSS_MERGE)

Input:  6_taxonomy.csv (with NLDs) + optionally 6b_relations.csv
Output: 6c_taxonomy_cleaned.csv + 6c_critic_log.csv (audit trail)
        + optionally 6c_relations_cleaned.csv
"""

import json
import os
import time

import pandas as pd
from tqdm import tqdm

from src.utils.gemini_client import get_client, generate
from src.utils import log
from src.utils.prompt_loader import load_prompt


# ── Shared helpers ──────────────────────────────────────────────────────

def _nld_summary(nld_raw, max_len: int = 250) -> str:
    """Extract first sentence of NLD, truncated to max_len."""
    nld = nld_raw if isinstance(nld_raw, str) else ""
    first_sentence = nld.split(". ")[0] + "." if nld else ""
    return first_sentence[:max_len]


def _safe_parse_actions(response_text: str, context: str) -> list[dict]:
    """Parse JSON response into action list, with error handling."""
    try:
        actions = json.loads(response_text)
        if not isinstance(actions, list):
            log.warn(f"Critic returned non-list for {context}: {type(actions)}")
            return []
        return actions
    except Exception as e:
        log.warn(f"Critic JSON parse failed for {context}: {e}")
        return []


# ── Branch helpers ──────────────────────────────────────────────────────

def _build_branch_json(group_df: pd.DataFrame) -> list[dict]:
    """Build a JSON-serializable branch representation for LLM input."""
    entries = []
    for _, row in group_df.iterrows():
        entry = {
            "term": row["Term"],
            "parent": row["Parent_Term"],
            "relationship": row["Relationship_Type"],
            "is_intermediate": bool(row.get("Is_Intermediate", False)),
        }
        nld = row.get("NLD", "") or ""
        if nld:
            entry["nld"] = nld
        entries.append(entry)
    return entries


# ── LLM calls ──────────────────────────────────────────────────────────

def _critique_branch(
    category: str,
    branch: list[dict],
    model_name: str,
    temperature: float,
) -> list[dict]:
    """Send a category branch to the LLM for critique. Returns list of actions."""
    system_instruction, prompt_template = load_prompt("ontology_critic.txt")

    prompt = prompt_template.format(
        category=category,
        n_entries=len(branch),
        branch_json=json.dumps(branch, indent=2),
    )

    try:
        response_text = generate(
            prompt,
            model=model_name,
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
        )
        return _safe_parse_actions(response_text, f"branch '{category}'")
    except Exception as e:
        log.warn(f"Critic failed for '{category}': {e}")
        return []


def _cross_category_review(
    all_terms: list[dict],
    model_name: str,
    temperature: float,
) -> list[dict]:
    """Check for duplicate/miscategorised concepts across categories."""
    system_instruction, prompt_template = load_prompt("ontology_critic_cross.txt")

    compact = []
    for t in all_terms:
        compact.append({
            "term": t["term"],
            "category": t["category"],
            "nld_summary": _nld_summary(t.get("nld", "")),
        })

    prompt = prompt_template.format(
        n_terms=len(compact),
        terms_json=json.dumps(compact, indent=2),
    )

    try:
        response_text = generate(
            prompt,
            model=model_name,
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
        )
        return _safe_parse_actions(response_text, "cross-category")
    except Exception as e:
        log.warn(f"Cross-category critic failed: {e}")
        return []


def _essentiality_review(
    all_terms: list[dict],
    model_name: str,
    temperature: float,
) -> list[dict]:
    """Final pass: ask LLM which remaining terms do NOT earn their place."""
    system_instruction, prompt_template = load_prompt(
        "ontology_critic_essentiality.txt"
    )

    compact = []
    for t in all_terms:
        compact.append({
            "term": t["term"],
            "category": t["category"],
            "parent": t.get("parent", ""),
            "has_children": t.get("has_children", False),
            "is_individual": t.get("is_individual", False),
            "nld_summary": _nld_summary(t.get("nld", "")),
        })

    prompt = prompt_template.format(
        n_terms=len(compact),
        terms_json=json.dumps(compact, indent=2),
    )

    try:
        response_text = generate(
            prompt,
            model=model_name,
            system_instruction=system_instruction,
            temperature=temperature,
            response_mime_type="application/json",
        )
        return _safe_parse_actions(response_text, "essentiality")
    except Exception as e:
        log.warn(f"Essentiality review failed: {e}")
        return []


# ── Action application ─────────────────────────────────────────────────

def _apply_actions(
    df: pd.DataFrame,
    actions: list[dict],
    log_rows: list[dict],
) -> pd.DataFrame:
    """Apply critic actions to the taxonomy DataFrame. Returns modified copy."""
    df = df.copy()

    for action in actions:
        act_type = action.get("action", "")
        reason = action.get("reason", "")

        if act_type == "MERGE":
            terms = action.get("terms", [])
            survivor = action.get("surviving_term", terms[0] if terms else "")
            to_remove = [t for t in terms if t != survivor]

            # Validate: survivor must exist in df
            if survivor not in df["Term"].values:
                log.warn(f"MERGE skipped: survivor '{survivor}' not in taxonomy")
                continue

            for dead_term in to_remove:
                if dead_term not in df["Term"].values:
                    continue
                # Reparent children of dead_term to survivor
                df.loc[df["Parent_Term"] == dead_term, "Parent_Term"] = survivor
                # Remove dead_term row
                df = df[df["Term"] != dead_term]
                log_rows.append({
                    "Action": "MERGE",
                    "Term": dead_term,
                    "Detail": f"Merged into '{survivor}'",
                    "Reason": reason,
                    "_target": survivor,
                })

        elif act_type == "REMOVE":
            term = action.get("term", "")
            if not term or term not in df["Term"].values:
                continue
            removed_rows = df[df["Term"] == term]
            parent = removed_rows.iloc[0]["Parent_Term"]
            # If parent is NaN/empty, fall back to the term's Category (only if valid)
            if not parent or (isinstance(parent, float) and pd.isna(parent)) or not str(parent).strip():
                category_val = str(removed_rows.iloc[0].get("Category", "")).strip()
                if category_val and category_val in df["Term"].values:
                    parent = category_val
                else:
                    parent = ""  # Will become a root class
            # Reparent children to the removed term's parent
            df.loc[df["Parent_Term"] == term, "Parent_Term"] = parent
            # Remove the term
            df = df[df["Term"] != term]
            log_rows.append({
                "Action": "REMOVE",
                "Term": term,
                "Detail": f"Removed (children reparented to '{parent}')",
                "Reason": reason,
            })

        elif act_type == "MOVE":
            term = action.get("term", "")
            new_parent = action.get("new_parent", "")
            if not term or not new_parent or term not in df["Term"].values:
                continue
            old_parent_vals = df.loc[df["Term"] == term, "Parent_Term"].values
            old_parent = old_parent_vals[0] if len(old_parent_vals) > 0 else "?"
            df.loc[df["Term"] == term, "Parent_Term"] = new_parent
            log_rows.append({
                "Action": "MOVE",
                "Term": term,
                "Detail": f"Moved from '{old_parent}' to '{new_parent}'",
                "Reason": reason,
            })

        elif act_type == "RENAME":
            old_name = action.get("term", "")
            new_name = action.get("new_name", "")
            if not old_name or not new_name or old_name not in df["Term"].values:
                continue
            # Rename in Term column
            df.loc[df["Term"] == old_name, "Term"] = new_name
            # Rename in Parent_Term column (for children)
            df.loc[df["Parent_Term"] == old_name, "Parent_Term"] = new_name
            log_rows.append({
                "Action": "RENAME",
                "Term": old_name,
                "Detail": f"Renamed to '{new_name}'",
                "Reason": reason,
                "_target": new_name,
            })

        elif act_type == "CROSS_MERGE":
            terms_info = action.get("terms", [])
            survivor = action.get("surviving_term", "")
            if len(terms_info) < 2 or not survivor:
                continue
            for t_info in terms_info:
                t_name = t_info.get("term", "")
                if t_name != survivor and t_name in df["Term"].values:
                    # Reparent children
                    df.loc[df["Parent_Term"] == t_name, "Parent_Term"] = survivor
                    # Remove
                    df = df[df["Term"] != t_name]
                    log_rows.append({
                        "Action": "CROSS_MERGE",
                        "Term": t_name,
                        "Detail": f"Cross-category merge into '{survivor}'",
                        "Reason": reason,
                        "_target": survivor,
                    })

        elif act_type == "CROSS_MOVE":
            term = action.get("term", "")
            new_cat = action.get("new_category", "")
            new_parent = action.get("new_parent", "")
            if not term or not new_cat or term not in df["Term"].values:
                continue
            old_cat = df.loc[df["Term"] == term, "Category"].values[0]
            old_parent = df.loc[df["Term"] == term, "Parent_Term"].values[0]
            # Update category and parent
            df.loc[df["Term"] == term, "Category"] = new_cat
            if new_parent:
                df.loc[df["Term"] == term, "Parent_Term"] = new_parent
            # Cascade: update category for direct children of moved term
            df.loc[df["Parent_Term"] == term, "Category"] = new_cat
            log_rows.append({
                "Action": "CROSS_MOVE",
                "Term": term,
                "Detail": f"Moved from '{old_cat}' (parent '{old_parent}') to '{new_cat}' (parent '{new_parent}')",
                "Reason": reason,
            })

    return df


def _apply_actions_to_relations(
    relations_df: pd.DataFrame,
    log_rows: list[dict],
) -> pd.DataFrame:
    """Apply renames, merges, removals to the relations CSV."""
    df = relations_df.copy()

    # Build rename/merge map and removal set from log
    rewrite_map = {}
    removed = set()

    for row in log_rows:
        if row["Action"] in ("RENAME", "MERGE", "CROSS_MERGE"):
            target = row.get("_target", "")
            if target:
                rewrite_map[row["Term"]] = target
        elif row["Action"] == "REMOVE":
            removed.add(row["Term"])

    # Apply rewrites to Term and Filler columns
    if "Term" in df.columns:
        df["Term"] = df["Term"].replace(rewrite_map)
    if "Filler" in df.columns:
        df["Filler"] = df["Filler"].replace(rewrite_map)

    # Remove rows for removed terms — both when the removed term is the
    # subject (Term) and when it is the object (Filler). Dropping only on
    # Term would leave dangling restrictions whose filler no longer exists
    # as a class in the taxonomy, producing phantom orphans under owl:Thing.
    if removed:
        if "Term" in df.columns:
            df = df[~df["Term"].isin(removed)]
        if "Filler" in df.columns:
            df = df[~df["Filler"].isin(removed)]

    return df


# ── Main entry point ───────────────────────────────────────────────────

def run_ontology_critic(
    taxonomy_csv: str,
    output_path: str | None = None,
    relations_csv: str | None = None,
    relations_output: str | None = None,
) -> str:
    """
    Run the ontology critic on a taxonomy CSV.

    Args:
        taxonomy_csv: Path to 6_taxonomy.csv
        output_path: Output path for cleaned taxonomy (default: 6c_taxonomy_cleaned.csv)
        relations_csv: Optional path to 6b_relations.csv to also clean
        relations_output: Output path for cleaned relations

    Returns:
        Path to the cleaned taxonomy CSV
    """
    from dotenv import load_dotenv
    load_dotenv()
    get_client()

    base_dir = os.path.dirname(taxonomy_csv)
    if output_path is None:
        output_path = os.path.join(base_dir, "6c_taxonomy_cleaned.csv")
    log_path = os.path.join(base_dir, "6c_critic_log.csv")

    MODEL_NAME = os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    MODEL_TEMPERATURE = float(os.environ.get("LLM_GENERATION_TEMPERATURE", 0))

    df = pd.read_csv(taxonomy_csv, encoding="utf-8-sig")
    n_original = len(df)
    log.info(f"Ontology critic: reviewing {n_original} taxonomy entries")

    all_actions = []
    log_rows = []
    categories = df["Category"].unique()

    # ── Pass 1: Intra-category critique ─────────────────────────────────
    log.info("Pass 1: Intra-category review...")
    for cat in tqdm(sorted(categories), desc="Reviewing branches", unit="cat"):
        group = df[df["Category"] == cat]
        if len(group) < 2:
            continue  # Skip singleton categories

        branch = _build_branch_json(group)
        actions = _critique_branch(cat, branch, MODEL_NAME, MODEL_TEMPERATURE)

        if actions:
            tqdm.write(f"  {cat}: {len(actions)} actions proposed")
            for a in actions:
                a["_category"] = cat
            all_actions.extend(actions)
        time.sleep(2)  # Rate limiting

    # Apply intra-category actions
    n_intra = len(all_actions)
    if all_actions:
        df = _apply_actions(df, all_actions, log_rows)
        log.info(f"Pass 1: {len(log_rows)} changes applied from {n_intra} proposed actions")

    # ── Pass 2: Cross-category review ─────────────────────────────────
    log.info("Pass 2: Cross-category review (duplicates + miscategorisation)...")
    non_intermediate = df[df.get("Is_Intermediate", pd.Series(dtype=bool)) != True]
    all_terms = []
    for _, row in non_intermediate.iterrows():
        nld_raw = row.get("NLD", "")
        all_terms.append({
            "term": row["Term"],
            "category": row["Category"],
            "nld": nld_raw if isinstance(nld_raw, str) else "",
        })

    cross_actions = _cross_category_review(all_terms, MODEL_NAME, MODEL_TEMPERATURE)
    if cross_actions:
        pre_cross = len(log_rows)
        df = _apply_actions(df, cross_actions, log_rows)
        log.info(f"Pass 2: {len(log_rows) - pre_cross} cross-category changes applied")
    else:
        log.info("Pass 2: no cross-category issues found")

    # ── Pass 3: Essentiality review ─────────────────────────────────────
    log.info("Pass 3: Essentiality review (final quality gate)...")
    # Build term list with parent/children info for the LLM
    child_counts = df["Parent_Term"].value_counts()
    ess_terms = []
    for _, row in df.iterrows():
        nld_raw = row.get("NLD", "")
        term_name = row["Term"]
        ess_terms.append({
            "term": term_name,
            "category": row["Category"],
            "parent": row["Parent_Term"],
            "has_children": term_name in child_counts.index,
            "is_individual": row.get("Relationship_Type", "") == "rdf:type",
            "nld": nld_raw if isinstance(nld_raw, str) else "",
        })

    ess_actions = _essentiality_review(ess_terms, MODEL_NAME, MODEL_TEMPERATURE)
    if ess_actions:
        pre_ess = len(log_rows)
        df = _apply_actions(df, ess_actions, log_rows)
        log.info(f"Pass 3: {len(log_rows) - pre_ess} essentiality removals applied")
    else:
        log.info("Pass 3: all terms passed essentiality check")

    # ── Remove orphaned intermediate nodes ──────────────────────────────
    # After merges/removals, some intermediate nodes may have lost all children
    changed = True
    while changed:
        changed = False
        intermediates = df[df["Is_Intermediate"] == True]
        for _, irow in intermediates.iterrows():
            i_term = irow["Term"]
            children = df[df["Parent_Term"] == i_term]
            if children.empty:
                parent = irow["Parent_Term"]
                df = df[df["Term"] != i_term]
                log_rows.append({
                    "Action": "REMOVE",
                    "Term": i_term,
                    "Detail": f"Orphaned intermediate removed (was under '{parent}')",
                    "Reason": "No children remaining after other operations",
                })
                changed = True

    # ── Repair broken parent references ─────────────────────────────────
    # After all passes, some terms may reference parents that were removed.
    # Re-parent them to their Category (upper-ontology anchor).
    from src.modules.taxonomy_builder import UPPER_IRIS
    _upper_lower = {k.lower(): k for k in UPPER_IRIS}
    valid_terms = set(df["Term"].values)

    for idx, row in df.iterrows():
        parent = row["Parent_Term"]
        if not parent or (isinstance(parent, float) and pd.isna(parent)) or not str(parent).strip():
            continue  # No parent — will be a root class
        parent_str = str(parent).strip()
        # Parent is valid if it exists in the taxonomy OR is an upper-ontology term
        if parent_str in valid_terms or parent_str.lower() in _upper_lower:
            continue
        # Broken reference — reparent to Category (if valid), else root
        category_val = str(row.get("Category", "")).strip()
        old_parent = parent_str
        if category_val and (category_val in valid_terms or category_val.lower() in _upper_lower):
            df.at[idx, "Parent_Term"] = category_val
        else:
            df.at[idx, "Parent_Term"] = ""
        log_rows.append({
            "Action": "REPARENT",
            "Term": row["Term"],
            "Detail": f"Broken parent '{old_parent}' → reparented to category '{category_val}'",
            "Reason": "Parent no longer exists in taxonomy after critic passes",
        })

    # ── Save outputs ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Audit log
    if log_rows:
        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
        log.detail(f"Critic log: {log_path}")

    # Clean relations if provided
    if relations_csv and os.path.exists(relations_csv):
        if relations_output is None:
            relations_output = os.path.join(base_dir, "6c_relations_cleaned.csv")
        rel_df = pd.read_csv(relations_csv, encoding="utf-8-sig")
        n_rel_before = len(rel_df)
        if log_rows:
            rel_df = _apply_actions_to_relations(rel_df, log_rows)
        rel_df.to_csv(relations_output, index=False, encoding="utf-8-sig")
        log.detail(f"Relations cleaned: {n_rel_before} → {len(rel_df)} rows → {relations_output}")

    # Summary
    n_merges = sum(1 for r in log_rows if r["Action"] in ("MERGE", "CROSS_MERGE"))
    n_removes = sum(1 for r in log_rows if r["Action"] == "REMOVE")
    n_moves = sum(1 for r in log_rows if r["Action"] in ("MOVE", "CROSS_MOVE"))
    n_renames = sum(1 for r in log_rows if r["Action"] == "RENAME")
    n_reparents = sum(1 for r in log_rows if r["Action"] == "REPARENT")

    log.success(
        f"Ontology critic: {n_original} → {len(df)} entries "
        f"(merged={n_merges}, removed={n_removes}, moved={n_moves}, "
        f"renamed={n_renames}, reparented={n_reparents})"
    )
    log.detail(f"Cleaned taxonomy: {output_path}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ontology critic on a taxonomy CSV")
    parser.add_argument("taxonomy_csv", help="Path to 6_taxonomy.csv")
    parser.add_argument("--output", default=None, help="Output path for cleaned taxonomy")
    parser.add_argument("--relations", default=None, help="Path to 6b_relations.csv")
    args = parser.parse_args()
    run_ontology_critic(args.taxonomy_csv, output_path=args.output, relations_csv=args.relations)
