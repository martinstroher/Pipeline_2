"""
Taxonomy Builder — Step 6 of the PreSaltOntoLearn pipeline.

Takes the winning condition's categorized output (flat: Term -> Category)
and builds IS-A hierarchies WITHIN each category using LLM group reasoning.

Key design decisions:
  - Processes all terms in a category GROUP (not one-by-one) for tree consistency
  - Explicitly distinguishes classes vs named individuals
  - Uses published BFO/GeoCore/GeoReservoir IRIs for upper-level anchoring

Output: construct_taxonomy.csv with columns (Term, Parent_Term, Relationship_Type, Category)
"""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from src.utils.csv_io import read_csv, write_csv
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils.gemini_client import get_client, generate
from src.utils import log
from src.utils.prompt_loader import load_prompt
from src.utils.ontology_config import get_config

# Published OWL IRIs for upper-level ontology anchoring.
# Sourced from `ontology_config.yaml` — edit that file to add/remove classes.
# Backed by BFO (http://purl.obolibrary.org/obo/bfo.owl),
# GeoCore (https://www.inf.ufrgs.br/bdi/ontologies/geocore.owl),
# GeoReservoir (https://www.inf.ufrgs.br/bdi/ontologies/geores.owl).
UPPER_IRIS = get_config().upper_iris()





def build_taxonomy_for_group(
    category: str, terms_with_nlds: list[dict],
    model_name: str, model_temperature: float,
    hints: list[dict] | None = None,
) -> list[dict]:
    """
    Build IS-A hierarchy for a group of terms within the same category.

    Args:
        category: The ontology category (e.g., "Sedimentary Rock")
        terms_with_nlds: List of {"term": ..., "nld": ...}
        model_name: Gemini model name
        model_temperature: Temperature for generation
        hints: Optional list of {"general": ..., "specific": ...} specialization hints

    Returns:
        List of {"Term", "Parent_Term", "Relationship_Type", "Category"}
    """
    system_instruction, prompt_template = load_prompt("taxonomy_building.txt")

    upper_vocab = "\n   ".join(f"- {k}" for k in sorted(UPPER_IRIS.keys()))

    # Build optional hints section
    hints_section = ""
    if hints:
        relevant = [h for h in hints
                     if h["general"] in {t["term"] for t in terms_with_nlds}
                     or h["specific"] in {t["term"] for t in terms_with_nlds}]
        if relevant:
            pairs = "\n".join(f"   - \"{h['specific']}\" is a specialization of \"{h['general']}\"" for h in relevant)
            hints_section = (
                f"\n\n**PRE-IDENTIFIED SPECIALIZATIONS (use as parent-child constraints):**\n"
                f"{pairs}\n"
                f"These pairs were identified by synonym analysis. Place the specific term "
                f"under the general term in the hierarchy unless the NLDs clearly contradict this.\n"
            )

    prompt = prompt_template.format(
        category=category,
        upper_vocab=upper_vocab,
        terms_json=json.dumps(terms_with_nlds, indent=2),
    )
    # Append hints after the formatted prompt (before the output format section won't work
    # since format() already resolved placeholders, so append at the end of terms_json area)
    if hints_section:
        prompt = prompt + hints_section

    try:
        response_text = generate(
            prompt,
            model=model_name,
            system_instruction=system_instruction,
            temperature=model_temperature,
            response_mime_type="application/json",
        )
        result = json.loads(response_text)

        rows = []
        for item in result:
            is_intermediate = bool(item.get("is_intermediate", False))
            # Intermediates carry their NLD inline (the LLM writes one in the same call);
            # input terms reuse their existing NLD downstream via nld_lookup.
            row_nld = item.get("intermediate_nld", "") if is_intermediate else ""
            rows.append({
                "Term": item.get("term", ""),
                "Parent_Term": item.get("parent_term", category),
                "Relationship_Type": item.get("relationship_type", "rdfs:subClassOf"),
                "Category": category,
                "Is_Intermediate": is_intermediate,
                "NLD": row_nld,
                "FALLBACK": False,
            })

        # --- Casing normalization + duplicate-intermediate drop ---
        # The LLM may invent an intermediate (Title Case) that collides
        # case-insensitively with an existing input term (lowercase). In that
        # case the input term wins (it has the canonical NLD); we rewrite any
        # references to the duplicate and drop the synthetic row.
        input_terms_lower = {t["term"].lower(): t["term"] for t in terms_with_nlds}
        for row in rows:
            parent_lower = str(row["Parent_Term"]).lower()
            if parent_lower in input_terms_lower:
                row["Parent_Term"] = input_terms_lower[parent_lower]
        duplicate_intermediates = set()
        for row in rows:
            if row["Is_Intermediate"]:
                term_lower = str(row["Term"]).lower()
                if term_lower in input_terms_lower and row["Term"] != input_terms_lower[term_lower]:
                    duplicate_intermediates.add(row["Term"])
                    log.detail(
                        f"Dropped duplicate intermediate '{row['Term']}' in '{category}' "
                        f"— collides with input term '{input_terms_lower[term_lower]}'"
                    )
        if duplicate_intermediates:
            rows = [r for r in rows if r["Term"] not in duplicate_intermediates]

        # --- Cycle detection: break any cycles by re-parenting to category root ---
        child_to_parent = {r["Term"]: r["Parent_Term"] for r in rows}
        for row in rows:
            visited = set()
            node = row["Term"]
            while node in child_to_parent:
                if node in visited:
                    # Cycle detected — break it by re-parenting this row
                    log.warn(f"Cycle detected involving '{row['Term']}' in category '{category}' — re-parenting to root")
                    row["Parent_Term"] = category
                    child_to_parent[row["Term"]] = category
                    break
                visited.add(node)
                node = child_to_parent.get(node)

        return rows
    except Exception as e:
        log.warn(f"Flat fallback for '{category}' ({len(terms_with_nlds)} terms): {e}")
        # Fallback: flat hierarchy (all terms directly under category)
        return [
            {
                "Term": t.get("term", ""),
                "Parent_Term": category,
                "Relationship_Type": "rdfs:subClassOf",
                "Category": category,
                "Is_Intermediate": False,
                "NLD": "",
                "FALLBACK": True,
            }
            for t in terms_with_nlds
        ]


def run_taxonomy_builder(categorized_csv: str, output_path: str | None = None, hints_csv: str | None = None):
    """
    Build taxonomy from a categorized CSV.

    Args:
        categorized_csv: Path to categorized output (e.g., cat_A.csv or classify_categories.csv)
        output_path: Output path (default: derived from input)
        hints_csv: Optional path to specialization hints CSV (General_Term, Specific_Term)
    """
    load_dotenv()
    get_client()

    if output_path is None:
        base = os.path.splitext(categorized_csv)[0]
        output_path = base.replace("cat_", "construct_taxonomy_").replace(
            "classify_categories", "construct_taxonomy"
        ) + ".csv"

    df = read_csv(categorized_csv)
    log.info(f"Taxonomy builder: {len(df)} terms from {categorized_csv}")

    # Filter out errors and NOT_CLASSIFIED
    df_valid = df[
        ~df["Category"].str.startswith("ERROR", na=False)
        & (df["Category"] != "NOT_CLASSIFIED")
    ].copy()
    log.detail(f"{len(df_valid)} valid terms (excluding errors and NOT_CLASSIFIED)")

    MODEL_NAME = os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    MODEL_TEMPERATURE = float(os.environ.get("LLM_GENERATION_TEMPERATURE", 0))

    # Load optional specialization hints
    hints = None
    if hints_csv and os.path.exists(hints_csv):
        hints_df = read_csv(hints_csv)
        hints = [
            {"general": row["General_Term"], "specific": row["Specific_Term"]}
            for _, row in hints_df.iterrows()
        ]
        log.info(f"Loaded {len(hints)} specialization hints from '{hints_csv}'")

    all_taxonomy_rows = []
    categories = sorted(df_valid["Category"].unique())
    MAX_WORKERS = int(os.environ.get("MAX_CONCURRENT_TAXONOMY", 5))
    lock = threading.Lock()

    def _process_category(cat: str) -> list[dict]:
        group = df_valid[df_valid["Category"] == cat]
        terms_with_nlds = [
            {"term": row["Term"], "nld": row.get("NLD", "")}
            for _, row in group.iterrows()
        ]
        nld_lookup = {t["term"]: t.get("nld", "") for t in terms_with_nlds}
        cat_rows: list[dict] = []
        # Process in chunks of 150 for large groups (sequential within category)
        for chunk_start in range(0, len(terms_with_nlds), 150):
            chunk = terms_with_nlds[chunk_start : chunk_start + 150]
            rows = build_taxonomy_for_group(
                cat, chunk, MODEL_NAME, MODEL_TEMPERATURE, hints=hints,
            )
            for row in rows:
                # Input terms get their NLD from the categorized CSV;
                # intermediates keep the one-sentence NLD the LLM wrote inline.
                if not row.get("Is_Intermediate"):
                    row["NLD"] = nld_lookup.get(row["Term"], "")
            cat_rows.extend(rows)
        return cat_rows

    max_workers = min(MAX_WORKERS, len(categories)) if categories else 1
    log.info(
        f"Building taxonomy for {len(categories)} categories with {max_workers} workers"
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_category, cat): cat for cat in categories}
        with tqdm(total=len(categories), desc="Building taxonomy", unit="category") as pbar:
            for future in as_completed(futures):
                cat = futures[future]
                try:
                    cat_rows = future.result()
                    with lock:
                        all_taxonomy_rows.extend(cat_rows)
                    pbar.set_postfix_str(cat[:30])
                except Exception as e:
                    tqdm.write(f"  [error] Category '{cat}': {e}")
                pbar.update(1)

    # Add upper-level IRI mappings
    for row in all_taxonomy_rows:
        cat_key = row["Category"]
        if cat_key in UPPER_IRIS:
            row["Category_IRI"] = UPPER_IRIS[cat_key]
        else:
            row["Category_IRI"] = ""
        parent_key = row["Parent_Term"]
        if parent_key in UPPER_IRIS:
            row["Parent_IRI"] = UPPER_IRIS[parent_key]
        else:
            row["Parent_IRI"] = ""

    taxonomy_df = pd.DataFrame(all_taxonomy_rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_csv(taxonomy_df, output_path)

    # Stats
    n_classes = len(taxonomy_df[taxonomy_df["Relationship_Type"] == "rdfs:subClassOf"])
    n_individuals = len(taxonomy_df[taxonomy_df["Relationship_Type"] == "rdf:type"])
    n_intermediate = len(taxonomy_df[taxonomy_df["Is_Intermediate"] == True])
    log.success(f"Taxonomy built: {len(taxonomy_df)} entries (classes={n_classes}, individuals={n_individuals}, intermediate={n_intermediate})")
    n_fallback = int(taxonomy_df["FALLBACK"].sum()) if "FALLBACK" in taxonomy_df.columns else 0
    if n_fallback:
        log.warn(f"{n_fallback} entries used flat fallback (LLM error) — FALLBACK=True in CSV marks affected rows")
    log.detail(f"Saved to: {output_path}")

    return taxonomy_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to categorized CSV")
    parser.add_argument("--output", default=None, help="Output path")
    args = parser.parse_args()
    run_taxonomy_builder(args.input_csv, args.output)
