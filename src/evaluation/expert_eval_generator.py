"""
Expert Evaluation Spreadsheet Generator (Layer 2).

Generates a blinded, multi-sheet Excel workbook for domain expert evaluation:
  Sheet 1 — Instructions: guidelines, Likert scales, calibration examples
  Sheet 2 — Term_Relevance: 200 terms, condition-independent relevance rating
  Sheet 3 — NLD_Quality: 200 terms, blinded A-vs-B paired definition comparison
  Sheet 4 — Category_Correct: stratified by ontology tier (GeoReservoir binary
            + GeoCore/BFO simplified 3-way), deduplicated across 4 conditions
  Sheet 5 — Taxonomy_Correct: parent-child IS-A pairs for hierarchy validation

Outputs:
  - Excel workbook per expert (send to expert)
  - Blinding key CSV (keep for analysis, never share with experts)
"""

import os
import random
import hashlib

import pandas as pd

OUTPUT_DIR = os.environ.get("ABLATION_OUTPUT_DIR", "output/ablation")

# Ontology tier classification for stratified evaluation
GEORESERVOIR_CATEGORIES = {
    "Sedimentary Geological Object", "Depositional Unit", "Channel Unit",
    "Lobe Unit", "Levee Unit", "Mound Unit", "Overbank Unit",
    "Sedimentary Rock", "Sediment", "Depositional System", "Channel Surface",
    "Dimension", "Length", "Thickness", "Geometry", "Geometry Value",
    "Channel Geometry", "Lobe Geometry", "Mound Geometry", "Wedge Geometry",
    "Sinuosity", "Facies", "Sedimentary Facies", "Facies Association",
    "Sedimentary Structure", "Sedimentary Environment", "Lithology",
    "Formation", "Stratigraphic Unit", "Fossil",
}

GEOCORE_CATEGORIES = {
    "Geological Object", "Earth Material", "Rock", "Earth Fluid",
    "Geological Boundary", "Geological Structure", "Geological Contact",
    "Geological Process", "Geological Time Interval", "Geological Age",
}

# Everything else falls to BFO tier


# ---------------------------------------------------------------------------
# Term Selection
# ---------------------------------------------------------------------------

def select_terms(n_terms: int = 200, seed: int = 42) -> list[str]:
    """Select top-N terms by frequency from the filtered terms file.

    Falls back to the first N terms if frequency column is flat.
    Excludes terms where any condition produced an ERROR result.
    """
    terms_file = os.environ.get("FILTERED_TERMS_OUTPUT", "output/3_filtered_top_terms.csv")
    df = pd.read_csv(terms_file, encoding="utf-8")
    # Sort by frequency descending (should already be sorted, but ensure)
    df = df.sort_values("Frequency", ascending=False).reset_index(drop=True)

    # Filter out terms with errors in any condition
    error_terms = set()
    for cond in ["A", "B", "C", "D"]:
        cat_path = os.path.join(OUTPUT_DIR, f"cat_{cond}.csv")
        if os.path.exists(cat_path):
            cat_df = pd.read_csv(cat_path, encoding="utf-8-sig")
            err_mask = cat_df["Category"].str.startswith("ERROR", na=False)
            error_terms.update(cat_df.loc[err_mask, "Term"].tolist())

    candidates = df[~df["Readable_Term"].isin(error_terms)]
    selected = candidates.head(n_terms)["Readable_Term"].tolist()

    if len(selected) < n_terms:
        print(f"  Warning: only {len(selected)} error-free terms available (requested {n_terms})")

    return selected


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_all_conditions() -> dict:
    """Load NLD and categorization CSVs for all 4 conditions.

    Returns:
        {
            "nld": {"A": DataFrame, "B": DataFrame, ...},
            "cat": {"A": DataFrame, "B": DataFrame, ...},
        }
    """
    data = {"nld": {}, "cat": {}}
    for cond in ["A", "B", "C", "D"]:
        nld_path = os.path.join(OUTPUT_DIR, f"nld_{cond}.csv")
        cat_path = os.path.join(OUTPUT_DIR, f"cat_{cond}.csv")

        if os.path.exists(nld_path):
            data["nld"][cond] = pd.read_csv(nld_path, encoding="utf-8-sig")
        else:
            print(f"  Warning: {nld_path} not found, skipping condition {cond}")

        if os.path.exists(cat_path):
            data["cat"][cond] = pd.read_csv(cat_path, encoding="utf-8-sig")
        else:
            print(f"  Warning: {cat_path} not found, skipping condition {cond}")

    return data


# ---------------------------------------------------------------------------
# Sheet Builders
# ---------------------------------------------------------------------------

def build_term_relevance_sheet(terms: list[str]) -> pd.DataFrame:
    """Sheet 2: Term relevance evaluation (condition-independent).

    100 rows, sorted alphabetically.
    """
    rows = []
    for term in sorted(terms):
        rows.append({
            "Term": term,
            "Relevance (1-5)": "",
            "Notes": "",
        })
    return pd.DataFrame(rows)


def build_nld_quality_sheet(
    terms: list[str],
    nld_data: dict[str, pd.DataFrame],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sheet 3: Blinded A-vs-B NLD quality comparison.

    Returns (expert_sheet, key_sheet).
    Only conditions A and B are included (C and D break blinding).
    """
    if "A" not in nld_data or "B" not in nld_data:
        raise ValueError("NLD data for conditions A and B is required for NLD quality sheet")

    nld_a = nld_data["A"].set_index("Term")
    nld_b = nld_data["B"].set_index("Term")

    rng = random.Random(seed)
    expert_rows = []
    key_rows = []

    # Shuffle term order for blinding
    shuffled_terms = list(terms)
    rng.shuffle(shuffled_terms)

    for idx, term in enumerate(shuffled_terms, 1):
        row_id = f"NLD-{idx:03d}"

        def_a = nld_a.loc[term, "NLD"] if term in nld_a.index else "N/A"
        def_b = nld_b.loc[term, "NLD"] if term in nld_b.index else "N/A"
        ctx_used = nld_a.loc[term, "Context_Used"] if term in nld_a.index else ""

        # Coin flip: which definition goes first
        if rng.random() < 0.5:
            d1, d2 = def_a, def_b
            order = "A_first"
        else:
            d1, d2 = def_b, def_a
            order = "B_first"

        expert_rows.append({
            "Row_ID": row_id,
            "Term": term,
            "Definition_1": d1,
            "Definition_2": d2,
            "Quality_1 (1-5)": "",
            "Quality_2 (1-5)": "",
            "Preference (1/2/Tie)": "",
            "Notes": "",
        })

        key_rows.append({
            "Sheet": "NLD_Quality",
            "Row_ID": row_id,
            "Term": term,
            "Order": order,
            "Context_Used_A": ctx_used,
            "NLD_A": def_a,
            "NLD_B": def_b,
        })

    return pd.DataFrame(expert_rows), pd.DataFrame(key_rows)


def _classify_tier(category: str) -> str:
    """Classify a category into its ontology tier."""
    if category in GEORESERVOIR_CATEGORIES:
        return "GeoReservoir"
    elif category in GEOCORE_CATEGORIES:
        return "GeoCore"
    else:
        return "BFO"


def _load_category_descriptions() -> dict[str, str]:
    """Load one-line descriptions for GeoReservoir categories."""
    defs = {}
    path = os.environ.get("GEORESERVOIR_DEFS_PATH", "resources/georeservoir-definitions.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    name, desc = line.split(":", 1)
                    defs[name.strip()] = desc.strip()
    return defs


def build_category_sheet(
    terms: list[str],
    cat_data: dict[str, pd.DataFrame],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sheet 4: Stratified category correctness evaluation (all 4 conditions).

    Evaluation is stratified by ontology tier:
      - GeoReservoir terms: binary validation ("Does this term fit this category?")
        with full category description shown. If No/Partial, expert picks from
        GeoReservoir alternatives only.
      - GeoCore/BFO terms: simplified 3-way question ("Is this best understood as
        a physical thing, a process, or a property?") plus binary validation of
        the assigned category with its description.

    Returns (expert_sheet, key_sheet).
    """
    rng = random.Random(seed + 100)
    geores_defs = _load_category_descriptions()
    expert_rows = []
    key_rows = []
    row_counter = 0

    shuffled_terms = list(terms)
    rng.shuffle(shuffled_terms)

    for term in shuffled_terms:
        term_categories: dict[str, list[str]] = {}
        for cond in ["A", "B", "C", "D"]:
            if cond not in cat_data:
                continue
            df = cat_data[cond]
            match = df[df["Term"] == term]
            if match.empty:
                continue
            cat = match.iloc[0]["Category"]
            if pd.isna(cat) or str(cat).startswith("ERROR"):
                continue
            cat_str = str(cat)
            if cat_str not in term_categories:
                term_categories[cat_str] = []
            term_categories[cat_str].append(cond)

        for category, conditions in term_categories.items():
            row_counter += 1
            row_id = f"CAT-{row_counter:04d}"
            tier = _classify_tier(category)

            if tier == "GeoReservoir":
                cat_desc = geores_defs.get(category, "")
                expert_rows.append({
                    "Row_ID": row_id,
                    "Term": term,
                    "Tier": "GeoReservoir",
                    "Assigned_Category": category,
                    "Category_Description": cat_desc,
                    "Correct (Yes/No/Partial)": "",
                    "Suggested_Category": "",
                    "Notes": "",
                })
            else:
                # GeoCore/BFO: show assigned category + simplified 3-way
                expert_rows.append({
                    "Row_ID": row_id,
                    "Term": term,
                    "Tier": tier,
                    "Assigned_Category": category,
                    "Category_Description": "",
                    "Correct (Yes/No/Partial)": "",
                    "Suggested_Category": "",
                    "Notes": "",
                })

            key_rows.append({
                "Sheet": "Category_Correct",
                "Row_ID": row_id,
                "Term": term,
                "Assigned_Category": category,
                "Tier": tier,
                "Conditions": ",".join(conditions),
            })

    # Shuffle rows so terms aren't grouped together
    combined = list(zip(expert_rows, key_rows))
    rng.shuffle(combined)
    expert_rows = [c[0] for c in combined]
    key_rows = [c[1] for c in combined]

    return pd.DataFrame(expert_rows), pd.DataFrame(key_rows)


# ---------------------------------------------------------------------------
# Sheet 5: Taxonomy Correctness
# ---------------------------------------------------------------------------

def build_taxonomy_sheet(
    terms: list[str],
    seed: int = 42,
    n_pairs: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sheet 5: Parent-child IS-A pair validation from the taxonomy.

    Loads the taxonomy CSV, filters to edges involving the selected terms,
    stratifies across categories and depths, and asks experts:
    'Is [Term] a type of [Parent]?'

    Args:
        terms: List of selected terms for evaluation.
        seed: Random seed for reproducibility.
        n_pairs: Target number of parent-child pairs to include.

    Returns (expert_sheet, key_sheet).
    """
    rng = random.Random(seed + 200)

    # Load taxonomy
    taxonomy_path = os.environ.get("TAXONOMY_OUTPUT", "output/6_taxonomy.csv")
    if not os.path.exists(taxonomy_path):
        print(f"  Warning: {taxonomy_path} not found, skipping taxonomy sheet")
        return pd.DataFrame(), pd.DataFrame()

    tax_df = pd.read_csv(taxonomy_path, encoding="utf-8-sig")

    # Filter to IS-A relationships (rdfs:subClassOf and rdf:type)
    isa_mask = tax_df["Relationship_Type"].isin(["rdfs:subClassOf", "rdf:type"])
    isa_df = tax_df[isa_mask].copy()

    if isa_df.empty:
        print("  Warning: No IS-A relationships found in taxonomy")
        return pd.DataFrame(), pd.DataFrame()

    # Prefer edges involving the selected terms, but also include
    # intermediate-node edges for hierarchy depth coverage
    term_set = set(terms)
    involves_selected = isa_df["Term"].isin(term_set) | isa_df["Parent_Term"].isin(term_set)

    # Split: edges involving selected terms vs. intermediate-only
    primary = isa_df[involves_selected]
    intermediate = isa_df[~involves_selected]

    # Stratify primary by category to ensure broad coverage
    selected_pairs = []
    if not primary.empty:
        categories = primary["Category"].unique()
        per_cat = max(1, n_pairs * 3 // (4 * len(categories)))  # ~75% from primary
        for cat in categories:
            cat_edges = primary[primary["Category"] == cat]
            sample_n = min(per_cat, len(cat_edges))
            selected_pairs.extend(
                cat_edges.sample(n=sample_n, random_state=seed).to_dict("records")
            )

    # Fill remaining slots from intermediate edges
    remaining = n_pairs - len(selected_pairs)
    if remaining > 0 and not intermediate.empty:
        fill_n = min(remaining, len(intermediate))
        selected_pairs.extend(
            intermediate.sample(n=fill_n, random_state=seed).to_dict("records")
        )

    # Trim to target
    if len(selected_pairs) > n_pairs:
        rng.shuffle(selected_pairs)
        selected_pairs = selected_pairs[:n_pairs]

    # Build sheets
    expert_rows = []
    key_rows = []
    rng.shuffle(selected_pairs)

    for idx, pair in enumerate(selected_pairs, 1):
        row_id = f"TAX-{idx:04d}"
        term = pair["Term"]
        parent = pair["Parent_Term"]
        category = pair.get("Category", "")
        rel_type = pair.get("Relationship_Type", "")
        is_intermediate = pair.get("Is_Intermediate", False)

        expert_rows.append({
            "Row_ID": row_id,
            "Term": term,
            "Parent_Term": parent,
            "Question": f"Is '{term}' a type of '{parent}'?",
            "Correct (Yes/No/Partial)": "",
            "Suggested_Parent": "",
            "Notes": "",
        })

        key_rows.append({
            "Sheet": "Taxonomy_Correct",
            "Row_ID": row_id,
            "Term": term,
            "Parent_Term": parent,
            "Category": category,
            "Relationship_Type": rel_type,
            "Is_Intermediate": is_intermediate,
        })

    return pd.DataFrame(expert_rows), pd.DataFrame(key_rows)


def build_instructions_sheet() -> pd.DataFrame:
    """Sheet 1: Evaluation instructions with Likert scale definitions."""
    instructions = [
        ["EXPERT EVALUATION — PreSaltOntoLearn Pipeline", ""],
        ["", ""],
        ["PURPOSE", "Evaluate the quality of automatically generated definitions, ontological classifications, and taxonomic hierarchy for Pre-Salt petroleum geology terms. Your evaluations are BLINDED — do not attempt to identify which system produced which output."],
        ["", ""],
        ["=== SHEET 2: TERM RELEVANCE ===", ""],
        ["Task", "Rate how relevant each term is to Brazilian Pre-Salt petroleum geology."],
        ["Scale", ""],
        ["1 = Not relevant", "Term has no connection to Pre-Salt petroleum geology"],
        ["2 = Marginal", "Loosely related; general geology term with no Pre-Salt specificity"],
        ["3 = Relevant", "Clearly geological, applicable to Pre-Salt context"],
        ["4 = Important", "Commonly used in Pre-Salt studies"],
        ["5 = Core concept", "Fundamental to Pre-Salt petroleum geology understanding"],
        ["", ""],
        ["=== SHEET 3: NLD QUALITY ===", ""],
        ["Task", "For each term, two Natural Language Definitions (NLDs) are shown. Rate the quality of EACH definition independently, then indicate your overall preference."],
        ["Scale", ""],
        ["1 = Incorrect", "Definition contains fundamental geological errors or is incoherent"],
        ["2 = Poor", "Major inaccuracies, severe imprecision, or missing essential properties"],
        ["3 = Acceptable", "Broadly correct but with notable gaps or minor inaccuracies"],
        ["4 = Good", "Accurate and precise, only minor issues"],
        ["5 = Excellent", "Publication-quality definition; correctly captures the concept for Pre-Salt"],
        ["Preference", "Choose 1, 2, or Tie based on which definition is better overall"],
        ["", ""],
        ["=== SHEET 4: CATEGORY CORRECTNESS ===", ""],
        ["Task", "For each (Term, Category) pair, judge whether the assigned ontological category is correct."],
        ["", ""],
        ["Tier column", "Shows which ontology level the category comes from: GeoReservoir (domain-specific), GeoCore (mid-level geological), or BFO (abstract foundational). Focus your effort on GeoReservoir categories — those are directly in your area of expertise."],
        ["Category_Description", "For GeoReservoir categories, the formal description is provided for reference."],
        ["", ""],
        ["Correct", "The assigned category correctly classifies this term"],
        ["Partial", "The category is in the right direction but too broad, too narrow, or in the wrong layer"],
        ["No", "The assigned category is incorrect for this term"],
        ["Suggested_Category", "If you chose No or Partial, suggest the correct category"],
        ["", ""],
        ["NOTE on GeoCore/BFO tiers", "For terms classified into abstract categories (GeoCore or BFO), a simplified judgment is sufficient: does the category broadly make sense? You are NOT expected to know BFO formal definitions — use your geological intuition."],
        ["", ""],
        ["=== SHEET 5: TAXONOMY CORRECTNESS ===", ""],
        ["Task", "Evaluate parent-child relationships in the generated taxonomy. For each pair, judge: 'Is [Term] a type of [Parent]?'"],
        ["", ""],
        ["Yes", "The IS-A relationship is geologically correct (e.g., Grainstone IS-A Carbonate Rock)"],
        ["No", "The IS-A relationship is geologically incorrect"],
        ["Partial", "Broadly reasonable but imprecise (e.g., too broad or too narrow)"],
        ["Suggested_Parent", "If No or Partial, suggest a better parent term"],
        ["", ""],
        ["=== CALIBRATION EXAMPLES ===", ""],
        ["Example 1 (Relevance)", "Term: 'Grainstone' — Core concept (Relevance=5). A correct category would be GeoReservoir > Sedimentary Rock."],
        ["Example 2 (Relevance)", "Term: 'Coquina' — Core concept (Relevance=5). A correct category would be GeoReservoir > Sedimentary Rock."],
        ["Example 3 (Taxonomy)", "'Grainstone' IS-A 'Carbonate Rock' → Yes. 'Grainstone' IS-A 'Geological Process' → No."],
        ["", ""],
        ["NOTES", "Use the Notes column to explain non-obvious judgments. Thank you for your expertise!"],
    ]
    return pd.DataFrame(instructions, columns=["Section", "Details"])


# ---------------------------------------------------------------------------
# Workbook Generation
# ---------------------------------------------------------------------------

def generate_expert_evaluation(
    n_terms: int = 200,
    seed: int = 42,
    output_dir: str | None = None,
) -> tuple[str, str]:
    """Generate the expert evaluation workbook and blinding key.

    Args:
        n_terms: Number of terms to include.
        seed: Random seed for reproducibility.
        output_dir: Output directory (default: output/ablation/).

    Returns:
        (workbook_path, key_path)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    # 1. Select terms
    print(f"\nExpert Evaluation Generator (seed={seed})")
    terms = select_terms(n_terms, seed)
    print(f"  Selected {len(terms)} terms (top by frequency, error-free)")

    # 2. Load ablation data
    data = load_all_conditions()

    # 3. Build sheets
    instructions_df = build_instructions_sheet()
    relevance_df = build_term_relevance_sheet(terms)

    nld_expert_df, nld_key_df = build_nld_quality_sheet(terms, data["nld"], seed)
    cat_expert_df, cat_key_df = build_category_sheet(terms, data["cat"], seed)
    tax_expert_df, tax_key_df = build_taxonomy_sheet(terms, seed)

    # 4. Write expert workbook
    workbook_path = os.path.join(output_dir, "expert_evaluation.xlsx")

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        instructions_df.to_excel(writer, sheet_name="Instructions", index=False)
        relevance_df.to_excel(writer, sheet_name="Term_Relevance", index=False)
        nld_expert_df.to_excel(writer, sheet_name="NLD_Quality", index=False)
        cat_expert_df.to_excel(writer, sheet_name="Category_Correct", index=False)
        if not tax_expert_df.empty:
            tax_expert_df.to_excel(writer, sheet_name="Taxonomy_Correct", index=False)

    # 5. Write blinding key (separate file)
    key_path = os.path.join(output_dir, f"blinding_key_{seed}.csv")
    key_parts = [nld_key_df, cat_key_df]
    if not tax_key_df.empty:
        key_parts.append(tax_key_df)
    key_combined = pd.concat(key_parts, ignore_index=True)
    key_combined.to_csv(key_path, index=False, encoding="utf-8-sig")

    # 6. Summary
    n_geores = len(cat_expert_df[cat_expert_df["Tier"] == "GeoReservoir"]) if "Tier" in cat_expert_df.columns else 0
    n_upper = len(cat_expert_df) - n_geores if not cat_expert_df.empty else 0
    print(f"\n  Workbook: {workbook_path}")
    print(f"    Sheet 'Instructions': evaluation guidelines and Likert scales")
    print(f"    Sheet 'Term_Relevance': {len(relevance_df)} terms")
    print(f"    Sheet 'NLD_Quality': {len(nld_expert_df)} blinded A-vs-B comparisons")
    print(f"    Sheet 'Category_Correct': {len(cat_expert_df)} pairs ({n_geores} GeoReservoir, {n_upper} GeoCore/BFO)")
    if not tax_expert_df.empty:
        print(f"    Sheet 'Taxonomy_Correct': {len(tax_expert_df)} parent-child IS-A pairs")
    print(f"  Blinding key: {key_path} (DO NOT share with experts)")

    return workbook_path, key_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate expert evaluation spreadsheet")
    parser.add_argument("--n-terms", type=int, default=200, help="Number of terms to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    generate_expert_evaluation(n_terms=args.n_terms, seed=args.seed)
