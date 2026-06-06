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

from src.utils.csv_io import read_csv
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from src.utils.ontology_config import get_config

OUTPUT_DIR = os.environ.get("ABLATION_OUTPUT_DIR", "output/ablation")

# Ontology tier classification for stratified evaluation. Sourced from
# `ontology_config.yaml` — each ontology key contributes its valid categories.
GEORESERVOIR_CATEGORIES = get_config().categories_for("georeservoir")
GEOCORE_CATEGORIES = get_config().categories_for("geocore")
# Everything else falls to BFO tier


# ---------------------------------------------------------------------------
# Term Selection
# ---------------------------------------------------------------------------

def select_terms(n_terms: int = 200, seed: int = 42) -> list[str]:
    """Select top-N terms by frequency from the filtered terms file.

    Falls back to the first N terms if frequency column is flat.
    Excludes terms where any condition produced an ERROR result.
    """
    terms_file = os.environ.get("FILTERED_TERMS_OUTPUT", "output/extract_filtered.csv")
    df = pd.read_csv(terms_file, encoding="utf-8")
    # Sort by frequency descending (should already be sorted, but ensure)
    df = df.sort_values("Frequency", ascending=False).reset_index(drop=True)

    # Filter out terms with errors in any condition
    error_terms = set()
    for cond in ["A", "B", "C", "D"]:
        cat_path = os.path.join(OUTPUT_DIR, f"cat_{cond}.csv")
        if os.path.exists(cat_path):
            cat_df = read_csv(cat_path)
            err_mask = cat_df["Category"].str.startswith("ERROR", na=False)
            error_terms.update(cat_df.loc[err_mask, "Term"].tolist())

    candidates = df[~df["Readable_Term"].isin(error_terms)]
    selected = candidates.head(n_terms)["Readable_Term"].tolist()

    if len(selected) < n_terms:
        print(f"  Warning: only {len(selected)} error-free terms available (requested {n_terms})")

    return selected


def select_terms_refined(
    cq_matrix_csv: str,
    threshold: int,
    n_kept: int = 150,
    n_removed: int = 50,
    seed: int = 42,
) -> tuple[list[str], dict[str, bool]]:
    """Select terms for combined ablation + CQ-filter evaluation.

    Samples `n_kept` terms with CQ_Count >= threshold (kept) and
    `n_removed` terms with CQ_Count < threshold (removed), stratified
    by category within each group.

    Args:
        cq_matrix_csv: Path to 5b_cq_matrix.csv.
        threshold: CQ count threshold (e.g. 1).
        n_kept: Target number of kept terms.
        n_removed: Target number of removed terms.
        seed: Random seed.

    Returns:
        (terms_list, status_map) where status_map maps term → True (kept) / False (removed).
    """
    rng = random.Random(seed)

    df_cq = read_csv(cq_matrix_csv)

    # Load categorized CSV for category info
    cat_csv = os.environ.get("CATEGORIZED_LLM_TERMS", "output/classify_categories.csv")
    df_cat = read_csv(cat_csv)
    cat_map = dict(zip(df_cat["Term"], df_cat["Category"]))
    df_cq["Category"] = df_cq["Term"].map(cat_map)

    # Filter out error/unclassified
    df_cq = df_cq[~df_cq["Category"].str.startswith("ERROR", na=True)].copy()
    df_cq = df_cq[df_cq["Category"] != "NOT_CLASSIFIED"].copy()

    # Also exclude terms with errors in ablation conditions
    error_terms = set()
    for cond in ["A", "B", "C", "D"]:
        cat_path = os.path.join(OUTPUT_DIR, f"cat_{cond}.csv")
        if os.path.exists(cat_path):
            abl_df = read_csv(cat_path)
            err_mask = abl_df["Category"].str.startswith("ERROR", na=False)
            error_terms.update(abl_df.loc[err_mask, "Term"].tolist())
    df_cq = df_cq[~df_cq["Term"].isin(error_terms)].copy()

    kept_pool = df_cq[df_cq["CQ_Count"] >= threshold]
    removed_pool = df_cq[df_cq["CQ_Count"] < threshold]

    def _stratified_sample(pool: pd.DataFrame, n: int) -> list[str]:
        """Sample up to n terms, stratified by category."""
        if len(pool) <= n:
            return pool["Term"].tolist()
        cats = pool["Category"].unique()
        per_cat = max(1, n // len(cats))
        sampled = []
        for cat in cats:
            cat_rows = pool[pool["Category"] == cat]
            take = min(per_cat, len(cat_rows))
            sampled.extend(cat_rows.sample(n=take, random_state=seed)["Term"].tolist())
        # Fill remaining slots
        remaining = n - len(sampled)
        if remaining > 0:
            leftover = pool[~pool["Term"].isin(sampled)]
            if len(leftover) > 0:
                extra = leftover.sample(n=min(remaining, len(leftover)), random_state=seed)
                sampled.extend(extra["Term"].tolist())
        return sampled[:n]

    kept_terms = _stratified_sample(kept_pool, n_kept)
    removed_terms = _stratified_sample(removed_pool, n_removed)

    # If removed pool is too small, backfill from kept
    total = n_kept + n_removed
    if len(kept_terms) + len(removed_terms) < total:
        shortfall = total - len(kept_terms) - len(removed_terms)
        backfill_pool = kept_pool[~kept_pool["Term"].isin(kept_terms)]
        if len(backfill_pool) >= shortfall:
            extra = backfill_pool.sample(n=shortfall, random_state=seed)["Term"].tolist()
            kept_terms.extend(extra)

    all_terms = kept_terms + removed_terms
    rng.shuffle(all_terms)

    status_map = {}
    for t in kept_terms:
        status_map[t] = True
    for t in removed_terms:
        status_map[t] = False

    print(f"  Selected {len(kept_terms)} kept + {len(removed_terms)} removed = {len(all_terms)} terms")
    return all_terms, status_map


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
            data["nld"][cond] = read_csv(nld_path)
        else:
            print(f"  Warning: {nld_path} not found, skipping condition {cond}")

        if os.path.exists(cat_path):
            data["cat"][cond] = read_csv(cat_path)
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
    """Load descriptions for all upper-ontology classes from ontology_config."""
    cfg = get_config()
    defs: dict[str, str] = {}
    for ontology_key in ("georeservoir", "geocore", "bfo"):
        for line in cfg.llm_definitions_block(ontology_key).split("\n"):
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
            cat_desc = geores_defs.get(category, "")

            expert_rows.append({
                "Row_ID": row_id,
                "Term": term,
                "Tier": tier,
                "Assigned_Category": category,
                "Category_Description": cat_desc,
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
    taxonomy_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sheet 5: Parent-child IS-A pair validation from the taxonomy.

    Loads the taxonomy CSV, filters to edges involving the selected terms,
    stratifies across categories and depths, and asks experts:
    'Is [Term] a type of [Parent]?'

    Args:
        terms: List of selected terms for evaluation.
        seed: Random seed for reproducibility.
        n_pairs: Target number of parent-child pairs to include.
        taxonomy_path: Explicit path to taxonomy CSV (default: TAXONOMY_OUTPUT env var).

    Returns (expert_sheet, key_sheet).
    """
    rng = random.Random(seed + 200)

    # Load taxonomy
    if taxonomy_path is None:
        taxonomy_path = os.environ.get("TAXONOMY_OUTPUT", "output/construct_taxonomy.csv")
    if not os.path.exists(taxonomy_path):
        print(f"  Warning: {taxonomy_path} not found, skipping taxonomy sheet")
        return pd.DataFrame(), pd.DataFrame()

    tax_df = read_csv(taxonomy_path)

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


def build_instructions_sheet() -> list[list[str]]:
    """Sheet 1: Evaluation instructions with Likert scale definitions.

    Returns raw rows (list of [section, details]) for custom formatting.
    Content is sourced from `studies/expert_eval.yaml`
    (`instructions_sheet.rows`) so the workbook is retargetable without
    editing this module.
    """
    from src.utils.study_config import get_study_config
    rows = get_study_config().instruction_rows
    return [[section, details] for section, details in rows]


# ---------------------------------------------------------------------------
# Excel Formatting
# ---------------------------------------------------------------------------

# Reusable styles
_HEADER_FONT = Font(bold=True, size=11)
_HEADER_FILL = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
_HEADER_FONT_WHITE = Font(bold=True, size=11, color="FFFFFF")
_SECTION_FONT = Font(bold=True, size=12, color="1F4E79")
_TITLE_FONT = Font(bold=True, size=14, color="1F4E79")
_WRAP_ALIGN = Alignment(wrap_text=True, vertical="top")
_CENTER_ALIGN = Alignment(horizontal="center", vertical="center")
_INPUT_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")  # light yellow
_THIN_BORDER = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"), bottom=Side(style="thin"),
)


def _style_header_row(ws, ncols: int):
    """Apply blue header with white bold font to row 1."""
    for col in range(1, ncols + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = _HEADER_FONT_WHITE
        cell.fill = _HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
        cell.border = _THIN_BORDER


def _highlight_input_cells(ws, col_letters: list[str], data_rows: int):
    """Highlight input columns with light yellow and add thin borders."""
    for col_letter in col_letters:
        for row in range(2, data_rows + 2):
            cell = ws[f"{col_letter}{row}"]
            cell.fill = _INPUT_FILL
            cell.alignment = _CENTER_ALIGN
            cell.border = _THIN_BORDER


def _set_column_widths(ws, widths: dict[str, int]):
    """Set column widths. Keys are column letters, values are character widths."""
    for col_letter, width in widths.items():
        ws.column_dimensions[col_letter].width = width


def _add_data_validation(ws, col_letter: str, formula: str, data_rows: int):
    """Add a dropdown data validation to a column."""
    dv = DataValidation(
        type="list", formula1=f'"{formula}"', allow_blank=True,
        showErrorMessage=True, errorTitle="Invalid input",
        error=f"Please select from: {formula}",
    )
    dv.add(f"{col_letter}2:{col_letter}{data_rows + 1}")
    ws.add_data_validation(dv)


def _freeze_and_filter(ws, freeze_cell: str = "A2"):
    """Freeze top row and enable auto-filter."""
    ws.freeze_panes = freeze_cell
    ws.auto_filter.ref = ws.dimensions


def _format_workbook(wb, relevance_df, nld_df, cat_df, tax_df):
    """Apply formatting, data validation, and styling to all sheets."""

    # --- Instructions sheet ---
    ws = wb["Instructions"]
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 90
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=2):
        for cell in row:
            cell.alignment = _WRAP_ALIGN
            cell.border = _THIN_BORDER
        section_text = str(row[0].value or "")
        if section_text.startswith("SHEET") or section_text in (
            "PURPOSE", "CALIBRATION EXAMPLES", "NOTES",
            "EXPERT EVALUATION — PreSaltOntoLearn Pipeline",
        ):
            row[0].font = _SECTION_FONT
            if section_text.startswith("EXPERT"):
                row[0].font = _TITLE_FONT
        elif section_text.startswith(("1 ", "2 ", "3 ", "4 ", "5 ")):
            row[0].font = Font(bold=True, size=11)

    # --- Term Relevance ---
    ws = wb["Term_Relevance"]
    n_rel = len(relevance_df)
    _style_header_row(ws, 3)
    _set_column_widths(ws, {"A": 35, "B": 16, "C": 40})
    _add_data_validation(ws, "B", "1,2,3,4,5", n_rel)
    _highlight_input_cells(ws, ["B", "C"], n_rel)
    _freeze_and_filter(ws)
    for row in range(2, n_rel + 2):
        ws.cell(row=row, column=1).alignment = _WRAP_ALIGN
        ws.cell(row=row, column=1).border = _THIN_BORDER

    # --- NLD Quality ---
    ws = wb["NLD_Quality"]
    n_nld = len(nld_df)
    _style_header_row(ws, 8)
    _set_column_widths(ws, {
        "A": 12, "B": 30, "C": 65, "D": 65,
        "E": 16, "F": 16, "G": 18, "H": 35,
    })
    _add_data_validation(ws, "E", "1,2,3,4,5", n_nld)
    _add_data_validation(ws, "F", "1,2,3,4,5", n_nld)
    _add_data_validation(ws, "G", "1,2,Tie", n_nld)
    _highlight_input_cells(ws, ["E", "F", "G", "H"], n_nld)
    _freeze_and_filter(ws)
    # Wrap definition text
    for row in range(2, n_nld + 2):
        for col in (1, 2, 3, 4):
            cell = ws.cell(row=row, column=col)
            cell.alignment = _WRAP_ALIGN
            cell.border = _THIN_BORDER
    ws.sheet_properties.defaultRowHeight = 80

    # --- Category Correct ---
    ws = wb["Category_Correct"]
    n_cat = len(cat_df)
    _style_header_row(ws, 8)
    _set_column_widths(ws, {
        "A": 12, "B": 30, "C": 14, "D": 30,
        "E": 50, "F": 18, "G": 30, "H": 35,
    })
    _add_data_validation(ws, "F", "Yes,No,Partial", n_cat)
    _highlight_input_cells(ws, ["F", "G", "H"], n_cat)
    _freeze_and_filter(ws)
    for row in range(2, n_cat + 2):
        for col in range(1, 9):
            cell = ws.cell(row=row, column=col)
            cell.alignment = _WRAP_ALIGN
            cell.border = _THIN_BORDER

    # --- Taxonomy Correct ---
    if "Taxonomy_Correct" in wb.sheetnames:
        ws = wb["Taxonomy_Correct"]
        n_tax = len(tax_df)
        _style_header_row(ws, 7)
        _set_column_widths(ws, {
            "A": 12, "B": 30, "C": 30, "D": 50,
            "E": 20, "F": 30, "G": 35,
        })
        _add_data_validation(ws, "E", "Yes,No,Partial", n_tax)
        _highlight_input_cells(ws, ["E", "F", "G"], n_tax)
        _freeze_and_filter(ws)
        for row in range(2, n_tax + 2):
            for col in range(1, 8):
                cell = ws.cell(row=row, column=col)
                cell.alignment = _WRAP_ALIGN
                cell.border = _THIN_BORDER


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
    instructions_data = build_instructions_sheet()
    relevance_df = build_term_relevance_sheet(terms)

    nld_expert_df, nld_key_df = build_nld_quality_sheet(terms, data["nld"], seed)
    cat_expert_df, cat_key_df = build_category_sheet(terms, data["cat"], seed)
    tax_expert_df, tax_key_df = build_taxonomy_sheet(terms, seed)

    # 4. Write expert workbook with formatting
    workbook_path = os.path.join(output_dir, "expert_evaluation.xlsx")

    # Write raw data first, then apply formatting
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        # Instructions — write as raw rows
        instr_rows = instructions_data
        instr_df = pd.DataFrame(instr_rows, columns=["Section", "Details"])
        instr_df.to_excel(writer, sheet_name="Instructions", index=False)

        relevance_df.to_excel(writer, sheet_name="Term_Relevance", index=False)
        nld_expert_df.to_excel(writer, sheet_name="NLD_Quality", index=False)
        cat_expert_df.to_excel(writer, sheet_name="Category_Correct", index=False)
        if not tax_expert_df.empty:
            tax_expert_df.to_excel(writer, sheet_name="Taxonomy_Correct", index=False)

        wb = writer.book
        _format_workbook(wb, relevance_df, nld_expert_df, cat_expert_df, tax_expert_df)

    # 5. Write blinding key (separate file)
    key_path = os.path.join(output_dir, f"blinding_key_{seed}.csv")
    key_parts = [nld_key_df, cat_key_df]
    if not tax_key_df.empty:
        key_parts.append(tax_key_df)
    key_combined = pd.concat(key_parts, ignore_index=True)
    write_csv(key_combined, key_path)

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


def generate_refined_evaluation(
    cq_matrix_csv: str,
    threshold: int,
    taxonomy_path: str,
    seed: int = 42,
    output_dir: str | None = None,
) -> tuple[str, str]:
    """Generate a combined expert workbook for ablation + CQ-filter validation.

    Samples ~150 kept + ~50 removed terms (blinded). Sheets:
      - Term_Relevance: all 200 terms (validates both ablation + CQ filter)
      - NLD_Quality: blinded A-vs-B (ablation)
      - Category_Correct: stratified by tier (ablation, uses all 4 conditions)
      - Taxonomy_Correct: IS-A pairs from the refined taxonomy

    The blinding key includes CQ_Status (kept/removed) and CQ_Count columns
    so post-hoc analysis can split kept vs removed.

    Args:
        cq_matrix_csv: Path to 5b_cq_matrix.csv.
        threshold: CQ count threshold (e.g., 1).
        taxonomy_path: Path to the refined threshold's construct_taxonomy.csv.
        seed: Random seed.
        output_dir: Output directory (default: output/refined/).

    Returns:
        (workbook_path, key_path)
    """
    if output_dir is None:
        output_dir = os.path.join("output", "refined")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRefined Expert Evaluation Generator (threshold≥{threshold}, seed={seed})")

    # 1. Select terms: 150 kept + 50 removed, stratified by category
    terms, status_map = select_terms_refined(cq_matrix_csv, threshold, seed=seed)

    # Build CQ_Count lookup for blinding key
    df_cq = read_csv(cq_matrix_csv)
    cq_count_map = dict(zip(df_cq["Term"], df_cq["CQ_Count"]))

    # 2. Load ablation data (for NLD_Quality + Category_Correct sheets)
    data = load_all_conditions()

    # 3. Build sheets
    instructions_data = build_instructions_sheet()
    relevance_df = build_term_relevance_sheet(terms)
    nld_expert_df, nld_key_df = build_nld_quality_sheet(terms, data["nld"], seed)
    cat_expert_df, cat_key_df = build_category_sheet(terms, data["cat"], seed)
    tax_expert_df, tax_key_df = build_taxonomy_sheet(
        terms, seed, taxonomy_path=taxonomy_path
    )

    # 4. Add CQ_Status and CQ_Count to blinding key
    # Term-level key rows (one per term, for CQ filter validation)
    cq_key_rows = []
    for term in terms:
        cq_key_rows.append({
            "Sheet": "CQ_Filter",
            "Row_ID": f"CQ-{term}",
            "Term": term,
            "CQ_Status": "kept" if status_map.get(term, True) else "removed",
            "CQ_Count": cq_count_map.get(term, 0),
        })
    cq_key_df = pd.DataFrame(cq_key_rows)

    # 5. Write workbook
    workbook_path = os.path.join(output_dir, "expert_evaluation_refined.xlsx")

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        instr_df = pd.DataFrame(instructions_data, columns=["Section", "Details"])
        instr_df.to_excel(writer, sheet_name="Instructions", index=False)
        relevance_df.to_excel(writer, sheet_name="Term_Relevance", index=False)
        nld_expert_df.to_excel(writer, sheet_name="NLD_Quality", index=False)
        cat_expert_df.to_excel(writer, sheet_name="Category_Correct", index=False)
        if not tax_expert_df.empty:
            tax_expert_df.to_excel(writer, sheet_name="Taxonomy_Correct", index=False)
        wb = writer.book
        _format_workbook(wb, relevance_df, nld_expert_df, cat_expert_df, tax_expert_df)

    # 6. Write blinding key (includes CQ_Filter section)
    key_path = os.path.join(output_dir, f"blinding_key_refined_{seed}.csv")
    key_parts = [nld_key_df, cat_key_df, cq_key_df]
    if not tax_key_df.empty:
        key_parts.append(tax_key_df)
    key_combined = pd.concat(key_parts, ignore_index=True)
    write_csv(key_combined, key_path)

    # 7. Summary
    n_kept = sum(1 for v in status_map.values() if v)
    n_removed = sum(1 for v in status_map.values() if not v)
    n_geores = len(cat_expert_df[cat_expert_df["Tier"] == "GeoReservoir"]) if "Tier" in cat_expert_df.columns else 0
    n_upper = len(cat_expert_df) - n_geores if not cat_expert_df.empty else 0
    print(f"\n  Workbook: {workbook_path}")
    print(f"    Terms: {n_kept} kept + {n_removed} removed (blinded)")
    print(f"    Sheet 'Term_Relevance': {len(relevance_df)} terms")
    print(f"    Sheet 'NLD_Quality': {len(nld_expert_df)} blinded A-vs-B comparisons")
    print(f"    Sheet 'Category_Correct': {len(cat_expert_df)} pairs ({n_geores} GeoReservoir, {n_upper} GeoCore/BFO)")
    if not tax_expert_df.empty:
        print(f"    Sheet 'Taxonomy_Correct': {len(tax_expert_df)} parent-child IS-A pairs")
    print(f"  Blinding key: {key_path} (DO NOT share with experts)")
    print(f"    CQ_Filter section: {n_kept} kept, {n_removed} removed (for Mann-Whitney U)")

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
