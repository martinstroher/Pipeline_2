"""
Taxonomy Builder — Step 6 of the PreSaltOntoLearn pipeline.

Takes the winning condition's categorized output (flat: Term -> Category)
and builds IS-A hierarchies WITHIN each category using LLM group reasoning.

Key design decisions:
  - Processes all terms in a category GROUP (not one-by-one) for tree consistency
  - Explicitly distinguishes classes vs named individuals
  - Uses published BFO/GeoCore/GeoReservoir IRIs for upper-level anchoring

Output: 6_taxonomy.csv with columns (Term, Parent_Term, Relationship_Type, Category)
"""

import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils.gemini_client import get_client, generate
from src.utils import log

# Published OWL IRIs for upper-level ontology anchoring
# Sources: BFO (http://purl.obolibrary.org/obo/bfo.owl)
#           GeoCore (https://www.inf.ufrgs.br/bdi/ontologies/geocore.owl)
#           GeoReservoir (https://www.inf.ufrgs.br/bdi/ontologies/geores.owl)
UPPER_IRIS = {
    # BFO top-level
    "entity": "http://purl.obolibrary.org/obo/BFO_0000001",
    "continuant": "http://purl.obolibrary.org/obo/BFO_0000002",
    "occurrent": "http://purl.obolibrary.org/obo/BFO_0000003",
    "independent continuant": "http://purl.obolibrary.org/obo/BFO_0000004",
    "spatial region": "http://purl.obolibrary.org/obo/BFO_0000006",
    "temporal region": "http://purl.obolibrary.org/obo/BFO_0000008",
    "spatiotemporal region": "http://purl.obolibrary.org/obo/BFO_0000011",
    "process": "http://purl.obolibrary.org/obo/BFO_0000015",
    "quality": "http://purl.obolibrary.org/obo/BFO_0000019",
    "specifically dependent continuant": "http://purl.obolibrary.org/obo/BFO_0000020",
    "fiat object part": "http://purl.obolibrary.org/obo/BFO_0000024",
    "object aggregate": "http://purl.obolibrary.org/obo/BFO_0000027",
    "site": "http://purl.obolibrary.org/obo/BFO_0000029",
    "object": "http://purl.obolibrary.org/obo/BFO_0000030",
    "generically dependent continuant": "http://purl.obolibrary.org/obo/BFO_0000031",
    "process boundary": "http://purl.obolibrary.org/obo/BFO_0000035",
    "one-dimensional temporal region": "http://purl.obolibrary.org/obo/BFO_0000038",
    "material entity": "http://purl.obolibrary.org/obo/BFO_0000040",
    "continuant fiat boundary": "http://purl.obolibrary.org/obo/BFO_0000140",
    "immaterial entity": "http://purl.obolibrary.org/obo/BFO_0000141",
    "relational quality": "http://purl.obolibrary.org/obo/BFO_0000145",
    "fiat surface": "http://purl.obolibrary.org/obo/BFO_0000146",
    # GeoCore (namespace: https://www.inf.ufrgs.br/bdi/ontologies/)
    "Geological Object": "https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000001",
    "Geological Process": "https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000002",
    "Geological Age": "https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000003",
    "Geological Structure": "https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000004",
    "Geological Time Interval": "https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000005",
    "Earth Material": "https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000006",
    "Rock": "https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000008",
    "Earth Fluid": "https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000009",
    "Geological Boundary": "https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000011",
    "Geological Contact": "https://www.inf.ufrgs.br/bdi/ontologies/GEOCORE_0000012",
    # GeoReservoir (namespace: https://www.inf.ufrgs.br/bdi/ontologies/)
    "Sedimentary Rock": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000032",
    "Sediment": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000031",
    "Depositional Unit": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000006",
    "Channel Unit": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000008",
    "Lobe Unit": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000014",
    "Levee Unit": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000013",
    "Mound Unit": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000015",
    "Depositional System": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000020",
    "Sedimentary Facies": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000029",
    "Facies Association": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000030",
    "Sedimentary Structure": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000025",
    "Fossil": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000018",
    "Facies": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000028",
    "Geometry": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000023",
    "Dimension": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000022",
    "Sinuosity": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000027",
    "Sedimentary Environment": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000021",
    "Lithology": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000024",
    "Formation": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000004",
    "Stratigraphic Unit": "https://www.inf.ufrgs.br/bdi/ontologies/GEORES_0000002",
}


def _configure_genai():
    """Configure Gemini client."""
    get_client()


def build_taxonomy_for_group(
    category: str, terms_with_nlds: list[dict],
    model_name: str, model_temperature: float,
) -> list[dict]:
    """
    Build IS-A hierarchy for a group of terms within the same category.

    Args:
        category: The ontology category (e.g., "Sedimentary Rock")
        terms_with_nlds: List of {"term": ..., "nld": ...}
        model: Configured Gemini model

    Returns:
        List of {"Term", "Parent_Term", "Relationship_Type", "Category"}
    """
    system_instruction = (
        "You are an expert ontology engineer building a geological taxonomy. "
        "You must arrange terms into IS-A (subClassOf) hierarchies. "
        "You distinguish between classes (types/kinds) and named individuals (specific instances)."
    )

    upper_vocab = ", ".join(sorted(UPPER_IRIS.keys()))

    prompt = f"""You are given a set of geological terms, all pre-classified under the ontology category "{category}".
Your task is to arrange them into an IS-A hierarchy (taxonomy tree).

**RULES:**
1. Every term MUST have exactly one parent. The root parent is "{category}" (the category itself).
2. Create intermediate classes if needed for a natural hierarchy.
   - **NLD-guided naming:** If a term's NLD follows the Aristotelian pattern "X is a Y that Z",
     use the genus Y as the intermediate class name
     (e.g., NLD "Grainstone is a grain-supported carbonate rock that lacks mud matrix"
     → intermediate class = "Carbonate Rock", not "CarbonateSubtype" or "GrainRock").
3. **Class vs Individual distinction (CRITICAL):**
   - Named geological time periods (e.g., "Cretaceous", "Aptian", "Albian") are INDIVIDUALS, not classes.
     Use relationship_type = "rdf:type" (not "rdfs:subClassOf").
   - Named locations, basins, formations are INDIVIDUALS.
   - General types/kinds (e.g., "Grainstone", "Fault", "Porosity") are CLASSES.
     Use relationship_type = "rdfs:subClassOf".
4. Intermediate classes you create should use Title Case.
5. Keep the hierarchy depth reasonable (2-4 levels below the category root).
6. **Canonical vocabulary:** When one of the following established class names is the natural parent
   for a term or intermediate node, use it verbatim (exact label, Title Case):
   {upper_vocab}

**INPUT (terms and their NLDs):**
{json.dumps(terms_with_nlds, indent=2)}

**OUTPUT FORMAT:**
Return a JSON array where each object has:
- "term": the original term
- "parent_term": immediate parent (either another term, an intermediate class you created, or "{category}")
- "relationship_type": either "rdfs:subClassOf" (for classes) or "rdf:type" (for individuals)
- "is_intermediate": false for input terms, true for intermediate classes you created

Include intermediate classes you create as separate entries in the array.

Return ONLY the JSON array, nothing else.
"""

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
            rows.append({
                "Term": item["term"],
                "Parent_Term": item["parent_term"],
                "Relationship_Type": item.get("relationship_type", "rdfs:subClassOf"),
                "Category": category,
                "Is_Intermediate": item.get("is_intermediate", False),
            })
        return rows
    except Exception as e:
        log.error(f"Building taxonomy for '{category}': {e}")
        # Fallback: flat hierarchy (all terms directly under category)
        return [
            {
                "Term": t["term"],
                "Parent_Term": category,
                "Relationship_Type": "rdfs:subClassOf",
                "Category": category,
                "Is_Intermediate": False,
            }
            for t in terms_with_nlds
        ]


def run_taxonomy_builder(categorized_csv: str, output_path: str | None = None):
    """
    Build taxonomy from a categorized CSV.

    Args:
        categorized_csv: Path to categorized output (e.g., cat_A.csv or 5_categorized_ontology.csv)
        output_path: Output path (default: derived from input)
    """
    load_dotenv()
    _configure_genai()

    if output_path is None:
        base = os.path.splitext(categorized_csv)[0]
        output_path = base.replace("cat_", "6_taxonomy_").replace(
            "5_categorized_ontology", "6_taxonomy"
        ) + ".csv"

    df = pd.read_csv(categorized_csv, encoding="utf-8-sig")
    log.info(f"Taxonomy builder: {len(df)} terms from {categorized_csv}")

    # Filter out errors and NOT_CLASSIFIED
    df_valid = df[
        ~df["Category"].str.startswith("ERROR", na=False)
        & (df["Category"] != "NOT_CLASSIFIED")
    ].copy()
    log.detail(f"{len(df_valid)} valid terms (excluding errors and NOT_CLASSIFIED)")

    MODEL_NAME = os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    MODEL_TEMPERATURE = float(os.environ.get("LLM_GENERATION_TEMPERATURE", 0))

    all_taxonomy_rows = []
    categories = df_valid["Category"].unique()

    for cat in tqdm(sorted(categories), desc="Building taxonomy", unit="category"):
        group = df_valid[df_valid["Category"] == cat]

        terms_with_nlds = []
        for _, row in group.iterrows():
            terms_with_nlds.append({
                "term": row["Term"],
                "nld": row.get("NLD", ""),
            })

        # For large groups, process in chunks of 150
        nld_lookup = {t["term"]: t.get("nld", "") for t in terms_with_nlds}
        if len(terms_with_nlds) > 150:
            for chunk_start in range(0, len(terms_with_nlds), 150):
                chunk = terms_with_nlds[chunk_start : chunk_start + 150]
                rows = build_taxonomy_for_group(cat, chunk, MODEL_NAME, MODEL_TEMPERATURE)
                for row in rows:
                    row["NLD"] = nld_lookup.get(row["Term"], "")
                all_taxonomy_rows.extend(rows)
                time.sleep(2)
        else:
            rows = build_taxonomy_for_group(cat, terms_with_nlds, MODEL_NAME, MODEL_TEMPERATURE)
            for row in rows:
                row["NLD"] = nld_lookup.get(row["Term"], "")
            all_taxonomy_rows.extend(rows)
            time.sleep(2)

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
    taxonomy_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # Stats
    n_classes = len(taxonomy_df[taxonomy_df["Relationship_Type"] == "rdfs:subClassOf"])
    n_individuals = len(taxonomy_df[taxonomy_df["Relationship_Type"] == "rdf:type"])
    n_intermediate = len(taxonomy_df[taxonomy_df["Is_Intermediate"] == True])
    log.success(f"Taxonomy built: {len(taxonomy_df)} entries (classes={n_classes}, individuals={n_individuals}, intermediate={n_intermediate})")
    log.detail(f"Saved to: {output_path}")

    return taxonomy_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to categorized CSV")
    parser.add_argument("--output", default=None, help="Output path")
    args = parser.parse_args()
    run_taxonomy_builder(args.input_csv, args.output)
