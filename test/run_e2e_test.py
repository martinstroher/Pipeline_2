import csv
import glob
import json
import os
import shutil
import subprocess
import sys
from dotenv import load_dotenv


# ~1500 chars of domain content across 5 sections.
# Each section has a distinct key term with definitional "X is a Y that..." phrasing
# so the LLM can use RAG context (Context_Used=True).
# 5 headers → 5 distinct chunks at chunk_size=1024.
# Target: ~15-20 extracted terms for a fast test run.
_TEST_MARKDOWN = """# Pre-Salt Petroleum Geology: Key Concepts

## Carbonate Reservoirs

Carbonate reservoir is a type of petroleum reservoir rock composed of carbonate minerals
such as calcite and dolomite. Microbialite is a sedimentary rock formed by the metabolic
activity of microbial communities, serving as the primary reservoir facies in the
Santos Basin Pre-Salt sequences.

## Diagenesis and Porosity

Porosity is a physical property of rock defined as the ratio of void space to total rock
volume. Dolomitization is a diagenetic process involving replacement of calcite by dolomite,
often enhancing reservoir porosity and permeability.

## Structural Geology and Traps

Structural trap is a type of petroleum trap formed by deformation of rock layers through
tectonic processes. Fault is a planar fracture in rock along which displacement has occurred,
serving as both migration pathway and barrier in Pre-Salt reservoirs.

## Depositional Environments

Lacustrine environment is a continental depositional setting associated with lake systems.
Evaporite is a chemical sedimentary rock formed by precipitation of dissolved minerals from
evaporating water bodies, acting as the regional seal in the Pre-Salt play.

## Stratigraphy and Basin Evolution

Rifting is a tectonic process involving extensional deformation of the lithosphere.
Reservoir quality is a composite property describing a rock's capacity to store and
transmit fluids, primarily controlled by porosity and permeability.
"""


def generate_test_content(inputs_dir):
    """Write the test .md file directly for RAG ingestion.

    We write the markdown directly (with proper ## headers) so the
    MarkdownHeaderTextSplitter can create multiple distinct chunks.
    The pipeline is run with --skip-pdf to bypass PDF->MD conversion.
    """
    md_path = os.path.join(inputs_dir, "test_doc.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_TEST_MARKDOWN)
    print(f"Generated test markdown at: {md_path} ({len(_TEST_MARKDOWN)} chars)")


def validate_csv(filepath, expected_columns, min_rows=1, label=""):
    """Validate a CSV file has expected columns and minimum row count."""
    errors = []
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            # Check expected columns
            missing_cols = set(expected_columns) - set(headers)
            if missing_cols:
                errors.append(f"Missing columns: {missing_cols}")

            # Count data rows and basic checks
            rows = list(reader)
            if len(rows) < min_rows:
                errors.append(f"Expected >= {min_rows} data rows, got {len(rows)}")

            # Check no row has all-empty values
            empty_rows = sum(1 for r in rows if all(v.strip() == "" for v in r.values()))
            if empty_rows > 0:
                errors.append(f"{empty_rows} completely empty rows found")

    except Exception as e:
        errors.append(f"Failed to parse CSV: {e}")

    if errors:
        print(f"  [FAIL] {label or filepath}: {'; '.join(errors)}")
    else:
        print(f"  [OK]   {label or filepath}: {len(rows)} rows, columns {headers}")
    return len(errors) == 0


def run_test():
    # 1. Read existing env (to keep PATH, etc.)
    load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
    env = os.environ.copy()

    # Check if key exists
    if not env.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not found in environment or ../.env!")
        sys.exit(1)

    # 2. Load .env-test manually to override
    print("Loading test configuration...")
    test_env_path = os.path.join(os.path.dirname(__file__), ".env-test")
    with open(test_env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, val = line.split("=", 1)
            # Expand ${VAR} references from parent environment
            if val.startswith("${") and val.endswith("}"):
                ref_var = val[2:-1]
                val = os.environ.get(ref_var, "")
            env[key] = val

    # Verify critical vars
    print(f"Test Configuration:")
    print(f"  DOCS_DIR: {env.get('DOCS_DIR')}")
    print(f"  LLM_INPUT_DIR: {env.get('LLM_INPUT_DIR')}")
    print(f"  OUTPUT: {env.get('LLM_OUTPUT_FILE')}")

    # Setup Paths
    base_test_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(base_test_dir)
    inputs_dir = os.path.join(base_test_dir, "inputs_test")
    output_dir = os.path.join(base_test_dir, "output_test")
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Generate test markdown content (written directly with ## headers for RAG chunking)
    generate_test_content(inputs_dir)

    # 3. Clean previous test outputs and intermediary files (Python-native: rm -rf silently fails on Windows)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for d in glob.glob(os.path.join(root_dir, "test", "chroma_db_test*")):
        if os.path.isdir(d):
            shutil.rmtree(d)

    for pattern in ["*.txt.ia", "*.ocr", "*.json", "*.dat"]:
        for f in glob.glob(os.path.join(inputs_dir, pattern)):
            os.remove(f)

    # Ensure subprocess uses UTF-8 for ANSI escape codes in log output
    env["PYTHONIOENCODING"] = "utf-8"

    # 4. Run full pipeline (Steps 0-7) as subprocess from ROOT
    print("\nRunning pipeline subprocess (Steps 0-7)...")
    result = subprocess.run(
        [sys.executable, "pipeline.py", "--skip-pdf"],
        env=env,
        cwd=root_dir,
        capture_output=False,
    )
    if result.returncode != 0:
        print("\n[FAIL] Pipeline (Steps 0-7) exited with non-zero return code!")
        sys.exit(result.returncode)

    # Step 5b rebases all post-classify artifacts under <input-dir>/refined/.
    # That subfolder holds: classify_categories.csv (filtered), construct_taxonomy.csv,
    # construct_relations.csv, validate_taxonomy.csv, validate_relations.csv,
    # validate_edits.csv, validate_taxonomy.ttl, emit_verification.json.
    cat_csv_relative = env.get("CATEGORIZED_LLM_TERMS", "test/output_test/classify_categories.csv")
    refined_dir_rel = os.path.join(os.path.dirname(cat_csv_relative), "refined")
    refined_dir_abs = os.path.join(root_dir, refined_dir_rel)
    taxonomy_csv_abs = os.path.join(refined_dir_abs, "construct_taxonomy.csv")
    validate_tax_abs = os.path.join(refined_dir_abs, "validate_taxonomy.csv")
    validate_rel_abs = os.path.join(refined_dir_abs, "validate_relations.csv")
    # OWL exporter uses the critic's validate_taxonomy.csv -> validate_taxonomy.ttl by
    # default (the construct_taxonomy -> emit_ontology rename only fires when the input
    # name still contains 'construct_taxonomy').
    _ttl_candidates = [
        os.path.join(refined_dir_abs, "validate_taxonomy.ttl"),
        os.path.join(refined_dir_abs, "emit_ontology.ttl"),
        os.path.join(refined_dir_abs, "construct_taxonomy.ttl"),
    ]
    owl_ttl_abs = next(
        (p for p in _ttl_candidates if os.path.exists(p)),
        _ttl_candidates[0],
    )

    # 5. Assertions — file existence + content validation
    print("\nPipeline execution finished. Verifying artifacts...\n")

    all_passed = True

    # --- Step 0: Verify .md file exists (we provided it directly) ---
    md_files = [f for f in os.listdir(inputs_dir) if f.endswith(".md")]
    if md_files:
        md_path = os.path.join(inputs_dir, md_files[0])
        size = os.path.getsize(md_path)
        if size > 500:  # Rich test doc should be > 3000 chars
            print(f"  [OK]   Test markdown: {md_files[0]} ({size} bytes)")
        else:
            print(f"  [FAIL] Test markdown: {md_files[0]} too small ({size} bytes)")
            all_passed = False
    else:
        print("  [FAIL] No .md file found in inputs_test/")
        all_passed = False

    # --- Step 1: Raw extraction (CSV with .json extension) ---
    f1 = os.path.join(root_dir, "test/output_test/extract_raw.json")
    if os.path.exists(f1):
        ok = validate_csv(f1, expected_columns=["Entity"], min_rows=1, label="Step 1 (extraction)")
        all_passed = all_passed and ok
    else:
        print("  [FAIL] Step 1 (extraction): file missing")
        all_passed = False

    # --- Step 2: Aggregated counts ---
    f2 = os.path.join(root_dir, "test/output_test/extract_aggregated.csv")
    if os.path.exists(f2):
        ok = validate_csv(f2, expected_columns=["Readable_Term", "Frequency"], min_rows=1, label="Step 2 (aggregation)")
        if ok:
            # Check frequencies are positive integers
            with open(f2, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                bad_freqs = [r for r in reader if not r["Frequency"].strip().isdigit() or int(r["Frequency"]) < 1]
                if bad_freqs:
                    print(f"  [FAIL] Step 2: {len(bad_freqs)} rows with invalid Frequency")
                    ok = False
        all_passed = all_passed and ok
    else:
        print("  [FAIL] Step 2 (aggregation): file missing")
        all_passed = False

    # --- Step 3: Filtered terms ---
    f3 = os.path.join(root_dir, "test/output_test/extract_filtered.csv")
    if os.path.exists(f3):
        ok = validate_csv(f3, expected_columns=["Readable_Term", "Frequency"], min_rows=1, label="Step 3 (filter)")
        all_passed = all_passed and ok
    else:
        print("  [FAIL] Step 3 (filter): file missing")
        all_passed = False

    # --- Step 4: NLD generation ---
    f4 = os.path.join(root_dir, "test/output_test/define_nld.csv")
    if os.path.exists(f4):
        ok = validate_csv(f4, expected_columns=["Term", "NLD", "Context_Used", "Context"], min_rows=1, label="Step 4 (NLD)")
        if ok:
            # Check NLDs are not empty strings
            with open(f4, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                empty_nlds = [r for r in rows if not r.get("NLD", "").strip()]
                if empty_nlds:
                    print(f"  [WARN] Step 4: {len(empty_nlds)}/{len(rows)} terms have empty NLDs")

            # Check RAG discrimination: not all Context values should be identical
            contexts = set(r.get("Context", "").strip() for r in rows)
            if len(contexts) <= 1 and len(rows) > 1:
                print(f"  [WARN] Step 4: All {len(rows)} terms received identical RAG context (RAG not discriminating)")
            else:
                print(f"  [OK]   Step 4 RAG discrimination: {len(contexts)} distinct contexts across {len(rows)} terms")

        all_passed = all_passed and ok
    else:
        print("  [FAIL] Step 4 (NLD): file missing")
        all_passed = False

    # --- Step 5: Categorization ---
    f5 = os.path.join(root_dir, "test/output_test/classify_categories.csv")
    if os.path.exists(f5):
        ok = validate_csv(
            f5,
            expected_columns=["Term", "RAG_Context_Used", "Category", "Reasoning", "NLD"],
            min_rows=1,
            label="Step 5 (categorization)",
        )
        if ok:
            # Check categories are not all ERROR
            with open(f5, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                error_cats = [r for r in rows if r.get("Category", "").startswith("ERROR")]
                if error_cats and len(error_cats) == len(rows):
                    print(f"  [FAIL] Step 5: ALL {len(rows)} terms have ERROR categories")
                    ok = False
                elif error_cats:
                    print(f"  [WARN] Step 5: {len(error_cats)}/{len(rows)} terms have ERROR categories")
        all_passed = all_passed and ok
    else:
        print("  [FAIL] Step 5 (categorization): file missing")
        all_passed = False

    # --- Step 6: Taxonomy CSV ---
    if os.path.exists(taxonomy_csv_abs):
        ok = validate_csv(
            taxonomy_csv_abs,
            expected_columns=["Term", "Parent_Term", "Relationship_Type", "Category", "Is_Intermediate", "FALLBACK", "NLD"],
            min_rows=1,
            label="Step 6 (taxonomy)",
        )
        if ok:
            with open(taxonomy_csv_abs, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            classes = [r for r in rows if r.get("Relationship_Type") == "rdfs:subClassOf"]
            individuals = [r for r in rows if r.get("Relationship_Type") == "rdf:type"]
            intermediates = [r for r in rows if r.get("Is_Intermediate", "").strip().lower() == "true"]
            with_nld = [r for r in rows if r.get("NLD", "").strip()]
            print(f"  [OK]   Step 6: {len(classes)} classes, {len(individuals)} individuals, {len(intermediates)} intermediate nodes")
            print(f"  [OK]   Step 6: {len(with_nld)}/{len(rows)} rows have NLDs")
            # Report flat fallbacks
            fallbacks = [r for r in rows if r.get("FALLBACK", "").strip().lower() == "true"]
            if fallbacks:
                print(f"  [WARN] Step 6: {len(fallbacks)} entries used flat fallback (LLM error) — check taxonomy quality")
            # Non-intermediate terms should all have NLDs (warn if gap)
            missing_nld = [r for r in rows if r.get("Is_Intermediate", "").strip().lower() != "true" and not r.get("NLD", "").strip()]
            if missing_nld:
                print(f"  [WARN] Step 6: {len(missing_nld)} non-intermediate terms missing NLD")
        all_passed = all_passed and ok
    else:
        print("  [FAIL] Step 6 (taxonomy): file missing")
        all_passed = False

    # --- Step 6b: Relation Extraction ---
    f6b = os.path.join(refined_dir_abs, "construct_relations.csv")
    if os.path.exists(f6b):
        ok = validate_csv(
            f6b,
            expected_columns=["Term", "Category", "Property", "Property_IRI", "Filler",
                              "Filler_Source", "Confidence", "Evidence",
                              "Validation_Status", "Validation_Reason"],
            min_rows=1,
            label="Step 6b (relations)",
        )
        if ok:
            with open(f6b, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rel_rows = list(reader)
            accepted = [r for r in rel_rows if r.get("Validation_Status") == "ACCEPTED"]
            rejected = [r for r in rel_rows if r.get("Validation_Status") == "REJECTED"]
            errors = [r for r in rel_rows if r.get("Validation_Status") == "ERROR"]
            # Property distribution
            prop_counts = {}
            for r in accepted:
                prop = r.get("Property", "")
                prop_counts[prop] = prop_counts.get(prop, 0) + 1
            print(f"  [OK]   Step 6b: {len(accepted)} accepted, {len(rejected)} rejected, {len(errors)} errors")
            if prop_counts:
                top_props = sorted(prop_counts.items(), key=lambda x: -x[1])[:5]
                print(f"  [OK]   Step 6b top properties: {', '.join(f'{p}={c}' for p, c in top_props)}")
            # Filler source distribution
            domain_fillers = sum(1 for r in accepted if r.get("Filler_Source") == "domain_term")
            external_fillers = sum(1 for r in accepted if r.get("Filler_Source") == "external")
            print(f"  [OK]   Step 6b fillers: {domain_fillers} domain, {external_fillers} external")
        all_passed = all_passed and ok
    else:
        print("  [WARN] Step 6b (relations): file missing — relation extraction may have been skipped")

    # --- Step 7: OWL Turtle ---
    if os.path.exists(owl_ttl_abs):
        try:
            from rdflib import Graph
            from rdflib.namespace import OWL, RDF, RDFS
            g = Graph()
            g.parse(owl_ttl_abs, format="turtle")
            n_classes = len(list(g.subjects(RDF.type, OWL.Class)))
            n_individuals = len(list(g.subjects(RDF.type, OWL.NamedIndividual)))
            n_comments = len(list(g.triples((None, RDFS.comment, None))))
            n_labels = len(list(g.triples((None, RDFS.label, None))))
            n_restrictions = len(list(g.subjects(RDF.type, OWL.Restriction)))
            n_triples = len(g)
            if n_classes > 0 and n_triples > 0:
                print(f"  [OK]   Step 7 (OWL): {n_triples} triples — {n_classes} classes, {n_individuals} individuals")
                print(f"  [OK]   Step 7: {n_labels} rdfs:label, {n_comments} rdfs:comment (NLDs)")
                if n_restrictions > 0:
                    print(f"  [OK]   Step 7: {n_restrictions} owl:Restriction nodes (from relation extraction)")
                if n_comments == 0:
                    print(f"  [WARN] Step 7: no rdfs:comment — NLDs not propagating to OWL")

                # Check ontology is anchored to upper ontologies (BFO / GeoCore / GeoReservoir)
                _BFO_PREFIX = "http://purl.obolibrary.org/obo/"
                _ONTO_PREFIX = "https://www.inf.ufrgs.br/bdi/ontologies/"
                upper_iris_used = set(
                    str(o) for _, _, o in g
                    if str(o).startswith(_BFO_PREFIX) or str(o).startswith(_ONTO_PREFIX)
                )
                # Exclude the owl:imports declaration itself (that's the ontology header, not a class link)
                upper_iris_used.discard("http://purl.obolibrary.org/obo/bfo.owl")
                if upper_iris_used:
                    print(f"  [OK]   Step 7 upper ontology anchoring: {len(upper_iris_used)} distinct BFO/GeoCore/GeoReservoir IRIs referenced in OWL")
                else:
                    print(f"  [FAIL] Step 7: no triples reference BFO/GeoCore/GeoReservoir IRIs — ontology not anchored to upper ontologies")
                    all_passed = False
            else:
                print(f"  [FAIL] Step 7 (OWL): parsed but empty ({n_triples} triples, {n_classes} classes)")
                all_passed = False
        except Exception as e:
            print(f"  [FAIL] Step 7 (OWL): parse error: {e}")
            all_passed = False
    else:
        print("  [FAIL] Step 7 (OWL): file missing")
        all_passed = False

    # --- Step 7b: Verification report ---
    verify_report = os.path.join(refined_dir_abs, "emit_verification.json")
    if os.path.exists(verify_report):
        try:
            with open(verify_report, "r", encoding="utf-8") as f:
                vr = json.load(f)
            overall = vr.get("overall_status", "UNKNOWN")
            syntax_status = vr.get("layers", {}).get("syntax", {}).get("status", "UNKNOWN")
            struct = vr.get("layers", {}).get("structure", {})
            struct_status = struct.get("status", "UNKNOWN")
            n_issues = len(struct.get("issues", []))
            summary = struct.get("issue_summary", {})
            print(f"  [OK]   Step 7b (verification): overall={overall}, syntax={syntax_status}, structure={struct_status}")
            if summary:
                print(f"  [OK]   Step 7b issues: {summary}")
            if overall == "FAIL":
                print(f"  [WARN] Step 7b: verification FAILED — {n_issues} issue(s) found")

        except Exception as e:
            print(f"  [WARN] Step 7b: could not parse report: {e}")
    else:
        print("  [WARN] Step 7b (verification): report missing")

    # --- Cross-step validation: term counts should be consistent ---
    print("\n--- Cross-step consistency checks ---")
    try:
        with open(f3, "r", encoding="utf-8-sig") as f:
            filtered_count = sum(1 for _ in csv.DictReader(f))
        with open(f4, "r", encoding="utf-8-sig") as f:
            nld_count = sum(1 for _ in csv.DictReader(f))
        with open(f5, "r", encoding="utf-8-sig") as f:
            cat_count = sum(1 for _ in csv.DictReader(f))

        if nld_count == filtered_count:
            print(f"  [OK]   Steps 3->4: filtered ({filtered_count}) == NLD ({nld_count})")
        else:
            print(f"  [WARN] Steps 3->4: filtered ({filtered_count}) != NLD ({nld_count})")

        if cat_count == nld_count:
            print(f"  [OK]   Steps 4->5: NLD ({nld_count}) == categorized ({cat_count})")
        else:
            print(f"  [WARN] Steps 4->5: NLD ({nld_count}) != categorized ({cat_count})")

        if os.path.exists(taxonomy_csv_abs):
            with open(taxonomy_csv_abs, "r", encoding="utf-8-sig") as f:
                tax_rows = list(csv.DictReader(f))
            tax_input_count = sum(1 for r in tax_rows if r.get("Is_Intermediate", "").strip().lower() != "true")
            with open(f5, "r", encoding="utf-8-sig") as f:
                valid_cat_count = sum(
                    1 for r in csv.DictReader(f)
                    if not r.get("Category", "").startswith("ERROR") and r.get("Category") != "NOT_CLASSIFIED"
                )
            print(f"  [OK]   Steps 5->6: {valid_cat_count} valid categorized -> {tax_input_count} taxonomy input terms, +{len(tax_rows)-tax_input_count} intermediate nodes")

    except Exception as e:
        print(f"  [WARN] Could not run cross-step checks: {e}")

    # --- Final verdict ---
    print()
    if all_passed:
        print("E2E Test Passed Successfully!")
    else:
        print("E2E Test Failed.")
        sys.exit(1)


if __name__ == "__main__":
    run_test()
