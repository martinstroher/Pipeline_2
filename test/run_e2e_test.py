import csv
import os
import subprocess
import sys
from dotenv import load_dotenv


# ~4200 chars of domain content across 5 sections.
# Each section has distinct terms with definitional "X is a Y that..." phrasing
# so the LLM can use RAG context (Context_Used=True).
# At chunk_size=1024 this should produce 4-5 chunks via MarkdownHeaderTextSplitter.
_TEST_MARKDOWN = """# Pre-Salt Petroleum Geology: Key Concepts

## Carbonate Reservoirs

Carbonate reservoir is a type of petroleum reservoir rock that is primarily composed of
carbonate minerals such as calcite and dolomite, formed through biological and chemical
precipitation in marine or lacustrine environments. In the Brazilian Pre-Salt context,
carbonate reservoirs are the dominant reservoir lithology, hosting the majority of
discovered hydrocarbon accumulations beneath the evaporite seal.

Microbialite is a sedimentary rock that is formed by the metabolic activity of
microbial communities, particularly cyanobacteria, which mediate carbonate precipitation.
Microbialites are the primary reservoir facies in the Santos and Campos Basin Pre-Salt
sequences, exhibiting high primary porosity due to their stromatolitic and thrombolitic
growth fabrics. Coquina is a bioclastic carbonate rock that is composed predominantly
of accumulated bivalve shells and shell fragments, deposited in high-energy lacustrine
shoreline environments during the rift phase of basin evolution.

## Diagenesis and Porosity

Diagenesis is a geological process that encompasses all physical, chemical, and
biological changes occurring in sediments after deposition and before metamorphism.
In Pre-Salt carbonates, diagenesis controls reservoir quality through processes such
as cementation, dissolution, dolomitization, and compaction. Early marine cementation
can reduce primary porosity, while later dissolution creates secondary porosity.

Porosity is a physical property of rock that is defined as the ratio of void space to
total rock volume, expressed as a percentage. In Pre-Salt reservoirs, porosity types include
interparticle, moldic, vuggy, and fracture porosity. Dolomitization is a diagenetic
process that involves the replacement of calcite by dolomite, often enhancing reservoir
porosity and permeability through the creation of intercrystalline pore space.

## Structural Geology and Traps

Structural trap is a type of petroleum trap that is formed by deformation of rock
layers through tectonic processes such as faulting and folding. In the Pre-Salt play,
structural traps are commonly associated with rift-related normal faults and horst-graben
geometries that compartmentalize the reservoir. Fault is a planar fracture in rock along
which displacement has occurred, and in the Pre-Salt context, faults serve dual roles as
both migration pathways and barriers to fluid flow depending on their sealing properties.

Source rock is a geological unit that contains sufficient organic matter to generate
hydrocarbons upon thermal maturation. The Pre-Salt source rocks are lacustrine shales
rich in Type I kerogen, deposited during the sag phase in anoxic lake environments.
These source rocks achieved thermal maturity during burial beneath thick evaporite
sequences, charging the overlying carbonate reservoirs.

## Depositional Environments

Lacustrine environment is a continental depositional setting that is associated with
lake systems, where sedimentation is controlled by climate, tectonics, and water chemistry.
The Pre-Salt lacustrine system of the South Atlantic rift basins hosted carbonate platforms,
siliciclastic fans, and organic-rich mudstones. Sedimentary facies is a body of rock with
specified characteristics that reflect the conditions of its formation, including grain
size, composition, sedimentary structures, and fossil content.

Evaporite is a chemical sedimentary rock that is formed by the precipitation of dissolved
minerals from evaporating water bodies. The massive salt layer (primarily halite) overlying
the Pre-Salt carbonates acts as a regional seal, trapping hydrocarbons in the underlying
reservoirs. This evaporite sequence can reach several kilometers in thickness in the
Santos Basin.

## Stratigraphy and Basin Evolution

Rifting is a tectonic process that involves the extensional deformation and thinning of
the lithosphere, creating rift basins through normal faulting and crustal stretching.
The rift phase of South Atlantic basin evolution produced the initial accommodation space
and deposited the lacustrine source rocks and early carbonate sequences during the
Early Cretaceous period.

Reservoir quality is a composite geological property that describes the capacity of a
rock to store and transmit fluids, determined primarily by porosity and permeability.
In the Pre-Salt carbonates, reservoir quality is controlled by the interplay of
depositional facies, diagenetic history, and structural deformation, with microbialites
and coquinas representing the highest-quality reservoir intervals.
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

    # 3. Clean previous test outputs and intermediary files
    subprocess.run("rm -rf test/output_test/* test/chroma_db_test*", shell=True, cwd=root_dir)
    subprocess.run("rm -f test/inputs_test/*.txt.ia test/inputs_test/*.ocr", shell=True, cwd=root_dir)
    subprocess.run("rm -f test/inputs_test/*.json test/inputs_test/*.dat", shell=True, cwd=root_dir)

    # Ensure subprocess uses UTF-8 for ANSI escape codes in log output
    env["PYTHONIOENCODING"] = "utf-8"

    # 4. Run Pipeline as subprocess from ROOT (--skip-pdf since we provide the .md directly)
    print("\nRunning pipeline subprocess...")
    result = subprocess.run(
        [sys.executable, "pipeline.py", "--skip-pdf"],
        env=env,
        cwd=root_dir,
        capture_output=False,
    )

    if result.returncode != 0:
        print("\n[FAIL] Pipeline exited with non-zero return code!")
        sys.exit(result.returncode)

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
    f1 = os.path.join(root_dir, "test/output_test/1_raw_llm_extraction.json")
    if os.path.exists(f1):
        ok = validate_csv(f1, expected_columns=["Entity"], min_rows=1, label="Step 1 (extraction)")
        all_passed = all_passed and ok
    else:
        print("  [FAIL] Step 1 (extraction): file missing")
        all_passed = False

    # --- Step 2: Aggregated counts ---
    f2 = os.path.join(root_dir, "test/output_test/2_aggregated_counts.csv")
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
    f3 = os.path.join(root_dir, "test/output_test/3_filtered_top_terms.csv")
    if os.path.exists(f3):
        ok = validate_csv(f3, expected_columns=["Readable_Term", "Frequency"], min_rows=1, label="Step 3 (filter)")
        all_passed = all_passed and ok
    else:
        print("  [FAIL] Step 3 (filter): file missing")
        all_passed = False

    # --- Step 4: NLD generation ---
    f4 = os.path.join(root_dir, "test/output_test/4_nld_generated_definitions.csv")
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
    f5 = os.path.join(root_dir, "test/output_test/5_categorized_ontology.csv")
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
