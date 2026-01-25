import os
import subprocess
import sys
from dotenv import load_dotenv

def run_test():
    # Load test environment variables
    # We use stream=True to avoid polluting the current process too much, 
    # but constructing a fresh env dict for the subprocess is better.
    
    # 1. Read existing env (to keep PATH, etc.)
    load_dotenv(".env") # Load production .env to get the real API Key
    env = os.environ.copy()
    
    # Check if key exists
    if not env.get("GEMINI_API_KEY"):
         print("WARNING: GEMINI_API_KEY not found in environment or .env!")

    # 2. Load .env-test manually to override
    # We can't easily use load_dotenv into a dict, so we'll parse it manually 
    # OR better: use load_dotenv then copy back.
    
    print("Loading configuration from .env-test...")
    # Parse .env-test manually to ensure we override everything
    with open(".env-test", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, val = line.split("=", 1)
            # Handle variable expansion if needed (simplistic here)
            if val == "${GEMINI_API_KEY}":
                val = os.environ.get("GEMINI_API_KEY", "")
            
            env[key] = val
            
    # Verify critical vars
    print(f"Test Configuration:")
    print(f"  DOCS_DIR: {env.get('DOCS_DIR')}")
    print(f"  LLM_INPUT_DIR: {env.get('LLM_INPUT_DIR')}")
    print(f"  OUTPUT: {env.get('LLM_OUTPUT_FILE')}")
    
    # 3. Clean previous test outputs
    subprocess.run("rm -rf output_test/* chroma_db_test", shell=True)
    
    # 4. Run Pipeline as subprocess
    print("\nRunning pipeline subprocess...")
    result = subprocess.run(
        [sys.executable, "pipeline.py"], 
        env=env,
        capture_output=False  # Stream output to console
    )
    
    if result.returncode != 0:
        print("\n❌ Pipeline failed!")
        sys.exit(result.returncode)
        
    # 5. Assetions
    print("\n✅ Pipeline execution finished. Verifying artifacts...")
    
    expected_files = [
        "output_test/1_raw_llm_extraction.json",
        "output_test/2_aggregated_counts.csv",
        "output_test/3_filtered_top_terms.csv",
        "output_test/4_nld_generated_definitions.csv",
        "output_test/5_categorized_ontology.csv"
    ]
    
    all_passed = True
    for f in expected_files:
        if os.path.exists(f):
            # Optional: Check size > 0
            if os.path.getsize(f) > 0:
                print(f"  [OK] Found {f}")
            else:
                print(f"  [WARN] Found {f} but it is empty")
        else:
            print(f"  [FAIL] Missing {f}")
            all_passed = False
            
    if all_passed:
        print("\n🎉 E2E Test Passed Successfully!")
    else:
        print("\n❌ E2E Test Failed: Missing artifacts.")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
