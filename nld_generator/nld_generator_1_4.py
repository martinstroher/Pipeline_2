import os
import time

import google.generativeai as genai
import pandas as pd

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    print("Gemini API Key configured successfully from environment variables.")
except KeyError:
    print("ERROR: The GEMINI_API_KEY environment variable was not found.")
    print("Please set it.")
    exit()
except Exception as e:
    print(f"ERROR configuring Gemini API: {e}")
    exit()

MODEL_NAME = os.environ.get("LLM_MODEL_NAME") # Use get with default
MODEL_TEMPERATURE = float(os.environ.get("LLM_MODEL_TEMPERATURE", 0.0))
INPUT_FILE = os.environ.get("FILTERED_TERMS_OUTPUT")
OUTPUT_FILE = os.environ.get("CONSOLIDATED_LLM_RESULTS_WITH_NLDS")
OUTPUT_FAILURE_FILE = os.environ.get("OUTPUT_FAILURE_FILE")

generation_config = genai.GenerationConfig(
    temperature=MODEL_TEMPERATURE,
)

def run_nld_generation():
    def load_terms_from_aggregator_csv(filepath):
        """Loads terms from the aggregator output CSV file."""
        if not os.path.exists(filepath):
            print(f"ERROR: The file '{filepath}' was not found.")
            return None
        try:
            df = pd.read_csv(filepath, encoding='utf-8', delimiter=',', header=0, usecols=['Readable_Term'])
            print(f"Success! {len(df)} terms loaded from '{filepath}'.")
            return df
        except ValueError as e:
            print(f"ERROR reading CSV: Column 'Readable_Term' likely not found in '{filepath}'. {e}")
            return None
        except Exception as e:
            print(f"ERROR reading the CSV file '{filepath}': {e}")
            return None


    system_instruction_definicao = "You are a senior geoscientist and ontology engineer. Your expertise is in oil and gas exploration geology, with a specific focus on the carbonate reservoirs of the Brazilian Pre-Salt."
    prompt_template_definicao = """Generate a concise and precise Natural Language Definition (NLD) in Portuguese for the provided geological term. The term may be in Portuguese or English.
    
    Mandatory Instructions:
    1. The definition must strictly follow the Aristotelian structure "X is a Y that Z". For example, "An amount of rock is a solid consolidated earth material that is constituted by an aggregate of particles made of mineral matter or material of biological origin".
    2. Base the definition on your knowledge of Brazilian Pre-Salt geology and petroleum systems.
    3. The definition should be technical yet clear, and a maximum of three sentences.
    4. Your response must contain only the generated NLD, without any extra text.
    
    Term to be defined: "{term}"
    """


    df_termos = load_terms_from_aggregator_csv(INPUT_FILE)

    if df_termos is not None:
        results = []
        terms_for_review = []

        model_definicao = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction_definicao,
                                                generation_config=generation_config)

        total_terms = len(df_termos)
        for index, row in df_termos.iterrows():
            # Get the term directly
            term = row['Readable_Term']

            print(f"Processing term {index + 1}/{total_terms}: '{term}'...")

            try:
                response_definicao = model_definicao.generate_content(
                    prompt_template_definicao.format(term=term)) # Pass term directly
                nld_generated = response_definicao.text.strip()

                print(f"  -> Definition generated successfully.")
                results.append({'Term': term, 'NLD': nld_generated}) # Save Term and NLD

                time.sleep(1)

            except Exception as e:
                print(f"  -> ERROR processing term '{term}': {e}")
                terms_for_review.append({'Term': term, 'Error': str(e)})

        print("\nProcessing complete. Saving results...")

        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df_results = pd.DataFrame(results)
        df_results.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"{len(df_results)} definitions saved to '{OUTPUT_FILE}'")

        if terms_for_review:
            failure_output_dir = os.path.dirname(OUTPUT_FAILURE_FILE)
            if failure_output_dir and not os.path.exists(failure_output_dir):
                 os.makedirs(failure_output_dir)

            df_review = pd.DataFrame(terms_for_review)
            df_review.to_csv(OUTPUT_FAILURE_FILE, index=False, encoding='utf-8-sig')
            print(f"{len(df_review)} terms marked for manual review saved to '{OUTPUT_FAILURE_FILE}'")