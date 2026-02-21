import os
import pandas as pd
import time
from src.utils.rag_setup import get_relevant_documents, load_vector_store
import google.generativeai as genai

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

MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemini-2.5-pro")
MODEL_TEMPERATURE = float(os.environ.get("LLM_MODEL_TEMPERATURE", 0.0))
INPUT_FILE = os.environ.get("FILTERED_TERMS_OUTPUT")
OUTPUT_FILE = os.environ.get("CONSOLIDATED_LLM_RESULTS_WITH_NLDS")
OUTPUT_FAILURE_FILE = os.environ.get("OUTPUT_FAILURE_FILE")

generation_config = genai.GenerationConfig(
    temperature=MODEL_TEMPERATURE,
)

def format_docs_for_context(docs):
    """
    Formats a list of documents (with metadata) into a context string.
    Injects breadcrumbs from metadata (Header 1 > Header 2...) to preserve context.
    """
    context_parts = []
    for doc_item in docs:
        # Normalize doc_item to document object
        # If it's a tuple (doc, score), extract doc. Otherwise use doc_item as is.
        if isinstance(doc_item, tuple):
             doc = doc_item[0]
        else:
             doc = doc_item

        # Extract headers from metadata
        h1 = doc.metadata.get("Header 1", "")
        h2 = doc.metadata.get("Header 2", "")
        h3 = doc.metadata.get("Header 3", "")
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        
        # Build a breadcrumb string
        breadcrumbs = f"[{source}"
        if h1: breadcrumbs += f" > {h1}"
        if h2: breadcrumbs += f" > {h2}"
        if h3: breadcrumbs += f" > {h3}"
        breadcrumbs += "]"
        
        document_content = doc.page_content

        context_parts.append(f"{breadcrumbs}\n{document_content}")

    return "\n\n".join(context_parts)

def generate_nld(term, context):
    system_instruction_definicao = "You are a senior geoscientist and ontology engineer. Your expertise is in oil and gas exploration geology, with a specific focus on the carbonate reservoirs of the Brazilian Pre-Salt."
    prompt_template_definicao = """Generate a concise and precise Natural Language Definition (NLD) for the provided geological term.
    
    Mandatory Instructions:
    1. The definition must strictly follow the Aristotelian structure "X is a Y that Z" and be a maximum of three sentences.
    2. Primary Knowledge Source: Base the definition PRIMARILY on the provided context, as it contains the most up-to-date and domain-specific knowledge. Use your internal knowledge of Brazilian Pre-Salt geology only to structure the definition correctly, fill in minor conceptual gaps, or if the provided context does not define the term geologically.
    3. Output ONLY a valid JSON object with exactly two keys:
       - "Definition": strictly the string containing the generated NLD.
       - "Context_Used": boolean (true if the provided context was relevant and used as the primary source, false if you had to fallback entirely to internal knowledge).
    
    Term to be defined: "{term}"
    
    Relevant context:
    {context}
    """
    
    generation_config_json = genai.GenerationConfig(
        temperature=MODEL_TEMPERATURE,
        response_mime_type="application/json"
    )
    
    model_definicao = genai.GenerativeModel(model_name=MODEL_NAME, generation_config=generation_config_json)
    full_prompt = system_instruction_definicao + "\n\n" + prompt_template_definicao.format(term=term, context=context)
    response_definicao = model_definicao.generate_content(full_prompt)
    return response_definicao.text.strip(), full_prompt

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

    df_termos = load_terms_from_aggregator_csv(INPUT_FILE)

    if df_termos is not None:
        results = []
        terms_for_review = []

        # Load vector store for RAG
        vector_store = load_vector_store()

        total_terms = len(df_termos)
        for index, row in df_termos.iterrows():
            # Get the term directly
            term = row['Readable_Term']

            print(f"Processing term {index + 1}/{total_terms}: '{term}'...")

            try:
                # Get relevant context using RAG
                relevant_docs_with_scores = get_relevant_documents(f"What is the definition of {term}?", vector_store)
                
                # Format context using the shared helper function
                context = format_docs_for_context(relevant_docs_with_scores)
                
                nld_json_str, _ = generate_nld(term, context)
                
                import json
                try:
                    nld_data = json.loads(nld_json_str)
                    nld_generated = nld_data.get("Definition", "")
                    context_used = nld_data.get("Context_Used", True)
                except json.JSONDecodeError:
                    print(f"  -> ERROR: Failed to parse JSON response. Raw: {nld_json_str}")
                    nld_generated = nld_json_str
                    context_used = "Error Parsing JSON"

                print(f"  -> Definition generated successfully. Context Used: {context_used}")
                results.append({'Term': term, 'NLD': nld_generated, 'Context_Used': context_used, 'Context': context})

                time.sleep(float(os.environ.get("NLD_SLEEP_SECONDS", 60)))

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
