import pandas as pd
import google.generativeai as genai
import os
import json
import time

def run_llm_term_extraction():
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        print("Gemini API Key configured successfully.")
    except Exception as e:
        print(f"ERROR configuring Gemini API: {e}")
        exit()

    LLM_MODEL_NAME = os.environ["LLM_MODEL_NAME"]
    LLM_MODEL_TEMPERATURE=float(os.environ["LLM_MODEL_TEMPERATURE"])
    PAPER_END_DELIMITER = os.environ["PAPER_END_DELIMITER"]
    LLM_INPUT_FILE=os.environ["LLM_INPUT_FILE"]
    LLM_OUTPUT_FILE=os.environ["LLM_OUTPUT_FILE"]

    generation_config = genai.GenerationConfig(
        temperature=LLM_MODEL_TEMPERATURE,
        response_mime_type="application/json"
    )


    def load_text_from_file(filepath):
        if not os.path.exists(filepath):
            print(f"ERROR: The file '{filepath}' was not found.")
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"Success! Loaded {len(content)} characters from '{filepath}'.")
            return content
        except Exception as e:
            print(f"ERROR reading the text file: {e}")
            return None


    system_instruction = "You are an expert geologist and ontology engineer specializing in South Atlantic Pre-Salt petroleum systems. Your task is to extract core geological concepts from technical texts suitable for building a domain ontology. This ontology's primary purpose is to assist geologists in describing and comparing analog reservoirs geological settings."


    prompt_template = """**METHODOLOGY**
    1.  **Identify Conceptual Entities:** Identify all terms or phrases representing geological concepts. Focus on identifying *types* or *classes* of entities relevant to petroleum geology and pre-salt context.
    2.  **Normalize Terms:** Return all extracted concepts translated to English and, where appropriate, in their singular, base form (e.g., "carbonates" -> "Carbonate", "faults" -> "Fault"). Use title case for concepts.
    3.  **Strict Filtering:** You MUST exclude:
        * Specific, non-conceptual proper nouns (e.g., individual well names like 'Well 1-BRSA-123', specific field names unless used generically, basin names like 'Santos Basin', author names, company names).
        * Units of measure, numerical values, and codes (e.g., 'mD', 'API', '10%', 'SiO2').
    4.  **Focus:** Prioritize terms that represent reusable classes within an ontology framework. Do not rank or limit the number extracted from this snippet.
    
    **OUTPUT FORMAT:**
    Your response MUST BE a valid JSON array of unique strings.
    
    **Example of output array:**
    ["Microbial Carbonate", "Diagenesis", "Source Rock", "Structural Trap", "Porosity", "Lacustrine Environment", "Aptian"]
    
    ---
    **TEXT SNIPPET TO ANALYZE:**
    {chunk_text}
    """


    print("Loading the text corpus...")
    full_text = load_text_from_file(LLM_INPUT_FILE)

    if full_text:
        all_extracted_terms = []

        model = genai.GenerativeModel(
            model_name=LLM_MODEL_NAME,
            system_instruction=system_instruction,
            generation_config=generation_config
        )

        papers = full_text.split(PAPER_END_DELIMITER)
        papers = [paper.strip() for paper in papers if paper.strip()]
        num_papers = len(papers)
        print(f"\nText has been split into {num_papers} separate papers.")

        for i, paper_text in enumerate(papers):
            paper_num = i + 1
            print(f"Processing paper {paper_num}/{num_papers}...")

            try:
                final_prompt = prompt_template.format(chunk_text=paper_text)

                response = model.generate_content(final_prompt)

                if not response.parts:
                    print(
                        f"  -> ERROR: API call for paper {paper_num} was blocked. Reason: {response.prompt_feedback.block_reason}")
                    continue

                terms_from_paper = json.loads(response.text)
                all_extracted_terms.extend(terms_from_paper)
                print(f"  -> Extracted {len(terms_from_paper)} terms from this paper.")

            except Exception as e:
                print(f"  -> An error occurred processing paper {paper_num}: {e}")

        print("\nExtraction complete. Saving all extracted terms...")

        df_raw_results = pd.DataFrame(all_extracted_terms, columns=['Entity'])
        df_raw_results.to_csv(LLM_OUTPUT_FILE, index=False, encoding='utf-8-sig')

        print(
            f"\nSuccess! A total of {len(all_extracted_terms)} raw terms were extracted and saved to '{LLM_OUTPUT_FILE}'.")