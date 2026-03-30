import json
import os
import pandas as pd
import time
from tqdm import tqdm
from src.utils.rag_setup import get_relevant_documents, load_vector_store
from src.utils.gemini_client import get_client, generate
from src.utils import log

# Module-level state (configured lazily inside functions, not at import time)
_genai_configured = False
_MODEL_NAME = None
_MODEL_TEMPERATURE = None


def _ensure_genai_configured():
    """Configure Gemini client once per process. Raises RuntimeError on failure."""
    global _genai_configured, _MODEL_NAME, _MODEL_TEMPERATURE
    if _genai_configured:
        return
    get_client()  # triggers client init + prints confirmation
    _MODEL_NAME = os.environ.get("LLM_GENERATION_MODEL", "gemini-2.5-pro")
    _MODEL_TEMPERATURE = float(os.environ.get("LLM_GENERATION_TEMPERATURE", 0.0))
    _genai_configured = True


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
    _ensure_genai_configured()

    system_instruction_definicao = "You are a senior geoscientist and ontology engineer. Your expertise spans oil and gas exploration geology, structural geology, stratigraphy, and petroleum systems, with a specific focus on the carbonate reservoirs of the Brazilian Pre-Salt."

    prompt_template_definicao = """Generate a concise and precise Natural Language Definition (NLD) for the provided geological term.

    Mandatory Instructions:
    1. The FIRST sentence MUST follow the Aristotelian pattern "X is a Y that Z", where Y is the proximate genus and Z is the differentia. Up to two additional sentences may provide domain-specific elaboration. Minimum two sentences total.
    2. The definition MUST be written in English.
    3. If a term is polysemous, define the sense most relevant to Pre-Salt petroleum geology.
    4. Primary Knowledge Source: Base the definition PRIMARILY on the provided context, as it contains the most up-to-date and domain-specific knowledge. Use your internal knowledge of Brazilian Pre-Salt geology only to structure the definition correctly, fill in minor conceptual gaps, or if the provided context does not define the term geologically. If the context section contains no geologically relevant information about the term (including when it reads 'No additional context available.'), rely entirely on your domain expertise and set Context_Used to false.
    5. Output ONLY a valid JSON object with exactly two keys:
       - "Definition": strictly the string containing the generated NLD.
       - "Context_Used": boolean (true if the provided context was relevant and used as the primary source, false if you had to fallback entirely to internal knowledge).

    **EXAMPLES:**
    Term: "Grainstone"
    Context: [paper.md > Carbonate Classification] Grainstones are grain-supported carbonate rocks lacking mud matrix...
    Output: {{"Definition": "Grainstone is a grain-supported sedimentary carbonate rock that is characterized by the absence of micrite matrix, with grains typically consisting of allochems such as bivalves, ostracods, gastropods, ooids, or intraclasts.", "Context_Used": true}}

    Term: "Coquina"
    Context: [paper.md > Reservoir Facies] Coquinas from the Santos Basin are bioclastic carbonates composed predominantly of bivalve shells...
    Output: {{"Definition": "Coquina is a sedimentary rock that is primarily composed of accumulated mollusk shells (bivalves and gastropods) and their fragments, deposited in brackish to saline lacustrine environments, commonly occurring as fragmented shell beds and serving as a significant reservoir rock type.", "Context_Used": true}}

    Term: "Dolomitization"
    Context: No additional context available.
    Output: {{"Definition": "Dolomitization is a diagenetic process that replaces calcium carbonate minerals with dolomite through the substitution of calcium ions by magnesium ions from Mg-rich fluids, commonly occurring in burial or hydrothermal settings and frequently enhancing reservoir porosity and permeability.", "Context_Used": false}}

    ---
    Term to be defined: "{term}"

    Relevant context:
    {context}
    """

    full_prompt = prompt_template_definicao.format(term=term, context=context)
    response_text = generate(
        full_prompt,
        model=_MODEL_NAME,
        system_instruction=system_instruction_definicao,
        temperature=_MODEL_TEMPERATURE,
        response_mime_type="application/json",
    )
    return response_text.strip(), full_prompt

def run_nld_generation(vector_store=None, bm25_retriever=None):
    _ensure_genai_configured()

    INPUT_FILE = os.environ.get("FILTERED_TERMS_OUTPUT")
    OUTPUT_FILE = os.environ.get("CONSOLIDATED_LLM_RESULTS_WITH_NLDS")
    OUTPUT_FAILURE_FILE = os.environ.get("OUTPUT_FAILURE_FILE")

    def load_terms_from_aggregator_csv(filepath):
        """Loads terms from the aggregator output CSV file."""
        if not os.path.exists(filepath):
            log.error(f"File '{filepath}' not found.")
            return None
        try:
            df = pd.read_csv(filepath, encoding='utf-8', delimiter=',', header=0, usecols=['Readable_Term'])
            log.info(f"{len(df)} terms loaded from '{filepath}'.")
            return df
        except ValueError as e:
            log.error(f"CSV column error in '{filepath}': {e}")
            return None
        except Exception as e:
            log.error(f"Reading CSV '{filepath}': {e}")
            return None

    df_termos = load_terms_from_aggregator_csv(INPUT_FILE)

    if df_termos is not None:
        # Load previously checkpointed results if they exist
        completed_terms = set()
        results = []
        if os.path.exists(OUTPUT_FILE):
            try:
                df_existing = pd.read_csv(OUTPUT_FILE, encoding='utf-8-sig')
                completed_terms = set(df_existing['Term'].tolist())
                results = df_existing.to_dict('records')
                log.info(f"Resuming: {len(completed_terms)} terms already processed.")
            except Exception:
                pass

        terms_for_review = []

        # Load vector store for RAG if not passed
        if vector_store is None:
            log.warn("No vector_store passed, loading from disk (dense-only).")
            vector_store = load_vector_store()

        if bm25_retriever is not None:
            log.info("BM25 hybrid retrieval is ACTIVE.")
        else:
            log.warn("BM25 not available, using dense-only retrieval.")

        total_terms = len(df_termos)
        pbar = tqdm(total=total_terms, desc="Generating NLDs")
        for index, row in df_termos.iterrows():
            # Get the term directly
            term = row['Readable_Term']
            pbar.set_postfix_str(term[:30])

            # Skip already checkpointed terms
            if term in completed_terms:
                pbar.update(1)
                continue

            try:
                # Get relevant context using RAG (with BM25 if available)
                # Use term-only query (no question-form noise for BM25)
                relevant_docs_with_scores = get_relevant_documents(
                    term,
                    vector_store,
                    bm25_retriever=bm25_retriever
                )

                # Format context using the shared helper function
                context = format_docs_for_context(relevant_docs_with_scores)

                nld_json_str, _ = generate_nld(term, context)

                try:
                    nld_data = json.loads(nld_json_str)
                    nld_generated = nld_data.get("Definition", "")
                    context_used = nld_data.get("Context_Used", True)
                except json.JSONDecodeError:
                    tqdm.write("")
                    log.warn(f"JSON parse failed for '{term}': {nld_json_str[:80]}")
                    nld_generated = nld_json_str
                    context_used = "Error Parsing JSON"

                result_row = {'Term': term, 'NLD': nld_generated, 'Context_Used': context_used, 'Context': context}
                results.append(result_row)

                # Checkpoint: atomic append to CSV
                single_row_df = pd.DataFrame([result_row])
                single_row_df.to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE) or len(results) == 1, index=False, encoding='utf-8-sig')

                time.sleep(float(os.environ.get("NLD_SLEEP_SECONDS", 4)))

            except Exception as e:
                tqdm.write("")
                log.error(f"Term '{term}': {e}")
                terms_for_review.append({'Term': term, 'Error': str(e)})

            pbar.update(1)
        pbar.close()

        # Write final consolidated output (overwrites checkpoint file with clean version)
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df_results = pd.DataFrame(results)
        df_results.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        log.success(f"{len(df_results)} definitions saved to '{OUTPUT_FILE}'")

        if terms_for_review:
            failure_output_dir = os.path.dirname(OUTPUT_FAILURE_FILE)
            if failure_output_dir and not os.path.exists(failure_output_dir):
                 os.makedirs(failure_output_dir)

            df_review = pd.DataFrame(terms_for_review)
            df_review.to_csv(OUTPUT_FAILURE_FILE, index=False, encoding='utf-8-sig')
            log.warn(f"{len(df_review)} terms need manual review -> '{OUTPUT_FAILURE_FILE}'")
