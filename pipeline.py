from llm_term_extractor.llm_term_extractor_1_1 import run_llm_term_extraction
from term_aggregators import term_aggregator_for_llm_output_1_2
from filter_top_terms.filter_top_terms_1_3 import filter_top_terms
from nld_generator.nld_generator_1_4 import run_nld_generation
from term_categorizer.term_categorizer_1_5 import run_term_categorization
from rag_setup import setup_rag, DOCS_DIR
import text_converted

def main():
    # Run Robust Text Extraction first
    print(f"Running text extraction on {DOCS_DIR}...")
    text_converted.process_folder(DOCS_DIR)

    # Set up RAG system
    setup_rag()

    # Run existing pipeline steps
    run_llm_term_extraction()
    term_aggregator_for_llm_output_1_2.run_term_aggregation()
    filter_top_terms()
    
    # Run NLD generation with RAG
    run_nld_generation()
    
    # Run term categorization
    run_term_categorization()

if __name__ == "__main__":
    main()
