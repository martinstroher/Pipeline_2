from src.modules.term_extractor import run_llm_term_extraction
from src.modules import term_aggregator
from src.modules.term_filter import filter_top_terms
from src.modules.nld_generator import run_nld_generation
from src.modules.term_categorizer import run_term_categorization
from src.utils.rag_setup import setup_rag, DOCS_DIR
from src.utils import pdf_processor as text_converted

def main():
    # Run Robust Text Extraction first
    print(f"Running text extraction on {DOCS_DIR}...")
    text_converted.process_folder(DOCS_DIR)

    # Set up RAG system
    setup_rag()

    # Run existing pipeline steps
    run_llm_term_extraction()
    term_aggregator.run_term_aggregation()
    filter_top_terms()
    
    # Run NLD generation with RAG
    run_nld_generation()
    
    # Run term categorization
    run_term_categorization()

if __name__ == "__main__":
    main()
