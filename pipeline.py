from llm_term_extractor.llm_term_extractor_1_1 import run_llm_term_extraction
from term_aggregators import term_aggregator_for_llm_output_1_2
from filter_top_terms.filter_top_terms_1_3 import filter_top_terms
from nld_generator.nld_generator_1_4 import run_nld_generation
from term_categorizer.term_categorizer_1_5 import run_term_categorization

def main():
    run_llm_term_extraction()
    term_aggregator_for_llm_output_1_2.run_term_aggregation()
    filter_top_terms()
    run_nld_generation()
    run_term_categorization()

if __name__ == "__main__":
    main()
