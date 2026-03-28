import argparse
import os
from dotenv import load_dotenv

load_dotenv()

from src.modules.term_extractor import run_llm_term_extraction
from src.modules import term_aggregator
from src.modules.term_filter import filter_top_terms
from src.modules.nld_generator import run_nld_generation
from src.modules.term_categorizer import run_term_categorization
from src.utils.rag_setup import setup_rag, DOCS_DIR
from src.utils import pdf_processor as text_converted


def main():
    parser = argparse.ArgumentParser(description="PreSaltOntoLearn Pipeline")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run 4-condition ablation study instead of standard pipeline",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="A,B,C,D",
        help="Comma-separated ablation conditions (default: A,B,C,D)",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Run Layer 1 statistical analysis on ablation results",
    )
    parser.add_argument(
        "--expert-eval",
        action="store_true",
        help="Generate expert evaluation spreadsheet from ablation results",
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        default=None,
        help="Build taxonomy from a categorized CSV (e.g., output/ablation/cat_A.csv)",
    )
    parser.add_argument(
        "--owl",
        type=str,
        default=None,
        help="Export OWL from taxonomy CSV (e.g., output/ablation/6_taxonomy_A.csv)",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip Steps 1-3 (use existing filtered terms)",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip PDF->Markdown conversion (use existing .md files)",
    )
    args = parser.parse_args()

    # --- Ablation mode ---
    if args.ablation:
        from src.evaluation.ablation_study import run_ablation
        conds = [c.strip().upper() for c in args.conditions.split(",")]
        run_ablation(conditions=conds)
        return

    # --- Layer 1 analysis ---
    if args.analysis:
        from src.evaluation.layer1_analysis import run_layer1_analysis
        run_layer1_analysis()
        return

    # --- Expert evaluation spreadsheet ---
    if args.expert_eval:
        from src.evaluation.expert_eval_generator import generate_expert_spreadsheet
        generate_expert_spreadsheet()
        return

    # --- Taxonomy builder ---
    if args.taxonomy:
        from src.modules.taxonomy_builder import run_taxonomy_builder
        run_taxonomy_builder(args.taxonomy)
        return

    # --- OWL export ---
    if args.owl:
        from src.modules.owl_exporter import run_owl_export
        run_owl_export(args.owl)
        return

    # --- Standard pipeline ---
    if not args.skip_pdf:
        print(f"Running text extraction on {DOCS_DIR}...")
        text_converted.process_folder(DOCS_DIR)

    # Set up RAG system
    vector_store, bm25 = setup_rag()

    if not args.skip_extraction:
        # Run existing pipeline steps (Steps 1-3)
        run_llm_term_extraction()
        term_aggregator.run_term_aggregation()
        filter_top_terms()

    # Step 4: NLD generation with RAG
    run_nld_generation(vector_store=vector_store, bm25_retriever=bm25)

    # Step 5: Term categorization
    run_term_categorization()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
