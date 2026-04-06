import argparse
import os
import warnings

# Suppress noisy library warnings before any src/ imports (they trigger langchain)
warnings.filterwarnings("ignore", message=".*Pydantic.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*HuggingFaceEmbeddings.*was deprecated.*")
warnings.filterwarnings("ignore", message=r".*class `Chroma`.*was deprecated.*")

from dotenv import load_dotenv

load_dotenv()

from src.utils import log
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
        "--layer2-analysis",
        nargs="+",
        default=None,
        metavar="WORKBOOK",
        help="Run Layer 2 analysis on completed expert workbooks (e.g., --layer2-analysis expert1.xlsx expert2.xlsx --layer2-key blinding_key_42.csv)",
    )
    parser.add_argument(
        "--layer2-key",
        type=str,
        default=None,
        help="Path to blinding key CSV (required with --layer2-analysis)",
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
        "--verify",
        type=str,
        default=None,
        help="Verify an OWL .ttl file (e.g., output/7_ontology.ttl)",
    )
    parser.add_argument(
        "--skip-oops",
        action="store_true",
        help="Skip OOPS! API call during verification (offline mode)",
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
        from src.evaluation.layer1_analysis import run_layer1_analysis
        from src.evaluation.expert_eval_generator import generate_expert_evaluation
        conds = [c.strip().upper() for c in args.conditions.split(",")]
        run_ablation(conditions=conds)
        run_layer1_analysis()
        generate_expert_evaluation()
        return

    # --- Layer 1 analysis ---
    if args.analysis:
        from src.evaluation.layer1_analysis import run_layer1_analysis
        run_layer1_analysis()
        return

    # --- Expert evaluation spreadsheet ---
    if args.expert_eval:
        from src.evaluation.expert_eval_generator import generate_expert_evaluation
        generate_expert_evaluation()
        return

    # --- Layer 2 analysis ---
    if args.layer2_analysis:
        from src.evaluation.expert_eval_analyzer import run_layer2_analysis
        if not args.layer2_key:
            parser.error("--layer2-key is required with --layer2-analysis")
        run_layer2_analysis(args.layer2_analysis, args.layer2_key)
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

    # --- Ontology verification ---
    if args.verify:
        from src.modules.ontology_verifier import run_ontology_verification
        run_ontology_verification(args.verify, skip_oops=args.skip_oops)
        return

    # --- Standard pipeline ---
    if not args.skip_pdf:
        log.banner(0, "PDF Text Extraction")
        text_converted.process_folder(DOCS_DIR)

    # Set up RAG system
    log.banner("R", "RAG Setup")
    vector_store, bm25 = setup_rag()

    if not args.skip_extraction:
        log.banner(1, "Term Extraction")
        run_llm_term_extraction()

        log.banner(2, "Term Aggregation")
        term_aggregator.run_term_aggregation()

        log.banner(3, "Term Filtering")
        filter_top_terms()

    log.banner(4, "NLD Generation")
    run_nld_generation(vector_store=vector_store, bm25_retriever=bm25)

    log.banner(5, "Term Categorization")
    run_term_categorization()

    log.banner(6, "Taxonomy Builder")
    from src.modules.taxonomy_builder import run_taxonomy_builder
    cat_csv = os.environ["CATEGORIZED_LLM_TERMS"]
    run_taxonomy_builder(cat_csv)

    log.banner(7, "OWL Export")
    from src.modules.owl_exporter import run_owl_export
    tax_csv = (
        os.path.splitext(cat_csv)[0]
        .replace("5_categorized_ontology", "6_taxonomy") + ".csv"
    )
    owl_path = run_owl_export(tax_csv)

    # Step 7b: Ontology Verification
    from src.modules.ontology_verifier import run_ontology_verification
    run_ontology_verification(owl_path, skip_oops=args.skip_oops)

    log.success("\nPipeline complete.")


if __name__ == "__main__":
    main()
