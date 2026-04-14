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
from src.utils.pdf_processor import process_folder as convert_pdfs


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
        "--relations",
        type=str,
        default=None,
        help="Extract relations from categorized CSV (e.g., output/5_categorized_ontology.csv)",
    )
    parser.add_argument(
        "--relation-analysis",
        type=str,
        default=None,
        nargs="?",
        const="output/6b_relations.csv",
        help="Run descriptive stats + precision sample on 6b_relations.csv",
    )
    parser.add_argument(
        "--skip-relations",
        action="store_true",
        help="Skip Step 6b relation extraction in the standard pipeline",
    )
    parser.add_argument(
        "--skip-oops",
        action="store_true",
        help="Skip OOPS! API call during verification (offline mode)",
    )
    parser.add_argument(
        "--skip-reasoner",
        action="store_true",
        help="Skip HermiT reasoner check during verification",
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
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clean start: remove all output files, ChromaDB, and .md inputs. Forces full rebuild.",
    )
    parser.add_argument(
        "--stop-after",
        type=str,
        default=None,
        choices=["0", "R", "1", "2", "3", "4", "5", "6", "6b", "7", "7b"],
        help="Stop pipeline after this step (e.g., --stop-after 3 to run only Steps 0-3)",
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
        run_ontology_verification(args.verify, skip_oops=args.skip_oops, skip_reasoner=args.skip_reasoner)
        return

    # --- Standalone relation extraction ---
    if args.relations:
        from src.modules.relation_extractor import run_relation_extraction
        run_relation_extraction(args.relations)
        return

    # --- Relation analysis ---
    if args.relation_analysis:
        from src.evaluation.relation_analysis import run_relation_analysis
        run_relation_analysis(args.relation_analysis)
        return

    # --- Fresh start: clean all outputs and caches ---
    if args.fresh:
        import glob
        import shutil
        log.banner("X", "Fresh Start — Cleaning Previous Run")
        # Remove output files (Steps 1-7b)
        output_dir = os.path.dirname(os.environ.get("LLM_OUTPUT_FILE", "output/1_raw_llm_extraction.json")) or "output"
        output_patterns = [
            os.path.join(output_dir, "1_raw_llm_extraction.json"),
            os.path.join(output_dir, "2_aggregated_counts.csv"),
            os.path.join(output_dir, "3_filtered_top_terms.csv"),
            os.path.join(output_dir, "4_nld_generated_definitions.csv"),
            os.path.join(output_dir, "4_nld_generation_failures.csv"),
            os.path.join(output_dir, "5_categorized_ontology.csv"),
            os.path.join(output_dir, "6_taxonomy.csv"),
            os.path.join(output_dir, "6b_relations.csv"),
            os.path.join(output_dir, "7_ontology.ttl"),
            os.path.join(output_dir, "7b_verification_report.json"),
        ]
        removed = 0
        for f in output_patterns:
            if os.path.exists(f):
                os.remove(f)
                removed += 1
        log.info(f"Removed {removed} output files from {output_dir}/")
        # Remove ChromaDB caches
        for d in glob.glob("chroma_db_*"):
            if os.path.isdir(d):
                shutil.rmtree(d)
                log.info(f"Removed ChromaDB cache: {d}")
        # Remove converted .md files from inputs (keep .pdf)
        md_files = glob.glob(os.path.join(DOCS_DIR, "*.md"))
        for f in md_files:
            os.remove(f)
        if md_files:
            log.info(f"Removed {len(md_files)} .md files from {DOCS_DIR}")
        log.success("Clean start complete — all caches and outputs removed.")

    # --- Standard pipeline ---
    _stop = args.stop_after

    if not args.skip_pdf:
        log.banner(0, "PDF Text Extraction")
        convert_pdfs(DOCS_DIR)
        if _stop == "0":
            log.success("\nStopped after Step 0 (--stop-after 0)."); return

    # Set up RAG system (force rebuild after --fresh)
    log.banner("R", "RAG Setup")
    vector_store, bm25 = setup_rag(force_rebuild=args.fresh)
    if _stop == "R":
        log.success("\nStopped after Step R (--stop-after R)."); return

    if not args.skip_extraction:
        log.banner(1, "Term Extraction")
        run_llm_term_extraction()
        if _stop == "1":
            log.success("\nStopped after Step 1 (--stop-after 1)."); return

        log.banner(2, "Term Aggregation")
        term_aggregator.run_term_aggregation()
        if _stop == "2":
            log.success("\nStopped after Step 2 (--stop-after 2)."); return

        log.banner(3, "Term Filtering")
        filter_top_terms()
        if _stop == "3":
            log.success("\nStopped after Step 3 (--stop-after 3)."); return

    log.banner(4, "NLD Generation")
    run_nld_generation(vector_store=vector_store, bm25_retriever=bm25)
    if _stop == "4":
        log.success("\nStopped after Step 4 (--stop-after 4)."); return

    log.banner(5, "Term Categorization")
    run_term_categorization()
    if _stop == "5":
        log.success("\nStopped after Step 5 (--stop-after 5)."); return

    log.banner(6, "Taxonomy Builder")
    from src.modules.taxonomy_builder import run_taxonomy_builder
    cat_csv = os.environ["CATEGORIZED_LLM_TERMS"]
    run_taxonomy_builder(cat_csv)
    if _stop == "6":
        log.success("\nStopped after Step 6 (--stop-after 6)."); return

    # Step 6b: Relation Extraction
    relations_csv = None
    if not args.skip_relations:
        log.banner("6b", "Relation Extraction")
        from src.modules.relation_extractor import run_relation_extraction
        relations_csv = run_relation_extraction(cat_csv)
    else:
        log.info("Step 6b: Relation extraction skipped (--skip-relations)")
    if _stop == "6b":
        log.success("\nStopped after Step 6b (--stop-after 6b)."); return

    log.banner(7, "OWL Export")
    from src.modules.owl_exporter import run_owl_export
    tax_csv = (
        os.path.splitext(cat_csv)[0]
        .replace("5_categorized_ontology", "6_taxonomy") + ".csv"
    )
    owl_path = run_owl_export(tax_csv, relations_csv=relations_csv)
    if _stop == "7":
        log.success("\nStopped after Step 7 (--stop-after 7)."); return

    # Step 7b: Ontology Verification
    from src.modules.ontology_verifier import run_ontology_verification
    run_ontology_verification(owl_path, skip_oops=args.skip_oops, skip_reasoner=args.skip_reasoner)

    log.success("\nPipeline complete.")


if __name__ == "__main__":
    main()
