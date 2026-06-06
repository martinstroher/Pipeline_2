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
from src.modules.extract.term_extractor import run_llm_term_extraction
from src.modules.extract import term_aggregator
from src.modules.extract.term_filter import filter_top_terms
from src.modules.define.nld_generator import run_nld_generation
from src.modules.classify.category_assigner import run_term_categorization
from src.utils.rag_setup import setup_rag, DOCS_DIR
from src.utils.pdf_processor import process_folder as convert_pdfs


def _clean_outputs() -> None:
    """Remove all generated outputs, ChromaDB caches, and converted .md files.

    Triggered by --fresh. Forces a full rebuild on the next run.
    """
    import glob
    import shutil

    log.banner("X", "Fresh Start — Cleaning Previous Run")
    output_dir = os.path.dirname(
        os.environ.get("LLM_OUTPUT_FILE", "output/extract_raw.json")
    ) or "output"
    output_files = [
        "extract_raw.json",
        "extract_aggregated.csv",
        "extract_filtered.csv",
        "define_nld.csv",
        "define_failures.csv",
        "classify_categories.csv",
        "construct_taxonomy.csv",
        "construct_relations.csv",
        "emit_ontology.ttl",
        "emit_verification.json",
    ]
    removed = 0
    for name in output_files:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)
            removed += 1
    log.info(f"Removed {removed} output files from {output_dir}/")

    for d in glob.glob("chroma_db_*"):
        if os.path.isdir(d):
            shutil.rmtree(d)
            log.info(f"Removed ChromaDB cache: {d}")

    md_files = glob.glob(os.path.join(DOCS_DIR, "*.md"))
    for f in md_files:
        os.remove(f)
    if md_files:
        log.info(f"Removed {len(md_files)} .md files from {DOCS_DIR}")
    log.success("Clean start complete — all caches and outputs removed.")


def _check_stop(stop: str | None, step: str) -> bool:
    """If `stop` matches `step`, log success and return True so caller can return.

    Used to dedupe the repeating `if _stop == "X": log.success(...); return`
    pattern after every pipeline step.
    """
    if stop == step:
        log.success(f"\nStopped after Step {step} (--stop-after {step}).")
        return True
    return False


def _dispatch_subcommand(args, parser) -> bool:
    """Run a one-shot subcommand (--ablation, --analysis, --taxonomy, ...) and
    return True if one fired. Returns False so the caller runs the standard
    pipeline.
    """
    if args.ablation:
        from src.evaluation.ablation_study import run_ablation
        from src.evaluation.layer1_analysis import run_layer1_analysis
        from src.evaluation.expert_eval_generator import generate_expert_evaluation
        conds = [c.strip().upper() for c in args.conditions.split(",")]
        run_ablation(conditions=conds)
        run_layer1_analysis()
        generate_expert_evaluation()
        return True

    if args.analysis:
        from src.evaluation.layer1_analysis import run_layer1_analysis
        run_layer1_analysis()
        return True

    if args.expert_eval:
        if args.refine and args.threshold is not None:
            from src.evaluation.expert_eval_generator import generate_refined_evaluation
            t = args.threshold
            cq_matrix = os.path.join("output", "refined", "5b_cq_matrix.csv")
            tax_path = os.path.join("output", "refined", f"t{t}", "construct_taxonomy.csv")
            if not os.path.exists(cq_matrix):
                parser.error(f"CQ matrix not found: {cq_matrix}. Run --refine first.")
            if not os.path.exists(tax_path):
                parser.error(f"Taxonomy not found: {tax_path}. Run --refine with full pipeline first.")
            generate_refined_evaluation(cq_matrix, t, tax_path)
        else:
            from src.evaluation.expert_eval_generator import generate_expert_evaluation
            generate_expert_evaluation()
        return True

    if args.layer2_analysis:
        from src.evaluation.expert_eval_analyzer import run_layer2_analysis
        if not args.layer2_key:
            parser.error("--layer2-key is required with --layer2-analysis")
        run_layer2_analysis(args.layer2_analysis, args.layer2_key)
        return True

    if args.taxonomy:
        from src.modules.construct.taxonomy_builder import run_taxonomy_builder
        run_taxonomy_builder(args.taxonomy)
        return True

    if args.owl:
        from src.modules.emit.owl_exporter import run_owl_export
        run_owl_export(args.owl)
        return True

    if args.verify:
        from src.modules.emit.verifier import run_ontology_verification
        run_ontology_verification(args.verify, skip_oops=args.skip_oops, skip_reasoner=args.skip_reasoner)
        return True

    if args.relations:
        from src.modules.construct.relation_extractor import run_relation_extraction
        run_relation_extraction(args.relations)
        return True

    if args.relation_analysis:
        from src.evaluation.relation_analysis import run_relation_analysis
        run_relation_analysis(args.relation_analysis)
        return True

    return False


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser. Kept separate so main() stays focused on flow."""
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
        help="Export OWL from taxonomy CSV (e.g., output/ablation/construct_taxonomy_A.csv)",
    )
    parser.add_argument(
        "--verify",
        type=str,
        default=None,
        help="Verify an OWL .ttl file (e.g., output/emit_ontology.ttl)",
    )
    parser.add_argument(
        "--relations",
        type=str,
        default=None,
        help="Extract relations from categorized CSV (e.g., output/classify_categories.csv)",
    )
    parser.add_argument(
        "--relation-analysis",
        type=str,
        default=None,
        nargs="?",
        const="output/construct_relations.csv",
        help="Run descriptive stats + precision sample on construct_relations.csv",
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
        "--refine",
        action="store_true",
        help="Run CQ-driven refinement: Step 5b cleanup/scoring + multi-threshold generation (Steps 6-7b per threshold).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="Pick a single CQ threshold. With --refine: run Steps 6-7b only for this T. With --expert-eval --refine: generate evaluation workbook for this T.",
    )
    parser.add_argument(
        "--stop-after",
        type=str,
        default=None,
        choices=["0", "R", "1", "2", "3", "4", "5", "5b", "6", "6b", "6c", "7", "7b"],
        help="Stop pipeline after this step (e.g., --stop-after 3 to run only Steps 0-3)",
    )
    return parser


def _run_refinement_pipeline(args, _stop) -> None:
    """Run Step 5b (CQ refinement) then per-threshold Steps 6→6b→6c→6d→7→7b."""
    from src.modules.classify.cq_scorer import run_cq_refinement
    log.banner("5b", "CQ-Driven Refinement")
    cat_csv = os.environ["CATEGORIZED_LLM_TERMS"]
    threshold_paths = run_cq_refinement(cat_csv)
    if _check_stop(_stop, "5b"): return

    from src.modules.construct.taxonomy_builder import run_taxonomy_builder
    from src.modules.emit.owl_exporter import run_owl_export
    from src.modules.emit.verifier import run_ontology_verification
    import pandas as _pd

    # Specialization hints from synonym triage (may not exist)
    hints_csv = os.path.join("output", "refined", "5b_specialization_hints.csv")
    if not os.path.exists(hints_csv):
        hints_csv = None

    if args.threshold is not None:
        picked = args.threshold
        if picked not in threshold_paths:
            log.error(f"Threshold T={picked} not found in 5b output. Available: {sorted(threshold_paths.keys())}")
            return
        threshold_paths = {picked: threshold_paths[picked]}

    comparison_rows = []
    for t, t_cat_csv in sorted(threshold_paths.items()):
        t_dir = os.path.dirname(t_cat_csv)
        log.banner(f"T{t}-6", f"Taxonomy Builder (threshold ≥{t})")
        tax_csv = os.path.join(t_dir, "construct_taxonomy.csv")
        run_taxonomy_builder(t_cat_csv, output_path=tax_csv, hints_csv=hints_csv)

        rel_csv = None
        if not args.skip_relations:
            log.banner(f"T{t}-6b", f"Relation Extraction (threshold ≥{t})")
            from src.modules.construct.relation_extractor import run_relation_extraction
            rel_csv = os.path.join(t_dir, "construct_relations.csv")
            run_relation_extraction(t_cat_csv, output_path=rel_csv)

        log.banner(f"T{t}-6c", f"Ontology Critic (threshold ≥{t})")
        from src.modules.validate.ontology_critic import run_ontology_critic
        cleaned_tax = os.path.join(t_dir, "validate_critic_taxonomy.csv")
        cleaned_rel = os.path.join(t_dir, "validate_critic_relations.csv") if rel_csv else None
        run_ontology_critic(
            tax_csv,
            output_path=cleaned_tax,
            relations_csv=rel_csv,
            relations_output=cleaned_rel,
        )
        if _check_stop(_stop, "6c"): return

        reclass_tax = cleaned_tax  # fallback if no relations
        if cleaned_rel and os.path.exists(cleaned_rel):
            log.banner(f"T{t}-6d", f"Relation Reclassification (threshold ≥{t})")
            from src.modules.validate.relation_reclassifier import run_relation_reclassification
            reclass_tax = os.path.join(t_dir, "validate_reclassified_taxonomy.csv")
            run_relation_reclassification(
                cleaned_tax,
                cleaned_rel,
                output_path=reclass_tax,
            )
        if _check_stop(_stop, "6d"): return

        final_tax = reclass_tax if os.path.exists(reclass_tax) else cleaned_tax
        final_rel = cleaned_rel if (cleaned_rel and os.path.exists(cleaned_rel)) else rel_csv

        log.banner(f"T{t}-7", f"OWL Export (threshold ≥{t})")
        owl_path = os.path.join(t_dir, "emit_ontology.ttl")
        run_owl_export(final_tax, relations_csv=final_rel, output_path=owl_path)

        log.banner(f"T{t}-7b", f"Verification (threshold ≥{t})")
        report_path = os.path.join(t_dir, "emit_verification.json")
        report = run_ontology_verification(
            owl_path, output_path=report_path,
            skip_oops=args.skip_oops, skip_reasoner=args.skip_reasoner,
        )

        structure = report.get("layers", {}).get("structure", {})
        comparison_rows.append({
            "Threshold": f"T>={t}",
            "Terms": _pd.read_csv(t_cat_csv, encoding="utf-8-sig").shape[0],
            "Classes": structure.get("classes", 0),
            "Individuals": structure.get("individuals", 0),
            "Triples": structure.get("triples", 0),
            "HermiT": report.get("layers", {}).get("reasoner", {}).get("status", "SKIP"),
        })

    comp_path = os.path.join("output", "refined", "comparison.csv")
    _pd.DataFrame(comparison_rows).to_csv(comp_path, index=False, encoding="utf-8-sig")
    log.success(f"\nComparison report → '{comp_path}'")
    log.success("\nRefinement pipeline complete.")


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if _dispatch_subcommand(args, parser):
        return

    # --- Fresh start: clean all outputs and caches ---
    if args.fresh:
        _clean_outputs()

    # --- Standard pipeline ---
    _stop = args.stop_after

    if not args.skip_pdf:
        log.banner(0, "PDF Text Extraction")
        convert_pdfs(DOCS_DIR)
        if _check_stop(_stop, "0"): return

    # Set up RAG system (force rebuild after --fresh)
    # Skip RAG if --refine and categorized output already exists (5b doesn't need RAG)
    _cat_exists = os.path.exists(os.environ.get("CATEGORIZED_LLM_TERMS", "output/classify_categories.csv"))
    if args.refine and _cat_exists and args.skip_extraction:
        log.info("Skipping RAG setup (not needed for Step 5b with existing outputs)")
        vector_store, bm25 = None, None
    else:
        log.banner("R", "RAG Setup")
        vector_store, bm25 = setup_rag(force_rebuild=args.fresh)
        if _check_stop(_stop, "R"): return

    if not args.skip_extraction:
        log.banner(1, "Term Extraction")
        run_llm_term_extraction()
        if _check_stop(_stop, "1"): return

        log.banner(2, "Term Aggregation")
        term_aggregator.run_term_aggregation()
        if _check_stop(_stop, "2"): return

        log.banner(3, "Term Filtering")
        filter_top_terms()
        if _check_stop(_stop, "3"): return

    # When --refine is set and categorized output exists, skip Steps 4-5
    cat_csv_path = os.environ.get("CATEGORIZED_LLM_TERMS", "output/classify_categories.csv")
    if args.refine and os.path.exists(cat_csv_path) and _stop != "4" and _stop != "5":
        log.info(f"Skipping Steps 4-5: categorized output exists at '{cat_csv_path}'")
    else:
        log.banner(4, "NLD Generation")
        run_nld_generation(vector_store=vector_store, bm25_retriever=bm25)
        if _check_stop(_stop, "4"): return

        log.banner(5, "Term Categorization")
        run_term_categorization()
        if _check_stop(_stop, "5"): return

    # --- CQ-Driven Refinement mode ---
    if args.refine:
        _run_refinement_pipeline(args, _stop)
        return

    log.banner(6, "Taxonomy Builder")
    from src.modules.construct.taxonomy_builder import run_taxonomy_builder
    cat_csv = os.environ["CATEGORIZED_LLM_TERMS"]
    run_taxonomy_builder(cat_csv)
    if _check_stop(_stop, "6"): return

    # Step 6b: Relation Extraction
    relations_csv = None
    if not args.skip_relations:
        log.banner("6b", "Relation Extraction")
        from src.modules.construct.relation_extractor import run_relation_extraction
        relations_csv = run_relation_extraction(cat_csv)
    else:
        log.info("Step 6b: Relation extraction skipped (--skip-relations)")
    if _check_stop(_stop, "6b"): return

    # Step 6c: Ontology Critic
    log.banner("6c", "Ontology Critic")
    from src.modules.validate.ontology_critic import run_ontology_critic
    tax_csv = (
        os.path.splitext(cat_csv)[0]
        .replace("classify_categories", "construct_taxonomy") + ".csv"
    )
    cleaned_tax = tax_csv.replace("construct_taxonomy", "validate_critic_taxonomy")
    cleaned_rel = relations_csv.replace("construct_relations", "validate_critic_relations") if relations_csv else None
    run_ontology_critic(
        tax_csv,
        output_path=cleaned_tax,
        relations_csv=relations_csv,
        relations_output=cleaned_rel,
    )
    if _check_stop(_stop, "6c"): return

    # Step 6d: Relation-based reclassification
    reclass_tax = cleaned_tax  # fallback if no relations
    if cleaned_rel and os.path.exists(cleaned_rel):
        log.banner("6d", "Relation Reclassification")
        from src.modules.validate.relation_reclassifier import run_relation_reclassification
        reclass_tax = cleaned_tax.replace("validate_critic_taxonomy", "validate_reclassified_taxonomy")
        run_relation_reclassification(
            cleaned_tax,
            cleaned_rel,
            output_path=reclass_tax,
        )
    if _check_stop(_stop, "6d"): return

    # Use best available outputs for OWL export
    final_tax = reclass_tax if os.path.exists(reclass_tax) else cleaned_tax
    final_rel = cleaned_rel if (cleaned_rel and os.path.exists(cleaned_rel)) else relations_csv

    log.banner(7, "OWL Export")
    from src.modules.emit.owl_exporter import run_owl_export
    owl_path = run_owl_export(final_tax, relations_csv=final_rel)
    if _check_stop(_stop, "7"): return

    # Step 7b: Ontology Verification
    from src.modules.emit.verifier import run_ontology_verification
    run_ontology_verification(owl_path, skip_oops=args.skip_oops, skip_reasoner=args.skip_reasoner)

    log.success("\nPipeline complete.")


if __name__ == "__main__":
    main()
