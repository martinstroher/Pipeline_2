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


# Verb-name aliases for --stop-after. Each verb maps to the canonical step ID.
_STEP_ALIASES: dict[str, str] = {
    "extract": "3",
    "define": "4",
    "classify": "5b",
    "construct": "6b",
    "validate": "validate",
    "emit": "7b",
}

_NUMERIC_STOP_CHOICES = ("0", "R", "1", "2", "3", "4", "5", "5b", "6", "6b", "7", "7b")
_VERB_STOP_CHOICES = tuple(_STEP_ALIASES.keys())


def _resolve_stop_alias(stop: str | None) -> str | None:
    """Resolve verb-name aliases to numeric step IDs. Emits a deprecation warning
    when the legacy numeric form is used so callers migrate to verb names.
    """
    if stop is None:
        return None
    if stop in _STEP_ALIASES:
        return _STEP_ALIASES[stop]
    if stop in _NUMERIC_STOP_CHOICES:
        log.warn(
            f"--stop-after {stop!r} uses the legacy numeric form; "
            f"prefer verb names ({', '.join(_VERB_STOP_CHOICES)}). "
            f"Numeric aliases will be removed in a future release."
        )
        return stop
    return stop


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
        "--stop-after",
        type=str,
        default=None,
        choices=list(_NUMERIC_STOP_CHOICES) + list(_VERB_STOP_CHOICES),
        help=(
            "Stop pipeline after this step. Prefer verb names "
            "(extract|define|classify|construct|validate|emit). "
            "Numeric step IDs (0|R|1..7b) are accepted as deprecated aliases."
        ),
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if _dispatch_subcommand(args, parser):
        return

    # --- Fresh start: clean all outputs and caches ---
    if args.fresh:
        _clean_outputs()

    # --- Standard pipeline ---
    _stop = _resolve_stop_alias(args.stop_after)

    if not args.skip_pdf:
        log.banner(0, "PDF Text Extraction")
        convert_pdfs(DOCS_DIR)
        if _check_stop(_stop, "0"): return

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

    log.banner(4, "NLD Generation")
    run_nld_generation(vector_store=vector_store, bm25_retriever=bm25)
    if _check_stop(_stop, "4"): return

    log.banner(5, "Term Categorization")
    run_term_categorization()
    if _check_stop(_stop, "5"): return

    # Step 5b: CQ-driven refinement (mandatory). Cleans duplicates, scores every
    # surviving term against the 10 competency questions, and writes a filtered
    # categorized CSV containing only terms with CQ_Count >= 1. Downstream
    # steps consume that filtered CSV instead of the raw Step 5 output.
    log.banner("5b", "CQ-Driven Refinement")
    from src.modules.classify.cq_scorer import run_cq_refinement
    cat_csv = run_cq_refinement(os.environ["CATEGORIZED_LLM_TERMS"])
    hints_csv = os.path.join(os.path.dirname(cat_csv), "5b_specialization_hints.csv")
    if not os.path.exists(hints_csv):
        hints_csv = None
    if _check_stop(_stop, "5b"): return

    log.banner(6, "Taxonomy Builder")
    from src.modules.construct.taxonomy_builder import run_taxonomy_builder
    tax_csv = os.path.join(os.path.dirname(cat_csv), "construct_taxonomy.csv")
    run_taxonomy_builder(cat_csv, output_path=tax_csv, hints_csv=hints_csv)
    if _check_stop(_stop, "6"): return

    # Step 6b: Relation Extraction
    relations_csv = None
    if not args.skip_relations:
        log.banner("6b", "Relation Extraction")
        from src.modules.construct.relation_extractor import run_relation_extraction
        relations_csv = os.path.join(os.path.dirname(cat_csv), "construct_relations.csv")
        run_relation_extraction(cat_csv, output_path=relations_csv)
    else:
        log.info("Step 6b: Relation extraction skipped (--skip-relations)")
    if _check_stop(_stop, "6b"): return

    # Step validate: two-pass critic per category (taxonomy then relations)
    log.banner("validate", "Validate (taxonomy + relation critic per category)")
    from src.modules.validate.critic import run_critic
    tax_csv = (
        os.path.splitext(cat_csv)[0]
        .replace("classify_categories", "construct_taxonomy") + ".csv"
    )
    output_dir = os.path.dirname(tax_csv)
    final_tax, final_rel = run_critic(
        tax_csv,
        output_dir,
        relations_csv=relations_csv,
    )
    if _check_stop(_stop, "validate"): return

    # Defensive fallbacks if critic produced nothing writable.
    if not (final_tax and os.path.exists(final_tax)):
        final_tax = tax_csv
    if relations_csv and not (final_rel and os.path.exists(final_rel)):
        final_rel = relations_csv

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
