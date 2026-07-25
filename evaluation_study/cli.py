"""Command-line interface for the standalone thesis evaluation study."""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from evaluation_study.paths import REPO_ROOT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PreSaltOntoLearn Evaluation Study")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ablation = subparsers.add_parser("ablation", help="Run controlled A/B/C/D ablation")
    ablation.add_argument("--conditions", default="A,B,C,D")

    subparsers.add_parser("layer1", help="Run automated sensitivity analysis")

    expert = subparsers.add_parser("expert-workbooks", help="Generate expert workbooks")
    expert.add_argument("--terms", type=int, default=100)
    expert.add_argument("--experts", type=int, default=3)
    expert.add_argument("--seed", type=int, default=42)

    layer2 = subparsers.add_parser("layer2", help="Analyze completed expert workbooks")
    layer2.add_argument("workbooks", nargs="+")
    layer2.add_argument("--key", required=True)
    layer2.add_argument("--output-dir", default=None)

    rehearsal = subparsers.add_parser("rehearsal", help="Run zero-Azure synthetic rehearsal")
    rehearsal.add_argument("--overwrite", action="store_true")
    rehearsal.add_argument("--bootstrap-iterations", type=int, default=5000)

    relations = subparsers.add_parser("relation-analysis", help="Analyze relation output")
    relations.add_argument("relations_csv", nargs="?", default=None)
    return parser


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    args = _build_parser().parse_args()

    if args.command == "ablation":
        from evaluation_study.ablation_study import run_ablation

        run_ablation([value.strip().upper() for value in args.conditions.split(",")])
    elif args.command == "layer1":
        from evaluation_study.layer1_analysis import run_layer1_analysis

        run_layer1_analysis()
    elif args.command == "expert-workbooks":
        from evaluation_study.expert_eval_generator import generate_expert_evaluation

        generate_expert_evaluation(
            n_terms=args.terms,
            seed=args.seed,
            n_experts=args.experts,
        )
    elif args.command == "layer2":
        from evaluation_study.expert_eval_analyzer import run_layer2_analysis

        run_layer2_analysis(args.workbooks, args.key, args.output_dir)
    elif args.command == "rehearsal":
        from evaluation_study.offline_rehearsal import run_offline_rehearsal

        run_offline_rehearsal(
            overwrite=args.overwrite,
            bootstrap_iterations=args.bootstrap_iterations,
        )
    elif args.command == "relation-analysis":
        from evaluation_study.relation_analysis import run_relation_analysis

        run_relation_analysis(args.relations_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
