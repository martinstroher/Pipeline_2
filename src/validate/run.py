"""High-level validate orchestrator — the public entry point of the
six-verb pipeline's `validate` stage.

Runs in order:
  1. rule_applier      → validate_rule_verdicts.csv
  2. domain_filters    → validate_filter_verdicts.csv
  3. aggregator        → validate_{taxonomy,relations}.csv,
                         validate_log.csv,
                         validate_per_condition_stats.csv

Replaces the legacy two-step pair (ontology_critic + relation_reclassifier).
"""

from __future__ import annotations

import os

from src.utils import log
from src.validate.aggregator import run_aggregator
from src.validate.domain_filters import run_domain_filters
from src.validate.rule_applier import run_rule_applier


def run_validate(
    taxonomy_csv: str,
    output_dir: str,
    *,
    relations_csv: str | None = None,
    cq_covered_terms_csv: str | None = None,
) -> tuple[str, str | None]:
    """Run the full validate stage. Returns (final_taxonomy, final_relations_or_None).

    `output_dir` is where every validate_*.csv lands; callers typically
    pass the same directory that holds the construct_*.csv inputs.
    """
    os.makedirs(output_dir, exist_ok=True)

    log.banner("validate", "Validate (rules → filters → aggregate)")

    rule_csv = run_rule_applier(
        taxonomy_csv,
        relations_csv or "",
        output_dir,
        cq_covered_terms_csv=cq_covered_terms_csv,
    )
    filter_csv = run_domain_filters(
        taxonomy_csv,
        output_dir,
        relations_csv=relations_csv,
    )
    tax_out, rel_out, _, _ = run_aggregator(
        taxonomy_csv,
        rule_csv,
        filter_csv,
        output_dir,
        relations_csv=relations_csv,
    )
    return tax_out, rel_out
