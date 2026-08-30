"""Execute domain-configured competency questions against an emitted ontology."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from rdflib import Graph


_VALID_MODES = {"enumeration", "relations_exist", "relational"}
_VERDICT_ORDER = ("FULL", "PARTIAL", "NOT ANSWERABLE")
_VALID_VERDICTS = set(_VERDICT_ORDER)


def _run_query(graph: Graph, query: str) -> tuple[int, list[list[str]]]:
    rows = [[str(value) for value in row] for row in graph.query(query)]
    return len(rows), rows[:20]


def _load_graph(ttl_path: str | Path, closure_paths: list[Path]) -> Graph:
    graph = Graph()
    graph.parse(str(ttl_path), format="turtle")
    for closure_path in closure_paths:
        graph.parse(str(closure_path))
    return graph


def _classify(mode: str, inventory_count: int, answer_count: int) -> str:
    if mode == "enumeration":
        return "FULL" if inventory_count else "NOT ANSWERABLE"
    if mode == "relations_exist":
        return "FULL" if answer_count else "NOT ANSWERABLE"
    return "PARTIAL" if answer_count else "NOT ANSWERABLE"


def evaluate_competency_questions(
    ttl_path: str | Path,
    spec_path: str | Path,
    closure_paths: list[Path] | None = None,
) -> dict:
    """Return FULL, PARTIAL, or NOT ANSWERABLE for each configured question."""
    data = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
    questions = data.get("questions", []) if isinstance(data, dict) else []
    if not questions:
        raise ValueError("CQ answerability spec requires a questions list")

    resolved_closure_paths = closure_paths or []
    graph = _load_graph(ttl_path, resolved_closure_paths)
    baseline_value = str(data.get("baseline_artifact", "")).strip()
    if not baseline_value:
        raise ValueError("CQ answerability spec requires baseline_artifact")
    baseline_path = (Path(spec_path).parent / baseline_value).resolve()
    if not baseline_path.exists():
        raise ValueError(f"CQ baseline artifact does not exist: {baseline_path}")
    baseline_graph = _load_graph(baseline_path, resolved_closure_paths)

    results: list[dict] = []
    for item in questions:
        cq_id = str(item.get("id", "")).strip()
        mode = str(item.get("mode", "")).strip()
        if not cq_id or mode not in _VALID_MODES:
            raise ValueError(f"Invalid CQ specification: {item!r}")
        if mode == "relational" and not item.get("inventory_query"):
            raise ValueError(f"Relational CQ {cq_id} requires an inventory_query")
        inventory_count = 0
        inventory_rows: list[list[str]] = []
        if item.get("inventory_query"):
            inventory_count, inventory_rows = _run_query(
                graph, str(item["inventory_query"])
            )
            baseline_inventory_count, baseline_inventory_rows = _run_query(
                baseline_graph, str(item["inventory_query"])
            )
        else:
            baseline_inventory_count = 0
            baseline_inventory_rows = []
        answer_count = 0
        answer_rows: list[list[str]] = []
        if item.get("answer_query"):
            answer_count, answer_rows = _run_query(graph, str(item["answer_query"]))
            baseline_answer_count, baseline_answer_rows = _run_query(
                baseline_graph, str(item["answer_query"])
            )
        else:
            baseline_answer_count = 0
            baseline_answer_rows = []

        verdict = _classify(mode, inventory_count, answer_count)
        baseline_verdict = _classify(
            mode, baseline_inventory_count, baseline_answer_count
        )

        expected_baseline_verdict = str(
            item.get("expected_baseline_verdict", "")
        ).strip()
        if (
            expected_baseline_verdict
            and expected_baseline_verdict not in _VALID_VERDICTS
        ):
            raise ValueError(
                f"CQ {cq_id} has invalid expected_baseline_verdict "
                f"{expected_baseline_verdict!r}"
            )
        if (
            expected_baseline_verdict
            and expected_baseline_verdict != baseline_verdict
        ):
            raise ValueError(
                f"CQ {cq_id} measured baseline verdict {baseline_verdict}, "
                f"expected {expected_baseline_verdict}"
            )
        result = {
            "id": cq_id,
            "question": str(item.get("question", "")).strip(),
            "mode": mode,
            "inventory_count": inventory_count,
            "baseline_inventory_count": baseline_inventory_count,
            "inventory_count_delta": inventory_count - baseline_inventory_count,
            "answer_bindings": answer_count,
            "baseline_answer_bindings": baseline_answer_count,
            "answer_bindings_delta": answer_count - baseline_answer_count,
            "verdict": verdict,
            "baseline_verdict": baseline_verdict,
            "verdict_changed": baseline_verdict != verdict,
            "inventory_sample": inventory_rows,
            "baseline_inventory_sample": baseline_inventory_rows,
            "answer_sample": answer_rows,
            "baseline_answer_sample": baseline_answer_rows,
            "absent_relation_pattern": (
                "" if answer_count else str(item.get("absent_relation_pattern", ""))
            ),
        }

        change_spec = item.get("verdict_change")
        if result["verdict_changed"]:
            if not isinstance(change_spec, dict):
                raise ValueError(
                    f"CQ {cq_id} changed verdict but has no verdict_change record"
                )
            retained_query = str(change_spec.get("retained_axiom_query", "")).strip()
            if not retained_query:
                raise ValueError(
                    f"CQ {cq_id} verdict_change requires retained_axiom_query"
                )
            retained_count, retained_rows = _run_query(graph, retained_query)
            baseline_retained_count, baseline_retained_rows = _run_query(
                baseline_graph, retained_query
            )
            retained = retained_count > 0
            if change_spec.get("require_retained_axiom", True) and not retained:
                raise ValueError(
                    f"CQ {cq_id} expected its supporting axiom to remain present"
                )
            result["verdict_change"] = {
                "from": baseline_verdict,
                "to": verdict,
                "correction_ids": [
                    str(value) for value in change_spec.get("correction_ids", [])
                ],
                "cause": str(change_spec.get("cause", "")).strip(),
                "underlying_axiom": str(
                    change_spec.get("underlying_axiom", "")
                ).strip(),
                "underlying_axiom_retained": retained,
                "baseline_axiom_bindings": baseline_retained_count,
                "retained_axiom_bindings": retained_count,
                "baseline_axiom_sample": baseline_retained_rows,
                "retained_axiom_sample": retained_rows,
            }
        elif change_spec:
            raise ValueError(
                f"CQ {cq_id} has a verdict_change record but its verdict did not change"
            )

        results.append(result)

    counts = {
        verdict: sum(result["verdict"] == verdict for result in results)
        for verdict in _VERDICT_ORDER
    }
    baseline_counts = {
        verdict: sum(result["baseline_verdict"] == verdict for result in results)
        for verdict in _VERDICT_ORDER
    }
    expected_baseline = data.get("expected_baseline_summary")
    if expected_baseline and baseline_counts != expected_baseline:
        raise ValueError(
            f"Configured baseline summary {expected_baseline} does not match "
            f"question records {baseline_counts}"
        )
    verdict_changes = [
        {
            "id": result["id"],
            **result["verdict_change"],
        }
        for result in results
        if result["verdict_changed"]
    ]
    return {
        "spec": os.path.relpath(spec_path, Path.cwd()),
        "baseline_artifact": os.path.relpath(baseline_path, Path.cwd()),
        "questions": results,
        "baseline_summary": baseline_counts,
        "summary": counts,
        "verdict_changes": verdict_changes,
    }
