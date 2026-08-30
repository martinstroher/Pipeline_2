"""Execute domain-configured competency questions against an emitted ontology."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from rdflib import Graph


_VALID_MODES = {"enumeration", "relations_exist", "relational"}


def _run_query(graph: Graph, query: str) -> tuple[int, list[list[str]]]:
    rows = [[str(value) for value in row] for row in graph.query(query)]
    return len(rows), rows[:20]


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

    graph = Graph()
    graph.parse(str(ttl_path), format="turtle")
    for closure_path in closure_paths or []:
        graph.parse(str(closure_path))

    results: list[dict] = []
    for item in questions:
        cq_id = str(item.get("id", "")).strip()
        mode = str(item.get("mode", "")).strip()
        if not cq_id or mode not in _VALID_MODES:
            raise ValueError(f"Invalid CQ specification: {item!r}")
        inventory_count = 0
        inventory_rows: list[list[str]] = []
        if item.get("inventory_query"):
            inventory_count, inventory_rows = _run_query(
                graph, str(item["inventory_query"])
            )
        answer_count = 0
        answer_rows: list[list[str]] = []
        if item.get("answer_query"):
            answer_count, answer_rows = _run_query(graph, str(item["answer_query"]))

        if mode == "enumeration":
            verdict = "FULL" if inventory_count else "NOT ANSWERABLE"
        elif mode == "relations_exist":
            verdict = "FULL" if answer_count else "NOT ANSWERABLE"
        else:
            verdict = "PARTIAL" if answer_count else "NOT ANSWERABLE"

        results.append(
            {
                "id": cq_id,
                "question": str(item.get("question", "")).strip(),
                "mode": mode,
                "inventory_count": inventory_count,
                "answer_bindings": answer_count,
                "verdict": verdict,
                "inventory_sample": inventory_rows,
                "answer_sample": answer_rows,
                "absent_relation_pattern": (
                    "" if answer_count else str(item.get("absent_relation_pattern", ""))
                ),
            }
        )

    counts = {
        verdict: sum(result["verdict"] == verdict for result in results)
        for verdict in ("FULL", "PARTIAL", "NOT ANSWERABLE")
    }
    return {
        "spec": os.path.relpath(spec_path, Path.cwd()),
        "questions": results,
        "summary": counts,
    }
