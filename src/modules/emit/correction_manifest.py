"""Validated, domain-configured graph corrections for deterministic releases."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import yaml
from rdflib import BNode, Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from src.utils.csv_io import read_csv


_REQUIRED_DECISION_FIELDS = {
    "id",
    "defect_id",
    "action",
    "subject",
    "resolution",
    "critic_verdict",
    "critic_reasoning",
    "confidence",
    "justification",
    "status",
}
GRAPH_ACTIONS = {
    "drop_restriction",
    "reparent_class",
    "replace_equivalent_genus",
}
_EVIDENCE_ACTIONS = {
    "pipeline_fix",
    "demotion_trace",
    "cq_deferred",
}
_SUPPORTED_ACTIONS = GRAPH_ACTIONS | _EVIDENCE_ACTIONS


def load_correction_manifest(path: str | Path) -> dict:
    """Load and validate a versioned correction manifest."""
    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")
    if "\u2014" in text:
        raise ValueError(f"Correction manifest contains a forbidden em dash: {manifest_path}")
    data = yaml.safe_load(text)
    if not isinstance(data, dict) or not str(data.get("version", "")).strip():
        raise ValueError("Correction manifest requires a non-empty version")
    decisions = list(data.get("decisions") or [])
    for relative_path in data.get("decision_files", []) or []:
        decision_path = manifest_path.parent / str(relative_path)
        frame = read_csv(decision_path).fillna("")
        decisions.extend(frame.to_dict(orient="records"))
    data["decisions"] = decisions
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("Correction manifest requires at least one decision")

    seen: set[str] = set()
    for index, decision in enumerate(decisions, start=1):
        if not isinstance(decision, dict):
            raise ValueError(f"Correction manifest decision {index} is not a mapping")
        missing = sorted(_REQUIRED_DECISION_FIELDS - set(decision))
        if missing:
            raise ValueError(
                f"Correction manifest decision {index} is missing fields: {missing}"
            )
        decision_id = str(decision["id"]).strip()
        if not decision_id or decision_id in seen:
            raise ValueError(f"Duplicate or empty correction decision id: {decision_id!r}")
        seen.add(decision_id)
        if decision["confidence"] not in {"final", "provisional"}:
            raise ValueError(
                f"Decision {decision_id} confidence must be final or provisional"
            )
        if decision["status"] not in {"fixed", "deferred", "recorded"}:
            raise ValueError(
                f"Decision {decision_id} status must be fixed, deferred, or recorded"
            )
        if decision["action"] not in _SUPPORTED_ACTIONS:
            raise ValueError(
                f"Decision {decision_id} has unsupported action {decision['action']!r}"
            )
    return data


def _remove_orphaned_bnode_subgraph(graph: Graph, node: BNode) -> None:
    if any(True for _ in graph.triples((None, None, node))):
        return
    children = [
        obj
        for _, _, obj in graph.triples((node, None, None))
        if isinstance(obj, BNode)
    ]
    graph.remove((node, None, None))
    for child in children:
        _remove_orphaned_bnode_subgraph(graph, child)


def _resolve_property(value: str, relation_iris: dict[str, str]) -> URIRef:
    value = value.strip()
    iri = relation_iris.get(value, value)
    if not iri.startswith(("http://", "https://")):
        raise ValueError(f"Unknown correction-manifest property: {value}")
    return URIRef(iri)


def apply_correction_manifest(
    graph: Graph,
    manifest: dict,
    resolve_entity: Callable[[str], URIRef],
    relation_iris: dict[str, str],
) -> list[dict]:
    """Apply graph-changing decisions and return an auditable action log."""
    applied: list[dict] = []
    for decision in manifest["decisions"]:
        action = str(decision["action"]).strip()
        decision_id = str(decision["id"]).strip()
        if hasattr(graph, "set_source"):
            graph.set_source(f"manifest:{decision_id}")
        if action not in GRAPH_ACTIONS:
            applied.append(
                {
                    "decision_id": decision_id,
                    "action": action,
                    "matches": 0,
                    "graph_change": False,
                }
            )
            continue

        subject = resolve_entity(str(decision["subject"]))
        matches = 0
        if action == "drop_restriction":
            prop = _resolve_property(str(decision.get("predicate", "")), relation_iris)
            filler = resolve_entity(str(decision.get("object", "")))
            for restriction in list(graph.objects(subject, RDFS.subClassOf)):
                if not isinstance(restriction, BNode):
                    continue
                if (restriction, OWL.onProperty, prop) not in graph:
                    continue
                has_filler = (
                    (restriction, OWL.someValuesFrom, filler) in graph
                    or (restriction, OWL.hasValue, filler) in graph
                )
                if not has_filler:
                    continue
                graph.remove((subject, RDFS.subClassOf, restriction))
                _remove_orphaned_bnode_subgraph(graph, restriction)
                matches += 1
        elif action == "reparent_class":
            old_parent = resolve_entity(str(decision.get("object", "")))
            new_parent = resolve_entity(str(decision.get("replacement_object", "")))
            if (subject, RDFS.subClassOf, old_parent) in graph:
                graph.remove((subject, RDFS.subClassOf, old_parent))
                graph.add((subject, RDFS.subClassOf, new_parent))
                matches = 1
        elif action == "replace_equivalent_genus":
            old_genus = resolve_entity(str(decision.get("object", "")))
            new_genus = resolve_entity(str(decision.get("replacement_object", "")))
            for expression in graph.objects(subject, OWL.equivalentClass):
                for head in graph.objects(expression, OWL.intersectionOf):
                    node = head
                    while node and node != RDF.nil:
                        first = next(graph.objects(node, RDF.first), None)
                        if first == old_genus:
                            graph.set((node, RDF.first, new_genus))
                            matches += 1
                        node = next(graph.objects(node, RDF.rest), None)

        if matches != 1:
            raise ValueError(
                f"Correction decision {decision_id} expected one graph match, got {matches}"
            )
        applied.append(
            {
                "decision_id": decision_id,
                "action": action,
                "matches": matches,
                "graph_change": True,
            }
        )
    if hasattr(graph, "set_source"):
        graph.set_source("pipeline:src.modules.emit.owl_exporter")
    return applied
