"""Stable, causal triple provenance for emitted RDF graphs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from rdflib import BNode, Graph
from rdflib.compare import _TripleCanonicalizer

from src.utils.csv_io import write_csv


class ProvenanceGraph(Graph):
    """RDF graph that records the active source for every added triple."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active_source = "pipeline:src.modules.emit.owl_exporter"
        self._triple_sources: dict[tuple, set[str]] = defaultdict(set)

    @contextmanager
    def source(self, source_ref: str):
        previous = self._active_source
        self._active_source = source_ref
        try:
            yield self
        finally:
            self._active_source = previous

    def add(self, triple):
        super().add(triple)
        self._triple_sources[triple].add(self._active_source)
        return self

    def remove(self, triple):
        removed = list(self.triples(triple))
        super().remove(triple)
        for item in removed:
            self._triple_sources.pop(item, None)
        return self

    def sources_for(self, triple: tuple) -> list[str]:
        if triple not in self._triple_sources:
            raise RuntimeError(f"Emitted triple has no causal provenance: {triple!r}")
        return sorted(self._triple_sources[triple])

    def set_source(self, source_ref: str) -> None:
        self._active_source = source_ref


@contextmanager
def source_scope(graph: Graph, source_ref: str):
    """Set a provenance source when the graph supports source tracking."""
    if isinstance(graph, ProvenanceGraph):
        with graph.source(source_ref):
            yield graph
    else:
        yield graph


def _canonical_triple_map(graph: Graph) -> dict[tuple, tuple]:
    canonicalizer = _TripleCanonicalizer(graph)
    coloring = canonicalizer._initial_color()
    coloring = canonicalizer._refine(coloring, coloring[:])
    if not canonicalizer._discrete(coloring):
        coloring = canonicalizer._traces(coloring)
    labels = {color.nodes[0]: color.hash_color() for color in coloring}
    return {
        triple: tuple(canonicalizer._canonicalize_bnodes(triple, labels))
        for triple in graph
    }


def _triple_text(triple: tuple) -> str:
    return " ".join(node.n3() for node in triple) + " ."


def write_triple_provenance(
    graph: Graph,
    output_path: str | Path,
    *,
    config_source: str,
    manifest_source: str | None,
    input_sources: list[str],
) -> dict:
    """Write one deterministic causal provenance row for every graph triple."""
    if not isinstance(graph, ProvenanceGraph):
        raise TypeError("Triple provenance requires a ProvenanceGraph")
    canonical_map = _canonical_triple_map(graph)
    rows: list[dict] = []
    seen: set[str] = set()
    for triple, canonical_triple in canonical_map.items():
        triple_text = _triple_text(canonical_triple)
        if triple_text in seen:
            raise RuntimeError(f"Canonical triple collision: {triple_text}")
        seen.add(triple_text)
        sources = graph.sources_for(triple)
        rows.append(
            {
                "triple": triple_text,
                "sources": json.dumps(sources, ensure_ascii=False),
                "config": config_source,
                "manifest": manifest_source or "",
                "inputs": json.dumps(sorted(input_sources), ensure_ascii=False),
            }
        )
    rows.sort(key=lambda row: row["triple"])
    if len(rows) != len(graph):
        raise RuntimeError(
            f"Triple provenance incomplete: {len(rows)} rows for {len(graph)} triples"
        )
    digest = hashlib.sha256(
        "\n".join(
            f"{row['triple']}\t{row['sources']}" for row in rows
        ).encode("utf-8")
    ).hexdigest()
    write_csv(pd.DataFrame(rows), output_path)
    return {
        "triple_count": len(rows),
        "provenance_complete": True,
        "canonical_provenance_sha256": digest,
        "path": str(output_path),
    }
