"""Unit tests for src.validate.engines.hybrid.

Tests the cosine-similarity clustering logic with synthetic embeddings.
The embedding-model load path (`embed_terms`) is NOT exercised here —
that's an integration test deferred to the e2e Phase 2.8 run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.validate.engines.hybrid import (  # noqa: E402
    TermCluster,
    cluster_terms_by_similarity,
)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(matrix, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return matrix / n


def test_no_pairs_above_threshold_returns_empty():
    terms = ["alpha", "beta", "gamma"]
    embeddings = _normalise(np.eye(3, dtype=np.float32))  # orthogonal — cos sim 0
    clusters = cluster_terms_by_similarity(terms, embeddings, threshold=0.5)
    assert clusters == [], f"Expected empty, got {clusters}"
    print("  ✓ orthogonal terms produce no clusters")


def test_identical_terms_cluster():
    terms = ["sandstone", "Sandstone", "limestone"]
    v_sand = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_lime = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    emb = _normalise(np.vstack([v_sand, v_sand + 0.001, v_lime]).astype(np.float32))
    clusters = cluster_terms_by_similarity(terms, emb, threshold=0.92)
    assert len(clusters) == 1, f"Expected 1 cluster, got {len(clusters)}"
    assert set(clusters[0].members) == {"sandstone", "Sandstone"}
    assert "limestone" not in clusters[0].members
    assert clusters[0].pairwise_min_similarity >= 0.92
    print("  ✓ near-identical terms cluster correctly")


def test_max_cluster_size_caps_grouping():
    # 4 highly similar terms; cap at size 2 should produce 2 clusters of 2
    terms = ["a", "b", "c", "d"]
    base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    perturbations = np.array([
        base,
        base + np.array([0.001, 0.0, 0.0]),
        base + np.array([0.0, 0.001, 0.0]),
        base + np.array([0.0, 0.0, 0.001]),
    ], dtype=np.float32)
    emb = _normalise(perturbations)
    clusters = cluster_terms_by_similarity(
        terms, emb, threshold=0.92, max_cluster_size=2
    )
    sizes = sorted(len(c.members) for c in clusters)
    assert sizes == [2, 2], f"Expected sizes [2, 2], got {sizes}"
    # Every member appears exactly once
    members_flat = [m for c in clusters for m in c.members]
    assert sorted(members_flat) == ["a", "b", "c", "d"]
    print("  ✓ max_cluster_size caps grouping")


def test_returns_termcluster_dataclass():
    terms = ["x", "y"]
    emb = _normalise(np.ones((2, 3), dtype=np.float32))
    clusters = cluster_terms_by_similarity(terms, emb, threshold=0.5)
    assert len(clusters) == 1
    c = clusters[0]
    assert isinstance(c, TermCluster)
    assert c.cluster_id.startswith("dup_")
    assert isinstance(c.members, tuple)
    assert 0.0 <= c.pairwise_min_similarity <= 1.0001
    print("  ✓ returns TermCluster dataclass with cluster_id")


def test_length_mismatch_raises():
    try:
        cluster_terms_by_similarity(["a", "b"], np.zeros((3, 4), dtype=np.float32))
    except ValueError:
        print("  ✓ length mismatch raises ValueError")
        return
    raise AssertionError("Expected ValueError")


def test_fewer_than_two_terms_returns_empty():
    assert cluster_terms_by_similarity([], np.zeros((0, 3), dtype=np.float32)) == []
    assert cluster_terms_by_similarity(["only"], np.zeros((1, 3), dtype=np.float32)) == []
    print("  ✓ trivial inputs return empty")


def test_deterministic_ordering():
    terms = ["a", "b", "c", "d"]
    # Two clusters: {a, b} very similar, {c, d} less similar
    e_a = np.array([1.0, 0.0], dtype=np.float32)
    e_b = np.array([0.999, 0.045], dtype=np.float32)
    e_c = np.array([0.0, 1.0], dtype=np.float32)
    e_d = np.array([0.06, 0.998], dtype=np.float32)
    emb = _normalise(np.vstack([e_a, e_b, e_c, e_d]))
    clusters = cluster_terms_by_similarity(terms, emb, threshold=0.92)
    assert len(clusters) == 2
    # Higher min-sim cluster comes first
    assert clusters[0].pairwise_min_similarity >= clusters[1].pairwise_min_similarity
    print("  ✓ deterministic ordering by min-similarity desc")


def main() -> int:
    tests = [
        test_no_pairs_above_threshold_returns_empty,
        test_identical_terms_cluster,
        test_max_cluster_size_caps_grouping,
        test_returns_termcluster_dataclass,
        test_length_mismatch_raises,
        test_fewer_than_two_terms_returns_empty,
        test_deterministic_ordering,
    ]
    print(f"=== test_hybrid_engine.py — {len(tests)} tests ===")
    for t in tests:
        t()
    print(f"=== ALL {len(tests)} TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
