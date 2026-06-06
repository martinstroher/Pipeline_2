"""Hybrid validation engine — embedding-based pre-clustering.

The only hybrid rule today is `near_duplicate`. This module computes
term embeddings with BAAI/bge-m3 (the same model the RAG pipeline uses,
so no extra weights are downloaded), groups terms whose cosine
similarity exceeds the threshold into candidate clusters, and returns
those clusters for downstream LLM adjudication via
`src/validate/engines/llm.evaluate_batch()`.

Embeddings are computed in a single batched call and held in memory for
the duration of the run; we do not persist them between runs because the
term set typically changes between validator invocations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TermCluster:
    """A candidate near-duplicate cluster proposed by similarity sweep."""

    cluster_id: str
    members: tuple[str, ...]
    pairwise_min_similarity: float


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_terms(
    terms: list[str],
    *,
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Return a (len(terms), dim) float32 numpy array of L2-normalised embeddings.

    Loads the embedding model lazily — callers are expected to be operating
    in the same environment the rest of the pipeline runs in (where the
    BGE-M3 weights are already cached locally by the RAG indexer).
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings  # heavy import

    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
    )
    vectors = embedder.embed_documents(terms)
    arr = np.asarray(vectors, dtype=np.float32)
    return _normalize(arr)


def cluster_terms_by_similarity(
    terms: list[str],
    embeddings: np.ndarray,
    *,
    threshold: float = 0.92,
    max_cluster_size: int = 4,
) -> list[TermCluster]:
    """Group terms whose pairwise cosine similarity ≥ `threshold`.

    Returns a list of `TermCluster` with ≥2 members each. Clusters are
    formed by a simple greedy union-find pass over the similarity matrix:
    every pair with similarity above the threshold joins its members'
    components. Components larger than `max_cluster_size` are split into
    chunks of that size (preserving the highest-similarity pairs first).
    """
    if len(terms) != embeddings.shape[0]:
        raise ValueError(
            f"terms ({len(terms)}) and embeddings ({embeddings.shape[0]}) length mismatch"
        )
    if len(terms) < 2:
        return []

    # Cosine sim on normalised vectors = dot product.
    sim = embeddings @ embeddings.T

    # Collect candidate pairs above threshold, sorted by similarity desc.
    n = len(terms)
    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim[i, j])
            if s >= threshold:
                pairs.append((s, i, j))
    if not pairs:
        return []
    pairs.sort(reverse=True)

    # Union-find with size cap.
    parent = list(range(n))
    size = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if size[ra] + size[rb] > max_cluster_size:
            return False
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        return True

    for _s, i, j in pairs:
        union(i, j)

    # Build clusters from components.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clusters: list[TermCluster] = []
    for cid, idxs in groups.items():
        if len(idxs) < 2:
            continue
        members = tuple(terms[i] for i in idxs)
        # Min pairwise sim across the cluster (worst link).
        min_s = 1.0
        for a_i, ai in enumerate(idxs):
            for aj in idxs[a_i + 1 :]:
                s = float(sim[ai, aj])
                if s < min_s:
                    min_s = s
        clusters.append(
            TermCluster(
                cluster_id=f"dup_{cid:04d}",
                members=members,
                pairwise_min_similarity=min_s,
            )
        )
    # Deterministic ordering: by min similarity desc, then by member tuple.
    clusters.sort(key=lambda c: (-c.pairwise_min_similarity, c.members))
    return clusters


def cluster_terms(
    terms: list[str],
    *,
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
    threshold: float = 0.92,
    max_cluster_size: int = 4,
) -> list[TermCluster]:
    """End-to-end helper: embed terms and cluster them.

    Convenience wrapper for callers that don't need the raw embeddings.
    """
    if len(terms) < 2:
        return []
    embeddings = embed_terms(terms, model_name=model_name, device=device)
    return cluster_terms_by_similarity(
        terms,
        embeddings,
        threshold=threshold,
        max_cluster_size=max_cluster_size,
    )
