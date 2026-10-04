"""Embedding-quality metrics for hierarchy reconstruction.

Given ground-truth graph distances ``d_G`` (e.g. hop counts on the embodiment > task > primitive
tree) and an embedding on a manifold, we report

* **average distortion**: mean over pairs of ``|d_M(i, j) - d_G(i, j)| / d_G(i, j)`` after fitting
  a single global scale, and
* **mAP**: for each node, precision at retrieving its true graph neighbours by manifold distance,
  averaged (Nickel & Kiela, 2017).

Both use ``manifold.dist`` and are therefore reported in the native geometry of the model.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hyperbolic_world_model.geometry.base import Manifold


def _fit_scale(emb_dist: Tensor, true_dist: Tensor, mask: Tensor) -> Tensor:
    """Least-squares scale ``s`` minimising ``|s * emb - true|`` over ``mask``."""
    num = (emb_dist[mask] * true_dist[mask]).sum()
    den = (emb_dist[mask] ** 2).sum().clamp_min(torch.finfo(emb_dist.dtype).eps)
    return num / den


def pairwise_distances(embeddings: Tensor, manifold: Manifold) -> Tensor:
    """``(n, n)`` float64 manifold distance matrix; share it between the metrics below."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be (n, d)")
    with torch.no_grad():
        return manifold.pairwise_dist(embeddings).to(torch.float64)


def average_distortion(
    embeddings: Tensor,
    true_dist: Tensor,
    manifold: Manifold,
    fit_scale: bool = True,
    pairwise: Tensor | None = None,
) -> float:
    """Average relative distortion between manifold distances and ground-truth distances.

    Args:
        embeddings: ``(n, d)`` points on ``manifold``.
        true_dist: ``(n, n)`` ground-truth distances (graph hop counts or a tree metric).
        manifold: geometry of the embeddings.
        fit_scale: if true, a single global scale is fitted first so the metric is
            scale-invariant (embeddings that are a perfect scaled copy score ``0``).
        pairwise: optional precomputed :func:`pairwise_distances` of ``embeddings`` (the
            hierarchy task scores distortion and mAP from one matrix).

    Returns:
        Mean relative distortion over off-diagonal pairs with ``true_dist > 0``.
    """
    if embeddings.ndim != 2 or true_dist.shape != (embeddings.shape[0], embeddings.shape[0]):
        raise ValueError("embeddings must be (n, d) and true_dist (n, n)")
    emb_dist = pairwise_distances(embeddings, manifold) if pairwise is None else pairwise
    emb_dist = emb_dist.to(torch.float64)
    true_dist = true_dist.to(torch.float64)
    mask = (true_dist > 0) & ~torch.eye(
        true_dist.shape[0], dtype=torch.bool, device=true_dist.device
    )
    if fit_scale:
        emb_dist = emb_dist * _fit_scale(emb_dist, true_dist, mask)
    rel = (emb_dist[mask] - true_dist[mask]).abs() / true_dist[mask]
    return float(rel.mean())


def expected_average_precision(dist: Tensor, relevant: Tensor) -> float:
    """Average precision of ranking ``dist`` ascending, exact in expectation over tied distances.

    Candidates at equal distance are ranked in a uniformly random order, so the result does not
    depend on how a sort breaks ties (the hierarchy pipeline produces exact ties structurally:
    a node with a single child shares its Fréchet mean). Without ties this is the usual AP.

    Args:
        dist: ``(m,)`` distances of the candidates (the query itself excluded).
        relevant: ``(m,)`` boolean relevance of each candidate; at least one must be true.
    """
    dist = dist.to(torch.float64)
    relevant = relevant.bool()
    n_rel = int(relevant.sum())
    if n_rel == 0:
        raise ValueError("no relevant candidates")
    order = torch.argsort(dist, stable=True)
    d_sorted, rel_sorted = dist[order], relevant[order]
    counts = torch.unique_consecutive(d_sorted, return_counts=True)[1].tolist()
    if len(counts) == len(d_sorted):  # no ties: the vectorised textbook formula
        hits = rel_sorted.to(torch.float64)
        ranks = torch.arange(1, len(hits) + 1, dtype=torch.float64, device=hits.device)
        return float((hits.cumsum(0) / ranks)[rel_sorted].mean())
    total, n_before, r_before, pos = 0.0, 0, 0, 0
    for g in counts:
        r = int(rel_sorted[pos : pos + g].sum())
        if r:
            k = torch.arange(1, g + 1, dtype=torch.float64)
            # A relevant item uniformly at position k of its tie group is preceded, within the
            # group, by a hypergeometric number of other relevant items with mean (k-1)(r-1)/(g-1).
            others = (k - 1) * (r - 1) / (g - 1) if g > 1 else torch.zeros_like(k)
            total += r * float(((r_before + 1 + others) / (n_before + k)).mean())
        n_before, r_before, pos = n_before + g, r_before + r, pos + g
    return total / n_rel


def mean_average_precision(
    embeddings: Tensor,
    adjacency: Tensor,
    manifold: Manifold,
    pairwise: Tensor | None = None,
) -> float:
    """mAP for neighbourhood retrieval: rank nodes by manifold distance, score true edges.

    Ties in distance are scored by :func:`expected_average_precision`, so equal embeddings give
    the same mAP whatever order a sort happens to put them in.

    Args:
        embeddings: ``(n, d)`` points on ``manifold``.
        adjacency: ``(n, n)`` boolean matrix of ground-truth edges (symmetric, zero diagonal).
        manifold: geometry of the embeddings.
        pairwise: optional precomputed :func:`pairwise_distances` of ``embeddings``.

    Returns:
        Mean over nodes with at least one neighbour of the average precision of the ranking.
    """
    n = embeddings.shape[0]
    if adjacency.shape != (n, n):
        raise ValueError("adjacency must be (n, n)")
    dist = pairwise_distances(embeddings, manifold) if pairwise is None else pairwise
    dist = dist.to(torch.float64)
    adjacency = adjacency.bool()
    not_self = ~torch.eye(n, dtype=torch.bool, device=dist.device)
    aps = []
    for i in range(n):
        rel = adjacency[i] & not_self[i]
        if not bool(rel.any()):
            continue
        aps.append(expected_average_precision(dist[i][not_self[i]], rel[not_self[i]]))
    if not aps:
        raise ValueError("adjacency has no edges")
    return float(sum(aps) / len(aps))


__all__ = [
    "average_distortion",
    "expected_average_precision",
    "mean_average_precision",
    "pairwise_distances",
]
