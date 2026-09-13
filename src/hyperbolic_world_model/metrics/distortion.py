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


def average_distortion(
    embeddings: Tensor,
    true_dist: Tensor,
    manifold: Manifold,
    fit_scale: bool = True,
) -> float:
    """Average relative distortion between manifold distances and ground-truth distances.

    Args:
        embeddings: ``(n, d)`` points on ``manifold``.
        true_dist: ``(n, n)`` ground-truth distances (graph hop counts or a tree metric).
        manifold: geometry of the embeddings.
        fit_scale: if true, a single global scale is fitted first so the metric is
            scale-invariant (embeddings that are a perfect scaled copy score ``0``).

    Returns:
        Mean relative distortion over off-diagonal pairs with ``true_dist > 0``.
    """
    if embeddings.ndim != 2 or true_dist.shape != (embeddings.shape[0], embeddings.shape[0]):
        raise ValueError("embeddings must be (n, d) and true_dist (n, n)")
    with torch.no_grad():
        emb_dist = manifold.pairwise_dist(embeddings).to(torch.float64)
    true_dist = true_dist.to(torch.float64)
    mask = (true_dist > 0) & ~torch.eye(
        true_dist.shape[0], dtype=torch.bool, device=true_dist.device
    )
    if fit_scale:
        emb_dist = emb_dist * _fit_scale(emb_dist, true_dist, mask)
    rel = (emb_dist[mask] - true_dist[mask]).abs() / true_dist[mask]
    return float(rel.mean())


def mean_average_precision(embeddings: Tensor, adjacency: Tensor, manifold: Manifold) -> float:
    """mAP for neighbourhood retrieval: rank nodes by manifold distance, score true edges.

    Args:
        embeddings: ``(n, d)`` points on ``manifold``.
        adjacency: ``(n, n)`` boolean matrix of ground-truth edges (symmetric, zero diagonal).
        manifold: geometry of the embeddings.

    Returns:
        Mean over nodes with at least one neighbour of the average precision of the ranking.
    """
    n = embeddings.shape[0]
    if adjacency.shape != (n, n):
        raise ValueError("adjacency must be (n, n)")
    with torch.no_grad():
        dist = manifold.pairwise_dist(embeddings).to(torch.float64)
    adjacency = adjacency.bool()
    aps = []
    for i in range(n):
        rel = adjacency[i].clone()
        rel[i] = False
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue
        d = dist[i].clone()
        d[i] = float("inf")
        order = torch.argsort(d)
        hits = rel[order].to(torch.float64)
        cum_hits = hits.cumsum(0)
        ranks = torch.arange(1, n + 1, dtype=torch.float64, device=hits.device)
        precision_at_hit = (cum_hits / ranks)[hits.bool()]
        aps.append(float(precision_at_hit.mean()))
    if not aps:
        raise ValueError("adjacency has no edges")
    return float(sum(aps) / len(aps))


__all__ = ["average_distortion", "mean_average_precision"]
