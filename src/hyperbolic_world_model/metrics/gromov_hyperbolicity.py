"""Gromov delta-hyperbolicity estimator.

A metric space is delta-hyperbolic if every geodesic triangle is delta-thin. Trees are 0-hyperbolic;
a Euclidean grid is not hyperbolic (delta grows with the diameter). We use the base-point
formulation (Gromov products) that admits a vectorised max-min matrix product:

    (x|y)_w  = 1/2 (d(x,w) + d(y,w) - d(x,y))
    delta_w  = max_{x,y,z} [ min((x|z)_w, (y|z)_w) - (x|y)_w ]

``delta_w`` is within a factor of two of the four-point delta for any base point ``w``. The
estimator is cubic in the number of points, so callers subsample.

Reference: Chami et al. (2020), "From Trees to Continuous Embeddings and Back", Appendix.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor

from hyperbolic_world_model.geometry.base import Manifold


def _gromov_products(dist: Tensor, base: int) -> Tensor:
    """Matrix ``G[i, j] = (i|j)_base``."""
    row = dist[base]  # d(., base)
    return 0.5 * (row.unsqueeze(1) + row.unsqueeze(0) - dist)


#: Elements of the ``(rows, n, n)`` intermediate allowed per chunk (16M float64 = 128 MB).
MAXMIN_BUDGET = 16_000_000


def _chunk_rows(n: int, budget: int = MAXMIN_BUDGET) -> int:
    """Rows per chunk so that ``rows * n * n`` stays within ``budget`` (at least one row)."""
    return max(1, min(n, budget // max(n * n, 1)))


def _max_min_product(a: Tensor, b: Tensor) -> Tensor:
    """``(A (*) B)[i, j] = max_k min(A[i, k], B[k, j])`` computed in row chunks to bound memory.

    The intermediate is ``(rows, n, n)``; the previous ``512 * 512 // n`` rule bounded ``rows * n``
    instead and allocated 1 GB at ``n = 500``.
    """
    n = a.shape[0]
    out = torch.empty_like(a)
    chunk = _chunk_rows(n)
    for start in range(0, n, chunk):
        sl = slice(start, start + chunk)
        out[sl] = torch.minimum(a[sl].unsqueeze(2), b.unsqueeze(0)).amax(dim=1)
    return out


def delta_hyperbolicity_from_distances(dist: Tensor, base: int | None = None) -> float:
    """Gromov delta of a finite metric space given its full pairwise distance matrix.

    Args:
        dist: ``(n, n)`` symmetric distance matrix with zero diagonal.
        base: index of the base point ``w``; defaults to ``0``.

    Returns:
        ``delta_w >= 0``. Exactly ``0`` (up to float error) for tree metrics.
    """
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"expected square distance matrix, got {tuple(dist.shape)}")
    dist = dist.to(torch.float64)
    base = 0 if base is None else base
    g = _gromov_products(dist, base)
    delta = (_max_min_product(g, g) - g).amax()
    return float(delta.clamp_min(0.0))


def _subsampled_distance_matrices(
    points: Tensor, manifold: Manifold, n_samples: int, n_trials: int, seed: int
) -> Iterator[Tensor]:
    """Yield the pairwise distance matrix of ``n_trials`` random subsamples of ``points``."""
    if points.ndim != 2:
        raise ValueError("points must be (N, d)")
    if n_samples < 2 or n_trials < 1:
        raise ValueError("n_samples must be >= 2 and n_trials >= 1")
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    for _ in range(n_trials):
        idx = rng.choice(n, size=min(n_samples, n), replace=False)
        sub = points[torch.as_tensor(idx, device=points.device)]
        with torch.no_grad():
            yield manifold.pairwise_dist(sub)


def delta_hyperbolicity(
    points: Tensor,
    manifold: Manifold,
    n_samples: int = 200,
    n_trials: int = 5,
    seed: int = 0,
) -> tuple[float, float]:
    """Estimate delta for a point cloud on ``manifold`` by repeated subsampling.

    Args:
        points: ``(N, d)`` points on ``manifold`` (for probing an encoder, its latents).
        manifold: geometry used to compute pairwise distances. Pass ``Euclidean()`` to probe raw
            encoder outputs, or the model's geometry to probe a hyperbolic head.
        n_samples: points per subsample (cubic cost, keep ``<= 500``).
        n_trials: number of subsamples; the mean and standard deviation are returned.
        seed: RNG seed for the subsampling.

    Returns:
        ``(mean_delta, std_delta)`` across trials.
    """
    deltas = [
        delta_hyperbolicity_from_distances(dist)
        for dist in _subsampled_distance_matrices(points, manifold, n_samples, n_trials, seed)
    ]
    arr = np.asarray(deltas)
    return float(arr.mean()), float(arr.std())


def relative_delta_hyperbolicity(
    points: Tensor,
    manifold: Manifold,
    n_samples: int = 200,
    n_trials: int = 5,
    seed: int = 0,
) -> tuple[float, float]:
    """Diameter-normalised delta ``2 * delta / diam`` in ``[0, 1]``, comparable across scales.

    Values near ``0`` indicate tree-like structure; ``1`` is maximally non-hyperbolic. Uses the
    same subsamples as :func:`delta_hyperbolicity` for the same ``seed``.
    """
    vals = []
    for dist in _subsampled_distance_matrices(points, manifold, n_samples, n_trials, seed):
        diam = float(dist.max())
        vals.append(0.0 if diam <= 0 else 2.0 * delta_hyperbolicity_from_distances(dist) / diam)
    arr = np.asarray(vals)
    return float(arr.mean()), float(arr.std())


__all__ = [
    "delta_hyperbolicity",
    "delta_hyperbolicity_from_distances",
    "relative_delta_hyperbolicity",
]
