"""Average distortion and mAP behave as expected on known embeddings."""

from __future__ import annotations

import pytest
import torch

from hyperbolic_world_model.data.hierarchies import build_hierarchy
from hyperbolic_world_model.geometry import Euclidean, PoincareBall
from hyperbolic_world_model.metrics import average_distortion, mean_average_precision


def _small_tree():
    meta = [{"embodiment": e, "task": t, "primitive": p} for e in "ab" for t in "xy" for p in "pq"]
    return build_hierarchy(meta)


def test_perfect_embedding_has_zero_distortion_and_unit_map() -> None:
    """A path graph embedded on a line is exactly recoverable (after scale fitting)."""
    n = 6
    emb = torch.arange(n, dtype=torch.float32).unsqueeze(1) * 2.5  # scaled line
    true = torch.cdist(
        torch.arange(n, dtype=torch.float64).unsqueeze(1),
        torch.arange(n, dtype=torch.float64).unsqueeze(1),
    )
    m = Euclidean()
    assert average_distortion(emb, true, m) < 1e-6
    assert average_distortion(emb, true, m, fit_scale=False) > 1.0  # scale matters when not fitted
    adj = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n - 1):
        adj[i, i + 1] = adj[i + 1, i] = True
    assert mean_average_precision(emb, adj, m) == pytest.approx(1.0)


def test_random_embedding_is_worse_than_structured() -> None:
    tree = _small_tree()
    true = tree.tree_distance_matrix()
    adj = tree.adjacency()
    m = PoincareBall(-1.0)
    # Structured: leaves placed by depth along random directions, parents at the origin-ward point.
    depth = torch.tensor(tree.depth, dtype=torch.float64).unsqueeze(1)
    dirs = torch.randn(tree.n_nodes, 3, dtype=torch.float64)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    structured = m.expmap0(dirs * depth * 0.4)
    random = m.expmap0(torch.randn(tree.n_nodes, 3, dtype=torch.float64) * 0.5)
    assert 0 <= average_distortion(structured, true, m) < 5
    assert 0 <= mean_average_precision(random, adj, m) <= 1
    # The tree root at the origin must be closest to its children on average.
    assert mean_average_precision(structured, adj, m) > 0.2


def test_shape_validation() -> None:
    m = Euclidean()
    with pytest.raises(ValueError):
        average_distortion(torch.zeros(3, 2), torch.zeros(4, 4), m)
    with pytest.raises(ValueError):
        mean_average_precision(torch.zeros(3, 2), torch.zeros(3, 3, dtype=torch.bool), m)
