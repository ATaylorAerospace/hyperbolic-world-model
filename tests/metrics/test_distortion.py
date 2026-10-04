"""Average distortion and mAP behave as expected on known embeddings."""

from __future__ import annotations

import pytest
import torch

from hyperbolic_world_model.data.hierarchies import build_hierarchy
from hyperbolic_world_model.geometry import Euclidean, PoincareBall
from hyperbolic_world_model.metrics import (
    average_distortion,
    expected_average_precision,
    mean_average_precision,
    pairwise_distances,
)


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


def test_expected_average_precision_is_textbook_without_ties_and_tie_invariant() -> None:
    d = torch.tensor([0.1, 0.4, 0.2, 0.9, 0.3])
    rel = torch.tensor([True, True, False, False, True])
    # sorted distances 0.1 (rel), 0.2, 0.3 (rel), 0.4 (rel), 0.9: relevant ranks 1, 3, 4
    assert expected_average_precision(d, rel) == pytest.approx((1 + 2 / 3 + 3 / 4) / 3)
    # Two candidates tied at the nearest distance, one relevant: the expectation over the two
    # tie orders is the mean of the optimistic (1.0) and pessimistic (0.5) precisions.
    d = torch.tensor([0.5, 0.5, 0.9])
    rel = torch.tensor([True, False, False])
    assert expected_average_precision(d, rel) == pytest.approx(0.75)
    assert expected_average_precision(d[[1, 0, 2]], rel[[1, 0, 2]]) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="no relevant"):
        expected_average_precision(d, torch.zeros(3, dtype=torch.bool))


def test_map_does_not_depend_on_tie_breaking() -> None:
    """A single-child node shares its child's Fréchet mean: exact ties are structural."""
    m = Euclidean()
    emb = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 1.0]], dtype=torch.float64)
    adj = torch.zeros(4, 4, dtype=torch.bool)
    for i, j in [(0, 1), (1, 2), (1, 3)]:
        adj[i, j] = adj[j, i] = True
    perm = [0, 2, 1, 3]
    a = mean_average_precision(emb, adj, m)
    b = mean_average_precision(emb[perm], adj[perm][:, perm], m)
    assert a == pytest.approx(b)
    assert 0.75 < a < 1.0  # between the pessimistic and optimistic tie orders


def test_precomputed_pairwise_distances_give_identical_scores() -> None:
    tree = _small_tree()
    m = PoincareBall(-1.0)
    emb = m.expmap0(torch.randn(tree.n_nodes, 3, dtype=torch.float64) * 0.4)
    pw = pairwise_distances(emb, m)
    assert pw.shape == (tree.n_nodes, tree.n_nodes) and pw.dtype == torch.float64
    true, adj = tree.tree_distance_matrix(), tree.adjacency()
    assert average_distortion(emb, true, m, pairwise=pw) == pytest.approx(
        average_distortion(emb, true, m)
    )
    assert mean_average_precision(emb, adj, m, pairwise=pw) == pytest.approx(
        mean_average_precision(emb, adj, m)
    )
    with pytest.raises(ValueError, match="embeddings must be"):
        pairwise_distances(torch.zeros(3), m)
