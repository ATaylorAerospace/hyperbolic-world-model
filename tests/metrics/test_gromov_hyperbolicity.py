"""delta is 0 for a tree metric and positive for a grid."""

from __future__ import annotations

import itertools

import torch

from hyperbolic_world_model.data.hierarchies import build_hierarchy
from hyperbolic_world_model.geometry import Euclidean
from hyperbolic_world_model.metrics import delta_hyperbolicity, relative_delta_hyperbolicity
from hyperbolic_world_model.metrics.gromov_hyperbolicity import (
    MAXMIN_BUDGET,
    _chunk_rows,
    _max_min_product,
    delta_hyperbolicity_from_distances,
)


def _tree_distances() -> torch.Tensor:
    meta = [
        {"embodiment": e, "task": t, "primitive": p}
        for e in ("arm_a", "arm_b")
        for t in ("t0", "t1", "t2")
        for p in ("reach", "grasp")
    ]
    return build_hierarchy(meta).tree_distance_matrix()


def test_tree_metric_has_zero_delta() -> None:
    d = _tree_distances()
    assert d.shape[0] == 1 + 2 + 6 + 12
    assert delta_hyperbolicity_from_distances(d) == 0.0
    for base in (3, 10):
        assert delta_hyperbolicity_from_distances(d, base=base) == 0.0


def test_tree_distances_match_reference_lca_loop() -> None:
    rng = torch.Generator().manual_seed(0)
    meta = [
        {
            "embodiment": f"e{int(torch.randint(0, 3, (1,), generator=rng))}",
            "task": f"t{int(torch.randint(0, 5, (1,), generator=rng))}",
            "primitive": f"p{i}",
        }
        for i in range(60)
    ]
    tree = build_hierarchy(meta)
    fast = tree.tree_distance_matrix()
    n = tree.n_nodes
    ref = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            ref[i, j] = tree.depth[i] + tree.depth[j] - 2 * tree.lca_depth(i, j)
    assert torch.equal(fast, ref)
    assert tree.ancestor_matrix().shape == (n, 4) and tree.ancestor_matrix()[0].tolist() == [
        0,
        -1,
        -1,
        -1,
    ]


def test_grid_has_positive_delta() -> None:
    n = 8
    pts = torch.tensor(list(itertools.product(range(n), range(n))), dtype=torch.float64)
    d = torch.cdist(pts, pts, p=1)  # L1 grid metric
    delta = delta_hyperbolicity_from_distances(d)
    assert delta > 1.0  # an 8x8 L1 grid has delta of order n/2
    d2 = torch.cdist(pts, pts)  # Euclidean grid metric
    assert delta_hyperbolicity_from_distances(d2) > 0.5


def test_max_min_product_chunking_bounds_memory_and_matches_dense() -> None:
    for n in (7, 300, 500, 2000):
        rows = _chunk_rows(n)
        assert 1 <= rows <= n and (rows * n * n <= MAXMIN_BUDGET or rows == 1)
    assert _chunk_rows(500) < 500  # the old rule used all 500 rows (1 GB intermediate)
    a = torch.rand(40, 40, dtype=torch.float64)
    dense = torch.minimum(a.unsqueeze(2), a.unsqueeze(0)).amax(dim=1)
    assert torch.equal(_max_min_product(a, a), dense)
    assert torch.equal(
        _max_min_product(a, a), torch.stack([_max_min_product(a, a)[i] for i in range(40)])
    )


def test_delta_is_scale_covariant() -> None:
    pts = torch.randn(40, 3, dtype=torch.float64)
    d = torch.cdist(pts, pts)
    assert (
        abs(delta_hyperbolicity_from_distances(3 * d) - 3 * delta_hyperbolicity_from_distances(d))
        < 1e-9
    )


def test_subsampled_estimator_on_point_clouds() -> None:
    m = Euclidean()
    grid = torch.tensor(list(itertools.product(range(12), range(12))), dtype=torch.float32)
    mean, std = delta_hyperbolicity(grid, m, n_samples=60, n_trials=4, seed=1)
    assert mean > 0 and std >= 0
    rel, _ = relative_delta_hyperbolicity(grid, m, n_samples=60, n_trials=4, seed=1)
    assert 0 < rel <= 1
    # Points on a line form a tree metric (a path): delta must be 0.
    line = torch.stack([torch.linspace(0, 10, 50), torch.zeros(50)], dim=1)
    mean_line, _ = delta_hyperbolicity(line, m, n_samples=50, n_trials=2)
    assert mean_line < 1e-6
