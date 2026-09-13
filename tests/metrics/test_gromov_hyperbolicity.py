"""delta is 0 for a tree metric and positive for a grid."""

from __future__ import annotations

import itertools

import torch

from hyperbolic_world_model.data.hierarchies import build_hierarchy
from hyperbolic_world_model.geometry import Euclidean
from hyperbolic_world_model.metrics import delta_hyperbolicity, relative_delta_hyperbolicity
from hyperbolic_world_model.metrics.gromov_hyperbolicity import delta_hyperbolicity_from_distances


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


def test_grid_has_positive_delta() -> None:
    n = 8
    pts = torch.tensor(list(itertools.product(range(n), range(n))), dtype=torch.float64)
    d = torch.cdist(pts, pts, p=1)  # L1 grid metric
    delta = delta_hyperbolicity_from_distances(d)
    assert delta > 1.0  # an 8x8 L1 grid has delta of order n/2
    d2 = torch.cdist(pts, pts)  # Euclidean grid metric
    assert delta_hyperbolicity_from_distances(d2) > 0.5


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
