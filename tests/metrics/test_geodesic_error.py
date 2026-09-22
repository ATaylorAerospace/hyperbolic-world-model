"""Geodesic rollout error: always ``manifold.dist``, never a Euclidean fallback."""

from __future__ import annotations

import ast
import importlib
import inspect

import pytest
import torch

from hyperbolic_world_model.geometry import Euclidean, Lorentz, Manifold, PoincareBall
from hyperbolic_world_model.metrics import (
    geodesic_error,
    geodesic_error_per_horizon,
    normalised_geodesic_error,
    normalised_geodesic_error_per_horizon,
    static_baseline_error,
)

MANIFOLDS = [
    pytest.param(Euclidean(), id="euclidean"),
    pytest.param(PoincareBall(c=-1.0), id="poincare"),
    pytest.param(PoincareBall(c=-0.5), id="poincare(c=-0.5)"),
    pytest.param(Lorentz(c=-1.0), id="lorentz"),
]


def _points(m: Manifold, *shape: int, scale: float = 0.6) -> torch.Tensor:
    v = torch.randn(*shape, dtype=torch.float64) * scale
    return m.expmap0(m.tangent0_from_euclidean(v))


@pytest.mark.parametrize("m", MANIFOLDS)
def test_error_is_the_manifold_distance(m: Manifold) -> None:
    pred, target = _points(m, 5, 3), _points(m, 5, 3)
    err = geodesic_error(m, pred, target)
    assert err.shape == (5,)
    assert torch.allclose(err, m.dist(pred, target))
    euclid = (pred - target).norm(dim=-1)
    if isinstance(m, Euclidean):
        assert torch.allclose(err, euclid)
    else:
        # Curved geometries must NOT reduce to the coordinate norm.
        assert not torch.allclose(err, euclid, rtol=1e-3)


def test_rejects_anything_that_is_not_a_manifold() -> None:
    x = torch.zeros(2, 3)
    for bad in (None, "euclidean", Euclidean().to_geoopt(), torch.cdist):
        with pytest.raises(TypeError, match="Manifold"):
            geodesic_error(bad, x, x)
        with pytest.raises(TypeError, match="Manifold"):
            static_baseline_error(bad, x[:, None], x[:, None])


def test_module_source_has_no_euclidean_fallback() -> None:
    """The only distance called in this module is ``manifold.dist`` (checked on the AST)."""
    module = importlib.import_module("hyperbolic_world_model.metrics.geodesic_error")
    tree = ast.parse(inspect.getsource(module))
    called_attrs, called_names, dist_receivers = set(), set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            called_attrs.add(node.func.attr)
            if node.func.attr == "dist" and isinstance(node.func.value, ast.Name):
                dist_receivers.add(node.func.value.id)
        elif isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
    forbidden = {
        "cdist",
        "norm",
        "vector_norm",
        "mse_loss",
        "pairwise_distance",
        "cosine_similarity",
    }
    assert not (called_attrs | called_names) & forbidden
    assert dist_receivers == {"manifold"}
    assert "isinstance" in called_names  # the Manifold type check


def test_shape_mismatch_and_bad_ranks_raise() -> None:
    m = Euclidean()
    with pytest.raises(ValueError, match="shape mismatch"):
        geodesic_error(m, torch.zeros(2, 3), torch.zeros(3, 3))
    with pytest.raises(ValueError, match="batch, horizon, d"):
        geodesic_error_per_horizon(m, torch.zeros(2, 3), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="batch, horizon, d"):
        normalised_geodesic_error_per_horizon(
            m, torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3)
        )
    with pytest.raises(ValueError, match="unknown reduce"):
        geodesic_error_per_horizon(m, torch.zeros(2, 4, 3), torch.zeros(2, 4, 3), reduce="max")
    with pytest.raises(ValueError, match="reduce='none'"):
        normalised_geodesic_error_per_horizon(
            m, torch.zeros(2, 4, 3), torch.zeros(2, 4, 3), torch.zeros(2, 3), reduce="none"
        )
    with pytest.raises(ValueError, match="broadcast"):
        static_baseline_error(m, torch.zeros(2, 5), torch.zeros(2, 4, 3))


@pytest.mark.parametrize("m", MANIFOLDS)
def test_per_horizon_reductions(m: Manifold) -> None:
    pred, target = _points(m, 6, 4, 3), _points(m, 6, 4, 3)
    full = geodesic_error(m, pred, target)
    assert torch.allclose(geodesic_error_per_horizon(m, pred, target), full.mean(0))
    assert torch.allclose(
        geodesic_error_per_horizon(m, pred, target, reduce="median"), full.median(0).values
    )
    assert torch.equal(geodesic_error_per_horizon(m, pred, target, reduce="none"), full)


@pytest.mark.parametrize("m", MANIFOLDS)
def test_static_predictor_scores_exactly_one_in_every_geometry(m: Manifold) -> None:
    context = _points(m, 6, 3)
    target = _points(m, 6, 4, 3)
    static = context.unsqueeze(1).expand_as(target)
    moved = static_baseline_error(m, context, target)
    assert moved.shape == (6, 4)
    assert torch.allclose(moved, m.dist(static, target))
    # The (..., 1, d) form of the context is accepted too.
    assert torch.allclose(static_baseline_error(m, context.unsqueeze(1), target), moved)
    norm = normalised_geodesic_error(m, static, target, context)
    assert norm.shape == (6, 4) and torch.allclose(norm, torch.ones_like(norm))
    per_h = normalised_geodesic_error_per_horizon(m, static, target, context)
    assert per_h.shape == (4,) and torch.allclose(per_h, torch.ones_like(per_h))
    # A perfect predictor scores 0 (up to the hyperboloid's float self-distance noise).
    assert torch.all(normalised_geodesic_error(m, target, target, context) < 1e-6)


def test_normalised_error_handles_targets_that_did_not_move() -> None:
    m = Euclidean()
    context = torch.zeros(3, 2)
    target = torch.zeros(3, 2, 2)  # nothing moved: denominator is clamped, no inf/nan
    pred = torch.ones(3, 2, 2)
    out = normalised_geodesic_error(m, pred, target, context)
    assert torch.isfinite(out).all()
    assert torch.isfinite(normalised_geodesic_error_per_horizon(m, pred, target, context)).all()
