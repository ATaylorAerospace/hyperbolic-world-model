"""Tests for geometry/utils.py: every helper, including the per-dtype tables."""

from __future__ import annotations

import pytest
import torch

from hyperbolic_world_model.geometry.utils import (
    BOUNDARY_EPS,
    MIN_NORM,
    arcosh,
    artanh,
    clip_norm,
    eps,
    min_norm,
    reliable_radius,
    safe_norm,
)


def test_eps_and_min_norm_tables() -> None:
    assert eps(torch.float32) == BOUNDARY_EPS[torch.float32] == 4e-3
    assert eps(torch.float64) == BOUNDARY_EPS[torch.float64] == 1e-5
    assert min_norm(torch.float32) == MIN_NORM[torch.float32] == 1e-7
    assert min_norm(torch.float64) == MIN_NORM[torch.float64] == 1e-15
    # Unknown dtypes fall back to the float32 entry rather than raising.
    assert eps(torch.int64) == eps(torch.float32) and min_norm(torch.int64) == min_norm(
        torch.float32
    )
    assert eps(torch.float64) < eps(torch.float32) and min_norm(torch.float64) < min_norm(
        torch.float32
    )


def test_artanh_matches_torch_inside_and_is_finite_at_the_boundary(dtype: torch.dtype) -> None:
    x = torch.linspace(-0.9, 0.9, 25, dtype=dtype)
    assert torch.allclose(artanh(x), torch.atanh(x), atol=1e-6)
    edge = torch.tensor([-1.0, 1.0, -2.0, 2.0], dtype=dtype)
    out = artanh(edge)
    assert torch.isfinite(out).all()
    assert out[1] > 8 and out[0] < -8  # clamped at 1 - 1e-7 -> artanh ~ 8.3
    a = torch.tensor([0.5, 0.999999], dtype=torch.float64, requires_grad=True)
    artanh(a).sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.allclose(a.grad[0], 1 / (1 - 0.25 * torch.ones((), dtype=torch.float64)))


def test_arcosh_matches_torch_and_is_finite_at_one(dtype: torch.dtype) -> None:
    x = torch.linspace(1.5, 50.0, 25, dtype=dtype)
    assert torch.allclose(arcosh(x), torch.acosh(x), atol=1e-5)
    edge = arcosh(torch.tensor([1.0, 0.5], dtype=dtype))
    assert torch.isfinite(edge).all() and edge[0] >= 0
    assert arcosh(x).dtype == dtype  # evaluated in float64 internally, returned in the input dtype
    a = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    arcosh(a).backward()
    assert torch.allclose(a.grad, 1 / torch.sqrt(torch.tensor([3.0], dtype=torch.float64)))


def test_safe_norm_never_returns_zero(dtype: torch.dtype) -> None:
    x = torch.zeros(4, 3, dtype=dtype)
    assert torch.all(safe_norm(x) == min_norm(dtype))
    y = torch.tensor([[3.0, 4.0]], dtype=dtype)
    assert torch.allclose(safe_norm(y), torch.tensor([[5.0]], dtype=dtype))
    assert safe_norm(y, keepdim=False).shape == (1,)
    assert safe_norm(torch.ones(2, 3, 4, dtype=dtype), dim=1).shape == (2, 1, 4)


def test_clip_norm_only_shrinks(dtype: torch.dtype) -> None:
    big = torch.randn(8, 3, dtype=dtype) * 10
    small = torch.randn(8, 3, dtype=dtype) * 0.01
    assert torch.allclose(clip_norm(big, 1.0).norm(dim=-1), torch.ones(8, dtype=dtype), atol=1e-6)
    assert torch.equal(clip_norm(small, 1.0), small)
    # Direction is preserved and a tensor max_norm is accepted.
    c = clip_norm(big, torch.tensor(2.0, dtype=dtype))
    cos = (c * big).sum(-1) / (c.norm(dim=-1) * big.norm(dim=-1))
    assert torch.allclose(cos, torch.ones(8, dtype=dtype), atol=1e-6)
    assert clip_norm(torch.ones(2, 5, 3, dtype=dtype) * 3, 1.0, dim=1).norm(dim=1).max() <= 1 + 1e-6


@pytest.mark.parametrize("fn", [artanh, arcosh])
def test_helpers_keep_shape(fn) -> None:
    x = torch.full((2, 3, 4), 1.2 if fn is arcosh else 0.3)
    assert fn(x).shape == x.shape


@pytest.mark.parametrize("c", [-0.5, -1.0, -2.0, -4.0])
def test_reliable_radius_is_where_the_ball_clip_lands(c: float, dtype: torch.dtype) -> None:
    """A tangent far beyond the boundary lands exactly at the clip radius; flat space is unbounded."""
    from hyperbolic_world_model.geometry import Euclidean, PoincareBall

    m = PoincareBall(c=c)
    huge = torch.zeros(3, dtype=dtype)
    huge[0] = 1e3
    landed = float(m.dist0(m.expmap0(huge)))
    assert landed == pytest.approx(reliable_radius(c, dtype), rel=1e-3)
    assert reliable_radius(c, dtype) < reliable_radius(c / 2, dtype)  # weaker curvature, larger
    assert reliable_radius(c, torch.float32) < reliable_radius(c, torch.float64)
    assert reliable_radius(Euclidean().curvature, dtype) == float("inf")
    assert reliable_radius(-1.0, torch.float32) == pytest.approx(6.21, abs=0.01)
