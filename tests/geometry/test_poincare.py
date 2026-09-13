"""Poincaré ball: construction, Möbius helpers, gradcheck, and float32 vs float64 at 0.999 R."""

from __future__ import annotations

import math

import geoopt
import pytest
import torch

from hyperbolic_world_model.geometry import PoincareBall, build_manifold
from hyperbolic_world_model.geometry.utils import BOUNDARY_EPS
from tests.geometry.helpers import random_points, random_tangents

CURVATURES = [-0.5, -1.0, -2.0]


@pytest.mark.parametrize("c", CURVATURES)
def test_construction_radius_and_geoopt_parameter(c: float) -> None:
    m = PoincareBall(c=c)
    assert m.curvature == c and m.name == "poincare" and m.ambient_dim_offset == 0
    assert m.radius == pytest.approx(1 / math.sqrt(-c))
    g = m.to_geoopt()
    assert (
        isinstance(g, geoopt.PoincareBall)
        and float(g.c) == pytest.approx(-c)
        and g.k.dtype == torch.float64
    )
    assert isinstance(build_manifold({"name": "poincare", "curvature": c}), PoincareBall)
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="c < 0"):
            PoincareBall(c=bad)


@pytest.mark.parametrize("c", CURVATURES)
def test_lambda_x_closed_form(c: float, dtype: torch.dtype) -> None:
    m = PoincareBall(c=c)
    x = random_points(m, 16, 4, dtype)
    expected = 2 / (1 + c * (x * x).sum(-1, keepdim=True))
    assert torch.allclose(m.lambda_x(x), expected, rtol=1e-5)
    assert m.lambda_x(x, keepdim=False).shape == (16,)
    assert torch.allclose(
        m.lambda_x(torch.zeros(1, 4, dtype=dtype)), torch.full((1, 1), 2.0, dtype=dtype)
    )


@pytest.mark.parametrize("c", CURVATURES)
def test_mobius_add_identity_inverse_and_geoopt(c: float, dtype: torch.dtype) -> None:
    m = PoincareBall(c=c)
    x, y = random_points(m, 16, 3, dtype), random_points(m, 16, 3, dtype)
    zero = torch.zeros_like(x)
    a = 1e-6 if dtype == torch.float32 else 1e-12
    assert torch.allclose(m.mobius_add(zero, x), x, atol=a)
    assert torch.allclose(m.mobius_add(x, zero), x, atol=a)
    assert torch.allclose(m.mobius_add(m.mobius_neg(x), x), zero, atol=a)
    assert torch.equal(m.mobius_neg(x), -x)
    assert torch.allclose(m.mobius_add(x, y), m.to_geoopt().mobius_add(x, y))
    # Möbius addition is not commutative in general.
    assert not torch.allclose(m.mobius_add(x, y), m.mobius_add(y, x))
    # Left cancellation law: (-x) (+) (x (+) y) = y.
    assert torch.allclose(
        m.mobius_add(-x, m.mobius_add(x, y)), y, atol=1e-5 if dtype == torch.float32 else 1e-10
    )


@pytest.mark.parametrize("c", CURVATURES)
def test_gyration_is_an_isometry_and_trivial_at_zero(c: float) -> None:
    m = PoincareBall(c=c)
    dt = torch.float64
    u, v = random_points(m, 16, 3, dt), random_points(m, 16, 3, dt)
    w = torch.randn(16, 3, dtype=dt)
    zero = torch.zeros_like(u)
    assert torch.allclose(m.gyration(u, zero, w), w, atol=1e-10)
    assert torch.allclose(m.gyration(zero, v, w), w, atol=1e-10)
    assert torch.allclose(m.gyration(u, v, w).norm(dim=-1), w.norm(dim=-1), atol=1e-10)
    assert torch.allclose(m.gyration(u, v, w), m.to_geoopt().gyration(u, v, w))
    # Gyration defines parallel transport: P = (lambda_x / lambda_y) gyr[y, -x].
    x, y = random_points(m, 16, 3, dt), random_points(m, 16, 3, dt)
    assert torch.allclose(
        m.ptransp(x, y, w), m.lambda_x(x) / m.lambda_x(y) * m.gyration(y, -x, w), atol=1e-10
    )


@pytest.mark.parametrize("c", CURVATURES)
def test_ptransp_preserves_conformal_norm(c: float, dtype: torch.dtype) -> None:
    m = PoincareBall(c=c)
    x, y = random_points(m, 16, 4, dtype), random_points(m, 16, 4, dtype)
    u = random_tangents(m, x)
    pu = m.ptransp(x, y, u)
    assert torch.allclose(
        m.lambda_x(x) * u.norm(dim=-1, keepdim=True),
        m.lambda_x(y) * pu.norm(dim=-1, keepdim=True),
        rtol=1e-4,
    )


def test_distance_closed_form_from_origin() -> None:
    m = PoincareBall(c=-1.0)
    x = random_points(m, 16, 3, torch.float64)
    assert torch.allclose(m.dist0(x), 2 * torch.atanh(x.norm(dim=-1)), atol=1e-10)
    assert torch.allclose(m.dist(torch.zeros_like(x), x), m.dist0(x), atol=1e-10)


@pytest.mark.parametrize("c", CURVATURES)
def test_gradients_match_finite_differences(c: float) -> None:
    m = PoincareBall(c=c)
    dt = torch.float64
    x = random_points(m, 4, 3, dt).requires_grad_(True)
    y = random_points(m, 4, 3, dt).requires_grad_(True)
    u = random_tangents(m, x.detach(), max_len=0.5).requires_grad_(True)
    v = (torch.randn(4, 3, dtype=dt) * 0.3).requires_grad_(True)
    kw = dict(eps=1e-6, atol=1e-5)
    assert torch.autograd.gradcheck(lambda a, b: m.dist(a, b), (x, y), **kw)
    assert torch.autograd.gradcheck(lambda a, t: m.expmap(a, t), (x, u), **kw)
    assert torch.autograd.gradcheck(lambda a, b: m.logmap(a, b), (x, y), **kw)
    assert torch.autograd.gradcheck(lambda t: m.expmap0(t), (v,), **kw)
    assert torch.autograd.gradcheck(lambda b: m.logmap0(b), (y,), **kw)
    assert torch.autograd.gradcheck(lambda a, b, t: m.ptransp(a, b, t), (x, y, u), **kw)
    assert torch.autograd.gradcheck(lambda a: m.dist0(a), (x,), **kw)


def test_proj_clips_to_boundary(dtype: torch.dtype) -> None:
    m = PoincareBall(c=-1.0)
    far = torch.randn(8, 3, dtype=dtype) * 100
    p = m.proj(far)
    assert torch.all(p.norm(dim=-1) < m.radius) and m.check_point(p).all()
    assert torch.allclose(
        p.norm(dim=-1),
        torch.full((8,), (1 - BOUNDARY_EPS[dtype]) * m.radius, dtype=dtype),
        rtol=1e-5,
    )
    inside = torch.randn(8, 3, dtype=dtype) * 0.1
    assert torch.equal(m.proj(inside), inside)
    assert not m.check_point(far).any()
    assert m.check_point(inside, atol=0.5).all() and not m.check_point(inside * 60, atol=0.5).any()


@pytest.mark.parametrize("c", [-1.0, -2.0])
def test_float32_vs_float64_at_0999_of_the_boundary_radius(c: float) -> None:
    """Points at ``0.999 * radius``, i.e. inside the float32 clip margin (``BOUNDARY_EPS = 4e-3``).

    Measured contract:
    * quantities that do not go through Möbius addition (``dist0``, ``logmap0``, ``lambda_x``)
      agree with float64 to 1e-4;
    * pairwise ``dist``/``logmap`` between two boundary points lose ~1% in float32;
    * everything stays finite, including gradients;
    * the float64 exp/log round trip still holds to 1e-5, while the float32 round trip cannot,
      because ``expmap`` projects back to ``(1 - 4e-3) * radius``; float32 users must keep
      latents inside that radius (``HyperbolicHead.max_step`` / ``embed_scale`` do).
    """
    m = PoincareBall(c=c)
    r = 0.999 * m.radius
    g = torch.Generator().manual_seed(7)
    dirn = torch.randn(16, 4, dtype=torch.float64, generator=g)
    dirn = dirn / dirn.norm(dim=-1, keepdim=True)
    a64, b64, inner64 = r * dirn, r * torch.roll(dirn, 1, 0), 0.5 * m.radius * dirn
    a32, b32, inner32 = a64.float(), b64.float(), inner64.float()

    def close(x32: torch.Tensor, x64: torch.Tensor, rtol: float) -> bool:
        return torch.allclose(x32.double(), x64, rtol=rtol, atol=1e-6)

    assert close(m.dist0(a32), m.dist0(a64), 1e-4)
    assert close(m.logmap0(a32), m.logmap0(a64), 1e-4)
    assert close(m.lambda_x(a32), m.lambda_x(a64), 1e-3)
    assert close(m.dist(a32, inner32), m.dist(a64, inner64), 1e-4)
    assert close(m.dist(a32, b32), m.dist(a64, b64), 3e-2)
    assert close(m.dist(a32, -a32), m.dist(a64, -a64), 3e-2)
    assert close(m.logmap(a32, b32), m.logmap(a64, b64), 3e-2)
    assert torch.all(m.dist(a64, b64) > 2 * m.dist0(a64) - 5)  # boundary pairs are far apart

    for fn in (
        m.dist(a32, b32),
        m.logmap(a32, b32),
        m.expmap(a32, m.logmap(a32, b32)),
        m.ptransp(a32, b32, m.logmap(a32, b32)),
    ):
        assert torch.isfinite(fn).all()
    aa = a32.clone().requires_grad_(True)
    m.dist(aa, b32).sum().backward()
    assert torch.isfinite(aa.grad).all()

    rt64 = m.expmap(a64, m.logmap(a64, b64))
    assert torch.all(m.dist(rt64, b64) < 1e-5)
    rt32 = m.expmap(a32, m.logmap(a32, b32))
    assert torch.all(rt32.norm(dim=-1) <= (1 - BOUNDARY_EPS[torch.float32]) * m.radius + 1e-6)
    assert torch.all(
        m.proj(b32).norm(dim=-1) < b32.norm(dim=-1)
    )  # b32 itself is beyond the float32 clip


@pytest.mark.parametrize("c", [-1.0, -2.0])
def test_float32_round_trip_holds_inside_the_clip_margin(c: float) -> None:
    """At ``0.99 * radius`` (outside the 4e-3 margin) the float32 exp/log round trip is accurate."""
    m = PoincareBall(c=c)
    g = torch.Generator().manual_seed(3)
    dirn = torch.randn(16, 4, generator=g)
    dirn = dirn / dirn.norm(dim=-1, keepdim=True)
    a, b = 0.99 * m.radius * dirn, 0.99 * m.radius * torch.roll(dirn, 1, 0)
    rt = m.expmap(a, m.logmap(a, b))
    assert torch.all(m.dist(rt, b) < 5e-2 * m.dist(a, b))
