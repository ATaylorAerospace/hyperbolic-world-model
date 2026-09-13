"""Lorentz hyperboloid: construction, Minkowski inner product, own transport vs geoopt, isometry
to the ball, gradcheck, and float32 vs float64 at the image of 0.999 R."""

from __future__ import annotations

import math

import geoopt
import pytest
import torch
from tests.geometry.helpers import random_points, random_tangents

from hyperbolic_world_model.geometry import Lorentz, PoincareBall, build_manifold
from hyperbolic_world_model.geometry.lorentz import (
    lorentz_to_poincare,
    minkowski_inner,
    poincare_to_lorentz,
)

CURVATURES = [-0.5, -1.0, -2.0]


@pytest.mark.parametrize("c", CURVATURES)
def test_construction_k_and_geoopt_parameter(c: float) -> None:
    m = Lorentz(c=c)
    assert m.curvature == c and m.name == "lorentz" and m.ambient_dim_offset == 1
    assert m.k == pytest.approx(-1 / c) and m.sqrt_k == pytest.approx(math.sqrt(-1 / c))
    g = m.to_geoopt()
    assert (
        isinstance(g, geoopt.Lorentz)
        and float(g.k) == pytest.approx(m.k)
        and g.k.dtype == torch.float64
    )
    assert isinstance(build_manifold({"name": "lorentz", "curvature": c}), Lorentz)
    for bad in (0.0, 2.0):
        with pytest.raises(ValueError, match="c < 0"):
            Lorentz(c=bad)


def test_minkowski_inner_and_inner(dtype: torch.dtype) -> None:
    u = torch.tensor([[1.0, 2.0, 3.0]], dtype=dtype)
    v = torch.tensor([[4.0, 5.0, 6.0]], dtype=dtype)
    expected = torch.tensor([[-4.0 + 10.0 + 18.0]], dtype=dtype)
    assert torch.equal(minkowski_inner(u, v), expected)
    assert minkowski_inner(u, v, keepdim=False).shape == (1,)
    m = Lorentz(c=-1.0)
    assert torch.equal(m.inner(u, v), expected) and m.inner(u, v, keepdim=False).shape == (1,)
    x = random_points(m, 8, 4, dtype)
    assert torch.allclose(
        m.inner(x, x, keepdim=False), torch.full((8,), -m.k, dtype=dtype), atol=1e-5
    )


@pytest.mark.parametrize("c", CURVATURES)
def test_points_and_origin_lie_on_the_hyperboloid(c: float, dtype: torch.dtype) -> None:
    m = Lorentz(c=c)
    x = random_points(m, 32, 5, dtype)
    assert m.check_point(x).all()
    assert torch.all(x[:, 0] > 0)
    o = m.origin(3, 6, dtype=dtype)
    assert torch.equal(o[:, 0], torch.full((3,), m.sqrt_k, dtype=dtype)) and torch.all(
        o[:, 1:] == 0
    )
    assert torch.allclose(m.inner(o, o, keepdim=False), torch.full((3,), -m.k, dtype=dtype))
    assert not m.check_point(x + 1.0).any()
    assert m.check_point(x + 1e-3, atol=1e-1).all()


@pytest.mark.parametrize("c", CURVATURES)
def test_logmap_is_tangent_and_dist_is_its_length(c: float) -> None:
    m = Lorentz(c=c)
    x, y = random_points(m, 16, 4, torch.float64), random_points(m, 16, 4, torch.float64)
    u = m.logmap(x, y)
    assert torch.allclose(
        m.inner(x, u, keepdim=False), torch.zeros(16, dtype=torch.float64), atol=1e-10
    )
    assert torch.allclose(m.dist(x, y), m.inner(u, u, keepdim=False).sqrt(), atol=1e-9)


@pytest.mark.parametrize("c", CURVATURES)
def test_own_ptransp_matches_geoopt_and_is_finite_at_coincident_points(
    c: float, dtype: torch.dtype
) -> None:
    m = Lorentz(c=c)
    x, y = random_points(m, 16, 4, dtype), random_points(m, 16, 4, dtype)
    v = random_tangents(m, x)
    w = m.ptransp(x, y, v)
    assert torch.allclose(
        w, m.to_geoopt().transp(x, y, v), atol=1e-4 if dtype == torch.float32 else 1e-10
    )
    assert torch.allclose(
        m.inner(y, w, keepdim=False),
        torch.zeros(16, dtype=dtype),
        atol=1e-4 if dtype == torch.float32 else 1e-10,
    )
    assert torch.allclose(m.inner(v, v), m.inner(w, w), rtol=1e-4)
    same = m.ptransp(x, x, v)
    assert torch.isfinite(same).all() and torch.allclose(
        same, v, atol=1e-5 if dtype == torch.float32 else 1e-12
    )


@pytest.mark.parametrize("c", CURVATURES)
def test_isometry_to_poincare_ball(c: float) -> None:
    lor, ball = Lorentz(c=c), PoincareBall(c=c)
    xb, yb = (ball.expmap0(torch.randn(32, 4, dtype=torch.float64) * 0.5) for _ in range(2))
    xl, yl = poincare_to_lorentz(xb, c), poincare_to_lorentz(yb, c)
    assert lor.check_point(xl).all() and lor.check_point(yl).all()
    assert torch.allclose(lorentz_to_poincare(xl, c), xb, atol=1e-12)
    assert torch.allclose(lor.dist(xl, yl), ball.dist(xb, yb), atol=1e-9)
    assert torch.allclose(lor.dist0(xl), ball.dist0(xb), atol=1e-9)
    assert torch.allclose(
        lorentz_to_poincare(lor.origin(1, 5, dtype=torch.float64), c),
        torch.zeros(1, 4, dtype=torch.float64),
    )
    # Geodesic midpoints correspond under the isometry too.
    assert torch.allclose(
        lorentz_to_poincare(lor.geodesic(xl, yl, 0.5), c), ball.geodesic(xb, yb, 0.5), atol=1e-9
    )


@pytest.mark.parametrize("c", CURVATURES)
def test_gradients_match_finite_differences(c: float) -> None:
    m = Lorentz(c=c)
    dt = torch.float64
    x = random_points(m, 4, 3, dt).requires_grad_(True)
    y = random_points(m, 4, 3, dt).requires_grad_(True)
    u = random_tangents(m, x.detach(), max_len=0.5).requires_grad_(True)
    v = m.tangent0_from_euclidean(torch.randn(4, 3, dtype=dt) * 0.3).requires_grad_(True)
    kw = dict(eps=1e-6, atol=1e-5)
    assert torch.autograd.gradcheck(lambda a, b: m.dist(a, b), (x, y), **kw)
    assert torch.autograd.gradcheck(lambda a, t: m.expmap(a, t), (x, u), **kw)
    assert torch.autograd.gradcheck(lambda a, b: m.logmap(a, b), (x, y), **kw)
    assert torch.autograd.gradcheck(lambda t: m.expmap0(t), (v,), **kw)
    assert torch.autograd.gradcheck(lambda b: m.logmap0(b), (y,), **kw)
    assert torch.autograd.gradcheck(lambda a, b, t: m.ptransp(a, b, t), (x, y, u), **kw)
    assert torch.autograd.gradcheck(lambda a: m.dist0(a), (x,), **kw)


def test_tangent_lift_and_euclidean_projection() -> None:
    m = Lorentz(c=-1.0)
    v = torch.randn(8, 3)
    u = m.tangent0_from_euclidean(v)
    assert u.shape == (8, 4) and torch.all(u[:, 0] == 0)
    assert torch.equal(m.euclidean_from_tangent0(u), v)
    o = m.origin(8, 4)
    assert torch.equal(m.proj_tan(o, u), u)


@pytest.mark.parametrize("c", [-1.0, -2.0])
def test_float32_vs_float64_at_the_image_of_0999_radius(c: float) -> None:
    """Hyperboloid images of ball points at ``0.999 * radius`` (``x_0 ~ 1e3`` for ``c = -1``).

    Measured contract:
    * ``dist0`` and pairwise ``dist`` agree with float64 to 1e-5 (no Möbius cancellation here);
    * float32 self-distance noise is bounded by ``2e-3 * x_0`` because ``<x, x>_L`` cancels two
      terms of size ``x_0^2``;
    * ``logmap`` between two such points loses up to 20% in float32;
    * everything stays finite and on-manifold, including gradients;
    * even in float64 the exp/log round trip degrades exponentially with distance (below 1e-5 at
      4 units from the origin, below 1 at the 0.999-radius image), so hyperboloid latents should
      stay within a few units of the origin.
    """
    m = Lorentz(c=c)
    ball = PoincareBall(c=c)
    g = torch.Generator().manual_seed(7)
    dirn = torch.randn(16, 4, dtype=torch.float64, generator=g)
    dirn = dirn / dirn.norm(dim=-1, keepdim=True)
    a64 = poincare_to_lorentz(0.999 * ball.radius * dirn, c)
    b64 = poincare_to_lorentz(0.999 * ball.radius * torch.roll(dirn, 1, 0), c)
    a32, b32 = a64.float(), b64.float()
    assert m.check_point(a64).all() and m.check_point(a32).all()

    def close(x32: torch.Tensor, x64: torch.Tensor, rtol: float) -> bool:
        return torch.allclose(x32.double(), x64, rtol=rtol, atol=1e-6)

    assert close(m.dist0(a32), m.dist0(a64), 1e-5)
    assert close(m.dist(a32, b32), m.dist(a64, b64), 1e-5)
    assert torch.all(m.dist(a32, a32) <= 2e-3 * a32[:, 0] + 1e-3)
    assert close(m.logmap(a32, b32), m.logmap(a64, b64), 2e-1)
    for fn in (
        m.dist(a32, b32),
        m.logmap(a32, b32),
        m.expmap(a32, m.logmap(a32, b32)),
        m.ptransp(a32, b32, m.logmap(a32, b32)),
    ):
        assert torch.isfinite(fn).all()
    assert m.check_point(m.expmap(a32, m.logmap(a32, b32))).all()
    aa = a32.clone().requires_grad_(True)
    m.dist(aa, b32).sum().backward()
    assert torch.isfinite(aa.grad).all()

    rt64 = m.expmap(a64, m.logmap(a64, b64))
    assert torch.all(m.dist(rt64, b64) < 1.0)
    near = random_points(m, 16, 4, torch.float64, max_dist=4.0)
    near2 = random_points(m, 16, 4, torch.float64, max_dist=4.0)
    assert torch.all(m.dist(m.expmap(near, m.logmap(near, near2)), near2) < 1e-5)
