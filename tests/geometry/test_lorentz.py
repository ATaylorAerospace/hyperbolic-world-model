"""Contract tests for the Lorentz hyperboloid, including the isometry to the Poincaré ball."""

from __future__ import annotations

import pytest
import torch

from hyperbolic_world_model.geometry import Lorentz, PoincareBall
from hyperbolic_world_model.geometry.lorentz import (
    lorentz_to_poincare,
    minkowski_inner,
    poincare_to_lorentz,
)

CURVATURES = [-0.5, -1.0, -2.0]


def random_points(
    m: Lorentz, n: int, d: int, dtype: torch.dtype, max_dist: float = 1.5
) -> torch.Tensor:
    """Points at geodesic distance uniform in ``[0, max_dist]`` from the origin."""
    v = torch.randn(n, d, dtype=dtype)
    v = v / v.norm(dim=-1, keepdim=True) * (torch.rand(n, 1, dtype=dtype) * max_dist)
    return m.expmap0(m.tangent0_from_euclidean(v))


def random_tangents(m: Lorentz, x: torch.Tensor, max_len: float = 1.0) -> torch.Tensor:
    """Tangent vectors at ``x`` of Lorentz length uniform in ``[0, max_len]``."""
    n, d = x.shape[0], x.shape[1] - 1
    v = torch.randn(n, d, dtype=x.dtype)
    v = v / v.norm(dim=-1, keepdim=True) * (torch.rand(n, 1, dtype=x.dtype) * max_len)
    origin = m.origin(*x.shape, dtype=x.dtype)
    return m.ptransp(origin, x, m.tangent0_from_euclidean(v))


@pytest.mark.parametrize("curvature", CURVATURES)
def test_points_lie_on_hyperboloid(curvature: float, dtype: torch.dtype) -> None:
    m = Lorentz(curvature)
    x = random_points(m, 32, 5, dtype)
    assert m.check_point(x).all()
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    assert torch.allclose(
        minkowski_inner(x, x, keepdim=False), torch.full((32,), -m.k, dtype=dtype), atol=tol
    )


@pytest.mark.parametrize("curvature", CURVATURES)
def test_exp_log_inverse(curvature: float, dtype: torch.dtype) -> None:
    m = Lorentz(curvature)
    x, y = random_points(m, 32, 5, dtype), random_points(m, 32, 5, dtype)
    tol = 1e-4 if dtype == torch.float32 else 1e-9
    assert torch.allclose(m.expmap(x, m.logmap(x, y)), y, atol=tol)
    u = random_tangents(m, x)
    assert torch.allclose(m.logmap(x, m.expmap(x, u)), u, atol=tol)


@pytest.mark.parametrize("curvature", CURVATURES)
def test_logmap_is_tangent(curvature: float) -> None:
    m = Lorentz(curvature)
    x, y = random_points(m, 16, 4, torch.float64), random_points(m, 16, 4, torch.float64)
    u = m.logmap(x, y)
    assert torch.allclose(
        minkowski_inner(x, u, keepdim=False), torch.zeros(16, dtype=torch.float64), atol=1e-10
    )


@pytest.mark.parametrize("curvature", CURVATURES)
def test_distance_symmetric_and_triangle(curvature: float) -> None:
    m = Lorentz(curvature)
    dt = torch.float64
    x, y, z = (random_points(m, 64, 4, dt) for _ in range(3))
    assert torch.allclose(m.dist(x, y), m.dist(y, x))
    assert torch.all(m.dist(x, x) < 1e-6)
    assert torch.all(m.dist(x, z) <= m.dist(x, y) + m.dist(y, z) + 1e-9)


def test_dist_equals_tangent_norm_of_logmap() -> None:
    m = Lorentz(-1.0)
    x, y = random_points(m, 16, 4, torch.float64), random_points(m, 16, 4, torch.float64)
    u = m.logmap(x, y)
    assert torch.allclose(m.dist(x, y), minkowski_inner(u, u, keepdim=False).sqrt(), atol=1e-9)


@pytest.mark.parametrize("curvature", CURVATURES)
def test_ptransp_is_isometric_and_tangent(curvature: float) -> None:
    m = Lorentz(curvature)
    dt = torch.float64
    x, y = random_points(m, 16, 4, dt), random_points(m, 16, 4, dt)
    v = random_tangents(m, x)
    w = m.ptransp(x, y, v)
    assert torch.allclose(
        minkowski_inner(y, w, keepdim=False), torch.zeros(16, dtype=dt), atol=1e-9
    )
    assert torch.allclose(minkowski_inner(v, v), minkowski_inner(w, w), atol=1e-9)
    # Transporting the geodesic direction gives minus the reverse direction.
    assert torch.allclose(m.ptransp(x, y, m.logmap(x, y)), -m.logmap(y, x), atol=1e-8)


@pytest.mark.parametrize("curvature", CURVATURES)
def test_isometric_to_poincare_ball(curvature: float) -> None:
    lor, ball = Lorentz(curvature), PoincareBall(curvature)
    xb, yb = (ball.expmap0(torch.randn(32, 4, dtype=torch.float64) * 0.5) for _ in range(2))
    xl, yl = poincare_to_lorentz(xb, k=lor.k), poincare_to_lorentz(yb, k=lor.k)
    assert lor.check_point(xl).all()
    assert torch.allclose(lorentz_to_poincare(xl, k=lor.k), xb, atol=1e-10)
    assert torch.allclose(lor.dist(xl, yl), ball.dist(xb, yb), atol=1e-9)
    assert torch.allclose(lor.dist0(xl), ball.dist0(xb), atol=1e-9)


@pytest.mark.parametrize("curvature", CURVATURES)
def test_gradients_match_finite_differences(curvature: float) -> None:
    m = Lorentz(curvature)
    dt = torch.float64
    x = random_points(m, 4, 3, dt).requires_grad_(True)
    y = random_points(m, 4, 3, dt).requires_grad_(True)
    u = random_tangents(m, x.detach(), max_len=0.5).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a, b: m.dist(a, b), (x, y), eps=1e-6, atol=1e-5)
    assert torch.autograd.gradcheck(lambda a, v: m.expmap(a, v), (x, u), eps=1e-6, atol=1e-5)
    assert torch.autograd.gradcheck(lambda a, b: m.logmap(a, b), (x, y), eps=1e-6, atol=1e-5)


def test_float32_stable_far_from_origin() -> None:
    """At ~6 units from the origin (x_0 ~ cosh 6 ~ 200) float32 must stay finite and on-manifold.

    Documented limit: ``<x, x>_L`` cancels two terms of size ``x_0^2``, so float32 self-distance
    noise grows like ``x_0 * sqrt(eps32) ~ 1e-3 * x_0``. Keep float32 latents within ~5 units of
    the origin (``max_step``/``embed_scale``) or use float64 for the geometry.
    """
    m = Lorentz(-1.0)
    far = random_points(m, 16, 4, torch.float32, max_dist=6.0)
    near = random_points(m, 16, 4, torch.float32, max_dist=0.2)
    for fn in (m.dist(far, near), m.logmap(far, near), m.expmap(far, m.logmap(far, near))):
        assert torch.isfinite(fn).all()
    assert m.check_point(far).all()
    assert m.check_point(m.proj(far + 1e-3)).all()
    self_dist = m.dist(far, far)
    assert torch.isfinite(self_dist).all()
    assert torch.all(self_dist <= 2e-3 * far[:, 0] + 1e-3)
    # Float32 agrees with float64 on the same inputs to within the cancellation noise.
    d64 = m.dist(m.proj(far.double()), m.proj(near.double()))
    assert torch.allclose(
        m.dist(far, near).double(), d64, atol=2e-3 * float(far[:, 0].max()), rtol=1e-3
    )


def test_tangent_lift_roundtrip_and_offsets() -> None:
    m = Lorentz(-1.0)
    assert m.ambient_dim_offset == 1
    v = torch.randn(8, 3)
    u = m.tangent0_from_euclidean(v)
    assert u.shape == (8, 4) and torch.all(u[:, 0] == 0)
    assert torch.equal(m.euclidean_from_tangent0(u), v)
    assert torch.equal(m.origin(8, 4)[:, 0], torch.ones(8))
    with pytest.raises(ValueError):
        Lorentz(1.0)
