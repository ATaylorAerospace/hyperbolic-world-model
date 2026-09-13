"""Contract tests for the Poincaré ball.

Covers: exp/log inverse identity, distance symmetry, triangle inequality, gradient check against
finite differences, and float32 vs float64 stability near the boundary.
"""

from __future__ import annotations

import pytest
import torch

from hyperbolic_world_model.geometry import PoincareBall, build_manifold
from hyperbolic_world_model.geometry.utils import BOUNDARY_EPS

CURVATURES = [-0.5, -1.0, -2.0]


def random_points(
    m: PoincareBall, n: int, d: int, dtype: torch.dtype, max_dist: float = 1.0
) -> torch.Tensor:
    """Points at geodesic distance uniform in ``[0, max_dist]`` from the origin."""
    v = torch.randn(n, d, dtype=dtype)
    v = v / v.norm(dim=-1, keepdim=True) * (torch.rand(n, 1, dtype=dtype) * max_dist)
    return m.expmap0(v)


@pytest.mark.parametrize("curvature", CURVATURES)
def test_exp_log_inverse(curvature: float, dtype: torch.dtype) -> None:
    m = PoincareBall(curvature)
    x, y = random_points(m, 32, 6, dtype), random_points(m, 32, 6, dtype)
    tol = 1e-4 if dtype == torch.float32 else 1e-9
    assert torch.allclose(m.expmap(x, m.logmap(x, y)), y, atol=tol)
    # Tangent vectors of bounded *Riemannian* length: |u|_g = lambda_x |u| <= 1.
    u = torch.randn(32, 6, dtype=dtype)
    u = u / u.norm(dim=-1, keepdim=True) / m.lambda_x(x) * torch.rand(32, 1, dtype=dtype)
    assert torch.allclose(m.logmap(x, m.expmap(x, u)), u, atol=tol)


@pytest.mark.parametrize("curvature", CURVATURES)
def test_origin_closed_forms_match_general(curvature: float, dtype: torch.dtype) -> None:
    m = PoincareBall(curvature)
    u = torch.randn(16, 5, dtype=dtype) * 0.5
    o = m.origin(16, 5, dtype=dtype)
    tol = 1e-5 if dtype == torch.float32 else 1e-10
    assert torch.allclose(m.expmap0(u), m.expmap(o, u), atol=tol)
    y = m.expmap0(u)
    assert torch.allclose(m.logmap0(y), m.logmap(o, y), atol=tol)
    assert torch.allclose(m.dist0(y), m.dist(o, y), atol=tol)


@pytest.mark.parametrize("curvature", CURVATURES)
def test_distance_symmetric_and_zero_on_diagonal(curvature: float, dtype: torch.dtype) -> None:
    m = PoincareBall(curvature)
    x, y = random_points(m, 32, 4, dtype), random_points(m, 32, 4, dtype)
    tol = 1e-5 if dtype == torch.float32 else 1e-10
    assert torch.allclose(m.dist(x, y), m.dist(y, x), atol=tol)
    assert torch.all(m.dist(x, x) < 1e-3)
    assert torch.all(m.dist(x, y) >= 0)


@pytest.mark.parametrize("curvature", CURVATURES)
def test_triangle_inequality(curvature: float) -> None:
    m = PoincareBall(curvature)
    dt = torch.float64
    x, y, z = (random_points(m, 64, 4, dt) for _ in range(3))
    assert torch.all(m.dist(x, z) <= m.dist(x, y) + m.dist(y, z) + 1e-9)


def test_distance_matches_closed_form_from_origin() -> None:
    """d(0, x) = (2/sqrt(c)) artanh(sqrt(c) |x|)."""
    m = PoincareBall(-1.0)
    x = random_points(m, 16, 3, torch.float64)
    expected = 2 * torch.atanh(x.norm(dim=-1))
    assert torch.allclose(m.dist0(x), expected, atol=1e-10)


def test_mobius_add_identity_and_inverse() -> None:
    m = PoincareBall(-1.0)
    x = random_points(m, 8, 3, torch.float64)
    zero = torch.zeros_like(x)
    assert torch.allclose(m.mobius_add(zero, x), x)
    assert torch.allclose(m.mobius_add(x, zero), x)
    assert torch.allclose(m.mobius_add(-x, x), zero, atol=1e-12)


def test_ptransp_preserves_norm_in_metric() -> None:
    """Parallel transport is an isometry: lambda_x |u| = lambda_y |P u|."""
    m = PoincareBall(-1.0)
    x, y = random_points(m, 16, 4, torch.float64), random_points(m, 16, 4, torch.float64)
    u = torch.randn(16, 4, dtype=torch.float64)
    pu = m.ptransp(x, y, u)
    assert torch.allclose(
        m.lambda_x(x) * u.norm(dim=-1, keepdim=True),
        m.lambda_x(y) * pu.norm(dim=-1, keepdim=True),
        atol=1e-9,
    )


@pytest.mark.parametrize("curvature", CURVATURES)
def test_gradients_match_finite_differences(curvature: float) -> None:
    m = PoincareBall(curvature)
    dt = torch.float64
    x = random_points(m, 4, 3, dt).requires_grad_(True)
    y = random_points(m, 4, 3, dt).requires_grad_(True)
    u = (torch.randn(4, 3, dtype=dt) * 0.3).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda a, b: m.dist(a, b), (x, y), eps=1e-6, atol=1e-5)
    assert torch.autograd.gradcheck(lambda a, v: m.expmap(a, v), (x, u), eps=1e-6, atol=1e-5)
    assert torch.autograd.gradcheck(lambda a, b: m.logmap(a, b), (x, y), eps=1e-6, atol=1e-5)


def test_proj_clips_to_boundary(dtype: torch.dtype) -> None:
    m = PoincareBall(-1.0)
    far = torch.randn(8, 3, dtype=dtype) * 100
    p = m.proj(far)
    assert torch.all(p.norm(dim=-1) < m.radius)
    assert torch.all(p.norm(dim=-1) >= (1 - BOUNDARY_EPS[dtype]) * m.radius - 1e-6)
    inside = torch.randn(8, 3, dtype=dtype) * 0.1
    assert torch.equal(m.proj(inside), inside)


def test_float32_stable_near_boundary_vs_float64() -> None:
    """Float32 must agree with float64 while the Möbius sum stays outside the float32 clamp
    margin, and degrade gracefully (finite, bounded, monotone) once it enters it.

    Documented limit: with ``BOUNDARY_EPS[float32] = 4e-3`` the largest float32 distance the
    unit ball can represent is ``2 * artanh(1 - 4e-3) ~ 6.2``. Train in float32 only if latents
    stay well inside that (the hyperbolic head's ``max_step``/``embed_scale`` do this), or use
    Lorentz/float64.
    """
    m = PoincareBall(-1.0)
    direction = torch.randn(16, 4, dtype=torch.float64)
    direction = direction / direction.norm(dim=-1, keepdim=True)

    # Antipodal points at radius 0.9: |(-x) (+) y| = 0.9945 < 1 - 4e-3, so both dtypes agree.
    x64, y64 = direction * 0.9, -direction * 0.9
    x32, y32 = x64.float(), y64.float()
    d32, d64 = m.dist(x32, y32), m.dist(x64, y64)
    assert torch.isfinite(d32).all()
    assert torch.allclose(d64, 4 * torch.atanh(torch.tensor(0.9, dtype=torch.float64)))
    assert torch.allclose(d32.double(), d64, rtol=2e-2)
    assert torch.isfinite(m.logmap(x32, y32)).all()
    assert torch.isfinite(m.expmap(x32, m.logmap(x32, y32))).all()

    # Radius 0.99: the Möbius sum lands inside the clamp margin. Float32 must be finite, capped
    # at the documented maximum, and never exceed the float64 value; float64 keeps resolving.
    z64, w64 = direction * 0.99, -direction * 0.99
    z32, w32 = z64.float(), w64.float()
    dz32, dz64 = m.dist(z32, w32), m.dist(z64, w64)
    max_f32 = 2 * torch.atanh(torch.tensor(1 - BOUNDARY_EPS[torch.float32]))
    assert torch.isfinite(dz32).all() and torch.isfinite(dz64).all()
    assert torch.all(dz64 > d64) and torch.all(dz32 <= dz64.float())
    assert torch.all(dz32 >= d32) and torch.all(dz32 <= max_f32 + 1e-4)

    # Gradients through the clamped artanh stay finite in float32 at both radii.
    for a0, b0 in ((x32, y32), (z32, w32)):
        a = a0.clone().requires_grad_(True)
        m.dist(a, b0).sum().backward()
        assert torch.isfinite(a.grad).all()


def test_build_manifold_from_config() -> None:
    m = build_manifold({"name": "poincare", "curvature": -0.25})
    assert isinstance(m, PoincareBall)
    assert m.curvature == -0.25 and m.c == 0.25
    with pytest.raises(ValueError):
        PoincareBall(0.0)
    with pytest.raises(KeyError):
        build_manifold({"name": "spherical", "curvature": 1.0})
