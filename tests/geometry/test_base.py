"""Contract tests every Manifold must satisfy, run over all registered geometries and dtypes.

Adding a geometry means adding it to ``MANIFOLDS`` below; nothing else in this file changes.
"""

from __future__ import annotations

import geoopt
import pytest
import torch

from hyperbolic_world_model.geometry import (
    MANIFOLDS,
    Euclidean,
    Lorentz,
    Manifold,
    PoincareBall,
    build_manifold,
)
from tests.geometry.helpers import random_points, random_tangents, tol

MANIFOLD_CASES = [
    pytest.param(Euclidean(), id="euclidean"),
    pytest.param(PoincareBall(c=-1.0), id="poincare(c=-1)"),
    pytest.param(PoincareBall(c=-0.5), id="poincare(c=-0.5)"),
    pytest.param(PoincareBall(c=-2.0), id="poincare(c=-2)"),
    pytest.param(Lorentz(c=-1.0), id="lorentz(c=-1)"),
    pytest.param(Lorentz(c=-0.5), id="lorentz(c=-0.5)"),
    pytest.param(Lorentz(c=-2.0), id="lorentz(c=-2)"),
]
D = 5


@pytest.fixture(params=MANIFOLD_CASES)
def m(request: pytest.FixtureRequest) -> Manifold:
    return request.param


# ---------------------------------------------------------------------- registry
def test_registry_covers_every_case_and_build_manifold_accepts_both_keys() -> None:
    assert set(MANIFOLDS) == {"euclidean", "poincare", "lorentz"}
    for case in MANIFOLD_CASES:
        inst = case.values[0]
        assert MANIFOLDS[inst.name] is type(inst)
        rebuilt = build_manifold({"name": inst.name, "curvature": inst.curvature})
        assert type(rebuilt) is type(inst) and rebuilt.curvature == inst.curvature
        assert build_manifold({"name": inst.name, "c": inst.curvature}).curvature == inst.curvature
    assert (
        build_manifold({"name": "poincare", "curvature": -0.25, "_target_": "ignored"}).curvature
        == -0.25
    )
    with pytest.raises(KeyError, match="unknown geometry"):
        build_manifold({"name": "spherical", "curvature": 1.0})
    with pytest.raises(TypeError):
        build_manifold({"name": "poincare", "c": -1.0, "curvature": -1.0})


def test_lambda0_is_the_origin_metric_scale(m: Manifold, dtype: torch.dtype) -> None:
    v = torch.randn(16, D, dtype=dtype)
    v = v / v.norm(dim=-1, keepdim=True) * 0.3
    y = m.expmap0(m.tangent0_from_euclidean(v))
    assert torch.allclose(m.dist0(y), torch.full((16,), m.lambda0 * 0.3, dtype=dtype), rtol=1e-4)
    assert m.lambda0 == (2.0 if isinstance(m, PoincareBall) else 1.0)


def test_curvature_name_offset_and_repr(m: Manifold) -> None:
    assert isinstance(m.curvature, float)
    assert (m.curvature == 0.0) == isinstance(m, Euclidean)
    assert m.ambient_dim_offset == (1 if isinstance(m, Lorentz) else 0)
    assert m.name in MANIFOLDS
    assert repr(m) == m.__repr__() == f"{type(m).__name__}(c={m.curvature})"


# ---------------------------------------------------------------------- primitives
def test_expmap_logmap_are_inverse(m: Manifold, dtype: torch.dtype) -> None:
    x, y = random_points(m, 32, D, dtype), random_points(m, 32, D, dtype)
    assert torch.allclose(m.expmap(x, m.logmap(x, y)), y, atol=tol(dtype, 1e-4, 1e-9))
    u = random_tangents(m, x)
    assert torch.allclose(m.logmap(x, m.expmap(x, u)), u, atol=tol(dtype, 1e-4, 1e-9))


def test_distance_is_a_metric_on_random_points(m: Manifold, dtype: torch.dtype) -> None:
    x, y, z = (random_points(m, 64, D, dtype) for _ in range(3))
    dxy, dyx = m.dist(x, y), m.dist(y, x)
    assert dxy.shape == (64,)
    assert torch.allclose(dxy, dyx, atol=tol(dtype, 1e-5, 1e-12))  # symmetry
    assert torch.all(dxy >= 0)
    assert torch.all(m.dist(x, x) <= tol(dtype, 5e-3, 1e-6))  # identity (hyperboloid noise in f32)
    assert torch.all(
        m.dist(x, z) <= m.dist(x, y) + m.dist(y, z) + tol(dtype, 1e-4, 1e-9)
    )  # triangle


def test_sqdist_dist0_and_pairwise_are_consistent_with_dist(
    m: Manifold, dtype: torch.dtype
) -> None:
    x, y = random_points(m, 16, D, dtype), random_points(m, 16, D, dtype)
    assert torch.allclose(m.sqdist(x, y), m.dist(x, y) ** 2, atol=tol(dtype, 1e-5, 1e-12))
    o = m.origin(*x.shape, dtype=dtype)
    assert torch.allclose(m.dist0(x), m.dist(o, x), atol=tol(dtype, 1e-5, 1e-12))
    pd = m.pairwise_dist(x, y)
    assert pd.shape == (16, 16)
    for i in (0, 5):
        assert torch.allclose(
            pd[i], m.dist(x[i : i + 1].expand_as(y), y), atol=tol(dtype, 1e-5, 1e-12)
        )
    assert torch.allclose(
        m.pairwise_dist(x),
        m.pairwise_dist(x).T,
        rtol=tol(dtype, 1e-4, 1e-10),
        atol=tol(dtype, 1e-5, 1e-12),
    )
    # Row-chunked evaluation (memory bound) equals the single broadcast.
    assert torch.allclose(m.pairwise_dist(x, y, chunk_rows=3), pd, atol=tol(dtype, 1e-5, 1e-12))
    m.pairwise_budget, saved = 10, m.pairwise_budget
    try:
        assert torch.allclose(m.pairwise_dist(x, y), pd, atol=tol(dtype, 1e-5, 1e-12))
    finally:
        m.pairwise_budget = saved


def test_origin_closed_forms_match_general_maps(m: Manifold, dtype: torch.dtype) -> None:
    u = m.tangent0_from_euclidean(torch.randn(16, D, dtype=dtype) * 0.5)
    o = m.origin(*u.shape, dtype=dtype)
    assert m.check_point(o).all()
    assert torch.allclose(m.expmap0(u), m.expmap(o, u), atol=tol(dtype, 1e-5, 1e-10))
    y = m.expmap0(u)
    assert torch.allclose(m.logmap0(y), m.logmap(o, y), atol=tol(dtype, 1e-5, 1e-10))
    assert torch.allclose(m.logmap0(m.expmap0(u)), u, atol=tol(dtype, 1e-4, 1e-10))


def test_geodesic_endpoints_and_midpoint(m: Manifold, dtype: torch.dtype) -> None:
    x, y = random_points(m, 8, D, dtype), random_points(m, 8, D, dtype)
    a = tol(dtype, 1e-4, 1e-9)
    assert torch.allclose(m.geodesic(x, y, 0.0), x, atol=a)
    assert torch.allclose(m.geodesic(x, y, 1.0), y, atol=a)
    mid = m.geodesic(x, y, 0.5)
    assert torch.allclose(m.dist(x, mid), m.dist(mid, y), atol=tol(dtype, 1e-4, 1e-9))
    assert torch.allclose(
        m.dist(x, mid) + m.dist(mid, y), m.dist(x, y), atol=tol(dtype, 1e-4, 1e-9)
    )
    t = torch.linspace(0, 1, 8, dtype=dtype)  # per-row tensor t
    assert torch.allclose(m.geodesic(x, y, t)[0], x[0], atol=a) and torch.allclose(
        m.geodesic(x, y, t)[-1], y[-1], atol=a
    )


def test_proj_is_idempotent_and_returns_points_on_the_manifold(
    m: Manifold, dtype: torch.dtype
) -> None:
    raw = torch.randn(32, D + m.ambient_dim_offset, dtype=dtype) * 3
    p = m.proj(raw)
    assert m.check_point(p).all()
    assert torch.allclose(m.proj(p), p, atol=tol(dtype, 1e-6, 1e-12))
    assert m.check_point(random_points(m, 32, D, dtype)).all()


def test_proj_tan_is_idempotent_and_logmap_is_tangent(m: Manifold, dtype: torch.dtype) -> None:
    x, y = random_points(m, 16, D, dtype), random_points(m, 16, D, dtype)
    u = m.proj_tan(x, torch.randn(16, D + m.ambient_dim_offset, dtype=dtype))
    assert torch.allclose(m.proj_tan(x, u), u, atol=tol(dtype, 1e-6, 1e-12))
    lg = m.logmap(x, y)
    assert torch.allclose(m.proj_tan(x, lg), lg, atol=tol(dtype, 1e-5, 1e-10))


def test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic(
    m: Manifold, dtype: torch.dtype
) -> None:
    x, y = random_points(m, 16, D, dtype), random_points(m, 16, D, dtype)
    u = random_tangents(m, x)
    assert torch.allclose(m.ptransp(x, x, u), u, atol=tol(dtype, 1e-5, 1e-12))
    # Transporting the geodesic direction from x to y gives minus the direction back.
    assert torch.allclose(
        m.ptransp(x, y, m.logmap(x, y)), -m.logmap(y, x), atol=tol(dtype, 1e-4, 1e-9)
    )
    # Transport is an isometry: geodesic lengths are preserved (measured via expmap of scaled vectors).
    w = m.ptransp(x, y, u)
    assert torch.allclose(
        m.dist(y, m.expmap(y, w)), m.dist(x, m.expmap(x, u)), atol=tol(dtype, 1e-4, 1e-9)
    )


def test_egrad2rgrad_is_tangent_and_matches_geoopt(m: Manifold, dtype: torch.dtype) -> None:
    x = random_points(m, 16, D, dtype)
    g = torch.randn_like(x)
    rg = m.egrad2rgrad(x, g)
    assert rg.shape == x.shape
    assert torch.allclose(m.proj_tan(x, rg), rg, atol=tol(dtype, 1e-5, 1e-10))
    assert torch.allclose(rg, m.to_geoopt().egrad2rgrad(x, g.clone()), atol=tol(dtype, 1e-6, 1e-12))
    if isinstance(m, Euclidean):
        assert torch.equal(rg, g)


def test_tangent0_lift_round_trip(m: Manifold, dtype: torch.dtype) -> None:
    v = torch.randn(8, D, dtype=dtype)
    u = m.tangent0_from_euclidean(v)
    assert u.shape == (8, D + m.ambient_dim_offset)
    assert torch.equal(m.euclidean_from_tangent0(u), v)
    o = m.origin(*u.shape, dtype=dtype)
    assert torch.allclose(m.proj_tan(o, u), u, atol=tol(dtype, 1e-6, 1e-12))


def test_to_geoopt_returns_a_matching_manifold(m: Manifold) -> None:
    g = m.to_geoopt()
    assert isinstance(g, geoopt.manifolds.base.Manifold)
    if isinstance(m, PoincareBall):
        assert isinstance(g, geoopt.PoincareBall) and float(g.c) == pytest.approx(-m.curvature)
    elif isinstance(m, Lorentz):
        assert isinstance(g, geoopt.Lorentz) and float(g.k) == pytest.approx(-1.0 / m.curvature)
    else:
        assert isinstance(g, geoopt.Euclidean)
    # It is usable as a ManifoldParameter's manifold.
    p = geoopt.ManifoldParameter(random_points(m, 2, D, torch.float32), manifold=g)
    assert m.check_point(p.data).all()


def test_check_point_rejects_points_off_the_manifold(m: Manifold) -> None:
    x = random_points(m, 8, D, torch.float64)
    assert m.check_point(x).all()
    if isinstance(m, Euclidean):
        assert m.check_point(x * 1e6).all()  # nothing is off-manifold in flat space
    elif isinstance(m, PoincareBall):
        outside = x / x.norm(dim=-1, keepdim=True) * 2 * m.radius
        assert not m.check_point(outside).any()
    else:
        assert not m.check_point(x * 5).any()  # <x, x>_L scales by 25, so it is no longer -k


def test_outputs_keep_input_dtype(m: Manifold, dtype: torch.dtype) -> None:
    x, y = random_points(m, 4, D, dtype), random_points(m, 4, D, dtype)
    for out in (
        m.expmap(x, m.logmap(x, y)),
        m.dist(x, y),
        m.proj(x),
        m.ptransp(x, y, m.logmap(x, y)),
        m.dist0(x),
    ):
        assert out.dtype == dtype
