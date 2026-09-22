"""The flat baseline: same contract as the curved geometries, reducing to vector arithmetic."""

from __future__ import annotations

import geoopt
import pytest
import torch

from hyperbolic_world_model.geometry import Euclidean, Manifold, build_manifold


def test_construction_and_curvature() -> None:
    m = Euclidean()
    assert isinstance(m, Manifold) and m.curvature == 0.0 and m.ambient_dim_offset == 0
    assert Euclidean(c=0.0).curvature == 0.0
    with pytest.raises(ValueError, match="curvature 0"):
        Euclidean(c=-1.0)
    assert isinstance(build_manifold({"name": "euclidean", "curvature": 0.0}), Euclidean)


def test_primitives_reduce_to_vector_arithmetic(dtype: torch.dtype) -> None:
    m = Euclidean()
    x, y, u = (torch.randn(16, 5, dtype=dtype) for _ in range(3))
    assert torch.equal(m.expmap(x, u), x + u)
    assert torch.equal(m.logmap(x, y), y - x)
    assert torch.allclose(m.dist(x, y), (x - y).norm(dim=-1))
    assert torch.allclose(m.sqdist(x, y), ((x - y) ** 2).sum(-1))
    assert torch.equal(m.proj(x), x)
    assert torch.equal(m.proj_tan(x, u), u)
    assert torch.equal(m.ptransp(x, y, u), u)
    assert torch.equal(m.egrad2rgrad(x, u), u)
    assert torch.equal(m.expmap0(u), u) and torch.equal(m.logmap0(y), y)
    assert torch.equal(m.origin(16, 5, dtype=dtype), torch.zeros(16, 5, dtype=dtype))
    assert torch.equal(m.tangent0_from_euclidean(u), u) and torch.equal(
        m.euclidean_from_tangent0(u), u
    )


def test_geodesic_pairwise_and_check_point() -> None:
    m = Euclidean()
    x, y = torch.zeros(4, 3), torch.ones(4, 3)
    assert torch.allclose(m.geodesic(x, y, 0.5), torch.full((4, 3), 0.5))
    pts = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    assert torch.allclose(m.pairwise_dist(pts), torch.tensor([[0.0, 5.0], [5.0, 0.0]]))
    big = torch.randn(64, 300)
    ref = (big[:, None] - big[None]).norm(dim=-1)
    assert torch.allclose(
        m.pairwise_dist(big), ref, atol=1e-4
    )  # cdist exact mode, no (n, n, d) broadcast
    assert torch.allclose(m.pairwise_dist(big[:5], big[:9]), ref[:5, :9], atol=1e-4)
    assert m.check_point(pts * 1e9).all()


def test_gradients_match_finite_differences() -> None:
    m = Euclidean()
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: m.dist(a, b), (x, y))
    assert torch.autograd.gradcheck(lambda a, b: m.expmap(a, m.logmap(a, b)), (x, y))


def test_to_geoopt_and_repr() -> None:
    m = Euclidean()
    assert isinstance(m.to_geoopt(), geoopt.Euclidean)
    assert repr(m) == "Euclidean(c=0.0)"
