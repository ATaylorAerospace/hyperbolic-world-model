"""The flat baseline must satisfy the same contract as the curved geometries."""

from __future__ import annotations

import pytest
import torch

from hyperbolic_world_model.geometry import Euclidean, Manifold, build_manifold


def test_is_a_manifold_with_zero_curvature() -> None:
    m = Euclidean()
    assert isinstance(m, Manifold)
    assert m.curvature == 0.0 and m.ambient_dim_offset == 0
    with pytest.raises(ValueError):
        Euclidean(curvature=-1.0)
    assert isinstance(build_manifold({"name": "euclidean", "curvature": 0.0}), Euclidean)


def test_primitives_reduce_to_vector_arithmetic(dtype: torch.dtype) -> None:
    m = Euclidean()
    x, y, u = (torch.randn(16, 5, dtype=dtype) for _ in range(3))
    assert torch.equal(m.expmap(x, u), x + u)
    assert torch.equal(m.logmap(x, y), y - x)
    assert torch.allclose(m.dist(x, y), (x - y).norm(dim=-1))
    assert torch.allclose(m.sqdist(x, y), ((x - y) ** 2).sum(-1))
    assert torch.equal(m.proj(x), x)
    assert torch.equal(m.ptransp(x, y, u), u)
    assert torch.equal(m.expmap0(u), u) and torch.equal(m.logmap0(y), y)


def test_geodesic_and_pairwise() -> None:
    m = Euclidean()
    x, y = torch.zeros(4, 3), torch.ones(4, 3)
    assert torch.allclose(m.geodesic(x, y, 0.5), torch.full((4, 3), 0.5))
    pts = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    assert torch.allclose(m.pairwise_dist(pts), torch.tensor([[0.0, 5.0], [5.0, 0.0]]))


def test_to_geoopt_roundtrip() -> None:
    import geoopt

    assert isinstance(Euclidean().to_geoopt(), geoopt.Euclidean)
