"""Shared point/tangent samplers for the geometry contract tests (not a test module)."""

from __future__ import annotations

import torch
from torch import Tensor

from hyperbolic_world_model.geometry import Manifold


def random_points(m: Manifold, n: int, d: int, dtype: torch.dtype, max_dist: float = 1.5) -> Tensor:
    """``n`` points at geodesic distance uniform in ``[0, max_dist]`` from the origin, intrinsic dim ``d``."""
    v = torch.randn(n, d, dtype=dtype)
    v = v / v.norm(dim=-1, keepdim=True) * (torch.rand(n, 1, dtype=dtype) * max_dist)
    return m.expmap0(m.tangent0_from_euclidean(v))


def random_tangents(m: Manifold, x: Tensor, max_len: float = 1.0) -> Tensor:
    """Tangent vectors at ``x`` of Riemannian length uniform in ``[0, max_len]`` (transported from 0)."""
    n = x.shape[0]
    d = x.shape[1] - m.ambient_dim_offset
    v = torch.randn(n, d, dtype=x.dtype)
    v = v / v.norm(dim=-1, keepdim=True) * (torch.rand(n, 1, dtype=x.dtype) * max_len)
    origin = m.origin(*x.shape, dtype=x.dtype)
    return m.proj_tan(x, m.ptransp(origin, x, m.tangent0_from_euclidean(v)))


def tol(dtype: torch.dtype, f32: float, f64: float) -> float:
    return f32 if dtype == torch.float32 else f64
