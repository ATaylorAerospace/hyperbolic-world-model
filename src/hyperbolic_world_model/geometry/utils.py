"""Numerical helpers shared by the manifolds and the predictor heads.

``artanh`` and ``arcosh`` are geoopt's well-tested, TorchScript-compiled versions re-exported under
one import path; the remaining helpers are ours because geoopt has no equivalent.
"""

from __future__ import annotations

import math

import torch
from geoopt.manifolds.lorentz.math import arcosh as _geoopt_arcosh
from geoopt.manifolds.stereographic.math import artanh as _geoopt_artanh
from torch import Tensor

# Distance from the Poincaré-ball boundary at which geoopt's ``project`` clips points, per dtype.
# Mirrors ``geoopt.manifolds.stereographic.math._project`` so tests and ``check_point`` agree
# with what ``proj`` actually does.
BOUNDARY_EPS: dict[torch.dtype, float] = {
    torch.float16: 4e-3,
    torch.bfloat16: 4e-3,
    torch.float32: 4e-3,
    torch.float64: 1e-5,
}

# Smallest positive value we divide by, per dtype.
MIN_NORM: dict[torch.dtype, float] = {
    torch.float16: 1e-4,
    torch.bfloat16: 1e-4,
    torch.float32: 1e-7,
    torch.float64: 1e-15,
}


def eps(dtype: torch.dtype) -> float:
    """Boundary clipping margin for ``dtype`` (falls back to the float32 value)."""
    return BOUNDARY_EPS.get(dtype, BOUNDARY_EPS[torch.float32])


def min_norm(dtype: torch.dtype) -> float:
    """Smallest norm we are willing to divide by for ``dtype``."""
    return MIN_NORM.get(dtype, MIN_NORM[torch.float32])


def reliable_radius(curvature: float, dtype: torch.dtype = torch.float32) -> float:
    """Largest geodesic distance from the origin representable at ``curvature`` in ``dtype``.

    The Poincaré ball of curvature ``c < 0`` clips points to ``(1 - eps) * radius`` (``eps`` from
    :data:`BOUNDARY_EPS`), so no latent can lie farther than
    ``(2 / sqrt(-c)) * artanh(1 - eps)`` from the origin whatever a config asks for: 6.2 units at
    ``c = -1`` in float32, 3.1 at ``c = -4``. The predictor heads cap their ``max_radius`` at this
    value, in every geometry, so the Poincaré and Lorentz heads at the same curvature keep
    identical guards. Flat space (``c >= 0``) has no bound and returns ``inf``.
    """
    if curvature >= 0:
        return math.inf
    return (2.0 / math.sqrt(-curvature)) * math.atanh(1.0 - eps(dtype))


def artanh(x: Tensor) -> Tensor:
    """Inverse hyperbolic tangent with the argument clamped to ``(-1 + 1e-7, 1 - 1e-7)`` (geoopt)."""
    return _geoopt_artanh(x)


def arcosh(x: Tensor) -> Tensor:
    """Inverse hyperbolic cosine, evaluated in float64 internally and safe at ``x = 1`` (geoopt)."""
    return _geoopt_arcosh(x)


def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = True) -> Tensor:
    """Euclidean norm clamped from below by :func:`min_norm` so it can be used as a divisor."""
    return x.norm(dim=dim, keepdim=keepdim).clamp_min(min_norm(x.dtype))


def clip_norm(x: Tensor, max_norm: float | Tensor, dim: int = -1) -> Tensor:
    """Rescale ``x`` along ``dim`` so its norm is at most ``max_norm``; a no-op when already smaller.

    Used by the hyperbolic head to bound tangent updates before ``expmap`` so ``tanh``/``sinh`` do
    not saturate, and by tests to construct points at a prescribed radius.
    """
    norm = x.norm(dim=dim, keepdim=True).clamp_min(min_norm(x.dtype))
    max_norm_t = torch.as_tensor(max_norm, dtype=x.dtype, device=x.device)
    scale = torch.where(norm > max_norm_t, max_norm_t / norm, torch.ones_like(norm))
    return x * scale


__all__ = [
    "BOUNDARY_EPS",
    "MIN_NORM",
    "arcosh",
    "artanh",
    "clip_norm",
    "eps",
    "min_norm",
    "reliable_radius",
    "safe_norm",
]
