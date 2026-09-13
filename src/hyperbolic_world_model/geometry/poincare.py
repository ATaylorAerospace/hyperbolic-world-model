"""Poincaré ball model of hyperbolic space.

The ball of radius ``1/sqrt(c)`` with metric ``g_x = lambda_x^2 I``, ``lambda_x = 2 / (1 - c|x|^2)``,
has constant sectional curvature ``-c``. We follow Ganea et al. (2018) for Möbius arithmetic and
geoopt for the numerical conventions. ``curvature`` in configs is the sectional curvature ``K < 0``
so that ``c = -K``; the default ``-1.0`` gives the unit ball.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from hyperbolic_world_model.geometry.base import Manifold
from hyperbolic_world_model.geometry.utils import artanh, clip_norm, eps, safe_norm, tanh


class PoincareBall(Manifold):
    """Poincaré ball of curvature ``curvature < 0``.

    Args:
        curvature: sectional curvature ``K``. Must be strictly negative. Swept in experiments.
    """

    name = "poincare"

    def __init__(self, curvature: float = -1.0) -> None:
        if curvature >= 0:
            raise ValueError(f"PoincareBall needs curvature < 0, got {curvature}")
        super().__init__(curvature=curvature)
        self.c = -float(curvature)
        self.sqrt_c = math.sqrt(self.c)
        self.radius = 1.0 / self.sqrt_c

    # ------------------------------------------------------------------ conformal factor
    def lambda_x(self, x: Tensor, keepdim: bool = True) -> Tensor:
        """Conformal factor ``2 / (1 - c |x|^2)``."""
        x2 = (x * x).sum(dim=-1, keepdim=keepdim)
        return 2.0 / (1.0 - self.c * x2).clamp_min(eps(x.dtype))

    # ------------------------------------------------------------------ Möbius arithmetic
    def mobius_add(self, x: Tensor, y: Tensor) -> Tensor:
        """Möbius addition ``x (+)_c y`` (non-commutative, non-associative)."""
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        c = self.c
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + c * c * x2 * y2
        return num / den.clamp_min(eps(x.dtype))

    def mobius_neg(self, x: Tensor) -> Tensor:
        """Möbius negation is ordinary negation."""
        return -x

    def gyration(self, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        """Thomas gyration ``gyr[u, v] w = -(u (+) v) (+) (u (+) (v (+) w))``."""
        return self.mobius_add(-self.mobius_add(u, v), self.mobius_add(u, self.mobius_add(v, w)))

    # ------------------------------------------------------------------ primitives
    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        u_norm = safe_norm(u)
        lam = self.lambda_x(x)
        second = tanh(self.sqrt_c * lam * u_norm / 2) * u / (self.sqrt_c * u_norm)
        return self.proj(self.mobius_add(x, second))

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        sub = self.mobius_add(-x, y)
        sub_norm = safe_norm(sub)
        lam = self.lambda_x(x)
        return (2.0 / (self.sqrt_c * lam)) * artanh(self.sqrt_c * sub_norm) * sub / sub_norm

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        sub_norm = self.mobius_add(-x, y).norm(dim=-1)
        return (2.0 / self.sqrt_c) * artanh(self.sqrt_c * sub_norm)

    def proj(self, x: Tensor) -> Tensor:
        """Boundary clipping: pull points with ``|x| >= radius`` back inside by ``eps``."""
        return clip_norm(x, (1.0 - eps(x.dtype)) * self.radius)

    def ptransp(self, x: Tensor, y: Tensor, u: Tensor) -> Tensor:
        """Parallel transport via gyration: ``P_{x->y}(u) = (lambda_x / lambda_y) gyr[y, -x] u``."""
        return (self.lambda_x(x) / self.lambda_x(y)) * self.gyration(y, -x, u)

    def proj_tan(self, x: Tensor, u: Tensor) -> Tensor:
        return u

    def origin(self, *shape: int, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
        return torch.zeros(*shape, dtype=dtype, device=device)

    def egrad2rgrad(self, x: Tensor, grad: Tensor) -> Tensor:
        return grad / self.lambda_x(x) ** 2

    # ------------------------------------------------------------------ closed forms at the origin
    def expmap0(self, u: Tensor) -> Tensor:
        u_norm = safe_norm(u)
        return self.proj(tanh(self.sqrt_c * u_norm) * u / (self.sqrt_c * u_norm))

    def logmap0(self, y: Tensor) -> Tensor:
        y_norm = safe_norm(y)
        return artanh(self.sqrt_c * y_norm) * y / (self.sqrt_c * y_norm)

    def dist0(self, x: Tensor) -> Tensor:
        return (2.0 / self.sqrt_c) * artanh(self.sqrt_c * x.norm(dim=-1))

    def check_point(self, x: Tensor, atol: float | None = None) -> Tensor:
        return x.norm(dim=-1) < self.radius

    def to_geoopt(self):
        import geoopt

        return geoopt.PoincareBall(c=self.c)


__all__ = ["PoincareBall"]
