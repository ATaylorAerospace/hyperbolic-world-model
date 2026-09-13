"""Lorentz (hyperboloid) model of hyperbolic space.

Points satisfy ``<x, x>_L = -k`` with ``x_0 > 0`` under the Minkowski inner product
``<x, y>_L = -x_0 y_0 + sum_i x_i y_i``. The space has sectional curvature ``-1/k``; configs give
``curvature = K`` and we set ``k = -1/K``. The ambient representation has one extra coordinate
(``ambient_dim_offset = 1``) compared with the Poincaré ball of the same curvature, to which it is
isometric via :func:`lorentz_to_poincare`.

The hyperboloid avoids the ``artanh`` boundary blow-up of the ball, which is why it is the
recommended geometry for optimisation (Nickel & Kiela, 2018); the ball remains useful for
visualisation and as an independent numerical cross-check.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from hyperbolic_world_model.geometry.base import Manifold
from hyperbolic_world_model.geometry.utils import arcosh, cosh, eps, min_norm, sinh


def minkowski_inner(u: Tensor, v: Tensor, keepdim: bool = True) -> Tensor:
    """Minkowski inner product ``-u_0 v_0 + sum_{i>0} u_i v_i`` along the last axis."""
    prod = u * v
    res = prod[..., 1:].sum(dim=-1, keepdim=keepdim) - prod[..., :1].sum(dim=-1, keepdim=keepdim)
    return res


class Lorentz(Manifold):
    """Hyperboloid of curvature ``curvature < 0`` (``k = -1 / curvature``)."""

    name = "lorentz"

    def __init__(self, curvature: float = -1.0) -> None:
        if curvature >= 0:
            raise ValueError(f"Lorentz needs curvature < 0, got {curvature}")
        super().__init__(curvature=curvature)
        self.k = -1.0 / float(curvature)
        self.sqrt_k = math.sqrt(self.k)

    @property
    def ambient_dim_offset(self) -> int:
        return 1

    # ------------------------------------------------------------------ primitives
    def inner(self, u: Tensor, v: Tensor, keepdim: bool = True) -> Tensor:
        """Minkowski inner product (exposed for metrics that want it directly)."""
        return minkowski_inner(u, v, keepdim=keepdim)

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        u_norm = self.inner(u, u).clamp_min(min_norm(u.dtype)).sqrt()
        theta = u_norm / self.sqrt_k
        y = cosh(theta) * x + self.sqrt_k * sinh(theta) * u / u_norm
        return self.proj(y)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        alpha = (-self.inner(x, y) / self.k).clamp_min(1.0 + min_norm(x.dtype))
        coef = arcosh(alpha) / (alpha * alpha - 1.0).clamp_min(min_norm(x.dtype)).sqrt()
        return self.proj_tan(x, coef * (y - alpha * x))

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        alpha = -self.inner(x, y, keepdim=False) / self.k
        return self.sqrt_k * arcosh(alpha)

    def proj(self, x: Tensor) -> Tensor:
        """Recompute the time-like coordinate so ``<x, x>_L = -k`` holds exactly."""
        spatial = x[..., 1:]
        x0 = (self.k + (spatial * spatial).sum(dim=-1, keepdim=True)).sqrt()
        return torch.cat([x0, spatial], dim=-1)

    def ptransp(self, x: Tensor, y: Tensor, u: Tensor) -> Tensor:
        """``P_{x->y}(u) = u + <y, u>_L / (k - <x, y>_L) (x + y)``."""
        denom = (self.k - self.inner(x, y)).clamp_min(min_norm(x.dtype))
        return u + self.inner(y, u) / denom * (x + y)

    def proj_tan(self, x: Tensor, u: Tensor) -> Tensor:
        return u + self.inner(x, u) / self.k * x

    def origin(self, *shape: int, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
        o = torch.zeros(*shape, dtype=dtype, device=device)
        o[..., 0] = self.sqrt_k
        return o

    def egrad2rgrad(self, x: Tensor, grad: Tensor) -> Tensor:
        g = grad.clone()
        g[..., 0] = -g[..., 0]
        return self.proj_tan(x, g)

    def tangent0_from_euclidean(self, v: Tensor) -> Tensor:
        return torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)

    def euclidean_from_tangent0(self, u: Tensor) -> Tensor:
        return u[..., 1:]

    def check_point(self, x: Tensor, atol: float | None = None) -> Tensor:
        """``|<x, x>_L + k| <= atol * max(1, x_0^2)``: relative because the two squared terms cancel."""
        tol = atol if atol is not None else 1e-5 if x.dtype == torch.float32 else 1e-10
        scale = x[..., 0].pow(2).clamp_min(1.0)
        return ((self.inner(x, x, keepdim=False) + self.k).abs() <= tol * scale) & (x[..., 0] > 0)

    def to_geoopt(self):
        import geoopt

        return geoopt.Lorentz(k=self.k)


# ---------------------------------------------------------------------- isometries
def lorentz_to_poincare(x: Tensor, k: float = 1.0) -> Tensor:
    """Stereographic projection from the hyperboloid (parameter ``k``) to the ball of ``c = 1/k``."""
    sqrt_k = math.sqrt(k)
    return sqrt_k * x[..., 1:] / (x[..., :1] + sqrt_k)


def poincare_to_lorentz(x: Tensor, k: float = 1.0) -> Tensor:
    """Inverse stereographic projection: ball of ``c = 1/k`` to the hyperboloid of parameter ``k``."""
    sqrt_k = math.sqrt(k)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    denom = (k - x2).clamp_min(eps(x.dtype))
    x0 = sqrt_k * (k + x2) / denom
    spatial = 2 * k * x / denom
    return torch.cat([x0, spatial], dim=-1)


__all__ = ["Lorentz", "lorentz_to_poincare", "minkowski_inner", "poincare_to_lorentz"]
