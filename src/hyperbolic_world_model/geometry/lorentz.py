"""Lorentz (hyperboloid) model of hyperbolic space, backed by ``geoopt.Lorentz``.

Points satisfy ``<x, x>_L = -k`` with ``x_0 > 0`` under the Minkowski inner product
``<x, y>_L = -x_0 y_0 + sum_i x_i y_i``, giving sectional curvature ``c = -1/k``. The ambient
representation has one extra coordinate (``ambient_dim_offset = 1``) compared with the Poincaré
ball of the same curvature, to which it is isometric via :func:`lorentz_to_poincare`.

Delegated to geoopt: ``expmap``, ``logmap``, ``dist``, ``proj`` (``projx``), ``proj_tan``
(``proju``), ``expmap0``, ``logmap0``, ``dist0``, ``egrad2rgrad`` and ``inner``.

Implemented here: ``ptransp`` and the isometries (see the note above each). For ``ptransp``, geoopt's ``transp`` uses ``v - <log_x y, v> / d(x, y)^2 (log_x y +
log_y x)``, which divides by zero when ``x == y``. Teacher-forced batches contain coincident
consecutive latents (a frame where nothing moved), so we use the equivalent closed form
``u + <y, u>_L / (k - <x, y>_L) (x + y)`` (Nickel & Kiela, 2018), which is finite there and agrees
with geoopt elsewhere (tested).

Numerical note: ``<x, x>_L`` cancels two terms of size ``x_0^2``, so float32 self-distance noise
grows like ``1e-3 * x_0``; keep float32 latents within a few units of the origin or use float64.
"""

from __future__ import annotations

import math

import geoopt
import torch
from geoopt.manifolds.lorentz import math as _lmath
from torch import Tensor

from hyperbolic_world_model.geometry.base import Manifold
from hyperbolic_world_model.geometry.utils import min_norm


def minkowski_inner(u: Tensor, v: Tensor, keepdim: bool = True) -> Tensor:
    """Minkowski inner product ``-u_0 v_0 + sum_{i>0} u_i v_i`` along the last axis (geoopt)."""
    return _lmath.inner(u, v, keepdim=keepdim)


class Lorentz(Manifold):
    """Hyperboloid of sectional curvature ``c < 0`` (``k = -1 / c``)."""

    name = "lorentz"

    def __init__(self, c: float = -1.0) -> None:
        if c >= 0:
            raise ValueError(f"Lorentz needs c < 0, got {c}")
        super().__init__(c=c)
        self.k = -1.0 / self.curvature
        self.sqrt_k = math.sqrt(self.k)
        # float64 parameter for the same reason as in poincare.py.
        self._g = geoopt.Lorentz(k=torch.tensor(self.k, dtype=torch.float64))

    @property
    def ambient_dim_offset(self) -> int:
        return 1

    # ------------------------------------------------------------------ primitives
    def inner(self, u: Tensor, v: Tensor, keepdim: bool = True) -> Tensor:
        """Minkowski inner product (exposed for metrics that want it directly)."""
        return minkowski_inner(u, v, keepdim=keepdim)

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        return self._g.expmap(x, u, project=True)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        return self._g.logmap(x, y)

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        return self._g.dist(x, y, keepdim=False)

    def proj(self, x: Tensor) -> Tensor:
        """Recompute the time-like coordinate so ``<x, x>_L = -k`` holds exactly."""
        return self._g.projx(x)

    def ptransp(self, x: Tensor, y: Tensor, u: Tensor) -> Tensor:
        """``u + <y, u>_L / (k - <x, y>_L) (x + y)``; finite at ``x == y`` (see module docstring)."""
        denom = (self.k - self.inner(x, y)).clamp_min(min_norm(x.dtype))
        return u + self.inner(y, u) / denom * (x + y)

    def proj_tan(self, x: Tensor, u: Tensor) -> Tensor:
        return self._g.proju(x, u)

    def origin(self, *shape: int, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
        o = torch.zeros(*shape, dtype=dtype, device=device)
        o[..., 0] = self.sqrt_k
        return o

    def egrad2rgrad(self, x: Tensor, grad: Tensor) -> Tensor:
        return self._g.egrad2rgrad(x, grad.clone())

    # ------------------------------------------------------------------ closed forms at the origin
    def expmap0(self, u: Tensor) -> Tensor:
        return self._g.expmap0(u, project=True)

    def logmap0(self, y: Tensor) -> Tensor:
        return self._g.logmap0(y)

    def dist0(self, x: Tensor) -> Tensor:
        return self._g.dist0(x, keepdim=False)

    def tangent0_from_euclidean(self, v: Tensor) -> Tensor:
        return torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)

    def euclidean_from_tangent0(self, u: Tensor) -> Tensor:
        return u[..., 1:]

    def check_point(self, x: Tensor, atol: float | None = None) -> Tensor:
        """``|<x, x>_L + k| <= atol * max(1, x_0^2)``: relative, because two squared terms cancel."""
        tol = atol if atol is not None else 1e-5 if x.dtype == torch.float32 else 1e-10
        scale = x[..., 0].pow(2).clamp_min(1.0)
        return ((self.inner(x, x, keepdim=False) + self.k).abs() <= tol * scale) & (x[..., 0] > 0)

    def to_geoopt(self) -> geoopt.Lorentz:
        return self._g


# ---------------------------------------------------------------------- isometries (ours)
# geoopt ships ``lorentz_to_poincare`` / ``poincare_to_lorentz``, but they map the *unit* ball
# with a k-scaled metric, not the radius-1/sqrt(-c) ball that ``geoopt.PoincareBall`` (and our
# ``PoincareBall``) use, so for any curvature other than -1 they land off-manifold. These two
# functions are the correct maps between ``PoincareBall(c)`` and ``Lorentz(c)`` (tested).
def lorentz_to_poincare(x: Tensor, c: float = -1.0) -> Tensor:
    """Stereographic projection from ``Lorentz(c)`` to ``PoincareBall(c)``: ``sqrt(k) x_{1:} / (x_0 + sqrt(k))``."""
    sqrt_k = math.sqrt(-1.0 / c)
    return sqrt_k * x[..., 1:] / (x[..., :1] + sqrt_k)


def poincare_to_lorentz(x: Tensor, c: float = -1.0) -> Tensor:
    """Inverse stereographic projection from ``PoincareBall(c)`` to ``Lorentz(c)``."""
    k = -1.0 / c
    sqrt_k = math.sqrt(k)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    denom = (k - x2).clamp_min(min_norm(x.dtype))
    return torch.cat([sqrt_k * (k + x2) / denom, 2 * k * x / denom], dim=-1)


__all__ = ["Lorentz", "lorentz_to_poincare", "minkowski_inner", "poincare_to_lorentz"]
