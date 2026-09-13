"""Poincaré ball model of hyperbolic space, backed by ``geoopt.PoincareBall``.

The ball of radius ``1/sqrt(-c)`` with conformal factor ``lambda_x = 2 / (1 + c |x|^2)`` has
constant sectional curvature ``c < 0``. Every operation delegates to geoopt's stereographic
implementation (Möbius arithmetic after Ganea et al., 2018), which is what the wider hyperbolic
deep-learning literature has validated. geoopt parameterises the ball by ``c_geoopt = -c > 0``.

Numerical notes (measured in ``tests/geometry/test_poincare.py``):

* ``proj`` clips points to ``(1 - eps) * radius`` with ``eps = 4e-3`` in float32 and ``1e-5`` in
  float64 (:data:`~hyperbolic_world_model.geometry.utils.BOUNDARY_EPS`).
* ``artanh`` is clamped at ``1 - 1e-7`` for every dtype, so float32 distances between points
  near the boundary saturate; the float32-vs-float64 test documents the resulting bound.
"""

from __future__ import annotations

import math

import geoopt
import torch
from torch import Tensor

from hyperbolic_world_model.geometry.base import Manifold


class PoincareBall(Manifold):
    """Poincaré ball of sectional curvature ``c < 0`` (swept in experiments)."""

    name = "poincare"

    def __init__(self, c: float = -1.0) -> None:
        if c >= 0:
            raise ValueError(f"PoincareBall needs c < 0, got {c}")
        super().__init__(c=c)
        # float64 parameter: 0-dim tensors never promote dimensioned float32 inputs, but a float32
        # parameter would silently cap float64 accuracy at ~1e-7 for curvatures like -0.5.
        self._g = geoopt.PoincareBall(c=torch.tensor(-self.curvature, dtype=torch.float64))
        self.radius = 1.0 / math.sqrt(-self.curvature)

    # ------------------------------------------------------------------ Möbius arithmetic
    def lambda_x(self, x: Tensor, keepdim: bool = True) -> Tensor:
        """Conformal factor ``2 / (1 + c |x|^2)``."""
        return self._g.lambda_x(x, keepdim=keepdim)

    def mobius_add(self, x: Tensor, y: Tensor) -> Tensor:
        """Möbius addition ``x (+) y`` (non-commutative, non-associative)."""
        return self._g.mobius_add(x, y)

    def mobius_neg(self, x: Tensor) -> Tensor:
        """Möbius negation is ordinary negation."""
        return -x

    def gyration(self, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        """Thomas gyration ``gyr[u, v] w = -(u (+) v) (+) (u (+) (v (+) w))``."""
        return self._g.gyration(u, v, w)

    # ------------------------------------------------------------------ primitives
    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        return self._g.expmap(x, u, project=True)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        return self._g.logmap(x, y)

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        return self._g.dist(x, y, keepdim=False)

    def proj(self, x: Tensor) -> Tensor:
        """Boundary clipping: points with ``|x| >= radius`` are pulled inside by ``eps``."""
        return self._g.projx(x)

    def ptransp(self, x: Tensor, y: Tensor, u: Tensor) -> Tensor:
        """Parallel transport via gyration: ``(lambda_x / lambda_y) gyr[y, -x] u``."""
        return self._g.transp(x, y, u)

    def proj_tan(self, x: Tensor, u: Tensor) -> Tensor:
        return self._g.proju(x, u)

    def origin(self, *shape: int, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
        return torch.zeros(*shape, dtype=dtype, device=device)

    def egrad2rgrad(self, x: Tensor, grad: Tensor) -> Tensor:
        return self._g.egrad2rgrad(x, grad)

    # ------------------------------------------------------------------ closed forms at the origin
    def expmap0(self, u: Tensor) -> Tensor:
        return self._g.expmap0(u, project=True)

    def logmap0(self, y: Tensor) -> Tensor:
        return self._g.logmap0(y)

    def dist0(self, x: Tensor) -> Tensor:
        return self._g.dist0(x, keepdim=False)

    def geodesic(self, x: Tensor, y: Tensor, t: Tensor | float) -> Tensor:
        t_t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        if t_t.ndim > 0:
            t_t = t_t.unsqueeze(-1)
        return self._g.geodesic(t_t, x, y)

    def check_point(self, x: Tensor, atol: float | None = None) -> Tensor:
        """True where ``|x| < radius`` (``atol`` tightens the radius if given)."""
        r = self.radius - (atol or 0.0)
        return x.norm(dim=-1) < r

    def to_geoopt(self) -> geoopt.PoincareBall:
        return self._g


__all__ = ["PoincareBall"]
