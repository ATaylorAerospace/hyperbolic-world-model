"""Poincaré ball model of hyperbolic space, backed by ``geoopt.PoincareBall``.

The ball of radius ``1/sqrt(-c)`` with conformal factor ``lambda_x = 2 / (1 + c |x|^2)`` has
constant sectional curvature ``c < 0``. Every operation delegates to geoopt's stereographic
implementation (Möbius arithmetic after Ganea et al., 2018), which is what the wider hyperbolic
deep-learning literature has validated. geoopt parameterises the ball by ``c_geoopt = -c > 0``.

Numerical notes (measured in ``tests/geometry/test_poincare.py``):

* ``proj`` clips points to ``(1 - eps) * radius`` with ``eps = 4e-3`` in float32 and ``1e-5`` in
  float64 (:data:`~hyperbolic_world_model.geometry.utils.BOUNDARY_EPS`).
* ``dist`` evaluates ``|-x (+) y|`` via the symmetric identity
  ``|x - y| / sqrt((1 - k |x|^2)(1 - k |y|^2) + k |x - y|^2)`` (``k = -c > 0``) so ``d(x, y) == d(y, x)``
  holds exactly in float32 without Möbius-addition subtraction cancellation near the boundary.
* ``artanh`` is clamped at ``1 - 1e-7`` for every dtype, so float32 distances between points
  near the boundary saturate; the float32-vs-float64 test documents the resulting bound.
"""

from __future__ import annotations

import math

import geoopt
import torch
from geoopt.manifolds.stereographic import math as _smath
from torch import Tensor

from hyperbolic_world_model.geometry.base import Manifold
from hyperbolic_world_model.geometry.utils import min_norm


class PoincareBall(Manifold):
    """Poincaré ball of sectional curvature ``c < 0`` (swept in experiments)."""

    name = "poincare"

    def __init__(self, c: float = -1.0) -> None:
        if c >= 0:
            raise ValueError(f"PoincareBall needs c < 0, got {c}")
        super().__init__(c=c)
        self._k = -self.curvature
        self._sqrt_k = math.sqrt(self._k)
        # float64 parameter: 0-dim tensors never promote dimensioned float32 inputs, but a float32
        # parameter would silently cap float64 accuracy at ~1e-7 for curvatures like -0.5.
        self._g = geoopt.PoincareBall(c=torch.tensor(self._k, dtype=torch.float64))
        self.radius = 1.0 / self._sqrt_k

    @property
    def lambda0(self) -> float:
        return 2.0

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
        """Closed-form distance avoiding Möbius-addition asymmetry and boundary cancellation."""
        diff = x - y
        sqdist = (diff * diff).sum(dim=-1)
        x2 = (x * x).sum(dim=-1)
        y2 = (y * y).sum(dim=-1)
        denom = ((1.0 - self._k * x2) * (1.0 - self._k * y2) + self._k * sqdist).clamp_min(
            min_norm(x.dtype)
        )
        mobius_norm = diff.norm(dim=-1) / denom.sqrt()
        return (2.0 / self._sqrt_k) * _smath.artanh(self._sqrt_k * mobius_norm)

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
