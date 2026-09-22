"""Flat Euclidean space: the baseline geometry.

Wraps ``geoopt.Euclidean(ndim=1)`` so the baseline goes through exactly the same ``Manifold``
interface as the curved geometries; no code path outside ``geometry/`` special-cases "flat".
"""

from __future__ import annotations

import geoopt
import torch
from torch import Tensor

from hyperbolic_world_model.geometry.base import Manifold


class Euclidean(Manifold):
    """R^n with the standard metric. ``c`` must be ``0``."""

    name = "euclidean"

    def __init__(self, c: float = 0.0) -> None:
        if c != 0.0:
            raise ValueError(f"Euclidean space has curvature 0, got c={c}")
        super().__init__(c=0.0)
        self._g = geoopt.Euclidean(ndim=1)

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        return self._g.expmap(x, u)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        return self._g.logmap(x, y)

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        return self._g.dist(x, y)

    def sqdist(self, x: Tensor, y: Tensor) -> Tensor:
        return self._g.dist2(x, y)

    def proj(self, x: Tensor) -> Tensor:
        return self._g.projx(x)

    def ptransp(self, x: Tensor, y: Tensor, u: Tensor) -> Tensor:
        return self._g.transp(x, y, u)

    def proj_tan(self, x: Tensor, u: Tensor) -> Tensor:
        return self._g.proju(x, u)

    def origin(self, *shape: int, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
        return torch.zeros(*shape, dtype=dtype, device=device)

    def egrad2rgrad(self, x: Tensor, grad: Tensor) -> Tensor:
        return self._g.egrad2rgrad(x, grad)

    def pairwise_dist(
        self, x: Tensor, y: Tensor | None = None, chunk_rows: int | None = None
    ) -> Tensor:
        """Exact ``torch.cdist`` (no ``(n, m, d)`` broadcast; the mm-based mode is avoided for accuracy)."""
        y = x if y is None else y
        return torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist")

    def expmap0(self, u: Tensor) -> Tensor:
        return u

    def logmap0(self, y: Tensor) -> Tensor:
        return y

    def to_geoopt(self) -> geoopt.Euclidean:
        return self._g


__all__ = ["Euclidean"]
