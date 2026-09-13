"""Flat Euclidean space: the baseline geometry.

Every primitive is the identity or a subtraction. Included so the Euclidean predictor head and the
Euclidean metrics go through exactly the same ``Manifold`` interface as the hyperbolic ones; no
code path outside ``geometry/`` should special-case "flat".
"""

from __future__ import annotations

import torch
from torch import Tensor

from hyperbolic_world_model.geometry.base import Manifold


class Euclidean(Manifold):
    """R^n with the standard metric. ``curvature`` is fixed at ``0``."""

    name = "euclidean"

    def __init__(self, curvature: float = 0.0) -> None:
        if curvature != 0.0:
            raise ValueError(f"Euclidean space has curvature 0, got {curvature}")
        super().__init__(curvature=0.0)

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        return x + u

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        return y - x

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        return (y - x).norm(dim=-1)

    def sqdist(self, x: Tensor, y: Tensor) -> Tensor:
        return ((y - x) ** 2).sum(dim=-1)

    def proj(self, x: Tensor) -> Tensor:
        return x

    def ptransp(self, x: Tensor, y: Tensor, u: Tensor) -> Tensor:
        return u

    def proj_tan(self, x: Tensor, u: Tensor) -> Tensor:
        return u

    def origin(self, *shape: int, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
        return torch.zeros(*shape, dtype=dtype, device=device)

    def egrad2rgrad(self, x: Tensor, grad: Tensor) -> Tensor:
        return grad

    def expmap0(self, u: Tensor) -> Tensor:
        return u

    def logmap0(self, y: Tensor) -> Tensor:
        return y

    def to_geoopt(self):
        import geoopt

        return geoopt.Euclidean(ndim=1)


__all__ = ["Euclidean"]
