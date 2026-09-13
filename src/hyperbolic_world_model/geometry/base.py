"""The ``Manifold`` abstraction.

This is the single interface every predictor head, metric and task depends on. Nothing outside
``geometry/`` may import ``PoincareBall``, ``Lorentz`` or ``Euclidean`` by name; they receive a
``Manifold`` and call the primitives below. Adding a geometry therefore requires:

1. a new subclass in ``geometry/<name>.py`` implementing the abstract methods,
2. a registry entry in ``geometry/__init__.py``,
3. a config file in ``configs/geometry/<name>.yaml``,
4. adding it to the parametrised contract tests in ``tests/geometry/test_base.py``.

Conventions
-----------
* Points and tangent vectors are tensors whose **last** dimension is the ambient coordinate axis.
  Leading dimensions are batch dimensions and broadcast.
* ``dist`` returns a tensor with the last dimension removed.
* ``c`` is the sectional curvature: ``0`` for Euclidean, ``< 0`` for hyperbolic. It is passed at
  construction and exposed as :attr:`Manifold.curvature`. geoopt's positive parameters
  (``c`` for the ball, ``k`` for the hyperboloid) are derived from it internally.
* All methods must be differentiable and finite for float32 inputs anywhere in the domain; the
  exact contract is ``tests/geometry/test_base.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class Manifold(ABC):
    """Abstract Riemannian manifold with the operations a world model needs.

    Subclasses implement the primitives ``expmap``, ``logmap``, ``dist``, ``proj``, ``ptransp``
    plus the tangent-space helpers ``proj_tan``, ``origin``, ``egrad2rgrad`` and ``to_geoopt``.
    Everything else here is derived and may be overridden only for numerical reasons.
    """

    #: Registry name used by configs (``configs/geometry/<name>.yaml``).
    name: str = "abstract"

    def __init__(self, c: float = 0.0) -> None:
        self._c = float(c)

    # ------------------------------------------------------------------ properties
    @property
    def curvature(self) -> float:
        """Sectional curvature ``c``. Negative for hyperbolic space, zero for Euclidean."""
        return self._c

    @property
    def ambient_dim_offset(self) -> int:
        """Extra ambient coordinates beyond the intrinsic dimension (``1`` for the hyperboloid)."""
        return 0

    # ------------------------------------------------------------------ primitives
    @abstractmethod
    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """Exponential map: move from point ``x`` along tangent vector ``u`` for unit time."""

    @abstractmethod
    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map: the tangent vector at ``x`` whose ``expmap`` reaches ``y``."""

    @abstractmethod
    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Geodesic distance between ``x`` and ``y`` (last dimension reduced)."""

    @abstractmethod
    def proj(self, x: Tensor) -> Tensor:
        """Project an ambient point back onto the manifold (boundary clipping, hyperboloid lift)."""

    @abstractmethod
    def ptransp(self, x: Tensor, y: Tensor, u: Tensor) -> Tensor:
        """Parallel-transport tangent vector ``u`` at ``x`` to the tangent space at ``y``."""

    # ------------------------------------------------------------------ tangent-space helpers
    @abstractmethod
    def proj_tan(self, x: Tensor, u: Tensor) -> Tensor:
        """Project an ambient vector ``u`` onto the tangent space at ``x``."""

    @abstractmethod
    def origin(self, *shape: int, dtype: torch.dtype = torch.float32, device=None) -> Tensor:
        """The canonical base point, broadcast to ``shape`` (last entry is the ambient dim)."""

    @abstractmethod
    def egrad2rgrad(self, x: Tensor, grad: Tensor) -> Tensor:
        """Convert a Euclidean gradient at ``x`` into a Riemannian gradient."""

    @abstractmethod
    def to_geoopt(self):
        """The equivalent ``geoopt`` manifold, for ``ManifoldParameter`` / ``RiemannianAdam``."""

    # ------------------------------------------------------------------ derived operations
    def expmap0(self, u: Tensor) -> Tensor:
        """Exponential map at the origin. Subclasses override with geoopt's closed form."""
        return self.expmap(self.origin(*u.shape, dtype=u.dtype, device=u.device), u)

    def logmap0(self, y: Tensor) -> Tensor:
        """Logarithmic map at the origin. Subclasses override with geoopt's closed form."""
        return self.logmap(self.origin(*y.shape, dtype=y.dtype, device=y.device), y)

    def sqdist(self, x: Tensor, y: Tensor) -> Tensor:
        """Squared geodesic distance; the regression loss used by predictor heads."""
        return self.dist(x, y) ** 2

    def dist0(self, x: Tensor) -> Tensor:
        """Distance from the origin (a norm-like quantity used for hierarchy depth probes)."""
        return self.dist(self.origin(*x.shape, dtype=x.dtype, device=x.device), x)

    def geodesic(self, x: Tensor, y: Tensor, t: Tensor | float) -> Tensor:
        """Point at fraction ``t`` along the geodesic from ``x`` to ``y``.

        ``t`` may be a scalar or a tensor broadcastable to the batch shape of ``x``; it is
        unsqueezed on the last axis so it scales whole tangent vectors.
        """
        t_t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
        if t_t.ndim > 0:
            t_t = t_t.unsqueeze(-1)
        return self.expmap(x, t_t * self.logmap(x, y))

    def pairwise_dist(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """All-pairs distance matrix ``(n, m)`` between rows of ``x`` ``(n, d)`` and ``y`` ``(m, d)``."""
        y = x if y is None else y
        return self.dist(x.unsqueeze(-2), y.unsqueeze(-3))

    def check_point(self, x: Tensor, atol: float | None = None) -> Tensor:
        """Boolean mask of which points lie on the manifold (within ``atol``). Default: all true."""
        return torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)

    def tangent0_from_euclidean(self, v: Tensor) -> Tensor:
        """Lift a Euclidean vector ``(..., d)`` into the tangent space at the origin.

        Identity when the ambient dimension equals the intrinsic one; the Lorentz model prepends
        a zero time-like coordinate. Heads call this so their linear layers never need to know
        the ambient representation.
        """
        return v

    def euclidean_from_tangent0(self, u: Tensor) -> Tensor:
        """Inverse of :meth:`tangent0_from_euclidean`: drop ambient-only coordinates."""
        return u

    # ------------------------------------------------------------------ dunder
    def __repr__(self) -> str:
        return f"{type(self).__name__}(c={self.curvature})"


__all__ = ["Manifold"]
