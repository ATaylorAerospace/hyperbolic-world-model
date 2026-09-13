"""Riemannian geometries used as latent spaces.

Three manifolds are provided, all sharing the :class:`Manifold` interface:

* :class:`Euclidean` - flat baseline (curvature 0).
* :class:`PoincareBall` - conformal model of hyperbolic space of curvature ``-c``.
* :class:`Lorentz` - hyperboloid model of the same space, numerically preferable for optimisation.

Use :func:`build_manifold` to construct one from a Hydra config group ``geometry``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hyperbolic_world_model.geometry.base import Manifold
from hyperbolic_world_model.geometry.euclidean import Euclidean
from hyperbolic_world_model.geometry.lorentz import Lorentz
from hyperbolic_world_model.geometry.poincare import PoincareBall

MANIFOLDS: dict[str, type[Manifold]] = {
    "euclidean": Euclidean,
    "poincare": PoincareBall,
    "lorentz": Lorentz,
}


def build_manifold(cfg: Mapping[str, Any]) -> Manifold:
    """Instantiate a manifold from a ``geometry`` config mapping.

    Args:
        cfg: mapping with a ``name`` key (one of :data:`MANIFOLDS`) and, for curved spaces, a
            ``curvature`` key (negative float). Extra keys are passed to the constructor.

    Raises:
        KeyError: if ``name`` is unknown. The error message lists the registered geometries.
    """
    cfg = dict(cfg)
    name = str(cfg.pop("name"))
    cfg.pop("_target_", None)
    if name not in MANIFOLDS:
        raise KeyError(f"unknown geometry {name!r}; registered: {sorted(MANIFOLDS)}")
    return MANIFOLDS[name](**cfg)


__all__ = ["MANIFOLDS", "Euclidean", "Lorentz", "Manifold", "PoincareBall", "build_manifold"]
