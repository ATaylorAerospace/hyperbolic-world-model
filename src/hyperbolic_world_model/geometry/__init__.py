"""Riemannian geometries used as latent spaces.

Three manifolds share the :class:`Manifold` interface:

* :class:`Euclidean` - flat baseline (``c = 0``).
* :class:`PoincareBall` - conformal model of hyperbolic space of curvature ``c < 0``.
* :class:`Lorentz` - hyperboloid model of the same space, numerically preferable for optimisation.

Curved geometries are thin wrappers over ``geoopt`` (``geoopt.PoincareBall``, ``geoopt.Lorentz``),
whose numerics are well tested upstream. We implement ourselves only what geoopt does not expose
or where its implementation is singular for inputs we hit in practice (see ``lorentz.py``).

Use :func:`build_manifold` to construct one from a Hydra ``geometry`` config group.
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
        cfg: mapping with a ``name`` key (one of :data:`MANIFOLDS`) and the curvature under
            either ``c`` or ``curvature`` (the config files use ``curvature``). Both are the
            sectional curvature: ``0`` for Euclidean, negative for hyperbolic. Any other keys are
            passed to the constructor unchanged.

    Raises:
        KeyError: if ``name`` is unknown; the message lists the registered geometries.
        TypeError: if both ``c`` and ``curvature`` are given.
    """
    cfg = dict(cfg)
    name = str(cfg.pop("name"))
    cfg.pop("_target_", None)
    if "curvature" in cfg:
        if "c" in cfg:
            raise TypeError("give either 'c' or 'curvature', not both")
        cfg["c"] = cfg.pop("curvature")
    if name not in MANIFOLDS:
        raise KeyError(f"unknown geometry {name!r}; registered: {sorted(MANIFOLDS)}")
    return MANIFOLDS[name](**cfg)


__all__ = ["MANIFOLDS", "Euclidean", "Lorentz", "Manifold", "PoincareBall", "build_manifold"]
