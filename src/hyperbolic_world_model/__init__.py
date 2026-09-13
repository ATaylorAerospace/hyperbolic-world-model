"""hyperbolic_world_model: evaluating hyperbolic latent spaces in action-conditioned world models.

The package is organised around a single abstraction, :class:`hyperbolic_world_model.geometry.Manifold`.
Every predictor head, every distance-based metric and every task consumes a ``Manifold`` and never
imports a concrete geometry directly, so adding a new geometry means adding one file under
``geometry/`` and one config under ``configs/geometry/``.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
