"""Model components: frozen encoders and trainable predictor heads.

Build everything through :func:`hyperbolic_world_model.models.registry.build_model`; nothing in
``tasks/`` or ``training/`` instantiates an encoder or head directly.
"""

from __future__ import annotations

from hyperbolic_world_model.models.registry import (
    ModelBundle,
    build_encoder,
    build_model,
    build_predictor,
)

__all__ = ["ModelBundle", "build_encoder", "build_model", "build_predictor"]
