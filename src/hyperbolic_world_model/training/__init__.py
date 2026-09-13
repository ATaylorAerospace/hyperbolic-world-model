"""Training of predictor heads only. Encoders are never trained anywhere in this package."""

from __future__ import annotations

from hyperbolic_world_model.training.riemannian_optim import build_optimizer

__all__ = ["build_optimizer"]
