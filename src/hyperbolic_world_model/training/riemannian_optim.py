"""Optimiser construction with geoopt integration.

Predictor heads keep their trainable weights in Euclidean parameter space (the MLP) and only the
*outputs* live on the manifold, so plain AdamW is correct for them. If a module registers a
``geoopt.ManifoldParameter`` (for example a learnable manifold embedding table in a later phase),
``RiemannianAdam`` is required, and this module picks it automatically so that a forgotten
optimiser choice can never silently apply Euclidean updates to manifold parameters.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from hyperbolic_world_model.geometry.base import Manifold


def manifold_parameter(tensor: torch.Tensor, manifold: Manifold, requires_grad: bool = True):
    """Wrap ``tensor`` as a ``geoopt.ManifoldParameter`` on the geoopt twin of ``manifold``."""
    import geoopt

    return geoopt.ManifoldParameter(
        tensor, manifold=manifold.to_geoopt(), requires_grad=requires_grad
    )


def has_manifold_parameters(params: Iterable[nn.Parameter]) -> bool:
    import geoopt

    return any(isinstance(p, geoopt.ManifoldParameter) for p in params)


def build_optimizer(
    params: Iterable[nn.Parameter],
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    stabilize: int | None = 10,
) -> torch.optim.Optimizer:
    """Return ``geoopt.optim.RiemannianAdam`` if any parameter lives on a manifold, else ``AdamW``.

    Args:
        params: parameters to optimise.
        lr, weight_decay, betas: standard Adam hyper-parameters.
        stabilize: every ``stabilize`` steps ``RiemannianAdam`` re-projects manifold parameters
            (ignored for AdamW).
    """
    params = list(params)
    if not params:
        raise ValueError("no trainable parameters were passed to build_optimizer")
    if has_manifold_parameters(params):
        from geoopt.optim import RiemannianAdam

        return RiemannianAdam(
            params, lr=lr, weight_decay=weight_decay, betas=betas, stabilize=stabilize
        )
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


__all__ = ["build_optimizer", "has_manifold_parameters", "manifold_parameter"]
