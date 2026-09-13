"""Euclidean baseline predictor head.

A residual MLP on ``[state, action]`` in flat space. Loss is squared Euclidean distance, which is
``Euclidean().sqdist`` and therefore the same code path every other head uses.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from hyperbolic_world_model.geometry.base import Manifold
from hyperbolic_world_model.geometry.euclidean import Euclidean
from hyperbolic_world_model.models.predictors.base import ActionConditionedPredictor


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    """``n_layers`` hidden layers of width ``hidden_dim`` with GELU, zero-initialised output."""
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden_dim), nn.GELU()]
        d = hidden_dim
    out = nn.Linear(d, out_dim)
    nn.init.zeros_(out.weight)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class EuclideanHead(ActionConditionedPredictor):
    """``s_{t+1} = s_t + MLP([s_t, a_t])`` in R^d."""

    def __init__(
        self,
        encoder_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        embed_scale: float = 1.0,
        seed: int = 0,
        manifold: Manifold | None = None,
    ) -> None:
        manifold = Euclidean() if manifold is None else manifold
        if not isinstance(manifold, Euclidean):
            raise TypeError(
                "EuclideanHead requires the Euclidean manifold; use HyperbolicHead otherwise"
            )
        super().__init__(manifold, encoder_dim, latent_dim, action_dim, embed_scale, seed)
        self.mlp = make_mlp(latent_dim + action_dim, hidden_dim, latent_dim, n_layers)

    def step(self, state: Tensor, action: Tensor) -> Tensor:
        return state + self.mlp(torch.cat([state, action.to(state.dtype)], dim=-1))


__all__ = ["EuclideanHead", "make_mlp"]
