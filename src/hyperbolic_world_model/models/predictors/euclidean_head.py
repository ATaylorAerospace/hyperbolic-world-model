"""Euclidean baseline predictor head.

Identical to :class:`~hyperbolic_world_model.models.predictors.hyperbolic_head.HyperbolicHead`
in every module and parameter; the only difference is :meth:`EuclideanHead.step`, which adds the
proposed update in flat space. The loss is squared Euclidean distance, i.e. ``Euclidean().sqdist``,
so it goes through the same ``Manifold`` code path as every other head.
"""

from __future__ import annotations

from torch import Tensor

from hyperbolic_world_model.geometry.base import Manifold
from hyperbolic_world_model.geometry.euclidean import Euclidean
from hyperbolic_world_model.models.predictors.base import ActionConditionedPredictor


class EuclideanHead(ActionConditionedPredictor):
    """``s_{t+1} = s_t + delta(s_t, a_t)`` in R^d, then the shared ``max_radius`` clip."""

    def __init__(
        self,
        encoder_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        action_embed_dim: int = 32,
        embed_scale: float = 1.0,
        max_step: float = 5.0,
        seed: int = 0,
        max_radius: float | None = 4.0,
        manifold: Manifold | None = None,
    ) -> None:
        manifold = Euclidean() if manifold is None else manifold
        if not isinstance(manifold, Euclidean):
            raise TypeError(
                "EuclideanHead requires the Euclidean manifold; use HyperbolicHead otherwise"
            )
        super().__init__(
            manifold=manifold,
            encoder_dim=encoder_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            action_embed_dim=action_embed_dim,
            embed_scale=embed_scale,
            max_step=max_step,
            seed=seed,
            max_radius=max_radius,
        )

    def step(self, state: Tensor, action: Tensor) -> Tensor:
        return self.retract(state + self.delta(state, action))


__all__ = ["EuclideanHead"]
