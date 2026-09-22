"""Hyperbolic predictor head: dynamics as geodesic steps on a curved manifold.

Given a manifold point ``s_t`` and action ``a_t`` (shared architecture in ``base.py``):

1. read the state in tangent coordinates at the origin, ``v = logmap0(s_t)``,
2. embed the action and concatenate, ``[v, phi(a_t)]``,
3. let the fusion MLP propose a bounded tangent update ``delta`` at the origin,
4. parallel-transport ``delta`` to ``s_t`` and follow the geodesic:
   ``s_{t+1} = proj(expmap(s_t, P_{0->s_t} delta))``,
5. retract to ``max_radius`` if the step left the numerically reliable region (shared guard).

The loss is the squared geodesic distance ``manifold.sqdist``. The head works for **any**
``Manifold``; with ``Euclidean`` it reduces exactly to :class:`EuclideanHead`, which
``tests/models/test_predictor_heads.py`` asserts.
"""

from __future__ import annotations

from torch import Tensor

from hyperbolic_world_model.geometry.base import Manifold
from hyperbolic_world_model.models.predictors.base import ActionConditionedPredictor


class HyperbolicHead(ActionConditionedPredictor):
    """Geodesic residual head on an arbitrary manifold (Poincaré, Lorentz, or Euclidean)."""

    def __init__(
        self,
        manifold: Manifold,
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
    ) -> None:
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
        m = self.manifold
        u0 = self.lift(self.delta(state, action))
        origin = m.origin(*state.shape, dtype=state.dtype, device=state.device)
        u = m.proj_tan(state, m.ptransp(origin, state, u0))
        return self.retract(m.proj(m.expmap(state, u)))


__all__ = ["HyperbolicHead"]
