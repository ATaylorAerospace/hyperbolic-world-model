"""Hyperbolic predictor head: dynamics as tangent-space updates on a curved manifold.

Given a manifold point ``s_t`` and action ``a_t``:

1. read the state in tangent coordinates at the origin, ``v = logmap0(s_t)`` (Euclidean vector),
2. let an MLP propose a tangent update ``u_0 = MLP([v, a_t])`` at the origin,
3. parallel-transport ``u_0`` to ``s_t`` and follow the geodesic: ``s_{t+1} = expmap(s_t, ptransp(0, s_t, u_0))``.

The loss is the squared geodesic distance ``manifold.sqdist``. The head works for **any**
``Manifold``; with ``Euclidean`` it reduces exactly to :class:`EuclideanHead`, which
``tests/test_smoke_experiment.py`` relies on.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hyperbolic_world_model.geometry.base import Manifold
from hyperbolic_world_model.geometry.utils import clip_norm
from hyperbolic_world_model.models.predictors.base import ActionConditionedPredictor
from hyperbolic_world_model.models.predictors.euclidean_head import make_mlp


class HyperbolicHead(ActionConditionedPredictor):
    """Geodesic residual MLP head on an arbitrary manifold.

    Args:
        manifold: any :class:`Manifold` (Poincaré, Lorentz, or Euclidean for consistency checks).
        max_step: cap on the tangent-update norm per step, keeps ``expmap`` well-conditioned in
            float32 far from the origin. Interpreted in the manifold's distance units.
    """

    def __init__(
        self,
        manifold: Manifold,
        encoder_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        embed_scale: float = 1.0,
        seed: int = 0,
        max_step: float = 5.0,
    ) -> None:
        super().__init__(manifold, encoder_dim, latent_dim, action_dim, embed_scale, seed)
        self.max_step = float(max_step)
        self.mlp = make_mlp(latent_dim + action_dim, hidden_dim, latent_dim, n_layers)

    def step(self, state: Tensor, action: Tensor) -> Tensor:
        m = self.manifold
        v = m.euclidean_from_tangent0(m.logmap0(state))
        u_eucl = clip_norm(self.mlp(torch.cat([v, action.to(v.dtype)], dim=-1)), self.max_step)
        u0 = m.tangent0_from_euclidean(u_eucl)
        origin = m.origin(*state.shape, dtype=state.dtype, device=state.device)
        u = m.ptransp(origin, state, u0)
        return m.proj(m.expmap(state, m.proj_tan(state, u)))


__all__ = ["HyperbolicHead"]
