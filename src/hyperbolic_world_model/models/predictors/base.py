"""Interface every predictor head implements.

A head owns a :class:`~hyperbolic_world_model.geometry.Manifold` and exposes three operations:

* :meth:`ActionConditionedPredictor.embed` - map a frozen encoder latent onto the manifold,
* :meth:`ActionConditionedPredictor.step` - one action-conditioned transition on the manifold,
* :meth:`ActionConditionedPredictor.loss` - squared geodesic distance in the head's own geometry.

The embedding is **frozen** (a seeded random projection followed by ``expmap0``). This is a
deliberate phase-1 design choice: if the embedding were trainable and the target were also produced
by it, the trivial solution "map everything to the origin" would achieve zero loss. Keeping the
embedding fixed means the head can only reduce the loss by learning the dynamics.

TODO(phase 3): add a learnable embedding with an anti-collapse term (e.g. VICReg-style variance
regularisation on ``logmap0`` coordinates) and compare against the frozen projection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from hyperbolic_world_model.geometry.base import Manifold


class ActionConditionedPredictor(nn.Module, ABC):
    """Base class for predictor heads.

    Args:
        manifold: geometry of the latent space the head predicts in.
        encoder_dim: dimension of the frozen encoder's output.
        latent_dim: intrinsic dimension of the manifold latent (swept for dimension efficiency).
        action_dim: dimension of the (normalised) action vector.
        embed_scale: multiplier applied to the projected latent before ``expmap0``. Controls how
            far from the origin encoder latents land; ``1.0`` is a safe default for unit-variance
            encoder outputs.
        seed: seed for the frozen projection.
    """

    def __init__(
        self,
        manifold: Manifold,
        encoder_dim: int,
        latent_dim: int,
        action_dim: int,
        embed_scale: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.encoder_dim = int(encoder_dim)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.embed_scale = float(embed_scale)
        gen = torch.Generator().manual_seed(seed)
        proj = torch.randn(self.encoder_dim, self.latent_dim, generator=gen)
        # Orthonormal columns so the projection neither inflates nor shrinks typical norms.
        q, _ = (
            torch.linalg.qr(proj)
            if self.encoder_dim >= self.latent_dim
            else torch.linalg.qr(proj.T)
        )
        proj = q if self.encoder_dim >= self.latent_dim else q.T
        self.register_buffer("projection", proj.contiguous())

    # ------------------------------------------------------------------ abstract API
    @abstractmethod
    def step(self, state: Tensor, action: Tensor) -> Tensor:
        """Predict the next manifold point from ``state`` ``(..., d)`` and ``action`` ``(..., a)``."""

    # ------------------------------------------------------------------ shared behaviour
    @property
    def ambient_dim(self) -> int:
        """Dimension of the manifold coordinates (``latent_dim`` plus any ambient offset)."""
        return self.latent_dim + self.manifold.ambient_dim_offset

    def embed(self, encoder_latent: Tensor) -> Tensor:
        """Frozen map: encoder latent ``(..., encoder_dim)`` -> point on the manifold ``(..., ambient_dim)``."""
        v = encoder_latent.to(self.projection.dtype) @ self.projection * self.embed_scale
        return self.manifold.expmap0(self.manifold.tangent0_from_euclidean(v))

    def loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Mean squared geodesic distance in the head's native geometry."""
        return self.manifold.sqdist(pred, target).mean()

    def rollout(self, state0: Tensor, actions: Tensor) -> Tensor:
        """Open-loop rollout: ``state0`` ``(B, d)`` and ``actions`` ``(B, H, a)`` -> ``(B, H, d)``."""
        if actions.ndim != 3:
            raise ValueError(f"actions must be (B, H, a), got {tuple(actions.shape)}")
        states = []
        s = state0
        for h in range(actions.shape[1]):
            s = self.step(s, actions[:, h])
            states.append(s)
        return torch.stack(states, dim=1)

    def forward(self, encoder_latents: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Teacher-forced one-step prediction.

        Args:
            encoder_latents: ``(B, T, encoder_dim)`` frozen encoder outputs.
            actions: ``(B, T - 1, action_dim)`` actions taken between consecutive frames.

        Returns:
            ``(pred, target)`` each ``(B, T - 1, ambient_dim)`` on the manifold.
        """
        z = self.embed(encoder_latents)
        pred = self.step(z[:, :-1], actions)
        return pred, z[:, 1:]

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters the optimiser should update (never includes the projection buffer)."""
        return [p for p in self.parameters() if p.requires_grad]


__all__ = ["ActionConditionedPredictor"]
