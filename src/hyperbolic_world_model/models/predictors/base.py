"""Interface and shared architecture of the predictor heads.

Both heads are **architecturally identical**; only the geometry differs. The shared parts live
here so the comparison is controlled by construction:

* a frozen, seeded projection ``encoder_dim -> latent_dim`` followed by ``expmap0`` onto the
  manifold (:meth:`ActionConditionedPredictor.embed`),
* an action embedding ``action_dim -> action_embed_dim`` (linear + GELU),
* a fusion MLP on the concatenation ``[state coordinates, action embedding]`` that proposes a
  bounded update ``delta`` in the tangent space at the origin (:meth:`delta`).

The only method a subclass implements is :meth:`step`, which turns ``(state, delta)`` into the next
point: ``state + delta`` in flat space, ``exp_state(P_{0->state} delta)`` on a curved manifold.
The loss is the squared geodesic distance of the head's own manifold.

The embedding is frozen on purpose: if it were trainable and the target were also produced by it,
"map everything to the origin" would achieve zero loss. Keeping it fixed means the head can only
reduce the loss by learning the dynamics.

TODO(phase 3): add a learnable embedding with an anti-collapse term (e.g. VICReg-style variance
regularisation on ``logmap0`` coordinates) and compare against the frozen projection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from hyperbolic_world_model.geometry.base import Manifold
from hyperbolic_world_model.geometry.utils import clip_norm


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


class ActionConditionedPredictor(nn.Module, ABC):
    """Base class for predictor heads.

    Args:
        manifold: geometry of the latent space the head predicts in.
        encoder_dim: dimension of the frozen encoder's pooled output.
        latent_dim: intrinsic dimension of the manifold latent (swept for dimension efficiency).
        action_dim: dimension of the (normalised) action vector.
        hidden_dim: width of the fusion MLP.
        n_layers: number of hidden layers of the fusion MLP.
        action_embed_dim: width of the action embedding that is concatenated to the state.
        embed_scale: multiplier applied to the projected latent before ``expmap0``; controls how
            far from the origin encoder latents land (``1.0`` for unit-variance encoder outputs).
        max_step: cap on the norm of the proposed update per step, in the manifold's distance
            units. Keeps ``expmap`` well-conditioned in float32 and is applied identically in
            both heads.
        max_radius: states (embedded or predicted) farther than this from the origin are
            retracted along their geodesic to the origin down to this distance. Float32
            hyperbolic geometries lose accuracy exponentially with distance (see
            ``docs/methodology.md``), so this bounds every latent to the reliable region; in flat
            space it is a norm clip. Applied identically in both heads. ``None`` disables it.
        seed: seed for the frozen projection **and** for the initial weights, so two heads built
            with the same seed start from identical parameters regardless of construction order.
    """

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
        max_radius: float | None = 8.0,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.encoder_dim = int(encoder_dim)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.action_embed_dim = int(action_embed_dim)
        self.embed_scale = float(embed_scale)
        self.max_step = float(max_step)
        self.max_radius = None if max_radius is None else float(max_radius)
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
        # Seeded module construction: same seed -> identical initial weights in every head.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.action_embed = nn.Sequential(
                nn.Linear(self.action_dim, self.action_embed_dim), nn.GELU()
            )
            self.dynamics = make_mlp(
                self.latent_dim + self.action_embed_dim, hidden_dim, self.latent_dim, n_layers
            )

    # ------------------------------------------------------------------ abstract API
    @abstractmethod
    def step(self, state: Tensor, action: Tensor) -> Tensor:
        """Predict the next manifold point from ``state`` ``(..., d)`` and ``action`` ``(..., a)``."""

    # ------------------------------------------------------------------ shared architecture
    @property
    def ambient_dim(self) -> int:
        """Dimension of the manifold coordinates (``latent_dim`` plus any ambient offset)."""
        return self.latent_dim + self.manifold.ambient_dim_offset

    def embed(self, encoder_latent: Tensor) -> Tensor:
        """Frozen map: pooled encoder latent ``(..., encoder_dim)`` -> manifold point ``(..., ambient_dim)``.

        The projection is applied, scaled, lifted into the tangent space at the origin and pushed
        onto the manifold with the exponential map at the origin.
        """
        v = encoder_latent.to(self.projection.dtype) @ self.projection * self.embed_scale
        if self.max_radius is not None:
            # |v| is the geodesic distance of lift(v), so clipping v is the retraction, applied
            # before expmap0 so the hyperboloid's cosh cannot overflow.
            v = clip_norm(v, self.max_radius)
        return self.manifold.expmap0(self.lift(v))

    def lift(self, v: Tensor) -> Tensor:
        """Euclidean vector ``(..., latent_dim)`` -> tangent vector at the origin of geodesic length ``|v|``."""
        return self.manifold.tangent0_from_euclidean(v / self.manifold.lambda0)

    def retract(self, state: Tensor) -> Tensor:
        """Pull points beyond ``max_radius`` back along their geodesic to the origin (see ``__init__``)."""
        if self.max_radius is None:
            return state
        m = self.manifold
        d = m.dist0(state).unsqueeze(-1)
        origin = m.origin(*state.shape, dtype=state.dtype, device=state.device)
        t = (self.max_radius / d.clamp_min(1e-12)).clamp_max(1.0)
        pulled = m.expmap(origin, t * m.logmap(origin, state))
        return torch.where(d > self.max_radius, pulled, state)

    def coordinates(self, state: Tensor) -> Tensor:
        """Inverse of :meth:`lift` after ``logmap0``: coordinates whose norm is the geodesic distance from the origin."""
        return (
            self.manifold.euclidean_from_tangent0(self.manifold.logmap0(state))
            * self.manifold.lambda0
        )

    def fuse(self, state: Tensor, action: Tensor) -> Tensor:
        """Concatenate state coordinates with the action embedding: ``(..., latent_dim + action_embed_dim)``."""
        coords = self.coordinates(state)
        return torch.cat([coords, self.action_embed(action.to(coords.dtype))], dim=-1)

    def delta(self, state: Tensor, action: Tensor) -> Tensor:
        """Bounded update ``(..., latent_dim)`` proposed by the fusion MLP in the origin's tangent basis."""
        return clip_norm(self.dynamics(self.fuse(state, action)), self.max_step)

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
            encoder_latents: ``(B, T, encoder_dim)`` pooled frozen-encoder outputs.
            actions: ``(B, T - 1, action_dim)`` actions taken between consecutive frames.

        Returns:
            ``(pred, target)`` each ``(B, T - 1, ambient_dim)`` on the manifold.
        """
        if (
            encoder_latents.ndim != 3
            or actions.ndim != 3
            or actions.shape[1] != encoder_latents.shape[1] - 1
        ):
            raise ValueError("expected latents (B, T, D) and actions (B, T - 1, a)")
        z = self.embed(encoder_latents)
        pred = self.step(z[:, :-1], actions)
        return pred, z[:, 1:]

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Parameters the optimiser should update (never includes the projection buffer)."""
        return [p for p in self.parameters() if p.requires_grad]

    def architecture_signature(self) -> list[tuple[str, tuple[int, ...]]]:
        """``(name, shape)`` of every parameter and buffer; equal across heads by construction."""
        return [(n, tuple(p.shape)) for n, p in self.named_parameters()] + [
            (n, tuple(b.shape)) for n, b in self.named_buffers()
        ]


__all__ = ["ActionConditionedPredictor", "make_mlp"]
