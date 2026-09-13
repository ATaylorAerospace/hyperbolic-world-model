"""Frozen video encoders.

Every encoder subclasses :class:`FrozenEncoder`, which fixes the contract used by the rest of
the harness: frames in, one pooled latent per frame out, and **no trainable parameters**.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class FrozenEncoder(nn.Module, ABC):
    """Base class for frozen encoders.

    Contract:
        * ``forward(frames)`` takes ``(batch, time, channels, height, width)`` float frames in
          ``[0, 1]`` and returns ``(batch, time, embed_dim)`` latents.
        * every parameter has ``requires_grad=False`` and the module is in ``eval()`` mode.
          :meth:`assert_frozen` is called by the trainer before the first optimiser step.
    """

    embed_dim: int

    @abstractmethod
    def forward(self, frames: Tensor) -> Tensor:
        """Encode ``(B, T, C, H, W)`` frames into ``(B, T, embed_dim)`` latents."""

    def freeze(self) -> FrozenEncoder:
        """Disable gradients for every parameter and switch to eval mode."""
        for p in self.parameters():
            p.requires_grad_(False)
        return self.eval()

    def train(self, mode: bool = True) -> FrozenEncoder:  # type: ignore[override]
        """Frozen encoders never enter training mode (keeps dropout/BN deterministic)."""
        return super().train(False)

    def assert_frozen(self) -> None:
        """Raise if any parameter is trainable. This is the frozen-encoder invariant."""
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        if trainable:
            raise RuntimeError(
                f"{type(self).__name__} has trainable parameters: {trainable[:5]}"
                f"{' ...' if len(trainable) > 5 else ''}. The encoder must stay frozen."
            )
        if self.training:
            raise RuntimeError(f"{type(self).__name__} is in training mode; call .eval()")

    @torch.no_grad()
    def encode(self, frames: Tensor) -> Tensor:
        """``forward`` under ``no_grad``; the only entry point tasks and trainers should use."""
        return self.forward(frames)


__all__ = ["FrozenEncoder"]
