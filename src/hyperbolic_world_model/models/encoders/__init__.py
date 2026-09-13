"""Frozen video encoders.

Every encoder subclasses :class:`FrozenEncoder`, which fixes the contract used by the rest of
the harness: frames in, an :class:`EncoderOutput` with patch-level and pooled embeddings out,
and **no trainable parameters**.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class EncoderOutput:
    """Both granularities every encoder exposes.

    Attributes:
        patch: ``(batch, time, tokens, embed_dim)`` patch-level embeddings per frame.
        pooled: ``(batch, time, embed_dim)`` one embedding per frame (the input to our heads).
    """

    patch: Tensor
    pooled: Tensor

    def __post_init__(self) -> None:
        if self.patch.ndim != 4 or self.pooled.ndim != 3:
            raise ValueError(
                f"patch must be 4-D and pooled 3-D, got {self.patch.shape} / {self.pooled.shape}"
            )
        if (
            self.patch.shape[:2] != self.pooled.shape[:2]
            or self.patch.shape[-1] != self.pooled.shape[-1]
        ):
            raise ValueError("patch and pooled must agree on (batch, time) and embed_dim")


class FrozenEncoder(nn.Module, ABC):
    """Base class for frozen encoders.

    Contract:
        * ``forward(frames)`` takes ``(batch, time, channels, height, width)`` float frames in
          ``[0, 1]`` and returns an :class:`EncoderOutput`.
        * every parameter has ``requires_grad=False`` and the module is in ``eval()`` mode.
          :meth:`assert_frozen` is called by the trainer before the first optimiser step.
    """

    embed_dim: int

    @abstractmethod
    def forward(self, frames: Tensor) -> EncoderOutput:
        """Encode ``(B, T, C, H, W)`` frames into patch-level and pooled embeddings."""

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
    def encode_all(self, frames: Tensor) -> EncoderOutput:
        """``forward`` under ``no_grad``."""
        return self.forward(frames)

    @torch.no_grad()
    def encode(self, frames: Tensor) -> Tensor:
        """Pooled ``(B, T, embed_dim)`` latents; the entry point tasks and trainers use."""
        return self.forward(frames).pooled

    @torch.no_grad()
    def encode_patches(self, frames: Tensor) -> Tensor:
        """Patch-level ``(B, T, N, embed_dim)`` latents (for Meta's reference predictor)."""
        return self.forward(frames).patch


__all__ = ["EncoderOutput", "FrozenEncoder"]
