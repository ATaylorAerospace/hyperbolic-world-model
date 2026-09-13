"""Deterministic synthetic encoder used by the CPU smoke experiment and tests.

It is a fixed, seeded random linear projection of the flattened frame. It has no learned
parameters, downloads nothing and runs in milliseconds, which is exactly what CI needs to
exercise the end-to-end path (encode -> project onto manifold -> predict -> score in native
geometry) without a real checkpoint. It is registered as ``models=synthetic`` and must never be
used for reported results. Its "patch" output is a single token per frame.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hyperbolic_world_model.models.encoders import EncoderOutput, FrozenEncoder


class SyntheticEncoder(FrozenEncoder):
    """Frozen random projection ``(C * H * W) -> embed_dim``.

    Args:
        frame_shape: ``(channels, height, width)`` of the input frames.
        embed_dim: output latent dimension.
        seed: seed for the projection so runs are reproducible.
    """

    def __init__(self, frame_shape: tuple[int, int, int], embed_dim: int, seed: int = 0) -> None:
        super().__init__()
        self.frame_shape = tuple(int(s) for s in frame_shape)
        self.embed_dim = int(embed_dim)
        in_dim = int(torch.tensor(self.frame_shape).prod())
        gen = torch.Generator().manual_seed(seed)
        weight = torch.randn(in_dim, self.embed_dim, generator=gen) / in_dim**0.5
        self.register_buffer("weight", weight)
        self.freeze()

    def forward(self, frames: Tensor) -> EncoderOutput:
        if frames.ndim != 5:
            raise ValueError(f"expected (B, T, C, H, W), got {tuple(frames.shape)}")
        b, t = frames.shape[:2]
        flat = frames.reshape(b, t, -1).to(self.weight.dtype)
        pooled = flat @ self.weight
        return EncoderOutput(patch=pooled.unsqueeze(2), pooled=pooled)


__all__ = ["SyntheticEncoder"]
