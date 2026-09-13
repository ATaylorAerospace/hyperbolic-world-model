"""Frozen DINOv2 image encoder for the DINO-WM secondary subject.

DINO-WM (Zhou et al., 2024) encodes each frame independently with DINOv2 and trains a ViT
predictor on the patch tokens. Here the encoder is frozen and its CLS embedding (or mean patch
embedding) per frame is exposed through the :class:`FrozenEncoder` contract so the same predictor
heads and metrics apply unchanged.

Weights: https://huggingface.co/facebook/dinov2-base (Apache 2.0). Nothing is downloaded at import.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import Tensor

from hyperbolic_world_model.models.encoders import FrozenEncoder

DEFAULT_MODEL_ID = "facebook/dinov2-base"


class DINOv2Encoder(FrozenEncoder):
    """Per-frame DINOv2 features.

    Args:
        model: a loaded ``transformers.Dinov2Model``.
        pooling: ``"cls"`` uses the CLS token; ``"mean"`` averages patch tokens.
        image_mean / image_std: ImageNet normalisation applied to ``[0, 1]`` frames.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pooling: str = "cls",
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        self.model = model
        self.pooling = pooling
        self.embed_dim = int(model.config.hidden_size)
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 3, 1, 1))
        self.freeze()

    def forward(self, frames: Tensor) -> Tensor:
        """Encode ``(B, T, 3, H, W)`` frames in ``[0, 1]`` to ``(B, T, embed_dim)``.

        TODO(phase 2): confirm the DINOv2 ``patch_size``-divisible resize policy used by DINO-WM
        (224x224 centre crop) and apply it here rather than assuming pre-resized inputs.
        """
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError(f"expected (B, T, 3, H, W), got {tuple(frames.shape)}")
        b, t = frames.shape[:2]
        images = frames.reshape(b * t, *frames.shape[2:])
        images = (images - self.image_mean) / self.image_std
        out = self.model(pixel_values=images)
        if self.pooling == "cls":
            feats = out.last_hidden_state[:, 0]
        else:
            feats = out.last_hidden_state[:, 1:].mean(dim=1)
        return feats.reshape(b, t, self.embed_dim)


def load_dinov2_encoder(
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    device: str | torch.device = "cpu",
    pooling: str = "cls",
) -> DINOv2Encoder:
    """Load a frozen DINOv2 encoder from Hugging Face (cached under ``CKPT_ROOT/encoders/dinov2``)."""
    from transformers import AutoModel

    cache = (
        Path(cache_dir)
        if cache_dir
        else Path(os.environ.get("CKPT_ROOT", "checkpoints")) / "encoders" / "dinov2"
    )
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=str(cache),
        token=os.environ.get("HF_TOKEN") or None,
    )
    model.to(device)
    encoder = DINOv2Encoder(model, pooling=pooling)
    encoder.assert_frozen()
    return encoder


__all__ = ["DEFAULT_MODEL_ID", "DINOv2Encoder", "load_dinov2_encoder"]
