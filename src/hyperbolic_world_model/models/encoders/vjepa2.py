"""Frozen V-JEPA 2 video encoder loaded from Hugging Face.

Only the encoder is used. The V-JEPA 2-AC predictor is **not** loaded from upstream: the whole
point of this repository is to retrain that head under different latent geometries, so the
predictor comes from ``models/predictors/`` and is trained by ``training/train_predictor.py``.

There is deliberately no training path in this module: :func:`load_vjepa2_encoder` freezes every
parameter and the :class:`VJEPA2Encoder` forward runs under ``torch.no_grad``.

Weights: https://huggingface.co/facebook/vjepa2-vitl-fpc64-256 (MIT-licensed code, see
THIRD_PARTY_LICENSES.md). Nothing is downloaded at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import Tensor

from hyperbolic_world_model.models.encoders import FrozenEncoder

DEFAULT_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"


class VJEPA2Encoder(FrozenEncoder):
    """Wraps a ``transformers`` V-JEPA 2 model and pools its tubelet tokens to per-frame latents.

    Args:
        model: a loaded ``transformers.VJEPA2Model`` (already frozen by the loader).
        pooling: ``"mean"`` averages the patch tokens of each temporal slot; ``"tokens"`` returns
            all tokens per slot (``(B, T', N, D)``) for tasks that need spatial structure.
    """

    def __init__(self, model: torch.nn.Module, pooling: str = "mean") -> None:
        super().__init__()
        self.model = model
        self.pooling = pooling
        self.embed_dim = int(model.config.hidden_size)
        self.tubelet_size = int(getattr(model.config, "tubelet_size", 2))
        self.freeze()

    def forward(self, frames: Tensor) -> Tensor:
        """Encode ``(B, T, C, H, W)`` frames; returns ``(B, T // tubelet_size, embed_dim)``.

        TODO(phase 2): validate the token layout (temporal-major, then spatial patches) against a
        real checkpoint and confirm the mean-pool matches the upstream V-JEPA 2-AC probe input.
        Until that check has been run on downloaded weights this method refuses to be called.
        """
        raise NotImplementedError(
            "VJEPA2Encoder.forward is pending validation against real weights "
            "(see TODO in models/encoders/vjepa2.py); use models=synthetic for smoke runs."
        )


def load_vjepa2_encoder(
    model_id: str = DEFAULT_MODEL_ID,
    revision: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> VJEPA2Encoder:
    """Download (if not cached) and load the V-JEPA 2 encoder with all parameters frozen.

    Args:
        model_id: Hugging Face repository id.
        revision: optional commit hash for reproducibility (recorded in ``outputs/*/config.yaml``).
        cache_dir: overrides ``CKPT_ROOT/encoders/vjepa2`` if given.
        device: target device.
        dtype: parameter dtype; float32 on CPU, bfloat16 is fine on GPU for inference.

    Returns:
        A frozen :class:`VJEPA2Encoder`. ``assert_frozen`` passes on the returned module.
    """
    from transformers import AutoModel

    cache = (
        Path(cache_dir)
        if cache_dir
        else Path(os.environ.get("CKPT_ROOT", "checkpoints")) / "encoders" / "vjepa2"
    )
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=str(cache),
        token=os.environ.get("HF_TOKEN") or None,
        torch_dtype=dtype,
    )
    model.to(device)
    encoder = VJEPA2Encoder(model)
    encoder.assert_frozen()
    return encoder


__all__ = ["DEFAULT_MODEL_ID", "VJEPA2Encoder", "load_vjepa2_encoder"]
