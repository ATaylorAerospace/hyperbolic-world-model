"""Extract Cosmos 3 tokenizer latents for hyperbolicity probing.

The Cosmos 3 video tokenizer maps a clip to a latent grid. We pool those latents per frame and
store them so :mod:`hyperbolic_world_model.metrics.gromov_hyperbolicity` can estimate their
delta-hyperbolicity. This answers a question independent of our own models: *does a large
generative world model already organise its latent space in a tree-like way?*

Usage::

    python -m hyperbolic_world_model.data.cosmos3.extract_latents \
        --rollouts data/cosmos3/rollouts --out data/cosmos3/latents

Output: ``latents/<id>.npz`` with ``latents`` ``(T, D)`` float16 and ``meta`` JSON string.

TODO(phase 2): implement :func:`load_tokenizer` and :func:`encode_video` against the official
Cosmos tokenizer API; the encoder half only, decoder never loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def load_tokenizer(model_id: str, revision: str | None, device: str) -> torch.nn.Module:
    """Load the Cosmos 3 tokenizer encoder (frozen). TODO(phase 2)."""
    raise NotImplementedError("load_tokenizer: pending phase 2")


def encode_video(tokenizer: torch.nn.Module, video: np.ndarray, pooling: str = "mean") -> np.ndarray:
    """``(T, H, W, 3)`` uint8 video -> ``(T', D)`` pooled latents. TODO(phase 2)."""
    raise NotImplementedError("encode_video: pending phase 2")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model-id", default="nvidia/Cosmos-3-Nano")
    p.add_argument("--revision", default=None)
    p.add_argument("--pooling", choices=["mean", "tokens"], default="mean")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raise NotImplementedError(f"Cosmos 3 latent extraction is pending phase 2 (args: {vars(args)})")


if __name__ == "__main__":
    main()
