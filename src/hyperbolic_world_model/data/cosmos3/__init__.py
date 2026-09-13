"""NVIDIA Cosmos as a data generator and a latent-space probe target.

Cosmos is **never fine-tuned, never modified, and its generation quality is never reported as a
result** (see ``docs/cosmos3_usage.md``). Two inference-only uses:

1. :mod:`generate` - batch script turning (start frame, action sequence) into a video rollout,
   producing controlled trajectories with branching futures.
2. :mod:`extract_latents` - pull the Cosmos video tokenizer's latents so
   :mod:`hyperbolic_world_model.metrics.gromov_hyperbolicity` can probe them.

:mod:`dataset` loads the generated trajectories in the shared batch format for the
long-horizon-consistency and compositional-generalisation tasks.

Layout under ``DATA_ROOT/cosmos3_generated/``::

    rollouts/<prompt_id>/<branch_id>/frames.npz    frames uint8 (T, H, W, 3), actions float32 (T-1, a)
    rollouts/<prompt_id>/<branch_id>/rollout.mp4   optional preview (only if torchvision is importable)
    rollouts/<prompt_id>/<branch_id>/meta.json     labels, seed, model, generation settings
    latents/<prompt_id>/<branch_id>.npz            tokenizer latents float16 (T', D) + meta
    manifest.jsonl                                 one line per rollout (source of truth for the dataset)
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_SUBDIR = "cosmos3_generated"


def default_root(data_root: str | Path | None = None) -> Path:
    """``DATA_ROOT/cosmos3_generated`` (``DATA_ROOT`` from the environment, default ``data``)."""
    import os

    base = Path(data_root) if data_root is not None else Path(os.environ.get("DATA_ROOT") or "data")
    return base / DEFAULT_SUBDIR


__all__ = ["DEFAULT_SUBDIR", "default_root"]
