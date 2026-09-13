"""Batch generation: start frame + action sequence -> video rollout via Cosmos 3 Nano.

Usage (see ``scripts/generate_cosmos3_trajectories.sh``)::

    python -m hyperbolic_world_model.data.cosmos3.generate \
        --prompts data/cosmos3/prompts --out data/cosmos3/rollouts --seed 0

Input layout (one prompt per directory)::

    prompts/<id>/frame_0.png       start frame
    prompts/<id>/actions.npy       (H, action_dim) action sequence
    prompts/<id>/meta.json         embodiment / task / primitive labels

Output::

    rollouts/<id>/rollout.mp4      generated video
    rollouts/<id>/meta.json        copy of input meta + generation seed + model revision
    manifest.jsonl                 appended one line per trajectory

Design constraints:
    * The Cosmos 3 model is loaded in inference mode only. No parameter of it is ever touched.
    * Generation is seeded per prompt so branching futures (same start frame, different action
      sequences) are reproducible.

TODO(phase 2): implement :func:`load_cosmos3_nano` (Hugging Face ``nvidia/Cosmos-3-Nano`` under
the OpenMDW licence, gated; uses ``HF_TOKEN``), :func:`generate_rollout` and :func:`main`.
The exact action-conditioning API of the Cosmos 3 Nano release must be checked against the model
card before writing the call; do not guess it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def load_cosmos3_nano(model_id: str, revision: str | None, device: str) -> torch.nn.Module:
    """Load Cosmos 3 Nano for inference only (frozen, ``eval()``).

    TODO(phase 2): implement via the official Cosmos inference API.
    """
    raise NotImplementedError("load_cosmos3_nano: pending phase 2 (see docs/cosmos3_usage.md)")


def generate_rollout(
    model: torch.nn.Module,
    start_frame: np.ndarray,
    actions: np.ndarray,
    seed: int,
    num_frames: int,
) -> np.ndarray:
    """Return a ``(num_frames, H, W, 3)`` uint8 video conditioned on ``start_frame`` and ``actions``.

    TODO(phase 2): implement; must be deterministic for a fixed ``seed``.
    """
    raise NotImplementedError("generate_rollout: pending phase 2")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", type=Path, required=True, help="directory of prompt sub-directories")
    p.add_argument("--out", type=Path, required=True, help="output directory for rollouts")
    p.add_argument("--model-id", default="nvidia/Cosmos-3-Nano")
    p.add_argument("--revision", default=None)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point. TODO(phase 2): iterate prompts, call :func:`generate_rollout`, write manifest."""
    args = parse_args(argv)
    raise NotImplementedError(f"Cosmos 3 generation is pending phase 2 (args: {vars(args)})")


if __name__ == "__main__":
    main()
