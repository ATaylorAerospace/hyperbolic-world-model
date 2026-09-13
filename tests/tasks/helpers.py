"""Tiny bundles and datasets for the task tests (CPU, no downloads, milliseconds)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from hyperbolic_world_model.data.synthetic import SyntheticSpec, SyntheticTrajectoryDataset
from hyperbolic_world_model.models.registry import ModelBundle, build_model

FRAME_SHAPE = (1, 8, 8)
ENC_DIM, LAT, ACT = 16, 8, 4

GEOMETRIES = [
    pytest.param(("euclidean", 0.0), id="euclidean"),
    pytest.param(("poincare", -1.0), id="poincare"),
    pytest.param(("poincare", -0.5), id="poincare(c=-0.5)"),
    pytest.param(("lorentz", -1.0), id="lorentz"),
]


def tiny_bundle(
    geometry: str = "poincare",
    curvature: float = -1.0,
    latent_dim: int = LAT,
    frame_shape: tuple[int, int, int] = FRAME_SHAPE,
    action_dim: int = ACT,
    seed: int = 0,
    max_step: float = 5.0,
    head_type: str | None = None,
    perturb: float = 0.0,
) -> ModelBundle:
    """Synthetic frozen encoder + head on the requested geometry.

    ``perturb`` adds seeded noise to the (zero-initialised) output layer so the head moves;
    without it every head is a static predictor.
    """
    head_type = head_type or ("euclidean" if geometry == "euclidean" else "hyperbolic")
    cfg = {
        "geometry": {"name": geometry, "curvature": curvature},
        "models": {
            "name": "synthetic",
            "embed_dim": ENC_DIM,
            "encoder": {"frozen": True, "frame_shape": list(frame_shape), "seed": 0},
            "head": {
                "type": head_type,
                "latent_dim": latent_dim,
                "hidden_dim": 32,
                "n_layers": 2,
                "action_embed_dim": 8,
                "max_step": max_step,
                "seed": seed,
            },
        },
    }
    bundle = build_model(cfg, action_dim=action_dim)
    if perturb:
        gen = torch.Generator().manual_seed(seed + 1)
        out = bundle.predictor.dynamics[-1]
        with torch.no_grad():
            out.weight.add_(perturb * torch.randn(out.weight.shape, generator=gen))
            out.bias.add_(perturb * torch.randn(out.bias.shape, generator=gen))
    return bundle


def synthetic_dataset(n_episodes: int = 32, horizon: int = 6, split: str = "val"):
    return SyntheticTrajectoryDataset(
        SyntheticSpec(
            n_episodes=n_episodes, horizon=horizon, frame_shape=FRAME_SHAPE, action_dim=ACT
        ),
        split=split,
    )


class BranchingDataset(Dataset):
    """In-memory branching rollouts: every branch of a prompt shares its start frame."""

    def __init__(
        self,
        n_prompts: int = 3,
        n_branches: int = 3,
        horizon: int = 5,
        frame_shape: tuple[int, int, int] = FRAME_SHAPE,
        action_dim: int = ACT,
        seed: int = 0,
    ) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.items: list[dict[str, Any]] = []
        self._groups: dict[str, list[int]] = {}
        emb = ["arm_a", "arm_b"]
        prim = ["reach", "grasp", "push"]
        for p in range(n_prompts):
            start = torch.rand(1, *frame_shape, generator=gen)
            for b in range(n_branches):
                rest = torch.rand(horizon, *frame_shape, generator=gen)
                actions = torch.randn(horizon, action_dim, generator=gen)
                self._groups.setdefault(f"p{p}", []).append(len(self.items))
                self.items.append(
                    {
                        "frames": torch.cat([start, rest]),
                        "actions": actions,
                        "meta": {
                            "prompt_id": f"p{p}",
                            "branch_id": f"b{b}",
                            "embodiment": emb[p % 2],
                            "task": f"task_{p}",
                            "primitive": prim[b % 3],
                        },
                    }
                )
        self.action_dim = action_dim

    def branches(self) -> dict[str, list[int]]:
        return dict(self._groups)

    def branch_pairs(self) -> list[tuple[int, int]]:
        pairs = []
        for idxs in self._groups.values():
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    pairs.append((idxs[a], idxs[b]))
        return pairs

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


def write_cosmos_fixture(
    root: Path,
    n_prompts: int = 2,
    n_branches: int = 2,
    n_frames: int = 6,
    size: int = 8,
    action_dim: int = ACT,
    seed: int = 0,
) -> Path:
    """Hand-written ``manifest.jsonl`` + ``frames.npz`` in the Cosmos 3 generated-data layout."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for p in range(n_prompts):
        start = rng.integers(0, 255, size=(1, size, size, 3), dtype=np.uint8)
        for b in range(n_branches):
            rest = rng.integers(0, 255, size=(n_frames - 1, size, size, 3), dtype=np.uint8)
            frames = np.concatenate([start, rest])
            actions = rng.normal(size=(n_frames - 1, action_dim)).astype(np.float32)
            d = root / "rollouts" / f"ep{p}" / f"b{b}"
            d.mkdir(parents=True)
            np.savez_compressed(d / "frames.npz", frames=frames, actions=actions)
            records.append(
                {
                    "prompt_id": f"ep{p}",
                    "branch_id": f"b{b}",
                    "embodiment": ["franka", "ur5"][p % 2],
                    "task": "pick",
                    "primitive": ["reach", "grasp"][b % 2],
                    "frames_path": str((d / "frames.npz").relative_to(root)),
                    "mp4_path": None,
                    "latents_path": None,
                    "num_frames": n_frames,
                    "height": size,
                    "width": size,
                    "action_dim": action_dim,
                    "seed": seed,
                }
            )
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return root
