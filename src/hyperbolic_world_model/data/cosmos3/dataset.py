"""Loader for Cosmos 3 generated trajectories (``data/cosmos3/manifest.jsonl``).

Yields the shared batch format. Because every rollout shares its start frame with its siblings
(branching futures), the loader exposes :meth:`Cosmos3TrajectoryDataset.branches` grouping
trajectories by prompt id, which the long-horizon consistency task uses.

TODO(phase 2): implement video decoding (``.mp4`` -> uint8 frames -> float ``[0, 1]``) and the
manifest index. Depends on :mod:`generate` having produced data.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class Cosmos3TrajectoryDataset(Dataset):
    """Generated trajectories with branching structure.

    TODO(phase 2): implement ``__init__`` (read manifest), ``__len__``, ``__getitem__``, ``branches``.
    """

    def __init__(self, root: Path, split: str, horizon: int, image_size: int) -> None:
        raise NotImplementedError(
            "Cosmos3TrajectoryDataset is pending phase 2; generate data first with "
            "scripts/generate_cosmos3_trajectories.sh"
        )

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], split: str = "train") -> Cosmos3TrajectoryDataset:
        return cls(
            root=Path(cfg["root"]),
            split=split,
            horizon=int(cfg["horizon"]),
            image_size=int(cfg.get("image_size", 224)),
        )

    def branches(self) -> dict[str, list[int]]:
        """Map prompt id -> indices of trajectories sharing that start frame. TODO(phase 2)."""
        raise NotImplementedError


__all__ = ["Cosmos3TrajectoryDataset"]
