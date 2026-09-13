"""Datasets: DROID episodes, Cosmos 3 generated trajectories, and a synthetic CPU fixture.

All datasets yield the same batch dict::

    {
        "frames":  float tensor (B, T, C, H, W) in [0, 1],
        "actions": float tensor (B, T - 1, action_dim), normalised,
        "meta":    list of dicts with at least embodiment / task / primitive labels,
    }

so encoders, heads and tasks never branch on the data source.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.utils.data import Dataset


def build_dataset(data_cfg: Mapping[str, Any], split: str = "train") -> Dataset:
    """Instantiate the dataset named by ``data_cfg['name']``."""
    name = data_cfg["name"]
    if name == "synthetic":
        from hyperbolic_world_model.data.synthetic import SyntheticTrajectoryDataset

        return SyntheticTrajectoryDataset.from_config(data_cfg, split=split)
    if name == "droid":
        from hyperbolic_world_model.data.droid import DroidDataset

        return DroidDataset.from_config(data_cfg, split=split)
    if name == "cosmos3_generated":
        from hyperbolic_world_model.data.cosmos3.dataset import Cosmos3TrajectoryDataset

        return Cosmos3TrajectoryDataset.from_config(data_cfg, split=split)
    raise KeyError(f"unknown dataset {name!r}; expected synthetic, droid or cosmos3_generated")


__all__ = ["build_dataset"]
