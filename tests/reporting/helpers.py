"""A fake ``outputs/`` tree with a Euclidean baseline and a Poincaré curvature sweep."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

DIMS = (8, 16)
SEEDS = (0, 1)
CURVATURES = (-0.5, -1.0)


def _value(geometry: str, curvature: float, dim: int, seed: int, metric: str) -> float:
    """Deterministic, ordered values: Poincaré K=-1 is best; seeds differ by 0.01."""
    boost = 0.0 if geometry == "euclidean" else (0.2 if curvature == -1.0 else 0.1)
    if metric == "geodesic_error_hmax":
        return 1.0 / math.log2(dim) - boost + 0.01 * seed
    if metric == "map":
        return 0.5 + 0.05 * math.log2(dim) + boost + 0.01 * seed
    raise KeyError(metric)


def write_run(
    root: Path,
    experiment: str,
    geometry: str,
    curvature: float,
    dim: int,
    seed: int,
    model: str = "synthetic",
    offset: float = 0.0,
) -> Path:
    """One fake run; ``offset`` shifts every value so a second model has different numbers."""
    d = root / experiment / f"geometry={geometry},K={curvature},dim={dim},seed={seed}"
    d.mkdir(parents=True, exist_ok=True)
    err = _value(geometry, curvature, dim, seed, "geodesic_error_hmax") + offset
    payload = {
        "experiment": experiment,
        "geometry": geometry,
        "curvature": curvature,
        "latent_dim": dim,
        "seed": seed,
        "model": model,
        "train_history": [{"epoch": 0, "loss": 1.0}],
        "tasks": {
            "latent_rollout": {
                "task": "latent_rollout",
                "geometry": geometry,
                "curvature": curvature,
                "metrics": {"geodesic_error_hmax": err, "n_items": 32},
            },
            "hierarchy_reconstruction": {
                "task": "hierarchy_reconstruction",
                "geometry": geometry,
                "curvature": curvature,
                "metrics": {
                    "map": _value(geometry, curvature, dim, seed, "map") + offset,
                    "map_std": 0.01,
                    "n_nodes": 27,
                },
            },
        },
    }
    (d / "metrics.json").write_text(json.dumps(payload, indent=2))
    horizons = [1, 2, 3, 4]
    pd.DataFrame(
        {
            "horizon": horizons,
            "geodesic_error": [err * h / 4 for h in horizons],
            "static_baseline_error": [err * h / 4 * 1.1 for h in horizons],
            "normalised_error": [1 / 1.1] * 4,
        }
    ).to_csv(d / "latent_rollout_curves.csv", index=False)
    pd.DataFrame(
        {
            "depth": [0, 1, 2],
            "level": ["root", "embodiment", "task"],
            "n_nodes": [1, 2, 4],
            "mean_dist0": [0.0, 0.5 + 0.1 * seed, 1.0],
            "std_dist0": [0.0, 0.1, 0.1],
        }
    ).to_csv(d / "hierarchy_reconstruction_curves.csv", index=False)
    return d


def write_outputs_tree(
    root: Path, with_sweep: bool = True, model: str = "synthetic", offset: float = 0.0
) -> Path:
    for dim in DIMS:
        for seed in SEEDS:
            write_run(root, "baseline_euclidean", "euclidean", 0.0, dim, seed, model, offset)
            if with_sweep:
                for k in CURVATURES:
                    write_run(root, "poincare_sweep", "poincare", k, dim, seed, model, offset)
    return root
