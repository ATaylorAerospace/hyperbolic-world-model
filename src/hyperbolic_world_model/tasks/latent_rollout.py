"""Latent rollout: open-loop prediction error versus horizon.

Hypothesis: for the same latent dimension, the hyperbolic head has lower geodesic rollout error
than the Euclidean head at long horizons because branching futures spread out exponentially in
hyperbolic space and so remain distinguishable. Falsified if, after sweeping curvature, no
curvature beats the Euclidean baseline's normalised error at any horizon beyond the first.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from hyperbolic_world_model.data.synthetic import collate
from hyperbolic_world_model.metrics.geodesic_error import geodesic_error, geodesic_error_per_horizon
from hyperbolic_world_model.models.registry import ModelBundle
from hyperbolic_world_model.tasks.base import Task, TaskResult


class LatentRolloutTask(Task):
    """Open-loop rollout scored with the geodesic error of ``bundle.manifold``.

    Args:
        horizon: number of steps to roll out (``<=`` dataset horizon).
        batch_size: evaluation batch size.
    """

    name = "latent_rollout"

    def __init__(self, horizon: int = 8, batch_size: int = 32) -> None:
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)

    @torch.no_grad()
    def run(self, bundle: ModelBundle, dataset: Dataset, device: str = "cpu") -> TaskResult:
        bundle.encoder.assert_frozen()
        head = bundle.predictor.eval()
        m = bundle.manifold
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate)
        errs, static_errs = [], []
        for batch in loader:
            frames = batch["frames"].to(device)
            actions = batch["actions"].to(device)[:, : self.horizon]
            z = head.embed(bundle.encoder.encode(frames))  # (B, T, d) on the manifold
            target = z[:, 1 : self.horizon + 1]
            pred = head.rollout(z[:, 0], actions)
            errs.append(geodesic_error(m, pred, target))
            # "Nothing moves" baseline: predict the context latent at every horizon.
            static = z[:, :1].expand_as(target)
            static_errs.append(geodesic_error(m, static, target))
        err = torch.cat(errs)  # (N, H)
        static = torch.cat(static_errs)
        per_h = err.mean(0)
        static_per_h = static.mean(0)
        curves = pd.DataFrame(
            {
                "horizon": range(1, self.horizon + 1),
                "geodesic_error": per_h.cpu().numpy(),
                "static_baseline_error": static_per_h.cpu().numpy(),
            }
        )
        res = self.result(
            bundle,
            geodesic_error_h1=float(per_h[0]),
            geodesic_error_hmax=float(per_h[-1]),
            geodesic_error_mean=float(per_h.mean()),
            normalised_error_hmax=float(per_h[-1] / static_per_h[-1].clamp_min(1e-12)),
        )
        res.curves = curves
        return res


__all__ = ["LatentRolloutTask", "geodesic_error_per_horizon"]
