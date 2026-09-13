"""Rollout error measured in the native geometry of each model.

The core rule of the harness: a Euclidean head is scored with Euclidean distance, a Poincaré head
with Poincaré distance, a Lorentz head with Lorentz distance. Both quantities are geodesic
distances on their own manifold, so they are comparable *as regression errors in the space the
model actually predicts in*, without ever mapping one model's latents into another model's space.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hyperbolic_world_model.geometry.base import Manifold


def geodesic_error(manifold: Manifold, pred: Tensor, target: Tensor) -> Tensor:
    """Per-sample geodesic distance between predicted and target latents.

    Args:
        manifold: geometry in which both ``pred`` and ``target`` live.
        pred: ``(..., d)`` predicted points on ``manifold``.
        target: ``(..., d)`` ground-truth points on ``manifold``.

    Returns:
        ``(...)`` tensor of distances, never reduced so callers can bootstrap or bucket it.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
        )
    return manifold.dist(pred, target)


def geodesic_error_per_horizon(
    manifold: Manifold, pred: Tensor, target: Tensor, reduce: str = "mean"
) -> Tensor:
    """Rollout error as a function of horizon.

    Args:
        manifold: geometry of the latents.
        pred: ``(batch, horizon, d)`` open-loop predictions.
        target: ``(batch, horizon, d)`` encoder latents of the true future frames.
        reduce: ``"mean"`` or ``"median"`` over the batch axis.

    Returns:
        ``(horizon,)`` tensor: error at step ``h`` averaged over the batch.
    """
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError("expected (batch, horizon, d) tensors")
    err = geodesic_error(manifold, pred, target)  # (batch, horizon)
    if reduce == "mean":
        return err.mean(dim=0)
    if reduce == "median":
        return err.median(dim=0).values
    raise ValueError(f"unknown reduce {reduce!r}")


def normalised_geodesic_error(
    manifold: Manifold, pred: Tensor, target: Tensor, context: Tensor
) -> Tensor:
    """Geodesic error divided by the distance the latent actually travelled from ``context``.

    A rollout that predicts "nothing moves" scores exactly 1 on this metric regardless of
    geometry, which makes it the right quantity to compare across curvatures where absolute
    distance scales differ.
    """
    moved = manifold.dist(context, target).clamp_min(torch.finfo(target.dtype).eps)
    return geodesic_error(manifold, pred, target) / moved


__all__ = ["geodesic_error", "geodesic_error_per_horizon", "normalised_geodesic_error"]
