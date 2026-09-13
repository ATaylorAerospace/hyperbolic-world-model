"""Rollout error measured in the native geometry of each model.

The core rule of the harness: a Euclidean head is scored with Euclidean distance, a Poincaré head
with Poincaré distance, a Lorentz head with Lorentz distance. Both quantities are geodesic
distances on their own manifold, so they are comparable *as regression errors in the space the
model actually predicts in*, without ever mapping one model's latents into another model's space.

Every function here takes a :class:`~hyperbolic_world_model.geometry.base.Manifold` and computes
distances with ``manifold.dist`` only. There is no ``torch.cdist``, no ``.norm()`` and no other
Euclidean fallback in this module; ``tests/metrics/test_geodesic_error.py`` checks the source for
exactly that. Passing anything that is not a ``Manifold`` raises ``TypeError``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hyperbolic_world_model.geometry.base import Manifold

_REDUCTIONS = ("mean", "median", "none")


def _check_manifold(manifold: object) -> Manifold:
    if not isinstance(manifold, Manifold):
        raise TypeError(
            "geodesic errors are computed with manifold.dist and need a "
            f"hyperbolic_world_model.geometry.Manifold; got {type(manifold).__name__}. "
            "There is deliberately no Euclidean fallback."
        )
    return manifold


def _reduce_batch(err: Tensor, reduce: str) -> Tensor:
    if reduce == "mean":
        return err.mean(dim=0)
    if reduce == "median":
        return err.median(dim=0).values
    if reduce == "none":
        return err
    raise ValueError(f"unknown reduce {reduce!r}; expected one of {_REDUCTIONS}")


def _broadcast_context(context: Tensor, target: Tensor) -> Tensor:
    """``(..., d)`` context -> ``(..., 1, d)`` so it broadcasts against ``(..., horizon, d)``."""
    if context.ndim == target.ndim - 1:
        context = context.unsqueeze(-2)
    if context.ndim != target.ndim or context.shape[-1] != target.shape[-1]:
        raise ValueError(
            f"context {tuple(context.shape)} does not broadcast against target "
            f"{tuple(target.shape)}; expected (..., d) or (..., 1, d)"
        )
    return context


def geodesic_error(manifold: Manifold, pred: Tensor, target: Tensor) -> Tensor:
    """Per-sample geodesic distance between predicted and target latents.

    Args:
        manifold: geometry in which both ``pred`` and ``target`` live.
        pred: ``(..., d)`` predicted points on ``manifold``.
        target: ``(..., d)`` ground-truth points on ``manifold``.

    Returns:
        ``(...)`` tensor of distances, never reduced so callers can bootstrap or bucket it.
    """
    manifold = _check_manifold(manifold)
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
        reduce: ``"mean"`` or ``"median"`` over the batch axis, or ``"none"`` to keep it.

    Returns:
        ``(horizon,)`` tensor: error at step ``h`` reduced over the batch (``(batch, horizon)``
        with ``reduce="none"``).
    """
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError("expected (batch, horizon, d) tensors")
    err = geodesic_error(manifold, pred, target)  # (batch, horizon)
    return _reduce_batch(err, reduce)


def static_baseline_error(manifold: Manifold, context: Tensor, target: Tensor) -> Tensor:
    """Error of the "nothing moves" predictor: the distance from ``context`` to each target.

    Args:
        manifold: geometry of the latents.
        context: ``(..., d)`` (or ``(..., 1, d)``) latent of the frame the rollout starts from.
        target: ``(..., horizon, d)`` true future latents.

    Returns:
        ``(..., horizon)`` distances the true latent travelled from the context at each step.
        This is the denominator of every normalised error and the baseline every rollout curve
        is plotted against.
    """
    manifold = _check_manifold(manifold)
    context = _broadcast_context(context, target)
    return manifold.dist(context.expand_as(target), target)


def normalised_geodesic_error(
    manifold: Manifold, pred: Tensor, target: Tensor, context: Tensor
) -> Tensor:
    """Geodesic error divided by the distance the latent actually travelled from ``context``.

    A rollout that predicts "nothing moves" scores exactly 1 on this metric regardless of
    geometry, which makes it the right quantity to compare across curvatures where absolute
    distance scales differ. The ratio is taken per sample; use
    :func:`normalised_geodesic_error_per_horizon` for the batch-level (and better conditioned)
    ratio of means.

    Args:
        manifold: geometry of the latents.
        pred: ``(..., horizon, d)`` predictions.
        target: ``(..., horizon, d)`` true future latents.
        context: ``(..., d)`` or ``(..., 1, d)`` latent the rollout started from.
    """
    moved = static_baseline_error(manifold, context, target)
    moved = moved.clamp_min(torch.finfo(moved.dtype).eps)
    return geodesic_error(manifold, pred, target) / moved


def normalised_geodesic_error_per_horizon(
    manifold: Manifold, pred: Tensor, target: Tensor, context: Tensor, reduce: str = "mean"
) -> Tensor:
    """Ratio of the reduced rollout error to the reduced static-baseline error at each horizon.

    ``reduce`` is applied to numerator and denominator separately before dividing, so the result
    is the batch-level normalised error (exactly 1 for a static predictor in every geometry and
    stable when individual targets barely move).

    Returns:
        ``(horizon,)`` tensor.
    """
    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError("expected (batch, horizon, d) tensors")
    if reduce == "none":
        raise ValueError("reduce='none' is not defined for a ratio of batch statistics")
    err = _reduce_batch(geodesic_error(manifold, pred, target), reduce)
    moved = _reduce_batch(static_baseline_error(manifold, context, target), reduce)
    return err / moved.clamp_min(torch.finfo(moved.dtype).eps)


__all__ = [
    "geodesic_error",
    "geodesic_error_per_horizon",
    "normalised_geodesic_error",
    "normalised_geodesic_error_per_horizon",
    "static_baseline_error",
]
