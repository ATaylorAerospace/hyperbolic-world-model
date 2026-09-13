"""Task interface and the helpers every task shares.

A task evaluates a :class:`~hyperbolic_world_model.models.registry.ModelBundle` on a dataset and
returns a :class:`TaskResult`. Everything distance-based inside a task goes through
``bundle.manifold`` (the geometry the head was trained in), and every result carries that
geometry and curvature so a table can never present a number without its space.

Shared helpers:

* :meth:`Task.iter_embedded` streams the dataset through the frozen encoder and the head's frozen
  embedding in batches, so every task sees ``(B, T, d)`` manifold points and the batch metadata
  without touching the data source.
* :func:`frechet_mean` pools points on the manifold (a Karcher mean by iterated
  ``logmap``/``expmap``) instead of averaging coordinates, which would leave the manifold or bias
  towards the origin.
* :func:`spearman` is the rank correlation used by the hierarchy and long-horizon tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from hyperbolic_world_model.data.synthetic import collate
from hyperbolic_world_model.geometry.base import Manifold
from hyperbolic_world_model.models.registry import ModelBundle


@dataclass
class TaskResult:
    """Scalar metrics plus optional per-horizon curves.

    Attributes:
        task: task name.
        geometry: ``manifold.name`` the metrics were computed in (always recorded so a table can
            never present a number without its geometry).
        curvature: curvature the metrics were computed at.
        metrics: flat ``{name: value}`` mapping of scalars.
        curves: optional tidy DataFrame (e.g. ``horizon, error``) for plotting.
    """

    task: str
    geometry: str
    curvature: float
    metrics: dict[str, float] = field(default_factory=dict)
    curves: pd.DataFrame | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "geometry": self.geometry,
            "curvature": self.curvature,
            "metrics": self.metrics,
        }


@dataclass
class EmbeddedBatch:
    """One batch after the frozen encoder and the head's frozen embedding.

    Attributes:
        latents: ``(B, T, d)`` points on ``bundle.manifold`` (``d`` is the ambient dimension).
        encoder_latents: ``(B, T, D)`` pooled frozen-encoder outputs (Euclidean, the encoder's
            own space).
        actions: ``(B, T - 1, a)`` actions between consecutive frames.
        frames: ``(B, T, C, H, W)`` input frames, on ``device``.
        meta: per-item metadata dicts.
        indices: dataset indices of the items in this batch.
    """

    latents: Tensor
    encoder_latents: Tensor
    actions: Tensor
    frames: Tensor
    meta: list[dict[str, Any]]
    indices: list[int]


class Task(ABC):
    """A task is a callable that evaluates a bundle on a dataset."""

    name: str = "abstract"

    @abstractmethod
    def run(self, bundle: ModelBundle, dataset: Dataset, device: str = "cpu") -> TaskResult:
        """Evaluate and return metrics in the bundle's native geometry."""

    def result(self, bundle: ModelBundle, **metrics: float) -> TaskResult:
        """Helper stamping the geometry and curvature onto a result."""
        return TaskResult(
            task=self.name,
            geometry=bundle.manifold.name,
            curvature=bundle.manifold.curvature,
            metrics={k: float(v) for k, v in metrics.items()},
        )

    @staticmethod
    def loader(
        dataset: Dataset, batch_size: int, indices: Sequence[int] | None = None
    ) -> DataLoader:
        """Deterministic (unshuffled) loader over ``dataset`` or a subset of its indices."""
        ds = dataset if indices is None else Subset(dataset, list(indices))
        return DataLoader(ds, batch_size=int(batch_size), shuffle=False, collate_fn=collate)

    @staticmethod
    @torch.no_grad()
    def iter_embedded(
        bundle: ModelBundle,
        dataset: Dataset,
        batch_size: int,
        indices: Sequence[int] | None = None,
        device: str = "cpu",
    ) -> Iterator[EmbeddedBatch]:
        """Stream ``dataset`` through the frozen encoder and the head's embedding.

        Asserts the encoder is frozen first and puts the head in eval mode; both are invariants
        every task relies on.
        """
        bundle.encoder.assert_frozen()
        head = bundle.predictor.eval()
        idx = list(range(len(dataset))) if indices is None else list(indices)
        loader = Task.loader(dataset, batch_size, idx)
        offset = 0
        for batch in loader:
            frames = batch["frames"].to(device)
            actions = batch["actions"].to(device)
            enc = bundle.encoder.encode(frames)  # (B, T, D)
            z = head.embed(enc)  # (B, T, d) on the manifold
            n = frames.shape[0]
            yield EmbeddedBatch(
                latents=z,
                encoder_latents=enc,
                actions=actions,
                frames=frames,
                meta=batch["meta"],
                indices=idx[offset : offset + n],
            )
            offset += n


def frechet_mean(
    manifold: Manifold,
    points: Tensor,
    dim: int = -2,
    n_iter: int = 100,
    tol: float = 1e-7,
    weights: Tensor | None = None,
    max_backtracks: int = 20,
) -> Tensor:
    """Fréchet (Karcher) mean of ``points`` along ``dim`` on ``manifold``.

    Minimises ``f(x) = sum_i w_i d(x, p_i)^2`` by Riemannian gradient descent: starting from the
    exponential map of the tangent-space mean at the origin, each iteration proposes the classic
    update ``x <- exp_x(sum_i w_i log_x(p_i))`` and halves the step until ``f`` does not increase
    (a backtracking line search). The safeguard matters in hyperbolic space: for points more than
    about two units from the current iterate the Hessian of ``d^2`` exceeds two, so the unit step
    of the plain fixed-point iteration overshoots and diverges. ``f`` is geodesically convex on
    every geometry here, so the descent converges to the unique mean. On the Euclidean manifold
    the first unit step lands on the arithmetic mean exactly.

    Tasks use this to pool latents over time and over trajectories: a coordinate average would
    leave the hyperboloid and pull ball points towards the origin.

    Args:
        manifold: geometry of ``points``.
        points: ``(..., n, d)`` points on ``manifold`` (with ``dim=-2``; any ``dim`` other than
            the last is allowed).
        dim: axis to reduce.
        n_iter: maximum number of descent iterations.
        tol: stop when every mean moved by less than ``tol`` (geodesic units).
        weights: optional non-negative weights broadcastable to ``points.shape[:-1]``; ``None``
            is uniform.
        max_backtracks: maximum step halvings per iteration.

    Returns:
        ``points`` with ``dim`` removed.
    """
    if points.ndim < 2:
        raise ValueError("points must have at least a reduction axis and a coordinate axis")
    dim = dim % points.ndim
    if dim == points.ndim - 1:
        raise ValueError("dim must not be the coordinate (last) axis")
    if points.shape[dim] == 0:
        raise ValueError("cannot take the mean of zero points")
    if weights is None:
        w = torch.full(points.shape[:-1], 1.0, dtype=points.dtype, device=points.device)
    else:
        w = weights.to(points.dtype).expand(points.shape[:-1])
    w = w / w.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(points.dtype).eps)

    def objective(x: Tensor) -> Tensor:
        return (w * manifold.dist(x.unsqueeze(dim), points) ** 2).sum(dim=dim)

    x = manifold.proj(manifold.expmap0((w.unsqueeze(-1) * manifold.logmap0(points)).sum(dim=dim)))
    f = objective(x)
    slack = 16 * torch.finfo(points.dtype).eps
    for _ in range(n_iter):
        u = (w.unsqueeze(-1) * manifold.logmap(x.unsqueeze(dim), points)).sum(dim=dim)
        u = manifold.proj_tan(x, u)
        t = torch.ones_like(f)
        accepted = torch.zeros_like(f, dtype=torch.bool)
        x_try, f_try = x, f
        for _ in range(max_backtracks):
            x_try = manifold.proj(manifold.expmap(x, t.unsqueeze(-1) * u))
            f_try = objective(x_try)
            accepted = f_try <= f * (1 + slack) + slack
            if bool(accepted.all()):
                break
            t = torch.where(accepted, t, t / 2)
        keep = accepted.unsqueeze(-1)
        x_new = torch.where(keep, x_try, x)
        step = manifold.dist(x, x_new).max()
        x, f = x_new, torch.where(accepted, f_try, f)
        if float(step) < tol:
            break
    return x


def spearman(a: Sequence[float] | Tensor, b: Sequence[float] | Tensor) -> float:
    """Spearman rank correlation (average ranks for ties); ``NaN`` if either input is constant."""
    sa = pd.Series(torch.as_tensor(a).detach().cpu().to(torch.float64).numpy())
    sb = pd.Series(torch.as_tensor(b).detach().cpu().to(torch.float64).numpy())
    if len(sa) != len(sb):
        raise ValueError("inputs must have the same length")
    if len(sa) < 2 or sa.nunique() < 2 or sb.nunique() < 2:
        return float("nan")
    return float(sa.corr(sb, method="spearman"))


__all__ = ["EmbeddedBatch", "Task", "TaskResult", "frechet_mean", "spearman"]
