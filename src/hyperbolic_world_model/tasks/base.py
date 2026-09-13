"""Task interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from torch.utils.data import Dataset

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
            metrics=metrics,
        )


__all__ = ["Task", "TaskResult"]
