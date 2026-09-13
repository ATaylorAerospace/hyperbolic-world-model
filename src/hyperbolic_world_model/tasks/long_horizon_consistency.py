"""Long-horizon consistency on branching futures (Cosmos 3 generated trajectories).

Hypothesis: two rollouts that share a start frame and diverge in their action sequences should
diverge in latent space at a rate that matches the divergence of the generated videos. In
hyperbolic space a geodesic ball's volume grows exponentially, so many branches fit without
collapsing onto one another. We measure the correlation between latent divergence (geodesic
distance between the two rollouts at horizon ``h``) and pixel/encoder divergence of the
corresponding generated frames. Falsified if the hyperbolic head's correlation is not higher than
the Euclidean head's, or if its latent divergence saturates earlier.

TODO(phase 2): implement using ``Cosmos3TrajectoryDataset.branches`` (requires generated data).
"""

from __future__ import annotations

from torch.utils.data import Dataset

from hyperbolic_world_model.models.registry import ModelBundle
from hyperbolic_world_model.tasks.base import Task, TaskResult


class LongHorizonConsistencyTask(Task):
    name = "long_horizon_consistency"

    def __init__(self, horizon: int = 32, batch_size: int = 16) -> None:
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)

    def run(self, bundle: ModelBundle, dataset: Dataset, device: str = "cpu") -> TaskResult:
        raise NotImplementedError(
            "long_horizon_consistency is pending phase 2; needs Cosmos 3 rollouts"
        )


__all__ = ["LongHorizonConsistencyTask"]
