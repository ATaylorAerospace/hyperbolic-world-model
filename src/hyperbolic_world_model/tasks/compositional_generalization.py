"""Compositional generalisation: held-out (embodiment, primitive) combinations.

Hypothesis: if the latent space encodes the hierarchy, a head trained on all embodiments and all
primitives but *not* on every combination will roll out unseen combinations with an error gap
(unseen minus seen) that is smaller in hyperbolic space than in Euclidean space. Falsified if the
gap is not smaller for the best swept curvature, or if it is smaller only because seen-combination
error got worse.

TODO(phase 2): implement with ``Cosmos3TrajectoryDataset.split_by_combination(cfg.data.holdout_combinations)``
(or the DROID equivalent) and reuse
:class:`~hyperbolic_world_model.tasks.latent_rollout.LatentRolloutTask` on each subset.
"""

from __future__ import annotations

from torch.utils.data import Dataset

from hyperbolic_world_model.models.registry import ModelBundle
from hyperbolic_world_model.tasks.base import Task, TaskResult


class CompositionalGeneralizationTask(Task):
    name = "compositional_generalization"

    def __init__(
        self, horizon: int = 8, batch_size: int = 32, holdout: tuple[str, ...] = ()
    ) -> None:
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self.holdout = tuple(holdout)

    def run(self, bundle: ModelBundle, dataset: Dataset, device: str = "cpu") -> TaskResult:
        raise NotImplementedError(
            "compositional_generalization is pending phase 2; see module docstring"
        )


__all__ = ["CompositionalGeneralizationTask"]
