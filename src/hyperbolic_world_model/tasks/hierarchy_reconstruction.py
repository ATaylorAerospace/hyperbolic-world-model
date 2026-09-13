"""Hierarchy reconstruction: does the latent space recover embodiment > task > primitive?

Hypothesis: distances between per-trajectory latents (mean-pooled over time) reflect the tree
distance between their metadata nodes better in hyperbolic space than in Euclidean space, at equal
dimension, measured by average distortion (lower is better) and mAP (higher is better) from
``metrics/distortion.py``. Falsified if the best swept curvature does not improve either metric
over the Euclidean head at the same dimension by more than the seed standard deviation.

TODO(phase 2): implement :meth:`HierarchyReconstructionTask.run`:
    1. encode every trajectory, embed with the head, mean-pool over time (Fréchet mean on the
       manifold via iterated ``logmap``/``expmap``, not a Euclidean mean),
    2. build the tree with :func:`hyperbolic_world_model.data.hierarchies.build_hierarchy`,
    3. average leaf latents per tree node, use ``dist0`` for the depth ordering probe,
    4. report ``average_distortion``, ``mean_average_precision`` and Spearman(depth, dist0).
"""

from __future__ import annotations

from torch.utils.data import Dataset

from hyperbolic_world_model.models.registry import ModelBundle
from hyperbolic_world_model.tasks.base import Task, TaskResult


class HierarchyReconstructionTask(Task):
    name = "hierarchy_reconstruction"

    def __init__(self, batch_size: int = 32, n_seeds: int = 3) -> None:
        self.batch_size = int(batch_size)
        self.n_seeds = int(n_seeds)

    def run(self, bundle: ModelBundle, dataset: Dataset, device: str = "cpu") -> TaskResult:
        raise NotImplementedError(
            "hierarchy_reconstruction is pending phase 2; see module docstring"
        )


__all__ = ["HierarchyReconstructionTask"]
