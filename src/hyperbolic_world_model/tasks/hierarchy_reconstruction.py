"""Hierarchy reconstruction: does the latent space recover embodiment > task > primitive?

Hypothesis: distances between per-trajectory latents (Fréchet mean over time) reflect the tree
distance between their metadata nodes better in hyperbolic space than in Euclidean space, at equal
dimension, measured by average distortion (lower is better) and mAP (higher is better) from
``metrics/distortion.py``, and the distance from the origin (``manifold.dist0``) increases with
tree depth. Falsified if the best swept curvature does not improve both distortion and mAP over
the Euclidean head at the same dimension by more than the seed standard deviation, or if the
depth correlation is not positive.

Procedure (all pooling is a Fréchet mean on ``bundle.manifold``, never a coordinate average):

1. encode every trajectory, embed it with the head, pool over time -> one point per trajectory;
2. build the tree with :func:`~hyperbolic_world_model.data.hierarchies.build_hierarchy` from the
   trajectory metadata and map each trajectory to its leaf;
3. embed every tree node as the Fréchet mean of the trajectories in its subtree (the root is the
   mean of everything);
4. score the node embedding against the hop-count metric and adjacency of the tree
   (``average_distortion``, ``mean_average_precision``) and correlate ``dist0`` with depth;
5. repeat 3-4 on ``n_seeds`` bootstrap resamples of the trajectories within each leaf to report
   the within-run spread of every metric.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from hyperbolic_world_model.data.hierarchies import (
    LEVELS,
    Hierarchy,
    build_hierarchy,
    leaf_index_for_metadata,
)
from hyperbolic_world_model.geometry.base import Manifold
from hyperbolic_world_model.metrics.distortion import average_distortion, mean_average_precision
from hyperbolic_world_model.models.registry import ModelBundle
from hyperbolic_world_model.tasks.base import Task, TaskResult, frechet_mean, spearman

SCORES = ("average_distortion", "map", "depth_spearman")


def subtree_membership(tree: Hierarchy, leaf_of_item: Tensor) -> Tensor:
    """Boolean ``(n_nodes, n_items)``: item ``j`` lies in the subtree of node ``i``."""
    n_items = int(leaf_of_item.shape[0])
    member = torch.zeros(tree.n_nodes, n_items, dtype=torch.bool)
    for j in range(n_items):
        for node in tree.ancestors(int(leaf_of_item[j])):
            member[node, j] = True
    return member


def node_embeddings(
    manifold: Manifold,
    member: Tensor,
    points: Tensor,
    weights: Tensor | None = None,
    n_iter: int = 100,
) -> Tensor:
    """Fréchet mean of the member points of every node: ``(n_nodes, d)``.

    Args:
        manifold: geometry of ``points``.
        member: ``(n_nodes, n_items)`` boolean membership (see :func:`subtree_membership`).
        points: ``(n_items, d)`` per-trajectory points on ``manifold``.
        weights: optional ``(n_items,)`` non-negative weights (bootstrap counts); every node
            must keep positive total weight.
    """
    out = []
    for node in range(member.shape[0]):
        idx = member[node].nonzero(as_tuple=True)[0].to(points.device)
        w = None if weights is None else weights.to(points.device)[idx]
        if idx.numel() == 0 or (w is not None and float(w.sum()) <= 0):
            raise ValueError(f"tree node {node} has no trajectories")
        out.append(frechet_mean(manifold, points[idx], dim=0, weights=w, n_iter=n_iter))
    return torch.stack(out)


def score_tree_embedding(
    manifold: Manifold, tree: Hierarchy, emb: Tensor, tree_dist: Tensor, adjacency: Tensor
) -> dict[str, float]:
    """Distortion, mAP and Spearman(depth, dist0) of a node embedding, in ``manifold``."""
    with torch.no_grad():
        radius = manifold.dist0(emb)
    return {
        "average_distortion": average_distortion(emb, tree_dist.to(emb.device), manifold),
        "map": mean_average_precision(emb, adjacency.to(emb.device), manifold),
        "depth_spearman": spearman(torch.tensor(tree.depth, dtype=torch.float64), radius),
    }


class HierarchyReconstructionTask(Task):
    """Score the head's latents against the metadata tree in the head's own geometry.

    Args:
        batch_size: encoder batch size.
        n_seeds: number of bootstrap resamples (within each leaf) used for the ``*_std`` metrics;
            ``0`` reports point estimates only.
        levels: metadata keys, outermost first (defaults to embodiment > task > primitive).
        n_iter: Fréchet-mean iterations.
    """

    name = "hierarchy_reconstruction"

    def __init__(
        self,
        batch_size: int = 32,
        n_seeds: int = 3,
        levels: Sequence[str] = LEVELS,
        n_iter: int = 100,
    ) -> None:
        self.batch_size = int(batch_size)
        self.n_seeds = int(n_seeds)
        self.levels = tuple(str(k) for k in levels)
        self.n_iter = int(n_iter)
        if self.n_seeds < 0:
            raise ValueError("n_seeds must be >= 0")

    @torch.no_grad()
    def trajectory_latents(
        self, bundle: ModelBundle, dataset: Dataset, device: str = "cpu"
    ) -> tuple[Tensor, list[dict]]:
        """One point per trajectory (Fréchet mean over time) and the matching metadata."""
        m = bundle.manifold
        pts, metas = [], []
        for batch in self.iter_embedded(bundle, dataset, self.batch_size, None, device):
            pts.append(frechet_mean(m, batch.latents, dim=1, n_iter=self.n_iter))
            metas.extend(batch.meta)
        if not pts:
            raise ValueError("dataset is empty")
        return torch.cat(pts), metas

    @staticmethod
    def bootstrap_weights(leaf_of_item: Tensor, seed: int) -> Tensor:
        """Resample trajectories with replacement within each leaf; returns per-item counts."""
        gen = torch.Generator().manual_seed(int(seed))
        counts = torch.zeros(leaf_of_item.shape[0], dtype=torch.float64)
        for leaf in leaf_of_item.unique():
            idx = (leaf_of_item == leaf).nonzero(as_tuple=True)[0]
            draw = idx[torch.randint(len(idx), (len(idx),), generator=gen)]
            counts.index_add_(0, draw, torch.ones(len(draw), dtype=torch.float64))
        return counts

    @torch.no_grad()
    def run(self, bundle: ModelBundle, dataset: Dataset, device: str = "cpu") -> TaskResult:
        m = bundle.manifold
        points, metas = self.trajectory_latents(bundle, dataset, device)
        tree = build_hierarchy(metas, self.levels)
        leaf_of_item = leaf_index_for_metadata(tree, metas, self.levels)
        member = subtree_membership(tree, leaf_of_item)
        tree_dist, adjacency = tree.tree_distance_matrix(), tree.adjacency()

        emb = node_embeddings(m, member, points, n_iter=self.n_iter)
        point = score_tree_embedding(m, tree, emb, tree_dist, adjacency)

        boots = []
        for seed in range(self.n_seeds):
            w = self.bootstrap_weights(leaf_of_item, seed)
            emb_s = node_embeddings(m, member, points, weights=w, n_iter=self.n_iter)
            boots.append(score_tree_embedding(m, tree, emb_s, tree_dist, adjacency))
        spread = {}
        for key in SCORES:
            vals = torch.tensor([b[key] for b in boots], dtype=torch.float64)
            spread[f"{key}_std"] = float(vals.std()) if len(vals) >= 2 else float("nan")

        radius = m.dist0(emb).cpu()
        depth = torch.tensor(tree.depth, dtype=torch.float64)
        rows = []
        for d in sorted(set(tree.depth)):
            mask = depth == d
            rows.append(
                {
                    "depth": int(d),
                    "level": tree.levels[int(mask.nonzero()[0])],
                    "n_nodes": int(mask.sum()),
                    "mean_dist0": float(radius[mask].mean()),
                    "std_dist0": float(radius[mask].std()) if int(mask.sum()) > 1 else 0.0,
                }
            )
        res = self.result(
            bundle,
            **point,
            **spread,
            n_nodes=tree.n_nodes,
            n_leaves=len(tree.leaves()),
            n_trajectories=points.shape[0],
            n_bootstrap=self.n_seeds,
        )
        res.curves = pd.DataFrame(rows)
        return res


__all__ = [
    "SCORES",
    "HierarchyReconstructionTask",
    "node_embeddings",
    "score_tree_embedding",
    "subtree_membership",
]
