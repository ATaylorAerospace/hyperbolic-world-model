"""Hierarchy reconstruction: tree recovery scored in the head's geometry with Fréchet pooling."""

from __future__ import annotations

import math

import pytest
import torch

from hyperbolic_world_model.data.hierarchies import build_hierarchy, leaf_index_for_metadata
from hyperbolic_world_model.geometry import Euclidean, PoincareBall
from hyperbolic_world_model.tasks import HierarchyReconstructionTask
from hyperbolic_world_model.tasks.hierarchy_reconstruction import (
    SCORES,
    node_embeddings,
    score_tree_embedding,
    subtree_membership,
)
from tests.tasks.helpers import GEOMETRIES, synthetic_dataset, tiny_bundle

METRICS = {
    *SCORES,
    *(f"{s}_std" for s in SCORES),
    "n_nodes",
    "n_leaves",
    "n_trajectories",
    "n_bootstrap",
}


def _chain_tree():
    """root -> a -> b -> c: a path graph whose hop metric a line recovers exactly."""
    return build_hierarchy([{"embodiment": "a", "task": "b", "primitive": "c"}])


def test_score_tree_embedding_is_perfect_on_a_perfectly_embedded_chain() -> None:
    tree = _chain_tree()
    emb = torch.arange(4, dtype=torch.float64).unsqueeze(1) * 3.0  # scaled line, root at origin
    m = Euclidean()
    scores = score_tree_embedding(m, tree, emb, tree.tree_distance_matrix(), tree.adjacency())
    assert scores["average_distortion"] == pytest.approx(0.0, abs=1e-9)
    assert scores["map"] == pytest.approx(1.0)
    assert scores["depth_spearman"] == pytest.approx(1.0)
    # Reversed depth ordering flips the correlation while distances stay perfect.
    rev = emb.flip(0)
    rev_scores = score_tree_embedding(m, tree, rev, tree.tree_distance_matrix(), tree.adjacency())
    assert rev_scores["average_distortion"] == pytest.approx(0.0, abs=1e-9)
    assert rev_scores["depth_spearman"] == pytest.approx(-1.0)


def test_subtree_membership_and_node_embeddings() -> None:
    meta = [{"embodiment": e, "task": t, "primitive": p} for e in "ab" for t in "xy" for p in "pq"]
    meta = meta * 2  # two trajectories per leaf
    tree = build_hierarchy(meta)
    leaf = leaf_index_for_metadata(tree, meta)
    member = subtree_membership(tree, leaf)
    assert member.shape == (tree.n_nodes, len(meta))
    assert member[0].all()  # root contains everything
    for j, node in enumerate(leaf.tolist()):
        assert member[node, j] and member[tree.parent[node], j]
    assert member[0].sum() == 16
    assert member[tree.index_of(("a",))].sum() == 8 and member[tree.index_of(("b",))].sum() == 8
    assert member[tree.index_of(("a", "x"))].sum() == 4
    pts = torch.randn(len(meta), 3, dtype=torch.float64)
    m = Euclidean()
    emb = node_embeddings(m, member, pts)
    assert emb.shape == (tree.n_nodes, 3)
    assert torch.allclose(emb[0], pts.mean(0))
    for node in range(tree.n_nodes):
        assert torch.allclose(emb[node], pts[member[node]].mean(0))
    # Uniform weights reproduce the unweighted mean; a node with zero total weight is an error.
    assert torch.allclose(node_embeddings(m, member, pts, weights=torch.ones(len(meta))), emb)
    w = torch.ones(len(meta))
    w[leaf == leaf[0]] = 0.0
    with pytest.raises(ValueError, match="no trajectories"):
        node_embeddings(m, member, pts, weights=w)


def test_bootstrap_weights_resample_within_leaves() -> None:
    leaf = torch.tensor([3, 3, 3, 5, 5, 8])
    w0 = HierarchyReconstructionTask.bootstrap_weights(leaf, seed=0)
    assert w0.shape == (6,) and w0.sum() == 6
    assert w0[:3].sum() == 3 and w0[3:5].sum() == 2 and w0[5] == 1
    assert torch.equal(w0, HierarchyReconstructionTask.bootstrap_weights(leaf, seed=0))
    assert any(
        not torch.equal(w0, HierarchyReconstructionTask.bootstrap_weights(leaf, seed=s))
        for s in range(1, 20)
    )


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_run_on_the_synthetic_hierarchy(geom: tuple[str, float]) -> None:
    bundle, ds = tiny_bundle(*geom, perturb=0.2), synthetic_dataset(n_episodes=32, horizon=4)
    res = HierarchyReconstructionTask(batch_size=8, n_seeds=3).run(bundle, ds)
    assert res.geometry == geom[0] and res.curvature == geom[1]
    assert set(res.metrics) == METRICS
    # root + 2 embodiments + 2x4 tasks + 8x2 primitives, 32 trajectories, 16 leaves.
    assert res.metrics["n_nodes"] == 27 and res.metrics["n_leaves"] == 16
    assert res.metrics["n_trajectories"] == 32 and res.metrics["n_bootstrap"] == 3
    assert res.metrics["average_distortion"] >= 0
    assert 0 <= res.metrics["map"] <= 1
    assert -1 <= res.metrics["depth_spearman"] <= 1
    assert all(res.metrics[f"{s}_std"] >= 0 for s in SCORES)
    assert list(res.curves.columns) == ["depth", "level", "n_nodes", "mean_dist0", "std_dist0"]
    assert res.curves["depth"].tolist() == [0, 1, 2, 3]
    assert res.curves["level"].tolist() == ["root", "embodiment", "task", "primitive"]
    assert res.curves["n_nodes"].tolist() == [1, 2, 8, 16]
    assert (res.curves["mean_dist0"] >= 0).all() and res.curves["std_dist0"].iloc[0] == 0.0


def test_run_is_deterministic_and_n_seeds_zero_gives_nan_spread() -> None:
    bundle, ds = tiny_bundle("poincare", -1.0), synthetic_dataset(n_episodes=16, horizon=3)
    a = HierarchyReconstructionTask(batch_size=4, n_seeds=2).run(bundle, ds)
    b = HierarchyReconstructionTask(batch_size=16, n_seeds=2).run(bundle, ds)
    for key in SCORES:
        assert a.metrics[key] == pytest.approx(b.metrics[key], abs=1e-5)
    c = HierarchyReconstructionTask(n_seeds=0).run(bundle, ds)
    assert all(math.isnan(c.metrics[f"{s}_std"]) for s in SCORES)
    with pytest.raises(ValueError, match="n_seeds"):
        HierarchyReconstructionTask(n_seeds=-1)


def test_pooling_happens_on_the_manifold() -> None:
    """Per-trajectory points are Fréchet means over time, not coordinate averages."""
    bundle, ds = (
        tiny_bundle("poincare", -1.0, perturb=0.2),
        synthetic_dataset(n_episodes=6, horizon=3),
    )
    task = HierarchyReconstructionTask(batch_size=6)
    pts, metas = task.trajectory_latents(bundle, ds)
    m = bundle.manifold
    assert isinstance(m, PoincareBall) and pts.shape == (6, bundle.predictor.ambient_dim)
    assert m.check_point(pts).all() and len(metas) == 6
    with torch.no_grad():
        z = bundle.predictor.embed(bundle.encoder.encode(ds.frames[:6]))
    naive = z.mean(1)
    sq = lambda x: (m.dist(x.unsqueeze(1), z) ** 2).sum(1)  # noqa: E731
    assert torch.all(sq(pts) <= sq(naive) + 1e-6)
    assert torch.all(torch.linalg.vector_norm(m.logmap(pts.unsqueeze(1), z).mean(1), dim=-1) < 1e-4)


def test_custom_levels() -> None:
    bundle, ds = tiny_bundle("euclidean", 0.0), synthetic_dataset(n_episodes=16, horizon=2)
    res = HierarchyReconstructionTask(n_seeds=0, levels=("embodiment", "primitive")).run(bundle, ds)
    assert res.metrics["n_nodes"] == 1 + 2 + 4 and res.curves["level"].tolist() == [
        "root",
        "embodiment",
        "task",
    ]
