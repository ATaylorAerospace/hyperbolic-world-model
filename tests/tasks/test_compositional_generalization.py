"""Compositional generalisation: seen vs held-out (embodiment, primitive) rollout error."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from hyperbolic_world_model.data.cosmos3.dataset import Cosmos3TrajectoryDataset
from hyperbolic_world_model.tasks import CompositionalGeneralizationTask, LatentRolloutTask
from hyperbolic_world_model.tasks.compositional_generalization import (
    normalise_holdout,
    split_by_combination,
    split_dataset_by_combination,
)
from hyperbolic_world_model.training.train_predictor import CONFIG_DIR, run, training_subset
from tests.tasks.helpers import GEOMETRIES, synthetic_dataset, tiny_bundle, write_cosmos_fixture

HOLDOUT = [["arm_b", "grasp"]]


def test_normalise_holdout() -> None:
    assert normalise_holdout(None) == ()
    assert normalise_holdout([("a", "b"), ["c", 1]]) == (("a", "b"), ("c", "1"))
    with pytest.raises(ValueError, match="embodiment, primitive"):
        normalise_holdout([["a"]])


def test_split_by_combination_from_metadata() -> None:
    meta = [{"embodiment": e, "primitive": p} for e in ("a", "b") for p in ("x", "y")] * 2
    seen, held = split_by_combination(meta, [["b", "y"]])
    assert held == [3, 7] and seen == [0, 1, 2, 4, 5, 6]
    with pytest.raises(ValueError, match="not present"):
        split_by_combination(meta, [["b", "z"]])
    with pytest.raises(ValueError, match="removes"):
        split_by_combination(meta, [["b", "x"], ["b", "y"]])
    assert split_by_combination(meta, []) == (list(range(8)), [])


def test_split_dataset_prefers_the_dataset_method(tmp_path: Path) -> None:
    ds = synthetic_dataset(n_episodes=16, horizon=2)
    seen, held = split_dataset_by_combination(ds, HOLDOUT)
    assert len(seen) == 12 and len(held) == 4
    assert all(
        ds.meta[i]["embodiment"] == "arm_b" and ds.meta[i]["primitive"] == "grasp" for i in held
    )

    class Fake:
        def split_by_combination(self, holdout):
            assert holdout == [["arm_b", "grasp"]]
            return [0, 2], [1]

    assert split_dataset_by_combination(Fake(), HOLDOUT) == ([0, 2], [1])
    root = write_cosmos_fixture(tmp_path / "gen", n_prompts=2, n_branches=2, n_frames=4)
    cds = Cosmos3TrajectoryDataset(root, horizon=3, image_size=8)
    seen, held = split_dataset_by_combination(cds, [["ur5", "grasp"]])
    assert len(seen) == 3 and len(held) == 1 and cds.combination(held[0]) == ("ur5", "grasp")


def test_task_requires_a_holdout() -> None:
    with pytest.raises(ValueError, match="holdout_combinations"):
        CompositionalGeneralizationTask(holdout=[])
    with pytest.raises(ValueError, match="holdout_combinations"):
        CompositionalGeneralizationTask()


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_run_reports_seen_unseen_and_gap(geom: tuple[str, float]) -> None:
    bundle, ds = tiny_bundle(*geom, perturb=0.3), synthetic_dataset(n_episodes=32, horizon=5)
    task = CompositionalGeneralizationTask(horizon=4, batch_size=8, holdout=HOLDOUT)
    res = task.run(bundle, ds)
    assert res.geometry == geom[0] and res.curvature == geom[1]
    mt = res.metrics
    assert mt["n_seen"] == 24 and mt["n_unseen"] == 8 and mt["n_holdout_combinations"] == 1
    for prefix in ("seen_", "unseen_"):
        for name in ("geodesic_error_h1", "geodesic_error_hmax", "geodesic_error_mean"):
            assert mt[f"{prefix}{name}"] > 0
    assert mt["gap_geodesic_error_hmax"] == pytest.approx(
        mt["unseen_geodesic_error_hmax"] - mt["seen_geodesic_error_hmax"]
    )
    assert mt["gap_normalised_error_hmax"] == pytest.approx(
        mt["unseen_normalised_error_hmax"] - mt["seen_normalised_error_hmax"]
    )
    assert mt["unseen_to_seen_ratio_hmax"] == pytest.approx(
        mt["unseen_normalised_error_hmax"] / mt["seen_normalised_error_hmax"]
    )
    assert list(res.curves.columns) == [
        "horizon",
        "seen_error",
        "unseen_error",
        "seen_static_baseline_error",
        "unseen_static_baseline_error",
        "seen_normalised_error",
        "unseen_normalised_error",
        "gap_normalised_error",
    ]
    assert res.curves["horizon"].tolist() == [1, 2, 3, 4]
    # Each subset is exactly the latent-rollout task restricted to it.
    seen, held = task.split(ds)
    ref = LatentRolloutTask(horizon=4, batch_size=8).evaluate(bundle, ds, seen).summary("seen_")
    for k, v in ref.items():
        assert mt[k] == pytest.approx(v, abs=1e-6)
    ref_u = LatentRolloutTask(horizon=4, batch_size=8).evaluate(bundle, ds, held).summary("unseen_")
    for k, v in ref_u.items():
        assert mt[k] == pytest.approx(v, abs=1e-6)


def test_run_rejects_missing_combinations() -> None:
    bundle, ds = tiny_bundle(), synthetic_dataset(n_episodes=8, horizon=2)
    with pytest.raises(ValueError, match="not present"):
        CompositionalGeneralizationTask(horizon=2, holdout=[["arm_b", "fly"]]).run(bundle, ds)


def test_training_subset_drops_held_out_combinations() -> None:
    ds = synthetic_dataset(n_episodes=32, horizon=2, split="train")
    assert training_subset(ds, []) is ds
    assert training_subset(ds, None) is ds
    sub = training_subset(ds, HOLDOUT)
    assert len(sub) == 24
    assert all(
        not (item["meta"]["embodiment"] == "arm_b" and item["meta"]["primitive"] == "grasp")
        for item in (sub[i] for i in range(len(sub)))
    )


def test_trainer_runs_the_task_on_a_head_trained_without_the_held_out_pairs(tmp_path: Path) -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiments=smoke",
                "tasks=compositional_generalization",
                "data.holdout_combinations=[[arm_b,grasp]]",
                "training.epochs=2",
                "seed=3",
                f"output_dir={tmp_path}",
            ],
        )
    out = run(cfg)
    assert out["n_train"] == 48  # 64 synthetic episodes minus the 16 held-out ones
    (res,) = out["results"]
    assert res.task == "compositional_generalization"
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["seed"] == 3 and payload["n_train"] == 48 and payload["geometry"] == "poincare"
    metrics = payload["tasks"]["compositional_generalization"]["metrics"]
    assert metrics["n_seen"] == 48 and metrics["n_unseen"] == 16
    assert (tmp_path / "compositional_generalization_curves.csv").exists()


def test_trainer_builds_tasks_before_training(tmp_path: Path) -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiments=smoke",
                "tasks=compositional_generalization",
                "training.epochs=1000",  # would take long: must fail before training starts
                f"output_dir={tmp_path}",
            ],
        )
    with pytest.raises(ValueError, match="holdout_combinations"):
        run(cfg)
