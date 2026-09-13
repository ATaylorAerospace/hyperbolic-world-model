"""Runs configs/experiments/smoke.yaml end to end on CPU and checks the harness invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from hyperbolic_world_model.geometry import Euclidean
from hyperbolic_world_model.models.predictors import EuclideanHead, HyperbolicHead
from hyperbolic_world_model.training.train_predictor import CONFIG_DIR, run

pytestmark = pytest.mark.smoke


def _compose(tmp_path: Path, *overrides: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["experiments=smoke", f"output_dir={tmp_path}", *overrides],
        )
    return cfg


def test_smoke_config_is_tiny_and_cpu() -> None:
    cfg = _compose(Path("outputs/smoke-test"))
    assert cfg.models.name == "synthetic" and cfg.data.name == "synthetic"
    assert cfg.device == "cpu" and cfg.training.epochs <= 20
    assert cfg.geometry.name == "poincare" and cfg.geometry.curvature < 0


@pytest.mark.parametrize("geometry", ["poincare", "lorentz", "euclidean"])
def test_smoke_runs_end_to_end(tmp_path: Path, geometry: str) -> None:
    overrides = [f"geometry={geometry}"]
    if geometry == "euclidean":
        overrides.append("models.head.type=euclidean")
    cfg = _compose(tmp_path / geometry, *overrides)
    out = run(cfg)
    history = out["history"]
    assert history[-1]["loss"] < history[0]["loss"], "training must reduce the geodesic loss"
    (res,) = out["results"]
    assert res.geometry == geometry
    assert res.metrics["geodesic_error_h1"] < res.metrics["geodesic_error_hmax"]
    payload = json.loads((Path(out["output_dir"]) / "metrics.json").read_text())
    assert payload["geometry"] == geometry and "curvature" in payload
    assert (Path(out["output_dir"]) / "latent_rollout_curves.csv").exists()
    saved = OmegaConf.load(Path(out["output_dir"]) / "config.yaml")
    assert saved.geometry.name == geometry


def test_encoder_stays_frozen(tmp_path: Path) -> None:
    """The frozen-encoder invariant: build_encoder refuses frozen=false and assert_frozen catches drift."""
    from hyperbolic_world_model.models.registry import build_encoder

    cfg = OmegaConf.to_container(_compose(tmp_path).models, resolve=True)
    enc = build_encoder(cfg)
    assert all(not p.requires_grad for p in enc.parameters())
    cfg["encoder"]["frozen"] = False
    with pytest.raises(ValueError, match="frozen"):
        build_encoder(cfg)


def test_hyperbolic_head_on_euclidean_manifold_matches_euclidean_head() -> None:
    """With the flat manifold, the generic head must reduce to the baseline exactly."""
    m = Euclidean()
    kw = dict(encoder_dim=16, latent_dim=8, action_dim=4, hidden_dim=32, n_layers=2, seed=3)
    a, b = EuclideanHead(manifold=m, **kw), HyperbolicHead(manifold=m, max_step=1e9, **kw)
    b.load_state_dict(a.state_dict())
    z, act = torch.randn(5, 4, 16), torch.randn(5, 3, 4)
    pa, ta = a(z, act)
    pb, tb = b(z, act)
    assert torch.allclose(pa, pb, atol=1e-6) and torch.allclose(ta, tb)
    assert torch.allclose(a.loss(pa, ta), b.loss(pb, tb))


def test_smoke_runs_all_four_tasks_on_branched_synthetic_data(tmp_path: Path) -> None:
    """``tasks=all`` needs branching rollouts and a held-out combination; both exist on CPU."""
    cfg = _compose(
        tmp_path,
        "tasks=all",
        "data.n_branches=2",
        "data.holdout_combinations=[[arm_b,grasp]]",
        "training.epochs=3",
    )
    out = run(cfg)
    assert out["n_train"] == 48
    names = [r.task for r in out["results"]]
    assert names == [
        "latent_rollout",
        "hierarchy_reconstruction",
        "long_horizon_consistency",
        "compositional_generalization",
    ]
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["seed"] == 0 and set(payload["tasks"]) == set(names)
    assert (
        payload["tasks"]["long_horizon_consistency"]["metrics"]["n_pairs"] == 32
    )  # 64 / 2 prompts
    for name in names:
        assert (tmp_path / f"{name}_curves.csv").exists()
