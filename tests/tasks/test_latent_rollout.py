"""Latent rollout task: native-geometry error vs horizon with the static baseline."""

from __future__ import annotations

import pytest
import torch

from hyperbolic_world_model.metrics import geodesic_error, static_baseline_error
from hyperbolic_world_model.tasks import LatentRolloutTask
from hyperbolic_world_model.tasks.latent_rollout import RolloutErrors
from tests.tasks.helpers import GEOMETRIES, synthetic_dataset, tiny_bundle

METRICS = {
    "geodesic_error_h1",
    "geodesic_error_hmax",
    "geodesic_error_mean",
    "normalised_error_h1",
    "normalised_error_hmax",
    "normalised_error_mean",
    "n_items",
}


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_run_reports_curves_and_metrics_in_the_bundle_geometry(geom: tuple[str, float]) -> None:
    bundle, ds = tiny_bundle(*geom, perturb=0.3), synthetic_dataset(n_episodes=20, horizon=6)
    res = LatentRolloutTask(horizon=5, batch_size=8).run(bundle, ds)
    assert res.task == "latent_rollout" and res.geometry == geom[0] and res.curvature == geom[1]
    assert set(res.metrics) == METRICS and res.metrics["n_items"] == 20
    assert list(res.curves.columns) == [
        "horizon",
        "geodesic_error",
        "static_baseline_error",
        "normalised_error",
    ]
    assert res.curves["horizon"].tolist() == [1, 2, 3, 4, 5]
    assert res.metrics["geodesic_error_h1"] == pytest.approx(res.curves["geodesic_error"].iloc[0])
    assert res.metrics["geodesic_error_hmax"] == pytest.approx(
        res.curves["geodesic_error"].iloc[-1]
    )
    assert res.metrics["normalised_error_hmax"] == pytest.approx(
        res.curves["geodesic_error"].iloc[-1] / res.curves["static_baseline_error"].iloc[-1]
    )
    assert (res.curves["geodesic_error"] > 0).all() and (res.curves["normalised_error"] > 0).all()


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_static_head_has_normalised_error_one_in_every_geometry(geom: tuple[str, float]) -> None:
    """A zero-initialised head never moves, so its error is the static baseline exactly."""
    bundle, ds = tiny_bundle(*geom), synthetic_dataset(n_episodes=12, horizon=4)
    errors = LatentRolloutTask(horizon=4, batch_size=5).evaluate(bundle, ds)
    assert errors.error.shape == errors.static.shape == (12, 4)
    assert torch.allclose(errors.error, errors.static)
    assert torch.allclose(errors.normalised_per_horizon(), torch.ones(4))


@pytest.mark.parametrize("geom", GEOMETRIES)
def test_evaluate_matches_a_manual_rollout(geom: tuple[str, float]) -> None:
    bundle, ds = tiny_bundle(*geom, perturb=0.2), synthetic_dataset(n_episodes=6, horizon=3)
    errors = LatentRolloutTask(horizon=3, batch_size=6).evaluate(bundle, ds, indices=[4, 1])
    m, head = bundle.manifold, bundle.predictor
    frames = torch.stack([ds[4]["frames"], ds[1]["frames"]])
    actions = torch.stack([ds[4]["actions"], ds[1]["actions"]])
    with torch.no_grad():
        z = head.embed(bundle.encoder.encode(frames))
        pred = head.rollout(z[:, 0], actions)
    assert torch.allclose(errors.error, geodesic_error(m, pred, z[:, 1:]), atol=1e-6)
    assert torch.allclose(errors.static, static_baseline_error(m, z[:, 0], z[:, 1:]), atol=1e-6)
    assert errors.n == 2 and errors.horizon == 3


def test_horizon_validation() -> None:
    bundle, ds = tiny_bundle(), synthetic_dataset(n_episodes=4, horizon=3)
    with pytest.raises(ValueError, match="horizon"):
        LatentRolloutTask(horizon=0)
    with pytest.raises(ValueError, match="exceeds the dataset horizon"):
        LatentRolloutTask(horizon=4).evaluate(bundle, ds)
    with pytest.raises(ValueError, match="no items"):
        LatentRolloutTask(horizon=2).evaluate(bundle, ds, indices=[])


def test_rollout_errors_summary_prefix() -> None:
    err = RolloutErrors(error=torch.ones(3, 2), static=torch.full((3, 2), 2.0))
    s = err.summary("seen_")
    assert set(s) == {
        "seen_geodesic_error_h1",
        "seen_geodesic_error_hmax",
        "seen_geodesic_error_mean",
        "seen_normalised_error_h1",
        "seen_normalised_error_hmax",
        "seen_normalised_error_mean",
    }
    assert s["seen_normalised_error_mean"] == pytest.approx(0.5)
