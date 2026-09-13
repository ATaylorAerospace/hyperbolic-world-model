"""Train a predictor head on top of a frozen encoder, then run the configured tasks.

Hydra entry point::

    python -m hyperbolic_world_model.training.train_predictor experiments=smoke
    python -m hyperbolic_world_model.training.train_predictor experiments=poincare_sweep --multirun

Invariants enforced here (see docs/methodology.md):
    * ``encoder.assert_frozen()`` runs before the first optimiser step and again before writing
      any checkpoint; only ``predictor.state_dict()`` is ever saved.
    * the loss is ``predictor.loss`` = squared geodesic distance in the head's geometry.
    * the geometry and curvature are written next to every metric.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from hyperbolic_world_model.data import build_dataset
from hyperbolic_world_model.data.synthetic import collate
from hyperbolic_world_model.models.registry import ModelBundle, build_model
from hyperbolic_world_model.tasks import build_task
from hyperbolic_world_model.training.riemannian_optim import build_optimizer

log = logging.getLogger(__name__)
CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "configs")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(bundle: ModelBundle, dataset, cfg: DictConfig, device: str) -> list[dict[str, float]]:
    """Optimise the head only. Returns a per-epoch log."""
    bundle.encoder.assert_frozen()
    head = bundle.predictor.train()
    opt = build_optimizer(
        head.trainable_parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    history: list[dict[str, float]] = []
    for epoch in range(int(cfg.training.epochs)):
        total, n = 0.0, 0
        t0 = time.time()
        for batch in loader:
            frames = batch["frames"].to(device)
            actions = batch["actions"].to(device)
            with torch.no_grad():
                z = bundle.encoder.encode(frames)
            pred, target = head(z, actions)
            loss = head.loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.training.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    head.trainable_parameters(), float(cfg.training.grad_clip)
                )
            opt.step()
            total += loss.item() * frames.shape[0]
            n += frames.shape[0]
        rec = {"epoch": epoch, "loss": total / max(n, 1), "seconds": time.time() - t0}
        history.append(rec)
        log.info("epoch %d  loss=%.5f  (%.1fs)", epoch, rec["loss"], rec["seconds"])
    bundle.encoder.assert_frozen()
    return history


def evaluate(bundle: ModelBundle, dataset, cfg: DictConfig, device: str) -> list[dict[str, Any]]:
    """Run every task listed in ``cfg.tasks`` (a list of task configs) and return JSON-able results."""
    out = []
    for task_cfg in cfg.tasks:
        task = build_task(OmegaConf.to_container(task_cfg, resolve=True))
        res = task.run(bundle, dataset, device=device)
        log.info("task %s (%s, K=%s): %s", res.task, res.geometry, res.curvature, res.metrics)
        out.append(res)
    return out


def write_outputs(out_dir: Path, cfg: DictConfig, bundle: ModelBundle, history, results) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    payload = {
        "experiment": cfg.experiment_name,
        "geometry": bundle.manifold.name,
        "curvature": bundle.manifold.curvature,
        "latent_dim": bundle.predictor.latent_dim,
        "model": bundle.name,
        "train_history": history,
        "tasks": {r.task: r.to_json_dict() for r in results},
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    for r in results:
        if r.curves is not None:
            r.curves.to_csv(out_dir / f"{r.task}_curves.csv", index=False)
    if cfg.training.save_head:
        bundle.encoder.assert_frozen()
        ckpt_root = (
            Path(os.environ.get("CKPT_ROOT") or "checkpoints") / "predictors" / cfg.experiment_name
        )
        ckpt_root.mkdir(parents=True, exist_ok=True)
        torch.save(bundle.predictor.state_dict(), ckpt_root / "head.pt")


def run(cfg: DictConfig) -> dict[str, Any]:
    """Programmatic entry point (used by ``tests/test_smoke_experiment.py``)."""
    device = os.environ.get("HWM_DEVICE", cfg.get("device", "cpu"))
    set_seed(int(cfg.seed))
    train_ds = build_dataset(cfg.data, split="train")
    eval_ds = build_dataset(cfg.data, split="val")
    bundle = build_model(cfg, action_dim=int(train_ds.action_dim), device=device)
    history = train(bundle, train_ds, cfg, device)
    results = evaluate(bundle, eval_ds, cfg, device)
    out_dir = Path(cfg.output_dir)
    write_outputs(out_dir, cfg, bundle, history, results)
    return {"history": history, "results": results, "output_dir": str(out_dir)}


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
