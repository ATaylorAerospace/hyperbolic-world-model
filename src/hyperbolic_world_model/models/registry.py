"""Build encoders, predictor heads and manifolds from Hydra configs.

Config shape (see ``configs/models/*.yaml``)::

    models:
      name: vjepa2_ac | dino_wm | synthetic
      encoder: {model_id: ..., revision: ..., frozen: true, ...}
      embed_dim: 1024
      head: {type: euclidean | hyperbolic, latent_dim: 64, hidden_dim: 256, n_layers: 2, embed_scale: 1.0}
    geometry:
      name: euclidean | poincare | lorentz
      curvature: -1.0
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hyperbolic_world_model.geometry import Manifold, build_manifold
from hyperbolic_world_model.geometry.euclidean import Euclidean
from hyperbolic_world_model.models.encoders import FrozenEncoder
from hyperbolic_world_model.models.predictors import (
    ActionConditionedPredictor,
    EuclideanHead,
    HyperbolicHead,
)


@dataclass
class ModelBundle:
    """Everything a task needs: the frozen encoder, the trainable head and their shared geometry."""

    encoder: FrozenEncoder
    predictor: ActionConditionedPredictor
    manifold: Manifold
    name: str


def _as_dict(cfg: Mapping[str, Any] | Any) -> dict[str, Any]:
    if hasattr(cfg, "items"):
        return {k: (_as_dict(v) if hasattr(v, "items") else v) for k, v in cfg.items()}
    return dict(cfg)


def build_encoder(model_cfg: Mapping[str, Any], device: str = "cpu") -> FrozenEncoder:
    """Instantiate the frozen encoder named by ``model_cfg['name']``."""
    cfg = _as_dict(model_cfg)
    name = cfg["name"]
    enc = cfg.get("encoder", {})
    if not enc.get("frozen", True):
        raise ValueError(
            f"models={name}: encoder.frozen must be true; this harness never trains encoders"
        )

    if name == "synthetic":
        from hyperbolic_world_model.models.encoders.synthetic import SyntheticEncoder

        encoder = SyntheticEncoder(
            frame_shape=tuple(enc["frame_shape"]),
            embed_dim=int(cfg["embed_dim"]),
            seed=int(enc.get("seed", 0)),
        )
    elif name == "vjepa2_ac":
        from hyperbolic_world_model.models.encoders.vjepa2 import load_vjepa2_encoder

        encoder = load_vjepa2_encoder(
            model_id=enc["model_id"],
            revision=enc.get("revision"),
            cache_dir=enc.get("cache_dir"),
            device=device,
        )
    elif name == "dino_wm":
        from hyperbolic_world_model.models.encoders.dino import load_dinov2_encoder

        encoder = load_dinov2_encoder(
            model_id=enc["model_id"],
            revision=enc.get("revision"),
            cache_dir=enc.get("cache_dir"),
            device=device,
            pooling=enc.get("pooling", "cls"),
        )
    else:
        raise KeyError(f"unknown model {name!r}; expected one of synthetic, vjepa2_ac, dino_wm")

    encoder.to(device)
    encoder.assert_frozen()
    if encoder.embed_dim != int(cfg["embed_dim"]):
        raise ValueError(
            f"config embed_dim={cfg['embed_dim']} but encoder reports {encoder.embed_dim}"
        )
    return encoder


def build_predictor(
    head_cfg: Mapping[str, Any], manifold: Manifold, encoder_dim: int, action_dim: int
) -> ActionConditionedPredictor:
    """Instantiate a predictor head on ``manifold``."""
    cfg = _as_dict(head_cfg)
    head_type = cfg.get("type", "hyperbolic")
    common = dict(
        encoder_dim=encoder_dim,
        latent_dim=int(cfg["latent_dim"]),
        action_dim=action_dim,
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        n_layers=int(cfg.get("n_layers", 2)),
        embed_scale=float(cfg.get("embed_scale", 1.0)),
        seed=int(cfg.get("seed", 0)),
    )
    if head_type == "euclidean":
        if not isinstance(manifold, Euclidean):
            raise ValueError("head.type=euclidean requires geometry=euclidean")
        return EuclideanHead(manifold=manifold, **common)
    if head_type == "hyperbolic":
        return HyperbolicHead(manifold=manifold, max_step=float(cfg.get("max_step", 5.0)), **common)
    raise KeyError(f"unknown head type {head_type!r}; expected euclidean or hyperbolic")


def build_model(cfg: Mapping[str, Any], action_dim: int, device: str = "cpu") -> ModelBundle:
    """Build the full bundle from a resolved experiment config with ``models`` and ``geometry`` groups."""
    cfg = _as_dict(cfg)
    manifold = build_manifold(cfg["geometry"])
    encoder = build_encoder(cfg["models"], device=device)
    predictor = build_predictor(cfg["models"]["head"], manifold, encoder.embed_dim, action_dim).to(
        device
    )
    return ModelBundle(
        encoder=encoder, predictor=predictor, manifold=manifold, name=cfg["models"]["name"]
    )


__all__ = ["ModelBundle", "build_encoder", "build_model", "build_predictor"]
