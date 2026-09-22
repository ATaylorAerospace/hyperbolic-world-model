"""Frozen V-JEPA 2-AC encoder and Meta's reference action-conditioned predictor, via torch.hub.

The action-conditioned release (``vjepa2_ac_vit_giant``, https://github.com/facebookresearch/vjepa2)
bundles a ViT-g/16 encoder post-trained on DROID together with the action-conditioned predictor
described in the V-JEPA 2 paper. We use **that** encoder, not the general V-JEPA 2 Hugging Face
release, so every head in this repository sees exactly the features Meta's predictor was trained on
and the comparison is controlled. Meta's predictor is kept, frozen, as the reference Euclidean
baseline (:class:`MetaACPredictor`).

How the wrapper follows Meta's own inference code (``notebooks/utils/world_model_wrapper.py`` and
``app/vjepa_droid/train.py`` in the upstream repository):

* each frame is encoded on its own by duplicating it into a ``tubelet_size``-frame clip,
  ``(B*T, 3, 2, H, W)``, giving ``N = (H / 16) * (W / 16)`` tokens per frame (256 at 256 px);
* token representations are layer-normalised over the embedding axis (``normalize_reps``);
* the predictor takes ``(tokens (B, T*N, D), actions (B, T, 7), states (B, T, 7))`` and, being
  frame-causal, its output at frame ``t`` is the prediction for frame ``t + 1``; Meta trains it
  with an L1 loss on the layer-normalised target tokens.

Loading. The model *code* comes from torch.hub (``torch.hub.load("facebookresearch/vjepa2",
"vjepa2_ac_vit_giant", pretrained=False)``); the *weights* are fetched by us from Meta's official
checkpoint URL with ``torch.hub.load_state_dict_from_url`` and cached under ``CKPT_ROOT``. We do
not pass ``pretrained=True`` to hub because upstream resolves the checkpoint URL from a module
constant that has, on ``main``, pointed at a local test server; fetching the official file
directly is deterministic and lets us pin ``hub_ref`` to an inspected commit. Hub's own dependency
check requires ``timm`` and ``einops`` to be importable on the loading machine; ``timm`` is the
``vjepa2`` optional extra of this repository (``uv sync --extra vjepa2``).

There is deliberately no training path in this module: :func:`load_vjepa2_ac` freezes every
parameter of both modules and :meth:`VJEPA2ACEncoder.assert_frozen` is called by the trainer
before the first optimiser step. Nothing is downloaded at import time.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hyperbolic_world_model.geometry.euclidean import Euclidean
from hyperbolic_world_model.models.encoders import EncoderOutput, FrozenEncoder

HUB_REPO = "facebookresearch/vjepa2"
#: Commit of the upstream repository whose ``hubconf`` and model code this wrapper was written against.
HUB_REF = "204698b45b3712590f06245fbfba32d3be539812"
HUB_ENTRY = "vjepa2_ac_vit_giant"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
#: ViT-g/16 hidden size; 7-DoF end-effector deltas and 7-D proprioceptive state, as in DROID.
EMBED_DIM = 1408
ACTION_DIM = 7
STATE_DIM = 7


def clean_state_dict_keys(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Strip the ``module.`` / ``backbone.`` prefixes Meta's checkpoints carry (as upstream does)."""
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}


class MetaACPredictor(nn.Module):
    """Meta's shipped action-conditioned predictor, frozen: the reference Euclidean baseline.

    It operates on patch tokens, not on our pooled latents, and it conditions on both actions and
    proprioceptive states, exactly as trained. Its geometry is Euclidean and its loss is Meta's
    L1 on layer-normalised tokens, which is the native-geometry rule applied to Meta's model.

    Args:
        predictor: the ``VisionTransformerPredictorAC`` returned by torch.hub.
        normalize_reps: layer-normalise predictions as Meta's rollout code does.
    """

    def __init__(self, predictor: nn.Module, normalize_reps: bool = True) -> None:
        super().__init__()
        self.model = predictor
        self.normalize_reps = bool(normalize_reps)
        self.manifold = Euclidean()
        self.action_dim = int(predictor.action_encoder.in_features)
        self.state_dim = int(predictor.state_encoder.in_features)
        self.embed_dim = int(predictor.predictor_embed.in_features)
        self.tokens_per_frame = int(predictor.grid_height * predictor.grid_width)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> MetaACPredictor:  # type: ignore[override]
        return super().train(False)

    def assert_frozen(self) -> None:
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        if trainable or self.training:
            raise RuntimeError("MetaACPredictor must stay frozen; it is the reference baseline")

    def _check(self, tokens: Tensor, actions: Tensor, states: Tensor) -> None:
        if (
            tokens.ndim != 4
            or tokens.shape[2] != self.tokens_per_frame
            or tokens.shape[3] != self.embed_dim
        ):
            raise ValueError(
                f"tokens must be (B, T, {self.tokens_per_frame}, {self.embed_dim}), got {tuple(tokens.shape)}"
            )
        b, t = tokens.shape[:2]
        if actions.shape != (b, t, self.action_dim) or states.shape != (b, t, self.state_dim):
            raise ValueError(
                f"actions must be {(b, t, self.action_dim)} and states {(b, t, self.state_dim)}, "
                f"got {tuple(actions.shape)} / {tuple(states.shape)}"
            )

    def forward(self, tokens: Tensor, actions: Tensor, states: Tensor) -> Tensor:
        """Frame-causal prediction: output ``[:, t]`` is the predicted token set of frame ``t + 1``.

        Args:
            tokens: ``(B, T, N, D)`` layer-normalised encoder tokens of frames ``0..T-1``.
            actions: ``(B, T, 7)`` action taken after each frame.
            states: ``(B, T, 7)`` proprioceptive state at each frame.
        """
        self._check(tokens, actions, states)
        b, t, n, d = tokens.shape
        dt = next(self.model.parameters()).dtype
        out = self.model(tokens.flatten(1, 2).to(dt), actions.to(dt), states.to(dt))
        if self.normalize_reps:
            out = F.layer_norm(out, (out.shape[-1],))
        return out.view(b, t, n, d)

    def predict_next(self, tokens: Tensor, actions: Tensor, states: Tensor) -> Tensor:
        """Tokens of the frame after the last context frame, ``(B, N, D)``."""
        return self.forward(tokens, actions, states)[:, -1]

    def rollout(self, context: Tensor, actions: Tensor, states: Tensor) -> Tensor:
        """Autoregressive rollout as in Meta's training code.

        Args:
            context: ``(B, Tc, N, D)`` context frames (``Tc >= 1``).
            actions: ``(B, Tc + H - 1, 7)`` actions after each context and predicted frame.
            states: ``(B, Tc + H - 1, 7)`` states aligned with ``actions``.

        Returns:
            ``(B, H, N, D)`` predicted tokens for the ``H`` frames after the context.
        """
        if context.ndim != 4:
            raise ValueError("context must be (B, Tc, N, D)")
        tc = context.shape[1]
        horizon = actions.shape[1] - tc + 1
        if horizon < 1 or states.shape[1] != actions.shape[1]:
            raise ValueError("actions and states must have Tc + H - 1 steps with H >= 1")
        z = context
        preds = []
        for h in range(horizon):
            t = tc + h
            nxt = self.predict_next(z, actions[:, :t], states[:, :t])
            preds.append(nxt)
            z = torch.cat([z, nxt.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)

    def loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Meta's objective: mean absolute error on layer-normalised tokens (``loss_exp = 1``)."""
        if self.normalize_reps:
            target = F.layer_norm(target, (target.shape[-1],))
        return (pred - target).abs().mean()


class VJEPA2ACEncoder(FrozenEncoder):
    """Frozen V-JEPA 2-AC ViT-g encoder producing per-frame patch tokens and pooled embeddings.

    Args:
        encoder: the ``VisionTransformer`` returned by torch.hub for ``vjepa2_ac_vit_giant``.
        reference_predictor: Meta's predictor from the same checkpoint (kept frozen alongside).
        normalize_reps: layer-normalise tokens over the embedding axis, as Meta does before the
            predictor sees them.
        image_mean / image_std: normalisation applied to ``[0, 1]`` frames (ImageNet statistics,
            as in Meta's evaluation transform).
    """

    def __init__(
        self,
        encoder: nn.Module,
        reference_predictor: MetaACPredictor | None = None,
        normalize_reps: bool = True,
        image_mean: tuple[float, float, float] = IMAGENET_MEAN,
        image_std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.model = encoder
        self.reference_predictor = reference_predictor
        self.normalize_reps = bool(normalize_reps)
        self.embed_dim = int(encoder.embed_dim)
        self.patch_size = int(encoder.patch_size)
        self.tubelet_size = int(encoder.tubelet_size)
        self.register_buffer("image_mean", torch.tensor(image_mean).view(1, 1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).view(1, 1, 3, 1, 1))
        self.freeze()

    def tokens_per_frame(self, height: int, width: int) -> int:
        """Number of patch tokens the encoder produces for one ``height x width`` frame."""
        return (height // self.patch_size) * (width // self.patch_size)

    def forward(self, frames: Tensor) -> EncoderOutput:
        """Encode ``(B, T, 3, H, W)`` frames in ``[0, 1]``; ``H`` and ``W`` must be multiples of 16.

        Returns patch tokens ``(B, T, N, D)`` and their mean ``(B, T, D)``.
        """
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError(f"expected (B, T, 3, H, W), got {tuple(frames.shape)}")
        b, t, _, h, w = frames.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(f"H and W must be multiples of {self.patch_size}, got {(h, w)}")
        x = (frames.to(self.image_mean.dtype) - self.image_mean) / self.image_std
        # Meta encodes each frame alone as a `tubelet_size`-frame clip of the same frame.
        clip = (
            x.flatten(0, 1).unsqueeze(2).repeat(1, 1, self.tubelet_size, 1, 1)
        )  # (B*T, 3, tubelet, H, W)
        # The wrapped encoder may run in bfloat16 on GPU while the normalisation buffers stay
        # float32; feed it its own parameter dtype and compute the tokens back in float32.
        param = next(self.model.parameters(), None)
        if param is not None and param.dtype != clip.dtype:
            clip = clip.to(param.dtype)
        tokens = self.model(clip).to(x.dtype)  # (B*T, N, D)
        tokens = tokens.reshape(b, t, -1, self.embed_dim)
        if self.normalize_reps:
            tokens = F.layer_norm(tokens, (self.embed_dim,))
        return EncoderOutput(patch=tokens, pooled=tokens.mean(dim=2))


def _hub_load(hub_repo: str, hub_ref: str | None, **kwargs: Any) -> tuple[nn.Module, nn.Module]:
    """``torch.hub.load`` of the AC entry point; a local checkout path is accepted in place of ``owner/name``."""
    if Path(
        hub_repo
    ).is_dir():  # air-gapped machines: a clone of facebookresearch/vjepa2 at hub_ref
        repo, extra = str(hub_repo), {"source": "local"}
    else:
        repo, extra = (f"{hub_repo}:{hub_ref}" if hub_ref else hub_repo), {}
    try:
        return torch.hub.load(repo, HUB_ENTRY, trust_repo=True, **extra, **kwargs)
    except RuntimeError as e:
        if "Missing dependencies" in str(e):
            raise RuntimeError(
                "torch.hub needs the upstream vjepa2 dependencies importable on this machine "
                "(`timm`, `einops`). Install the `vjepa2` extra before loading the V-JEPA 2-AC "
                "checkpoint: uv sync --extra vjepa2 (or: uv pip install timm)"
            ) from e
        raise


def load_vjepa2_ac(
    hub_repo: str = HUB_REPO,
    hub_ref: str | None = HUB_REF,
    pretrained: bool = True,
    checkpoint_url: str = CHECKPOINT_URL,
    cache_dir: str | os.PathLike[str] | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    normalize_reps: bool = True,
) -> VJEPA2ACEncoder:
    """Load the V-JEPA 2-AC checkpoint (frozen encoder + frozen reference predictor) via torch.hub.

    Args:
        hub_repo: GitHub ``owner/name`` served by torch.hub, or the path of a local clone of it
            (loaded with ``source="local"``; ``hub_ref`` is then the caller's responsibility).
        hub_ref: branch, tag or commit of ``hub_repo``; defaults to the inspected commit.
        pretrained: download Meta's weights from ``checkpoint_url`` (cached under ``cache_dir``).
            ``False`` builds the randomly initialised architecture (tests only).
        checkpoint_url: the official ``vjepa2-ac-vitg.pt`` URL.
        cache_dir: overrides ``CKPT_ROOT/encoders/vjepa2_ac``.
        device / dtype: placement for both modules; bfloat16 is fine on GPU for inference.
        normalize_reps: see :class:`VJEPA2ACEncoder`.

    Returns:
        A :class:`VJEPA2ACEncoder` whose ``reference_predictor`` is the frozen
        :class:`MetaACPredictor`. ``assert_frozen`` passes on the returned module.
    """
    encoder, predictor = _hub_load(hub_repo, hub_ref, pretrained=False)
    if pretrained:
        cache = (
            Path(cache_dir)
            if cache_dir
            else Path(os.environ.get("CKPT_ROOT") or "checkpoints") / "encoders" / "vjepa2_ac"
        )
        cache.mkdir(parents=True, exist_ok=True)
        state = torch.hub.load_state_dict_from_url(
            checkpoint_url, model_dir=str(cache), map_location="cpu"
        )
        enc_res = encoder.load_state_dict(clean_state_dict_keys(state["encoder"]), strict=False)
        if enc_res.missing_keys:
            raise RuntimeError(
                f"V-JEPA 2-AC encoder checkpoint is missing keys: {enc_res.missing_keys[:5]}"
            )
        predictor.load_state_dict(clean_state_dict_keys(state["predictor"]), strict=True)
    encoder.to(device=device, dtype=dtype)
    predictor.to(device=device, dtype=dtype)
    wrapped = VJEPA2ACEncoder(
        encoder,
        reference_predictor=MetaACPredictor(predictor, normalize_reps=normalize_reps),
        normalize_reps=normalize_reps,
    ).to(device)
    wrapped.assert_frozen()
    wrapped.reference_predictor.assert_frozen()
    return wrapped


__all__ = [
    "ACTION_DIM",
    "CHECKPOINT_URL",
    "EMBED_DIM",
    "HUB_ENTRY",
    "HUB_REF",
    "HUB_REPO",
    "STATE_DIM",
    "MetaACPredictor",
    "VJEPA2ACEncoder",
    "clean_state_dict_keys",
    "load_vjepa2_ac",
]
