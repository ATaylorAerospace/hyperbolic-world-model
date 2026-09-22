"""V-JEPA 2-AC wrapper tests with tiny stand-ins for the hub modules (no download, no timm).

The fakes reproduce the interface our wrapper relies on: an encoder with ``embed_dim``,
``patch_size``, ``tubelet_size`` mapping ``(B, 3, tubelet, H, W)`` clips to ``(B, N, D)`` tokens,
and a predictor with ``predictor_embed``, ``action_encoder``, ``state_encoder``, ``grid_height``,
``grid_width`` taking ``(tokens, actions, states)``.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from hyperbolic_world_model.models.encoders import EncoderOutput
from hyperbolic_world_model.models.encoders import vjepa2 as v
from hyperbolic_world_model.models.registry import build_encoder, build_model

D, PATCH, TUBELET, IMG = 16, 8, 2, 16
N = (IMG // PATCH) ** 2


class FakeViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim, self.patch_size, self.tubelet_size = D, PATCH, TUBELET
        self.patch_embed = nn.Conv3d(
            3, D, kernel_size=(TUBELET, PATCH, PATCH), stride=(TUBELET, PATCH, PATCH)
        )
        self.norm = nn.Linear(
            D, D
        )  # named like the real model's final norm so key-cleaning tests can drop it
        self.seen_shapes: list[tuple[int, ...]] = []

    def forward(self, x: Tensor) -> Tensor:
        self.seen_shapes.append(tuple(x.shape))
        return self.norm(self.patch_embed(x).flatten(2).transpose(1, 2))  # (B, N, D)


class FakeACPredictor(nn.Module):
    def __init__(self, action_dim: int = 7) -> None:
        super().__init__()
        p = 12
        self.predictor_embed = nn.Linear(D, p)
        self.action_encoder = nn.Linear(action_dim, p)
        self.state_encoder = nn.Linear(action_dim, p)
        self.predictor_proj = nn.Linear(p, D)
        self.grid_height = self.grid_width = IMG // PATCH

    def forward(self, x: Tensor, actions: Tensor, states: Tensor) -> Tensor:
        b, tn, _ = x.shape
        t = tn // (self.grid_height * self.grid_width)
        h = self.predictor_embed(x).view(b, t, -1, self.predictor_embed.out_features)
        h = h + (self.action_encoder(actions) + self.state_encoder(states)).unsqueeze(2)
        return self.predictor_proj(torch.tanh(h)).flatten(1, 2)


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict = {}

    def fake_load(repo: str, entry: str, **kw):
        calls.update(repo=repo, entry=entry, kwargs=kw)
        return FakeViT(), FakeACPredictor()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    return calls


def test_loader_uses_torch_hub_entry_and_freezes_everything(fake_hub: dict) -> None:
    enc = v.load_vjepa2_ac(pretrained=False)
    assert (
        fake_hub["entry"] == "vjepa2_ac_vit_giant"
        and fake_hub["repo"] == f"{v.HUB_REPO}:{v.HUB_REF}"
    )
    assert fake_hub["kwargs"]["pretrained"] is False and fake_hub["kwargs"]["trust_repo"] is True
    assert isinstance(enc, v.VJEPA2ACEncoder) and isinstance(
        enc.reference_predictor, v.MetaACPredictor
    )
    assert all(not p.requires_grad for p in enc.parameters())
    assert all(not p.requires_grad for p in enc.reference_predictor.parameters())
    assert not enc.training and not enc.reference_predictor.training
    enc.assert_frozen()
    enc.reference_predictor.assert_frozen()
    # train() can never switch either module into training mode.
    assert not enc.train().training and not enc.reference_predictor.train().training


def test_encoder_forward_returns_patch_and_pooled_embeddings(fake_hub: dict) -> None:
    enc = v.load_vjepa2_ac(pretrained=False)
    frames = torch.rand(2, 3, 3, IMG, IMG)
    out = enc(frames)
    assert isinstance(out, EncoderOutput)
    assert out.patch.shape == (2, 3, N, D) and out.pooled.shape == (2, 3, D)
    assert enc.tokens_per_frame(IMG, IMG) == N and enc.embed_dim == D
    # Each frame was encoded alone as a `tubelet_size`-frame clip, as in Meta's inference code.
    assert enc.model.seen_shapes[-1] == (2 * 3, 3, TUBELET, IMG, IMG)
    # Tokens are layer-normalised over the embedding axis, and pooled is their mean.
    assert torch.allclose(out.patch.mean(-1), torch.zeros(2, 3, N), atol=1e-5)
    assert torch.allclose(out.patch.var(-1, unbiased=False), torch.ones(2, 3, N), atol=1e-3)
    assert torch.allclose(out.pooled, out.patch.mean(2))
    assert torch.equal(enc.encode(frames), out.pooled) and torch.equal(
        enc.encode_patches(frames), out.patch
    )
    with pytest.raises(ValueError):
        enc(torch.rand(2, 3, 3, IMG + 1, IMG))
    with pytest.raises(ValueError):
        enc(torch.rand(2, 3, IMG, IMG))


def test_bf16_model_accepts_float32_frames_and_actions(fake_hub: dict) -> None:
    enc = v.load_vjepa2_ac(pretrained=False, dtype=torch.bfloat16)
    assert enc.param_dtype == torch.bfloat16 and enc.image_mean.dtype == torch.float32
    out = enc(torch.rand(1, 2, 3, IMG, IMG))
    assert out.patch.dtype == torch.bfloat16 and out.patch.shape == (1, 2, N, D)
    ref = enc.reference_predictor
    pred = ref(out.patch, torch.randn(1, 2, ref.action_dim), torch.randn(1, 2, ref.state_dim))
    assert pred.shape == (1, 2, N, D) and torch.isfinite(pred.float()).all()


def test_normalize_reps_off_keeps_raw_tokens(fake_hub: dict) -> None:
    enc = v.load_vjepa2_ac(pretrained=False, normalize_reps=False)
    frames = torch.rand(1, 2, 3, IMG, IMG)
    out = enc(frames)
    assert not torch.allclose(out.patch.var(-1, unbiased=False), torch.ones(1, 2, N), atol=1e-3)
    clip = (
        ((frames - enc.image_mean) / enc.image_std)
        .flatten(0, 1)
        .unsqueeze(2)
        .repeat(1, 1, TUBELET, 1, 1)
    )
    with torch.no_grad():
        raw = enc.model(clip).reshape(1, 2, N, D)
    assert torch.allclose(out.patch, raw)


def test_assert_frozen_catches_a_thawed_parameter(fake_hub: dict) -> None:
    enc = v.load_vjepa2_ac(pretrained=False)
    next(enc.model.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable parameters"):
        enc.assert_frozen()
    next(enc.reference_predictor.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="frozen"):
        enc.reference_predictor.assert_frozen()


def test_reference_predictor_shapes_rollout_and_loss(fake_hub: dict) -> None:
    enc = v.load_vjepa2_ac(pretrained=False)
    ref = enc.reference_predictor
    b, t = 2, 4
    tokens = enc.encode_patches(torch.rand(b, t, 3, IMG, IMG))
    actions, states = torch.randn(b, t, ref.action_dim), torch.randn(b, t, ref.state_dim)
    out = ref(tokens, actions, states)
    assert out.shape == (b, t, N, D) and ref.tokens_per_frame == N and ref.embed_dim == D
    assert ref.predict_next(tokens, actions, states).shape == (b, N, D)
    roll = ref.rollout(tokens[:, :1], actions[:, :3], states[:, :3])  # Tc = 1, H = 3
    assert roll.shape == (b, 3, N, D)
    roll2 = ref.rollout(tokens[:, :2], actions[:, :3], states[:, :3])  # Tc = 2, H = 2
    assert roll2.shape == (b, 2, N, D)
    loss = ref.loss(out[:, :-1], tokens[:, 1:])
    assert loss.shape == () and loss >= 0
    assert ref.manifold.name == "euclidean"
    with pytest.raises(ValueError):
        ref(tokens, actions[:, :-1], states)
    with pytest.raises(ValueError):
        ref.rollout(tokens[:, :2], actions[:, :1], states[:, :1])


def test_pretrained_path_cleans_keys_and_loads_strictly(
    fake_hub: dict, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    src_enc, src_pred = FakeViT(), FakeACPredictor()
    with torch.no_grad():
        for p in list(src_enc.parameters()) + list(src_pred.parameters()):
            p.normal_()
    ckpt = {
        "encoder": {"module.backbone." + k: v_ for k, v_ in src_enc.state_dict().items()},
        "predictor": {"module." + k: v_ for k, v_ in src_pred.state_dict().items()},
    }
    urls: list[str] = []

    def fake_fetch(url: str, model_dir: str, map_location: str):
        urls.append(url)
        assert model_dir == str(tmp_path)
        return ckpt

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fake_fetch)
    enc = v.load_vjepa2_ac(pretrained=True, cache_dir=tmp_path)
    assert urls == [v.CHECKPOINT_URL]
    for k, val in src_enc.state_dict().items():
        assert torch.equal(enc.model.state_dict()[k], val)
    for k, val in src_pred.state_dict().items():
        assert torch.equal(enc.reference_predictor.model.state_dict()[k], val)
    assert v.clean_state_dict_keys({"module.backbone.a.b": 1, "c": 2}) == {"a.b": 1, "c": 2}
    # A checkpoint missing an encoder key is refused rather than silently partially loaded.
    del ckpt["encoder"]["module.backbone.norm.weight"]
    with pytest.raises(RuntimeError, match="missing keys"):
        v.load_vjepa2_ac(pretrained=True, cache_dir=tmp_path)


def test_local_checkout_path_uses_hub_local_source(fake_hub: dict, tmp_path) -> None:
    enc = v.load_vjepa2_ac(hub_repo=str(tmp_path), pretrained=False)
    assert fake_hub["repo"] == str(tmp_path) and fake_hub["kwargs"]["source"] == "local"
    assert "source" not in v.__dict__  # sanity: no module-level leakage
    assert isinstance(enc, v.VJEPA2ACEncoder)
    v.load_vjepa2_ac(hub_repo="owner/name", hub_ref=None, pretrained=False)
    assert fake_hub["repo"] == "owner/name" and "source" not in fake_hub["kwargs"]


def test_missing_hub_dependencies_produce_an_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load(*a, **k):
        raise RuntimeError("Missing dependencies: timm")

    monkeypatch.setattr(torch.hub, "load", fake_load)
    with pytest.raises(RuntimeError, match="uv sync --extra vjepa2"):
        v.load_vjepa2_ac(pretrained=False)


def test_registry_builds_vjepa2_ac_bundle_with_reference_predictor(fake_hub: dict) -> None:
    model_cfg = {
        "name": "vjepa2_ac",
        "embed_dim": D,
        "encoder": {"frozen": True, "pretrained": False, "hub_ref": "abc123"},
        "head": {
            "type": "hyperbolic",
            "latent_dim": 8,
            "hidden_dim": 16,
            "n_layers": 1,
            "action_embed_dim": 8,
        },
    }
    enc = build_encoder(model_cfg)
    assert fake_hub["repo"].endswith(":abc123") and isinstance(enc, v.VJEPA2ACEncoder)
    bundle = build_model(
        {"models": model_cfg, "geometry": {"name": "poincare", "curvature": -1.0}}, action_dim=7
    )
    assert isinstance(bundle.reference_predictor, v.MetaACPredictor)
    assert bundle.reference_predictor is bundle.encoder.reference_predictor
    assert bundle.manifold.name == "poincare" and bundle.predictor.encoder_dim == D
    with pytest.raises(ValueError, match="frozen"):
        build_encoder({**model_cfg, "encoder": {"frozen": False, "pretrained": False}})
    with pytest.raises(ValueError, match="embed_dim"):
        build_encoder({**model_cfg, "embed_dim": D + 1})
