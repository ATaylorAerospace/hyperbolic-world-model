"""Cosmos generation pipeline, latent extraction and dataset, exercised end to end with fakes.

No model is loaded. A deterministic fake world model stands in for the NVIDIA adapter's client so
the adapter's chunking, padding, seeding and stitching logic (which mirrors upstream) is verified
against the ``generate_vid2world`` contract it was written for.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hyperbolic_world_model.data import build_dataset
from hyperbolic_world_model.data.cosmos3 import default_root, generate as g
from hyperbolic_world_model.data.cosmos3.dataset import Cosmos3TrajectoryDataset
from hyperbolic_world_model.data.cosmos3.extract_latents import (
    CosmosTokenizerBackend,
    extract_latents,
    load_latents,
    pool_latent,
)
from hyperbolic_world_model.data.cosmos3.extract_latents import main as extract_main
from hyperbolic_world_model.data.hierarchies import build_hierarchy

H = W = 16
A = 7


class FakeGenerator:
    """Deterministic stand-in: each action shifts the frame's mean brightness by its first component."""

    model_info = {"backend": "fake", "model": "fake-nano"}

    def generate(self, start_frame: np.ndarray, actions: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        frames = [start_frame]
        cur = start_frame.astype(np.float32)
        for a in actions:
            cur = np.clip(cur + 10 * a[0] + rng.normal(0, 0.5, cur.shape), 0, 255)
            frames.append(cur.astype(np.uint8))
        return np.stack(frames)


class FakeTokenizer:
    model_info = {"backend": "fake-tokenizer"}

    def encode(self, video: np.ndarray) -> np.ndarray:  # (T, H, W, 3) -> (C, T', H', W')
        t = 1 + (video.shape[0] - 1) // 4
        c, h, w = 6, video.shape[1] // 8, video.shape[2] // 8
        lat = np.zeros((c, t, h, w), dtype=np.float32)
        for k in range(t):
            frame = video[min(k * 4, video.shape[0] - 1)].astype(np.float32) / 255
            lat[:3, k] = frame[::8, ::8].transpose(2, 0, 1)
            lat[3:, k] = frame[::8, ::8].transpose(2, 0, 1) ** 2
        return lat


@pytest.fixture
def prompt_dir(tmp_path: Path) -> tuple[Path, Path]:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    from PIL import Image

    rng = np.random.default_rng(0)
    for pid in ("ep0", "ep1", "ep2"):
        img = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        Image.fromarray(img).save(frames_dir / f"{pid}.png")
    np.save(frames_dir / "ep3.npy", rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8))
    spec = {
        "action_dim": A,
        "prompts": [
            {
                "prompt_id": pid,
                "start_frame": f"{pid}.{'npy' if pid == 'ep3' else 'png'}",
                "embodiment": emb,
                "task": task,
                "branches": [
                    {"branch_id": "b0", "primitive": "reach", "actions": rng.normal(size=(6, A)).tolist(), "note": "x"},
                    {"branch_id": "b1", "primitive": "grasp", "actions": rng.normal(size=(9, A)).tolist()},
                ],
            }
            for pid, emb, task in (("ep0", "franka", "pick"), ("ep1", "franka", "place"), ("ep2", "ur5", "pick"), ("ep3", "ur5", "place"))
        ],
    }
    spec_path = tmp_path / "actions.json"
    spec_path.write_text(json.dumps(spec))
    return frames_dir, spec_path


# ---------------------------------------------------------------------------- inputs
def test_load_action_spec_validates(prompt_dir: tuple[Path, Path], tmp_path: Path) -> None:
    frames_dir, spec_path = prompt_dir
    prompts = g.load_action_spec(spec_path, frames_dir)
    assert [p.prompt_id for p in prompts] == ["ep0", "ep1", "ep2", "ep3"]
    assert prompts[0].branches[0].actions.shape == (6, A) and prompts[0].branches[0].extra == {"note": "x"}
    assert g.load_start_frame(prompts[0].start_frame_path).shape == (H, W, 3)
    assert g.load_start_frame(prompts[3].start_frame_path).dtype == np.uint8
    spec = json.loads(spec_path.read_text())
    bad = tmp_path / "bad.json"
    spec["prompts"][0]["branches"][0]["actions"] = [[0.0] * (A - 1)]
    bad.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="actions must be"):
        g.load_action_spec(bad, frames_dir)
    spec = json.loads(spec_path.read_text())
    spec["prompts"][0]["start_frame"] = "missing.png"
    bad.write_text(json.dumps(spec))
    with pytest.raises(FileNotFoundError):
        g.load_action_spec(bad, frames_dir)
    spec = json.loads(spec_path.read_text())
    spec["prompts"].append(dict(spec["prompts"][0]))
    bad.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        g.load_action_spec(bad, frames_dir)
    with pytest.raises(ValueError, match="unsupported"):
        g.load_start_frame(tmp_path / "x.gif")


def test_branch_seed_is_deterministic_and_distinct() -> None:
    assert g.branch_seed(0, "ep0", "b0") == g.branch_seed(0, "ep0", "b0")
    assert g.branch_seed(0, "ep0", "b0") != g.branch_seed(0, "ep0", "b1")
    assert g.branch_seed(0, "ep0", "b0") != g.branch_seed(1, "ep0", "b0")
    assert 0 <= g.branch_seed(123, "p", "b") < 2**31 - 1


# ---------------------------------------------------------------------------- generation driver
def test_generate_rollouts_writes_frames_meta_and_manifest(prompt_dir: tuple[Path, Path], tmp_path: Path) -> None:
    frames_dir, spec_path = prompt_dir
    out = tmp_path / "gen"
    prompts = g.load_action_spec(spec_path, frames_dir)
    records = g.generate_rollouts(prompts, FakeGenerator(), out, seed=0, write_mp4=False)
    assert len(records) == 8
    manifest = g.read_manifest(out)
    assert [(r["prompt_id"], r["branch_id"]) for r in manifest] == [(p, b) for p in ("ep0", "ep1", "ep2", "ep3") for b in ("b0", "b1")]
    r = manifest[1]
    assert r["num_frames"] == 10 and r["action_dim"] == A and r["primitive"] == "grasp" and r["latents_path"] is None
    assert r["seed"] == g.branch_seed(0, "ep0", "b1") and r["model_info"]["model"] == "fake-nano"
    with np.load(out / r["frames_path"]) as z:
        assert z["frames"].shape == (10, H, W, 3) and z["frames"].dtype == np.uint8 and z["actions"].shape == (9, A)
        assert np.array_equal(z["frames"][0], g.load_start_frame(prompts[0].start_frame_path))  # shared start frame
    meta = json.loads((out / "rollouts" / "ep0" / "b1" / "meta.json").read_text())
    assert meta["embodiment"] == "franka" and meta["task"] == "pick" and "generated_at" in meta
    assert json.loads((out / "rollouts" / "ep0" / "b0" / "meta.json").read_text())["note"] == "x"
    # Branches of one prompt share the start frame but diverge.
    with np.load(out / manifest[0]["frames_path"]) as z0, np.load(out / manifest[1]["frames_path"]) as z1:
        assert np.array_equal(z0["frames"][0], z1["frames"][0]) and not np.array_equal(z0["frames"][1], z1["frames"][1])
    # Idempotent: a second run skips everything already in the manifest; --overwrite regenerates identically.
    assert g.generate_rollouts(prompts, FakeGenerator(), out, seed=0, write_mp4=False) == []
    again = g.generate_rollouts(prompts, FakeGenerator(), out, seed=0, write_mp4=False, skip_existing=False)
    assert len(again) == 8
    with np.load(out / again[3]["frames_path"]) as z:
        with np.load(out / manifest[3]["frames_path"]) as z_old:
            assert np.array_equal(z["frames"], z_old["frames"])  # same seed -> same rollout


def test_write_rollout_rejects_bad_frames(prompt_dir: tuple[Path, Path], tmp_path: Path) -> None:
    frames_dir, spec_path = prompt_dir
    p = g.load_action_spec(spec_path, frames_dir)[0]
    with pytest.raises(ValueError, match="frames must be"):
        g.write_rollout(tmp_path, p, p.branches[0], np.zeros((3, H, W, 3), np.uint8), 0, {}, write_mp4=False)


def test_cli_dry_run_and_injected_generator(prompt_dir: tuple[Path, Path], tmp_path: Path, capsys) -> None:  # noqa: ANN001
    frames_dir, spec_path = prompt_dir
    out = tmp_path / "gen"
    assert g.main(["--start-frames", str(frames_dir), "--actions", str(spec_path), "--out", str(out), "--dry-run"]) == []
    plan = json.loads((out / "generation_plan.json").read_text())
    assert len(plan["rollouts"]) == 8 and plan["model"] == g.DEFAULT_MODEL and not (out / "manifest.jsonl").exists()
    records = g.main(["--start-frames", str(frames_dir), "--actions", str(spec_path), "--out", str(out), "--no-mp4", "--seed", "3"], generator=FakeGenerator())
    assert len(records) == 8 and records[0]["seed"] == g.branch_seed(3, "ep0", "b0")
    assert "generated 8 rollouts" in capsys.readouterr().out


def test_default_root_follows_data_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert default_root() == tmp_path / "cosmos3_generated"
    assert default_root("/x") == Path("/x/cosmos3_generated")


# ---------------------------------------------------------------------------- NVIDIA adapter logic
class FakeVideo2World:
    """Mimics ``Video2WorldInference.generate_vid2world``: records calls, returns a video in [-1, 1]."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate_vid2world(self, **kw):  # noqa: ANN003, ANN202
        self.calls.append(kw)
        vid = kw["input_path"]  # (B, C, T, H, W) uint8
        b, c, t, h, w = vid.shape
        first = vid[:, :, :1].float() / 127.5 - 1  # (B, C, 1, H, W)
        step = kw["action"][:, 0].view(1, 1, -1, 1, 1) * 0.1  # (1, 1, T-1, 1, 1)
        frames = first + torch.cumsum(step, dim=2)
        return torch.cat([first, frames], dim=2).clamp(-1, 1)


def test_cosmos_generator_reproduces_upstream_chunk_loop() -> None:
    client = FakeVideo2World()
    gen = g.CosmosGenerator(chunk_size=4, guidance=5.0, num_steps=3, client=client)
    start = np.full((H, W, 3), 100, dtype=np.uint8)
    actions = np.ones((10, A), dtype=np.float32)  # 10 actions -> chunks of 4, 4, 2 (padded to 4)
    video = gen.generate(start, actions, seed=7)
    assert video.shape == (11, H, W, 3) and video.dtype == np.uint8
    assert np.array_equal(video[0], start)
    assert len(client.calls) == 3
    for i, call in enumerate(client.calls):
        assert call["input_path"].shape == (1, 3, 5, H, W) and call["input_path"].dtype == torch.uint8
        assert torch.all(call["input_path"][:, :, 1:] == 0)  # only the first frame is real
        assert call["action"].shape == (4, A) and call["num_video_frames"] == 5
        assert call["seed"] == 7 + 4 * i and call["guidance"] == 5.0 and call["num_steps"] == 3
        assert call["num_latent_conditional_frames"] == 1 and call["prompt"] == ""
    assert torch.all(client.calls[2]["action"][2:] == 0)  # zero padding of the last chunk
    # Each chunk starts from the previous chunk's last generated frame.
    assert np.array_equal(client.calls[1]["input_path"][0, :, 0].permute(1, 2, 0).numpy(), video[4])
    # Brightness increases monotonically with the constant positive action: no duplicated boundary frame
    # (which upstream's preview stitching would produce) and no dropped frame.
    means = video.astype(np.float32).mean(axis=(1, 2, 3))
    assert np.all(np.diff(means) > 0)
    assert gen.model_info["chunk_size"] == 4 and gen.model_info["model"] == g.DEFAULT_MODEL
    with pytest.raises(ValueError):
        gen.generate(start.astype(np.float32), actions, 0)
    with pytest.raises(ValueError):
        g.CosmosGenerator(chunk_size=0)


def test_cosmos_generator_without_package_gives_actionable_error() -> None:
    gen = g.CosmosGenerator()
    with pytest.raises(RuntimeError, match="cosmos_predict2 package is not installed"):
        _ = gen.client


# ---------------------------------------------------------------------------- latents
def test_pool_latent_and_extract(prompt_dir: tuple[Path, Path], tmp_path: Path) -> None:
    lat = np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2)
    mean = pool_latent(lat, "mean")
    assert mean.shape == (3, 2) and np.allclose(mean[0], lat[:, 0].mean(axis=(1, 2)))
    flat = pool_latent(lat, "flatten")
    assert flat.shape == (3, 8) and np.allclose(flat[1], lat[:, 1].reshape(-1))
    with pytest.raises(ValueError):
        pool_latent(lat, "max")
    with pytest.raises(ValueError):
        pool_latent(lat[0], "mean")

    frames_dir, spec_path = prompt_dir
    out = tmp_path / "gen"
    g.generate_rollouts(g.load_action_spec(spec_path, frames_dir), FakeGenerator(), out, write_mp4=False)
    written = extract_latents(out, FakeTokenizer(), pooling="mean")
    assert len(written) == 8 and all(p.exists() for p in written)
    manifest = g.read_manifest(out)
    assert all(r["latents_path"] for r in manifest)
    lat, meta = load_latents(out / manifest[1]["latents_path"])
    assert lat.shape == (1 + 9 // 4, 6) and lat.dtype == np.float32 and meta["pooling"] == "mean"
    assert meta["primitive"] == "grasp" and meta["model_info"]["backend"] == "fake-tokenizer"
    assert extract_latents(out, FakeTokenizer()) == []  # skip existing
    assert len(extract_latents(out, FakeTokenizer(), pooling="flatten", skip_existing=False)) == 8
    lat2, meta2 = load_latents(out / manifest[1]["latents_path"])
    assert lat2.shape == (3, 6 * (H // 8) * (W // 8)) and meta2["pooling"] == "flatten"
    with pytest.raises(FileNotFoundError):
        extract_latents(tmp_path / "empty", FakeTokenizer())
    assert len(extract_main(["--root", str(out), "--overwrite"], backend=FakeTokenizer())) == 8


def test_tokenizer_backend_encodes_through_client_tokenizer() -> None:
    class Tok:
        def encode(self, x):  # noqa: ANN001, ANN202
            assert x.shape[:2] == (1, 3) and x.dtype == torch.bfloat16
            assert float(x.float().min()) >= -1 and float(x.float().max()) <= 1
            return x[:, :, ::4, ::8, ::8].float()

    class Client:
        class model:  # noqa: N801
            tokenizer = Tok()

    be = CosmosTokenizerBackend(client=Client())
    video = np.random.default_rng(0).integers(0, 255, size=(9, H, W, 3), dtype=np.uint8)
    lat = be.encode(video)
    assert lat.shape == (3, 3, H // 8, W // 8) and lat.dtype == np.float32
    with pytest.raises(ValueError):
        be.encode(video.astype(np.float32))


# ---------------------------------------------------------------------------- dataset
@pytest.fixture
def generated(prompt_dir: tuple[Path, Path], tmp_path: Path) -> Path:
    frames_dir, spec_path = prompt_dir
    out = tmp_path / "gen"
    g.generate_rollouts(g.load_action_spec(spec_path, frames_dir), FakeGenerator(), out, write_mp4=False)
    extract_latents(out, FakeTokenizer())
    return out


def test_dataset_windows_shapes_and_meta(generated: Path) -> None:
    ds = Cosmos3TrajectoryDataset(generated, horizon=4, stride=2, image_size=8)
    # ep*/b0 has 7 frames -> starts 0, 2 ; ep*/b1 has 10 frames -> starts 0, 2, 4 ; x4 prompts
    assert len(ds) == 4 * (2 + 3) and ds.action_dim == A
    item = ds[0]
    assert item["frames"].shape == (5, 3, 8, 8) and item["actions"].shape == (4, A)
    assert item["frames"].dtype == torch.float32 and 0 <= item["frames"].min() and item["frames"].max() <= 1
    assert item["meta"]["prompt_id"] == "ep0" and item["meta"]["branch_id"] == "b0" and item["meta"]["start"] == 0
    assert item["meta"]["latents_path"].endswith("ep0/b0.npz")
    whole = Cosmos3TrajectoryDataset(generated, horizon=None)
    assert len(whole) == 8 and whole[1]["frames"].shape == (10, 3, H, W)
    too_long = Cosmos3TrajectoryDataset(generated, horizon=8)
    assert len(too_long) == 4 * 2  # only the 10-frame branches fit
    with pytest.raises(FileNotFoundError):
        Cosmos3TrajectoryDataset(generated / "nope")
    with pytest.raises(ValueError):
        Cosmos3TrajectoryDataset(generated, horizon=0)
    assert "windows=" in repr(ds)


def test_dataset_branches_and_pairs_share_start_frames(generated: Path) -> None:
    ds = Cosmos3TrajectoryDataset(generated, horizon=4)
    groups = ds.branches()
    assert set(groups) == {"ep0", "ep1", "ep2", "ep3"} and all(len(v) == 2 for v in groups.values())
    for a, b in ds.branch_pairs():
        assert torch.equal(ds[a]["frames"][0], ds[b]["frames"][0]) and ds[a]["meta"]["start"] == 0
    assert len(ds.branch_pairs()) == 4
    assert len(ds.branches(start_only=False)["ep0"]) > 2


def test_dataset_compositional_split_and_hierarchy(generated: Path) -> None:
    ds = Cosmos3TrajectoryDataset(generated, horizon=4)
    assert ds.combinations() == {("franka", "reach"), ("franka", "grasp"), ("ur5", "reach"), ("ur5", "grasp")}
    seen, held = ds.split_by_combination([["franka", "grasp"]])
    assert seen and held and set(seen).isdisjoint(held) and len(seen) + len(held) == len(ds)
    assert all(ds.combination(i) == ("franka", "grasp") for i in held)
    with pytest.raises(ValueError, match="not present"):
        ds.split_by_combination([["franka", "fly"]])
    with pytest.raises(ValueError, match="removes"):
        ds.split_by_combination([["franka", "reach"], ["franka", "grasp"]])  # no franka left
    tree = build_hierarchy(ds.metadata())
    assert tree.n_nodes == 1 + 2 + 4 + 8  # root, 2 embodiments, 2x2 tasks, 2 primitives each
    lat, meta = ds.latents(0)
    assert lat.ndim == 2 and meta["prompt_id"] == "ep0"


def test_dataset_splits_and_config(generated: Path) -> None:
    parts = [Cosmos3TrajectoryDataset(generated, horizon=4, split=s) for s in ("train", "val", "test")]
    assert sum(len(p) for p in parts) == len(Cosmos3TrajectoryDataset(generated, horizon=4))
    with pytest.raises(ValueError):
        Cosmos3TrajectoryDataset(generated, split="dev")
    cfg = {"name": "cosmos3_generated", "root": str(generated), "horizon": 4, "image_size": 8, "stride": 1}
    ds = build_dataset(cfg, split="all")
    assert isinstance(ds, Cosmos3TrajectoryDataset) and ds[0]["frames"].shape[-1] == 8
    ds2 = Cosmos3TrajectoryDataset.from_config({"root": str(generated), "horizon": None, "image_size": [8, 12]})
    assert ds2[0]["frames"].shape[-2:] == (8, 12)
