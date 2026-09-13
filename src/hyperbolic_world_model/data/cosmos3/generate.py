"""Batch generation: start frame + action sequence -> video rollout with Cosmos, action-conditioned.

Standalone usage (see ``scripts/generate_cosmos3_trajectories.sh``)::

    python -m hyperbolic_world_model.data.cosmos3.generate \\
        --start-frames data/cosmos3_prompts/frames \\
        --actions data/cosmos3_prompts/actions.json \\
        --out "$DATA_ROOT/cosmos3_generated" --seed 0
    python -m ... --dry-run          # validate inputs and write the generation plan, no model

Inputs
------
``--start-frames``: a directory of images (``.png``/``.jpg``) or ``.npy`` arrays ``(H, W, 3)`` uint8.
``--actions``: a JSON file::

    {
      "action_dim": 7,
      "prompts": [
        {"prompt_id": "ep0001", "start_frame": "ep0001.png",
         "embodiment": "franka", "task": "pick_cup",
         "branches": [
            {"branch_id": "b0", "primitive": "reach", "actions": [[dx, dy, dz, droll, dpitch, dyaw, gripper], ...]},
            {"branch_id": "b1", "primitive": "grasp", "actions": [...]}
         ]}
      ]
    }

Every branch of a prompt shares the same start frame: that is what gives the long-horizon
consistency task its branching futures. Actions follow the convention of NVIDIA's action-conditioned
models: per-step relative end-effector deltas ``[xyz (3), euler (3), gripper (1)]`` (already scaled
as the model expects; NVIDIA multiplies raw deltas by ``action_scaler = 20``).

Outputs (under ``--out``): ``rollouts/<prompt_id>/<branch_id>/frames.npz`` (``frames`` uint8
``(T, H, W, 3)``, ``actions`` float32 ``(T - 1, a)``), ``meta.json``, an optional ``rollout.mp4``
preview when ``torchvision`` is importable, and one line per rollout appended to ``manifest.jsonl``.

The model call
--------------
Everything except the model call is plain numpy and is tested with a fake generator.
:class:`CosmosGenerator` wraps NVIDIA's ``cosmos_predict2`` action-conditioned inference API
(``Video2WorldInference.generate_vid2world`` with ``action=``), reproducing the chunked loop of
``cosmos_predict2/action_conditioned.py`` at the pinned upstream commit: the first frame is real,
the remaining frames of each chunk are zeros, actions are fed in chunks of ``chunk_size``, the last
generated frame seeds the next chunk. Chunks are stitched so that frame ``t + 1`` is the result of
action ``t`` (upstream's preview stitching repeats the boundary frame; see ``CosmosGenerator``). Which checkpoint
runs is the ``--model`` key registered in that package (``Cosmos-Predict2.5-2B/robot/action-cond``
today; select the Cosmos 3 Nano action-conditioned key once NVIDIA's package registers it). The
``cosmos_predict2`` package is not a dependency of this repository; install it separately per
NVIDIA's setup guide. Cosmos is loaded in inference mode only; no optimiser ever sees it.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hyperbolic_world_model.data.cosmos3 import default_root

DEFAULT_MODEL = "Cosmos-Predict2.5-2B/robot/action-cond"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
LEVEL_KEYS = ("embodiment", "task", "primitive")


# ---------------------------------------------------------------------------- inputs
@dataclass(frozen=True)
class Branch:
    branch_id: str
    actions: np.ndarray  # (T - 1, action_dim) float32
    primitive: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    start_frame_path: Path
    embodiment: str
    task: str
    branches: tuple[Branch, ...]


def load_start_frame(path: Path) -> np.ndarray:
    """Read a start frame as ``(H, W, 3)`` uint8 from ``.png``/``.jpg`` (via Pillow) or ``.npy``."""
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() in IMAGE_SUFFIXES:
        from PIL import Image  # Pillow ships with matplotlib, an allowed dependency

        arr = np.asarray(Image.open(path).convert("RGB"))
    else:
        raise ValueError(f"unsupported start frame {path}: expected .png, .jpg or .npy")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"start frame {path} must be (H, W, 3), got {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"start frame {path} must be uint8, got {arr.dtype}")
    return arr


def load_action_spec(actions_json: Path, start_frames_dir: Path) -> list[Prompt]:
    """Parse and validate the action-sequence JSON against the start-frame directory."""
    spec = json.loads(Path(actions_json).read_text())
    action_dim = int(spec["action_dim"])
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for p in spec["prompts"]:
        pid = str(p["prompt_id"])
        if pid in seen:
            raise ValueError(f"duplicate prompt_id {pid!r}")
        seen.add(pid)
        frame_path = Path(start_frames_dir) / p["start_frame"]
        if not frame_path.exists():
            raise FileNotFoundError(f"prompt {pid}: start frame {frame_path} not found")
        branches = []
        bids: set[str] = set()
        for b in p["branches"]:
            bid = str(b["branch_id"])
            if bid in bids:
                raise ValueError(f"prompt {pid}: duplicate branch_id {bid!r}")
            bids.add(bid)
            actions = np.asarray(b["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != action_dim or actions.shape[0] < 1:
                raise ValueError(
                    f"prompt {pid} branch {bid}: actions must be (T-1 >= 1, {action_dim}), got {actions.shape}"
                )
            extra = {k: v for k, v in b.items() if k not in {"branch_id", "actions", "primitive"}}
            branches.append(Branch(bid, actions, str(b["primitive"]), extra))
        if not branches:
            raise ValueError(f"prompt {pid}: no branches")
        prompts.append(Prompt(pid, frame_path, str(p["embodiment"]), str(p["task"]), tuple(branches)))
    if not prompts:
        raise ValueError("no prompts in action spec")
    return prompts


# ---------------------------------------------------------------------------- generator interface
class ActionConditionedGenerator(Protocol):
    """What :func:`generate_rollouts` needs from a world model. Inference only."""

    #: Free-form description recorded in every ``meta.json`` (model key, checkpoint, settings).
    model_info: dict[str, Any]

    def generate(self, start_frame: np.ndarray, actions: np.ndarray, seed: int) -> np.ndarray:
        """``(H, W, 3)`` uint8 start frame + ``(T - 1, a)`` actions -> ``(T, H, W, 3)`` uint8 video."""
        ...


class CosmosGenerator:
    """Adapter over NVIDIA's ``cosmos_predict2`` action-conditioned inference (see module docstring).

    Args:
        model: model key registered in ``cosmos_predict2.config.MODEL_CHECKPOINTS``.
        checkpoint_path: local checkpoint path; ``None`` resolves NVIDIA's registered URI.
        config_file: experiment config file of the action-conditioned model.
        chunk_size: actions per model call (NVIDIA default 12).
        guidance, num_steps, negative_prompt, resolution: sampler settings, NVIDIA defaults.
        client: an already constructed ``Video2WorldInference`` (tests inject a fake); built lazily
            from the package otherwise.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        checkpoint_path: str | None = None,
        config_file: str | None = None,
        chunk_size: int = 12,
        guidance: float = 7.0,
        num_steps: int = 35,
        negative_prompt: str | None = None,
        resolution: str = "none",
        client: Any | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        self.model = model
        self.checkpoint_path = checkpoint_path
        self.config_file = config_file
        self.chunk_size = int(chunk_size)
        self.guidance = float(guidance)
        self.num_steps = int(num_steps)
        self.negative_prompt = negative_prompt
        self.resolution = resolution
        self._client = client
        self.model_info = {
            "backend": "cosmos_predict2.Video2WorldInference.generate_vid2world",
            "model": model,
            "checkpoint_path": checkpoint_path,
            "chunk_size": self.chunk_size,
            "guidance": self.guidance,
            "num_steps": self.num_steps,
            "resolution": resolution,
        }

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        try:
            from cosmos_predict2.config import MODEL_KEYS, MODEL_CHECKPOINTS  # type: ignore[import-not-found]
            from cosmos_predict2._src.predict2.inference.video2world import (  # type: ignore[import-not-found]
                Video2WorldInference,
            )
        except ImportError as e:
            raise RuntimeError(
                "NVIDIA's cosmos_predict2 package is not installed. It is not a dependency of "
                "hyperbolic-world-model; install it per https://github.com/nvidia-cosmos/cosmos-predict2.5 "
                "(GPU required) before running generation."
            ) from e
        if self.model not in MODEL_KEYS:
            raise KeyError(f"model {self.model!r} is not registered; known: {sorted(MODEL_KEYS)}")
        ckpt = MODEL_CHECKPOINTS[MODEL_KEYS[self.model]]
        import torch

        torch.set_grad_enabled(False)  # inference only; Cosmos is never trained here
        kwargs = dict(
            experiment_name=ckpt.experiment,
            ckpt_path=self.checkpoint_path or ckpt.s3.uri,
            s3_credential_path="",
            context_parallel_size=1,
        )
        if self.config_file:
            kwargs["config_file"] = self.config_file
        return Video2WorldInference(**kwargs)

    def generate(self, start_frame: np.ndarray, actions: np.ndarray, seed: int) -> np.ndarray:
        """NVIDIA's chunked action-conditioned loop; returns ``(T, H, W, 3)`` uint8 with ``T = len(actions) + 1``."""
        import torch

        if start_frame.ndim != 3 or start_frame.dtype != np.uint8:
            raise ValueError("start_frame must be (H, W, 3) uint8")
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] < 1:
            raise ValueError("actions must be (T - 1 >= 1, action_dim)")
        n_actions = actions.shape[0]
        img = start_frame
        chunks: list[np.ndarray] = []
        for start in range(0, n_actions, self.chunk_size):
            chunk = actions[start : start + self.chunk_size]
            pad = self.chunk_size - chunk.shape[0]
            if pad > 0:  # upstream zero-pads the last chunk to chunk_size
                chunk = np.concatenate([chunk, np.zeros((pad, chunk.shape[1]), dtype=chunk.dtype)], axis=0)
            num_video_frames = self.chunk_size + 1
            img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W) uint8
            vid = torch.cat([img_t, torch.zeros_like(img_t).repeat(num_video_frames - 1, 1, 1, 1)], dim=0)
            vid = vid.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) uint8, first frame real
            video = self.client.generate_vid2world(
                prompt="",
                input_path=vid,
                action=torch.from_numpy(chunk).float(),
                guidance=self.guidance,
                num_video_frames=num_video_frames,
                num_latent_conditional_frames=1,
                resolution=self.resolution,
                seed=int(seed) + start,  # deterministic per (seed, chunk)
                negative_prompt=self.negative_prompt,
                num_steps=self.num_steps,
            )
            frames = ((video[0].float().clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
            chunks.append(frames)  # (chunk_size + 1, H, W, 3)
            img = frames[-1]
        # Stitch so that frame t + 1 is the result of action t: keep the whole first chunk, then drop
        # each later chunk's conditioning frame (it is the previous chunk's last frame). Upstream's
        # preview-video stitching keeps that frame and drops the chunk's last one instead, which
        # duplicates every chunk-boundary frame; for training data the alignment matters, so we differ.
        parts = [chunks[0]] + [c[1:] for c in chunks[1:]]
        video_np = np.concatenate(parts, axis=0)
        return video_np[: n_actions + 1]  # drop frames produced for zero-padded actions


# ---------------------------------------------------------------------------- outputs
def write_rollout(
    out_root: Path,
    prompt: Prompt,
    branch: Branch,
    frames: np.ndarray,
    seed: int,
    model_info: dict[str, Any],
    write_mp4: bool = True,
    fps: int = 10,
) -> dict[str, Any]:
    """Save ``frames.npz`` + ``meta.json`` (+ optional mp4) and return the manifest record."""
    if frames.ndim != 4 or frames.shape[0] != branch.actions.shape[0] + 1 or frames.dtype != np.uint8:
        raise ValueError(f"frames must be (T = len(actions) + 1, H, W, 3) uint8, got {frames.shape} {frames.dtype}")
    d = Path(out_root) / "rollouts" / prompt.prompt_id / branch.branch_id
    d.mkdir(parents=True, exist_ok=True)
    frames_path = d / "frames.npz"
    np.savez_compressed(frames_path, frames=frames, actions=branch.actions.astype(np.float32))
    mp4_path: str | None = None
    if write_mp4:
        mp4_path = _maybe_write_mp4(d / "rollout.mp4", frames, fps)
    record = {
        "prompt_id": prompt.prompt_id,
        "branch_id": branch.branch_id,
        "embodiment": prompt.embodiment,
        "task": prompt.task,
        "primitive": branch.primitive,
        "frames_path": str(frames_path.relative_to(out_root)),
        "mp4_path": mp4_path and str(Path(mp4_path).relative_to(out_root)),
        "latents_path": None,
        "num_frames": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
        "action_dim": int(branch.actions.shape[1]),
        "seed": int(seed),
        "start_frame": str(prompt.start_frame_path.name),
        "model_info": model_info,
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "host": platform.node(),
        **branch.extra,
    }
    (d / "meta.json").write_text(json.dumps(record, indent=2))
    return record


def _maybe_write_mp4(path: Path, frames: np.ndarray, fps: int) -> str | None:
    try:
        import torch
        from torchvision.io import write_video  # optional: comes with the vjepa2 extra
    except ImportError:
        return None
    try:
        write_video(str(path), torch.from_numpy(np.ascontiguousarray(frames)), fps=fps)
    except Exception:  # noqa: BLE001 - a missing codec must not fail generation; npz is canonical
        return None
    return str(path)


def append_manifest(out_root: Path, records: Iterable[dict[str, Any]]) -> Path:
    path = Path(out_root) / "manifest.jsonl"
    with path.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def read_manifest(out_root: Path) -> list[dict[str, Any]]:
    path = Path(out_root) / "manifest.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def branch_seed(base_seed: int, prompt_id: str, branch_id: str) -> int:
    """Deterministic per-branch seed: same inputs -> same seed, different branches -> different seeds."""
    import zlib

    return (int(base_seed) * 1_000_003 + zlib.crc32(f"{prompt_id}/{branch_id}".encode())) % (2**31 - 1)


# ---------------------------------------------------------------------------- driver
def generate_rollouts(
    prompts: list[Prompt],
    generator: ActionConditionedGenerator,
    out_root: Path,
    seed: int = 0,
    write_mp4: bool = True,
    fps: int = 10,
    skip_existing: bool = True,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Generate every branch of every prompt and append the manifest. Idempotent with ``skip_existing``."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    done = {(r["prompt_id"], r["branch_id"]) for r in read_manifest(out_root)} if skip_existing else set()
    records = []
    for prompt in prompts:
        start = load_start_frame(prompt.start_frame_path)
        for branch in prompt.branches:
            if (prompt.prompt_id, branch.branch_id) in done:
                continue
            s = branch_seed(seed, prompt.prompt_id, branch.branch_id)
            frames = generator.generate(start, branch.actions, s)
            rec = write_rollout(out_root, prompt, branch, frames, s, generator.model_info, write_mp4, fps)
            append_manifest(out_root, [rec])
            records.append(rec)
            if progress:
                progress(f"{prompt.prompt_id}/{branch.branch_id}: {frames.shape[0]} frames (seed {s})")
    return records


def write_plan(prompts: list[Prompt], out_root: Path, seed: int, model: str) -> Path:
    """Dry run: record what would be generated without loading any model."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "model": model,
        "seed": seed,
        "rollouts": [
            {
                "prompt_id": p.prompt_id,
                "branch_id": b.branch_id,
                "num_frames": int(b.actions.shape[0]) + 1,
                "action_dim": int(b.actions.shape[1]),
                "seed": branch_seed(seed, p.prompt_id, b.branch_id),
                "embodiment": p.embodiment,
                "task": p.task,
                "primitive": b.primitive,
            }
            for p in prompts
            for b in p.branches
        ],
    }
    path = out_root / "generation_plan.json"
    path.write_text(json.dumps(plan, indent=2))
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-frames", type=Path, required=True, help="directory of start frames (.png/.jpg/.npy)")
    p.add_argument("--actions", type=Path, required=True, help="JSON file of action sequences (see module doc)")
    p.add_argument("--out", type=Path, default=None, help="output root; default $DATA_ROOT/cosmos3_generated")
    p.add_argument("--model", default=DEFAULT_MODEL, help="cosmos_predict2 model key (action-conditioned)")
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--config-file", default=None)
    p.add_argument("--chunk-size", type=int, default=12)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--num-steps", type=int, default=35)
    p.add_argument("--resolution", default="none", help='"H,W" or "none" for the model default')
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=10, help="fps of the optional mp4 preview")
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="regenerate rollouts already in the manifest")
    p.add_argument("--dry-run", action="store_true", help="validate inputs and write generation_plan.json only")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, generator: ActionConditionedGenerator | None = None) -> list[dict[str, Any]]:
    args = parse_args(argv)
    out_root = args.out or default_root()
    prompts = load_action_spec(args.actions, args.start_frames)
    if args.dry_run:
        path = write_plan(prompts, out_root, args.seed, args.model)
        print(f"dry run: {sum(len(p.branches) for p in prompts)} rollouts planned -> {path}")
        return []
    gen = generator or CosmosGenerator(
        model=args.model,
        checkpoint_path=args.checkpoint_path,
        config_file=args.config_file,
        chunk_size=args.chunk_size,
        guidance=args.guidance,
        num_steps=args.num_steps,
        resolution=args.resolution,
    )
    records = generate_rollouts(
        prompts, gen, out_root, seed=args.seed, write_mp4=not args.no_mp4, fps=args.fps, skip_existing=not args.overwrite, progress=print
    )
    print(f"generated {len(records)} rollouts under {out_root} (manifest: {Path(out_root) / 'manifest.jsonl'})")
    return records


if __name__ == "__main__":
    main()

__all__ = [
    "DEFAULT_MODEL",
    "ActionConditionedGenerator",
    "Branch",
    "CosmosGenerator",
    "Prompt",
    "append_manifest",
    "branch_seed",
    "generate_rollouts",
    "load_action_spec",
    "load_start_frame",
    "main",
    "read_manifest",
    "write_plan",
    "write_rollout",
]
