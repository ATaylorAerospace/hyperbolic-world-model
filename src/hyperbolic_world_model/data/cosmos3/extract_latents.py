"""Save the Cosmos video tokenizer's latents for every generated rollout.

Usage::

    python -m hyperbolic_world_model.data.cosmos3.extract_latents \\
        --root "$DATA_ROOT/cosmos3_generated" [--pooling mean|flatten]

For each manifest entry the rollout video is encoded by the tokenizer encoder (decoder never
loaded) and saved to ``latents/<prompt_id>/<branch_id>.npz`` with ``latents`` float16 ``(T', D)``
and ``meta`` (JSON). The manifest is rewritten with ``latents_path`` filled in.

These latents answer a question independent of any head we train: *does a large generative world
model already organise its latent space in a tree-like way?* (:mod:`metrics.gromov_hyperbolicity`).
The tokenizer output is a latent video ``(C, T', H', W')`` with 4x temporal and 8x spatial
compression for the Cosmos-Predict2.5 tokenizer; ``pooling="mean"`` averages the spatial grid to
``(T', C)``, ``pooling="flatten"`` keeps it as ``(T', C * H' * W')``.

The model call is isolated in :class:`CosmosTokenizerBackend`, which reaches the tokenizer through
the same ``Video2WorldInference`` object as generation (``model.tokenizer.encode`` on a
``(1, 3, T, H, W)`` tensor in ``[-1, 1]``, per ``cosmos_predict2``'s tokenizer interface). Everything
else is numpy and is tested with a fake backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hyperbolic_world_model.data.cosmos3 import default_root
from hyperbolic_world_model.data.cosmos3.generate import DEFAULT_MODEL, read_manifest


class TokenizerBackend(Protocol):
    """What :func:`extract_latents` needs: a frozen video tokenizer encoder."""

    model_info: dict[str, Any]

    def encode(self, video: np.ndarray) -> np.ndarray:
        """``(T, H, W, 3)`` uint8 video -> latent video ``(C, T', H', W')`` float32."""
        ...


class CosmosTokenizerBackend:
    """Adapter over the tokenizer bundled with NVIDIA's ``cosmos_predict2`` models (encoder only)."""

    def __init__(self, model: str = DEFAULT_MODEL, checkpoint_path: str | None = None, client: Any | None = None) -> None:
        self.model = model
        self.checkpoint_path = checkpoint_path
        self._client = client
        self.model_info = {"backend": "cosmos_predict2 tokenizer.encode", "model": model, "checkpoint_path": checkpoint_path}

    @property
    def client(self) -> Any:
        if self._client is None:
            from hyperbolic_world_model.data.cosmos3.generate import CosmosGenerator

            self._client = CosmosGenerator(model=self.model, checkpoint_path=self.checkpoint_path).client
        return self._client

    def encode(self, video: np.ndarray) -> np.ndarray:
        import torch

        if video.ndim != 4 or video.dtype != np.uint8:
            raise ValueError("video must be (T, H, W, 3) uint8")
        tokenizer = self.client.model.tokenizer
        x = torch.from_numpy(np.ascontiguousarray(video)).permute(3, 0, 1, 2).unsqueeze(0)  # (1, 3, T, H, W)
        x = (x.float() / 127.5 - 1.0).to(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
        with torch.no_grad():
            latent = tokenizer.encode(x)  # (1, C, T', H', W')
        return latent[0].float().cpu().numpy()


def pool_latent(latent: np.ndarray, pooling: str = "mean") -> np.ndarray:
    """``(C, T', H', W')`` -> ``(T', D)`` with ``D = C`` (mean over space) or ``C * H' * W'`` (flatten)."""
    if latent.ndim != 4:
        raise ValueError(f"latent must be (C, T', H', W'), got {latent.shape}")
    c, t = latent.shape[:2]
    if pooling == "mean":
        return latent.mean(axis=(2, 3)).T.astype(np.float32)  # (T', C)
    if pooling == "flatten":
        return latent.transpose(1, 0, 2, 3).reshape(t, -1).astype(np.float32)
    raise ValueError(f"pooling must be 'mean' or 'flatten', got {pooling!r}")


def save_latents(root: Path, record: dict[str, Any], pooled: np.ndarray, pooling: str, model_info: dict[str, Any]) -> Path:
    d = Path(root) / "latents" / record["prompt_id"]
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{record['branch_id']}.npz"
    meta = {
        "prompt_id": record["prompt_id"],
        "branch_id": record["branch_id"],
        "embodiment": record["embodiment"],
        "task": record["task"],
        "primitive": record["primitive"],
        "num_frames": record["num_frames"],
        "pooling": pooling,
        "latent_shape": list(pooled.shape),
        "model_info": model_info,
    }
    np.savez_compressed(path, latents=pooled.astype(np.float16), meta=json.dumps(meta))
    return path


def load_latents(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read one latents file: ``(T', D)`` float32 and its meta dict."""
    with np.load(path) as z:
        return z["latents"].astype(np.float32), json.loads(str(z["meta"]))


def rewrite_manifest(root: Path, records: list[dict[str, Any]]) -> Path:
    path = Path(root) / "manifest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def extract_latents(root: Path, backend: TokenizerBackend, pooling: str = "mean", skip_existing: bool = True) -> list[Path]:
    """Encode every rollout in the manifest; returns the latents files written."""
    root = Path(root)
    records = read_manifest(root)
    if not records:
        raise FileNotFoundError(f"no manifest.jsonl under {root}; run generate.py first")
    written: list[Path] = []
    for rec in records:
        if skip_existing and rec.get("latents_path") and (root / rec["latents_path"]).exists():
            continue
        with np.load(root / rec["frames_path"]) as z:
            frames = z["frames"]
        latent = backend.encode(frames)
        pooled = pool_latent(latent, pooling)
        path = save_latents(root, rec, pooled, pooling, backend.model_info)
        rec["latents_path"] = str(path.relative_to(root))
        rec["latent_shape"] = list(pooled.shape)
        written.append(path)
    rewrite_manifest(root, records)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=None, help="generated data root; default $DATA_ROOT/cosmos3_generated")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--pooling", choices=["mean", "flatten"], default="mean")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, backend: TokenizerBackend | None = None) -> list[Path]:
    args = parse_args(argv)
    root = args.root or default_root()
    be = backend or CosmosTokenizerBackend(model=args.model, checkpoint_path=args.checkpoint_path)
    written = extract_latents(root, be, pooling=args.pooling, skip_existing=not args.overwrite)
    print(f"wrote {len(written)} latent files under {Path(root) / 'latents'}")
    return written


if __name__ == "__main__":
    main()

__all__ = [
    "CosmosTokenizerBackend",
    "TokenizerBackend",
    "extract_latents",
    "load_latents",
    "main",
    "pool_latent",
    "rewrite_manifest",
    "save_latents",
]
