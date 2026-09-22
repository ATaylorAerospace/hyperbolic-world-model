# Reproducibility

## One-command paths

```bash
# Environment (exact versions from uv.lock); the vjepa2 extra adds timm for torch.hub
uv sync --extra vjepa2

# Unit tests + smoke experiment (CPU, no downloads, < 1 minute)
uv run pytest -v
bash scripts/run_smoke.sh

# Full pipeline (GPU, requires weights and data)
bash scripts/download_weights.sh          # V-JEPA 2, DINOv2, Cosmos 3 Nano (HF_TOKEN for Cosmos)
bash scripts/generate_cosmos3_trajectories.sh
bash scripts/run_baseline.sh
bash scripts/run_curvature_sweep.sh both
bash scripts/make_report.sh               # regenerates every table and figure from outputs/
```

## Seeds

- `seed` in the experiment config seeds `torch` (CPU and CUDA) at the start of `run()`.
- The synthetic dataset and synthetic encoder are seeded independently of the training seed so
  that changing `seed` changes only the head initialisation and data order.
- The frozen projection inside every predictor head is seeded by `models.head.seed` so that the
  Euclidean and hyperbolic heads see the *same* projected latents at the same seed.
- Sweeps run `seed ∈ {0, 1, 2}`; tables report mean ± std over seeds.
- Cosmos 3 generation is seeded per prompt; the seed is stored in `manifest.jsonl`.

Full determinism on GPU additionally requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` and
`torch.use_deterministic_algorithms(True)`; these are not enabled by default because they slow
attention kernels, and seed-to-seed variance is reported anyway.

## Lockfile

`uv.lock` pins every dependency (including transitive ones) and is committed. `uv sync` installs
exactly those versions. CI installs CPU-only torch from the PyTorch wheel index for speed; the
resolved versions of everything else match the lockfile. Do not edit `uv.lock` by hand; run
`uv lock` after changing `pyproject.toml`.

## Weights and data provenance

- Encoder weights: model id and optional `revision` (commit hash) in `configs/models/*.yaml`.
  Pin a `revision` before reporting results; the resolved config is saved next to every run.
- DROID: version recorded in `data/droid/metadata.json` by the user's download.
- Cosmos 3 outputs: model revision and seed in `data/cosmos3/manifest.jsonl`.

## What every run writes

```text
outputs/<experiment>/geometry=<name>,K=<curvature>,dim=<latent_dim>,seed=<seed>/
├── config.yaml                 # fully resolved Hydra config, including seed and model ids
├── metrics.json                # geometry + curvature + latent_dim + seed + per-task metrics + train loss history
├── <task>_curves.csv           # one per task: error vs horizon, dist0 vs depth, divergence vs horizon, seen vs unseen
└── hydra/                      # Hydra's own logs and override records
```

The seed is part of the directory name so the three seeds of a sweep never overwrite each other
and the report can show mean ± std. The report (`outputs/report/`) is a pure function of these
files: running `make_report.sh` twice gives byte-identical output (see `outputs/README.md` for
the list of tables and figures).

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `HF_TOKEN` | unset | Authenticates Hugging Face downloads (needed for gated Cosmos 3) |
| `WANDB_API_KEY` | unset | Enables Weights & Biases logging; JSON logs are always written |
| `DATA_ROOT` | `data` | Root for datasets |
| `CKPT_ROOT` | `checkpoints` | Root for encoder weights and trained heads |
| `HWM_DEVICE` | config `device` (`cpu`) | Forces the device (CI sets `cpu`) |

`training.cache_latents` (default `true`) encodes the dataset once with the frozen encoder and trains
the head on cached latents; set it to `false` only when the latents do not fit in memory.
