[![Hyperbolic Latent Evaluation](docs/header.png)](https://github.com/ATaylorAerospace/hyperbolic-world-model)

# 🌀 Hyperbolic Latent Evaluation for World Models 📐

[![CI](https://github.com/ATaylorAerospace/hyperbolic-world-model/actions/workflows/ci.yml/badge.svg)](https://github.com/ATaylorAerospace/hyperbolic-world-model/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![geoopt](https://img.shields.io/badge/geoopt-0.5%2B-6f42c1)](https://github.com/geoopt/geoopt)
[![Tests](https://img.shields.io/badge/tests-79%20passing-brightgreen)](#-verification)
[![Curvature: Swept](https://img.shields.io/badge/Curvature-Swept-orange)](docs/methodology.md)

**A research harness that retrains the action-conditioned predictor head of a frozen video world model (V-JEPA 2-AC, with DINO-WM as a lightweight second subject) in Euclidean, Poincaré and Lorentz latent spaces, and measures whether negative curvature buys better rollouts, hierarchy recovery and dimension efficiency, with every distance computed in the geometry of the model that produced it and curvature always swept.**

**Author: A Taylor**

> 🚧 **Status:** Geometry primitives stable · V-JEPA 2-AC head training · 79/79 tests passing.

---

## 🤔 The Problem

Embodied data is tree-shaped in at least three ways. **Embodiment hierarchies** (robot → arm → gripper → primitive skill) branch at every level. **Task decompositions** turn one instruction into sub-goals and sub-goals into primitives. **Branching futures** mean the set of states reachable from one frame grows roughly exponentially with the horizon. A tree with branching factor *b* has *b^h* leaves at depth *h*; the space that stores it needs volume that grows the same way.

Euclidean space does not have that volume. Its balls grow polynomially with radius, so any low-distortion embedding of a tree needs dimension that grows with the tree (Bourgain-type lower bounds). A world model with a Euclidean latent therefore has two bad options: crush distinct futures together, or spend latent dimensions to keep them apart. Either shows up as long-horizon rollout error, poor hierarchy recovery, and a steep dimension-versus-performance curve.

Hyperbolic space has exponential volume growth and embeds any tree with arbitrarily low distortion in two dimensions (Sarkar, 2011). The question this repository asks is empirical: **does a hyperbolic latent help a frozen video encoder's predictor head, and at which curvature and dimension?** The counter-hypothesis is equally live: the frozen encoder may already linearise the relevant structure and curvature may only add optimisation difficulty. The harness is built so either answer is visible and neither is baked in.

---

## 💡 The Solution

- 🧊 **Freeze** the upstream encoder (V-JEPA 2 ViT-L or DINOv2) so every geometry sees identical features. `FrozenEncoder.assert_frozen` guards every optimiser step and checkpoint.
- 📐 **Project** encoder latents onto the chosen manifold with a seeded, frozen projection followed by `expmap0`, so no geometry can win by collapsing its embedding.
- 🔮 **Predict** the next latent with an action-conditioned head whose update is a tangent vector transported along the manifold; loss is squared geodesic distance.
- 🎬 **Generate** controlled branching trajectories with NVIDIA Cosmos 3 Nano (start frame + action sequence → video), inference only.
- 🔬 **Probe** the delta-hyperbolicity of V-JEPA 2, DINOv2 and Cosmos 3 tokenizer latents, independent of any head we train.
- 🎛️ **Sweep** curvature × latent dimension × seed for both hyperbolic models via Hydra multirun. Curvature is never a fixed constant.
- 📏 **Measure** rollout error, distortion, mAP and dimension efficiency in each model's *native* geometry via a single `Manifold` interface.
- 📊 **Report** every table and figure by regenerating them from `outputs/` with one script, never by hand.

---

## 🏛️ Architecture

```text
 ┌──────────────────────────┐        ┌──────────────────────────────┐        ┌──────────────────────────────┐
 │  NVIDIA Cosmos 3 Nano    │        │  Frozen encoder              │        │  Metrics (native geometry)   │
 │  (inference only)        │        │  V-JEPA 2 ViT-L  (primary)   │        │                              │
 │                          │ video  │  DINOv2 base     (secondary) │        │  geodesic_error   ─┐         │
 │  start frame + actions ──┼───────▶│                              │        │  gromov_hyperb.    │         │
 │  ──▶ video rollouts      │        │  frames (B,T,C,H,W)          │        │  distortion / mAP  ├──▶ 📊   │
 │                          │        │     ──▶ latents (B,T,D)      │        │  dimension_effic.  │  report │
 │  tokenizer latents ──────┼───┐    └──────────────┬───────────────┘        │                   ─┘         │
 │  (δ-hyperbolicity probe) │   │                   │ z_t  (D-dim, Euclidean coords)                        │
 └──────────────────────────┘   │                   │                        └──────────────▲───────────────┘
                                │   ═══════════════ ▼ ═══ GEOMETRY BOUNDARY ═══════════════│═══════════════
                                │   ║ everything below lives on a Manifold; distances    ║ │
                                │   ║ never cross this line in Euclidean coordinates      ║ │
                                │   ═══════════════════════════════════════════════════════│═══════════════
                                │        frozen seeded projection + expmap0               │
                                │              ┌────────────┴────────────┐                │
                                │              ▼                         ▼                │
                                │   ┌────────────────────┐   ┌────────────────────────┐   │
                                │   │ EuclideanHead      │   │ HyperbolicHead         │   │
                                │   │ K = 0              │   │ Poincaré | Lorentz     │   │
                                │   │ s' = s + MLP(s,a)  │   │ K ∈ sweep (< 0)        │   │
                                │   │ loss: |s'-s*|²     │   │ s' = exp_s(P₀→s MLP)   │   │
                                │   └─────────┬──────────┘   │ loss: d_K(s', s*)²     │   │
                                │             │              └───────────┬────────────┘   │
                                │             └──────────────┬───────────┘                │
                                │                            │ predicted latents + manifold│
                                └────────────────────────────┴────────────────────────────┘
```

The geometry boundary is labelled like a trust boundary on purpose: code above it (encoders, data) knows nothing about curvature; code below it never computes a Euclidean norm on manifold coordinates. Crossing the boundary happens in exactly one place, `ActionConditionedPredictor.embed`.

```text
                    Manifold (geometry/base.py)  ── the only abstraction heads, metrics and tasks import
                    ┌─────────────────────────────────────────────────────────────────┐
                    │ expmap(x, u)   logmap(x, y)   dist(x, y)   proj(x)   ptransp(x, y, u) │
                    │ + derived: expmap0, logmap0, sqdist, dist0, geodesic, pairwise_dist   │
                    └───────────┬───────────────────────┬────────────────────────┬────────┘
                                │                       │                        │
                    ┌───────────▼─────────┐  ┌──────────▼───────────┐  ┌─────────▼──────────┐
                    │ Euclidean  (K = 0)  │  │ PoincareBall (K < 0) │  │ Lorentz   (K < 0)  │
                    │ x + u, y - x, |·|   │  │ Möbius ⊕, artanh,    │  │ Minkowski ⟨·,·⟩_L, │
                    │                     │  │ boundary clipping    │  │ arcosh, hyperboloid│
                    └─────────────────────┘  └──────────────────────┘  └────────────────────┘
                                              isometric to each other: tested to 1e-9
```

---

## 🧪 The Three Models

| Model | Role | Weights | License | Modified |
|---|---|---|---|---|
| NVIDIA Cosmos 3 | Action-conditioned trajectory generator (start frame + actions → video) and tokenizer-latent probe target for δ-hyperbolicity | `nvidia/Cosmos-3-Nano`, downloaded by the user, gated | OpenMDW 1.1 | No, frozen |
| V-JEPA 2-AC | Primary subject: frozen ViT-L video encoder; the action-conditioned predictor head is retrained from scratch in each geometry | `facebook/vjepa2-vitl-fpc64-256`, downloaded by the user | MIT (code) | Predictor head only |
| DINO-WM | Secondary lightweight subject: frozen DINOv2 per-frame encoder with the same retrained heads | `facebook/dinov2-base`, downloaded by the user | Apache 2.0 | Predictor head only |

---

## 📏 Metrics

| Metric | What it measures | Geometry-aware | Module |
|---|---|---|---|
| Geodesic rollout error | Open-loop prediction error vs horizon, plus a "nothing moves" static baseline for normalisation | Yes: `manifold.dist` of the head's own geometry | `metrics/geodesic_error.py` |
| Normalised geodesic error | Error divided by the distance the true latent travelled; equals 1 for a static predictor in every geometry, so it is comparable across curvatures | Yes | `metrics/geodesic_error.py` |
| Gromov δ-hyperbolicity | How tree-like a latent point cloud is (0 for trees), via Gromov products with subsampling; also diameter-normalised | Yes: pairwise distances from any `Manifold` | `metrics/gromov_hyperbolicity.py` |
| Average distortion | Scale-fitted relative error between manifold distances and tree hop counts | Yes | `metrics/distortion.py` |
| mAP | Precision of retrieving true tree neighbours by manifold distance | Yes | `metrics/distortion.py` |
| Dimension efficiency | Metric-vs-latent-dimension curves, area under curve on a log₂ axis, dimension needed to reach a threshold | Consumes the above | `metrics/dimension_efficiency.py` |

---

## 🗺️ Tasks

| Task | Hypothesis tested | Data source | Falsified if |
|---|---|---|---|
| Latent rollout | At equal dimension the best swept curvature gives lower normalised geodesic error than Euclidean at horizons > 1 | DROID, Cosmos 3 rollouts, synthetic (CI) | No curvature beats Euclidean at any horizon > 1 by more than one seed std, at any dimension |
| Hierarchy reconstruction | Per-trajectory latents recover embodiment > task > primitive with lower distortion and higher mAP, and distance from origin tracks depth | DROID metadata (synthetic tree in CI) | Best curvature does not improve both distortion and mAP beyond seed std, or depth correlation is not positive |
| Long-horizon consistency | Latent divergence between branching rollouts tracks video divergence with higher correlation and saturates later in hyperbolic space | Cosmos 3 generated branches | Correlation is not higher or saturation not later for the best curvature |
| Compositional generalisation | The unseen-minus-seen error gap on held-out (embodiment, primitive) pairs is smaller in hyperbolic space | DROID with `holdout_combinations` | Gap is not smaller, or is smaller only because seen error got worse |

---

## 🛡️ Experimental Guarantees

> **Rule 1: distances are computed in the model's native geometry.** A Euclidean head is scored with Euclidean distance, a Poincaré head with the Poincaré geodesic at its trained curvature, a Lorentz head with the hyperboloid geodesic. No metric maps one model's latents into another's space. Every metric takes a `Manifold` and calls `manifold.dist`; there is no Euclidean fallback path. Every `TaskResult` records `geometry` and `curvature` next to its numbers.
> *Enforced by* `tests/test_smoke_experiment.py::test_smoke_runs_end_to_end[poincare|lorentz|euclidean]` (asserts the recorded geometry matches the manifold that produced the metrics) and `tests/geometry/test_lorentz.py::test_isometric_to_poincare_ball` (the two hyperbolic geometries agree with each other to 1e-9, so "native" is not "arbitrary").
>
> **Rule 2: curvature is swept, never fixed.** Hyperbolic results come only from `configs/experiments/poincare_sweep.yaml` and `lorentz_sweep.yaml`, which cross `curvature ∈ {-0.1, -0.25, -0.5, -1.0, -2.0, -4.0}` with latent dimension and seed. Tables show the whole curve. The `-1.0` default in `configs/geometry/*.yaml` exists only so single debug runs have a value.
> *Enforced by* `tests/test_smoke_experiment.py::test_smoke_config_is_tiny_and_cpu` (curvature is read from config, never hard-coded) and `tests/geometry/test_poincare.py::test_build_manifold_from_config` (the manifold is built from the config value); the sweep grids live in version-controlled YAML.
>
> **Invariant: the encoder is frozen.** `build_encoder` refuses `encoder.frozen: false`. `FrozenEncoder.assert_frozen` runs before the first optimiser step and before every checkpoint write. Only `predictor.state_dict()` is ever saved.
> *Enforced by* `tests/test_smoke_experiment.py::test_encoder_stays_frozen`.

---

## 🔧 Configuration

Hydra config groups (root: `configs/config.yaml`; select with `group=option`, run an experiment with `experiments=<name>`):

| Group | Options | Default | Notes |
|---|---|---|---|
| `geometry` | `euclidean`, `poincare`, `lorentz` | `euclidean` | `curvature` is the sectional curvature K; `-1.0` in the hyperbolic files is a placeholder for sweeps |
| `models` | `vjepa2_ac`, `dino_wm`, `synthetic` | `vjepa2_ac` | `encoder.frozen` must be `true`; `head.type` is `euclidean` or `hyperbolic`; `head.latent_dim` is swept |
| `data` | `droid`, `cosmos3_generated`, `synthetic` | `droid` | Roots resolve from `DATA_ROOT` |
| `tasks` | `latent_rollout`, `hierarchy_reconstruction`, `long_horizon_consistency`, `compositional_generalization`, `all` | `latent_rollout` | Each option is a list of task configs |
| `experiments` | `smoke`, `baseline_euclidean`, `poincare_sweep`, `lorentz_sweep` | none | `_global_` packages that override the groups above; sweeps set `hydra.mode: MULTIRUN` |

Environment variables (copy `.env.example` to `.env`):

| Variable | Default | Effect |
|---|---|---|
| `HF_TOKEN` | unset | Authenticates Hugging Face downloads; required for the gated Cosmos 3 Nano weights |
| `WANDB_API_KEY` | unset | Enables Weights & Biases logging; JSON/CSV outputs are always written regardless |
| `DATA_ROOT` | `data` | Root for DROID shards, Cosmos 3 prompts/rollouts/latents |
| `CKPT_ROOT` | `checkpoints` | Root for downloaded encoder weights and trained predictor heads |

---

## 📁 Repository Layout

```text
.
├── README.md                          # this file; regenerated at the end of every phase
├── LICENSE                            # Apache 2.0, © 2026 A Taylor
├── THIRD_PARTY_LICENSES.md            # upstream code and weight licences; nothing redistributed
├── CONTRIBUTING.md                    # ground rules: Manifold-only geometry, frozen encoders, swept curvature
├── pyproject.toml                     # uv-compatible metadata, ruff and pytest config
├── uv.lock                            # generated by `uv lock`; pins every dependency
├── .gitignore                         # Python, data/, checkpoints/, outputs/, .env, wandb/
├── .env.example                       # HF_TOKEN, WANDB_API_KEY, DATA_ROOT, CKPT_ROOT
├── .github/workflows/ci.yml           # ruff, pytest on geometry/ and metrics/ (CPU), one smoke benchmark
├── configs/
│   ├── config.yaml                    # Hydra root: defaults list, output_dir pattern, training block
│   ├── geometry/                      # euclidean.yaml, poincare.yaml (K swept), lorentz.yaml
│   ├── models/                        # vjepa2_ac.yaml, dino_wm.yaml, synthetic.yaml (CI fixture)
│   ├── data/                          # droid.yaml, cosmos3_generated.yaml, synthetic.yaml
│   ├── tasks/                         # one YAML per task plus all.yaml
│   └── experiments/                   # smoke.yaml (CI), baseline_euclidean.yaml, poincare_sweep.yaml, lorentz_sweep.yaml
├── src/hyperbolic_world_model/
│   ├── __init__.py                    # package version
│   ├── geometry/
│   │   ├── base.py                    # Manifold ABC: expmap, logmap, dist, proj, ptransp (+ derived ops)
│   │   ├── euclidean.py               # flat baseline, K = 0
│   │   ├── poincare.py                # Möbius addition, exp/log maps, geodesic distance, boundary clipping
│   │   ├── lorentz.py                 # hyperboloid model, Minkowski inner product, isometry to the ball
│   │   └── utils.py                   # stable artanh/arcosh, norm clipping, per-dtype epsilons
│   ├── models/
│   │   ├── encoders/                  # FrozenEncoder contract; vjepa2.py, dino.py (HF loaders), synthetic.py (CI)
│   │   ├── predictors/                # ActionConditionedPredictor, EuclideanHead, HyperbolicHead (geodesic loss)
│   │   └── registry.py                # build_manifold / build_encoder / build_predictor / build_model from config
│   ├── data/
│   │   ├── droid.py                   # action normalisation + chunking (done); DROID loader (TODO phase 2)
│   │   ├── synthetic.py               # in-memory linear-dynamics trajectories for the smoke run
│   │   ├── hierarchies.py             # embodiment > task > primitive tree, hop-count metric, adjacency
│   │   └── cosmos3/                   # generate.py, extract_latents.py, dataset.py (CLI skeletons, TODO phase 2)
│   ├── metrics/
│   │   ├── geodesic_error.py          # rollout error in native geometry, per-horizon and normalised
│   │   ├── gromov_hyperbolicity.py    # δ estimator via Gromov products with subsampling
│   │   ├── distortion.py              # scale-fitted average distortion and mAP
│   │   └── dimension_efficiency.py    # metric-vs-dimension tables, AUC, dimension-to-reach
│   ├── tasks/
│   │   ├── base.py                    # Task ABC and TaskResult (geometry + curvature stamped on every result)
│   │   ├── latent_rollout.py          # implemented: open-loop rollout vs static baseline
│   │   ├── hierarchy_reconstruction.py       # hypothesis + plan documented, run() TODO phase 2
│   │   ├── long_horizon_consistency.py       # needs Cosmos 3 branches, run() TODO phase 2
│   │   └── compositional_generalization.py   # held-out combinations, run() TODO phase 2
│   ├── training/
│   │   ├── train_predictor.py         # Hydra entry point; trains the head only; asserts encoder frozen
│   │   └── riemannian_optim.py        # AdamW, or geoopt RiemannianAdam if any ManifoldParameter exists
│   └── reporting/
│       ├── tables.py                  # collect metrics.json → long table → Markdown
│       ├── curvature_sweep_plots.py   # metric vs curvature, error vs horizon (matplotlib, headless)
│       └── make_report.py             # regenerates every table and figure from outputs/
├── scripts/
│   ├── download_weights.sh            # V-JEPA 2, DINOv2, Cosmos 3 Nano via huggingface_hub; reads HF_TOKEN
│   ├── generate_cosmos3_trajectories.sh   # generate rollouts, then extract tokenizer latents
│   ├── run_smoke.sh                   # CPU end-to-end run used by CI
│   ├── run_baseline.sh                # Euclidean baseline on DROID
│   ├── run_curvature_sweep.sh         # Poincaré and/or Lorentz multirun sweeps
│   └── make_report.sh                 # wraps reporting/make_report.py
├── tests/
│   ├── conftest.py                    # seeds, float32/float64 fixture
│   ├── geometry/                      # test_poincare.py, test_lorentz.py, test_euclidean.py
│   ├── metrics/                       # test_gromov_hyperbolicity.py, test_distortion.py
│   └── test_smoke_experiment.py       # runs configs/experiments/smoke.yaml on CPU in all three geometries
├── data/README.md                     # expected dataset layout (directory gitignored)
├── checkpoints/README.md              # expected weight layout (directory gitignored)
├── outputs/README.md                  # what every run writes (directory gitignored)
└── docs/
    ├── header.png                     # README header graphic (generated; to be replaced by a designed one)
    ├── methodology.md                 # hypotheses, falsification criteria, the two rules, numerical caveats
    ├── cosmos3_usage.md               # how Cosmos 3 is used and, explicitly, how it is not
    └── reproducibility.md             # seeds, lockfile, one-command reproduction path
```

---

## 🏁 Quick Start

1. **Clone**

   ```bash
   git clone https://github.com/ATaylorAerospace/hyperbolic-world-model
   cd hyperbolic-world-model
   ```

2. **Install** (uv creates `.venv` from the lockfile; add `--index https://download.pytorch.org/whl/cpu` style overrides for a CPU-only torch as CI does)

   ```bash
   uv sync
   cp .env.example .env    # fill in HF_TOKEN if you need gated weights
   ```

3. **Download** frozen weights (no datasets are downloaded; nothing is downloaded during install or tests)

   ```bash
   bash scripts/download_weights.sh all      # or: vjepa2 | dinov2 | cosmos3
   ```

4. **Smoke** test the whole pipeline on CPU in seconds (synthetic data, synthetic frozen encoder, Poincaré head)

   ```bash
   uv run pytest -v
   uv run bash scripts/run_smoke.sh
   ```

5. **Baseline** the Euclidean head on DROID with all tasks

   ```bash
   bash scripts/run_baseline.sh
   ```

6. **Sweep** curvature × latent dimension × seed for both hyperbolic models

   ```bash
   bash scripts/run_curvature_sweep.sh both
   ```

7. **Report** by regenerating every table and figure from `outputs/`

   ```bash
   bash scripts/make_report.sh
   ```

---

## 🧩 Components

| Module | Role | Key Guarantees | Status |
|---|---|---|---|
| `geometry/base.py` | `Manifold` interface | Single abstraction for heads, metrics, tasks; adding a geometry touches only `geometry/` and one config | ✅ |
| `geometry/euclidean.py` | Flat baseline | Identical code path to curved geometries; rejects non-zero curvature | ✅ |
| `geometry/poincare.py` | Poincaré ball | exp/log inverse, symmetric distance, triangle inequality, gradcheck, boundary clipping; float32 distances capped at ≈ 6.2 by design | ✅ |
| `geometry/lorentz.py` | Hyperboloid | On-manifold after every op, transport is isometric, isometric to the ball to 1e-9, float32 self-distance noise ≤ 2e-3·x₀ | ✅ |
| `geometry/utils.py` | Stable primitives | Clamped `artanh`/`arcosh` with exact gradients; per-dtype epsilons | ✅ |
| `metrics/geodesic_error.py` | Rollout error | Takes a `Manifold`; no Euclidean fallback; static baseline for normalisation | ✅ |
| `metrics/gromov_hyperbolicity.py` | δ-hyperbolicity | δ = 0 on tree metrics, > 0 on grids; scale covariant; chunked max-min product | ✅ |
| `metrics/distortion.py` | Distortion, mAP | Scale-fitted; 0 distortion / 1.0 mAP on a perfect embedding | ✅ |
| `metrics/dimension_efficiency.py` | Dimension curves | Tidy tables, log₂ AUC, dimension-to-reach | ✅ |
| `models/encoders/` | Frozen encoders | `assert_frozen`, `train()` is a no-op; synthetic encoder for CI; V-JEPA 2 forward awaits validation on real weights | 🚧 |
| `models/predictors/` | Heads | Frozen seeded projection prevents collapse; `HyperbolicHead(Euclidean)` equals `EuclideanHead` exactly | ✅ |
| `models/registry.py` | Config → objects | Refuses `frozen: false`; checks `embed_dim` against the loaded encoder | ✅ |
| `data/hierarchies.py` | Metadata → tree | Hop-count metric is 0-hyperbolic (used as the δ reference) | ✅ |
| `data/synthetic.py` | CI trajectories | Deterministic, learnable linear dynamics, two-level hierarchy in metadata | ✅ |
| `data/droid.py` | DROID | Action stats and chunking done; loader raises `NotImplementedError` with plan | 🚧 |
| `data/cosmos3/` | Cosmos 3 I/O | CLI skeletons; inference-only by construction; no API guessed | 🚧 |
| `tasks/latent_rollout.py` | Rollout task | Native-geometry error vs horizon with static baseline; curves to CSV | ✅ |
| `tasks/hierarchy_reconstruction.py`, `long_horizon_consistency.py`, `compositional_generalization.py` | Remaining tasks | Hypotheses and falsification criteria documented; `run()` pending phase 2 | 🚧 |
| `training/train_predictor.py` | Hydra entry point | Head-only optimisation; frozen assert before first step and every save; geometry + curvature in every output | ✅ |
| `training/riemannian_optim.py` | Optimiser | Auto-selects `RiemannianAdam` when any `ManifoldParameter` exists | ✅ |
| `reporting/` | Tables and figures | Pure function of `outputs/`; deterministic | ✅ |

---

## ✅ Verification

```bash
uv run pytest -v
```

```text
============================= test session starts ==============================

tests/geometry/test_euclidean.py::test_is_a_manifold_with_zero_curvature PASSED [  1%]
tests/geometry/test_euclidean.py::test_primitives_reduce_to_vector_arithmetic[f32] PASSED [  2%]
tests/geometry/test_euclidean.py::test_primitives_reduce_to_vector_arithmetic[f64] PASSED [  3%]
tests/geometry/test_euclidean.py::test_geodesic_and_pairwise PASSED      [  5%]
tests/geometry/test_euclidean.py::test_to_geoopt_roundtrip PASSED        [  6%]
tests/geometry/test_lorentz.py::test_points_lie_on_hyperboloid[f32--0.5] PASSED [  7%]
tests/geometry/test_lorentz.py::test_points_lie_on_hyperboloid[f32--1.0] PASSED [  8%]
tests/geometry/test_lorentz.py::test_points_lie_on_hyperboloid[f32--2.0] PASSED [ 10%]
tests/geometry/test_lorentz.py::test_points_lie_on_hyperboloid[f64--0.5] PASSED [ 11%]
tests/geometry/test_lorentz.py::test_points_lie_on_hyperboloid[f64--1.0] PASSED [ 12%]
tests/geometry/test_lorentz.py::test_points_lie_on_hyperboloid[f64--2.0] PASSED [ 13%]
tests/geometry/test_lorentz.py::test_exp_log_inverse[f32--0.5] PASSED    [ 15%]
tests/geometry/test_lorentz.py::test_exp_log_inverse[f32--1.0] PASSED    [ 16%]
tests/geometry/test_lorentz.py::test_exp_log_inverse[f32--2.0] PASSED    [ 17%]
tests/geometry/test_lorentz.py::test_exp_log_inverse[f64--0.5] PASSED    [ 18%]
tests/geometry/test_lorentz.py::test_exp_log_inverse[f64--1.0] PASSED    [ 20%]
tests/geometry/test_lorentz.py::test_exp_log_inverse[f64--2.0] PASSED    [ 21%]
tests/geometry/test_lorentz.py::test_logmap_is_tangent[-0.5] PASSED      [ 22%]
tests/geometry/test_lorentz.py::test_logmap_is_tangent[-1.0] PASSED      [ 24%]
tests/geometry/test_lorentz.py::test_logmap_is_tangent[-2.0] PASSED      [ 25%]
tests/geometry/test_lorentz.py::test_distance_symmetric_and_triangle[-0.5] PASSED [ 26%]
tests/geometry/test_lorentz.py::test_distance_symmetric_and_triangle[-1.0] PASSED [ 27%]
tests/geometry/test_lorentz.py::test_distance_symmetric_and_triangle[-2.0] PASSED [ 29%]
tests/geometry/test_lorentz.py::test_dist_equals_tangent_norm_of_logmap PASSED [ 30%]
tests/geometry/test_lorentz.py::test_ptransp_is_isometric_and_tangent[-0.5] PASSED [ 31%]
tests/geometry/test_lorentz.py::test_ptransp_is_isometric_and_tangent[-1.0] PASSED [ 32%]
tests/geometry/test_lorentz.py::test_ptransp_is_isometric_and_tangent[-2.0] PASSED [ 34%]
tests/geometry/test_lorentz.py::test_isometric_to_poincare_ball[-0.5] PASSED [ 35%]
tests/geometry/test_lorentz.py::test_isometric_to_poincare_ball[-1.0] PASSED [ 36%]
tests/geometry/test_lorentz.py::test_isometric_to_poincare_ball[-2.0] PASSED [ 37%]
tests/geometry/test_lorentz.py::test_gradients_match_finite_differences[-0.5] PASSED [ 39%]
tests/geometry/test_lorentz.py::test_gradients_match_finite_differences[-1.0] PASSED [ 40%]
tests/geometry/test_lorentz.py::test_gradients_match_finite_differences[-2.0] PASSED [ 41%]
tests/geometry/test_lorentz.py::test_float32_stable_far_from_origin PASSED [ 43%]
tests/geometry/test_lorentz.py::test_tangent_lift_roundtrip_and_offsets PASSED [ 44%]
tests/geometry/test_poincare.py::test_exp_log_inverse[f32--0.5] PASSED   [ 45%]
tests/geometry/test_poincare.py::test_exp_log_inverse[f32--1.0] PASSED   [ 46%]
tests/geometry/test_poincare.py::test_exp_log_inverse[f32--2.0] PASSED   [ 48%]
tests/geometry/test_poincare.py::test_exp_log_inverse[f64--0.5] PASSED   [ 49%]
tests/geometry/test_poincare.py::test_exp_log_inverse[f64--1.0] PASSED   [ 50%]
tests/geometry/test_poincare.py::test_exp_log_inverse[f64--2.0] PASSED   [ 51%]
tests/geometry/test_poincare.py::test_origin_closed_forms_match_general[f32--0.5] PASSED [ 53%]
tests/geometry/test_poincare.py::test_origin_closed_forms_match_general[f32--1.0] PASSED [ 54%]
tests/geometry/test_poincare.py::test_origin_closed_forms_match_general[f32--2.0] PASSED [ 55%]
tests/geometry/test_poincare.py::test_origin_closed_forms_match_general[f64--0.5] PASSED [ 56%]
tests/geometry/test_poincare.py::test_origin_closed_forms_match_general[f64--1.0] PASSED [ 58%]
tests/geometry/test_poincare.py::test_origin_closed_forms_match_general[f64--2.0] PASSED [ 59%]
tests/geometry/test_poincare.py::test_distance_symmetric_and_zero_on_diagonal[f32--0.5] PASSED [ 60%]
tests/geometry/test_poincare.py::test_distance_symmetric_and_zero_on_diagonal[f32--1.0] PASSED [ 62%]
tests/geometry/test_poincare.py::test_distance_symmetric_and_zero_on_diagonal[f32--2.0] PASSED [ 63%]
tests/geometry/test_poincare.py::test_distance_symmetric_and_zero_on_diagonal[f64--0.5] PASSED [ 64%]
tests/geometry/test_poincare.py::test_distance_symmetric_and_zero_on_diagonal[f64--1.0] PASSED [ 65%]
tests/geometry/test_poincare.py::test_distance_symmetric_and_zero_on_diagonal[f64--2.0] PASSED [ 67%]
tests/geometry/test_poincare.py::test_triangle_inequality[-0.5] PASSED   [ 68%]
tests/geometry/test_poincare.py::test_triangle_inequality[-1.0] PASSED   [ 69%]
tests/geometry/test_poincare.py::test_triangle_inequality[-2.0] PASSED   [ 70%]
tests/geometry/test_poincare.py::test_distance_matches_closed_form_from_origin PASSED [ 72%]
tests/geometry/test_poincare.py::test_mobius_add_identity_and_inverse PASSED [ 73%]
tests/geometry/test_poincare.py::test_ptransp_preserves_norm_in_metric PASSED [ 74%]
tests/geometry/test_poincare.py::test_gradients_match_finite_differences[-0.5] PASSED [ 75%]
tests/geometry/test_poincare.py::test_gradients_match_finite_differences[-1.0] PASSED [ 77%]
tests/geometry/test_poincare.py::test_gradients_match_finite_differences[-2.0] PASSED [ 78%]
tests/geometry/test_poincare.py::test_proj_clips_to_boundary[f32] PASSED [ 79%]
tests/geometry/test_poincare.py::test_proj_clips_to_boundary[f64] PASSED [ 81%]
tests/geometry/test_poincare.py::test_float32_stable_near_boundary_vs_float64 PASSED [ 82%]
tests/geometry/test_poincare.py::test_build_manifold_from_config PASSED  [ 83%]
tests/metrics/test_distortion.py::test_perfect_embedding_has_zero_distortion_and_unit_map PASSED [ 84%]
tests/metrics/test_distortion.py::test_random_embedding_is_worse_than_structured PASSED [ 86%]
tests/metrics/test_distortion.py::test_shape_validation PASSED           [ 87%]
tests/metrics/test_gromov_hyperbolicity.py::test_tree_metric_has_zero_delta PASSED [ 88%]
tests/metrics/test_gromov_hyperbolicity.py::test_grid_has_positive_delta PASSED [ 89%]
tests/metrics/test_gromov_hyperbolicity.py::test_delta_is_scale_covariant PASSED [ 91%]
tests/metrics/test_gromov_hyperbolicity.py::test_subsampled_estimator_on_point_clouds PASSED [ 92%]
tests/test_smoke_experiment.py::test_smoke_config_is_tiny_and_cpu PASSED [ 93%]
tests/test_smoke_experiment.py::test_smoke_runs_end_to_end[poincare] PASSED [ 94%]
tests/test_smoke_experiment.py::test_smoke_runs_end_to_end[lorentz] PASSED [ 96%]
tests/test_smoke_experiment.py::test_smoke_runs_end_to_end[euclidean] PASSED [ 97%]
tests/test_smoke_experiment.py::test_encoder_stays_frozen PASSED         [ 98%]
tests/test_smoke_experiment.py::test_hyperbolic_head_on_euclidean_manifold_matches_euclidean_head PASSED [100%]

============================== 79 passed in 3.84s ==============================
```

What the tests cover:

- [x] Poincaré ball: exp/log inverse identity in float32 and float64 at three curvatures; distance symmetry and zero diagonal; triangle inequality; closed-form distance from the origin; Möbius identity and inverse; parallel transport is an isometry; `gradcheck` of `dist`, `expmap`, `logmap` against finite differences; boundary clipping; float32 vs float64 behaviour near the boundary (agreement outside the clamp margin, bounded and monotone inside it).
- [x] Lorentz: points stay on the hyperboloid; exp/log inverse; log map is tangent; distance symmetry and triangle inequality; distance equals the tangent norm of the log map; transport is isometric and tangent; isometry to the Poincaré ball to 1e-9; `gradcheck`; float32 stability far from the origin with the documented cancellation bound; tangent-space lift round trip.
- [x] Euclidean: same `Manifold` contract reduces to vector arithmetic; geodesics and pairwise distances; geoopt equivalent.
- [x] Gromov δ: exactly 0 on the embodiment > task > primitive tree metric from any base point; positive on L1 and L2 grids; scale covariant; subsampled estimator on point clouds, including a path (δ = 0).
- [x] Distortion and mAP: 0 and 1.0 on a perfectly embedded path graph; scale fitting matters; shape validation; structured beats random on a Poincaré tree embedding.
- [x] Smoke experiment: `configs/experiments/smoke.yaml` runs end to end on CPU in Poincaré, Lorentz and Euclidean geometry; loss decreases; error grows with horizon; `metrics.json`, curves CSV and resolved config are written with the geometry recorded; the encoder cannot be un-frozen; `HyperbolicHead` on the flat manifold equals `EuclideanHead` exactly.

CI runs `ruff check`, `ruff format --check`, `pytest tests/geometry tests/metrics` and `scripts/run_smoke.sh`, all on CPU.

---

## ⚖️ Weights and Licensing

No model weights are included in this repository. Every checkpoint is downloaded by the user from its upstream source under the upstream terms, and none is redistributed:

- **V-JEPA 2** (`facebook/vjepa2-vitl-fpc64-256`): MIT-licensed code; weights per the model card. Used frozen.
- **DINOv2** (`facebook/dinov2-base`): Apache 2.0. Used frozen.
- **NVIDIA Cosmos 3 Nano** (`nvidia/Cosmos-3-Nano`): OpenMDW 1.1, gated. Used for inference only; never modified.

Trained predictor heads written to `checkpoints/predictors/` contain only the head's own parameters and fall under this repository's Apache 2.0 licence. Full list: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

---

## 🤝 Contributing

Ground rules, setup and the review checklist are in [CONTRIBUTING.md](CONTRIBUTING.md). The short version: geometry goes through `Manifold`, encoders stay frozen, curvature is swept, Cosmos 3 is inference-only, no new dependencies without discussion, and the README is updated in the same PR as the code.

---

## 👤 Author

**A Taylor** · 2026

---

## 📜 License

Apache 2.0. See [LICENSE](LICENSE).

Copyright 2026 A Taylor.

---

## 📬 Contact

[![Contact A Taylor - Get In Touch](https://img.shields.io/badge/Contact%20A%20Taylor-Get%20In%20Touch-brightgreen)](https://ataylor.getform.com/5w8wz)
