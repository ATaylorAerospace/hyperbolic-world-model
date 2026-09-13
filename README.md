[![Hyperbolic Latent Evaluation](docs/header.png)](https://github.com/ATaylorAerospace/hyperbolic-world-model)

# 🌀 Hyperbolic Latent Evaluation for World Models 📐

[![CI](https://github.com/ATaylorAerospace/hyperbolic-world-model/actions/workflows/ci.yml/badge.svg)](https://github.com/ATaylorAerospace/hyperbolic-world-model/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![geoopt](https://img.shields.io/badge/geoopt-0.5%2B-6f42c1)](https://github.com/geoopt/geoopt)
[![Tests](https://img.shields.io/badge/tests-344%20passing-brightgreen)](#-verification)
[![Curvature: Swept](https://img.shields.io/badge/Curvature-Swept-orange)](docs/methodology.md)

**A research harness that retrains the action-conditioned predictor head of a frozen video world model (V-JEPA 2-AC, with DINO-WM as a lightweight second subject) in Euclidean, Poincaré and Lorentz latent spaces, and measures whether negative curvature buys better rollouts, hierarchy recovery and dimension efficiency, with every distance computed in the geometry of the model that produced it and curvature always swept.**

**Author: A Taylor**

> 🚧 **Status:** Geometry primitives stable · V-JEPA 2-AC head training · 344/344 tests passing.

---

## 🤔 The Problem

Embodied data is tree-shaped in at least three ways. **Embodiment hierarchies** (robot → arm → gripper → primitive skill) branch at every level. **Task decompositions** turn one instruction into sub-goals and sub-goals into primitives. **Branching futures** mean the set of states reachable from one frame grows roughly exponentially with the horizon. A tree with branching factor *b* has *b^h* leaves at depth *h*; the space that stores it needs volume that grows the same way.

Euclidean space does not have that volume. Its balls grow polynomially with radius, so any low-distortion embedding of a tree needs dimension that grows with the tree (Bourgain-type lower bounds). A world model with a Euclidean latent therefore has two bad options: crush distinct futures together, or spend latent dimensions to keep them apart. Either shows up as long-horizon rollout error, poor hierarchy recovery, and a steep dimension-versus-performance curve.

Hyperbolic space has exponential volume growth and embeds any tree with arbitrarily low distortion in two dimensions (Sarkar, 2011). The question this repository asks is empirical: **does a hyperbolic latent help a frozen video encoder's predictor head, and at which curvature and dimension?** The counter-hypothesis is equally live: the frozen encoder may already linearise the relevant structure and curvature may only add optimisation difficulty. The harness is built so either answer is visible and neither is baked in.

---

## 💡 The Solution

- 🧊 **Freeze** the upstream encoder (the V-JEPA 2-AC ViT-g from Meta's `vjepa2_ac_vit_giant` checkpoint, or DINOv2) so every geometry sees identical features. `FrozenEncoder.assert_frozen` guards every optimiser step and checkpoint; Meta's own predictor from the same checkpoint stays frozen as the reference Euclidean baseline.
- 📐 **Project** encoder latents onto the chosen manifold with a seeded, frozen projection followed by `expmap0`, in geodesic units so a step of norm *r* is a geodesic of length *r* in every geometry, so no geometry can win by collapsing its embedding.
- 🔮 **Predict** the next latent with an action-conditioned head that fuses the state coordinates with an action embedding, proposes a bounded tangent update, transports it to the state and follows the geodesic; loss is squared geodesic distance. The Euclidean head is the same network with `+` in place of `expmap`.
- 🎬 **Generate** controlled branching trajectories with NVIDIA Cosmos (start frame + action sequence → video) through its action-conditioned inference API, inference only, seeded per branch and recorded in a manifest; Cosmos is a data generator and probe target, never a subject.
- 🔬 **Probe** the delta-hyperbolicity of V-JEPA 2, DINOv2 and Cosmos 3 tokenizer latents, independent of any head we train.
- 🎛️ **Sweep** curvature × latent dimension × seed for both hyperbolic models via Hydra multirun. Curvature is never a fixed constant.
- 📏 **Measure** rollout error, distortion, mAP and dimension efficiency in each model's *native* geometry via a single `Manifold` interface.
- 📊 **Report** every table and figure by regenerating them from `outputs/` with one script, never by hand.

---

## 🏛️ Architecture

```text
 ┌──────────────────────────┐        ┌──────────────────────────────┐        ┌──────────────────────────────┐
 │  NVIDIA Cosmos 3 Nano    │        │  Frozen encoder              │        │  Metrics (native geometry)   │
 │  (inference only)        │        │  V-JEPA 2-AC ViT-g (primary) │        │                              │
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
| NVIDIA Cosmos 3 | Action-conditioned trajectory generator (start frame + actions → video) and tokenizer-latent probe target for δ-hyperbolicity; never fine-tuned, generation quality never reported | Downloaded by the user through NVIDIA's `cosmos_predict2` package (model key selects the action-conditioned checkpoint) | OpenMDW 1.1 | No, frozen |
| V-JEPA 2-AC | Primary subject: the frozen ViT-g/16 encoder from the action-conditioned checkpoint feeds every head; our predictor heads are trained from scratch in each geometry; Meta's bundled predictor is kept frozen as the reference Euclidean baseline | `vjepa2_ac_vit_giant` via `torch.hub` (`vjepa2-ac-vitg.pt`), downloaded by the user | MIT (code) | Predictor head only |
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
| Long-horizon consistency | Latent divergence between branching rollouts tracks video divergence with higher correlation and saturates later in hyperbolic space | Cosmos-generated branches (`Cosmos3TrajectoryDataset.branch_pairs`) | Correlation is not higher or saturation not later for the best curvature |
| Compositional generalisation | The unseen-minus-seen error gap on held-out (embodiment, primitive) pairs is smaller in hyperbolic space | DROID or Cosmos-generated data with `holdout_combinations` (`split_by_combination`) | Gap is not smaller, or is smaller only because seen error got worse |

---

## 🛡️ Experimental Guarantees

> **Rule 1: distances are computed in the model's native geometry.** A Euclidean head is scored with Euclidean distance, a Poincaré head with the Poincaré geodesic at its trained curvature, a Lorentz head with the hyperboloid geodesic. No metric maps one model's latents into another's space. Every metric takes a `Manifold` and calls `manifold.dist`; there is no Euclidean fallback path. Every `TaskResult` records `geometry` and `curvature` next to its numbers.
> *Enforced by* `tests/test_smoke_experiment.py::test_smoke_runs_end_to_end[poincare|lorentz|euclidean]` (asserts the recorded geometry matches the manifold that produced the metrics) and `tests/geometry/test_lorentz.py::test_isometry_to_poincare_ball` (the two hyperbolic geometries agree with each other to 1e-9, so "native" is not "arbitrary").
>
> **Rule 2: curvature is swept, never fixed.** Hyperbolic results come only from `configs/experiments/poincare_sweep.yaml` and `lorentz_sweep.yaml`, which cross `curvature ∈ {-0.1, -0.25, -0.5, -1.0, -2.0, -4.0}` with latent dimension and seed. Tables show the whole curve. The `-1.0` default in `configs/geometry/*.yaml` exists only so single debug runs have a value.
> *Enforced by* `tests/test_smoke_experiment.py::test_smoke_config_is_tiny_and_cpu` (curvature is read from config, never hard-coded) and `tests/geometry/test_base.py::test_registry_covers_every_case_and_build_manifold_accepts_both_keys` (the manifold is built from the config value); the sweep grids live in version-controlled YAML.
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
│   │   ├── encoders/                  # FrozenEncoder contract + EncoderOutput; vjepa2.py (torch.hub V-JEPA 2-AC + Meta's reference predictor), dino.py, synthetic.py (CI)
│   │   ├── predictors/                # shared architecture (base.py), EuclideanHead, HyperbolicHead (geodesic loss)
│   │   └── registry.py                # build_manifold / build_encoder / build_predictor / build_model from config
│   ├── data/
│   │   ├── droid.py                   # action normalisation + chunking (done); DROID loader (TODO phase 2)
│   │   ├── synthetic.py               # in-memory linear-dynamics trajectories for the smoke run
│   │   ├── hierarchies.py             # embodiment > task > primitive tree, hop-count metric, adjacency
│   │   └── cosmos3/                   # generate.py (batch rollouts via Cosmos, adapter over NVIDIA's API), extract_latents.py (tokenizer latents), dataset.py (manifest loader)
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
│   ├── download_weights.sh            # V-JEPA 2-AC via torch.hub, DINOv2 and Cosmos 3 Nano via huggingface_hub; reads HF_TOKEN
│   ├── generate_cosmos3_trajectories.sh   # generate rollouts into $DATA_ROOT/cosmos3_generated, then extract tokenizer latents
│   ├── run_smoke.sh                   # CPU end-to-end run used by CI
│   ├── run_baseline.sh                # Euclidean baseline on DROID
│   ├── run_curvature_sweep.sh         # Poincaré and/or Lorentz multirun sweeps
│   └── make_report.sh                 # wraps reporting/make_report.py
├── tests/
│   ├── conftest.py                    # seeds, float32/float64 fixture
│   ├── geometry/                      # test_base.py (contract over all geometries), test_poincare.py, test_lorentz.py, test_euclidean.py, test_utils.py, test_public_api.py, helpers.py
│   ├── models/                        # test_predictor_heads.py (shapes, identical architecture, isometric twins), test_vjepa2.py (hub loader with fakes)
│   ├── data/                          # test_cosmos3.py (generation pipeline, adapter chunk loop, latents, dataset; all with fakes)
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
   uv sync --extra vjepa2  # the extra adds timm, which torch.hub needs to import the V-JEPA 2-AC code
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
| `geometry/base.py` | `Manifold` interface | Single abstraction for heads, metrics, tasks; curvature `c` passed at construction; the full contract runs against every registered geometry in `tests/geometry/test_base.py` | ✅ |
| `geometry/euclidean.py` | Flat baseline | Wraps `geoopt.Euclidean`; identical code path to curved geometries; rejects non-zero curvature | ✅ |
| `geometry/poincare.py` | Poincaré ball | Wraps `geoopt.PoincareBall`; exp/log inverse, metric axioms, gradcheck; float32 agrees with float64 at 0.999 R except ~1% on boundary-to-boundary Möbius distances; float32 round trip holds inside the 4e-3 clip margin | ✅ |
| `geometry/lorentz.py` | Hyperboloid | Wraps `geoopt.Lorentz`; own transport (finite at coincident points) and own isometries (geoopt's assume the unit ball); isometric to the ball to 1e-9 at every curvature; float32 self-distance noise ≤ 2e-3·x₀ | ✅ |
| `geometry/utils.py` | Stable primitives | geoopt's `artanh`/`arcosh` re-exported; per-dtype epsilons; norm clipping | ✅ |
| `tests/geometry/test_public_api.py` | API coverage guard | Fails if any public geometry function, method or constant is not referenced by a geometry test | ✅ |
| `metrics/geodesic_error.py` | Rollout error | Takes a `Manifold`; no Euclidean fallback; static baseline for normalisation | ✅ |
| `metrics/gromov_hyperbolicity.py` | δ-hyperbolicity | δ = 0 on tree metrics, > 0 on grids; scale covariant; chunked max-min product | ✅ |
| `metrics/distortion.py` | Distortion, mAP | Scale-fitted; 0 distortion / 1.0 mAP on a perfect embedding | ✅ |
| `metrics/dimension_efficiency.py` | Dimension curves | Tidy tables, log₂ AUC, dimension-to-reach | ✅ |
| `models/encoders/vjepa2.py` | V-JEPA 2-AC via torch.hub | Loads `vjepa2_ac_vit_giant` code from hub and Meta's `vjepa2-ac-vitg.pt` from the official URL; encodes each frame as Meta's inference code does (2-frame tubelet, layer-normed tokens); returns patch and pooled embeddings; `assert_frozen` on encoder and reference predictor; refuses partially loaded checkpoints; accepts a local clone path for air-gapped machines; validated against Meta's upstream code at the pinned commit (random init: 1012M-param ViT-g, 256 tokens per 256 px frame, 305M-param reference predictor, all frozen); Meta's weights not yet run (`uv sync --extra vjepa2` provides `timm` for hub) | 🚧 |
| `models/encoders/` | Encoder contract | `EncoderOutput(patch, pooled)`; `train()` is a no-op; synthetic encoder for CI; DINOv2 patch + CLS/mean pooling | ✅ |
| `models/predictors/` | Heads | Identical architecture by construction (same seed gives identical weights; `architecture_signature` equal); action embedding concatenated to state coordinates; geodesic units in every geometry (Poincaré and Lorentz heads are isometric twins); `max_step` and `max_radius` guards shared; `HyperbolicHead(Euclidean)` equals `EuclideanHead` exactly | ✅ |
| `models/registry.py` | Config → objects | Refuses `frozen: false`; checks `embed_dim` against the loaded encoder | ✅ |
| `data/hierarchies.py` | Metadata → tree | Hop-count metric is 0-hyperbolic (used as the δ reference) | ✅ |
| `data/synthetic.py` | CI trajectories | Deterministic, learnable linear dynamics, two-level hierarchy in metadata | ✅ |
| `data/droid.py` | DROID | Action stats and chunking done; loader raises `NotImplementedError` with plan | 🚧 |
| `data/cosmos3/generate.py` | Cosmos rollouts | Standalone batch script: start frames + action JSON → `frames.npz`, `meta.json`, optional mp4, `manifest.jsonl`; per-branch deterministic seeds; idempotent; `--dry-run` plan; adapter replays NVIDIA's chunked `generate_vid2world` loop (first frame real, zero padding, per-chunk seeds) and stitches so frame t+1 is the result of action t; not run on a real model | 🚧 |
| `data/cosmos3/extract_latents.py` | Tokenizer latents | Encoder-only tokenizer adapter (`(1,3,T,H,W)` in [-1, 1]); mean or flatten pooling to `(T', D)`; float16 npz + meta; manifest updated; idempotent | 🚧 |
| `data/cosmos3/dataset.py` | Generated-trajectory loader | Manifest-driven windows in the shared batch format; bilinear resize; `branches`/`branch_pairs` for long-horizon consistency; `split_by_combination` for compositional generalisation (refuses splits that remove an embodiment or primitive); `latents(idx)`; deterministic train/val/test by prompt hash | ✅ |
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

tests/data/test_cosmos3.py::test_load_action_spec_validates PASSED       [  0%]
tests/data/test_cosmos3.py::test_branch_seed_is_deterministic_and_distinct PASSED [  0%]
tests/data/test_cosmos3.py::test_generate_rollouts_writes_frames_meta_and_manifest PASSED [  0%]
tests/data/test_cosmos3.py::test_write_rollout_rejects_bad_frames PASSED [  1%]
tests/data/test_cosmos3.py::test_cli_dry_run_and_injected_generator PASSED [  1%]
tests/data/test_cosmos3.py::test_default_root_follows_data_root PASSED   [  1%]
tests/data/test_cosmos3.py::test_cosmos_generator_reproduces_upstream_chunk_loop PASSED [  2%]
tests/data/test_cosmos3.py::test_cosmos_generator_without_package_gives_actionable_error PASSED [  2%]
tests/data/test_cosmos3.py::test_pool_latent_and_extract PASSED          [  2%]
tests/data/test_cosmos3.py::test_tokenizer_backend_encodes_through_client_tokenizer PASSED [  2%]
tests/data/test_cosmos3.py::test_dataset_windows_shapes_and_meta PASSED  [  3%]
tests/data/test_cosmos3.py::test_dataset_branches_and_pairs_share_start_frames PASSED [  3%]
tests/data/test_cosmos3.py::test_dataset_compositional_split_and_hierarchy PASSED [  3%]
tests/data/test_cosmos3.py::test_dataset_splits_and_config PASSED        [  4%]
tests/geometry/test_base.py::test_registry_covers_every_case_and_build_manifold_accepts_both_keys PASSED [  4%]
tests/geometry/test_base.py::test_lambda0_is_the_origin_metric_scale[euclidean-f32] PASSED [  4%]
tests/geometry/test_base.py::test_lambda0_is_the_origin_metric_scale[euclidean-f64] PASSED [  4%]
tests/geometry/test_base.py::test_lambda0_is_the_origin_metric_scale[poincare(c=-1)-f32] PASSED [  5%]
tests/geometry/test_base.py::test_lambda0_is_the_origin_metric_scale[poincare(c=-1)-f64] PASSED [  5%]
tests/geometry/test_base.py::test_lambda0_is_the_origin_metric_scale[poincare(c=-0.5)-f32] PASSED [  5%]
tests/geometry/test_base.py::test_lambda0_is_the_origin_metric_scale[poincare(c=-0.5)-f64] PASSED [  6%]
tests/geometry/test_base.py::test_lambda0_is_the_origin_metric_scale[poincare(c=-2)-f32] PASSED [  6%]
tests/geometry/test_base.py::test_lambda0_is_the_origin_metric_scale[poincare(c=-2)-f64] PASSED [  6%]
tests/geometry/test_base.py::test_lambda0_is_the_origin_metric_scale[lorentz(c=-1)-f32] PASSED [  6%]
tests/geometry/test_base.py::test_lambda0_is_the_origin_metric_scale[lorentz(c=-1)-f64] PASSED [  7%]
tests/geometry/test_base.py::test_lambda0_is_the_origin_metric_scale[lorentz(c=-0.5)-f32] PASSED [  7%]
tests/geometry/test_base.py::test_lambda0_is_the_origin_metric_scale[lorentz(c=-0.5)-f64] PASSED [  7%]
tests/geometry/test_base.py::test_lambda0_is_the_origin_metric_scale[lorentz(c=-2)-f32] PASSED [  8%]
tests/geometry/test_base.py::test_lambda0_is_the_origin_metric_scale[lorentz(c=-2)-f64] PASSED [  8%]
tests/geometry/test_base.py::test_curvature_name_offset_and_repr[euclidean] PASSED [  8%]
tests/geometry/test_base.py::test_curvature_name_offset_and_repr[poincare(c=-1)] PASSED [  9%]
tests/geometry/test_base.py::test_curvature_name_offset_and_repr[poincare(c=-0.5)] PASSED [  9%]
tests/geometry/test_base.py::test_curvature_name_offset_and_repr[poincare(c=-2)] PASSED [  9%]
tests/geometry/test_base.py::test_curvature_name_offset_and_repr[lorentz(c=-1)] PASSED [  9%]
tests/geometry/test_base.py::test_curvature_name_offset_and_repr[lorentz(c=-0.5)] PASSED [ 10%]
tests/geometry/test_base.py::test_curvature_name_offset_and_repr[lorentz(c=-2)] PASSED [ 10%]
tests/geometry/test_base.py::test_expmap_logmap_are_inverse[euclidean-f32] PASSED [ 10%]
tests/geometry/test_base.py::test_expmap_logmap_are_inverse[euclidean-f64] PASSED [ 11%]
tests/geometry/test_base.py::test_expmap_logmap_are_inverse[poincare(c=-1)-f32] PASSED [ 11%]
tests/geometry/test_base.py::test_expmap_logmap_are_inverse[poincare(c=-1)-f64] PASSED [ 11%]
tests/geometry/test_base.py::test_expmap_logmap_are_inverse[poincare(c=-0.5)-f32] PASSED [ 11%]
tests/geometry/test_base.py::test_expmap_logmap_are_inverse[poincare(c=-0.5)-f64] PASSED [ 12%]
tests/geometry/test_base.py::test_expmap_logmap_are_inverse[poincare(c=-2)-f32] PASSED [ 12%]
tests/geometry/test_base.py::test_expmap_logmap_are_inverse[poincare(c=-2)-f64] PASSED [ 12%]
tests/geometry/test_base.py::test_expmap_logmap_are_inverse[lorentz(c=-1)-f32] PASSED [ 13%]
tests/geometry/test_base.py::test_expmap_logmap_are_inverse[lorentz(c=-1)-f64] PASSED [ 13%]
tests/geometry/test_base.py::test_expmap_logmap_are_inverse[lorentz(c=-0.5)-f32] PASSED [ 13%]
tests/geometry/test_base.py::test_expmap_logmap_are_inverse[lorentz(c=-0.5)-f64] PASSED [ 13%]
tests/geometry/test_base.py::test_expmap_logmap_are_inverse[lorentz(c=-2)-f32] PASSED [ 14%]
tests/geometry/test_base.py::test_expmap_logmap_are_inverse[lorentz(c=-2)-f64] PASSED [ 14%]
tests/geometry/test_base.py::test_distance_is_a_metric_on_random_points[euclidean-f32] PASSED [ 14%]
tests/geometry/test_base.py::test_distance_is_a_metric_on_random_points[euclidean-f64] PASSED [ 15%]
tests/geometry/test_base.py::test_distance_is_a_metric_on_random_points[poincare(c=-1)-f32] PASSED [ 15%]
tests/geometry/test_base.py::test_distance_is_a_metric_on_random_points[poincare(c=-1)-f64] PASSED [ 15%]
tests/geometry/test_base.py::test_distance_is_a_metric_on_random_points[poincare(c=-0.5)-f32] PASSED [ 15%]
tests/geometry/test_base.py::test_distance_is_a_metric_on_random_points[poincare(c=-0.5)-f64] PASSED [ 16%]
tests/geometry/test_base.py::test_distance_is_a_metric_on_random_points[poincare(c=-2)-f32] PASSED [ 16%]
tests/geometry/test_base.py::test_distance_is_a_metric_on_random_points[poincare(c=-2)-f64] PASSED [ 16%]
tests/geometry/test_base.py::test_distance_is_a_metric_on_random_points[lorentz(c=-1)-f32] PASSED [ 17%]
tests/geometry/test_base.py::test_distance_is_a_metric_on_random_points[lorentz(c=-1)-f64] PASSED [ 17%]
tests/geometry/test_base.py::test_distance_is_a_metric_on_random_points[lorentz(c=-0.5)-f32] PASSED [ 17%]
tests/geometry/test_base.py::test_distance_is_a_metric_on_random_points[lorentz(c=-0.5)-f64] PASSED [ 18%]
tests/geometry/test_base.py::test_distance_is_a_metric_on_random_points[lorentz(c=-2)-f32] PASSED [ 18%]
tests/geometry/test_base.py::test_distance_is_a_metric_on_random_points[lorentz(c=-2)-f64] PASSED [ 18%]
tests/geometry/test_base.py::test_sqdist_dist0_and_pairwise_are_consistent_with_dist[euclidean-f32] PASSED [ 18%]
tests/geometry/test_base.py::test_sqdist_dist0_and_pairwise_are_consistent_with_dist[euclidean-f64] PASSED [ 19%]
tests/geometry/test_base.py::test_sqdist_dist0_and_pairwise_are_consistent_with_dist[poincare(c=-1)-f32] PASSED [ 19%]
tests/geometry/test_base.py::test_sqdist_dist0_and_pairwise_are_consistent_with_dist[poincare(c=-1)-f64] PASSED [ 19%]
tests/geometry/test_base.py::test_sqdist_dist0_and_pairwise_are_consistent_with_dist[poincare(c=-0.5)-f32] PASSED [ 20%]
tests/geometry/test_base.py::test_sqdist_dist0_and_pairwise_are_consistent_with_dist[poincare(c=-0.5)-f64] PASSED [ 20%]
tests/geometry/test_base.py::test_sqdist_dist0_and_pairwise_are_consistent_with_dist[poincare(c=-2)-f32] PASSED [ 20%]
tests/geometry/test_base.py::test_sqdist_dist0_and_pairwise_are_consistent_with_dist[poincare(c=-2)-f64] PASSED [ 20%]
tests/geometry/test_base.py::test_sqdist_dist0_and_pairwise_are_consistent_with_dist[lorentz(c=-1)-f32] PASSED [ 21%]
tests/geometry/test_base.py::test_sqdist_dist0_and_pairwise_are_consistent_with_dist[lorentz(c=-1)-f64] PASSED [ 21%]
tests/geometry/test_base.py::test_sqdist_dist0_and_pairwise_are_consistent_with_dist[lorentz(c=-0.5)-f32] PASSED [ 21%]
tests/geometry/test_base.py::test_sqdist_dist0_and_pairwise_are_consistent_with_dist[lorentz(c=-0.5)-f64] PASSED [ 22%]
tests/geometry/test_base.py::test_sqdist_dist0_and_pairwise_are_consistent_with_dist[lorentz(c=-2)-f32] PASSED [ 22%]
tests/geometry/test_base.py::test_sqdist_dist0_and_pairwise_are_consistent_with_dist[lorentz(c=-2)-f64] PASSED [ 22%]
tests/geometry/test_base.py::test_origin_closed_forms_match_general_maps[euclidean-f32] PASSED [ 22%]
tests/geometry/test_base.py::test_origin_closed_forms_match_general_maps[euclidean-f64] PASSED [ 23%]
tests/geometry/test_base.py::test_origin_closed_forms_match_general_maps[poincare(c=-1)-f32] PASSED [ 23%]
tests/geometry/test_base.py::test_origin_closed_forms_match_general_maps[poincare(c=-1)-f64] PASSED [ 23%]
tests/geometry/test_base.py::test_origin_closed_forms_match_general_maps[poincare(c=-0.5)-f32] PASSED [ 24%]
tests/geometry/test_base.py::test_origin_closed_forms_match_general_maps[poincare(c=-0.5)-f64] PASSED [ 24%]
tests/geometry/test_base.py::test_origin_closed_forms_match_general_maps[poincare(c=-2)-f32] PASSED [ 24%]
tests/geometry/test_base.py::test_origin_closed_forms_match_general_maps[poincare(c=-2)-f64] PASSED [ 25%]
tests/geometry/test_base.py::test_origin_closed_forms_match_general_maps[lorentz(c=-1)-f32] PASSED [ 25%]
tests/geometry/test_base.py::test_origin_closed_forms_match_general_maps[lorentz(c=-1)-f64] PASSED [ 25%]
tests/geometry/test_base.py::test_origin_closed_forms_match_general_maps[lorentz(c=-0.5)-f32] PASSED [ 25%]
tests/geometry/test_base.py::test_origin_closed_forms_match_general_maps[lorentz(c=-0.5)-f64] PASSED [ 26%]
tests/geometry/test_base.py::test_origin_closed_forms_match_general_maps[lorentz(c=-2)-f32] PASSED [ 26%]
tests/geometry/test_base.py::test_origin_closed_forms_match_general_maps[lorentz(c=-2)-f64] PASSED [ 26%]
tests/geometry/test_base.py::test_geodesic_endpoints_and_midpoint[euclidean-f32] PASSED [ 27%]
tests/geometry/test_base.py::test_geodesic_endpoints_and_midpoint[euclidean-f64] PASSED [ 27%]
tests/geometry/test_base.py::test_geodesic_endpoints_and_midpoint[poincare(c=-1)-f32] PASSED [ 27%]
tests/geometry/test_base.py::test_geodesic_endpoints_and_midpoint[poincare(c=-1)-f64] PASSED [ 27%]
tests/geometry/test_base.py::test_geodesic_endpoints_and_midpoint[poincare(c=-0.5)-f32] PASSED [ 28%]
tests/geometry/test_base.py::test_geodesic_endpoints_and_midpoint[poincare(c=-0.5)-f64] PASSED [ 28%]
tests/geometry/test_base.py::test_geodesic_endpoints_and_midpoint[poincare(c=-2)-f32] PASSED [ 28%]
tests/geometry/test_base.py::test_geodesic_endpoints_and_midpoint[poincare(c=-2)-f64] PASSED [ 29%]
tests/geometry/test_base.py::test_geodesic_endpoints_and_midpoint[lorentz(c=-1)-f32] PASSED [ 29%]
tests/geometry/test_base.py::test_geodesic_endpoints_and_midpoint[lorentz(c=-1)-f64] PASSED [ 29%]
tests/geometry/test_base.py::test_geodesic_endpoints_and_midpoint[lorentz(c=-0.5)-f32] PASSED [ 29%]
tests/geometry/test_base.py::test_geodesic_endpoints_and_midpoint[lorentz(c=-0.5)-f64] PASSED [ 30%]
tests/geometry/test_base.py::test_geodesic_endpoints_and_midpoint[lorentz(c=-2)-f32] PASSED [ 30%]
tests/geometry/test_base.py::test_geodesic_endpoints_and_midpoint[lorentz(c=-2)-f64] PASSED [ 30%]
tests/geometry/test_base.py::test_proj_is_idempotent_and_returns_points_on_the_manifold[euclidean-f32] PASSED [ 31%]
tests/geometry/test_base.py::test_proj_is_idempotent_and_returns_points_on_the_manifold[euclidean-f64] PASSED [ 31%]
tests/geometry/test_base.py::test_proj_is_idempotent_and_returns_points_on_the_manifold[poincare(c=-1)-f32] PASSED [ 31%]
tests/geometry/test_base.py::test_proj_is_idempotent_and_returns_points_on_the_manifold[poincare(c=-1)-f64] PASSED [ 31%]
tests/geometry/test_base.py::test_proj_is_idempotent_and_returns_points_on_the_manifold[poincare(c=-0.5)-f32] PASSED [ 32%]
tests/geometry/test_base.py::test_proj_is_idempotent_and_returns_points_on_the_manifold[poincare(c=-0.5)-f64] PASSED [ 32%]
tests/geometry/test_base.py::test_proj_is_idempotent_and_returns_points_on_the_manifold[poincare(c=-2)-f32] PASSED [ 32%]
tests/geometry/test_base.py::test_proj_is_idempotent_and_returns_points_on_the_manifold[poincare(c=-2)-f64] PASSED [ 33%]
tests/geometry/test_base.py::test_proj_is_idempotent_and_returns_points_on_the_manifold[lorentz(c=-1)-f32] PASSED [ 33%]
tests/geometry/test_base.py::test_proj_is_idempotent_and_returns_points_on_the_manifold[lorentz(c=-1)-f64] PASSED [ 33%]
tests/geometry/test_base.py::test_proj_is_idempotent_and_returns_points_on_the_manifold[lorentz(c=-0.5)-f32] PASSED [ 34%]
tests/geometry/test_base.py::test_proj_is_idempotent_and_returns_points_on_the_manifold[lorentz(c=-0.5)-f64] PASSED [ 34%]
tests/geometry/test_base.py::test_proj_is_idempotent_and_returns_points_on_the_manifold[lorentz(c=-2)-f32] PASSED [ 34%]
tests/geometry/test_base.py::test_proj_is_idempotent_and_returns_points_on_the_manifold[lorentz(c=-2)-f64] PASSED [ 34%]
tests/geometry/test_base.py::test_proj_tan_is_idempotent_and_logmap_is_tangent[euclidean-f32] PASSED [ 35%]
tests/geometry/test_base.py::test_proj_tan_is_idempotent_and_logmap_is_tangent[euclidean-f64] PASSED [ 35%]
tests/geometry/test_base.py::test_proj_tan_is_idempotent_and_logmap_is_tangent[poincare(c=-1)-f32] PASSED [ 35%]
tests/geometry/test_base.py::test_proj_tan_is_idempotent_and_logmap_is_tangent[poincare(c=-1)-f64] PASSED [ 36%]
tests/geometry/test_base.py::test_proj_tan_is_idempotent_and_logmap_is_tangent[poincare(c=-0.5)-f32] PASSED [ 36%]
tests/geometry/test_base.py::test_proj_tan_is_idempotent_and_logmap_is_tangent[poincare(c=-0.5)-f64] PASSED [ 36%]
tests/geometry/test_base.py::test_proj_tan_is_idempotent_and_logmap_is_tangent[poincare(c=-2)-f32] PASSED [ 36%]
tests/geometry/test_base.py::test_proj_tan_is_idempotent_and_logmap_is_tangent[poincare(c=-2)-f64] PASSED [ 37%]
tests/geometry/test_base.py::test_proj_tan_is_idempotent_and_logmap_is_tangent[lorentz(c=-1)-f32] PASSED [ 37%]
tests/geometry/test_base.py::test_proj_tan_is_idempotent_and_logmap_is_tangent[lorentz(c=-1)-f64] PASSED [ 37%]
tests/geometry/test_base.py::test_proj_tan_is_idempotent_and_logmap_is_tangent[lorentz(c=-0.5)-f32] PASSED [ 38%]
tests/geometry/test_base.py::test_proj_tan_is_idempotent_and_logmap_is_tangent[lorentz(c=-0.5)-f64] PASSED [ 38%]
tests/geometry/test_base.py::test_proj_tan_is_idempotent_and_logmap_is_tangent[lorentz(c=-2)-f32] PASSED [ 38%]
tests/geometry/test_base.py::test_proj_tan_is_idempotent_and_logmap_is_tangent[lorentz(c=-2)-f64] PASSED [ 38%]
tests/geometry/test_base.py::test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic[euclidean-f32] PASSED [ 39%]
tests/geometry/test_base.py::test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic[euclidean-f64] PASSED [ 39%]
tests/geometry/test_base.py::test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic[poincare(c=-1)-f32] PASSED [ 39%]
tests/geometry/test_base.py::test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic[poincare(c=-1)-f64] PASSED [ 40%]
tests/geometry/test_base.py::test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic[poincare(c=-0.5)-f32] PASSED [ 40%]
tests/geometry/test_base.py::test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic[poincare(c=-0.5)-f64] PASSED [ 40%]
tests/geometry/test_base.py::test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic[poincare(c=-2)-f32] PASSED [ 40%]
tests/geometry/test_base.py::test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic[poincare(c=-2)-f64] PASSED [ 41%]
tests/geometry/test_base.py::test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic[lorentz(c=-1)-f32] PASSED [ 41%]
tests/geometry/test_base.py::test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic[lorentz(c=-1)-f64] PASSED [ 41%]
tests/geometry/test_base.py::test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic[lorentz(c=-0.5)-f32] PASSED [ 42%]
tests/geometry/test_base.py::test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic[lorentz(c=-0.5)-f64] PASSED [ 42%]
tests/geometry/test_base.py::test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic[lorentz(c=-2)-f32] PASSED [ 42%]
tests/geometry/test_base.py::test_ptransp_is_identity_at_the_same_point_and_reverses_the_geodesic[lorentz(c=-2)-f64] PASSED [ 43%]
tests/geometry/test_base.py::test_egrad2rgrad_is_tangent_and_matches_geoopt[euclidean-f32] PASSED [ 43%]
tests/geometry/test_base.py::test_egrad2rgrad_is_tangent_and_matches_geoopt[euclidean-f64] PASSED [ 43%]
tests/geometry/test_base.py::test_egrad2rgrad_is_tangent_and_matches_geoopt[poincare(c=-1)-f32] PASSED [ 43%]
tests/geometry/test_base.py::test_egrad2rgrad_is_tangent_and_matches_geoopt[poincare(c=-1)-f64] PASSED [ 44%]
tests/geometry/test_base.py::test_egrad2rgrad_is_tangent_and_matches_geoopt[poincare(c=-0.5)-f32] PASSED [ 44%]
tests/geometry/test_base.py::test_egrad2rgrad_is_tangent_and_matches_geoopt[poincare(c=-0.5)-f64] PASSED [ 44%]
tests/geometry/test_base.py::test_egrad2rgrad_is_tangent_and_matches_geoopt[poincare(c=-2)-f32] PASSED [ 45%]
tests/geometry/test_base.py::test_egrad2rgrad_is_tangent_and_matches_geoopt[poincare(c=-2)-f64] PASSED [ 45%]
tests/geometry/test_base.py::test_egrad2rgrad_is_tangent_and_matches_geoopt[lorentz(c=-1)-f32] PASSED [ 45%]
tests/geometry/test_base.py::test_egrad2rgrad_is_tangent_and_matches_geoopt[lorentz(c=-1)-f64] PASSED [ 45%]
tests/geometry/test_base.py::test_egrad2rgrad_is_tangent_and_matches_geoopt[lorentz(c=-0.5)-f32] PASSED [ 46%]
tests/geometry/test_base.py::test_egrad2rgrad_is_tangent_and_matches_geoopt[lorentz(c=-0.5)-f64] PASSED [ 46%]
tests/geometry/test_base.py::test_egrad2rgrad_is_tangent_and_matches_geoopt[lorentz(c=-2)-f32] PASSED [ 46%]
tests/geometry/test_base.py::test_egrad2rgrad_is_tangent_and_matches_geoopt[lorentz(c=-2)-f64] PASSED [ 47%]
tests/geometry/test_base.py::test_tangent0_lift_round_trip[euclidean-f32] PASSED [ 47%]
tests/geometry/test_base.py::test_tangent0_lift_round_trip[euclidean-f64] PASSED [ 47%]
tests/geometry/test_base.py::test_tangent0_lift_round_trip[poincare(c=-1)-f32] PASSED [ 47%]
tests/geometry/test_base.py::test_tangent0_lift_round_trip[poincare(c=-1)-f64] PASSED [ 48%]
tests/geometry/test_base.py::test_tangent0_lift_round_trip[poincare(c=-0.5)-f32] PASSED [ 48%]
tests/geometry/test_base.py::test_tangent0_lift_round_trip[poincare(c=-0.5)-f64] PASSED [ 48%]
tests/geometry/test_base.py::test_tangent0_lift_round_trip[poincare(c=-2)-f32] PASSED [ 49%]
tests/geometry/test_base.py::test_tangent0_lift_round_trip[poincare(c=-2)-f64] PASSED [ 49%]
tests/geometry/test_base.py::test_tangent0_lift_round_trip[lorentz(c=-1)-f32] PASSED [ 49%]
tests/geometry/test_base.py::test_tangent0_lift_round_trip[lorentz(c=-1)-f64] PASSED [ 50%]
tests/geometry/test_base.py::test_tangent0_lift_round_trip[lorentz(c=-0.5)-f32] PASSED [ 50%]
tests/geometry/test_base.py::test_tangent0_lift_round_trip[lorentz(c=-0.5)-f64] PASSED [ 50%]
tests/geometry/test_base.py::test_tangent0_lift_round_trip[lorentz(c=-2)-f32] PASSED [ 50%]
tests/geometry/test_base.py::test_tangent0_lift_round_trip[lorentz(c=-2)-f64] PASSED [ 51%]
tests/geometry/test_base.py::test_to_geoopt_returns_a_matching_manifold[euclidean] PASSED [ 51%]
tests/geometry/test_base.py::test_to_geoopt_returns_a_matching_manifold[poincare(c=-1)] PASSED [ 51%]
tests/geometry/test_base.py::test_to_geoopt_returns_a_matching_manifold[poincare(c=-0.5)] PASSED [ 52%]
tests/geometry/test_base.py::test_to_geoopt_returns_a_matching_manifold[poincare(c=-2)] PASSED [ 52%]
tests/geometry/test_base.py::test_to_geoopt_returns_a_matching_manifold[lorentz(c=-1)] PASSED [ 52%]
tests/geometry/test_base.py::test_to_geoopt_returns_a_matching_manifold[lorentz(c=-0.5)] PASSED [ 52%]
tests/geometry/test_base.py::test_to_geoopt_returns_a_matching_manifold[lorentz(c=-2)] PASSED [ 53%]
tests/geometry/test_base.py::test_check_point_rejects_points_off_the_manifold[euclidean] PASSED [ 53%]
tests/geometry/test_base.py::test_check_point_rejects_points_off_the_manifold[poincare(c=-1)] PASSED [ 53%]
tests/geometry/test_base.py::test_check_point_rejects_points_off_the_manifold[poincare(c=-0.5)] PASSED [ 54%]
tests/geometry/test_base.py::test_check_point_rejects_points_off_the_manifold[poincare(c=-2)] PASSED [ 54%]
tests/geometry/test_base.py::test_check_point_rejects_points_off_the_manifold[lorentz(c=-1)] PASSED [ 54%]
tests/geometry/test_base.py::test_check_point_rejects_points_off_the_manifold[lorentz(c=-0.5)] PASSED [ 54%]
tests/geometry/test_base.py::test_check_point_rejects_points_off_the_manifold[lorentz(c=-2)] PASSED [ 55%]
tests/geometry/test_base.py::test_outputs_keep_input_dtype[euclidean-f32] PASSED [ 55%]
tests/geometry/test_base.py::test_outputs_keep_input_dtype[euclidean-f64] PASSED [ 55%]
tests/geometry/test_base.py::test_outputs_keep_input_dtype[poincare(c=-1)-f32] PASSED [ 56%]
tests/geometry/test_base.py::test_outputs_keep_input_dtype[poincare(c=-1)-f64] PASSED [ 56%]
tests/geometry/test_base.py::test_outputs_keep_input_dtype[poincare(c=-0.5)-f32] PASSED [ 56%]
tests/geometry/test_base.py::test_outputs_keep_input_dtype[poincare(c=-0.5)-f64] PASSED [ 56%]
tests/geometry/test_base.py::test_outputs_keep_input_dtype[poincare(c=-2)-f32] PASSED [ 57%]
tests/geometry/test_base.py::test_outputs_keep_input_dtype[poincare(c=-2)-f64] PASSED [ 57%]
tests/geometry/test_base.py::test_outputs_keep_input_dtype[lorentz(c=-1)-f32] PASSED [ 57%]
tests/geometry/test_base.py::test_outputs_keep_input_dtype[lorentz(c=-1)-f64] PASSED [ 58%]
tests/geometry/test_base.py::test_outputs_keep_input_dtype[lorentz(c=-0.5)-f32] PASSED [ 58%]
tests/geometry/test_base.py::test_outputs_keep_input_dtype[lorentz(c=-0.5)-f64] PASSED [ 58%]
tests/geometry/test_base.py::test_outputs_keep_input_dtype[lorentz(c=-2)-f32] PASSED [ 59%]
tests/geometry/test_base.py::test_outputs_keep_input_dtype[lorentz(c=-2)-f64] PASSED [ 59%]
tests/geometry/test_euclidean.py::test_construction_and_curvature PASSED [ 59%]
tests/geometry/test_euclidean.py::test_primitives_reduce_to_vector_arithmetic[f32] PASSED [ 59%]
tests/geometry/test_euclidean.py::test_primitives_reduce_to_vector_arithmetic[f64] PASSED [ 60%]
tests/geometry/test_euclidean.py::test_geodesic_pairwise_and_check_point PASSED [ 60%]
tests/geometry/test_euclidean.py::test_gradients_match_finite_differences PASSED [ 60%]
tests/geometry/test_euclidean.py::test_to_geoopt_and_repr PASSED         [ 61%]
tests/geometry/test_lorentz.py::test_construction_k_and_geoopt_parameter[-0.5] PASSED [ 61%]
tests/geometry/test_lorentz.py::test_construction_k_and_geoopt_parameter[-1.0] PASSED [ 61%]
tests/geometry/test_lorentz.py::test_construction_k_and_geoopt_parameter[-2.0] PASSED [ 61%]
tests/geometry/test_lorentz.py::test_minkowski_inner_and_inner[f32] PASSED [ 62%]
tests/geometry/test_lorentz.py::test_minkowski_inner_and_inner[f64] PASSED [ 62%]
tests/geometry/test_lorentz.py::test_points_and_origin_lie_on_the_hyperboloid[f32--0.5] PASSED [ 62%]
tests/geometry/test_lorentz.py::test_points_and_origin_lie_on_the_hyperboloid[f32--1.0] PASSED [ 63%]
tests/geometry/test_lorentz.py::test_points_and_origin_lie_on_the_hyperboloid[f32--2.0] PASSED [ 63%]
tests/geometry/test_lorentz.py::test_points_and_origin_lie_on_the_hyperboloid[f64--0.5] PASSED [ 63%]
tests/geometry/test_lorentz.py::test_points_and_origin_lie_on_the_hyperboloid[f64--1.0] PASSED [ 63%]
tests/geometry/test_lorentz.py::test_points_and_origin_lie_on_the_hyperboloid[f64--2.0] PASSED [ 64%]
tests/geometry/test_lorentz.py::test_logmap_is_tangent_and_dist_is_its_length[-0.5] PASSED [ 64%]
tests/geometry/test_lorentz.py::test_logmap_is_tangent_and_dist_is_its_length[-1.0] PASSED [ 64%]
tests/geometry/test_lorentz.py::test_logmap_is_tangent_and_dist_is_its_length[-2.0] PASSED [ 65%]
tests/geometry/test_lorentz.py::test_own_ptransp_matches_geoopt_and_is_finite_at_coincident_points[f32--0.5] PASSED [ 65%]
tests/geometry/test_lorentz.py::test_own_ptransp_matches_geoopt_and_is_finite_at_coincident_points[f32--1.0] PASSED [ 65%]
tests/geometry/test_lorentz.py::test_own_ptransp_matches_geoopt_and_is_finite_at_coincident_points[f32--2.0] PASSED [ 65%]
tests/geometry/test_lorentz.py::test_own_ptransp_matches_geoopt_and_is_finite_at_coincident_points[f64--0.5] PASSED [ 66%]
tests/geometry/test_lorentz.py::test_own_ptransp_matches_geoopt_and_is_finite_at_coincident_points[f64--1.0] PASSED [ 66%]
tests/geometry/test_lorentz.py::test_own_ptransp_matches_geoopt_and_is_finite_at_coincident_points[f64--2.0] PASSED [ 66%]
tests/geometry/test_lorentz.py::test_isometry_to_poincare_ball[-0.5] PASSED [ 67%]
tests/geometry/test_lorentz.py::test_isometry_to_poincare_ball[-1.0] PASSED [ 67%]
tests/geometry/test_lorentz.py::test_isometry_to_poincare_ball[-2.0] PASSED [ 67%]
tests/geometry/test_lorentz.py::test_gradients_match_finite_differences[-0.5] PASSED [ 68%]
tests/geometry/test_lorentz.py::test_gradients_match_finite_differences[-1.0] PASSED [ 68%]
tests/geometry/test_lorentz.py::test_gradients_match_finite_differences[-2.0] PASSED [ 68%]
tests/geometry/test_lorentz.py::test_tangent_lift_and_euclidean_projection PASSED [ 68%]
tests/geometry/test_lorentz.py::test_float32_vs_float64_at_the_image_of_0999_radius[-1.0] PASSED [ 69%]
tests/geometry/test_lorentz.py::test_float32_vs_float64_at_the_image_of_0999_radius[-2.0] PASSED [ 69%]
tests/geometry/test_poincare.py::test_construction_radius_and_geoopt_parameter[-0.5] PASSED [ 69%]
tests/geometry/test_poincare.py::test_construction_radius_and_geoopt_parameter[-1.0] PASSED [ 70%]
tests/geometry/test_poincare.py::test_construction_radius_and_geoopt_parameter[-2.0] PASSED [ 70%]
tests/geometry/test_poincare.py::test_lambda_x_closed_form[f32--0.5] PASSED [ 70%]
tests/geometry/test_poincare.py::test_lambda_x_closed_form[f32--1.0] PASSED [ 70%]
tests/geometry/test_poincare.py::test_lambda_x_closed_form[f32--2.0] PASSED [ 71%]
tests/geometry/test_poincare.py::test_lambda_x_closed_form[f64--0.5] PASSED [ 71%]
tests/geometry/test_poincare.py::test_lambda_x_closed_form[f64--1.0] PASSED [ 71%]
tests/geometry/test_poincare.py::test_lambda_x_closed_form[f64--2.0] PASSED [ 72%]
tests/geometry/test_poincare.py::test_mobius_add_identity_inverse_and_geoopt[f32--0.5] PASSED [ 72%]
tests/geometry/test_poincare.py::test_mobius_add_identity_inverse_and_geoopt[f32--1.0] PASSED [ 72%]
tests/geometry/test_poincare.py::test_mobius_add_identity_inverse_and_geoopt[f32--2.0] PASSED [ 72%]
tests/geometry/test_poincare.py::test_mobius_add_identity_inverse_and_geoopt[f64--0.5] PASSED [ 73%]
tests/geometry/test_poincare.py::test_mobius_add_identity_inverse_and_geoopt[f64--1.0] PASSED [ 73%]
tests/geometry/test_poincare.py::test_mobius_add_identity_inverse_and_geoopt[f64--2.0] PASSED [ 73%]
tests/geometry/test_poincare.py::test_gyration_is_an_isometry_and_trivial_at_zero[-0.5] PASSED [ 74%]
tests/geometry/test_poincare.py::test_gyration_is_an_isometry_and_trivial_at_zero[-1.0] PASSED [ 74%]
tests/geometry/test_poincare.py::test_gyration_is_an_isometry_and_trivial_at_zero[-2.0] PASSED [ 74%]
tests/geometry/test_poincare.py::test_ptransp_preserves_conformal_norm[f32--0.5] PASSED [ 75%]
tests/geometry/test_poincare.py::test_ptransp_preserves_conformal_norm[f32--1.0] PASSED [ 75%]
tests/geometry/test_poincare.py::test_ptransp_preserves_conformal_norm[f32--2.0] PASSED [ 75%]
tests/geometry/test_poincare.py::test_ptransp_preserves_conformal_norm[f64--0.5] PASSED [ 75%]
tests/geometry/test_poincare.py::test_ptransp_preserves_conformal_norm[f64--1.0] PASSED [ 76%]
tests/geometry/test_poincare.py::test_ptransp_preserves_conformal_norm[f64--2.0] PASSED [ 76%]
tests/geometry/test_poincare.py::test_distance_closed_form_from_origin PASSED [ 76%]
tests/geometry/test_poincare.py::test_gradients_match_finite_differences[-0.5] PASSED [ 77%]
tests/geometry/test_poincare.py::test_gradients_match_finite_differences[-1.0] PASSED [ 77%]
tests/geometry/test_poincare.py::test_gradients_match_finite_differences[-2.0] PASSED [ 77%]
tests/geometry/test_poincare.py::test_proj_clips_to_boundary[f32] PASSED [ 77%]
tests/geometry/test_poincare.py::test_proj_clips_to_boundary[f64] PASSED [ 78%]
tests/geometry/test_poincare.py::test_float32_vs_float64_at_0999_of_the_boundary_radius[-1.0] PASSED [ 78%]
tests/geometry/test_poincare.py::test_float32_vs_float64_at_0999_of_the_boundary_radius[-2.0] PASSED [ 78%]
tests/geometry/test_poincare.py::test_float32_round_trip_holds_inside_the_clip_margin[-1.0] PASSED [ 79%]
tests/geometry/test_poincare.py::test_float32_round_trip_holds_inside_the_clip_margin[-2.0] PASSED [ 79%]
tests/geometry/test_public_api.py::test_every_public_name_is_referenced_by_a_geometry_test PASSED [ 79%]
tests/geometry/test_public_api.py::test_public_api_is_non_trivial PASSED [ 79%]
tests/geometry/test_utils.py::test_eps_and_min_norm_tables PASSED        [ 80%]
tests/geometry/test_utils.py::test_artanh_matches_torch_inside_and_is_finite_at_the_boundary[f32] PASSED [ 80%]
tests/geometry/test_utils.py::test_artanh_matches_torch_inside_and_is_finite_at_the_boundary[f64] PASSED [ 80%]
tests/geometry/test_utils.py::test_arcosh_matches_torch_and_is_finite_at_one[f32] PASSED [ 81%]
tests/geometry/test_utils.py::test_arcosh_matches_torch_and_is_finite_at_one[f64] PASSED [ 81%]
tests/geometry/test_utils.py::test_safe_norm_never_returns_zero[f32] PASSED [ 81%]
tests/geometry/test_utils.py::test_safe_norm_never_returns_zero[f64] PASSED [ 81%]
tests/geometry/test_utils.py::test_clip_norm_only_shrinks[f32] PASSED    [ 82%]
tests/geometry/test_utils.py::test_clip_norm_only_shrinks[f64] PASSED    [ 82%]
tests/geometry/test_utils.py::test_helpers_keep_shape[artanh] PASSED     [ 82%]
tests/geometry/test_utils.py::test_helpers_keep_shape[arcosh] PASSED     [ 83%]
tests/metrics/test_distortion.py::test_perfect_embedding_has_zero_distortion_and_unit_map PASSED [ 83%]
tests/metrics/test_distortion.py::test_random_embedding_is_worse_than_structured PASSED [ 83%]
tests/metrics/test_distortion.py::test_shape_validation PASSED           [ 84%]
tests/metrics/test_gromov_hyperbolicity.py::test_tree_metric_has_zero_delta PASSED [ 84%]
tests/metrics/test_gromov_hyperbolicity.py::test_grid_has_positive_delta PASSED [ 84%]
tests/metrics/test_gromov_hyperbolicity.py::test_delta_is_scale_covariant PASSED [ 84%]
tests/metrics/test_gromov_hyperbolicity.py::test_subsampled_estimator_on_point_clouds PASSED [ 85%]
tests/models/test_predictor_heads.py::test_output_shapes_from_random_inputs_on_cpu[euclidean] PASSED [ 85%]
tests/models/test_predictor_heads.py::test_output_shapes_from_random_inputs_on_cpu[hyperbolic-on-euclidean] PASSED [ 85%]
tests/models/test_predictor_heads.py::test_output_shapes_from_random_inputs_on_cpu[hyperbolic-poincare] PASSED [ 86%]
tests/models/test_predictor_heads.py::test_output_shapes_from_random_inputs_on_cpu[hyperbolic-poincare(c=-0.5)] PASSED [ 86%]
tests/models/test_predictor_heads.py::test_output_shapes_from_random_inputs_on_cpu[hyperbolic-lorentz] PASSED [ 86%]
tests/models/test_predictor_heads.py::test_outputs_stay_on_the_manifold_and_updates_are_bounded[euclidean] PASSED [ 86%]
tests/models/test_predictor_heads.py::test_outputs_stay_on_the_manifold_and_updates_are_bounded[hyperbolic-on-euclidean] PASSED [ 87%]
tests/models/test_predictor_heads.py::test_outputs_stay_on_the_manifold_and_updates_are_bounded[hyperbolic-poincare] PASSED [ 87%]
tests/models/test_predictor_heads.py::test_outputs_stay_on_the_manifold_and_updates_are_bounded[hyperbolic-poincare(c=-0.5)] PASSED [ 87%]
tests/models/test_predictor_heads.py::test_outputs_stay_on_the_manifold_and_updates_are_bounded[hyperbolic-lorentz] PASSED [ 88%]
tests/models/test_predictor_heads.py::test_gradients_reach_every_trainable_parameter_and_not_the_projection[euclidean] PASSED [ 88%]
tests/models/test_predictor_heads.py::test_gradients_reach_every_trainable_parameter_and_not_the_projection[hyperbolic-on-euclidean] PASSED [ 88%]
tests/models/test_predictor_heads.py::test_gradients_reach_every_trainable_parameter_and_not_the_projection[hyperbolic-poincare] PASSED [ 88%]
tests/models/test_predictor_heads.py::test_gradients_reach_every_trainable_parameter_and_not_the_projection[hyperbolic-poincare(c=-0.5)] PASSED [ 89%]
tests/models/test_predictor_heads.py::test_gradients_reach_every_trainable_parameter_and_not_the_projection[hyperbolic-lorentz] PASSED [ 89%]
tests/models/test_predictor_heads.py::test_action_conditions_the_prediction[euclidean] PASSED [ 89%]
tests/models/test_predictor_heads.py::test_action_conditions_the_prediction[hyperbolic-on-euclidean] PASSED [ 90%]
tests/models/test_predictor_heads.py::test_action_conditions_the_prediction[hyperbolic-poincare] PASSED [ 90%]
tests/models/test_predictor_heads.py::test_action_conditions_the_prediction[hyperbolic-poincare(c=-0.5)] PASSED [ 90%]
tests/models/test_predictor_heads.py::test_action_conditions_the_prediction[hyperbolic-lorentz] PASSED [ 90%]
tests/models/test_predictor_heads.py::test_heads_are_architecturally_identical PASSED [ 91%]
tests/models/test_predictor_heads.py::test_same_seed_gives_identical_initial_weights_and_different_seeds_differ PASSED [ 91%]
tests/models/test_predictor_heads.py::test_max_radius_retraction_bounds_every_latent[euclidean] PASSED [ 91%]
tests/models/test_predictor_heads.py::test_max_radius_retraction_bounds_every_latent[hyperbolic-on-euclidean] PASSED [ 92%]
tests/models/test_predictor_heads.py::test_max_radius_retraction_bounds_every_latent[hyperbolic-poincare] PASSED [ 92%]
tests/models/test_predictor_heads.py::test_max_radius_retraction_bounds_every_latent[hyperbolic-poincare(c=-0.5)] PASSED [ 92%]
tests/models/test_predictor_heads.py::test_max_radius_retraction_bounds_every_latent[hyperbolic-lorentz] PASSED [ 93%]
tests/models/test_predictor_heads.py::test_steps_are_measured_in_geodesic_units_in_every_geometry[euclidean] PASSED [ 93%]
tests/models/test_predictor_heads.py::test_steps_are_measured_in_geodesic_units_in_every_geometry[hyperbolic-on-euclidean] PASSED [ 93%]
tests/models/test_predictor_heads.py::test_steps_are_measured_in_geodesic_units_in_every_geometry[hyperbolic-poincare] PASSED [ 93%]
tests/models/test_predictor_heads.py::test_steps_are_measured_in_geodesic_units_in_every_geometry[hyperbolic-poincare(c=-0.5)] PASSED [ 94%]
tests/models/test_predictor_heads.py::test_steps_are_measured_in_geodesic_units_in_every_geometry[hyperbolic-lorentz] PASSED [ 94%]
tests/models/test_predictor_heads.py::test_poincare_and_lorentz_heads_are_isometric_twins PASSED [ 94%]
tests/models/test_predictor_heads.py::test_hyperbolic_head_on_euclidean_manifold_equals_euclidean_head PASSED [ 95%]
tests/models/test_predictor_heads.py::test_embedding_uses_expmap0_and_is_frozen PASSED [ 95%]
tests/models/test_predictor_heads.py::test_euclidean_head_rejects_curved_manifold_and_registry_builds_both PASSED [ 95%]
tests/models/test_vjepa2.py::test_loader_uses_torch_hub_entry_and_freezes_everything PASSED [ 95%]
tests/models/test_vjepa2.py::test_encoder_forward_returns_patch_and_pooled_embeddings PASSED [ 96%]
tests/models/test_vjepa2.py::test_normalize_reps_off_keeps_raw_tokens PASSED [ 96%]
tests/models/test_vjepa2.py::test_assert_frozen_catches_a_thawed_parameter PASSED [ 96%]
tests/models/test_vjepa2.py::test_reference_predictor_shapes_rollout_and_loss PASSED [ 97%]
tests/models/test_vjepa2.py::test_pretrained_path_cleans_keys_and_loads_strictly PASSED [ 97%]
tests/models/test_vjepa2.py::test_local_checkout_path_uses_hub_local_source PASSED [ 97%]
tests/models/test_vjepa2.py::test_missing_hub_dependencies_produce_an_actionable_error PASSED [ 97%]
tests/models/test_vjepa2.py::test_registry_builds_vjepa2_ac_bundle_with_reference_predictor PASSED [ 98%]
tests/test_smoke_experiment.py::test_smoke_config_is_tiny_and_cpu PASSED [ 98%]
tests/test_smoke_experiment.py::test_smoke_runs_end_to_end[poincare] PASSED [ 98%]
tests/test_smoke_experiment.py::test_smoke_runs_end_to_end[lorentz] PASSED [ 99%]
tests/test_smoke_experiment.py::test_smoke_runs_end_to_end[euclidean] PASSED [ 99%]
tests/test_smoke_experiment.py::test_encoder_stays_frozen PASSED         [ 99%]
tests/test_smoke_experiment.py::test_hyperbolic_head_on_euclidean_manifold_matches_euclidean_head PASSED [100%]

============================= 344 passed in 5.44s ==============================
```

What the tests cover:

- [x] Manifold contract (`tests/geometry/test_base.py`), run over Euclidean, Poincaré at c ∈ {-0.5, -1, -2} and Lorentz at c ∈ {-0.5, -1, -2}, in float32 and float64: exp/log inverse identity on random points and random tangents; distance symmetry, non-negativity, zero self-distance and the triangle inequality on random points; `sqdist`, `dist0` and `pairwise_dist` consistent with `dist`; origin closed forms match the general maps; geodesic endpoints, midpoint and additivity; `proj` idempotent and on-manifold; `proj_tan` idempotent and `logmap` tangent; parallel transport is the identity at a point, reverses the geodesic and preserves lengths; `egrad2rgrad` tangent and equal to geoopt's; tangent-lift round trip; `to_geoopt` returns the matching geoopt manifold usable as a `ManifoldParameter`; `check_point` rejects off-manifold points; outputs keep the input dtype; `build_manifold` accepts `c` or `curvature` and rejects unknown names.
- [x] Poincaré ball (`test_poincare.py`): construction and the float64 geoopt parameter; `lambda_x` closed form; Möbius addition identity, inverse, left cancellation, non-commutativity and agreement with geoopt; gyration is an isometry, trivial at zero and defines transport; transport preserves the conformal norm; closed-form distance from the origin; `gradcheck` against finite differences for `dist`, `expmap`, `logmap`, `expmap0`, `logmap0`, `ptransp`, `dist0`; boundary clipping to `(1 - eps) * radius`; float32 vs float64 at 0.999 of the boundary radius (agreement to 1e-4 off the Möbius path, ~1% on it, everything finite including gradients, float64 round trip to 1e-5) and the float32 round trip inside the clip margin.
- [x] Lorentz hyperboloid (`test_lorentz.py`): construction and the float64 geoopt parameter; Minkowski inner product; points and origin on the hyperboloid; `logmap` tangent with length equal to `dist`; our transport matches geoopt's at distinct points and is finite at coincident points; isometry to the Poincaré ball at every curvature to 1e-9 (points, distances, midpoints); `gradcheck` for the same seven functions; tangent lift; float32 vs float64 at the image of 0.999 of the radius (`dist`/`dist0` to 1e-5, self-distance noise ≤ 2e-3·x₀, `logmap` within 20%, on-manifold and finite) and the documented exponential loss of the float64 round trip with distance.
- [x] Euclidean (`test_euclidean.py`): the same contract reduces to vector arithmetic; geodesics, pairwise distances, `gradcheck`, geoopt equivalent.
- [x] Utilities (`test_utils.py`): per-dtype epsilon tables; `artanh` and `arcosh` match torch inside the domain, stay finite at the boundary and have correct gradients; `safe_norm` never returns zero; `clip_norm` only shrinks and preserves direction.
- [x] API coverage guard (`test_public_api.py`): every public function, method and constant in the geometry package is referenced by a geometry test.
- [x] Predictor heads (`tests/models/test_predictor_heads.py`), over the Euclidean head and the hyperbolic head on Euclidean, Poincaré (c ∈ {-1, -0.5}) and Lorentz manifolds: output shapes of `embed`, `step`, `rollout`, `forward`, `fuse`, `delta` and `coordinates` from random CPU inputs; outputs stay on the manifold with `max_step` and `max_radius` respected even under forced maximal updates; gradients reach every trainable parameter and never the frozen projection; the action conditions the prediction; both heads have identical parameter signatures and identical initial weights for the same seed; steps are measured in geodesic units in every geometry; Poincaré and Lorentz heads with identical weights are isometric twins; `HyperbolicHead(Euclidean)` equals `EuclideanHead` exactly; the embedding is `expmap0` of the frozen orthonormal projection; the registry builds both heads and rejects mismatched geometry.
- [x] V-JEPA 2-AC loader (`tests/models/test_vjepa2.py`, with tiny stand-ins for the hub modules, no download): `torch.hub.load` is called with the `vjepa2_ac_vit_giant` entry, pinned ref and `pretrained=False`; encoder and Meta's reference predictor come back frozen and cannot be switched to training mode; `assert_frozen` catches a thawed parameter in either; `forward` returns patch `(B, T, N, D)` and pooled `(B, T, D)` embeddings, encodes each frame as a 2-frame tubelet and layer-normalises tokens (and does not when `normalize_reps` is off); the reference predictor's frame-causal forward, `predict_next`, autoregressive `rollout` and L1 loss have the expected shapes and reject misaligned inputs; the pretrained path cleans `module.`/`backbone.` prefixes, loads strictly and refuses a checkpoint with missing keys; a local checkout path is loaded with hub's `source="local"`; a missing `timm` produces an actionable error; the registry builds the bundle with the reference predictor attached.
- [x] Cosmos pipeline (`tests/data/test_cosmos3.py`, all with fakes, no model): action-spec parsing and validation (shapes, missing frames, duplicate ids, unsupported formats); deterministic distinct per-branch seeds; end-to-end generation writes `frames.npz`, `meta.json` and `manifest.jsonl` with shared start frames across branches, is idempotent and reproducible; the CLI's `--dry-run` plan and injected-generator path; the NVIDIA adapter replays the chunked `generate_vid2world` loop with the expected input tensors (first frame real, rest zero), chunk sizes, zero padding, per-chunk seeds, sampler settings and chaining, and stitches without duplicated or dropped frames; a missing `cosmos_predict2` package gives an actionable error; latent pooling (mean, flatten), extraction to float16 npz with manifest update, idempotence and the CLI; the tokenizer backend feeds a `[-1, 1]` bfloat16 `(1, 3, T, H, W)` tensor; the dataset's windows, shapes, resize, metadata, branch groups and pairs sharing a start frame, compositional split with its safety checks, hierarchy construction, latents access, deterministic splits and config construction.
- [x] Gromov δ (`tests/metrics`): exactly 0 on the embodiment > task > primitive tree metric from any base point; positive on L1 and L2 grids; scale covariant; subsampled estimator on point clouds, including a path (δ = 0).
- [x] Distortion and mAP: 0 and 1.0 on a perfectly embedded path graph; scale fitting matters; shape validation; structured beats random on a Poincaré tree embedding.
- [x] Smoke experiment: `configs/experiments/smoke.yaml` runs end to end on CPU in Poincaré, Lorentz and Euclidean geometry; loss decreases; error grows with horizon; `metrics.json`, curves CSV and resolved config are written with the geometry recorded; the encoder cannot be un-frozen; `HyperbolicHead` on the flat manifold equals `EuclideanHead` exactly.

CI runs `ruff check`, `ruff format --check`, `pytest tests/geometry tests/metrics` and `scripts/run_smoke.sh`, all on CPU.

---

## ⚖️ Weights and Licensing

No model weights are included in this repository. Every checkpoint is downloaded by the user from its upstream source under the upstream terms, and none is redistributed:

- **V-JEPA 2-AC** (`vjepa2_ac_vit_giant` via `torch.hub`, weights `vjepa2-ac-vitg.pt` from Meta's official URL): MIT-licensed code; weights per the upstream release. Encoder used frozen; Meta's bundled predictor kept frozen as the reference baseline.
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
