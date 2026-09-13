# Methodology

This document states the hypotheses each task tests, what result would falsify each one, and the
two rules under which every number in this repository is produced.

## The two non-negotiable rules

**Rule 1: native geometry.** All distance-based metrics are reported in the native geometry of
each baseline. A Euclidean head is scored with Euclidean distance; a Poincaré head with the
Poincaré geodesic distance at the curvature it was trained with; a Lorentz head with the
hyperboloid geodesic distance. No metric ever maps one model's latents into another model's space,
because such a map would be an arbitrary choice that favours whichever geometry it targets. The
code makes this structural: every metric in `src/hyperbolic_world_model/metrics/` takes a
`Manifold` argument and calls `manifold.dist`; there is no Euclidean fallback. The task result
records `geometry` and `curvature` next to every scalar (`tasks/base.py::TaskResult`), and the
report tables never show a value without those two columns.

Where a comparison *across* geometries is needed, we compare dimensionless quantities:
the error normalised by the distance actually travelled (`normalised_geodesic_error`, which is 1
for a static predictor in every geometry), ranking metrics (mAP), scale-fitted distortion, and
relative delta-hyperbolicity.

**Rule 2: curvature is swept, never fixed.** No hyperbolic result is reported at a single
curvature. Every hyperbolic experiment is a Hydra multirun over
`geometry.curvature ∈ {-0.1, -0.25, -0.5, -1.0, -2.0, -4.0}` crossed with latent dimension and
seed (`configs/experiments/poincare_sweep.yaml`, `lorentz_sweep.yaml`). Tables report the full
curve; "best curvature" is reported alongside the curve, never instead of it. The default
`curvature: -1.0` in `configs/geometry/*.yaml` exists only so single debug runs have a value.

**Invariant: the encoder is frozen.** V-JEPA 2 and DINOv2 are loaded with every parameter set to
`requires_grad=False` and in `eval()` mode. `FrozenEncoder.assert_frozen` runs before the first
optimiser step and before any checkpoint is written; only the predictor head's state dict is ever
saved. `build_encoder` refuses a config with `encoder.frozen: false`. The test
`tests/test_smoke_experiment.py::test_encoder_stays_frozen` enforces both.

## Why hyperbolic space might help a world model

Three structures in embodied data are tree-like:

1. **Embodiment hierarchies.** Robot > arm type > gripper > primitive skill. Distances between
   siblings should be small; distances between subtrees large; the number of leaves grows
   exponentially with depth.
2. **Task decompositions.** A language instruction decomposes into sub-goals, which decompose into
   primitives. The decomposition is a tree.
3. **Branching futures.** From one state, the set of reachable states after `h` steps grows
   roughly exponentially with `h`. A latent space that must keep those futures distinguishable
   needs volume that grows exponentially with radius.

Euclidean space has polynomial volume growth, so embedding any of these with low distortion
requires dimension that grows with the size of the tree (Bourgain-type lower bounds). Hyperbolic
space has exponential volume growth and embeds trees with arbitrarily low distortion in two
dimensions (Sarkar, 2011). The hypothesis is that a predictor head whose latent lives in
hyperbolic space will (a) need fewer dimensions for the same rollout error, (b) preserve the
hierarchy better, and (c) keep branching futures separated for longer.

The counter-hypothesis is equally plausible and is what the harness must be able to show: the
frozen encoder already linearises the relevant structure, the residual dynamics are locally
Euclidean, and curvature only adds optimisation difficulty. The metrics below are chosen so that
either outcome is visible.

## Tasks, hypotheses, falsification

### Latent rollout (`tasks/latent_rollout.py`)

- **Hypothesis.** At equal latent dimension, the best swept curvature gives lower normalised
  geodesic rollout error than the Euclidean head at horizons beyond one step.
- **Metric.** `geodesic_error_per_horizon` and `normalised_error_hmax` in each model's geometry.
- **Falsified if** no curvature in the sweep beats the Euclidean baseline's normalised error at
  any horizon `h > 1` by more than one seed standard deviation, at any latent dimension.
- **Data.** DROID (real), Cosmos 3 generated trajectories (controlled), synthetic (CI only).

### Hierarchy reconstruction (`tasks/hierarchy_reconstruction.py`)

- **Hypothesis.** Per-trajectory latents (Fréchet mean over time) recover the
  embodiment > task > primitive tree with lower average distortion and higher mAP in hyperbolic
  space than in Euclidean space at the same dimension, and `dist0` (distance from origin)
  correlates with tree depth.
- **Metric.** `average_distortion` (scale-fitted), `mean_average_precision`, Spearman(depth, dist0).
- **Falsified if** the best curvature does not improve both distortion and mAP over Euclidean by
  more than the seed standard deviation, or if the depth correlation is not positive.
- **Data.** DROID metadata; synthetic hierarchy for CI.

### Long-horizon consistency (`tasks/long_horizon_consistency.py`)

- **Hypothesis.** For rollouts sharing a start frame but differing in actions (Cosmos 3
  branches), latent divergence tracks video divergence with higher correlation, and saturates
  later, in hyperbolic space.
- **Metric.** Spearman correlation between geodesic latent divergence and encoder-space divergence
  of the generated frames as a function of horizon; horizon at which latent divergence saturates.
- **Falsified if** the correlation is not higher, or saturation is not later, for the best
  curvature versus Euclidean.
- **Data.** Cosmos 3 generated branching trajectories.

### Compositional generalisation (`tasks/compositional_generalization.py`)

- **Hypothesis.** Holding out (embodiment, primitive) combinations, the gap between unseen and
  seen rollout error is smaller in hyperbolic space.
- **Metric.** `unseen_error - seen_error` (normalised, native geometry), per curvature.
- **Falsified if** the gap is not smaller for the best curvature, or is smaller only because seen
  error got worse.
- **Data.** DROID with `data.holdout_combinations`.

### Latent-space probe: delta-hyperbolicity (`metrics/gromov_hyperbolicity.py`)

Independent of any model we train: we estimate the relative Gromov delta of (a) raw frozen
V-JEPA 2 latents, (b) raw DINOv2 latents, (c) Cosmos 3 tokenizer latents. Low delta means the
encoder already organises data tree-like, which predicts that the hyperbolic head should help;
delta near 1 predicts it should not. This is a pre-registered prediction: we report it before the
sweeps and state whether the sweep outcome agreed.

## Dimension efficiency

Every task is run across `latent_dim ∈ {8, 16, 32, 64, 128}`. `metrics/dimension_efficiency.py`
turns the results into metric-vs-dimension curves and their area under the curve (log2 dimension
axis). The headline claim, if the hypothesis holds, is "hyperbolic reaches error `e` at dimension
`d_h < d_e`", with both numbers and the full curves in the report.

## Statistical protocol

- Three seeds per configuration; tables show mean ± std.
- A difference counts only if it exceeds one std of the seeds at both configurations.
- The Euclidean baseline is run with the same head architecture, optimiser, schedule, latent
  dimensions and seeds as the hyperbolic heads; only the manifold differs.
- The embedding from encoder space onto the manifold is a frozen seeded projection in phase 1, so
  no geometry can "win" by collapsing its embedding. A learnable embedding with an anti-collapse
  regulariser is a phase-3 ablation, reported separately.

## Numerical caveats (measured, see `tests/geometry`)

- Poincaré ball in float32 cannot represent geodesic distances beyond ~6.2 (unit curvature)
  because the `artanh` argument is clamped at `1 - 4e-3`. Latents must stay well inside this;
  `HyperbolicHead.max_step` and `embed_scale` enforce it, and a Lorentz run is the cross-check.
- Lorentz in float32 has self-distance noise growing like `1e-3 · x_0`; beyond ~6 units from the
  origin use float64 for the geometry.
- Both models agree with each other under the isometry to 1e-9 in float64
  (`test_lorentz.py::test_isometric_to_poincare_ball`), which is the cross-geometry consistency check.
