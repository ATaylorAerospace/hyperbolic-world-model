#!/usr/bin/env bash
# Euclidean baseline: frozen V-JEPA 2 + Euclidean head on DROID, all tasks.
# Requires weights (scripts/download_weights.sh) and DROID under $DATA_ROOT/droid.
# tasks=all includes compositional generalisation, so pass a held-out pair, e.g.
#   bash scripts/run_baseline.sh 'data.holdout_combinations=[[franka,grasp]]'
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && source .env && set +a
uv run --no-sync python -m hyperbolic_world_model.training.train_predictor experiments=baseline_euclidean "$@"
