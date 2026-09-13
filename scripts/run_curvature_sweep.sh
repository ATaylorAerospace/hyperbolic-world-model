#!/usr/bin/env bash
# Sweep curvature x latent dimension x seed for both hyperbolic models (Hydra multirun).
# Curvature is never fixed: every reported hyperbolic number comes from this grid.
# tasks=all includes compositional generalisation, so pass a held-out pair, e.g.
#   bash scripts/run_curvature_sweep.sh both 'data.holdout_combinations=[[franka,grasp]]'
#
# Usage: bash scripts/run_curvature_sweep.sh [poincare|lorentz|both] [extra hydra overrides...]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && source .env && set +a
WHICH="${1:-both}"; shift || true

run() { uv run --no-sync python -m hyperbolic_world_model.training.train_predictor "experiments=$1" --multirun "$@"; }
case "$WHICH" in
  poincare) run poincare_sweep "$@" ;;
  lorentz)  run lorentz_sweep "$@" ;;
  both)     run poincare_sweep "$@"; run lorentz_sweep "$@" ;;
  *) echo "unknown sweep: $WHICH" >&2; exit 2 ;;
esac
