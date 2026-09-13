#!/usr/bin/env bash
# Euclidean baseline: frozen V-JEPA 2 + Euclidean head on DROID, all tasks.
# Requires weights (scripts/download_weights.sh) and DROID under $DATA_ROOT/droid.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && source .env && set +a
uv run --no-sync python -m hyperbolic_world_model.training.train_predictor experiments=baseline_euclidean "$@"
