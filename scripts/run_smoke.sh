#!/usr/bin/env bash
# Tiny CPU end-to-end run (synthetic data + synthetic frozen encoder + Poincaré head). ~10 s.
# This is the "one smoke benchmark" CI executes after the unit tests.
set -euo pipefail
cd "$(dirname "$0")/.."
export HWM_DEVICE=cpu
python -m hyperbolic_world_model.training.train_predictor experiments=smoke "$@"
echo
echo "smoke outputs:"
find outputs/smoke -name metrics.json -exec sh -c 'echo "  $1"; python -c "import json,sys; d=json.load(open(sys.argv[1])); print(\"   \", d[\"geometry\"], \"K=\", d[\"curvature\"], d[\"tasks\"][\"latent_rollout\"][\"metrics\"])" "$1"' _ {} \;
