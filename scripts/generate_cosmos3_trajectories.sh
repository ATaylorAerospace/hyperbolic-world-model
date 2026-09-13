#!/usr/bin/env bash
# Generate action-conditioned video rollouts with Cosmos 3 Nano, then extract tokenizer latents.
# Cosmos 3 is used in inference mode only (see docs/cosmos3_usage.md).
#
# Usage: bash scripts/generate_cosmos3_trajectories.sh [--num-frames N] [--seed S]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && source .env && set +a
DATA_ROOT="${DATA_ROOT:-data}"

uv run --no-sync python -m hyperbolic_world_model.data.cosmos3.generate \
  --prompts "$DATA_ROOT/cosmos3/prompts" \
  --out "$DATA_ROOT/cosmos3/rollouts" \
  "$@"

uv run --no-sync python -m hyperbolic_world_model.data.cosmos3.extract_latents \
  --rollouts "$DATA_ROOT/cosmos3/rollouts" \
  --out "$DATA_ROOT/cosmos3/latents"
