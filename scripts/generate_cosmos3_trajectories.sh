#!/usr/bin/env bash
# Generate action-conditioned video rollouts with Cosmos, then extract tokenizer latents.
# Cosmos is used in inference mode only (docs/cosmos3_usage.md). Requires NVIDIA's cosmos_predict2
# package and a GPU on this machine; nothing in this repository installs them.
#
# Usage: bash scripts/generate_cosmos3_trajectories.sh --start-frames DIR --actions FILE.json [generate.py options]
#        bash scripts/generate_cosmos3_trajectories.sh --start-frames DIR --actions FILE.json --dry-run   # plan only
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && source .env && set +a
DATA_ROOT="${DATA_ROOT:-data}"
OUT="$DATA_ROOT/cosmos3_generated"

uv run --no-sync python -m hyperbolic_world_model.data.cosmos3.generate --out "$OUT" "$@"

case " $* " in
  *" --dry-run "*) echo "dry run: skipping latent extraction" ;;
  *) uv run --no-sync python -m hyperbolic_world_model.data.cosmos3.extract_latents --root "$OUT" ;;
esac
