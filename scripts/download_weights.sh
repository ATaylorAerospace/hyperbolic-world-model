#!/usr/bin/env bash
# Download frozen encoder weights into ${CKPT_ROOT:-checkpoints}/encoders.
# Reads HF_TOKEN from the environment (or .env). Never downloads datasets.
#
#   V-JEPA 2-AC  vjepa2_ac_vit_giant via torch.hub (MIT code; weights from dl.fbaipublicfiles.com)
#                needs `timm` importable (upstream hub dependency): uv pip install timm
#   DINOv2     facebook/dinov2-base             (Apache 2.0)
#   Cosmos 3   nvidia/Cosmos-3-Nano              (OpenMDW 1.1, gated: accept terms on the Hub first)
#
# Usage: bash scripts/download_weights.sh [vjepa2|dinov2|cosmos3|all]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && source .env && set +a

CKPT_ROOT="${CKPT_ROOT:-checkpoints}"
TARGET="${1:-all}"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "warning: HF_TOKEN is not set; public repos will still download, gated ones (Cosmos 3) will fail" >&2
fi

dl() {  # dl <repo_id> <subdir>
  local repo="$1" sub="$2"
  echo ">> $repo -> $CKPT_ROOT/encoders/$sub"
  uv run --no-sync python - "$repo" "$CKPT_ROOT/encoders/$sub" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo, out = sys.argv[1], sys.argv[2]
path = snapshot_download(repo_id=repo, cache_dir=out, token=os.environ.get("HF_TOKEN") or None,
                         allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md", "*.py"])
print("   cached at", path)
PY
}

dl_vjepa2_ac() {
  echo ">> vjepa2_ac_vit_giant (torch.hub) -> $CKPT_ROOT/encoders/vjepa2_ac"
  CKPT_ROOT="$CKPT_ROOT" uv run --no-sync python - <<'PYX'
from hyperbolic_world_model.models.encoders.vjepa2 import load_vjepa2_ac
enc = load_vjepa2_ac()
print("   encoder embed_dim", enc.embed_dim, "| reference predictor tokens/frame", enc.reference_predictor.tokens_per_frame)
PYX
}

case "$TARGET" in
  vjepa2) dl_vjepa2_ac ;;
  dinov2) dl facebook/dinov2-base dinov2 ;;
  cosmos3) dl nvidia/Cosmos-3-Nano cosmos3-nano ;;
  all)
    dl_vjepa2_ac
    dl facebook/dinov2-base dinov2
    dl nvidia/Cosmos-3-Nano cosmos3-nano
    ;;
  *) echo "unknown target: $TARGET (vjepa2|dinov2|cosmos3|all)" >&2; exit 2 ;;
esac
echo "done. Weights are inference-only; nothing here is ever trained."
