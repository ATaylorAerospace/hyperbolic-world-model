#!/usr/bin/env bash
# Download frozen encoder weights into ${CKPT_ROOT:-checkpoints}/encoders.
# Reads HF_TOKEN from the environment (or .env). Never downloads datasets.
#
#   V-JEPA 2   facebook/vjepa2-vitl-fpc64-256   (MIT code; weights per model card)
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

case "$TARGET" in
  vjepa2) dl facebook/vjepa2-vitl-fpc64-256 vjepa2 ;;
  dinov2) dl facebook/dinov2-base dinov2 ;;
  cosmos3) dl nvidia/Cosmos-3-Nano cosmos3-nano ;;
  all)
    dl facebook/vjepa2-vitl-fpc64-256 vjepa2
    dl facebook/dinov2-base dinov2
    dl nvidia/Cosmos-3-Nano cosmos3-nano
    ;;
  *) echo "unknown target: $TARGET (vjepa2|dinov2|cosmos3|all)" >&2; exit 2 ;;
esac
echo "done. Weights are inference-only; nothing here is ever trained."
