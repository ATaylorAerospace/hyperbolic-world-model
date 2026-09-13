#!/usr/bin/env bash
# Regenerate every table and figure in outputs/report from outputs/**/metrics.json.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --no-sync python -m hyperbolic_world_model.reporting.make_report --outputs "${1:-outputs}" --report "${2:-outputs/report}"
