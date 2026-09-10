#!/usr/bin/env bash
# Serve a trained FastWAM G1 checkpoint using the shared robot-client protocol.
set -euo pipefail

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage: bash scripts/g1/serve_g1_policy.sh <checkpoint.pt> [server_options...]

Examples (from the FastWAM repository):
  CUDA_VISIBLE_DEVICES=0 bash scripts/g1/serve_g1_policy.sh runs/RUN_ID/checkpoints/weights/step_000666.pt
  bash scripts/g1/serve_g1_policy.sh /models/g1.pt --config /models/config.yaml --dataset-stats /models/dataset_stats.json
  bash scripts/g1/serve_g1_policy.sh /models/g1.pt --host 127.0.0.1 --port 8001 --num-steps 10

Loads config.yaml and dataset_stats.json from the checkpoint's run by default.
Local Wan checkpoints default to ./checkpoints; downloads are disabled by default.
Trailing server options override the G1 defaults. See experiments/g1/README.md.
HELP
  exit 0
fi
if [[ $# -lt 1 || "$1" == --* ]]; then
  echo "Usage: bash scripts/g1/serve_g1_policy.sh <checkpoint.pt> [server_options...]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
CHECKPOINT="$1"
shift

# --image-key is repeatable; avoid appending the default to caller camera keys.
HAS_IMAGE_KEY=false
for argument in "$@"; do
  if [[ "$argument" == --image-key || "$argument" == --image-key=* ]]; then
    HAS_IMAGE_KEY=true
    break
  fi
done
if [[ "$HAS_IMAGE_KEY" == false ]]; then
  set -- --image-key ego_view "$@"
fi

# Preserve an explicitly selected GPU, including an explicitly empty value.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

exec python scripts/serve.py \
  --checkpoint "$CHECKPOINT" \
  --device cuda:0 \
  --mixed-precision bf16 \
  --num-steps 10 \
  --embodiment unitree_g1_sonic \
  --state-key state \
  --action-key action \
  --fps 10 \
  --execute-horizon 16 \
  --host 0.0.0.0 \
  --port 8000 \
  "$@"
