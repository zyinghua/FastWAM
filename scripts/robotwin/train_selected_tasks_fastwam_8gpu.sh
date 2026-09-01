#!/usr/bin/env bash
# Usage:
#   bash scripts/robotwin/train_selected_tasks_fastwam_8gpu.sh [hydra_overrides...]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$REPO_ROOT/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

exec bash scripts/train_zero2.sh 8 \
  task=robotwin_selected_tasks_uncond_3cam_384_1e-4 \
  "$@"
