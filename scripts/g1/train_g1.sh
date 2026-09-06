#!/usr/bin/env bash
# Full G1 training with the original FastWAM model and DeepSpeed ZeRO-2.
set -euo pipefail

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage: bash scripts/g1/train_g1.sh [num_gpus=8] [hydra_overrides...]

Examples (from the FastWAM repository):
  bash scripts/g1/train_g1.sh 8
  bash scripts/g1/train_g1.sh 8 task=g1_pnp_pour_joint_1cam_320_1e-4
  CUDA_VISIBLE_DEVICES=2,3 bash scripts/g1/train_g1.sh 2 batch_size=2
  G1_DATA_ROOT=/data/g1_pnp_pour_v3 bash scripts/g1/train_g1.sh 8
  bash scripts/g1/train_g1.sh 8 'data.train.dataset_dirs=[/data/g1_pnp_pour_v3]' max_steps=20000

Precompute text embeddings before training. Local checkpoints default to
./checkpoints; downloads are disabled by default. Caller Hydra overrides win.
See experiments/g1/README.md for model preparation, cache paths, and resume.
HELP
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/g1/common.sh
g1_setup 8 "$@"
shift "$G1_GPU_ARG_COUNT"

exec bash scripts/train_zero2.sh "$G1_NPROC" \
  task=g1_pnp_pour_uncond_1cam_320_1e-4 \
  "$@"
