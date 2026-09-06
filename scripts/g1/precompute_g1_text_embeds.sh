#!/usr/bin/env bash
# Cache embeddings for all task instructions in the G1 LeRobot v3 metadata.
set -euo pipefail

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage: bash scripts/g1/precompute_g1_text_embeds.sh [num_gpus=1] [hydra_overrides...]

Examples:
  CUDA_VISIBLE_DEVICES=0 bash scripts/g1/precompute_g1_text_embeds.sh 1
  G1_DATA_ROOT=/data/g1_pnp_pour_v3 bash scripts/g1/precompute_g1_text_embeds.sh 1
  bash scripts/g1/precompute_g1_text_embeds.sh 1 ++overwrite=false

Uses local checkpoints and disables downloads by default. The cache defaults
to ./artifacts/g1_pnp_pour_v3/text_embeds_cache. Set G1_TEXT_CACHE or override
data.train.text_embedding_cache_dir consistently for precompute and training.
See experiments/g1/README.md for the required local text encoder/tokenizer.
HELP
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/g1/common.sh
g1_setup 1 "$@"
shift "$G1_GPU_ARG_COUNT"

exec torchrun --standalone --nproc_per_node="$G1_NPROC" \
  scripts/precompute_text_embeds.py \
  task=g1_pnp_pour_uncond_1cam_320_1e-4 \
  "$@"
