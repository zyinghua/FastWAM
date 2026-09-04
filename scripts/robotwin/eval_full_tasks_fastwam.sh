#!/usr/bin/env bash
# Usage:
#   bash scripts/robotwin/eval_full_tasks_fastwam.sh \
#     <shard_id:0-16> <gpu_id> <ckpt_path> [dataset_stats_path] \
#     [--skip-completed N] [hydra_overrides...]
# Example: 
# bash scripts/robotwin/eval_full_tasks_fastwam.sh \
#   3 \
#   0 \
#   /workspace/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
#   /workspace/FastWAM/checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
#   --skip-completed 2

set -euo pipefail

usage() {
  echo "Usage: bash $0 <shard_id:0-16> <gpu_id> <ckpt_path> [dataset_stats_path] [--skip-completed N] [hydra_overrides...]" >&2
}

fail() {
  echo "Error: $*" >&2
  exit 2
}

validate_extra_overrides() {
  local override
  local key
  for override in "$@"; do
    [[ "$override" == *=* ]] || \
      fail "extra overrides must use Hydra key=value syntax, got: $override"
    key="${override%%=*}"
    while [[ "$key" == [\+\~]* ]]; do
      key="${key:1}"
    done
    case "$key" in
      task|ckpt|gpu_id|hydra.*|MULTIRUN.*|\
      EVALUATION.dataset_stats_path|EVALUATION.robotwin_root|EVALUATION.policy_name|\
      EVALUATION.task_name|EVALUATION.task_config|EVALUATION.eval_num_episodes|\
      EVALUATION.output_dir|EVALUATION.skip_get_obs_within_replan|EVALUATION.sigma_shift)
        fail "override is controlled by this shard script: $key"
        ;;
    esac
  done
}

find_dataset_stats() {
  local checkpoint="$1"
  local sibling="${checkpoint%.*}_dataset_stats.json"
  local directory
  local depth

  if [[ -f "$sibling" ]]; then
    printf '%s\n' "$sibling"
    return 0
  fi

  directory="$(dirname "$checkpoint")"
  for depth in {1..6}; do
    if [[ -f "$directory/dataset_stats.json" ]]; then
      printf '%s\n' "$directory/dataset_stats.json"
      return 0
    fi
    [[ "$directory" != / ]] || break
    directory="$(dirname "$directory")"
  done
  return 1
}

if [[ $# -lt 3 ]]; then
  usage
  exit 2
fi

SHARD_ID="$1"
GPU_ID="$2"
CKPT="$3"
shift 3

SKIP_COMPLETED=0
SKIP_COMPLETED_SEEN=false
REMAINING_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-completed)
      [[ "$SKIP_COMPLETED_SEEN" == false ]] || fail "--skip-completed was provided more than once"
      [[ $# -ge 2 ]] || fail "--skip-completed requires an integer"
      SKIP_COMPLETED="$2"
      SKIP_COMPLETED_SEEN=true
      shift 2
      ;;
    --skip-completed=*)
      [[ "$SKIP_COMPLETED_SEEN" == false ]] || fail "--skip-completed was provided more than once"
      SKIP_COMPLETED="${1#*=}"
      SKIP_COMPLETED_SEEN=true
      shift
      ;;
    --*)
      fail "unknown option: $1"
      ;;
    *)
      REMAINING_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${REMAINING_ARGS[@]}"

[[ "$SKIP_COMPLETED" =~ ^(0|[1-9][0-9]*)$ ]] || \
  fail "--skip-completed must be a non-negative integer, got: $SKIP_COMPLETED"

[[ "$SHARD_ID" =~ ^([0-9]|1[0-6])$ ]] || \
  fail "shard_id must be an integer from 0 to 16, got: $SHARD_ID"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || \
  fail "gpu_id must be a non-negative integer, got: $GPU_ID"
[[ -f "$CKPT" ]] || fail "checkpoint not found: $CKPT"
CKPT="$(cd "$(dirname "$CKPT")" && pwd -P)/$(basename "$CKPT")"

if [[ $# -ge 1 && "$1" != *=* ]]; then
  DATASET_STATS="$1"
  shift
else
  DATASET_STATS="$(find_dataset_stats "$CKPT")" || \
    fail "could not find dataset stats; pass dataset_stats_path explicitly"
fi
[[ -f "$DATASET_STATS" ]] || fail "dataset stats not found: $DATASET_STATS"
DATASET_STATS="$(cd "$(dirname "$DATASET_STATS")" && pwd -P)/$(basename "$DATASET_STATS")"
validate_extra_overrides "$@"

# Three task names per shard, except shard 16, balanced by evaluation step limits.
TASKS=(
  adjust_bottle beat_block_hammer open_microwave
  click_alarmclock click_bell blocks_ranking_rgb
  grab_roller lift_pot blocks_ranking_size
  move_can_pot move_playingcard_away stack_blocks_three
  move_stapler_pad pick_diverse_bottles stack_bowls_three
  pick_dual_bottles place_bread_skillet hanging_mug
  place_a2b_left place_empty_cup stack_bowls_two
  place_a2b_right dump_bin_bigbin handover_block
  place_container_plate place_fan place_cans_plasticbox
  place_mouse_pad place_object_scale stack_blocks_two
  place_object_stand open_laptop place_bread_basket
  place_phone_stand place_can_basket place_object_basket
  move_pillbottle_pad place_burger_fries put_object_cabinet
  press_stapler place_shoe shake_bottle
  rotate_qrcode scan_object shake_bottle_horizontally
  stamp_seal handover_mic place_dual_shoes
  put_bottles_dustbin turn_switch
)

TASK_OFFSET=$((SHARD_ID * 3))
SHARD_TASKS=("${TASKS[@]:$TASK_OFFSET:3}")
PHASES=(demo_clean demo_randomized)
TOTAL_RUNS=$((${#SHARD_TASKS[@]} * ${#PHASES[@]}))
((SKIP_COMPLETED <= TOTAL_RUNS)) || \
  fail "--skip-completed cannot exceed $TOTAL_RUNS for shard $SHARD_ID"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$REPO_ROOT/third_party/RoboTwin}"
MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$REPO_ROOT/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-python}"
HYDRA_TASK="${FASTWAM_HYDRA_TASK:-robotwin_uncond_3cam_384_1e-4}"

[[ -d "$ROBOTWIN_ROOT" ]] || fail "RoboTwin root not found: $ROBOTWIN_ROOT"
[[ -d "$MODEL_BASE_PATH" ]] || fail "model base path not found: $MODEL_BASE_PATH"
ROBOTWIN_ROOT="$(cd "$ROBOTWIN_ROOT" && pwd -P)"
MODEL_BASE_PATH="$(cd "$MODEL_BASE_PATH" && pwd -P)"

printf -v SHARD_TAG '%02d' "$SHARD_ID"
EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
OUTPUT_LABEL="${EVAL_OUTPUT_LABEL:-fastwam_full_eval_shard_${SHARD_TAG}_${EVAL_RUN_ID}}"
[[ -n "$OUTPUT_LABEL" && "$OUTPUT_LABEL" != */* && "$OUTPUT_LABEL" != "." && "$OUTPUT_LABEL" != ".." ]] || \
  fail "EVAL_OUTPUT_LABEL must be a non-empty path-free label, got: $OUTPUT_LABEL"

cd "$REPO_ROOT"
echo "FastWAM full evaluation shard $SHARD_TAG on GPU $GPU_ID"
echo "Tasks: ${SHARD_TASKS[*]}"
echo "Output label: $OUTPUT_LABEL"
if ((SKIP_COMPLETED > 0)); then
  echo "Skipping the first $SKIP_COMPLETED of $TOTAL_RUNS completed evaluations."
fi

RUN_INDEX=0
for task_name in "${SHARD_TASKS[@]}"; do
  for task_config in "${PHASES[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))
    if ((RUN_INDEX <= SKIP_COMPLETED)); then
      echo "[$RUN_INDEX/$TOTAL_RUNS] skipping completed task=$task_name config=$task_config"
      continue
    fi
    echo "[$RUN_INDEX/$TOTAL_RUNS] task=$task_name config=$task_config"
    env \
      "CUDA_VISIBLE_DEVICES=$GPU_ID" \
      PYTHONDONTWRITEBYTECODE=1 \
      "DIFFSYNTH_MODEL_BASE_PATH=$MODEL_BASE_PATH" \
      DIFFSYNTH_SKIP_DOWNLOAD=true \
      "$PYTHON_BIN" experiments/robotwin/eval_robotwin_single.py \
        "task=$HYDRA_TASK" \
        "ckpt=$CKPT" \
        "EVALUATION.dataset_stats_path=$DATASET_STATS" \
        "EVALUATION.robotwin_root=$ROBOTWIN_ROOT" \
        "EVALUATION.task_name=$task_name" \
        "EVALUATION.task_config=$task_config" \
        EVALUATION.eval_num_episodes=100 \
        EVALUATION.sigma_shift=5.0 \
        EVALUATION.skip_get_obs_within_replan=true \
        "EVALUATION.output_dir=$OUTPUT_LABEL" \
        "gpu_id=$GPU_ID" \
        "$@"
  done
done

echo "Completed FastWAM shard $SHARD_TAG."
