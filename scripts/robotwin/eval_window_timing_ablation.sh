#!/usr/bin/env bash
# Time FastWAM and a FastWAM-Joint proxy over equivalent W=1 through 8 horizons.
# The joint path reuses the FastWAM-trained checkpoint; it is not an accuracy evaluation.

set -euo pipefail

usage() {
  echo "Usage: bash $0 <gpu_id> <fastwam_ckpt> [task_name=beat_block_hammer] [episodes=10] [result_dir]" >&2
}

fail() {
  echo "Error: $*" >&2
  exit 2
}

if [[ $# -lt 2 || $# -gt 5 ]]; then
  usage
  exit 2
fi

GPU_ID="$1"
FASTWAM_CKPT="$2"
TASK_NAME="${3:-beat_block_hammer}"
EPISODES="${4:-10}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-0}"

[[ "$GPU_ID" =~ ^[0-9]+$ ]] || fail "gpu_id must be a non-negative integer, got: $GPU_ID"
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || fail "episodes must be a positive integer, got: $EPISODES"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fail "DRY_RUN must be 0 or 1, got: $DRY_RUN"
[[ "$SEED" =~ ^[0-9]+$ ]] || fail "SEED must be a non-negative integer, got: $SEED"
[[ -n "$TASK_NAME" ]] || fail "task_name must not be empty"
[[ -f "$FASTWAM_CKPT" ]] || fail "FastWAM checkpoint not found: $FASTWAM_CKPT"

FASTWAM_CKPT="$(cd "$(dirname "$FASTWAM_CKPT")" && pwd -P)/$(basename "$FASTWAM_CKPT")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TIMING_INIT="$REPO_ROOT/experiments/robotwin/fastwam_policy/__init__.py"

if ! grep -Eq '^[[:space:]]*from[[:space:]]+\.deploy_policy_timing[[:space:]]+import[[:space:]]+\*[[:space:]]*$' "$TIMING_INIT"; then
  fail "timing policy is not wired. Set $TIMING_INIT to: from .deploy_policy_timing import *"
fi

find_dataset_stats() {
  local directory
  local depth
  directory="$(dirname "$1")"
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

if [[ -n "${FASTWAM_DATASET_STATS:-}" ]]; then
  [[ -f "$FASTWAM_DATASET_STATS" ]] || fail "dataset stats not found: $FASTWAM_DATASET_STATS"
  FASTWAM_DATASET_STATS="$(cd "$(dirname "$FASTWAM_DATASET_STATS")" && pwd -P)/$(basename "$FASTWAM_DATASET_STATS")"
else
  FASTWAM_DATASET_STATS="$(find_dataset_stats "$FASTWAM_CKPT")" || \
    fail "could not find dataset_stats.json above the checkpoint; set FASTWAM_DATASET_STATS"
fi

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$REPO_ROOT/third_party/RoboTwin}"
MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$REPO_ROOT/checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FASTWAM_HYDRA_TASK="${FASTWAM_HYDRA_TASK:-robotwin_selected_tasks_uncond_3cam_384_1e-4}"
FASTWAM_JOINT_HYDRA_TASK="${FASTWAM_JOINT_HYDRA_TASK:-robotwin_joint_3cam_384_1e-4}"

[[ -d "$ROBOTWIN_ROOT" ]] || fail "RoboTwin root not found: $ROBOTWIN_ROOT"
[[ -d "$MODEL_BASE_PATH" ]] || fail "model base path not found: $MODEL_BASE_PATH"
ROBOTWIN_ROOT="$(cd "$ROBOTWIN_ROOT" && pwd -P)"
MODEL_BASE_PATH="$(cd "$MODEL_BASE_PATH" && pwd -P)"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
if [[ $# -ge 5 ]]; then
  RESULT_DIR="$5"
else
  RESULT_DIR="$REPO_ROOT/evaluate_results/robotwin_timing/fastwam_${RUN_ID}"
fi
mkdir -p "$RESULT_DIR/fastwam" "$RESULT_DIR/fastwam_joint"
RESULT_DIR="$(cd "$RESULT_DIR" && pwd -P)"

cd "$REPO_ROOT"

echo "FastWAM timing sweep: task=$TASK_NAME episodes=$EPISODES seed=$SEED gpu=$GPU_ID"
echo "Schedule: K=floor(16/W); horizon=16*W; joint_video_frames=4*W+1"
echo "Checkpoint shared by both paths: $FASTWAM_CKPT"
echo "Results: $RESULT_DIR"

for window in {1..8}; do
  inference_steps=$((16 / window))
  action_horizon=$((16 * window))
  num_video_frames=$((4 * window + 1))
  run_name="w${window}_h${action_horizon}_k${inference_steps}"

  fastwam_json="$RESULT_DIR/fastwam/$run_name.json"
  fastwam_log="$RESULT_DIR/fastwam/$run_name.log"
  joint_json="$RESULT_DIR/fastwam_joint/$run_name.json"
  joint_log="$RESULT_DIR/fastwam_joint/$run_name.log"
  fastwam_eval_name="timing_fastwam_${run_name}_${RUN_ID}"
  joint_eval_name="timing_fastwam_joint_${run_name}_${RUN_ID}"

  fastwam_command=(
    "$PYTHON_BIN" experiments/robotwin/eval_robotwin_single.py
    "task=$FASTWAM_HYDRA_TASK"
    "ckpt=$FASTWAM_CKPT"
    "EVALUATION.dataset_stats_path=$FASTWAM_DATASET_STATS"
    "EVALUATION.robotwin_root=$ROBOTWIN_ROOT"
    "EVALUATION.task_name=$TASK_NAME"
    EVALUATION.task_config=demo_clean
    "EVALUATION.eval_num_episodes=$EPISODES"
    "EVALUATION.action_horizon=$action_horizon"
    EVALUATION.replan_steps=16
    "EVALUATION.num_inference_steps=$inference_steps"
    EVALUATION.timing_enabled=true
    EVALUATION.skip_get_obs_within_replan=true
    "seed=$SEED"
    "EVALUATION.output_dir=$fastwam_eval_name"
    "gpu_id=$GPU_ID"
  )

  joint_command=(
    "$PYTHON_BIN" experiments/robotwin/eval_robotwin_single.py
    "task=$FASTWAM_JOINT_HYDRA_TASK"
    "ckpt=$FASTWAM_CKPT"
    "EVALUATION.dataset_stats_path=$FASTWAM_DATASET_STATS"
    "EVALUATION.robotwin_root=$ROBOTWIN_ROOT"
    "EVALUATION.task_name=$TASK_NAME"
    EVALUATION.task_config=demo_clean
    "EVALUATION.eval_num_episodes=$EPISODES"
    "EVALUATION.action_horizon=$action_horizon"
    "EVALUATION.num_video_frames=$num_video_frames"
    EVALUATION.replan_steps=16
    "EVALUATION.num_inference_steps=$inference_steps"
    EVALUATION.timing_enabled=true
    EVALUATION.skip_get_obs_within_replan=true
    "seed=$SEED"
    "EVALUATION.output_dir=$joint_eval_name"
    "gpu_id=$GPU_ID"
  )

  echo
  echo "[FastWAM W=$window] horizon=$action_horizon steps=$inference_steps"
  printf '  '
  printf '%q ' env \
    "DIFFSYNTH_MODEL_BASE_PATH=$MODEL_BASE_PATH" \
    DIFFSYNTH_SKIP_DOWNLOAD=true \
    "WAM_TIMING_RESULT_PATH=$fastwam_json" \
    FASTWAM_TIMING_EXPECTED_MODEL_CLASS=FastWAM \
    "${fastwam_command[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "0" ]]; then
    env \
      "DIFFSYNTH_MODEL_BASE_PATH=$MODEL_BASE_PATH" \
      DIFFSYNTH_SKIP_DOWNLOAD=true \
      "WAM_TIMING_RESULT_PATH=$fastwam_json" \
      FASTWAM_TIMING_EXPECTED_MODEL_CLASS=FastWAM \
      "${fastwam_command[@]}" 2>&1 | tee "$fastwam_log"

    [[ -f "$fastwam_json" ]] || fail "FastWAM timing result was not written for W=$window: $fastwam_json"
  fi

  echo
  echo "[FastWAM-Joint proxy W=$window] horizon=$action_horizon video_frames=$num_video_frames steps=$inference_steps"
  printf '  '
  printf '%q ' env \
    "DIFFSYNTH_MODEL_BASE_PATH=$MODEL_BASE_PATH" \
    DIFFSYNTH_SKIP_DOWNLOAD=true \
    "WAM_TIMING_RESULT_PATH=$joint_json" \
    FASTWAM_TIMING_EXPECTED_MODEL_CLASS=FastWAMJoint \
    "${joint_command[@]}"
  printf '\n'

  if [[ "$DRY_RUN" == "0" ]]; then
    env \
      "DIFFSYNTH_MODEL_BASE_PATH=$MODEL_BASE_PATH" \
      DIFFSYNTH_SKIP_DOWNLOAD=true \
      "WAM_TIMING_RESULT_PATH=$joint_json" \
      FASTWAM_TIMING_EXPECTED_MODEL_CLASS=FastWAMJoint \
      "${joint_command[@]}" 2>&1 | tee "$joint_log"

    [[ -f "$joint_json" ]] || fail "FastWAM-Joint timing result was not written for W=$window: $joint_json"
  fi
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry run complete; no evaluations or summary were written."
  exit 0
fi

"$PYTHON_BIN" - "$RESULT_DIR" "$TASK_NAME" "$EPISODES" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
task_name = sys.argv[2]
episodes = int(sys.argv[3])
seed = int(sys.argv[4])
step_schedule = [16, 8, 5, 4, 3, 2, 2, 2]


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def read_result(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def optional_mean(result):
    value = result.get("aggregate", {}).get("steady_state", {}).get("mean_ms")
    return None if value is None else float(value)


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


runs = []
shared_checkpoint = None
for window, inference_steps in enumerate(step_schedule, start=1):
    action_horizon = 16 * window
    num_video_frames = 4 * window + 1
    run_name = f"w{window}_h{action_horizon}_k{inference_steps}"
    fastwam_path = result_dir / "fastwam" / f"{run_name}.json"
    joint_path = result_dir / "fastwam_joint" / f"{run_name}.json"
    fastwam = read_result(fastwam_path)
    joint = read_result(joint_path)

    require_equal(fastwam.get("policy"), "fastwam", f"W={window} FastWAM policy")
    require_equal(fastwam.get("model", {}).get("class"), "FastWAM", f"W={window} FastWAM class")
    require_equal(fastwam.get("model", {}).get("action_horizon"), action_horizon, f"W={window} FastWAM horizon")
    require_equal(fastwam.get("model", {}).get("executed_actions_per_replan"), 16, f"W={window} FastWAM executed actions")
    require_equal(fastwam.get("model", {}).get("num_inference_steps"), inference_steps, f"W={window} FastWAM steps")
    require_equal(fastwam.get("model", {}).get("denoising_steps_per_replan"), inference_steps, f"W={window} FastWAM denoising steps")

    require_equal(joint.get("policy"), "fastwam", f"W={window} FastWAM-Joint policy")
    require_equal(joint.get("model", {}).get("class"), "FastWAMJoint", f"W={window} FastWAM-Joint class")
    require_equal(joint.get("model", {}).get("action_horizon"), action_horizon, f"W={window} FastWAM-Joint horizon")
    require_equal(joint.get("model", {}).get("num_video_frames"), num_video_frames, f"W={window} FastWAM-Joint video frames")
    require_equal(joint.get("model", {}).get("executed_actions_per_replan"), 16, f"W={window} FastWAM-Joint executed actions")
    require_equal(joint.get("model", {}).get("num_inference_steps"), inference_steps, f"W={window} FastWAM-Joint steps")
    require_equal(joint.get("model", {}).get("denoising_steps_per_replan"), inference_steps, f"W={window} FastWAM-Joint denoising steps")
    require_equal(joint.get("checkpoint"), fastwam.get("checkpoint"), f"W={window} shared checkpoint")
    if shared_checkpoint is None:
        shared_checkpoint = fastwam.get("checkpoint")
    require_equal(fastwam.get("checkpoint"), shared_checkpoint, f"W={window} checkpoint")

    require_equal(fastwam.get("aggregate", {}).get("episodes"), episodes, f"W={window} FastWAM episodes")
    require_equal(joint.get("aggregate", {}).get("episodes"), episodes, f"W={window} FastWAM-Joint episodes")

    fastwam_mean = optional_mean(fastwam)
    joint_mean = optional_mean(joint)
    runs.append(
        {
            "equivalent_window_blocks": window,
            "action_horizon": action_horizon,
            "num_inference_steps": inference_steps,
            "fastwam": {
                "result_file": str(fastwam_path.relative_to(result_dir)),
                "model": fastwam.get("model"),
                "aggregate": fastwam.get("aggregate"),
            },
            "fastwam_joint": {
                "result_file": str(joint_path.relative_to(result_dir)),
                "model": joint.get("model"),
                "aggregate": joint.get("aggregate"),
                "weights": "FastWAM-trained checkpoint; joint inference timing proxy only",
            },
            "steady_state_comparison": {
                "fastwam_mean_ms": fastwam_mean,
                "fastwam_joint_mean_ms": joint_mean,
                "fastwam_joint_slowdown_vs_fastwam": ratio(joint_mean, fastwam_mean),
                "fastwam_ms_per_denoiser_call": (
                    None if fastwam_mean is None else fastwam_mean / inference_steps
                ),
                "fastwam_joint_ms_per_denoiser_call": (
                    None if joint_mean is None else joint_mean / inference_steps
                ),
            },
        }
    )

summary = {
    "schema_version": 1,
    "comparison": "fastwam_vs_fastwam_joint_proxy",
    "unit": "ms",
    "task": task_name,
    "requested_episodes_per_window": episodes,
    "seed": seed,
    "checkpoint": shared_checkpoint,
    "executed_actions_per_replan": 16,
    "schedule_rule": "K=floor(16/W); both FastWAM variants use S=K",
    "fastwam_joint_weights": "FastWAM-trained checkpoint; timing proxy only",
    "runs": runs,
}
summary_path = result_dir / "summary.json"
temporary_path = summary_path.with_suffix(".json.tmp")
with temporary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
temporary_path.replace(summary_path)
print(f"Summary written to: {summary_path}")
PY

echo "FastWAM timing sweep complete."
