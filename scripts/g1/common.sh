#!/usr/bin/env bash
# Shared argument and local-model setup; source from the G1 entrypoints.

g1_setup() {
  local default_nproc="$1"
  shift
  G1_NPROC="$default_nproc"
  G1_GPU_ARG_COUNT=0
  if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    G1_NPROC="$1"
    G1_GPU_ARG_COUNT=1
    shift
  elif [[ $# -gt 0 && "$1" != *=* && "$1" != --* ]]; then
    echo "Error: expected a positive GPU count or Hydra override, got: $1" >&2
    return 2
  fi
  if [[ ! "$G1_NPROC" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_gpus must be a positive integer, got: $G1_NPROC" >&2
    return 2
  fi
  # An explicit CUDA_VISIBLE_DEVICES (including an empty value) is preserved.
  if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    local gpu
    local gpu_ids=()
    for ((gpu = 0; gpu < G1_NPROC; gpu++)); do
      gpu_ids+=("$gpu")
    done
    CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${gpu_ids[*]}")"
  fi
  export CUDA_VISIBLE_DEVICES
  export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
  export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_ROOT}/checkpoints}"
  export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
}
