#!/usr/bin/env bash
# Thin positional wrapper around sagemaker/launch_sm.py.
#
#   bash sagemaker/run_sm.sh <CONFIG> <INSTANCE_COUNT> <NAME> [HYDRA OVERRIDES...]
#   bash sagemaker/run_sm.sh g1 1 bottle \
#       'data.dataset_dirs=[/opt/ml/input/data/g1/bottle]'
#
# Env:
#   SKIP_BUILD=1   reuse the pushed ECR image (no docker build/push)
#   BUILD_ONLY=1   build + push and exit (run this on the docker host)
#   DRY_RUN=1      print the job summary without submitting
#   QUEUE=<alias>  override the target YAML's queue
#   SM_USER        job-name prefix (default: $USER)
#
# Pull in secrets (WANDB_API_KEY etc.) — launch_sm.py forwards WANDB_*/HF_TOKEN
# from the host env into the job's environment.
set -a; [ -f .env ] && source .env; set +a

# A stale AWS_PROFILE in the shell would silently override --profile.
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE

set -euo pipefail

CONFIG="${1:?usage: run_sm.sh <CONFIG> <INSTANCE_COUNT> <NAME> [overrides...]}"
INSTANCE_COUNT="${2:-1}"
NAME="${3:-}"
shift 3 2>/dev/null || shift $#
OVERRIDES=("$@")

ACCOUNT="${ACCOUNT:-robotics-old}"
REGION="${REGION:-us-west-2}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
export SM_USER="${SM_USER:-${USER:-user}}"

EXTRA=()
[ "${SKIP_BUILD:-0}" = 1 ] && EXTRA+=(--skip-build)
[ "${BUILD_ONLY:-0}" = 1 ] && EXTRA+=(--build-only)
[ "${DRY_RUN:-0}" = 1 ]    && EXTRA+=(--dry-run)
[ -n "${NAME}" ]           && EXTRA+=(--name "${NAME}")
[ -n "${QUEUE:-}" ]        && EXTRA+=(--queue "${QUEUE}")
# Spot queues are rejected unless the job also asks for spot instances.
case "${QUEUE:-}" in *spot*) EXTRA+=(--spot);; esac

PY="${PY:-python3}"

AWS_DEFAULT_REGION="${REGION}" \
  "${PY}" sagemaker/launch_sm.py \
  --config "${CONFIG}" \
  --account "${ACCOUNT}" \
  --image-tag "${IMAGE_TAG}" \
  --instance-count "${INSTANCE_COUNT}" \
  --region "${REGION}" \
  "${EXTRA[@]}" \
  -- "${OVERRIDES[@]}"
