"""SageMaker in-container runtime helpers.

Pure stdlib — no torch, no boto3/SageMaker SDK. These run INSIDE a SageMaker
training job. The host-side launcher (``launch_sm.py``) lives alongside in this
folder but is never imported here (control plane vs. runtime).

Nothing in the FastWAM package imports this module: the SageMaker integration
is bolted on from the outside, and every path it computes is handed to
``scripts/train.py`` as a Hydra override.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

# SageMaker well-known container paths.
SM_RESOURCECONFIG = Path("/opt/ml/input/config/resourceconfig.json")
SM_HYPERPARAMS = Path("/opt/ml/input/config/hyperparameters.json")
SM_CHECKPOINT_DIR = "/opt/ml/checkpoints"
SM_INPUT_DIR = "/opt/ml/input/data"
SM_CODE_DIR = "/opt/ml/code"
# Local landing dir for weights pulled from S3 (the `s3sync` pretrained source).
SM_PRETRAINED_DIR = "/opt/ml/pretrained"

# Env-var namespace for the launcher -> container contract.
ENV_PREFIX = "FASTWAM_SM"


def is_sagemaker() -> bool:
    """True inside a SageMaker training container."""
    return "SM_HOSTS" in os.environ or SM_RESOURCECONFIG.exists()


def read_hyperparameters() -> dict:
    """Read ``/opt/ml/input/config/hyperparameters.json``.

    The SDK may JSON-encode string values a second time (e.g.
    ``{"train_argv": "\\"task=x batch_size=2\\""}``); decode each value, falling
    back to the raw string when it isn't valid JSON or decodes to a non-str.
    """
    if not SM_HYPERPARAMS.exists():
        return {}
    raw: dict = json.loads(SM_HYPERPARAMS.read_text())
    out: dict = {}
    for k, v in raw.items():
        if isinstance(v, str):
            try:
                decoded = json.loads(v)
                out[k] = decoded if isinstance(decoded, str) else v
            except (json.JSONDecodeError, ValueError):
                out[k] = v
        else:
            out[k] = v
    return out


def channel_map() -> dict[str, str]:
    """Read the launcher's ``FASTWAM_SM_INPUT_*`` contract -> ``{channel: mount}``.

    Empty when no channels were mounted (e.g. a purely local run).
    """
    count = int(os.environ.get(f"{ENV_PREFIX}_INPUT_CHANNEL_COUNT", "0"))
    mapping: dict[str, str] = {}
    for i in range(count):
        channel = os.environ.get(f"{ENV_PREFIX}_INPUT_CHANNEL_{i:03d}")
        if channel:
            mapping[channel] = os.path.join(SM_INPUT_DIR, channel)
    return mapping


def channel_path(channel: str, *parts: str) -> str:
    """Absolute path of ``parts`` inside a mounted input ``channel``."""
    return os.path.join(SM_INPUT_DIR, channel, *parts)


def job_name() -> str:
    """The SageMaker training job name.

    SageMaker exposes it inside ``SM_TRAINING_ENV`` (a JSON blob), not as a
    variable of its own; ``TRAINING_JOB_NAME`` is the older toolkit spelling.
    """
    blob = os.environ.get("SM_TRAINING_ENV")
    if blob:
        try:
            name = json.loads(blob).get("job_name")
            if name:
                return str(name)
        except (json.JSONDecodeError, AttributeError):
            pass
    return os.environ.get("TRAINING_JOB_NAME", "job")


def resource_config() -> dict:
    """Parse ``resourceconfig.json`` (hosts + current host), with env fallbacks."""
    if SM_RESOURCECONFIG.exists():
        return json.loads(SM_RESOURCECONFIG.read_text())
    hosts = json.loads(os.environ.get("SM_HOSTS", '["algo-1"]'))
    return {
        "hosts": hosts,
        "current_host": os.environ.get("SM_CURRENT_HOST", hosts[0]),
    }


def distributed_env(master_port: int = 29500) -> dict:
    """Derive the multi-node launch parameters from SageMaker's resource config.

    SageMaker names hosts ``algo-1..algo-N`` and resolves them over the training
    cluster's private network; ``algo-1`` is the rendezvous master. Returns the
    values ``accelerate launch`` needs for multi-node DeepSpeed.
    """
    cfg = resource_config()
    hosts = sorted(cfg["hosts"])
    current = cfg["current_host"]
    nproc = int(os.environ.get("SM_NUM_GPUS") or _visible_gpu_count())
    return {
        "num_machines": len(hosts),
        "machine_rank": hosts.index(current),
        "main_process_ip": hosts[0],
        "main_process_port": int(os.environ.get("MASTER_PORT", master_port)),
        "nproc_per_node": nproc,
    }


def _visible_gpu_count() -> int:
    """GPU count without importing torch (nvidia-smi, then a conservative 1)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--list-gpus"], capture_output=True, text=True, check=True
        ).stdout
        return max(1, len([ln for ln in out.splitlines() if ln.strip()]))
    except (OSError, subprocess.CalledProcessError):
        return 1


def sync_from_s3(s3_uri: str, local: str) -> str:
    """``aws s3 cp --recursive`` a prefix into ``local``; returns ``local``.

    Only the first process to arrive downloads; later arrivals block on a
    sentinel. Used for pretrained weights when they are not mounted as a
    channel. Safe to call once per host (the entry runs once per host).
    """
    dest = Path(local)
    sentinel = dest / ".sync_done"
    if sentinel.exists():
        return local
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "cp", "--recursive", "--only-show-errors",
        s3_uri.rstrip("/") + "/", str(dest) + "/",
    ]
    print(f"[sm_env] {shlex.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    sentinel.touch()
    return local


def wait_for(path: str, timeout_s: int = 1800, poll_s: int = 5) -> None:
    """Block until ``path`` exists (used to wait on another rank's download)."""
    deadline = time.time() + timeout_s
    while not Path(path).exists():
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(poll_s)

