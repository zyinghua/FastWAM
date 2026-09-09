#!/usr/bin/env python
"""SageMaker container entry point (``SAGEMAKER_PROGRAM``).

Runs ONCE PER HOST — the estimator deliberately does not enable SageMaker's
``torch_distributed``. FastWAM's trainer reads
``accelerator.state.deepspeed_plugin.deepspeed_config`` unconditionally
(src/fastwam/trainer.py), so the Accelerate DeepSpeed plugin must be active;
that only happens when Accelerate is launched with the repo's ``--config_file``.
So this entry reconstructs the repo's own launch line::

    accelerate launch --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
        --num_processes <G*N> --num_machines <N> --machine_rank <R> \
        --main_process_ip algo-1 --main_process_port 29500 \
        scripts/train.py <overrides...>

which is exactly what ``scripts/train_zero2.sh`` does locally, with the
multi-node arguments derived from SageMaker's ``resourceconfig.json``.

Nothing in the FastWAM package is modified: every SageMaker-specific path is
injected as a Hydra override.

Launcher -> container contract (hyperparameters):
  train_argv    shlex-joined user CLI:   ``task=<name> [hydra overrides ...]``
  sm_overrides  shlex-joined path overrides, with placeholders (see expand())
  sm_env        JSON dict of env vars, values may contain the same placeholders

Auto-resume: SageMaker restores ``/opt/ml/checkpoints`` from ``checkpoint_s3_uri``
at job (re)start — after a spot interruption the previous run's checkpoints are
back on disk, but the trainer never scans for them (``cfg.resume`` defaults to
null). So when the resolved ``output_dir`` already contains
``checkpoints/state/step_*`` dirs and no ``resume=`` override was given, the
newest COMPLETE one is injected as ``resume=<dir>`` (see
``_is_complete_state_dir``). Disable with ``FASTWAM_SM_AUTO_RESUME=false``.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sm_env  # noqa: E402

# The repo root inside the image (code is baked in; see sagemaker/Dockerfile).
REPO_DIR = os.environ.get("FASTWAM_REPO_DIR", sm_env.SM_CODE_DIR)
ACCELERATE_CONFIG = "scripts/accelerate_configs/accelerate_zero2_ds.yaml"
TRAIN_SCRIPT = "scripts/train.py"


def expand(template: str, *, task: str) -> str:
    """Resolve the launcher's placeholders against this container's real paths.

    ``{channel:NAME}``  -> /opt/ml/input/data/NAME   (a mounted input channel)
    ``{checkpoint_dir}``-> /opt/ml/checkpoints        (synced to checkpoint_s3_uri)
    ``{pretrained_dir}``-> /opt/ml/pretrained         (weights pulled via aws s3 cp)
    ``{task}``          -> the Hydra task name
    ``{job}``           -> the SageMaker training job name

    Templates (rather than absolute paths) keep the launcher free of any
    knowledge of SageMaker's in-container layout.
    """
    out = template
    while "{channel:" in out:
        head, _, rest = out.partition("{channel:")
        name, _, tail = rest.partition("}")
        out = head + sm_env.channel_path(name) + tail
    return (
        out.replace("{checkpoint_dir}", sm_env.SM_CHECKPOINT_DIR)
        .replace("{pretrained_dir}", sm_env.SM_PRETRAINED_DIR)
        .replace("{task}", task)
        .replace("{job}", sm_env.job_name())
    )


def override_key(token: str) -> str:
    """Hydra override key, ignoring the ``+`` / ``++`` / ``~`` prefixes.

    ``+data.train.foo=1`` and ``data.train.foo=2`` target the same key; Hydra
    rejects the pair, so they must be deduplicated before launch.
    """
    return token.lstrip("+~").split("=", 1)[0]


def merge_overrides(base: list[str], user: list[str]) -> list[str]:
    """Merge Hydra overrides, letting ``user`` win per key, preserving order."""
    merged: dict[str, str] = {}
    for token in [*base, *user]:
        merged[override_key(token)] = token
    return list(merged.values())


def resolve_task(user_argv: list[str]) -> str:
    """Extract ``task=<name>`` from the user CLI (for {task} in path templates)."""
    for token in user_argv:
        if override_key(token) == "task":
            return token.split("=", 1)[1]
    return "train"


def _is_complete_state_dir(path: Path, world_size: int) -> bool:
    """True when a ``step_*`` dir is a fully saved AND fully synced checkpoint.

    Two failure modes to exclude: (a) an interruption mid-save on a host, and
    (b) a partial S3 sync — each host uploads its own /opt/ml/checkpoints
    independently, so any subset of another host's files can be missing on
    restore. trainer_state.json (written last, main process) covers (a); one
    ZeRO ``zero_pp_rank_*_optim_states.pt`` shard per rank covers (b), and as a
    side effect rejects a checkpoint from a run with a different world size,
    which DeepSpeed could not load anyway.
    """
    if not (path / "trainer_state.json").is_file():
        return False
    # Leading * is required: DeepSpeed writes
    # bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt, and a glob is anchored to
    # the whole basename, so "zero_pp_rank_*" matched nothing and auto-resume
    # silently rejected every complete checkpoint.
    shards = list(path.rglob("*zero_pp_rank_*_optim_states.pt"))
    return len(shards) == world_size


def _s3_names(prefix: str, recursive: bool = False) -> list[str]:
    """``aws s3 ls`` a prefix; returns the last column of each line.

    Directories come back as ``name/``; with ``recursive`` the names are keys
    relative to the bucket. Returns [] on any error, so a missing prefix or an
    expired credential degrades to "no checkpoint" rather than killing the job.
    """
    cmd = ["aws", "s3", "ls", prefix.rstrip("/") + "/"]
    if recursive:
        cmd.append("--recursive")
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return []
    if done.returncode != 0:
        return []
    return [line.split()[-1] for line in done.stdout.splitlines() if line.strip()]


def _s3_state_is_complete(step_prefix: str, world_size: int) -> bool:
    """Same completeness rule as :func:`_is_complete_state_dir`, against S3."""
    names = _s3_names(step_prefix, recursive=True)
    if not any(n.endswith("trainer_state.json") for n in names):
        return False
    # Mirrors the glob in _is_complete_state_dir. Note the real filename is
    # bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt — the rank index is NOT
    # adjacent to the suffix, so the wildcard has to span _mp_rank_NN.
    shards = [n for n in names if re.search(r"zero_pp_rank_.*optim_states\.pt$", n)]
    return len(shards) == world_size


def fetch_resume_from_s3(overrides: list[str], world_size: int) -> str | None:
    """Pull the newest complete state checkpoint straight from S3.

    Why this exists: SageMaker restores ``checkpoint_s3_uri`` into
    /opt/ml/checkpoints in the BACKGROUND. The step_* directories and their
    small files appear almost immediately, but the ~9 GB-per-rank optimizer
    shards stream in for minutes afterwards, so :func:`find_auto_resume` runs
    at container start and sees an incomplete tree — measured on a real spot
    restart, which then retrained from step 0. Downloading the one checkpoint
    we actually need is deterministic and finishes before training starts.

    Returns the local dir to resume from, or None to start fresh. Any failure
    returns None: a fresh start is always preferable to a crashed job.
    """
    if os.environ.get(f"{sm_env.ENV_PREFIX}_AUTO_RESUME", "true").lower() == "false":
        return None
    base = os.environ.get(f"{sm_env.ENV_PREFIX}_CHECKPOINT_S3")
    if not base:
        return None

    output_dir: str | None = None
    for token in overrides:
        key = override_key(token)
        if key == "resume":
            return None
        if key == "output_dir":
            output_dir = token.split("=", 1)[1]
    if not output_dir:
        return None
    try:
        rel = Path(output_dir).resolve().relative_to(sm_env.SM_CHECKPOINT_DIR)
    except ValueError:
        # output_dir lives outside /opt/ml/checkpoints, so it is not mirrored.
        return None

    s3_state_root = f"{base.rstrip('/')}/{rel.as_posix()}/checkpoints/state"
    steps: list[int] = []
    for name in _s3_names(s3_state_root):
        match = re.fullmatch(r"step_(\d+)/", name)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        return None

    for step in sorted(steps, reverse=True):
        src = f"{s3_state_root}/step_{step:06d}"
        if not _s3_state_is_complete(src, world_size):
            print(f"[entry] s3 checkpoint step_{step:06d} incomplete, trying older", flush=True)
            continue
        dst = Path(output_dir) / "checkpoints" / "state" / f"step_{step:06d}"
        print(f"[entry] fetching resume checkpoint {src} -> {dst}", flush=True)
        try:
            dst.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["aws", "s3", "cp", "--recursive", "--only-show-errors",
                 src + "/", str(dst) + "/"],
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[entry] resume fetch failed ({exc}); starting fresh", flush=True)
            return None
        if not _is_complete_state_dir(dst, world_size):
            print("[entry] fetched checkpoint still incomplete; starting fresh", flush=True)
            return None
        return str(dst)
    return None


def find_auto_resume(overrides: list[str], world_size: int) -> str | None:
    """Newest complete state checkpoint under the job's output_dir, if any.

    Returns the ``.../checkpoints/state/step_NNNNNN`` dir with the highest step
    that passes ``_is_complete_state_dir``, or None when auto-resume is
    disabled, a ``resume=`` override is already present, or no complete state
    checkpoint exists (the normal fresh-start case).
    """
    if os.environ.get(f"{sm_env.ENV_PREFIX}_AUTO_RESUME", "true").lower() == "false":
        return None
    output_dir: str | None = None
    for token in overrides:
        key = override_key(token)
        if key == "resume":
            return None  # an explicit resume choice always wins
        if key == "output_dir":
            output_dir = token.split("=", 1)[1]
    if not output_dir:
        return None

    state_root = Path(output_dir) / "checkpoints" / "state"
    if not state_root.is_dir():
        return None
    steps = []
    for path in state_root.iterdir():
        match = re.fullmatch(r"step_(\d+)", path.name)
        if match and path.is_dir():
            steps.append((int(match.group(1)), path))
    for _, path in sorted(steps, reverse=True):
        if _is_complete_state_dir(path, world_size):
            return str(path)
        print(
            f"[entry] skipping incomplete/mismatched checkpoint {path.name} "
            f"(needs trainer_state.json + {world_size} optimizer shards)",
            flush=True,
        )
    return None


def main() -> None:
    if not sm_env.is_sagemaker():
        raise SystemExit(
            "sagemaker/entry.py is the in-container entry point; it found no "
            "SageMaker environment. Run scripts/train.py directly for local training."
        )

    hp = sm_env.read_hyperparameters()
    if "train_argv" not in hp:
        raise SystemExit(
            "Missing the 'train_argv' hyperparameter. launch_sm.py packs the "
            "training CLI via shlex.join(['task=<name>', *overrides])."
        )

    user_argv = shlex.split(hp["train_argv"])
    task = resolve_task(user_argv)

    # Env from the launcher's target config (e.g. DIFFSYNTH_MODEL_BASE_PATH).
    for key, value in json.loads(hp.get("sm_env", "{}")).items():
        os.environ[key] = expand(str(value), task=task)

    # Pretrained weights that are NOT mounted as a channel get pulled from S3.
    pretrained_s3 = os.environ.get(f"{sm_env.ENV_PREFIX}_PRETRAINED_S3")
    if pretrained_s3:
        sm_env.sync_from_s3(pretrained_s3, sm_env.SM_PRETRAINED_DIR)

    sm_overrides = [
        expand(token, task=task) for token in shlex.split(hp.get("sm_overrides", ""))
    ]
    overrides = merge_overrides(sm_overrides, user_argv)

    dist = sm_env.distributed_env()
    world_size = dist["nproc_per_node"] * dist["num_machines"]
    # Prefer whatever SageMaker's restore already delivered (no download); fall
    # back to pulling the checkpoint from S3 ourselves when that tree is still
    # incomplete, which is the normal case on a spot restart.
    resume_dir = find_auto_resume(overrides, world_size)
    if resume_dir:
        print(f"[entry] auto-resume from restored checkpoint: {resume_dir}", flush=True)
    else:
        resume_dir = fetch_resume_from_s3(overrides, world_size)
    if resume_dir:
        overrides.append(f"resume={resume_dir}")

    cmd = [
        "accelerate", "launch",
        "--config_file", ACCELERATE_CONFIG,
        "--num_processes", str(dist["nproc_per_node"] * dist["num_machines"]),
    ]
    if dist["num_machines"] > 1:
        cmd += [
            "--num_machines", str(dist["num_machines"]),
            "--machine_rank", str(dist["machine_rank"]),
            "--main_process_ip", dist["main_process_ip"],
            "--main_process_port", str(dist["main_process_port"]),
            # `standard` lets Accelerate launch the workers on every host.
            "--deepspeed_multinode_launcher",
            os.environ.get(f"{sm_env.ENV_PREFIX}_DS_LAUNCHER", "standard"),
        ]
    cmd += [TRAIN_SCRIPT, *overrides]

    os.makedirs(sm_env.SM_CHECKPOINT_DIR, exist_ok=True)
    os.chdir(REPO_DIR)

    print(f"[entry] cwd={REPO_DIR}", flush=True)
    print(f"[entry] channels={sm_env.channel_map()}", flush=True)
    print(f"[entry] dist={dist}", flush=True)
    print(f"[entry] {shlex.join(cmd)}", flush=True)

    # Keep checkpoint_s3_uri small enough that SageMaker can restore it on a
    # spot restart. Nothing prunes checkpoints otherwise, and past ~369 GB the
    # restore fails with an opaque InternalServerError before the container
    # starts (measured; see prune_checkpoints.py). Main host only, and every
    # failure inside is swallowed — this must never be able to kill training.
    if dist["machine_rank"] == 0 and os.environ.get(
        f"{sm_env.ENV_PREFIX}_PRUNE_CHECKPOINTS", "true"
    ).lower() != "false":
        out_dir = next(
            (t.split("=", 1)[1] for t in overrides if override_key(t) == "output_dir"), None
        )
        if out_dir:
            try:
                import prune_checkpoints

                prune_checkpoints.spawn(
                    out_dir,
                    os.environ.get(f"{sm_env.ENV_PREFIX}_CHECKPOINT_S3", ""),
                    keep=int(os.environ.get(f"{sm_env.ENV_PREFIX}_PRUNE_KEEP", "1")),
                )
            except Exception as exc:
                print(f"[entry] checkpoint pruner not started ({exc}); continuing", flush=True)

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
