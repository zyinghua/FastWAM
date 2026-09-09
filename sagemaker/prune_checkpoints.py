"""Keep the SageMaker checkpoint prefix small enough to restore.

Why this exists
---------------
SageMaker restores the ENTIRE ``checkpoint_s3_uri`` prefix into
/opt/ml/checkpoints *before* the container starts. Past some size that restore
fails outright, and the job dies with a completely opaque

    InternalServerError: We encountered an internal error. Please try again.

with ZERO CloudWatch log streams, because the container never ran. Measured on
the libero job (93 GB per checkpoint):

    92 GB  (1 checkpoint)   restore OK, container starts, auto-resume works
    369 GB (4 checkpoints)  InternalServerError, container never starts

The trainer has no retention policy, so any long run walks itself into this.

Policy
------
Newest checkpoint stays in ``checkpoint_s3_uri`` (small restore). The previous
one is MOVED to a sibling ``<prefix>-archive/`` — outside the restore path, so
it costs nothing at startup, but is still there if the newest turns out to have
been cut mid-write by a spot reclaim. Without that spare, a corrupt newest makes
auto-resume reject everything and silently restart from step 0.

Anything older than that is deleted. Local copies under /opt/ml/checkpoints are
removed too, otherwise SageMaker's sync just re-uploads what we pruned.

Runs as a forked daemon from entry.py (see ``spawn``), on the main host only.
Every failure is swallowed: pruning must never be able to kill training.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

STEP_RE = re.compile(r"^step_(\d+)/?$")


def _run(cmd: list[str], timeout: int = 900) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)


def s3_steps(prefix: str) -> list[int]:
    """Step numbers of the ``step_NNNNNN/`` dirs directly under ``prefix``."""
    rc, out = _run(["aws", "s3", "ls", prefix.rstrip("/") + "/"])
    if rc != 0:
        return []
    steps = []
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == "PRE":
            m = STEP_RE.match(parts[-1])
            if m:
                steps.append(int(m.group(1)))
    return sorted(steps)


def prune_once(state_s3: str, weights_s3: str, archive_s3: str, local_state: Path,
               local_weights: Path, keep: int = 1, verbose: bool = True) -> None:
    """Keep ``keep`` newest in place, archive the next, delete the rest."""
    steps = s3_steps(state_s3)
    if len(steps) <= keep:
        return
    newest = steps[-keep:]
    to_archive = steps[-(keep + 1):-keep]      # exactly one spare
    to_delete = steps[:-(keep + 1)]

    def say(msg: str) -> None:
        if verbose:
            print(f"[prune] {msg}", flush=True)

    for step in to_archive:
        name = f"step_{step:06d}"
        say(f"archiving {name} -> {archive_s3}")
        _run(["aws", "s3", "mv", f"{state_s3.rstrip('/')}/{name}/",
              f"{archive_s3.rstrip('/')}/state/{name}/", "--recursive", "--only-show-errors"])
        _run(["aws", "s3", "mv", f"{weights_s3.rstrip('/')}/{name}.pt",
              f"{archive_s3.rstrip('/')}/weights/{name}.pt", "--only-show-errors"])

    for step in to_delete:
        name = f"step_{step:06d}"
        say(f"deleting {name}")
        _run(["aws", "s3", "rm", f"{state_s3.rstrip('/')}/{name}/", "--recursive", "--only-show-errors"])
        _run(["aws", "s3", "rm", f"{weights_s3.rstrip('/')}/{name}.pt", "--only-show-errors"])

    # Drop the local copies too, or the checkpoint sync re-uploads them.
    for step in [*to_archive, *to_delete]:
        name = f"step_{step:06d}"
        try:
            d = local_state / name
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
            f = local_weights / f"{name}.pt"
            if f.is_file():
                f.unlink()
        except OSError as exc:
            say(f"local cleanup of {name} failed (harmless): {exc}")

    if to_archive or to_delete:
        say(f"kept {newest}, archived {to_archive}, deleted {to_delete}")


def loop(output_dir: str, checkpoint_s3: str, keep: int, interval: int) -> None:
    rel = ""
    try:
        rel = str(Path(output_dir).resolve().relative_to("/opt/ml/checkpoints"))
    except ValueError:
        return  # output_dir is not inside the mirrored dir; nothing to prune
    base = f"{checkpoint_s3.rstrip('/')}/{rel}/checkpoints"
    state_s3, weights_s3 = f"{base}/state", f"{base}/weights"
    archive_s3 = f"{checkpoint_s3.rstrip('/')}-archive/{rel}"
    local = Path(output_dir) / "checkpoints"
    print(f"[prune] watching {state_s3} (keep={keep}, every {interval}s, "
          f"archive -> {archive_s3})", flush=True)
    while True:
        time.sleep(interval)
        try:
            prune_once(state_s3, weights_s3, archive_s3, local / "state",
                       local / "weights", keep=keep)
        except Exception as exc:  # never let pruning kill the job
            print(f"[prune] error (ignored): {exc}", flush=True)


def spawn(output_dir: str, checkpoint_s3: str, *, keep: int = 1,
          interval: int = 900) -> None:
    """Fork a daemon that prunes periodically; returns immediately in the parent.

    entry.py calls this just before ``os.execvp``. exec replaces the parent's
    process image but leaves children running, so the daemon outlives it.
    """
    if not checkpoint_s3:
        return
    try:
        if os.fork() != 0:
            return                      # parent: carry on and exec training
    except OSError as exc:
        print(f"[prune] fork failed, pruning disabled: {exc}", flush=True)
        return
    try:
        os.setsid()
    except OSError:
        pass
    try:
        loop(output_dir, checkpoint_s3, keep, interval)
    except Exception as exc:
        print(f"[prune] daemon exiting: {exc}", flush=True)
    finally:
        os._exit(0)


if __name__ == "__main__":
    # Manual use: python prune_checkpoints.py <output_dir> <checkpoint_s3> [keep]
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit(__doc__)
    loop(a[0], a[1], int(a[2]) if len(a) > 2 else 1, 0)

