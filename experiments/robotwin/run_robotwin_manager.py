import csv
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
EVAL_STEP_LIMIT_FILE = PROJECT_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2
GIB = 1024**3


def _read_integer_file(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _host_available_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cgroup_available_memory_bytes() -> int | None:
    candidates = (
        (Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max")),
        (
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        ),
    )
    for current_path, limit_path in candidates:
        current = _read_integer_file(current_path)
        limit = _read_integer_file(limit_path)
        if current is None or limit is None or limit >= 1 << 60:
            continue
        return max(0, limit - current)
    return None


def _available_memory_bytes() -> int | None:
    candidates = [
        value
        for value in (_host_available_memory_bytes(), _cgroup_available_memory_bytes())
        if value is not None
    ]
    return min(candidates) if candidates else None


def _cgroup_memory_events() -> dict[str, int]:
    events_path = Path("/sys/fs/cgroup/memory.events")
    try:
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    events: dict[str, int] = {}
    for line in lines:
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            events[fields[0]] = int(fields[1])
        except ValueError:
            continue
    return events


def _format_memory(value: int | None) -> str:
    return "unknown" if value is None else f"{value / GIB:.1f} GiB"


def _raise_keyboard_interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _load_all_tasks() -> list[str]:
    if not EVAL_STEP_LIMIT_FILE.exists():
        raise FileNotFoundError(f"Task list file not found: {EVAL_STEP_LIMIT_FILE}")
    with EVAL_STEP_LIMIT_FILE.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {EVAL_STEP_LIMIT_FILE}")
    tasks = list(task_map.keys())
    # Keep original order and remove duplicates.
    seen = set()
    dedup_tasks: list[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup_tasks.append(task)
    return dedup_tasks


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return last_value


def _phase_result_filename(phase: str) -> str:
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    phase: str  # "clean" | "random"
    process: subprocess.Popen[str]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    launch_stagger_seconds = float(cfg.MULTIRUN.get("launch_stagger_seconds", 0))
    if launch_stagger_seconds < 0:
        raise ValueError("`MULTIRUN.launch_stagger_seconds` must be >= 0.")
    min_available_memory_gib = float(cfg.MULTIRUN.get("min_available_memory_gib", 0))
    if min_available_memory_gib < 0:
        raise ValueError("`MULTIRUN.min_available_memory_gib` must be >= 0.")
    min_available_memory_bytes = int(min_available_memory_gib * GIB)
    first_gpu_id = int(cfg.gpu_id)
    if first_gpu_id < 0:
        raise ValueError("`gpu_id` must be >= 0.")
    gpu_ids = list(range(first_gpu_id, first_gpu_id + num_gpus))

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin" / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"

    task_name_cfg = cfg.EVALUATION.task_name
    if task_name_cfg is None or str(task_name_cfg).strip() == "":
        tasks = _load_all_tasks()
    else:
        tasks = [str(task_name_cfg)]

    extra_overrides = _collect_worker_overrides()

    task_rates: dict[str, dict[str, float | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    pending_tasks = deque(tasks)
    running_states: list[RunningState] = []
    last_launch_time: float | None = None

    phase_to_task_config = {
        "clean": "demo_clean",
        "random": "demo_randomized",
    }

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def build_cmd(*, task_name: str, gpu_id: int, phase: str) -> list[str]:
        task_config = phase_to_task_config[phase]
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={str(ckpt_path)}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={task_config}",
            f"EVALUATION.output_dir={str(output_dir)}",
        ]
        cmd.extend(extra_overrides)
        return cmd

    def ensure_memory_headroom() -> None:
        if min_available_memory_bytes == 0:
            return
        available = _available_memory_bytes()
        if available is not None and available < min_available_memory_bytes:
            raise MemoryError(
                "Stopping before host/container OOM: effective available memory "
                f"is {_format_memory(available)}, below the configured "
                f"{min_available_memory_gib:.1f} GiB reserve."
            )

    def launch_phase(task_name: str, gpu_id: int, phase: str) -> RunningState:
        nonlocal last_launch_time
        if last_launch_time is not None and launch_stagger_seconds > 0:
            wait_seconds = launch_stagger_seconds - (time.monotonic() - last_launch_time)
            while wait_seconds > 0:
                ensure_memory_headroom()
                time.sleep(min(POLL_INTERVAL_SEC, wait_seconds))
                wait_seconds = launch_stagger_seconds - (
                    time.monotonic() - last_launch_time
                )
        ensure_memory_headroom()
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id, phase=phase)
        log(
            f"launch task={task_name} phase={phase} gpu={gpu_id} "
            f"cmd={' '.join(cmd)}"
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
            start_new_session=(os.name == "posix"),
        )
        last_launch_time = time.monotonic()
        state = RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            phase=phase,
            process=process,
        )
        running_states.append(state)
        return state

    def signal_worker(state: RunningState, sig: signal.Signals) -> None:
        if os.name == "posix":
            try:
                os.killpg(state.process.pid, sig)
            except ProcessLookupError:
                pass
            return
        if state.process.poll() is None:
            state.process.send_signal(sig)

    def worker_group_exists(state: RunningState) -> bool:
        if os.name != "posix":
            return state.process.poll() is None
        try:
            os.killpg(state.process.pid, 0)
        except ProcessLookupError:
            return False
        return True

    def terminate_all_running() -> None:
        for state in list(running_states):
            log(f"terminating task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
            signal_worker(state, signal.SIGTERM)
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass
        for state in list(running_states):
            if worker_group_exists(state):
                log(f"killing task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
                signal_worker(state, signal.SIGKILL)
            if state.process.poll() is None:
                state.process.wait()

    def gpu_running_count(gpu_id: int) -> int:
        count = 0
        for state in running_states:
            if state.gpu_id != gpu_id:
                continue
            if state.process.poll() is None:
                count += 1
        return count

    def try_launch_pending(gpu_id: int) -> None:
        while len(pending_tasks) > 0 and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            task_name = pending_tasks.popleft()
            try:
                launch_phase(task_name=task_name, gpu_id=gpu_id, phase="clean")
            except Exception:
                pending_tasks.appendleft(task_name)
                raise

    def write_outputs() -> None:
        clean_mean = _mean_or_none([task_rates[t]["clean"] for t in tasks])
        random_mean = _mean_or_none([task_rates[t]["random"] for t in tasks])

        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
            for task in tasks:
                writer.writerow(
                    [
                        task,
                        task_rates[task]["clean"],
                        task_rates[task]["random"],
                    ]
                )
            writer.writerow(["__overall__", clean_mean, random_mean])

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "clean_success_rate": _to_jsonable(task_rates[task]["clean"]),
                    "random_success_rate": _to_jsonable(task_rates[task]["random"]),
                }
                for task in tasks
            ],
            "overall": {
                "clean_mean_success_rate": _to_jsonable(clean_mean),
                "random_mean_success_rate": _to_jsonable(random_mean),
            },
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(
                    f"{rec['task_name']},{rec['phase']},gpu={rec['gpu_id']},"
                    f"return_code={rec['return_code']},reason={rec['reason']}\n"
                )

    has_failure = False
    failure_message = ""
    host_available = _host_available_memory_bytes()
    cgroup_available = _cgroup_available_memory_bytes()
    cgroup_events_at_start = _cgroup_memory_events()
    log(
        f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} "
        f"launch_stagger_seconds={launch_stagger_seconds:g} "
        f"host_available={_format_memory(host_available)} "
        f"cgroup_available={_format_memory(cgroup_available)} "
        f"cgroup_events={cgroup_events_at_start or 'unavailable'} "
        f"output_dir={run_output_dir}"
    )
    if max_tasks_per_gpu > 1:
        log(
            "warning: max_tasks_per_gpu > 1 runs multiple full model/SAPIEN "
            "processes on each GPU and may exhaust GPU or host memory"
        )

    previous_signal_handlers: dict[signal.Signals, Any] = {}
    if os.name == "posix":
        for shutdown_signal in (signal.SIGTERM, signal.SIGHUP):
            previous_signal_handlers[shutdown_signal] = signal.signal(
                shutdown_signal,
                _raise_keyboard_interrupt,
            )

    try:
        # Launch initial tasks for each GPU up to capacity.
        for gpu_id in gpu_ids:
            try_launch_pending(gpu_id)

        while len(running_states) > 0:
            ensure_memory_headroom()
            progressed = False
            for state in list(running_states):
                gpu_id = state.gpu_id
                return_code = state.process.poll()
                if return_code is None:
                    continue
                progressed = True
                running_states.remove(state)

                if return_code != 0:
                    signal_worker(state, signal.SIGKILL)
                    has_failure = True
                    failure_message = (
                        f"worker failed: task={state.task_name}, phase={state.phase}, "
                        f"gpu={gpu_id}, return_code={return_code}"
                    )
                    failed_records.append(
                        {
                            "task_name": state.task_name,
                            "phase": state.phase,
                            "gpu_id": gpu_id,
                            "return_code": return_code,
                            "reason": "process_failed",
                        }
                    )
                    log(failure_message)
                    log(
                        f"memory at failure: available="
                        f"{_format_memory(_available_memory_bytes())} "
                        f"cgroup_events={_cgroup_memory_events() or 'unavailable'}"
                    )
                    terminate_all_running()
                    running_states.clear()
                    break

                result_file = run_output_dir / state.task_name / _phase_result_filename(state.phase)
                try:
                    success_rate = _parse_success_rate(result_file)
                except Exception as exc:
                    has_failure = True
                    failure_message = (
                        f"result parse failed: task={state.task_name}, phase={state.phase}, "
                        f"gpu={gpu_id}, error={repr(exc)}"
                    )
                    failed_records.append(
                        {
                            "task_name": state.task_name,
                            "phase": state.phase,
                            "gpu_id": gpu_id,
                            "return_code": return_code,
                            "reason": "result_parse_failed",
                        }
                    )
                    log(failure_message)
                    terminate_all_running()
                    running_states.clear()
                    break

                task_rates[state.task_name][state.phase] = success_rate
                log(
                    f"done task={state.task_name} phase={state.phase} gpu={gpu_id} "
                    f"success_rate={success_rate:.4f}"
                )

                if state.phase == "clean":
                    try:
                        launch_phase(
                            task_name=state.task_name,
                            gpu_id=gpu_id,
                            phase="random",
                        )
                    except MemoryError:
                        failed_records.append(
                            {
                                "task_name": state.task_name,
                                "phase": "random",
                                "gpu_id": gpu_id,
                                "return_code": -1,
                                "reason": "memory_guard",
                            }
                        )
                        raise
                    continue

                try_launch_pending(gpu_id)

            if has_failure:
                break
            if not progressed:
                time.sleep(POLL_INTERVAL_SEC)
    except MemoryError as exc:
        has_failure = True
        failure_message = str(exc)
        log(failure_message)
        for state in running_states:
            failed_records.append(
                {
                    "task_name": state.task_name,
                    "phase": state.phase,
                    "gpu_id": state.gpu_id,
                    "return_code": -1,
                    "reason": "memory_guard",
                }
            )
    except KeyboardInterrupt:
        log("manager interrupted; terminating all worker process groups")
        raise
    finally:
        terminate_all_running()
        for shutdown_signal, previous_handler in previous_signal_handlers.items():
            signal.signal(shutdown_signal, previous_handler)

    # Mark not started tasks when failure happened.
    if has_failure:
        for task_name in pending_tasks:
            failed_records.append(
                {
                    "task_name": task_name,
                    "phase": "not_started",
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")

    if has_failure:
        raise RuntimeError(failure_message)

    log("manager finished successfully")


if __name__ == "__main__":
    main()
