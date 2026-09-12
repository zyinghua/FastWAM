"""Episode-first, task-balanced reports from the shared command metrics."""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Any

import numpy as np

from .metrics import RATIO_EPS, _rms

BASE_METRICS = (
    "jump_boundary_mean", "jump_boundary_p95", "jump_interior_mean",
    "curvature_boundary_mean", "curvature_boundary_p95", "curvature_interior_mean",
)


def comparison_diagnostics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose task coverage and recorded protocol differences without filtering data.

    Missing metadata cannot establish matched execution settings. Compare each
    task separately so intentionally different task protocols are not conflated.
    """
    comparisons = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for episode in episodes:
        metadata = episode["metadata"]
        task = str(metadata.get("task_name") or metadata.get("task") or metadata.get("instruction") or "unspecified")
        comparisons[(episode["embodiment"], episode["source"])][episode["method"]][task].append(metadata)
    coverage, warnings = [], []
    for (embodiment, source), methods in sorted(comparisons.items()):
        label = f"{embodiment} / {source}"
        task_sets = {method: sorted(tasks) for method, tasks in sorted(methods.items())}
        coverage.append({"embodiment": embodiment, "source": source, "tasks_by_method": task_sets})
        if len({tuple(tasks) for tasks in task_sets.values()}) > 1:
            warnings.append(f"{label}: methods have different task sets; their aggregate means are not a matched-task comparison.")
        if any("unspecified" in tasks for tasks in task_sets.values()):
            warnings.append(f"{label}: some episodes lack task labels; task-balanced aggregation cannot identify those tasks.")
        for task in sorted({task for tasks in methods.values() for task in tasks}):
            for field in ("task_config", "execute_horizon", "actions_per_chunk", "control_hz", "fps"):
                by_method = {}
                for method, tasks in sorted(methods.items()):
                    metadata_rows = tasks.get(task, [])
                    known = [metadata[field] for metadata in metadata_rows if metadata.get(field) is not None]
                    values = sorted({json.dumps(float(value) if isinstance(value, (int, float))
                                               and not isinstance(value, bool) else value, sort_keys=True)
                                     for value in known})
                    if len(values) > 1:
                        warnings.append(f"{label} / {task} / {method}: mixed {field} values {values} are pooled in one summary.")
                    if values:
                        by_method[method] = values
                if len(by_method) > 1 and len({tuple(values) for values in by_method.values()}) > 1:
                    warnings.append(f"{label} / {task}: recorded {field} differs across methods: {by_method}.")
    return {"task_coverage": coverage, "warnings": warnings,
            "scope": "Recorded metadata only; absent settings are unverified, not assumed matched."}


def flatten_episode(trace: dict[str, Any], measurement: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = trace["metadata"]
    task = str(metadata.get("task_name") or metadata.get("task") or metadata.get("instruction") or "unspecified")
    rows = []
    for name, result in measurement["groups"].items():
        row = {
            "episode_id": trace["episode_id"], "path": trace["path"],
            "method": trace["method"], "embodiment": trace["embodiment"],
            "source": trace["source"], "group": name, "task": task,
            "num_actions": measurement["num_actions"], "num_chunks": measurement["num_chunks"],
            "num_boundaries": measurement["num_boundaries"],
            "success": trace["success"], "complete": trace["complete"],
        }
        for metric in ("jump", "curvature"):
            item = result[metric]
            row[f"{metric}_boundary_mean"] = item["boundary"]["mean"]
            row[f"{metric}_boundary_p95"] = item["boundary"]["p95"]
            row[f"{metric}_interior_mean"] = item["interior"]["mean"]
            row[f"{metric}_boundary_count"] = item["boundary"]["count"]
            row[f"{metric}_interior_count"] = item["interior"]["count"]
            row[f"{metric}_boundary_to_interior"] = item["ratio"]
        rows.append(row)
    return rows


def _mean(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return None if not values else float(np.mean(values))


def _ratio(numerator, denominator) -> float | None:
    if numerator is None or denominator is None or denominator <= RATIO_EPS:
        return None
    with np.errstate(over="ignore"):
        ratio = float(np.float64(numerator) / denominator)
    return ratio if np.isfinite(ratio) else None


def _task_macro(tasks: list[list[dict[str, Any]]], key: str) -> tuple[float | None, int, int]:
    task_means = [_mean(row.get(key) for row in task) for task in tasks]
    return (_mean(task_means), sum(value is not None for value in task_means),
            sum(row.get(key) is not None for task in tasks for row in task))


def summarize(rows: list[dict[str, Any]], *, bootstrap: int = 1000, seed: int = 0) -> list[dict[str, Any]]:
    """Average episodes within each task, then tasks equally.

    Confidence intervals resample whole episodes within the fixed task set.
    Rows from predicted and executed commands, or different embodiments/groups,
    are never pooled. P95 summaries are task-balanced means of episode P95s,
    not pooled percentiles that overweight long/failed episodes. Summary ratios
    divide the two task-balanced means using only episodes with both measures.
    """
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    grouped: dict[tuple, list] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["embodiment"], row["source"], row["group"])].append(row)
    output = []
    rng = np.random.default_rng(seed)
    for identity, episodes in sorted(grouped.items()):
        by_task: dict[str, list] = defaultdict(list)
        for row in episodes:
            by_task[row["task"]].append(row)
        tasks = list(by_task.values())
        result = dict(zip(("method", "embodiment", "source", "group"), identity))
        result.update(num_tasks=len(tasks), num_episodes=len(episodes),
                      num_actions=sum(row["num_actions"] for row in episodes),
                      num_boundaries=sum(row["num_boundaries"] for row in episodes),
                      unknown_success_episodes=sum(row["success"] is None for row in episodes))
        for key in (*BASE_METRICS, "success", "num_actions"):
            value, n_tasks, n_episodes = _task_macro(tasks, key)
            label = {"success": "success_rate", "num_actions": "mean_episode_actions"}.get(key, key)
            result[label] = value
            result[f"{label}_episodes"] = n_episodes
            result[f"{label}_tasks"] = n_tasks
            # A single episode cannot estimate episode-level variation.
            result[f"{label}_ci95_low"] = result[f"{label}_ci95_high"] = None
            eligible = [[row[key] for row in task if row.get(key) is not None] for task in tasks]
            eligible = [np.asarray(values, dtype=float) for values in eligible if values]
            if bootstrap and eligible and all(len(values) >= 2 for values in eligible):
                samples = sum(
                    values[rng.integers(0, len(values), size=(bootstrap, len(values)))].mean(axis=1)
                    for values in eligible
                ) / len(eligible)
                lo, hi = np.quantile(samples, [0.025, 0.975])
                result[f"{label}_ci95_low"], result[f"{label}_ci95_high"] = float(lo), float(hi)
        for metric in ("jump", "curvature"):
            boundary_key, interior_key = f"{metric}_boundary_mean", f"{metric}_interior_mean"
            paired = [[row for row in task if row[boundary_key] is not None and row[interior_key] is not None]
                      for task in tasks]
            numerator, _, n = _task_macro(paired, boundary_key)
            denominator, _, _ = _task_macro(paired, interior_key)
            result[f"{metric}_boundary_to_interior"] = _ratio(numerator, denominator)
            result[f"{metric}_ratio_episodes"] = n
        output.append(result)
    return output


def boundary_profile(
    trace: dict[str, Any], measurement: dict[str, Any], *, scales=None, radius: int = 5,
) -> list[dict[str, Any]]:
    """Event-aligned values for plotting; offset 0 is the new chunk's first command.

    A jump at index t connects t-1 to t. The second difference at index t uses
    t-1,t,t+1. Consequently a boundary artifact can affect offsets -1 and 0.
    Rows at another chunk boundary can appear for short execution horizons;
    their IDs are retained so downstream plots can make the selection explicit.
    """
    actions = np.asarray(trace["actions"], dtype=np.float64)
    if scales is not None:
        actions = actions / np.asarray(scales)
    rows = []
    ids = trace["chunk_ids"]
    metadata = trace["metadata"]
    task = str(metadata.get("task_name") or metadata.get("task") or metadata.get("instruction") or "unspecified")
    for name, group in measurement["groups"].items():
        selected = actions[:, group["dimensions"]]
        first = _rms(np.diff(selected, axis=0))
        second = _rms(np.diff(selected, n=2, axis=0))
        for event in group["boundary_events"]:
            b = event["index"]
            for offset in range(-radius, radius + 1):
                t = b + offset
                if not 0 <= t < len(actions):
                    continue
                rows.append({"episode_id": trace["episode_id"], "method": trace["method"],
                             "embodiment": trace["embodiment"], "source": trace["source"],
                             "group": name, "task": task, "boundary_index": b, "offset": offset,
                             "action_index": t, "chunk_id": int(ids[t]),
                             "is_boundary": bool(t > 0 and ids[t] != ids[t - 1]),
                             "jump": None if t == 0 else float(first[t - 1]),
                             "second_difference": float(second[t - 1]) if 0 < t < len(actions) - 1 else None})
    return rows
