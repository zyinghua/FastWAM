"""Offline, boundary-centered second-difference comparisons of RoboTwin traces.

Only denormalized 14-channel qpos targets recorded before TOPP are supported by
this renderer. The generic measurement CLI remains available for other action
representations. Figures describe commands, not measured physical acceleration.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import RATIO_EPS, measure_actions
from .profiles import get_groups
from .recording import load_trace

OFFSETS = np.arange(-6, 6)
CROSSING = np.isin(OFFSETS, [-1, 0])
COLORS = {"Rolling-WAM": "#D67B2C", "Fast-WAM": "#4289A3"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mean(values: list[Any]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= RATIO_EPS:
        return None
    value = numerator / denominator
    return float(value) if np.isfinite(value) else None


def trace_paths(inputs: list[Path]) -> list[Path]:
    """Resolve files/directories once; repeated paths do not double-count data."""
    found: set[Path] = set()
    for source in inputs:
        source = source.expanduser()
        _require(source.exists(), f"Input does not exist: {source}")
        if source.is_dir():
            found.update(p.resolve() for p in source.rglob("*.jsonl"))
        else:
            _require(source.suffix == ".jsonl", f"Expected JSONL trace: {source}")
            found.add(source.resolve())
    _require(bool(found), "No JSONL traces found in the supplied inputs")
    return sorted(found)


def analyze_episode(trace: dict) -> dict:
    """Keep all boundary scores, but plot only complete isolated neighborhoods."""
    path, meta = trace["path"], trace["metadata"]
    actions, ids = trace["actions"], trace["chunk_ids"]
    _require(trace["embodiment"] == "robotwin" and trace["source"] == "executed_command",
             f"{path}: plotting requires executed RoboTwin commands")
    _require(meta.get("action_space") == "joint_position_target" and
             meta.get("capture_point") == "before_RoboTwin_TOPP_interpolation",
             f"{path}: expected denormalized joint targets captured before TOPP")
    arms = get_groups("robotwin", actions.shape[1])["arms"]
    horizon = meta.get("execute_horizon")
    _require(type(horizon) is int and horizon > 0,
             f"{path}: missing positive integer execute_horizon metadata")
    boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    starts = np.r_[0, boundaries]
    lengths = np.diff(np.r_[starts, len(ids)])
    _require(np.all(lengths[:-1] == horizon) and 1 <= lengths[-1] <= horizon,
             f"{path}: consumed chunk lengths disagree with execute_horizon={horizon}")
    group = measure_actions(actions, ids, groups={"arms": arms})["groups"]["arms"]
    summary = group["second_difference"]
    q = actions[:, arms]
    # Array entry t-1 is c[t], centered on the middle command of the triplet.
    second = np.linalg.norm(np.diff(q, n=2, axis=0), axis=1) / np.sqrt(len(arms))
    profiles, events = [], []
    for event in group["boundary_events"]:
        b = event["index"]
        lo, hi = b + int(OFFSETS[0]) - 1, b + int(OFFSETS[-1]) + 1
        reason = None
        if lo < 0 or hi >= len(q):
            reason = "incomplete neighborhood or triplet at episode edge"
        elif not (np.all(ids[lo:b] == ids[b - 1]) and np.all(ids[b:hi + 1] == ids[b])):
            reason = "neighborhood contains another chunk transition"
        row = {"boundary_index": b, "chunk_id": event["chunk_id"],
               "second_difference_rad": event["second_difference"],
               "included_in_profile": reason is None, "exclusion_reason": reason,
               "nearby_interior_second_difference_rad": None, "signed_excess_rad": None}
        if reason is None:
            values = second[b + OFFSETS - 1]
            boundary = float(np.mean(values[CROSSING]))
            _require(np.isclose(boundary, event["second_difference"], rtol=1e-10, atol=1e-13),
                     f"{path}: second-difference profile indexing disagrees at boundary {b}")
            baseline = float(np.mean(values[~CROSSING]))
            row.update(nearby_interior_second_difference_rad=baseline,
                       signed_excess_rad=boundary - baseline,
                       profile_rad=values.tolist())
            profiles.append(values)
        events.append(row)
    mean_profile = np.mean(profiles, axis=0) if profiles else None
    return {
        "trace": path, "episode_id": trace["episode_id"], "method": trace["method"],
        "task": meta["task"], "success": trace["success"], "metadata": meta,
        "end_metadata": trace["end_metadata"], "commands": len(q),
        "execute_horizon": horizon, "chunk_lengths": lengths.tolist(),
        "boundaries": len(boundaries),
        "second_difference_boundary_count": summary["boundary"]["count"],
        "boundary_second_difference_rad": summary["boundary"]["mean"],
        "interior_second_difference_rad": summary["interior"]["mean"],
        "second_difference_interior_count": summary["interior"]["count"],
        "boundary_to_interior_second_difference": summary["ratio"],
        "profile_boundaries_included": len(profiles),
        "profile_boundaries_excluded": len(boundaries) - len(profiles),
        "mean_profile_rad": None if mean_profile is None else mean_profile.tolist(),
        "events": events,
    }


def _aggregate(episodes: list[dict], task: str, method: str) -> dict:
    eligible = [e for e in episodes if e["mean_profile_rad"] is not None]
    _require(bool(eligible), f"{task}/{method}: no complete isolated boundary profiles")
    profile = np.mean([e["mean_profile_rad"] for e in eligible], axis=0)
    metric_episodes = [e for e in episodes if e["boundary_second_difference_rad"] is not None
                       and e["interior_second_difference_rad"] is not None]
    boundary = _mean([e["boundary_second_difference_rad"] for e in metric_episodes])
    interior = _mean([e["interior_second_difference_rad"] for e in metric_episodes])
    profile_boundary = float(np.mean(profile[CROSSING]))
    local = float(np.mean(profile[~CROSSING]))
    return {
        "task": task, "method": method, "episodes": len(episodes),
        "profile_episodes": len(eligible), "metric_episodes": len(metric_episodes), "tasks": 1,
        "execute_horizons": sorted({e["execute_horizon"] for e in episodes}),
        "commands": sum(e["commands"] for e in episodes),
        "boundaries": sum(e["boundaries"] for e in episodes),
        "profile_boundaries_included": sum(e["profile_boundaries_included"] for e in episodes),
        "profile_boundaries_excluded": sum(e["profile_boundaries_excluded"] for e in episodes),
        "boundary_second_difference_rad": boundary,
        "interior_second_difference_rad": interior,
        "boundary_to_interior_second_difference": _ratio(boundary, interior),
        "profile_boundary_second_difference_rad": profile_boundary,
        "nearby_interior_second_difference_rad": local,
        "signed_excess_rad": profile_boundary - local,
        "mean_profile_rad": profile.tolist(),
    }


def analyze_traces(paths: list[Path], *, tasks: list[str], methods: list[str],
                   success_only: bool, audit_paths: list[Path] | None = None) -> dict:
    """Analyze explicit inputs with episode-first, then equal-task aggregation.

    Outcome filtering is explicit. Every supplied trace appears in the selection
    audit, including failed/incomplete episodes excluded from success-only plots.
    Corrupt traces fail validation rather than disappearing from the report.
    """
    _require(bool(tasks) and len(tasks) == len(set(tasks)), "Tasks must be nonempty and unique")
    _require(bool(methods) and len(methods) == len(set(methods)), "Methods must be nonempty and unique")
    episodes, selection, seen = [], [], set()
    selected_paths = {p.resolve() for p in paths}
    for path in sorted(selected_paths | {p.resolve() for p in (audit_paths or [])}):
        trace = load_trace(path)
        identity = (trace["method"], trace["episode_id"])
        _require(identity not in seen, f"Duplicate episode ID in inputs: {identity}")
        seen.add(identity)
        task = trace["metadata"].get("task")
        reason = None
        if path not in selected_paths:
            reason = "audit only; not an explicit analysis input"
        elif task not in tasks or trace["method"] not in methods:
            reason = "task or method not selected"
        elif not trace["complete"]:
            reason = "incomplete episode"
        elif type(trace["success"]) is not bool:
            reason = "missing recorded outcome"
        elif success_only and not trace["success"]:
            reason = "failed episode excluded by --success-only"
        selection.append({"trace": trace["path"], "episode_id": trace["episode_id"],
                          "method": trace["method"], "task": task,
                          "complete": trace["complete"], "success": trace["success"],
                          "selected": reason is None, "exclusion_reason": reason,
                          "metadata": trace["metadata"], "end_metadata": trace["end_metadata"]})
        if reason is None:
            episodes.append(analyze_episode(trace))
    summaries, matched_conditions = [], {}
    condition_keys = ["episode_seed", "instruction", "task_config", "evaluation_seed",
                      "instruction_type", "num_inference_steps", "execute_horizon"]
    for task in tasks:
        reference = None
        for method in methods:
            rows = [e for e in episodes if e["task"] == task and e["method"] == method]
            _require(bool(rows), f"No selected completed episodes for {task}/{method}")
            conditions = []
            for episode in rows:
                metadata = episode["metadata"]
                _require(all(metadata.get(k) is not None for k in condition_keys),
                         f"{episode['trace']}: missing matched-comparison metadata: {condition_keys}")
                conditions.append({k: metadata[k] for k in condition_keys})
            conditions.sort(key=lambda value: json.dumps(value, sort_keys=True))
            if reference is None:
                reference = conditions
                matched_conditions[task] = conditions
            else:
                _require(conditions == reference,
                         f"{task}/{method}: scenes, instructions, inference settings, execution horizons, "
                         "or replicate counts do not match the first method. Select matched trace files explicitly.")
            summaries.append(_aggregate(rows, task, method))
    # Each task has equal weight, regardless of its action/boundary/episode count.
    if len(tasks) > 1:
        for method in methods:
            rows = [s for s in summaries if s["method"] == method]
            profile = np.mean([s["mean_profile_rad"] for s in rows], axis=0)
            boundary = _mean([s["boundary_second_difference_rad"] for s in rows])
            interior = _mean([s["interior_second_difference_rad"] for s in rows])
            local = float(np.mean(profile[~CROSSING]))
            profile_boundary = float(np.mean(profile[CROSSING]))
            combined = {"task": "combined", "method": method, "tasks": len(tasks),
                        "execute_horizons": sorted({v for s in rows for v in s["execute_horizons"]}),
                        "boundary_second_difference_rad": boundary,
                        "interior_second_difference_rad": interior,
                        "boundary_to_interior_second_difference": _ratio(boundary, interior),
                        "profile_boundary_second_difference_rad": profile_boundary,
                        "nearby_interior_second_difference_rad": local,
                        "signed_excess_rad": profile_boundary - local,
                        "mean_profile_rad": profile.tolist()}
            for field in ("episodes", "profile_episodes", "metric_episodes", "commands", "boundaries",
                          "profile_boundaries_included", "profile_boundaries_excluded"):
                combined[field] = sum(s[field] for s in rows)
            summaries.append(combined)
    return {
        "schema": "rollingwam.action_second_difference_profiles.v1",
        "tasks": tasks, "methods": methods, "success_only": success_only,
        "selection_scope": "Conditional on recorded success" if success_only else "All completed outcomes",
        "selected_episodes": len(episodes), "excluded_episodes": len(selection) - len(episodes),
        "scored_dimensions": get_groups("robotwin", 14)["arms"], "grippers_excluded": [6, 13],
        "raw_units": "radians of denormalized qpos targets before TOPP; no time derivative or channel scaling",
        "figure_units": "milliradians (raw score multiplied by 1000)",
        "formula": "c[t] = RMS_12(a[t+1] - 2*a[t] + a[t-1]); C[b] = (c[b-1] + c[b])/2",
        "profile_offsets": OFFSETS.tolist(), "cross_boundary_offsets": [-1, 0],
        "profile_inclusion": "Complete -6..5 centers and stencil neighbors; only focal chunk transition allowed",
        "aggregation": "Boundaries within episode, then equal episode means within task; combined uses equal task means",
        "baseline": "Mean of offsets -6..-2 and 1..5 of each aggregate curve",
        "signed_excess": "Mean at offsets -1 and 0 minus nearby interior baseline; not absolute distance",
        "interior_ratio": "Ratio of aggregate raw boundary mean to aggregate whole-episode interior mean",
        "scope": "Descriptive command-space diagnostic; no confidence interval, physical acceleration, or causal claim",
        "matched_conditions": matched_conditions,
        "selection": selection, "episodes": episodes, "summaries": summaries,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_report(report: dict, output: Path) -> None:
    """Write trace provenance and numbers underlying every curve and exclusion."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    scalar = lambda row: {k: v for k, v in row.items() if not isinstance(v, (list, dict))}
    _write_csv(output / "summary.csv", [scalar(s) for s in report["summaries"]])
    _write_csv(output / "episodes.csv", [scalar(e) for e in report["episodes"]])
    _write_csv(output / "selection.csv", [scalar(s) for s in report["selection"]])
    events, profiles = [], []
    for episode in report["episodes"]:
        identity = {k: episode[k] for k in ("task", "method", "episode_id", "trace")}
        for event in episode["events"]:
            events.append({**identity, **scalar(event)})
            for offset, value in zip(OFFSETS, event.get("profile_rad", [])):
                profiles.append({**identity, "boundary_index": event["boundary_index"],
                                 "offset": int(offset), "second_difference_rad": value})
    _write_csv(output / "boundaries.csv", events)
    _write_csv(output / "event_profiles.csv", profiles)
    _write_csv(output / "mean_profiles.csv", [
        {"task": row["task"], "method": row["method"], "offset": int(offset),
         "crosses_boundary": bool(crossing), "second_difference_rad": value}
        for row in report["summaries"]
        for offset, crossing, value in zip(OFFSETS, CROSSING, row["mean_profile_rad"])])


def render_figures(report: dict, output: Path) -> list[str]:
    """Overlay methods; paired variants use identical axes, including y limits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = report["methods"]
    fallback = ["#806AA6", "#63966B", "#A86C79", "#758291"]
    colors = {method: COLORS.get(method, fallback[i % len(fallback)])
              for i, method in enumerate(methods)}
    style = {"font.family": "serif", "font.serif": ["Lora", "DejaVu Serif"],
             "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 13,
             "xtick.labelsize": 9, "ytick.labelsize": 9,
             "pdf.fonttype": 42, "ps.fonttype": 42,
             "axes.spines.top": False, "axes.spines.right": False}
    outputs = []
    panels = report["tasks"] + (["combined"] if len(report["tasks"]) > 1 else [])
    with plt.rc_context(style):
        for task in panels:
            rows = [s for s in report["summaries"] if s["task"] == task]
            episodes = [e for e in report["episodes"] if task == "combined" or e["task"] == task]
            individual = {method: [v["profile_rad"] for e in episodes if e["method"] == method
                                   for v in e["events"] if v["included_in_profile"]]
                          for method in methods}
            maximum = max(float(np.max(v)) for values in individual.values() for v in values) * 1000
            ymax = max(maximum * 1.12, 0.001)
            for faint in (True, False):
                fig, ax = plt.subplots(figsize=(7.0, 4.35))
                ax.axvspan(-1.5, .5, facecolor="#E4E5E7", alpha=.65, zorder=0)
                for row in rows:
                    color = colors[row["method"]]
                    if faint:
                        for values in individual[row["method"]]:
                            ax.plot(OFFSETS, np.asarray(values) * 1000, color=color,
                                    alpha=.19, linewidth=.85, zorder=1)
                    ax.axhline(row["nearby_interior_second_difference_rad"] * 1000,
                               color=color, linestyle=(0, (4, 3)), linewidth=1.1, alpha=.9, zorder=2)
                    ax.plot(OFFSETS, np.asarray(row["mean_profile_rad"]) * 1000,
                            color=color, linewidth=2.5, marker="o", markersize=3.3, zorder=3)
                title = "Combined Tasks" if task == "combined" else task.replace("_", " ").title()
                fig.suptitle(title, y=.975, fontsize=14)
                handles = []
                for row in rows:
                    color = colors[row["method"]]
                    handles.extend([
                        Line2D([0], [0], color=color, linewidth=2.5, marker="o", markersize=3.3,
                               label=f"{row['method']} ({row['profile_boundaries_included']} boundaries)"),
                        Line2D([0], [0], color=color, linewidth=1.1, linestyle=(0, (4, 3)),
                               label=f"{row['method']} Nearby Mean"),
                    ])
                fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .92),
                           ncol=min(len(methods), 2), frameon=False, fontsize=9,
                           handlelength=2.4, columnspacing=2.2)
                ax.set_xlabel("Action Offset from Chunk Boundary", labelpad=8)
                ax.set_ylabel("Action Second Difference (mrad)", labelpad=9)
                ax.set_xlim(-6.3, 5.3)
                ax.set_ylim(0, ymax)
                ax.set_xticks([-6, -4, -2, -1, 0, 2, 4, 5])
                ax.grid(axis="y", color="#E4E7EB", linewidth=.7)
                ax.set_axisbelow(True)
                ax.tick_params(length=3, color="#6F747B")
                for spine in ax.spines.values():
                    spine.set_color("#9AA0A6")
                weighting = "equal task weight" if task == "combined" else "equal rollout weight"
                detail = f"Thick: {weighting}." + (" Faint: individual boundaries." if faint else "")
                detail += " 12 arm joints (RMS)."
                fig.text(.53, .063,
                         "Dashed: nearby within-chunk mean. Gray: triplets crossing the boundary.\n" + detail,
                         ha="center", va="bottom", fontsize=8.5, color="#555C64", linespacing=1.6)
                fig.subplots_adjust(left=.115, right=.975, bottom=.255, top=.79)
                stem = f"{task}_{'with_boundaries' if faint else 'means_only'}"
                for suffix in ("png", "pdf"):
                    path = output / f"{stem}.{suffix}"
                    fig.savefig(path, dpi=240, facecolor="white")
                    outputs.append(str(path.resolve()))
                plt.close(fig)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot raw RoboTwin boundary second differences from recorded JSONL traces; no inference.",
        epilog="Example: --input run/rolling/traces run/fast/traces --tasks beat_block_hammer lift_pot "
               "--methods Rolling-WAM Fast-WAM --success-only --output run/figures. "
               "Each task and the task-balanced combination produce PNG/PDF pairs with faint boundaries "
               "and means only. Failure/exclusion provenance is retained; no confidence intervals are inferred.")
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="Trace files or recursively searched trace directories")
    parser.add_argument("--audit-input", nargs="+", type=Path, default=[],
                        help="Additional attempt traces/directories retained only as exclusion provenance; never plotted")
    parser.add_argument("--tasks", nargs="+", required=True, help="Exact trace metadata task names, in display order")
    parser.add_argument("--methods", nargs="+", default=["Rolling-WAM", "Fast-WAM"], help="Exact trace method names")
    outcome = parser.add_mutually_exclusive_group(required=True)
    outcome.add_argument("--success-only", action="store_true", help="Select successful completed episodes; record all exclusions")
    outcome.add_argument("--all-outcomes", action="store_true", help="Include both successful and failed completed episodes")
    parser.add_argument("--output", type=Path, required=True, help="New or empty report/figure directory")
    args = parser.parse_args(argv)
    _require(not args.output.exists() or (args.output.is_dir() and not any(args.output.iterdir())),
             f"Output directory must be new or empty: {args.output}")
    report = analyze_traces(trace_paths(args.input), tasks=args.tasks, methods=args.methods,
                            success_only=args.success_only,
                            audit_paths=trace_paths(args.audit_input) if args.audit_input else None)
    write_report(report, args.output)
    figures = render_figures(report, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "selected_episodes": report["selected_episodes"],
                      "excluded_episodes": report["excluded_episodes"], "figures": figures}, indent=2))
    return 0
