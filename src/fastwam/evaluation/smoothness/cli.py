"""Offline interface; never loads a checkpoint or opens a robot/server connection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .metrics import measure_actions
from .profiles import get_groups
from .recording import load_trace
from .reporting import boundary_profile, comparison_diagnostics, flatten_episode, summarize


def _groups(specifications: list[str]) -> dict[str, list[int]]:
    groups = {}
    for specification in specifications:
        name, separator, ranges = specification.partition("=")
        if not separator or not name or name in groups:
            raise ValueError("--group must be unique NAME=0:6,7:13 (zero-based, end-exclusive)")
        dimensions = []
        for part in ranges.split(","):
            if ":" in part:
                start, end = part.split(":")
                start, end = int(start), int(end)
                if start < 0 or end <= start:
                    raise ValueError(f"Invalid dimension range: {part}")
                dimensions.extend(range(start, end))
            else:
                dimensions.append(int(part))
        groups[name] = dimensions
    return groups


def _load_npz(path: Path, args) -> dict:
    """Existing baselines can export one NPZ per episode without our logger."""
    with np.load(path, allow_pickle=False) as data:
        if "actions" not in data or "chunk_ids" not in data:
            raise ValueError(f"{path}: NPZ requires actions[T,D] and chunk_ids[T]")
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}: metadata_json must encode an object")
        identity = {}
        for key in ("method", "embodiment", "source"):
            identity[key] = getattr(args, key) or metadata.pop(key, None)
            if not isinstance(identity[key], str) or not identity[key].strip():
                raise ValueError(f"{path}: supply --{key} or {key!r} in metadata_json")
        success = metadata.pop("success", None)
        if success is not None and not isinstance(success, bool):
            raise ValueError(f"{path}: success must be boolean or null")
        return {**identity, "episode_id": str(path.resolve()), "path": str(path.resolve()),
                "metadata": metadata, "end_metadata": {}, "success": success, "complete": True,
                "actions": data["actions"].copy(), "chunk_ids": data["chunk_ids"].copy()}


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        if keys:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Episode JSONL/NPZ files or directories searched recursively")
    parser.add_argument("--output", required=True, type=Path, help="Report directory (use a new directory for each comparison)")
    parser.add_argument("--profile", help="Override automatic embodiment profile: robotwin or g1_sonic")
    parser.add_argument("--group", action="append", default=[], help="Custom dimensions, e.g. joints=0:29; repeat for multiple groups")
    parser.add_argument("--scale-file", type=Path, help="Shared positive per-channel divisors: JSON list or {\"scale\": [...]} (not inferred separately per method)")
    for key in ("method", "embodiment", "source"):
        parser.add_argument(f"--{key}", help=f"{key} for NPZ imports only; JSONL uses its recorded metadata")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Episode bootstrap replicates per fixed task set; 0 disables CIs")
    parser.add_argument("--seed", type=int, default=0, help="Offline bootstrap seed, unrelated to policy sampling")
    parser.add_argument("--boundary-radius", type=int, default=5, help="Offsets exported in boundary_profile.csv")
    parser.add_argument("--include-incomplete", action="store_true", help="Include valid JSONL prefixes without an end record; success stays unknown")
    args = parser.parse_args(argv)
    try:
        if args.bootstrap < 0 or args.boundary_radius < 0:
            raise ValueError("--bootstrap and --boundary-radius must be nonnegative")
        groups_override = _groups(args.group)
        scales = None
        scale_info = None
        if args.scale_file:
            payload = args.scale_file.read_bytes()
            values = json.loads(payload)
            scales = np.asarray(values["scale"] if isinstance(values, dict) else values, dtype=float)
            scale_info = {"path": str(args.scale_file.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
                          "scale": scales.tolist()}
        files = set()
        for path in args.inputs:
            if path.is_dir():
                files.update(p.resolve() for p in path.rglob("*") if p.suffix in (".jsonl", ".npz") and p.is_file())
            elif path.is_file() and path.suffix in (".jsonl", ".npz"):
                files.add(path.resolve())
            else:
                raise ValueError(f"No JSONL/NPZ input: {path}")
        if not files:
            raise ValueError("No episode trace files found")
        output_names = ("report.json", "summary.csv", "episodes.csv", "boundaries.csv", "boundary_profile.csv")
        if any((args.output / name).exists() for name in output_names):
            raise ValueError("Output reports already exist; choose a new --output directory")
        episode_rows, boundary_rows, profile_rows, episode_details, skipped = [], [], [], [], []
        seen = set()
        protocols = {}
        for path in sorted(files):
            trace = _load_npz(path, args) if path.suffix == ".npz" else load_trace(path)
            if not trace["complete"] and not args.include_incomplete:
                skipped.append(str(path))
                continue
            identifier = (trace["method"], trace["embodiment"], trace["source"], trace["episode_id"])
            if identifier in seen:
                raise ValueError(f"Duplicate episode (possibly copied trace): {identifier}")
            seen.add(identifier)
            actions = trace["actions"]
            if actions.ndim != 2 or len(actions) == 0:
                raise ValueError(f"{path}: actions must be nonempty [T,D]")
            groups = groups_override or get_groups(args.profile or trace["embodiment"], actions.shape[1])
            measured = measure_actions(actions, trace["chunk_ids"], groups=groups, scales=scales)
            # Prevent incompatible dimensions/scales from sharing a report series.
            for group, item in measured["groups"].items():
                key = (trace["embodiment"], trace["source"], group)
                dimensions = tuple(item["dimensions"])
                signature = (actions.shape[1], dimensions)
                if key in protocols and protocols[key] != signature:
                    raise ValueError(f"Incompatible action dimensions in series {key}")
                protocols[key] = signature
            flattened = flatten_episode(trace, measured)
            episode_rows.extend(flattened)
            for row in flattened:
                for event in measured["groups"][row["group"]]["boundary_events"]:
                    boundary_rows.append({key: row[key] for key in
                                          ("episode_id", "method", "embodiment", "source", "task", "group")} | event)
            profile_rows.extend(boundary_profile(trace, measured, scales=scales, radius=args.boundary_radius))
            detail = {key: value for key, value in trace.items() if key not in ("actions", "chunk_ids", "timestamps")}
            detail["measurement"] = measured
            episode_details.append(detail)
        if not episode_rows:
            raise ValueError("No usable episodes (use --include-incomplete to inspect truncated recordings)")
        summary = summarize(episode_rows, bootstrap=args.bootstrap, seed=args.seed)
        diagnostics = comparison_diagnostics(episode_details)
        report = {
            "schema": "rollingwam.action_smoothness_report.v1", "scale": scale_info,
            "measurement_target": "Action-command continuity at replan boundaries",
            "metric_labels": {"jump": "Boundary Jump", "curvature": "Boundary Second Difference"},
            "magnitude": "L2 / sqrt(number of selected dimensions)",
            "curvature": "mean of centered second differences at b-1 and b, requiring both valid",
            "aggregation": "episode means/percentiles, then equal task means; ratios use matched eligible episodes",
            "confidence_intervals": "95% percentile bootstrap of episodes within each fixed task; unavailable for singleton tasks",
            "bootstrap": args.bootstrap, "seed": args.seed, "skipped_incomplete": skipped,
            "notes": ["First/second command differences are not time-normalized physical derivatives or geometric curvature.",
                      "Predicted and executed command sources are separate series.",
                      "RoboTwin traces contain accepted joint targets before TOPP, not measured/reached joint positions.",
                      "G1 motion tokens are latent controls, not joint angles.",
                      "Inference pauses and physical controller timing are not scored by these command-index metrics.",
                      "A complete trace has an end record; it may still represent an interrupted episode with unknown success.",
                      "Ratios are null when interior means <= 1e-12; null is not zero.",
                      "Match horizons, controllers, action scaling and tasks before comparing methods.",
                      "Mean P95 is the task-balanced mean of episode P95s, not a pooled percentile."],
            "comparison_diagnostics": diagnostics, "summary": summary, "episodes": episode_details,
        }
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "report.json").write_text(encoded, encoding="utf-8")
        for name, rows in (("summary.csv", summary), ("episodes.csv", episode_rows),
                           ("boundaries.csv", boundary_rows), ("boundary_profile.csv", profile_rows)):
            _write_csv(args.output / name, rows)
        for warning in diagnostics["warnings"]:
            print(f"Warning: {warning}")
        for row in summary:
            jump = "n/a" if row["jump_boundary_mean"] is None else f"{row['jump_boundary_mean']:.6g}"
            curvature = "n/a" if row["curvature_boundary_mean"] is None else f"{row['curvature_boundary_mean']:.6g}"
            ratio = row["curvature_boundary_to_interior"]
            normalized_curvature = "n/a" if ratio is None else f"{ratio:.6g}"
            print(f"{row['method']} / {row['embodiment']} / {row['source']} / {row['group']}: "
                  f"boundary_jump={jump}, boundary_second_difference={curvature}, "
                  f"boundary_to_interior_second_difference={normalized_curvature}, episodes={row['num_episodes']}")
        print(f"Saved reports to {args.output.resolve()} (skipped {len(skipped)} incomplete traces)")
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
