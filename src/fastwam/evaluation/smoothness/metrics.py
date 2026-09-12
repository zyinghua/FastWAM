"""Boundary smoothness of one episode of executed action commands.

These are action first and second differences, not physical velocity, acceleration,
or jerk: their physical meaning depends on the action representation and timestep.
No temporal derivative or resampling is applied. Keep control rates and action
representations matched when comparing policies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np

RATIO_EPS = 1e-12


def _real_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _rms(rows: np.ndarray) -> np.ndarray:
    """Compute L2 / sqrt(D), avoiding overflow in the sum of squares."""
    if rows.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    maximum = np.max(np.abs(rows), axis=1)
    divisor = np.where(maximum > 0, maximum, 1.0)
    return maximum * np.sqrt(np.mean((rows / divisor[:, None]) ** 2, axis=1))


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"count": 0, "mean": None, "p95": None}
    maximum = float(np.max(values))
    mean = maximum * float(np.mean(values / maximum)) if maximum else 0.0
    return {
        "count": int(values.size),
        "mean": mean,
        "p95": float(np.percentile(values, 95)),
    }


def _summary(boundary: np.ndarray, interior: np.ndarray) -> dict:
    boundary_stats, interior_stats = _stats(boundary), _stats(interior)
    numerator, denominator = boundary_stats["mean"], interior_stats["mean"]
    ratio = None
    if numerator is not None and denominator is not None and denominator > RATIO_EPS:
        with np.errstate(over="ignore"):
            candidate = float(np.float64(numerator) / denominator)
        if np.isfinite(candidate):
            ratio = candidate
    return {"boundary": boundary_stats, "interior": interior_stats, "ratio": ratio}


def measure_actions(
    actions: np.ndarray,
    chunk_ids: np.ndarray,
    *,
    groups: Mapping[str, Sequence[int]],
    scales: np.ndarray | None = None,
) -> dict:
    """Measure boundary and within-chunk variation for a single episode.

    ``actions`` has shape [T, D] and contains commands actually executed, in order.
    ``chunk_ids`` has shape [T]; nonnegative integer IDs must be nondecreasing.
    Each change of ID marks the first action ``b`` from a new executed chunk.
    IDs may skip values. Never concatenate episodes before calling this function.

    Each named group is measured separately, after dividing each action channel
    by its positive ``scales`` entry (default: one). Magnitudes use L2 / sqrt(d)
    for that group's d dimensions. For z = actions / scales:

      jump[b] = RMS(z[b] - z[b-1])
      c[t] = RMS((z[t+1] - z[t]) - (z[t] - z[t-1]))
      curvature[b] = (c[b-1] + c[b]) / 2

    Boundary curvature is available only when both centers exist (b >= 2 and
    b + 1 < T). Its two stencils may include another boundary for short chunks.
    An interior jump uses two actions from one chunk; interior curvature uses
    three actions from one chunk. Curvature is an action second difference,
    rather than geometric curvature or a universally physical jerk measure.

    The ratio is the boundary mean divided by the corresponding interior mean.
    Missing statistics and ratios with interior means <= ``RATIO_EPS`` (1e-12,
    in scaled action coordinates) are ``None``. Raw
    scores should accompany ratios: a noisy interior can lower a ratio.
    All output is JSON serializable; input arrays are never modified.
    """
    values = _real_array(actions, "actions")
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("actions must have shape [T, D] with D > 0")
    num_actions, action_dim = values.shape

    ids = np.asarray(chunk_ids)
    if ids.ndim != 1 or ids.shape[0] != num_actions:
        raise ValueError("chunk_ids must have shape [T], matching actions")
    # An empty Python list has float dtype; it is valid for an empty episode.
    if ids.size and ids.dtype.kind not in "iu":
        raise ValueError("chunk_ids must contain nonnegative integers")
    if np.any(ids < 0) or np.any(ids[1:] < ids[:-1]):
        raise ValueError("chunk_ids must be nonnegative and nondecreasing")

    if scales is None:
        divisors = np.ones(action_dim, dtype=np.float64)
    else:
        divisors = _real_array(scales, "scales")
        if divisors.shape != (action_dim,) or np.any(divisors <= 0):
            raise ValueError("scales must have shape [D] with strictly positive values")

    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("groups must be a nonempty mapping of names to dimensions")
    checked_groups = {}
    for name, dimensions in groups.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("group names must be nonempty strings")
        try:
            indices = list(dimensions)
        except TypeError as exc:
            raise ValueError(f"dimensions for group {name!r} must be a sequence") from exc
        if not indices or any(
            not isinstance(index, Integral) or isinstance(index, (bool, np.bool_))
            for index in indices
        ):
            raise ValueError(f"group {name!r} must have nonempty integer dimensions")
        indices = [int(index) for index in indices]
        if len(set(indices)) != len(indices):
            raise ValueError(f"group {name!r} contains duplicate dimensions")
        if any(index < 0 or index >= action_dim for index in indices):
            raise ValueError(f"group {name!r} has dimensions outside [0, D)")
        checked_groups[name] = indices

    with np.errstate(over="ignore", invalid="ignore"):
        normalized = values / divisors
        differences = np.diff(normalized, axis=0)
        second_differences = np.diff(differences, axis=0)
    if not all(np.all(np.isfinite(array)) for array in
               (normalized, differences, second_differences)):
        raise ValueError("action differences overflow; use suitable channel scales")

    transitions = ids[1:] != ids[:-1]
    boundaries = np.flatnonzero(transitions) + 1
    curvature_valid = (boundaries >= 2) & (boundaries + 1 < num_actions)
    valid_boundaries = boundaries[curvature_valid]
    interior_centers = (ids[:-2] == ids[1:-1]) & (ids[1:-1] == ids[2:])
    result = {
        "num_actions": int(num_actions),
        "action_dim": int(action_dim),
        "num_chunks": int(boundaries.size + bool(num_actions)),
        "num_boundaries": int(boundaries.size),
        "groups": {},
    }
    for name, indices in checked_groups.items():
        jumps = _rms(differences[:, indices])
        curvature = _rms(second_differences[:, indices])
        # curvature array index t-1 corresponds to the stencil centered at t.
        boundary_curvature = (
            curvature[valid_boundaries - 2] / 2
            + curvature[valid_boundaries - 1] / 2
        )
        events = []
        for b, valid in zip(boundaries, curvature_valid):
            events.append({
                "index": int(b),
                "chunk_id": int(ids[b]),
                "jump": float(jumps[b - 1]),
                "curvature": float(curvature[b - 2] / 2 + curvature[b - 1] / 2)
                if valid else None,
            })
        result["groups"][name] = {
            "dimensions": indices,
            "jump": _summary(jumps[transitions], jumps[~transitions]),
            "curvature": _summary(boundary_curvature, curvature[interior_centers]),
            "boundary_events": events,
        }
    return result
