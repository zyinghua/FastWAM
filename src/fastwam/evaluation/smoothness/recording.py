"""One JSONL file per episode, containing only commands and small metadata.

The caller decides when an action is submitted/executed. A predicted chunk is
not evidence of execution: use source='predicted_command' for server logging.
This module does not import a policy, connect to a server, or actuate a robot.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

# Retain the common trace schema so either repository can analyze both methods.
SCHEMA = "rollingwam.action_trace.v1"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, default=_json_default, ensure_ascii=False)


def _metadata(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is not None and not isinstance(value, dict):
        raise ValueError("metadata must be a dict or None")
    return json.loads(_json(value if value is not None else {}))


class ActionTraceRecorder:
    """Opt-in recorder shared by RoboTwin, G1, Fast-WAM, and Joint-WAM clients.

    Call start_chunk() when switching to a newly planned chunk, then
    record_action() for each command actually consumed. Never record a whole
    predicted chunk as executed when its tail might be dropped. Files are
    opened lazily; resets before the first action produce no empty episodes.
    Timestamps default to monotonic wall time and are diagnostic metadata,
    not used as physical timesteps in the command-difference metrics.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        method: str = "Fast-WAM",
        embodiment: str = "robotwin",
        source: str = "executed_command",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        for name, value in (("method", method), ("embodiment", embodiment), ("source", source)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        self.directory = Path(directory).expanduser()
        self.method, self.embodiment, self.source = method, embodiment, source
        self.metadata = _metadata(metadata)
        self._episode_metadata: dict[str, Any] = {}
        self._file = None
        self.path: Path | None = None
        self._chunk_id = -1
        self._count = 0
        self._action_dim: int | None = None
        self._last_timestamp: float | None = None

    def start_episode(self, metadata: dict[str, Any] | None = None) -> None:
        episode_metadata = _metadata(metadata)
        self.finish_episode(metadata={"end_reason": "next_episode"})
        self._episode_metadata = episode_metadata

    def start_chunk(self) -> int:
        self._chunk_id += 1
        return self._chunk_id

    def record_action(self, action: Any, timestamp: float | None = None) -> None:
        if self._chunk_id < 0:
            raise ValueError("Call start_chunk() before recording an action")
        values = np.asarray(action, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("action must be a nonempty finite one-dimensional vector")
        if self._action_dim is not None and values.size != self._action_dim:
            raise ValueError(f"Action dimension changed from {self._action_dim} to {values.size}")
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("Timestamps must be nondecreasing within an episode")
        row = {"type": "action", "index": self._count, "chunk_id": self._chunk_id,
               "timestamp": timestamp, "action": values.tolist()}
        encoded = _json(row)  # Validate before creating a file or changing counters.
        if self._file is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            episode_id = uuid.uuid4().hex
            self.path = self.directory / f"episode-{episode_id}.jsonl"
            self._file = self.path.open("x", encoding="utf-8", buffering=1)
            header = {"type": "episode", "schema": SCHEMA, "episode_id": episode_id,
                      "method": self.method, "embodiment": self.embodiment, "source": self.source,
                      "metadata": {**self.metadata, **self._episode_metadata}}
            self._file.write(_json(header) + "\n")
        self._file.write(encoded + "\n")
        self._count += 1
        self._action_dim = int(values.size)
        self._last_timestamp = timestamp

    def finish_episode(
        self, success: bool | None = None, metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        if success is not None and not isinstance(success, (bool, np.bool_)):
            raise ValueError("success must be bool or None")
        end_metadata = _metadata(metadata)
        result = self.path if self._file is not None else None
        if self._file is not None:
            footer = {"type": "end", "num_actions": self._count,
                      "success": None if success is None else bool(success), "metadata": end_metadata}
            self._file.write(_json(footer) + "\n")
            self._file.close()
        self._file = None
        self.path = None
        self._episode_metadata = {}
        self._chunk_id = -1
        self._count = 0
        self._action_dim = None
        self._last_timestamp = None
        return result

    def close(self) -> None:
        self.finish_episode(metadata={"end_reason": "recorder_closed"})

    def __enter__(self) -> "ActionTraceRecorder":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.finish_episode(metadata={"end_reason": "exception" if exc_type else "context_exit"})


def load_trace(path: str | Path) -> dict[str, Any]:
    """Read and validate a recorded episode. Missing end records mark truncation.

    A trailing partial line or corrupt record is an error, not silently dropped.
    Returned arrays are plain NumPy, with no pickle or simulator dependency.
    """
    path = Path(path)
    header = footer = None
    actions, chunk_ids, timestamps = [], [], []
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON record") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: record must be an object")
            kind = row.get("type")
            if header is None:
                if kind != "episode" or row.get("schema") != SCHEMA:
                    raise ValueError(f"{path}: expected {SCHEMA} episode header")
                for key in ("episode_id", "method", "embodiment", "source"):
                    if not isinstance(row.get(key), str) or not row[key].strip():
                        raise ValueError(f"{path}: missing/non-string {key}")
                if not isinstance(row.get("metadata"), dict):
                    raise ValueError(f"{path}: metadata must be an object")
                header = row
            elif footer is not None:
                raise ValueError(f"{path}:{lineno}: records after episode end")
            elif kind == "action":
                if type(row.get("index")) is not int or row["index"] != len(actions):
                    raise ValueError(f"{path}:{lineno}: action indices must be consecutive from zero")
                chunk = row.get("chunk_id")
                if type(chunk) is not int or chunk < 0 or (chunk_ids and chunk < chunk_ids[-1]):
                    raise ValueError(f"{path}:{lineno}: chunk IDs must be nonnegative, nondecreasing integers")
                action = np.asarray(row.get("action"), dtype=np.float64)
                if action.ndim != 1 or action.size == 0 or not np.isfinite(action).all():
                    raise ValueError(f"{path}:{lineno}: invalid action vector")
                if actions and action.shape != actions[0].shape:
                    raise ValueError(f"{path}:{lineno}: action dimension changed")
                timestamp = row.get("timestamp")
                if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                    raise ValueError(f"{path}:{lineno}: invalid timestamp")
                if not math.isfinite(timestamp) or (timestamps and timestamp < timestamps[-1]):
                    raise ValueError(f"{path}:{lineno}: timestamps must be finite and nondecreasing")
                actions.append(action)
                chunk_ids.append(chunk)
                timestamps.append(timestamp)
            elif kind == "end":
                if type(row.get("num_actions")) is not int or row["num_actions"] != len(actions):
                    raise ValueError(f"{path}:{lineno}: end action count mismatch")
                if row.get("success") is not None and not isinstance(row["success"], bool):
                    raise ValueError(f"{path}:{lineno}: success must be bool or null")
                if not isinstance(row.get("metadata", {}), dict):
                    raise ValueError(f"{path}:{lineno}: end metadata must be an object")
                footer = row
            else:
                raise ValueError(f"{path}:{lineno}: unexpected record type {kind!r}")
    if header is None or not actions:
        raise ValueError(f"{path}: trace contains no actions")
    return {"path": str(path.resolve()), **header, "complete": footer is not None,
            "success": None if footer is None else footer.get("success"),
            "end_metadata": {} if footer is None else footer.get("metadata", {}),
            "actions": np.stack(actions), "chunk_ids": np.array(chunk_ids, dtype=np.int64),
            "timestamps": np.array(timestamps, dtype=np.float64)}
