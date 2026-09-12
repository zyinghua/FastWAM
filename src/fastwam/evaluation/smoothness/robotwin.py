"""Opt-in command tracing at RoboTwin's pre-interpolation action boundary.

This adapter also works around another policy's ``env.take_action`` call: call
``begin_chunk`` when a new predicted chunk is selected for execution, then
``take_action`` once per executed command. It never records unused predictions.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .recording import ActionTraceRecorder


class RoboTwinActionTrace:
    def __init__(
        self,
        directory: str | Path,
        *,
        method: str = "Fast-WAM",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.recorder = ActionTraceRecorder(
            directory,
            method=method,
            embodiment="robotwin",
            source="executed_command",
            metadata={
                "action_space": "joint_position_target",
                "capture_point": "before_RoboTwin_TOPP_interpolation",
                "timestamp_semantics": "monotonic_command_submission_not_simulation_time",
                **(metadata or {}),
            },
        )
        self._active = False

    @staticmethod
    def _terminal_status(env: Any) -> bool | None:
        if bool(getattr(env, "eval_success", False)):
            return True
        count = getattr(env, "take_action_cnt", None)
        limit = getattr(env, "step_lim", None)
        if count is not None and limit is not None and count >= limit:
            return False
        return None

    def finish_if_terminal(self, env: Any) -> bool:
        """Avoid inference/logging when RoboTwin would ignore another command."""
        success = self._terminal_status(env)
        if success is None:
            return False
        if self._active:
            self.recorder.finish_episode(
                success=success,
                metadata={"termination": "success" if success else "step_limit"},
            )
            self._active = False
        return True

    def begin_chunk(self, env: Any, instruction: str) -> None:
        if self.finish_if_terminal(env):
            return
        if not self._active:
            metadata = {"task": getattr(env, "task_name", None), "instruction": instruction}
            # RoboTwin's standard evaluator does not retain the sampled seed on
            # the env. Do not confuse its evaluation seed with an episode seed.
            for name in ("ep_num", "episode_seed", "seed"):
                value = getattr(env, name, None)
                if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                    key = "episode_index" if name == "ep_num" else "episode_seed"
                    metadata.setdefault(key, int(value))
            self.recorder.start_episode(metadata=metadata)
            self._active = True
        self.recorder.start_chunk()

    def take_action(self, env: Any, action: np.ndarray) -> None:
        """Submit once, recording only accepted commands in their original form."""
        if self.finish_if_terminal(env):
            return
        if not self._active:
            raise RuntimeError("Call begin_chunk before recording RoboTwin actions")
        command = np.asarray(action).copy()
        timestamp = time.monotonic()
        before = getattr(env, "take_action_cnt", None)
        env.take_action(action, action_type="qpos")
        after = getattr(env, "take_action_cnt", None)
        # RoboTwin increments this counter only after its early-return guards.
        # A failed call raises above, so it cannot create a successful trace row.
        if before is None or after is None or after > before:
            self.recorder.record_action(command, timestamp=timestamp)
        self.finish_if_terminal(env)

    def reset(self) -> None:
        if self._active:
            self.recorder.finish_episode(metadata={"termination": "reset_before_terminal"})
            self._active = False

    def close(self) -> None:
        if self._active:
            self.recorder.finish_episode(metadata={"termination": "closed_before_terminal"})
            self._active = False
        self.recorder.close()
