#!/usr/bin/env python3
"""G1 robot-side recording interface and a non-actuating synthetic example.

Import record_executed_chunk into the existing robot-side client. It deliberately
does not implement SONIC transport, rate control, or a robot connection. The
callback must return True only for commands actually accepted by the controller.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fastwam.evaluation.smoothness.recording import ActionTraceRecorder


def record_executed_chunk(
    recorder: ActionTraceRecorder,
    commands: np.ndarray,
    execute_one: Callable[[np.ndarray], bool],
) -> int:
    """Execute/log the consumed prefix of one final, postprocessed 78D G1 chunk.

    Apply clipping, filtering or other command changes before passing commands.
    The callback owns scheduling and delivery, and must not silently transform
    these values. For transformations performed inside a controller, record
    there instead. A False return stops without recording the rejected command
    or its unused tail. Exceptions also propagate without recording that row.
    """
    commands = np.asarray(commands, dtype=float)
    if commands.ndim != 2 or commands.shape[1] != 78 or not np.isfinite(commands).all():
        raise ValueError("commands must be a finite [T,78] array in the SONIC layout")
    if len(commands) == 0:
        return 0
    recorder.start_chunk()
    consumed = 0
    for row in commands:
        command = row.copy()
        timestamp = time.monotonic()
        accepted = execute_one(command.copy())
        if not isinstance(accepted, (bool, np.bool_)):
            raise ValueError("execute_one must return an explicit bool confirming command acceptance")
        if not accepted:
            break
        recorder.record_action(command, timestamp=timestamp)
        consumed += 1
    return consumed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic G1 traces locally; never connects to or actuates a robot.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with ActionTraceRecorder(args.output, method="Synthetic example", embodiment="unitree_g1_sonic",
                             source="synthetic_example", metadata={"task": "constant_ramp_with_boundary_offset"}) as recorder:
        for start in (0, 10, 18):
            values = np.arange(start, start + 8, dtype=float)[:, None]
            commands = np.broadcast_to(values, (8, 78)).copy() * 0.01
            record_executed_chunk(recorder, commands, lambda command: True)
        recorder.finish_episode(success=None)
    print(f"Synthetic traces saved to {args.output.resolve()}; no robot commands were sent.")


if __name__ == "__main__":
    main()
