#!/usr/bin/env bash
# Send one synthetic observation, print predictions, and perform no actuation.
set -euo pipefail

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage: bash scripts/g1/client_template.sh [server_uri] [task_instruction]

Example:
  bash scripts/g1/client_template.sh ws://127.0.0.1:8000 "pick up the container and pour its contents"

Sends one synthetic RGB frame and 43D state. Prints and validates the returned
action horizon using server metadata. Replace the arrays with live observations
when integrating a robot client; this template never commands a robot.
HELP
  exit 0
fi
if [[ $# -gt 2 ]]; then
  echo "Usage: bash scripts/g1/client_template.sh [server_uri] [task_instruction]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SERVER_URI="${1:-ws://127.0.0.1:8000}" \
TASK_INSTRUCTION="${2:-pick up the container and pour its contents}" \
PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
PYTHONDONTWRITEBYTECODE=1 \
python - <<'PY'
import asyncio
import os

import numpy as np
import websockets

from experiments.g1.action_layout import ACTION_DIM, STATE_DIM, split_action, validate_state
from fastwam.serving import msgpack_numpy


async def receive_message(websocket):
    raw = await websocket.recv()
    if isinstance(raw, str):
        raise RuntimeError(raw)
    message = msgpack_numpy.unpackb(raw)
    if not isinstance(message, dict):
        raise TypeError(f"Expected a dictionary, got {type(message).__name__}.")
    return message


async def main():
    async with websockets.connect(
        os.environ["SERVER_URI"],
        compression=None,
        max_size=None,
    ) as websocket:
        metadata = await receive_message(websocket)
        print("Server metadata:", metadata)
        embodiment = metadata["default_embodiment"]
        schema = metadata["embodiments"][embodiment]
        image_keys = schema["image_keys"]
        state_keys = schema["state_keys"]
        action_keys = schema["action_keys"]
        if len(image_keys) != 1 or len(state_keys) != 1 or len(action_keys) != 1:
            raise ValueError("The G1 template requires one camera, one state, and one action key.")
        state_key, state_dim = next(iter(state_keys.items()))
        action_key, action_dim = next(iter(action_keys.items()))
        if state_dim != STATE_DIM or action_dim != ACTION_DIM:
            raise ValueError(f"Expected G1 state/action dimensions 43/78, got {state_dim}/{action_dim}.")
        horizon = int(schema["action_horizon"])
        if horizon <= 0:
            raise ValueError(f"Expected a positive action horizon, got {horizon}.")

        request = {
            "images": {
                # Replace with the latest HWC RGB camera frame.
                image_keys[0]: np.zeros((480, 640, 3), dtype=np.uint8),
            },
            "states": {
                # Replace with the latest 43D robot state.
                state_key: validate_state(np.zeros((STATE_DIM,), dtype=np.float32)),
            },
            "text": os.environ["TASK_INSTRUCTION"],
            "embodiment_tag": embodiment,
        }

        await websocket.send(msgpack_numpy.packb(request))
        response = await receive_message(websocket)
        action = np.asarray(response[action_key])
        if action.shape != (horizon, ACTION_DIM):
            raise ValueError(f"Expected action shape {(horizon, ACTION_DIM)}, got {action.shape}.")
        if action.dtype != np.float32:
            raise TypeError(f"Expected float32 actions, got {action.dtype}.")
        if not np.isfinite(action).all():
            raise ValueError("Response contains non-finite actions.")

        parts = split_action(action)
        print("Response keys:", response.keys())
        print("Action shape:", action.shape)
        print("Action dtype:", action.dtype)
        print("Motion-token / left-hand / right-hand shapes:", *(part.shape for part in parts))
        print("Action:\n", action)
        # Robot-side code may execute a prefix, then request a fresh prediction
        # from updated observations. This template only prints predictions.


asyncio.run(main())
PY
