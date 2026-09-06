"""G1 state/action layout shared with the RollingWAM client integration.

The model predicts the complete continuous 78D action. These helpers preserve
its values and expose the SONIC motion-token and hand segments to robot clients.
Controller transport and actuation remain robot-side.
"""

from typing import Any, NamedTuple


STATE_DIM = 43
ACTION_DIM = 78
MOTION_TOKEN_DIM = 64
HAND_DIM = 7


class G1ActionParts(NamedTuple):
    motion_token: Any
    left_hand: Any
    right_hand: Any


def _require_last_dim(value: Any, expected: int, name: str) -> None:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) == 0:
        raise ValueError(f"`{name}` must be an array or tensor with a last dimension.")
    if int(shape[-1]) != expected:
        raise ValueError(f"`{name}` must have last dimension {expected}, got {shape[-1]}.")


def validate_state(state: Any) -> Any:
    """Validate and return a `[..., 43]` G1 state unchanged."""
    _require_last_dim(state, STATE_DIM, "state")
    return state


def split_action(action: Any) -> G1ActionParts:
    """Split `[motion token 64, left hand 7, right hand 7]` without conversion."""
    _require_last_dim(action, ACTION_DIM, "action")
    left_start = MOTION_TOKEN_DIM
    right_start = left_start + HAND_DIM
    return G1ActionParts(
        motion_token=action[..., :left_start],
        left_hand=action[..., left_start:right_start],
        right_hand=action[..., right_start:],
    )
