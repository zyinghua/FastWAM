"""Action coordinate groups for the repository's supported robot interfaces."""

from __future__ import annotations

from numbers import Integral


def get_groups(profile: str, action_dim: int) -> dict[str, list[int]]:
    """Return explicit, representation-specific groups; reject dimension mismatch.

    RoboTwin's 14 commands are left arm (6), left gripper, right arm (6),
    right gripper. Gripper commands are excluded from the arm smoothness score.

    G1 SONIC's 78 commands comprise a 64-dimensional motion token and two
    seven-dimensional hand commands. The motion token is a latent representation,
    not physical joint motion. Keep these three scores separate; never average
    the motion token with hand joints into a single physical smoothness score.
    """
    if not isinstance(action_dim, Integral) or isinstance(action_dim, bool) or action_dim <= 0:
        raise ValueError("action_dim must be a positive integer")
    if not isinstance(profile, str):
        raise ValueError("profile must be a string")
    name = profile.strip().lower()
    if name in {"robotwin", "robotwin14"}:
        if action_dim != 14:
            raise ValueError(f"RoboTwin profile expects 14 action dimensions, got {action_dim}")
        return {"arms": [*range(6), *range(7, 13)]}
    if name in {"g1", "g1_sonic", "g1_sonic78", "unitree_g1_sonic"}:
        if action_dim != 78:
            raise ValueError(f"G1 SONIC profile expects 78 action dimensions, got {action_dim}")
        return {
            "motion_token": list(range(64)),
            "left_hand": list(range(64, 71)),
            "right_hand": list(range(71, 78)),
        }
    raise ValueError(f"Unknown action profile {profile!r}; use 'robotwin' or 'g1_sonic'")
