"""Resolve evaluation labels without importing any model or GPU dependencies."""

from __future__ import annotations

import inspect
from typing import Any


def resolve_method_label(model: Any, override: str | None = None) -> str:
    """Identify the model's default inference path, or use an explicit run label."""
    if override is not None:
        if not isinstance(override, str) or not override.strip():
            raise ValueError("smoothness_method must be a nonempty string or None")
        if override.strip().lower() not in {"auto", "none", "null"}:
            return override.strip()
    for cls in type(model).__mro__:
        if cls.__name__ == "FastWAMOptionalIDM":
            parameter = inspect.signature(model.infer_action).parameters.get("action_infer_mode")
            mode = None if parameter is None else parameter.default
            if mode == "first_frame":
                return "Fast-WAM"
            if mode == "idm":
                return "IDM-WAM"
            raise ValueError("Cannot identify OptionalIDM inference mode; set smoothness_method explicitly")
        if cls.__name__ == "FastWAMIDM":
            return "IDM-WAM"
        if cls.__name__ == "FastWAMJoint":
            return "Joint-WAM"
        if cls.__name__ == "FastWAM":
            return "Fast-WAM"
    raise ValueError("Cannot identify model for tracing; set smoothness_method explicitly")
