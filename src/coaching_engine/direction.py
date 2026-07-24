"""Resolve correction direction from a CoachingTarget without tennis knowledge."""

from __future__ import annotations

from typing import Literal

from src.scoring_report_reader.context import CoachingTarget

Direction = Literal["too_low", "too_high"]

_DIRECTION_KEYS = ("direction", "deviation", "error_direction")


class CoachingDirectionError(ValueError):
    """Raised when a coaching target does not declare a usable direction."""


def resolve_direction(target: CoachingTarget) -> Direction:
    """Read ``too_low`` / ``too_high`` from target metadata provided by scoring.

    The Coaching Engine does not infer direction from scores. The Scoring Engine
    (or report reader pipeline) must place the direction on
    ``CoachingTarget.metadata`` under one of:

    - ``direction``
    - ``deviation``
    - ``error_direction``
    """
    for key in _DIRECTION_KEYS:
        if key not in target.metadata:
            continue
        value = target.metadata[key]
        if value in ("too_low", "too_high"):
            return value  # type: ignore[return-value]
        raise CoachingDirectionError(
            f"Coaching target {target.name!r} has invalid {key}={value!r}; "
            "expected 'too_low' or 'too_high'"
        )

    raise CoachingDirectionError(
        f"Coaching target {target.name!r} is missing direction metadata. "
        f"Set metadata[{_DIRECTION_KEYS[0]!r}] to 'too_low' or 'too_high'."
    )
