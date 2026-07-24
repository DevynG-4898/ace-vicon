"""Map Coaching Engine phases onto DTW synchronized-step phases."""

from __future__ import annotations

_COACHING_TO_DTW: dict[str, str] = {
    "Loading": "Loading",
    "Cocking": "Cocking",
    "Acceleration": "Acceleration",
    "Contact": "Contact",
    "Deceleration": "Deceleration_Finish",
    "Finish": "Deceleration_Finish",
}


def coaching_phase_to_dtw(phase: str) -> str | None:
    """Return the DTW step phase for a coaching recommendation phase.

    Unknown phases return ``None`` (caller should fall back to whole-serve playback).
    """
    return _COACHING_TO_DTW.get(phase)
