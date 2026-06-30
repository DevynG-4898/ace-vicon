from __future__ import annotations

import numpy as np

from src.features.velocity import finite_difference


def angular_velocity_from_degrees(angle_degrees: np.ndarray, fps: float | None) -> np.ndarray:
    """Return angular velocity in degrees per second from per-frame angle values."""
    return finite_difference(np.asarray(angle_degrees, dtype=float), fps)

