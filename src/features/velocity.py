from __future__ import annotations

import numpy as np


def finite_difference(values: np.ndarray, fps: float | None) -> np.ndarray:
    """Differentiate frame-major values with respect to time using FPS."""
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] <= 1:
        return np.zeros_like(arr)
    rate = 1.0 if fps is None else float(fps)
    return np.gradient(arr, axis=0, edge_order=1) * rate


def speed_from_velocity(velocity: np.ndarray) -> np.ndarray:
    """Return Euclidean speed from a vector velocity series."""
    return np.linalg.norm(np.asarray(velocity, dtype=float), axis=-1)

