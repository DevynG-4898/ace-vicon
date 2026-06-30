from __future__ import annotations

import numpy as np

from src.features.velocity import finite_difference


def acceleration_from_positions(positions: np.ndarray, fps: float | None) -> np.ndarray:
    """Return second finite difference of positions with respect to time."""
    return finite_difference(finite_difference(positions, fps), fps)

