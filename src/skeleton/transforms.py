from __future__ import annotations

import numpy as np


def translate_positions(joint_positions: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Translate each frame by subtracting a per-frame `(frames, 3)` vector."""
    positions = np.asarray(joint_positions, dtype=float)
    offsets = np.asarray(translation, dtype=float)
    if offsets.shape != (positions.shape[0], 3):
        raise ValueError(f"translation must have shape ({positions.shape[0]}, 3); got {offsets.shape}")
    return positions - offsets[:, None, :]


def rotate_positions(joint_positions: np.ndarray, body_axes: np.ndarray) -> np.ndarray:
    """Rotate positions into the per-frame body coordinate basis."""
    positions = np.asarray(joint_positions, dtype=float)
    axes = np.asarray(body_axes, dtype=float)
    if axes.shape != (positions.shape[0], 3, 3):
        raise ValueError(f"body_axes must have shape ({positions.shape[0]}, 3, 3); got {axes.shape}")
    return np.einsum("fjd,fad->fja", positions, axes)


def scale_positions(joint_positions: np.ndarray, scale_factor: float) -> np.ndarray:
    """Scale positions by a positive scalar body-size factor."""
    scale = float(scale_factor)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale_factor must be positive and finite; got {scale_factor}")
    return np.asarray(joint_positions, dtype=float) / scale

