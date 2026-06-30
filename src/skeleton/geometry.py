from __future__ import annotations

import numpy as np

EPSILON = 1e-9


def normalized_vectors(vectors: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    """Normalize a `(frames, 3)` vector array with a fallback for degenerate rows."""
    values = np.asarray(vectors, dtype=float)
    fallback_vec = np.asarray(fallback, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    out = np.divide(values, norms, out=np.zeros_like(values), where=norms > EPSILON)
    bad = norms[:, 0] <= EPSILON
    if np.any(bad):
        out[bad] = fallback_vec / max(float(np.linalg.norm(fallback_vec)), EPSILON)
    return out


def orthonormal_body_axes(
    left_hip: np.ndarray,
    right_hip: np.ndarray,
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
) -> np.ndarray:
    """Estimate per-frame body axes from shoulders and hips.

    Returns an array of shape `(frames, 3, 3)` where rows are canonical body
    right, up, and forward axes expressed in the source coordinate system.
    """
    left_side = (np.asarray(left_hip, dtype=float) + np.asarray(left_shoulder, dtype=float)) / 2.0
    right_side = (np.asarray(right_hip, dtype=float) + np.asarray(right_shoulder, dtype=float)) / 2.0
    hip_mid = (np.asarray(left_hip, dtype=float) + np.asarray(right_hip, dtype=float)) / 2.0
    shoulder_mid = (np.asarray(left_shoulder, dtype=float) + np.asarray(right_shoulder, dtype=float)) / 2.0

    x_axis = normalized_vectors(right_side - left_side, fallback=np.array([1.0, 0.0, 0.0]))
    up_raw = shoulder_mid - hip_mid
    up_projected = up_raw - np.sum(up_raw * x_axis, axis=1, keepdims=True) * x_axis
    y_axis = normalized_vectors(up_projected, fallback=np.array([0.0, 1.0, 0.0]))
    z_axis = normalized_vectors(np.cross(x_axis, y_axis), fallback=np.array([0.0, 0.0, 1.0]))
    y_axis = normalized_vectors(np.cross(z_axis, x_axis), fallback=np.array([0.0, 1.0, 0.0]))

    return np.stack([x_axis, y_axis, z_axis], axis=1)


def median_positive_length(lengths: np.ndarray) -> float | None:
    """Return the finite positive median length, or None if no valid value exists."""
    values = np.asarray(lengths, dtype=float)
    finite = values[np.isfinite(values) & (values > EPSILON)]
    if finite.size == 0:
        return None
    return float(np.median(finite))

