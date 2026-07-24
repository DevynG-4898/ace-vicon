"""Angle trajectory extractors for Knowledge Library angle features.

Each function returns a 1D series where **higher values mean more of the
coached quantity** (more flexion, more tilt, more extension, more coil, …)
so the Angle Kernel direction semantics stay consistent.
"""

from __future__ import annotations

import numpy as np

from src.skeleton import SkeletonSequence

EPSILON = 1e-9
Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=float)


def _as_1d(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    out = tuple(np.asarray(a, dtype=float) for a in arrays)
    shapes = {a.shape for a in out}
    if len(shapes) != 1:
        raise ValueError(f"series shape mismatch: {shapes}")
    return out


def _project_xz(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values)
    out[:, 0] = values[:, 0]
    out[:, 2] = values[:, 2]
    return out


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = np.maximum(a_norm * b_norm, EPSILON)
    cosang = np.clip(np.sum(a * b, axis=1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def flexion_from_interior(interior_deg: np.ndarray) -> np.ndarray:
    """Interior joint angle → flexion depth (higher = more flexed)."""
    return 180.0 - np.asarray(interior_deg, dtype=float)


def extension_from_interior(interior_deg: np.ndarray) -> np.ndarray:
    """Interior joint angle as extension (higher = more extended / straighter)."""
    return np.asarray(interior_deg, dtype=float)


def compute_hip_flexion_series(hip_flexion_deg: np.ndarray) -> np.ndarray:
    """Pass-through: FeatureSequence ``hip_flexion`` (higher = more flexed)."""
    (series,) = _as_1d(hip_flexion_deg)
    return series.copy()


def compute_shoulder_tilt_series(shoulder_line_angle_deg: np.ndarray) -> np.ndarray:
    """Shoulder tilt magnitude from the shoulder line angle (higher = more tilt)."""
    (series,) = _as_1d(shoulder_line_angle_deg)
    return np.abs(series)


def compute_toss_arm_extension_series(toss_elbow_interior_deg: np.ndarray) -> np.ndarray:
    """Toss-arm elbow extension (higher = more extended)."""
    return extension_from_interior(toss_elbow_interior_deg)


def compute_trunk_rotation_series(trunk_rotation_deg: np.ndarray) -> np.ndarray:
    """Trunk coil magnitude (higher = more rotation vs pelvis)."""
    (series,) = _as_1d(trunk_rotation_deg)
    return np.abs(series)


def compute_pelvis_rotation_series_from_skeleton(sequence: SkeletonSequence) -> np.ndarray:
    """Pelvis yaw from the left→right hip line in the X/Z plane (degrees)."""
    hips = _project_xz(sequence.joint("right_hip") - sequence.joint("left_hip"))
    yaw = np.degrees(np.arctan2(hips[:, 2], hips[:, 0]))
    # Score coil amount relative to the first finite sample.
    finite = np.isfinite(yaw)
    if not np.any(finite):
        return yaw
    baseline = float(yaw[finite][0])
    return np.abs(yaw - baseline)


def compute_pelvis_rotation_series(pelvis_yaw_deg: np.ndarray) -> np.ndarray:
    """Pelvis rotation magnitude from a yaw series (higher = more rotation)."""
    (series,) = _as_1d(pelvis_yaw_deg)
    finite = np.isfinite(series)
    if not np.any(finite):
        return np.abs(series)
    baseline = float(series[finite][0])
    return np.abs(series - baseline)


def compute_elbow_flexion_series(elbow_interior_deg: np.ndarray) -> np.ndarray:
    """Elbow flexion depth (higher = more bent)."""
    return flexion_from_interior(elbow_interior_deg)


def compute_elbow_extension_series(elbow_interior_deg: np.ndarray) -> np.ndarray:
    """Elbow extension (higher = straighter arm)."""
    return extension_from_interior(elbow_interior_deg)


def compute_shoulder_er_proxy_series_from_skeleton(
    sequence: SkeletonSequence,
    *,
    side: str = "right",
) -> np.ndarray:
    """Shoulder ER proxy: angle between upper arm and trunk (segmentation-style).

    Higher values approximate greater external rotation / layback. This is a
    geometric proxy, not true glenohumeral ER.
    """
    shoulder = sequence.joint(f"{side}_shoulder")
    elbow = sequence.joint(f"{side}_elbow")
    thorax = sequence.joint("thorax")
    upper_arm = elbow - shoulder
    trunk = thorax - shoulder
    return _angle_between(upper_arm, trunk)


def compute_shoulder_er_proxy_series(er_proxy_deg: np.ndarray) -> np.ndarray:
    """Pass-through for a precomputed ER proxy series."""
    (series,) = _as_1d(er_proxy_deg)
    return series.copy()


def compute_forearm_angle_series_from_skeleton(
    sequence: SkeletonSequence,
    *,
    side: str = "right",
) -> np.ndarray:
    """Forearm elevation vs vertical (higher = forearm more raised from down)."""
    elbow = sequence.joint(f"{side}_elbow")
    wrist = sequence.joint(f"{side}_wrist")
    forearm = wrist - elbow
    # Angle from downward vertical (−Y): 0 = hanging down, 90 = horizontal.
    down = np.repeat((-Y_AXIS)[None, :], forearm.shape[0], axis=0)
    return _angle_between(forearm, down)


def compute_forearm_angle_series(forearm_angle_deg: np.ndarray) -> np.ndarray:
    (series,) = _as_1d(forearm_angle_deg)
    return series.copy()


def compute_shoulder_ir_proxy_series(er_proxy_deg: np.ndarray) -> np.ndarray:
    """IR progress proxy: drop from running peak ER (higher = more IR release).

    True glenohumeral IR is unavailable; this tracks how far the ER proxy has
    unwound from its running maximum — useful in Acceleration.
    """
    (er,) = _as_1d(er_proxy_deg)
    out = np.full(er.shape, np.nan, dtype=float)
    peak = -np.inf
    for i, value in enumerate(er):
        if not np.isfinite(value):
            continue
        peak = max(peak, float(value))
        out[i] = peak - float(value)
    return out


def compute_trunk_flexion_series(trunk_tilt_deg: np.ndarray) -> np.ndarray:
    """Trunk flexion from vertical tilt (higher = more flexed/tilted)."""
    (series,) = _as_1d(trunk_tilt_deg)
    return np.abs(series)


def compute_arm_extension_series(hitting_elbow_interior_deg: np.ndarray) -> np.ndarray:
    """Hitting-arm extension for contact (higher = more extended)."""
    return extension_from_interior(hitting_elbow_interior_deg)
