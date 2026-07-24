"""Velocity / deceleration trajectory extractors for the Velocity family.

Trunk and hip use absolute angular speed (deg/s). Shoulder deceleration uses
the positive rate at which rotational *speed* decreases (deg/s²).
"""

from __future__ import annotations

import numpy as np

from src.features.angular_velocity import angular_velocity_from_degrees
from src.skeleton import SkeletonSequence


def _as_1d(values: np.ndarray) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    if series.ndim != 1:
        raise ValueError(f"expected 1D series; got shape {series.shape}")
    return series


def compute_angular_speed_series(
    angle_deg: np.ndarray,
    fps: float | None,
) -> np.ndarray:
    """Absolute angular velocity (deg/s) from an angle trajectory."""
    omega = angular_velocity_from_degrees(_as_1d(angle_deg), fps)
    return np.abs(omega)


def compute_trunk_rotation_velocity_series(
    trunk_rotation_deg: np.ndarray,
    fps: float | None,
) -> np.ndarray:
    """Trunk angular speed (deg/s) from ``trunk_rotation``."""
    return compute_angular_speed_series(trunk_rotation_deg, fps)


def compute_hip_yaw_series_from_skeleton(sequence: SkeletonSequence) -> np.ndarray:
    """Pelvis/hip yaw (degrees) from the left→right hip line in X/Z."""
    hips = sequence.joint("right_hip") - sequence.joint("left_hip")
    return np.degrees(np.arctan2(hips[:, 2], hips[:, 0]))


def compute_hip_rotation_velocity_series(
    hip_yaw_deg: np.ndarray,
    fps: float | None,
) -> np.ndarray:
    """Hip/pelvis angular speed (deg/s) from a yaw trajectory."""
    return compute_angular_speed_series(hip_yaw_deg, fps)


def compute_hip_rotation_velocity_series_from_skeleton(
    sequence: SkeletonSequence,
    fps: float | None = None,
) -> np.ndarray:
    """Hip angular speed from skeleton hip markers."""
    yaw = compute_hip_yaw_series_from_skeleton(sequence)
    return compute_hip_rotation_velocity_series(yaw, fps if fps is not None else sequence.fps)


def compute_deceleration_magnitude_series(
    angle_or_velocity_deg: np.ndarray,
    fps: float | None,
    *,
    input_is_velocity: bool = False,
) -> np.ndarray:
    """Positive deceleration of rotational speed (deg/s²).

    If ``input_is_velocity`` is False, ``angle_or_velocity_deg`` is an angle
    series (deg) that is differentiated to speed first. Deceleration is::

        speed = |ω|
        d(speed)/dt
        deceleration = max(0, −d(speed)/dt)

    so only slowing of rotational magnitude scores, independent of ω sign.
    """
    series = _as_1d(angle_or_velocity_deg)
    if input_is_velocity:
        speed = np.abs(series)
    else:
        speed = np.abs(angular_velocity_from_degrees(series, fps))
    dspeed = angular_velocity_from_degrees(speed, fps)
    return np.maximum(0.0, -dspeed)


def compute_shoulder_deceleration_series(
    shoulder_angle_proxy_deg: np.ndarray,
    fps: float | None,
) -> np.ndarray:
    """Shoulder deceleration magnitude from an angle proxy (e.g. ER proxy).

    Proxy — not true glenohumeral angular deceleration about the IR/ER axis.
    """
    return compute_deceleration_magnitude_series(
        shoulder_angle_proxy_deg, fps, input_is_velocity=False
    )
