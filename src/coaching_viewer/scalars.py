"""Feature Extraction Adapter: segmented ACE markers → ScoringEngine scalars.

Produces one finite scalar per ``FEATURE_SCORERS`` key. Does not score or coach.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from segmentation.result import SegmentationResult, phase_to_index_range
from src.coaching_viewer.markers_to_skeleton import ace_markers_to_skeleton
from src.features.joint_angles import (
    hip_flexion,
    joint_angle,
    line_angle_xy,
    trunk_rotation,
    trunk_tilt,
)
from src.features.velocity import finite_difference, speed_from_velocity
from src.scoring_engine.angle_series import (
    compute_elbow_extension_series,
    compute_elbow_flexion_series,
    compute_forearm_angle_series_from_skeleton,
    compute_pelvis_rotation_series_from_skeleton,
    compute_shoulder_er_proxy_series_from_skeleton,
    compute_shoulder_ir_proxy_series,
    compute_shoulder_tilt_series,
)
from src.scoring_engine.contact_series import contact_height, contact_position_offsets
from src.scoring_engine.feature_scorer import FEATURE_SCORERS
from src.scoring_engine.knee_flexion import compute_knee_flexion_series
from src.scoring_engine.path_posture import path_length, rms
from src.scoring_engine.velocity_series import (
    compute_angular_speed_series,
    compute_deceleration_magnitude_series,
    compute_hip_yaw_series_from_skeleton,
)
from src.skeleton import SkeletonSequence

_DEFAULT_FPS = 100.0


def _phase_bounds(seg: SegmentationResult, *candidates: str) -> tuple[int, int]:
    for name in candidates:
        if name in seg.phases:
            return seg.phases[name]
    raise ValueError(f"Missing phase window; tried {candidates}")


def _phase_slice(seg: SegmentationResult, *candidates: str) -> slice:
    bounds = _phase_bounds(seg, *candidates)
    i0, i1 = phase_to_index_range(seg.frames, bounds)
    return slice(i0, i1 + 1)


def _contact_index(seg: SegmentationResult) -> int:
    if "contact" in seg.event_indices:
        return int(seg.event_indices["contact"])
    if "contact" in seg.events:
        return int(phase_to_index_range(seg.frames, (seg.events["contact"], seg.events["contact"]))[0])
    # Fall back to start of Contact phase
    sl = _phase_slice(seg, "Contact")
    return int(sl.start)


def _peak_max(series: np.ndarray, sl: slice) -> float:
    window = np.asarray(series[sl], dtype=float)
    finite = window[np.isfinite(window)]
    if finite.size == 0:
        raise ValueError("empty peak window")
    return float(np.nanmax(finite))


def _peak_abs_max(series: np.ndarray, sl: slice) -> float:
    window = np.asarray(series[sl], dtype=float)
    finite = window[np.isfinite(window)]
    if finite.size == 0:
        raise ValueError("empty peak window")
    return float(np.nanmax(np.abs(finite)))


def _value_at(series: np.ndarray, idx: int) -> float:
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        raise ValueError("empty series")
    i = int(np.clip(idx, 0, arr.size - 1))
    value = float(arr[i])
    if np.isfinite(value):
        return value
    finite_idx = np.flatnonzero(np.isfinite(arr))
    if finite_idx.size == 0:
        raise ValueError(f"non-finite value at index {i}")
    nearest = int(finite_idx[np.argmin(np.abs(finite_idx - i))])
    return float(arr[nearest])


def extract_feature_scalars(
    markers: dict,
    segmentation: SegmentationResult,
    *,
    handedness: str = "right",
    fps: float | None = None,
) -> dict[str, float]:
    """Extract one scalar per ScoringEngine feature from a segmented serve."""
    rate = float(fps) if fps is not None else _DEFAULT_FPS
    seq = ace_markers_to_skeleton(markers, fps=rate)
    hit = "right" if handedness != "left" else "left"
    toss = "left" if hit == "right" else "right"

    loading = _phase_slice(segmentation, "Loading")
    cocking = _phase_slice(segmentation, "Cocking")
    accel = _phase_slice(segmentation, "Acceleration")
    contact_sl = _phase_slice(segmentation, "Contact")
    decel = _phase_slice(segmentation, "Deceleration_Finish", "Deceleration", "Finish")
    finish = _phase_slice(segmentation, "Finish", "Deceleration_Finish")
    contact_i = _contact_index(segmentation)

    left_knee = joint_angle(seq, "left_hip", "left_knee", "left_ankle")
    right_knee = joint_angle(seq, "right_hip", "right_knee", "right_ankle")
    knee_flex = compute_knee_flexion_series(left_knee, right_knee)

    left_elbow = joint_angle(seq, "left_shoulder", "left_elbow", "left_wrist")
    right_elbow = joint_angle(seq, "right_shoulder", "right_elbow", "right_wrist")
    hit_elbow = right_elbow if hit == "right" else left_elbow
    toss_elbow = left_elbow if toss == "left" else right_elbow

    trunk_rot = trunk_rotation(seq)
    trunk_tilt_s = trunk_tilt(seq)
    shoulder_line = line_angle_xy(seq, "left_shoulder", "right_shoulder")
    er_proxy = compute_shoulder_er_proxy_series_from_skeleton(seq, side=hit)
    ir_proxy = compute_shoulder_ir_proxy_series(er_proxy)
    forearm = compute_forearm_angle_series_from_skeleton(seq, side=hit)
    pelvis_rot = compute_pelvis_rotation_series_from_skeleton(seq)
    hip_flex = hip_flexion(seq)

    trunk_speed = compute_angular_speed_series(trunk_rot, rate)
    hip_yaw = compute_hip_yaw_series_from_skeleton(seq)
    hip_speed = compute_angular_speed_series(hip_yaw, rate)
    shoulder_decel = compute_deceleration_magnitude_series(er_proxy, rate)

    hit_wrist = seq.joint(f"{hit}_wrist")
    pelvis = seq.joint("pelvis")
    hit_shoulder = seq.joint(f"{hit}_shoulder")

    # Center of mass proxy: pelvis vertical drop in Loading
    pelvis_y = pelvis[:, 1]
    load_y = pelvis_y[loading]
    if load_y.size == 0 or not np.any(np.isfinite(load_y)):
        raise ValueError("Center of Mass: empty Loading window")
    com_drop = float(load_y[0] - np.nanmin(load_y))

    # Follow through: hitting wrist path length after contact
    follow_pts = hit_wrist[decel]
    follow_through = path_length(follow_pts)

    # Balance: inverse of pelvis horizontal RMS in finish (higher = more stable)
    finish_pelvis = pelvis[finish]
    horiz = finish_pelvis[:, [0, 2]] - finish_pelvis[0:1, [0, 2]]
    sway = rms(np.linalg.norm(horiz, axis=1)) if horiz.size else 0.0
    balance = 100.0 / (1.0 + sway / 50.0)

    # Weight transfer: forward (Z) pelvis displacement after contact
    post = pelvis[contact_i:]
    if post.shape[0] < 2:
        weight_transfer = 0.0
    else:
        start_z = post[0, 2]
        end_z = post[-1, 2]
        if not np.isfinite(start_z) or not np.isfinite(end_z):
            finite_z = post[np.isfinite(post[:, 2]), 2]
            weight_transfer = float(finite_z[-1] - finite_z[0]) if finite_z.size >= 2 else 0.0
        else:
            weight_transfer = float(end_z - start_z)

    # Recovery: fraction of post-contact time until foot speed settles
    recovery = _recovery_fraction(seq, contact_i, rate)

    height = contact_height(hit_wrist, contact_i, vertical_axis=1)
    offsets = contact_position_offsets(hit_wrist, pelvis, contact_i)
    arm_ext = compute_elbow_extension_series(hit_elbow)
    body_align = abs(_value_at(trunk_rot, contact_i))

    values: dict[str, float] = {
        "Knee Flexion": _peak_max(knee_flex, loading),
        "Hip Flexion": _peak_max(hip_flex, loading),
        "Shoulder Tilt": _peak_max(compute_shoulder_tilt_series(shoulder_line), loading),
        "Toss Arm Extension": _peak_max(compute_elbow_extension_series(toss_elbow), loading),
        "Center of Mass": com_drop,
        "Trunk Rotation": _peak_abs_max(trunk_rot, loading),
        "Pelvis Rotation": _peak_max(pelvis_rot, loading),
        "Right Elbow Flexion": _peak_max(compute_elbow_flexion_series(right_elbow), cocking),
        "Left Elbow Flexion": _peak_max(compute_elbow_flexion_series(left_elbow), cocking),
        "Shoulder External Rotation": _peak_max(er_proxy, cocking),
        "Forearm Angle": _peak_max(forearm, cocking),
        "Shoulder Internal Rotation": _peak_max(ir_proxy, accel),
        "Right Elbow Extension": _peak_max(compute_elbow_extension_series(right_elbow), accel),
        "Left Elbow Extension": _peak_max(compute_elbow_extension_series(left_elbow), accel),
        "Trunk Rotation Velocity": _peak_max(trunk_speed, accel),
        "Hip Rotation Velocity": _peak_max(hip_speed, accel),
        "Contact Height": float(height),
        "Contact Position": float(offsets["forward"]),
        "Arm Extension": _value_at(arm_ext, contact_i),
        "Body Alignment": body_align,
        "Follow Through": follow_through,
        "Shoulder Deceleration": _peak_max(shoulder_decel, decel),
        "Trunk Flexion": _peak_abs_max(trunk_tilt_s, decel),
        "Balance": balance,
        "Weight Transfer": weight_transfer,
        "Recovery Position": recovery,
    }

    # Silence unused local kept for clarity / future event windows
    _ = (contact_sl, hit_shoulder)

    missing = [k for k in FEATURE_SCORERS if k not in values]
    if missing:
        raise ValueError("Missing scalar(s): " + ", ".join(sorted(missing)))
    for name, value in values.items():
        if not np.isfinite(value):
            raise ValueError(f"Non-finite scalar for {name}: {value}")
    return values


def _recovery_fraction(seq: SkeletonSequence, contact_i: int, fps: float) -> float:
    """Return fraction of post-contact duration until foot speed drops below threshold."""
    n = seq.n_frames
    if contact_i >= n - 2:
        return 0.4
    left = seq.joint("left_ankle")
    right = seq.joint("right_ankle")
    feet = 0.5 * (left + right)
    vel = finite_difference(feet, fps)
    speed = speed_from_velocity(vel)
    post = speed[contact_i:]
    if post.size < 2:
        return 0.4
    peak = float(np.nanmax(post))
    if not np.isfinite(peak) or peak <= 0:
        return 0.4
    threshold = 0.15 * peak
    settled = np.where(post < threshold)[0]
    if settled.size == 0:
        return 1.0
    # First settle after the initial peak
    peak_i = int(np.nanargmax(post))
    after = settled[settled > peak_i]
    idx = int(after[0]) if after.size else int(settled[-1])
    return float(idx / max(post.size - 1, 1))
