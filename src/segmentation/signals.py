from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_DTW_DIR = Path(__file__).resolve().parent.parent.parent / "dtw"
if str(_DTW_DIR) not in sys.path:
    sys.path.insert(0, str(_DTW_DIR))

from constants import MARKER_ORDER  # noqa: E402

from segmentation.config import SegmentationConfig


def marker_names(serve: dict) -> list[str]:
    return [k for k in serve if k != "frames"]


def position(serve: dict, marker: str) -> np.ndarray:
    m = serve[marker]
    return np.column_stack(
        [
            m["TX"].astype(float),
            m["TY"].astype(float),
            m["TZ"].astype(float),
        ]
    )


def smooth(series: np.ndarray, window: int) -> np.ndarray:
    w = max(3, window | 1)
    filled = pd.Series(series, dtype=float).interpolate(limit_direction="both").bfill().ffill()
    return filled.rolling(window=w, center=True, min_periods=1).mean().values


def speed(pos: np.ndarray) -> np.ndarray:
    d = np.diff(pos, axis=0)
    sp = np.linalg.norm(d, axis=1)
    return np.concatenate([sp, [sp[-1] if len(sp) else 0.0]])


def angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at vertex b (degrees). Lower = more flexion at knee."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = np.maximum(n1 * n2, 1e-9)
    cosang = np.clip(np.sum(v1 * v2, axis=1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def shoulder_external_rotation_proxy(
    shoulder: np.ndarray, elbow: np.ndarray, chest: np.ndarray
) -> np.ndarray:
    upper_arm = elbow - shoulder
    trunk = chest - shoulder
    n1 = np.linalg.norm(upper_arm, axis=1)
    n2 = np.linalg.norm(trunk, axis=1)
    denom = np.maximum(n1 * n2, 1e-9)
    cosang = np.clip(np.sum(upper_arm * trunk, axis=1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def body_velocity(serve: dict) -> np.ndarray:
    speeds = []
    for name in marker_names(serve):
        if name in MARKER_ORDER:
            speeds.append(speed(position(serve, name)))
    return np.nanmean(np.stack(speeds, axis=0), axis=0)


def shoulder_marker_velocity(serve: dict, cfg: SegmentationConfig) -> np.ndarray:
    """Serving-shoulder linear speed (mm/frame), NaN-interpolated before differencing."""
    pos = position(serve, cfg.serving_shoulder).copy()
    for col in range(3):
        c = pos[:, col]
        bad = np.isnan(c)
        if bad.any():
            good = np.where(~bad)[0]
            if len(good) >= 2:
                c[bad] = np.interp(np.where(bad)[0], good, c[good])
                pos[:, col] = c
    return smooth(speed(pos), cfg.contact_smooth_window)


def upper_body_angular_velocity(signals: dict, cfg: SegmentationConfig) -> np.ndarray:
    """Sum of |shoulder ER rate| and |elbow extension rate| (deg/frame)."""
    d_er = np.abs(signals["shoulder_er_derivative"])
    elbow = signals.get("elbow_extension_angle")
    if elbow is not None:
        return smooth(d_er + np.abs(derivative(elbow)), cfg.contact_smooth_window)
    return smooth(d_er, cfg.contact_smooth_window)


def elbow_extension_angle(serve: dict, cfg: SegmentationConfig) -> np.ndarray:
    sh = position(serve, cfg.serving_shoulder)
    el = position(serve, cfg.serving_elbow)
    ha = position(serve, cfg.serving_hand)
    return angle_at_joint(sh, el, ha)


def hip_descent(serve: dict, cfg: SegmentationConfig) -> np.ndarray:
    return position(serve, cfg.serving_side_hip)[:, 2].astype(float)


def trunk_up_axis(serve: dict) -> np.ndarray:
    """Unit trunk-up vector (hip midpoint → chest) per frame."""
    chest = position(serve, "chest")
    lhip = position(serve, "left_hip")
    rhip = position(serve, "right_hip")
    hip_mid = (lhip + rhip) / 2.0
    trunk = chest - hip_mid
    norm = np.linalg.norm(trunk, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-9)
    return trunk / norm


def trunk_elevation(serve: dict, marker: str) -> np.ndarray:
    """Marker height along trunk-up from hip midpoint (mm)."""
    chest = position(serve, "chest")
    lhip = position(serve, "left_hip")
    rhip = position(serve, "right_hip")
    hip_mid = (lhip + rhip) / 2.0
    trunk_up = trunk_up_axis(serve)
    pos = position(serve, marker)
    return np.sum((pos - hip_mid) * trunk_up, axis=1)


def trunk_tilt_deg(serve: dict) -> np.ndarray:
    """Trunk angle from vertical (degrees). Larger = more tilted away from vertical."""
    chest = position(serve, "chest")
    lhip = position(serve, "left_hip")
    rhip = position(serve, "right_hip")
    hip_mid = (lhip + rhip) / 2.0
    trunk = chest - hip_mid
    norm = np.linalg.norm(trunk, axis=1)
    cosang = np.clip(trunk[:, 2] / np.maximum(norm, 1e-9), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def derivative(series: np.ndarray) -> np.ndarray:
    return np.gradient(np.asarray(series, dtype=float))


def marker_validity_mask(serve: dict) -> np.ndarray:
    """Per-frame fraction of valid marker axes (1.0 = all present)."""
    markers = [m for m in marker_names(serve) if m in MARKER_ORDER]
    n_frames = len(serve["frames"])
    if not markers:
        return np.zeros(n_frames)
    total = len(markers) * 3
    valid = np.zeros(n_frames)
    for marker in markers:
        for axis in ("TX", "TY", "TZ"):
            arr = serve[marker][axis].astype(float)
            valid += ~np.isnan(arr)
    return valid / total


def compute_legacy_signals(serve: dict, cfg: SegmentationConfig) -> dict[str, np.ndarray | None]:
    """Original five signals used by legacy segmentation."""
    hand_pos = position(serve, cfg.serving_hand)
    hand_tz = hand_pos[:, 2]

    knee_angles = []
    for hip, knee, foot in (
        ("right_hip", "right_knee", "right_foot"),
        ("left_hip", "left_knee", "left_foot"),
    ):
        if all(m in serve for m in (hip, knee, foot)):
            knee_angles.append(
                angle_at_joint(
                    position(serve, hip),
                    position(serve, knee),
                    position(serve, foot),
                )
            )
    knee_flexion = np.min(np.stack(knee_angles, axis=0), axis=0) if knee_angles else None

    ser_proxy = shoulder_external_rotation_proxy(
        position(serve, cfg.serving_shoulder),
        position(serve, cfg.serving_elbow),
        position(serve, "chest"),
    )

    body_v = body_velocity(serve)
    hand_v = speed(hand_pos)

    w = cfg.smooth_window
    return {
        "hand_tz": smooth(hand_tz, w),
        "knee_flexion_deg": smooth(knee_flexion, w) if knee_flexion is not None else None,
        "shoulder_er_proxy_deg": smooth(ser_proxy, w),
        "body_velocity": smooth(body_v, w),
        "hand_velocity": smooth(hand_v, w),
    }


def compute_all_signals(serve: dict, cfg: SegmentationConfig) -> dict[str, np.ndarray | None]:
    """Legacy signals plus v2 biomechanical signals."""
    out = dict(compute_legacy_signals(serve, cfg))

    toss_pos = position(serve, cfg.toss_hand)
    toss_tz = toss_pos[:, 2]
    out["toss_hand_height"] = smooth(toss_tz, cfg.toss_smooth_window)
    out["toss_hand_velocity"] = smooth(speed(toss_pos), cfg.smooth_window)
    out["hip_descent"] = smooth(hip_descent(serve, cfg), cfg.smooth_window)
    out["elbow_extension_angle"] = smooth(elbow_extension_angle(serve, cfg), cfg.knee_smooth_window)
    out["marker_validity_mask"] = marker_validity_mask(serve)

    # V2 knee: deepest either leg (unsmoothed min stack for detection; smoothed for output)
    knee_r, knee_l = None, None
    if all(m in serve for m in ("right_hip", "right_knee", "right_foot")):
        knee_r = angle_at_joint(
            position(serve, "right_hip"),
            position(serve, "right_knee"),
            position(serve, "right_foot"),
        )
    if all(m in serve for m in ("left_hip", "left_knee", "left_foot")):
        knee_l = angle_at_joint(
            position(serve, "left_hip"),
            position(serve, "left_knee"),
            position(serve, "left_foot"),
        )
    if knee_r is not None and knee_l is not None:
        knee_min = np.minimum(knee_r, knee_l)
        out["knee_flexion_min_lr_deg"] = smooth(knee_min, cfg.knee_smooth_window)
        out["knee_flexion_right_deg"] = smooth(knee_r, cfg.knee_smooth_window)
        out["knee_flexion_left_deg"] = smooth(knee_l, cfg.knee_smooth_window)
    elif knee_r is not None:
        out["knee_flexion_min_lr_deg"] = smooth(knee_r, cfg.knee_smooth_window)
        out["knee_flexion_right_deg"] = smooth(knee_r, cfg.knee_smooth_window)
        out["knee_flexion_left_deg"] = None
    elif knee_l is not None:
        out["knee_flexion_min_lr_deg"] = smooth(knee_l, cfg.knee_smooth_window)
        out["knee_flexion_right_deg"] = None
        out["knee_flexion_left_deg"] = smooth(knee_l, cfg.knee_smooth_window)
    else:
        out["knee_flexion_min_lr_deg"] = None
        out["knee_flexion_right_deg"] = None
        out["knee_flexion_left_deg"] = None

    # Composite initiation score for v2 E1
    hip_v = speed(position(serve, cfg.serving_side_hip))
    toss_v = speed(toss_pos)
    body_v = out["body_velocity"]
    w1, w2, w3 = cfg.initiation_weights
    out["initiation_score"] = smooth(w1 * hip_v + w2 * toss_v + w3 * body_v, cfg.toss_smooth_window)

    # Racket velocity with shorter smooth for contact
    hand_pos = position(serve, cfg.serving_hand)
    out["racket_hand_velocity"] = smooth(speed(hand_pos), cfg.contact_smooth_window)

    # Coaching phase model signals
    toss_shoulder = cfg.toss_shoulder
    if toss_shoulder in serve:
        out["left_hand_height"] = smooth(trunk_elevation(serve, cfg.toss_hand), cfg.toss_smooth_window)
        out["shoulder_height"] = smooth(trunk_elevation(serve, toss_shoulder), cfg.toss_smooth_window)
    else:
        out["left_hand_height"] = out["toss_hand_height"]
        out["shoulder_height"] = None

    if "head" in serve:
        out["head_height"] = smooth(trunk_elevation(serve, "head"), cfg.smooth_window)
    else:
        out["head_height"] = None

    out["left_hand_velocity"] = out["toss_hand_velocity"]
    out["hip_velocity"] = smooth(speed(position(serve, cfg.serving_side_hip)), cfg.smooth_window)

    knee_for_deriv = out.get("knee_flexion_min_lr_deg")
    if knee_for_deriv is None:
        knee_for_deriv = out.get("knee_flexion_deg")
    if knee_for_deriv is not None:
        out["knee_flexion_derivative"] = derivative(knee_for_deriv)
    else:
        out["knee_flexion_derivative"] = None

    out["shoulder_er_derivative"] = derivative(out["shoulder_er_proxy_deg"])
    out["shoulder_velocity"] = shoulder_marker_velocity(serve, cfg)
    out["upper_body_angular_velocity"] = upper_body_angular_velocity(out, cfg)
    out["trunk_tilt_deg"] = smooth(trunk_tilt_deg(serve), cfg.smooth_window)
    if out.get("elbow_extension_angle") is not None:
        out["racket_laid_back_score"] = smooth(
            cfg.cocking_elbow_extension_max_deg - out["elbow_extension_angle"],
            cfg.knee_smooth_window,
        )
    else:
        out["racket_laid_back_score"] = None

    return out


def racket_hand_velocity_series(signals: dict) -> np.ndarray:
    rv = signals.get("racket_hand_velocity")
    if rv is not None:
        return rv
    return signals["hand_velocity"]


def clip_index(idx: int, n: int) -> int:
    return int(min(max(idx, 0), max(n - 1, 0)))


def persist_above(series: np.ndarray, threshold: float, persist: int, start: int = 0) -> int | None:
    run = 0
    for i in range(start, len(series)):
        if series[i] > threshold:
            run += 1
            if run >= persist:
                return i - persist + 1
        else:
            run = 0
    return None


def persist_below(series: np.ndarray, threshold: float, persist: int, start: int = 0) -> int | None:
    run = 0
    for i in range(start, len(series)):
        if series[i] < threshold:
            run += 1
            if run >= persist:
                return i - persist + 1
        else:
            run = 0
    return None


def argmax_in_range(series: np.ndarray, low: int, high: int, mode: str = "max") -> int:
    low = max(0, low)
    high = min(high, len(series) - 1)
    if high < low:
        return low
    segment = np.asarray(series[low : high + 1], dtype=float)
    if np.all(np.isnan(segment)):
        return low + len(segment) // 2
    if mode == "max":
        return low + int(np.nanargmax(segment))
    return low + int(np.nanargmin(segment))

