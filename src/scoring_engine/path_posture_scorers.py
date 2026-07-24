"""Path / Posture feature scorers (CoM, Follow Through, Balance, Weight Transfer, Recovery)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.features.angular_velocity import angular_velocity_from_degrees
from src.features.velocity import finite_difference
from src.scoring_engine.path_posture import (
    build_result,
    compare_to_reference,
    consistency_score,
    path_length,
    peak_timing_normalized,
    resolve_window,
    rms,
    scalar_fallback,
    series_confidence,
    smoothness_vs_reference,
    timing_score,
)
from src.scoring_engine.result import FeatureScoreResult

_COM_MIN_TOL = 0.02
_FOLLOW_MIN_TOL = 0.05
_BALANCE_MIN_TOL = 0.01
_TRANSFER_MIN_TOL = 0.05
_RECOVERY_TIME_MIN_TOL = 0.05
_RECOVERY_POSE_MIN_TOL = 0.05


def score_center_of_mass_biomechanics(
    *,
    player_com: np.ndarray,
    reference_com: np.ndarray,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_drops: Sequence[float] | None = None,
    joint_confidence: np.ndarray | None = None,
    vertical_axis: int = 1,
    fps: float | None = None,
) -> FeatureScoreResult:
    """Score CoM lowering in Loading: vertical drop + timing + smoothness."""
    player = np.asarray(player_com, dtype=float)
    reference = np.asarray(reference_com, dtype=float)
    p_win = resolve_window(player, player_phase_slice)
    r_win = resolve_window(
        reference,
        reference_phase_slice if reference_phase_slice is not None else player_phase_slice,
    )

    p_y = p_win[:, vertical_axis]
    r_y = r_win[:, vertical_axis]
    player_drop = float(p_y[0] - np.nanmin(p_y)) if p_y.size else 0.0
    reference_drop = float(r_y[0] - np.nanmin(r_y)) if r_y.size else 0.0

    player_drop_series = p_y[0] - p_y
    reference_drop_series = r_y[0] - r_y
    _, p_tau, p_idx = peak_timing_normalized(player_drop_series, mode="max")
    _, r_tau, r_idx = peak_timing_normalized(reference_drop_series, mode="max")

    mag, direction, mag_meta = compare_to_reference(
        player_drop, reference_drop, value_label="com_drop", min_tolerance=_COM_MIN_TOL
    )
    t_score, t_meta = timing_score(p_tau, r_tau)
    s_score, s_meta = smoothness_vs_reference(p_y, r_y)
    c_score, c_meta = consistency_score(player_trial_drops)
    conf, conf_meta = series_confidence(
        player[:, vertical_axis],
        joint_confidence=joint_confidence,
        phase_slice=player_phase_slice,
    )

    horiz_axes = [i for i in range(p_win.shape[1]) if i != vertical_axis]
    player_horiz = path_length(p_win[:, horiz_axes]) if horiz_axes else 0.0
    reference_horiz = path_length(r_win[:, horiz_axes]) if horiz_axes else 0.0

    measurements = {
        "scoring_mode": "biomechanical",
        "fps": fps,
        "player_peak_local_index": p_idx,
        "reference_peak_local_index": r_idx,
        "player_horizontal_path": round(player_horiz, 6),
        "reference_horizontal_path": round(reference_horiz, 6),
        "vertical_axis": vertical_axis,
        **mag_meta,
        **t_meta,
        **s_meta,
        **c_meta,
        **conf_meta,
    }
    return build_result(
        feature_name="Center of Mass",
        phase="Loading",
        player_value=player_drop,
        reference_value=reference_drop,
        direction=direction,
        confidence=conf,
        components=(
            ("magnitude", mag, 0.50),
            ("timing", t_score, 0.20),
            ("smoothness", s_score, 0.15),
            ("consistency", c_score, 0.15),
        ),
        unit="normalized",
        measurements=measurements,
    )


def score_center_of_mass_from_scalars(
    player_value: float, reference_value: float
) -> FeatureScoreResult:
    return scalar_fallback(
        feature_name="Center of Mass",
        phase="Loading",
        player_value=player_value,
        reference_value=reference_value,
        value_label="com_drop",
        min_tolerance=_COM_MIN_TOL,
        unit="mm",
    )


def score_follow_through_biomechanics(
    *,
    player_hand: np.ndarray,
    reference_hand: np.ndarray,
    player_contact_frame: int,
    reference_contact_frame: int,
    player_phase_end: int | None = None,
    reference_phase_end: int | None = None,
    player_trial_path_lengths: Sequence[float] | None = None,
    joint_confidence: np.ndarray | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    """Score follow-through from hand path length and smoothness after contact."""
    player = np.asarray(player_hand, dtype=float)
    reference = np.asarray(reference_hand, dtype=float)
    p_end = player_phase_end if player_phase_end is not None else player.shape[0]
    r_end = reference_phase_end if reference_phase_end is not None else reference.shape[0]
    p_seg = player[int(player_contact_frame) : int(p_end)]
    r_seg = reference[int(reference_contact_frame) : int(r_end)]
    if p_seg.shape[0] < 2 or r_seg.shape[0] < 2:
        raise ValueError("Follow-through segment must contain at least 2 frames")

    player_len = path_length(p_seg)
    reference_len = path_length(r_seg)
    mag, direction, mag_meta = compare_to_reference(
        player_len, reference_len, value_label="path_length", min_tolerance=_FOLLOW_MIN_TOL
    )

    p_speed = np.linalg.norm(finite_difference(p_seg, fps), axis=1)
    r_speed = np.linalg.norm(finite_difference(r_seg, fps), axis=1)
    s_score, s_meta = smoothness_vs_reference(p_speed, r_speed)

    p_ds = angular_velocity_from_degrees(p_speed, fps)
    r_ds = angular_velocity_from_degrees(r_speed, fps)
    p_decel = float(np.nanmax(np.maximum(0.0, -p_ds))) if p_ds.size else 0.0
    r_decel = float(np.nanmax(np.maximum(0.0, -r_ds))) if r_ds.size else 0.0
    decel_score, _, decel_meta = compare_to_reference(
        p_decel, r_decel, value_label="hand_deceleration", min_tolerance=10.0
    )

    c_score, c_meta = consistency_score(player_trial_path_lengths)
    conf, conf_meta = series_confidence(
        player[:, 0],
        joint_confidence=joint_confidence,
        phase_slice=slice(int(player_contact_frame), int(p_end)),
    )

    blended_mag = 0.65 * mag + 0.35 * decel_score
    measurements = {
        "scoring_mode": "biomechanical",
        "fps": fps,
        "player_contact_frame": int(player_contact_frame),
        "reference_contact_frame": int(reference_contact_frame),
        "deceleration_score": round(decel_score, 2),
        "magnitude_score": round(blended_mag, 2),
        **mag_meta,
        **{f"decel_{k}": v for k, v in decel_meta.items() if k != "magnitude_score"},
        **s_meta,
        **c_meta,
        **conf_meta,
    }
    return build_result(
        feature_name="Follow Through",
        phase="Deceleration",
        player_value=player_len,
        reference_value=reference_len,
        direction=direction,
        confidence=conf,
        components=(
            ("magnitude", blended_mag, 0.55),
            ("smoothness", s_score, 0.25),
            ("consistency", c_score, 0.20),
        ),
        unit="normalized",
        measurements=measurements,
    )


def score_follow_through_from_scalars(
    player_value: float, reference_value: float
) -> FeatureScoreResult:
    return scalar_fallback(
        feature_name="Follow Through",
        phase="Deceleration",
        player_value=player_value,
        reference_value=reference_value,
        value_label="path_length",
        min_tolerance=_FOLLOW_MIN_TOL,
        unit="%",
    )


def score_balance_biomechanics(
    *,
    player_com: np.ndarray,
    reference_com: np.ndarray,
    player_left_foot: np.ndarray,
    player_right_foot: np.ndarray,
    reference_left_foot: np.ndarray,
    reference_right_foot: np.ndarray,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_sway: Sequence[float] | None = None,
    joint_confidence: np.ndarray | None = None,
    fps: float | None = None,
    vertical_axis: int = 1,
) -> FeatureScoreResult:
    """Score balance from CoM sway and distance to the foot-midpoint support proxy."""
    player = np.asarray(player_com, dtype=float)
    reference = np.asarray(reference_com, dtype=float)
    p_win = resolve_window(player, player_phase_slice)
    r_win = resolve_window(
        reference,
        reference_phase_slice if reference_phase_slice is not None else player_phase_slice,
    )
    horiz = [i for i in range(p_win.shape[1]) if i != vertical_axis]

    p_vel = finite_difference(p_win, fps)
    r_vel = finite_difference(r_win, fps)
    player_sway = rms(np.linalg.norm(p_vel[:, horiz], axis=1)) if horiz else 0.0
    reference_sway = rms(np.linalg.norm(r_vel[:, horiz], axis=1)) if horiz else 0.0

    def _support_error(com, left_foot, right_foot, slc):
        c = resolve_window(com, slc)
        lf = resolve_window(left_foot, slc)
        rf = resolve_window(right_foot, slc)
        mid = 0.5 * (lf + rf)
        delta = c[:, horiz] - mid[:, horiz]
        return rms(np.linalg.norm(delta, axis=1))

    player_support = _support_error(
        player, player_left_foot, player_right_foot, player_phase_slice
    )
    reference_support = _support_error(
        reference, reference_left_foot, reference_right_foot, reference_phase_slice
    )

    sway_score, direction, sway_meta = compare_to_reference(
        player_sway, reference_sway, value_label="sway_rms", min_tolerance=_BALANCE_MIN_TOL
    )
    support_score, _, support_meta = compare_to_reference(
        player_support,
        reference_support,
        value_label="support_error",
        min_tolerance=_BALANCE_MIN_TOL,
    )
    s_score, s_meta = smoothness_vs_reference(
        np.linalg.norm(p_win[:, horiz], axis=1) if horiz else p_win[:, 0],
        np.linalg.norm(r_win[:, horiz], axis=1) if horiz else r_win[:, 0],
    )
    c_score, c_meta = consistency_score(player_trial_sway)
    conf, conf_meta = series_confidence(
        player[:, 0], joint_confidence=joint_confidence, phase_slice=player_phase_slice
    )

    magnitude = 0.60 * sway_score + 0.40 * support_score
    measurements = {
        "scoring_mode": "biomechanical",
        "fps": fps,
        "magnitude_score": round(magnitude, 2),
        **sway_meta,
        **{k: v for k, v in support_meta.items() if k != "magnitude_score"},
        **s_meta,
        **c_meta,
        **conf_meta,
    }
    return build_result(
        feature_name="Balance",
        phase="Finish",
        player_value=player_sway,
        reference_value=reference_sway,
        direction=direction,
        confidence=conf,
        components=(
            ("magnitude", magnitude, 0.55),
            ("smoothness", s_score, 0.25),
            ("consistency", c_score, 0.20),
        ),
        unit="normalized",
        measurements=measurements,
    )


def score_balance_from_scalars(player_value: float, reference_value: float) -> FeatureScoreResult:
    return scalar_fallback(
        feature_name="Balance",
        phase="Finish",
        player_value=player_value,
        reference_value=reference_value,
        value_label="balance",
        min_tolerance=1.0,
        unit="%",
    )


def score_weight_transfer_biomechanics(
    *,
    player_com: np.ndarray,
    reference_com: np.ndarray,
    player_back_foot: np.ndarray,
    player_front_foot: np.ndarray,
    reference_back_foot: np.ndarray,
    reference_front_foot: np.ndarray,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_transfer: Sequence[float] | None = None,
    joint_confidence: np.ndarray | None = None,
    fps: float | None = None,
    forward_axis: int = 2,
) -> FeatureScoreResult:
    """Score A-P weight transfer as CoM progress from back foot toward front foot."""

    def _transfer_fraction(com, back_foot, front_foot, slc):
        c = resolve_window(com, slc)[:, forward_axis]
        b = resolve_window(back_foot, slc)[:, forward_axis]
        f = resolve_window(front_foot, slc)[:, forward_axis]
        back = float(np.nanmean(b[: max(1, len(b) // 4)]))
        front = float(np.nanmean(f[-max(1, len(f) // 4) :]))
        span = front - back
        if abs(span) < 1e-8:
            return 0.0, c, 0.0
        frac_series = (c - back) / span
        final_frac = float(np.nanmean(frac_series[-max(1, len(frac_series) // 5) :]))
        hits = np.where(np.isfinite(frac_series) & (frac_series >= 0.5))[0]
        tau = float(hits[0] / max(len(frac_series) - 1, 1)) if hits.size else 1.0
        return final_frac, frac_series, tau

    p_frac, p_series, p_tau = _transfer_fraction(
        player_com, player_back_foot, player_front_foot, player_phase_slice
    )
    r_frac, r_series, r_tau = _transfer_fraction(
        reference_com, reference_back_foot, reference_front_foot, reference_phase_slice
    )

    mag, direction, mag_meta = compare_to_reference(
        p_frac, r_frac, value_label="transfer_fraction", min_tolerance=_TRANSFER_MIN_TOL
    )
    t_score, t_meta = timing_score(p_tau, r_tau)
    s_score, s_meta = smoothness_vs_reference(p_series, r_series)
    c_score, c_meta = consistency_score(player_trial_transfer)
    conf, conf_meta = series_confidence(
        np.asarray(player_com, dtype=float)[:, forward_axis],
        joint_confidence=joint_confidence,
        phase_slice=player_phase_slice,
    )

    measurements = {
        "scoring_mode": "biomechanical",
        "fps": fps,
        "forward_axis": forward_axis,
        **mag_meta,
        **t_meta,
        **s_meta,
        **c_meta,
        **conf_meta,
    }
    return build_result(
        feature_name="Weight Transfer",
        phase="Finish",
        player_value=p_frac,
        reference_value=r_frac,
        direction=direction,
        confidence=conf,
        components=(
            ("magnitude", mag, 0.50),
            ("timing", t_score, 0.20),
            ("smoothness", s_score, 0.15),
            ("consistency", c_score, 0.15),
        ),
        unit="fraction",
        measurements=measurements,
    )


def score_weight_transfer_from_scalars(
    player_value: float, reference_value: float
) -> FeatureScoreResult:
    return scalar_fallback(
        feature_name="Weight Transfer",
        phase="Finish",
        player_value=player_value,
        reference_value=reference_value,
        value_label="transfer_fraction",
        min_tolerance=_TRANSFER_MIN_TOL,
        unit="%",
    )


def score_recovery_position_biomechanics(
    *,
    player_com: np.ndarray,
    reference_com: np.ndarray,
    player_left_foot: np.ndarray,
    player_right_foot: np.ndarray,
    reference_left_foot: np.ndarray,
    reference_right_foot: np.ndarray,
    player_contact_frame: int,
    reference_contact_frame: int,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_recovery_times: Sequence[float] | None = None,
    joint_confidence: np.ndarray | None = None,
    fps: float | None = None,
    vertical_axis: int = 1,
    speed_threshold: float = 0.15,
) -> FeatureScoreResult:
    """Score recovery via time-to-stable CoM and final ready-pose similarity."""
    player = np.asarray(player_com, dtype=float)
    reference = np.asarray(reference_com, dtype=float)
    fps_val = float(fps) if fps and fps > 0 else 1.0

    def _recovery_time(com, contact_frame, slc):
        window = resolve_window(com, slc)
        start = 0 if slc is None else (slc.start or 0)
        local_contact = max(0, int(contact_frame) - int(start))
        vel = finite_difference(window, fps)
        speed = np.linalg.norm(vel, axis=1)
        after = speed[local_contact:]
        hits = np.where(np.isfinite(after) & (after <= speed_threshold))[0]
        if hits.size == 0:
            return float(len(after) / fps_val), speed
        return float(hits[0] / fps_val), speed

    p_time, p_speed = _recovery_time(player, player_contact_frame, player_phase_slice)
    r_time, r_speed = _recovery_time(reference, reference_contact_frame, reference_phase_slice)

    # KL too_low = recover more quickly → score recovery_speed = 1/time
    p_speed_metric = 1.0 / max(p_time, 1e-3)
    r_speed_metric = 1.0 / max(r_time, 1e-3)
    time_score, direction, time_meta = compare_to_reference(
        p_speed_metric,
        r_speed_metric,
        value_label="recovery_speed",
        min_tolerance=_RECOVERY_TIME_MIN_TOL,
    )

    def _pose(com, left_f, right_f, slc):
        c = resolve_window(com, slc)
        lf = resolve_window(left_f, slc)
        rf = resolve_window(right_f, slc)
        height = float(c[-1, vertical_axis])
        width = float(np.linalg.norm(lf[-1, [0, 2]] - rf[-1, [0, 2]]))
        return height, width

    p_h, p_w = _pose(player, player_left_foot, player_right_foot, player_phase_slice)
    r_h, r_w = _pose(
        reference, reference_left_foot, reference_right_foot, reference_phase_slice
    )
    height_score, _, height_meta = compare_to_reference(
        p_h, r_h, value_label="ready_com_height", min_tolerance=_RECOVERY_POSE_MIN_TOL
    )
    width_score, _, width_meta = compare_to_reference(
        p_w, r_w, value_label="ready_stance_width", min_tolerance=_RECOVERY_POSE_MIN_TOL
    )
    pose_score = 0.5 * height_score + 0.5 * width_score

    s_score, s_meta = smoothness_vs_reference(p_speed, r_speed)
    c_score, c_meta = consistency_score(player_trial_recovery_times)
    conf, conf_meta = series_confidence(
        player[:, vertical_axis],
        joint_confidence=joint_confidence,
        phase_slice=player_phase_slice,
    )

    magnitude = 0.60 * time_score + 0.40 * pose_score
    measurements = {
        "scoring_mode": "biomechanical",
        "fps": fps,
        "player_recovery_time_s": round(p_time, 4),
        "reference_recovery_time_s": round(r_time, 4),
        "player_contact_frame": int(player_contact_frame),
        "reference_contact_frame": int(reference_contact_frame),
        "pose_score": round(pose_score, 2),
        "magnitude_score": round(magnitude, 2),
        "speed_threshold": speed_threshold,
        **time_meta,
        **{k: v for k, v in height_meta.items() if k != "magnitude_score"},
        **{k: v for k, v in width_meta.items() if k != "magnitude_score"},
        **s_meta,
        **c_meta,
        **conf_meta,
    }
    return build_result(
        feature_name="Recovery Position",
        phase="Finish",
        player_value=p_time,
        reference_value=r_time,
        direction=direction,
        confidence=conf,
        components=(
            ("magnitude", magnitude, 0.55),
            ("smoothness", s_score, 0.20),
            ("consistency", c_score, 0.25),
        ),
        unit="s",
        measurements=measurements,
    )


def score_recovery_position_from_scalars(
    player_value: float, reference_value: float
) -> FeatureScoreResult:
    p_speed = 1.0 / max(float(player_value), 1e-3)
    r_speed = 1.0 / max(float(reference_value), 1e-3)
    result = scalar_fallback(
        feature_name="Recovery Position",
        phase="Finish",
        player_value=p_speed,
        reference_value=r_speed,
        value_label="recovery_speed",
        min_tolerance=_RECOVERY_TIME_MIN_TOL,
        unit="s",
    )
    return build_result(
        feature_name="Recovery Position",
        phase="Finish",
        player_value=float(player_value),
        reference_value=float(reference_value),
        direction=result.direction,
        confidence=result.confidence,
        components=(
            ("magnitude", result.measurements["magnitude_score"], 0.70),
            ("consistency", result.measurements["consistency_score"], 0.30),
        ),
        unit="s",
        measurements={
            **result.measurements,
            "player_recovery_time_s": float(player_value),
            "reference_recovery_time_s": float(reference_value),
        },
    )
