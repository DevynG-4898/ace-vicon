"""Shared Path / Posture scoring utilities.

Not a rigid kernel — small helpers reused by Center of Mass, Follow Through,
Balance, Weight Transfer, and Recovery Position. Each feature keeps its own
biomechanical interpretation.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.scoring_engine.result import Direction, FeatureScoreResult
from src.scoring_engine.score_builder import FeatureScoreBuilder

NEUTRAL_CONSISTENCY_SCORE = 75.0
SCALAR_FALLBACK_CONFIDENCE = 0.55
_EPSILON = 1e-6
_RELATIVE_TOLERANCE = 0.15
_DEADZONE_FRACTION = 0.05
_SMOOTHNESS_RATIO_TOLERANCE = 0.5
_CONSISTENCY_CV_TOLERANCE = 0.08


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def gaussian_score(error: float, tolerance: float) -> float:
    tol = max(float(tolerance), _EPSILON)
    return float(clamp(100.0 * np.exp(-0.5 * (float(error) / tol) ** 2)))


def compare_to_reference(
    player_value: float,
    reference_value: float,
    *,
    value_label: str,
    min_tolerance: float,
) -> tuple[float, Direction, dict[str, Any]]:
    """Gaussian magnitude score + direction from signed player−reference error."""
    error = float(player_value) - float(reference_value)
    tolerance = max(float(min_tolerance), _RELATIVE_TOLERANCE * abs(float(reference_value)))
    score = gaussian_score(error, tolerance)
    deadzone = _DEADZONE_FRACTION * tolerance
    if abs(error) <= deadzone:
        direction: Direction = "acceptable"
    elif error < 0:
        direction = "too_low"
    else:
        direction = "too_high"
    return score, direction, {
        f"player_{value_label}": float(player_value),
        f"reference_{value_label}": float(reference_value),
        f"{value_label}_difference": round(error, 4),
        "magnitude_tolerance": round(tolerance, 4),
        "magnitude_score": score,
    }


def resolve_window(series: np.ndarray, phase_slice: slice | None) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    window = arr if phase_slice is None else arr[phase_slice]
    if window.size == 0:
        raise ValueError("Path/posture window is empty")
    return np.asarray(window, dtype=float)


def path_length(points: np.ndarray) -> float:
    """Arc length of an (N, D) trajectory."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 0.0
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def rms(series: np.ndarray) -> float:
    arr = np.asarray(series, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(finite**2)))


def jerk_rms(series: np.ndarray) -> float:
    arr = np.asarray(series, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size < 3:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(finite, n=2) ** 2)))


def smoothness_vs_reference(
    player_series: np.ndarray, reference_series: np.ndarray
) -> tuple[float, dict[str, Any]]:
    player_jerk = jerk_rms(player_series)
    reference_jerk = jerk_rms(reference_series)
    ratio = player_jerk / max(reference_jerk, _EPSILON)
    if ratio <= 1.0:
        score = 100.0
    else:
        score = gaussian_score(ratio - 1.0, _SMOOTHNESS_RATIO_TOLERANCE)
    return score, {
        "player_jerk_rms": round(player_jerk, 6),
        "reference_jerk_rms": round(reference_jerk, 6),
        "smoothness_metric": round(ratio, 6),
        "smoothness_score": score,
    }


def peak_timing_normalized(series: np.ndarray, *, mode: str = "max") -> tuple[float, float, int]:
    """Return (peak_value, τ∈[0,1], local_index)."""
    window = np.asarray(series, dtype=float)
    finite = np.isfinite(window)
    if not np.any(finite):
        raise ValueError("series has no finite samples")
    filled = window.copy()
    filled[~finite] = float(np.median(window[finite]))
    local_idx = int(np.argmax(filled) if mode == "max" else np.argmin(filled))
    peak = float(filled[local_idx])
    tau = local_idx / max(len(window) - 1, 1)
    return peak, tau, local_idx


def timing_score(player_tau: float, reference_tau: float, *, tol: float = 0.15) -> tuple[float, dict[str, Any]]:
    error = player_tau - reference_tau
    score = gaussian_score(error, tol)
    return score, {
        "player_timing_normalized": round(player_tau, 4),
        "reference_timing_normalized": round(reference_tau, 4),
        "timing_error_normalized": round(error, 4),
        "timing_score": score,
    }


def consistency_score(trial_values: Sequence[float] | None) -> tuple[float, dict[str, Any]]:
    if trial_values is None or len(trial_values) < 2:
        return NEUTRAL_CONSISTENCY_SCORE, {
            "consistency_score": NEUTRAL_CONSISTENCY_SCORE,
            "consistency_mode": "single_serve_neutral",
            "trial_value_count": 0 if trial_values is None else len(trial_values),
        }
    values = np.asarray(trial_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return NEUTRAL_CONSISTENCY_SCORE, {
            "consistency_score": NEUTRAL_CONSISTENCY_SCORE,
            "consistency_mode": "single_serve_neutral",
            "trial_value_count": int(values.size),
        }
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    cv = std / max(abs(mean), _EPSILON)
    score = gaussian_score(cv, _CONSISTENCY_CV_TOLERANCE)
    return score, {
        "consistency_score": score,
        "consistency_mode": "multi_serve",
        "trial_value_count": int(values.size),
        "trial_value_mean": round(mean, 4),
        "trial_value_std": round(std, 4),
        "trial_value_cv": round(cv, 6),
    }


def series_confidence(
    series: np.ndarray,
    *,
    joint_confidence: np.ndarray | None = None,
    phase_slice: slice | None = None,
) -> tuple[float, dict[str, Any]]:
    window = resolve_window(series, phase_slice)
    finite_frac = float(np.mean(np.isfinite(window)))
    conf = finite_frac
    joint_mean = None
    if joint_confidence is not None:
        jc = np.asarray(joint_confidence, dtype=float)
        if phase_slice is not None:
            jc = jc[phase_slice]
        finite_jc = jc[np.isfinite(jc)]
        if finite_jc.size:
            joint_mean = float(np.mean(finite_jc))
            conf *= joint_mean
    n = int(window.size)
    length_factor = 0.5 if n < 8 else 0.75 if n < 15 else 1.0
    conf = float(clamp(conf * length_factor))
    return conf, {
        "confidence": conf,
        "finite_sample_fraction": round(finite_frac, 4),
        "mean_joint_confidence": None if joint_mean is None else round(joint_mean, 4),
        "window_frame_count": n,
        "window_length_factor": length_factor,
        "confidence_notes": (
            "Future plugs: MotionBERT confidence, marker_validity_mask, "
            "force-plate / pressure data for balance and weight transfer."
        ),
    }


def build_result(
    *,
    feature_name: str,
    phase: str,
    player_value: float,
    reference_value: float,
    direction: Direction,
    confidence: float,
    components: Sequence[tuple[str, float, float]],
    unit: str,
    measurements: Mapping[str, Any],
) -> FeatureScoreResult:
    """Assemble a FeatureScoreResult from named weighted components."""
    builder = (
        FeatureScoreBuilder(
            feature_name,
            phase,
            unit=unit,
            confidence=confidence,
        )
        .set_direction(direction)
        .set_values(player_value=float(player_value), reference_value=float(reference_value))
    )
    for name, score, weight in components:
        builder.add_component(name, score, weight)
    for key, value in measurements.items():
        builder.add_measurement(str(key), value)
    return builder.build()


def scalar_fallback(
    *,
    feature_name: str,
    phase: str,
    player_value: float,
    reference_value: float,
    value_label: str,
    min_tolerance: float,
    unit: str,
) -> FeatureScoreResult:
    mag, direction, mag_meta = compare_to_reference(
        player_value,
        reference_value,
        value_label=value_label,
        min_tolerance=min_tolerance,
    )
    measurements = {
        "scoring_mode": "scalar_fallback",
        "consistency_score": NEUTRAL_CONSISTENCY_SCORE,
        "consistency_mode": "single_serve_neutral",
        "confidence": SCALAR_FALLBACK_CONFIDENCE,
        "confidence_notes": (
            "Scalar fallback — pass trajectories / phase windows for full "
            "path/posture biomechanics."
        ),
        **mag_meta,
    }
    return build_result(
        feature_name=feature_name,
        phase=phase,
        player_value=float(player_value),
        reference_value=float(reference_value),
        direction=direction,
        confidence=SCALAR_FALLBACK_CONFIDENCE,
        components=(
            ("magnitude", mag, 0.70),
            ("consistency", NEUTRAL_CONSISTENCY_SCORE, 0.30),
        ),
        unit=unit,
        measurements=measurements,
    )
