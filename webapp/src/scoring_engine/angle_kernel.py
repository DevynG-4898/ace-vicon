"""Reusable Angle Kernel for phase-window angle trajectory scoring.

Generic biomechanical scoring shared by Knee Flexion and future angle-based
features (hip, elbow, trunk, shoulder, pelvis, …). Joint-specific series
preparation stays in each feature module; this kernel only scores trajectories.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from src.scoring_engine.result import Direction, FeatureScoreResult
from src.scoring_engine.score_builder import DEFAULT_COMPONENT_WEIGHTS, FeatureScoreBuilder

NEUTRAL_CONSISTENCY_SCORE = 75.0
NEUTRAL_COMPONENT_SCORE = 75.0
SCALAR_FALLBACK_CONFIDENCE = 0.55

_MAGNITUDE_MIN_TOLERANCE_DEG = 8.0
_MAGNITUDE_RELATIVE_TOLERANCE = 0.15
_TIMING_TOLERANCE = 0.15
_SMOOTHNESS_RATIO_TOLERANCE = 0.5
_CONSISTENCY_CV_TOLERANCE = 0.08
_DIRECTION_DEADZONE_FRACTION = 0.05
_EPSILON = 1e-6

PeakMode = Literal["max", "min"]


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _gaussian_score(error: float, tolerance: float) -> float:
    tol = max(float(tolerance), _EPSILON)
    return float(_clamp(100.0 * np.exp(-0.5 * (float(error) / tol) ** 2)))


def resolve_phase_window(
    series: np.ndarray,
    phase_slice: slice | None,
) -> tuple[np.ndarray, slice]:
    """Return the phase sub-series and the resolved slice."""
    if phase_slice is None:
        resolved = slice(0, len(series))
    else:
        resolved = phase_slice
    window = np.asarray(series[resolved], dtype=float)
    if window.size == 0:
        raise ValueError("Phase window is empty")
    return window, resolved


def peak_and_timing(
    window: np.ndarray,
    *,
    peak_mode: PeakMode = "max",
) -> tuple[float, float, int]:
    """Return peak value, normalized timing τ ∈ [0, 1], and local index."""
    finite = np.isfinite(window)
    if not np.any(finite):
        raise ValueError("Angle window has no finite samples")
    filled = window.copy()
    filled[~finite] = float(np.median(window[finite]))
    if peak_mode == "max":
        local_idx = int(np.argmax(filled))
    elif peak_mode == "min":
        local_idx = int(np.argmin(filled))
    else:
        raise ValueError(f"peak_mode must be 'max' or 'min'; got {peak_mode!r}")
    peak = float(filled[local_idx])
    denom = max(len(window) - 1, 1)
    timing = local_idx / denom
    return peak, timing, local_idx


def _jerk_rms(window: np.ndarray) -> float:
    finite = window[np.isfinite(window)]
    if finite.size < 3:
        return 0.0
    second = np.diff(finite, n=2)
    return float(np.sqrt(np.mean(second**2)))


def _magnitude_component(
    player_peak: float,
    reference_peak: float,
    *,
    peak_label: str,
) -> tuple[float, Direction, dict]:
    error = player_peak - reference_peak
    tolerance = max(
        _MAGNITUDE_MIN_TOLERANCE_DEG,
        _MAGNITUDE_RELATIVE_TOLERANCE * abs(reference_peak),
    )
    score = _gaussian_score(error, tolerance)
    deadzone = _DIRECTION_DEADZONE_FRACTION * tolerance
    if abs(error) <= deadzone:
        direction: Direction = "acceptable"
    elif error < 0:
        direction = "too_low"
    else:
        direction = "too_high"
    return score, direction, {
        f"player_peak_{peak_label}_deg": player_peak,
        f"reference_peak_{peak_label}_deg": reference_peak,
        f"peak_{peak_label}_error_deg": round(error, 4),
        "magnitude_tolerance_deg": round(tolerance, 4),
        "magnitude_score": score,
    }


def _timing_component(player_tau: float, reference_tau: float) -> tuple[float, dict]:
    error = player_tau - reference_tau
    score = _gaussian_score(error, _TIMING_TOLERANCE)
    return score, {
        "player_peak_timing_normalized": round(player_tau, 4),
        "reference_peak_timing_normalized": round(reference_tau, 4),
        "peak_timing_error_normalized": round(error, 4),
        "timing_tolerance_normalized": _TIMING_TOLERANCE,
        "timing_score": score,
    }


def _smoothness_component(
    player_window: np.ndarray, reference_window: np.ndarray
) -> tuple[float, dict]:
    player_jerk = _jerk_rms(player_window)
    reference_jerk = _jerk_rms(reference_window)
    ratio = player_jerk / max(reference_jerk, _EPSILON)
    if ratio <= 1.0:
        score = 100.0
    else:
        score = _gaussian_score(ratio - 1.0, _SMOOTHNESS_RATIO_TOLERANCE)
    return score, {
        "player_jerk_rms": round(player_jerk, 6),
        "reference_jerk_rms": round(reference_jerk, 6),
        "smoothness_metric": round(ratio, 6),
        "smoothness_score": score,
    }


def _consistency_component(player_trial_peaks: Sequence[float] | None) -> tuple[float, dict]:
    if player_trial_peaks is None or len(player_trial_peaks) < 2:
        return NEUTRAL_CONSISTENCY_SCORE, {
            "consistency_score": NEUTRAL_CONSISTENCY_SCORE,
            "consistency_mode": "single_serve_neutral",
            "trial_peak_count": 0 if player_trial_peaks is None else len(player_trial_peaks),
        }
    peaks = np.asarray(player_trial_peaks, dtype=float)
    peaks = peaks[np.isfinite(peaks)]
    if peaks.size < 2:
        return NEUTRAL_CONSISTENCY_SCORE, {
            "consistency_score": NEUTRAL_CONSISTENCY_SCORE,
            "consistency_mode": "single_serve_neutral",
            "trial_peak_count": int(peaks.size),
        }
    mean = float(np.mean(peaks))
    std = float(np.std(peaks, ddof=1)) if peaks.size > 1 else 0.0
    cv = std / max(abs(mean), _EPSILON)
    score = _gaussian_score(cv, _CONSISTENCY_CV_TOLERANCE)
    return score, {
        "consistency_score": score,
        "consistency_mode": "multi_serve",
        "trial_peak_count": int(peaks.size),
        "trial_peak_mean_deg": round(mean, 4),
        "trial_peak_std_deg": round(std, 4),
        "trial_peak_cv": round(cv, 6),
    }


def _confidence_from_quality(
    player_window: np.ndarray,
    *,
    player_joint_confidence: np.ndarray | None,
    phase_slice: slice | None,
    phase: str,
) -> tuple[float, dict]:
    finite_frac = float(np.mean(np.isfinite(player_window)))
    conf = finite_frac

    joint_mean = None
    if player_joint_confidence is not None:
        jc = np.asarray(player_joint_confidence, dtype=float)
        if phase_slice is not None:
            jc = jc[phase_slice]
        finite_jc = jc[np.isfinite(jc)]
        if finite_jc.size:
            joint_mean = float(np.mean(finite_jc))
            conf *= joint_mean

    n = int(player_window.size)
    if n < 8:
        length_factor = 0.5
    elif n < 15:
        length_factor = 0.75
    else:
        length_factor = 1.0
    conf *= length_factor
    conf = float(_clamp(conf, 0.0, 1.0))

    phase_key = phase.lower().replace(" ", "_")
    return conf, {
        "confidence": conf,
        "finite_sample_fraction": round(finite_frac, 4),
        "mean_joint_confidence": None if joint_mean is None else round(joint_mean, 4),
        f"{phase_key}_frame_count": n,
        f"{phase_key}_length_factor": length_factor,
        "confidence_notes": (
            "Future plugs: MotionBERT frame confidence, segmentation "
            f"marker_validity_mask, incomplete {phase}-phase flags."
        ),
    }


def score_angle_feature(
    *,
    feature_name: str,
    phase: str,
    player_series: np.ndarray,
    reference_series: np.ndarray,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: Sequence[float] | None = None,
    player_joint_confidence: np.ndarray | None = None,
    fps: float | None = None,
    unit: str = "deg",
    peak_mode: PeakMode = "max",
    peak_label: str = "angle",
) -> FeatureScoreResult:
    """Score an angle-based feature from player/reference trajectories.

    Parameters
    ----------
    peak_label:
        Used only in measurement key names, e.g. ``\"flexion\"`` yields
        ``player_peak_flexion_deg``. Does not affect scoring math.
    """
    player = np.asarray(player_series, dtype=float)
    reference = np.asarray(reference_series, dtype=float)
    player_window, player_slice = resolve_phase_window(player, player_phase_slice)
    reference_window, _ = resolve_phase_window(
        reference,
        reference_phase_slice if reference_phase_slice is not None else player_phase_slice,
    )

    player_peak, player_tau, player_local_idx = peak_and_timing(
        player_window, peak_mode=peak_mode
    )
    reference_peak, reference_tau, reference_local_idx = peak_and_timing(
        reference_window, peak_mode=peak_mode
    )

    magnitude_score, direction, magnitude_meta = _magnitude_component(
        player_peak, reference_peak, peak_label=peak_label
    )
    timing_score, timing_meta = _timing_component(player_tau, reference_tau)
    smoothness_score, smoothness_meta = _smoothness_component(
        player_window, reference_window
    )
    consistency_score, consistency_meta = _consistency_component(player_trial_peaks)
    confidence, confidence_meta = _confidence_from_quality(
        player_window,
        player_joint_confidence=player_joint_confidence,
        phase_slice=player_slice,
        phase=phase,
    )

    builder = (
        FeatureScoreBuilder(
            feature_name,
            phase,
            unit=unit,
            confidence=confidence,
        )
        .set_direction(direction)
        .set_values(player_value=player_peak, reference_value=reference_peak)
        .add_component(
            "magnitude",
            magnitude_score,
            DEFAULT_COMPONENT_WEIGHTS["magnitude"],
        )
        .add_component(
            "timing",
            timing_score,
            DEFAULT_COMPONENT_WEIGHTS["timing"],
        )
        .add_component(
            "smoothness",
            smoothness_score,
            DEFAULT_COMPONENT_WEIGHTS["smoothness"],
        )
        .add_component(
            "consistency",
            consistency_score,
            DEFAULT_COMPONENT_WEIGHTS["consistency"],
        )
        .add_measurement("scoring_mode", "biomechanical")
        .add_measurement("player_peak_local_index", player_local_idx)
        .add_measurement("reference_peak_local_index", reference_local_idx)
        .add_measurement("fps", fps)
    )
    for meta in (
        magnitude_meta,
        timing_meta,
        smoothness_meta,
        consistency_meta,
        confidence_meta,
    ):
        for key, value in meta.items():
            builder.add_measurement(key, value)

    return builder.build()


def score_angle_feature_from_scalars(
    *,
    feature_name: str,
    phase: str,
    player_value: float,
    reference_value: float,
    unit: str = "deg",
    peak_label: str = "angle",
) -> FeatureScoreResult:
    """Scalar-only fallback when trajectories are unavailable."""
    magnitude_score, direction, magnitude_meta = _magnitude_component(
        float(player_value), float(reference_value), peak_label=peak_label
    )
    builder = (
        FeatureScoreBuilder(
            feature_name,
            phase,
            unit=unit,
            confidence=SCALAR_FALLBACK_CONFIDENCE,
        )
        .set_direction(direction)
        .set_values(player_value=float(player_value), reference_value=float(reference_value))
        .add_component(
            "magnitude",
            magnitude_score,
            DEFAULT_COMPONENT_WEIGHTS["magnitude"],
        )
        .add_component(
            "timing",
            NEUTRAL_COMPONENT_SCORE,
            DEFAULT_COMPONENT_WEIGHTS["timing"],
        )
        .add_component(
            "smoothness",
            NEUTRAL_COMPONENT_SCORE,
            DEFAULT_COMPONENT_WEIGHTS["smoothness"],
        )
        .add_component(
            "consistency",
            NEUTRAL_CONSISTENCY_SCORE,
            DEFAULT_COMPONENT_WEIGHTS["consistency"],
        )
        .add_measurement("scoring_mode", "scalar_fallback")
        .add_measurement("timing_score", NEUTRAL_COMPONENT_SCORE)
        .add_measurement("smoothness_score", NEUTRAL_COMPONENT_SCORE)
        .add_measurement("consistency_score", NEUTRAL_CONSISTENCY_SCORE)
        .add_measurement("consistency_mode", "single_serve_neutral")
        .add_measurement("confidence", SCALAR_FALLBACK_CONFIDENCE)
        .add_measurement(
            "confidence_notes",
            "Scalar fallback — pass player_series/reference_series for full "
            "timing, smoothness, and data-quality confidence.",
        )
    )
    for key, value in magnitude_meta.items():
        builder.add_measurement(key, value)
    return builder.build()
