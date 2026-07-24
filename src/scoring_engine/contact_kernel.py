"""Reusable Contact Kernel for event-locked Contact-phase scoring.

Mathematics
-----------
Contact features are evaluated at a single biomechanical event (CONTACT),
not over a full phase peak search. Timing is therefore omitted — the event
index *is* the temporal definition.

**Magnitude** (weight 0.70)::

    tolerance = max(min_tolerance, 0.15 · |R|)
    magnitude_score = 100 · exp(−½ · ((P − R) / tolerance)²)

Direction from signed ``P − R`` with a 5%·tolerance dead-zone.

**Smoothness** (optional, weight 0.15 when a local window is provided)::

    Same jerk-RMS ratio rule as the Angle/Velocity kernels on a short
    ±N-frame neighborhood around contact. Omitted when no local series
    is supplied (weights renormalize).

**Consistency** (weight 0.30, or 0.15 with smoothness)::

    Multi-serve CV of trial event values; else neutral 75.

**Confidence**
    Finite event value × optional joint confidence at contact × optional
    local-window coverage. Future: contact-detector quality, ball/racket
    tracking, MotionBERT frame confidence.

**Scalar fallback**
    Magnitude from scalars only; consistency = 75; confidence = 0.55.

Assumptions
-----------
- ``contact_frame`` indexes the same trajectory arrays passed by the caller.
- Hand position proxies racket/ball contact (no ball marker yet).
- Normalized skeleton: default vertical axis is Y; forward/lateral are
  caller-defined offsets in the body frame.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.scoring_engine.result import Direction, FeatureScoreResult
from src.scoring_engine.score_builder import FeatureScoreBuilder

NEUTRAL_CONSISTENCY_SCORE = 75.0
SCALAR_FALLBACK_CONFIDENCE = 0.55

_MAGNITUDE_RELATIVE_TOLERANCE = 0.15
_SMOOTHNESS_RATIO_TOLERANCE = 0.5
_CONSISTENCY_CV_TOLERANCE = 0.08
_DIRECTION_DEADZONE_FRACTION = 0.05
_EPSILON = 1e-6
_DEFAULT_MIN_TOLERANCE = 40.0

# No timing — contact defines the event.
_WEIGHT_MAGNITUDE = 0.70
_WEIGHT_CONSISTENCY = 0.30
_WEIGHT_SMOOTHNESS = 0.15  # used only when local series provided; others renormalize


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _gaussian_score(error: float, tolerance: float) -> float:
    tol = max(float(tolerance), _EPSILON)
    return float(_clamp(100.0 * np.exp(-0.5 * (float(error) / tol) ** 2)))


def _magnitude_component(
    player_value: float,
    reference_value: float,
    *,
    value_label: str,
    min_tolerance: float,
) -> tuple[float, Direction, dict[str, Any]]:
    error = player_value - reference_value
    tolerance = max(float(min_tolerance), _MAGNITUDE_RELATIVE_TOLERANCE * abs(reference_value))
    score = _gaussian_score(error, tolerance)
    deadzone = _DIRECTION_DEADZONE_FRACTION * tolerance
    if abs(error) <= deadzone:
        direction: Direction = "acceptable"
    elif error < 0:
        direction = "too_low"
    else:
        direction = "too_high"
    return score, direction, {
        f"player_{value_label}": player_value,
        f"reference_{value_label}": reference_value,
        f"{value_label}_difference": round(error, 4),
        "magnitude_tolerance": round(tolerance, 4),
        "magnitude_score": score,
    }


def _jerk_rms(window: np.ndarray) -> float:
    finite = window[np.isfinite(window)]
    if finite.size < 3:
        return 0.0
    second = np.diff(finite, n=2)
    return float(np.sqrt(np.mean(second**2)))


def _smoothness_component(
    player_window: np.ndarray, reference_window: np.ndarray
) -> tuple[float, dict[str, Any]]:
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


def _consistency_component(player_trial_values: Sequence[float] | None) -> tuple[float, dict[str, Any]]:
    if player_trial_values is None or len(player_trial_values) < 2:
        return NEUTRAL_CONSISTENCY_SCORE, {
            "consistency_score": NEUTRAL_CONSISTENCY_SCORE,
            "consistency_mode": "single_serve_neutral",
            "trial_value_count": 0 if player_trial_values is None else len(player_trial_values),
        }
    values = np.asarray(player_trial_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return NEUTRAL_CONSISTENCY_SCORE, {
            "consistency_score": NEUTRAL_CONSISTENCY_SCORE,
            "consistency_mode": "single_serve_neutral",
            "trial_value_count": int(values.size),
        }
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    cv = std / max(abs(mean), _EPSILON)
    score = _gaussian_score(cv, _CONSISTENCY_CV_TOLERANCE)
    return score, {
        "consistency_score": score,
        "consistency_mode": "multi_serve",
        "trial_value_count": int(values.size),
        "trial_value_mean": round(mean, 4),
        "trial_value_std": round(std, 4),
        "trial_value_cv": round(cv, 6),
    }


def _confidence_at_contact(
    *,
    player_value: float,
    joint_confidence_at_contact: float | None,
    local_finite_fraction: float | None,
) -> tuple[float, dict[str, Any]]:
    conf = 1.0 if np.isfinite(player_value) else 0.0
    if joint_confidence_at_contact is not None and np.isfinite(joint_confidence_at_contact):
        conf *= float(joint_confidence_at_contact)
    if local_finite_fraction is not None:
        conf *= float(local_finite_fraction)
    conf = float(_clamp(conf, 0.0, 1.0))
    return conf, {
        "confidence": conf,
        "joint_confidence_at_contact": (
            None
            if joint_confidence_at_contact is None
            else round(float(joint_confidence_at_contact), 4)
        ),
        "local_finite_fraction": (
            None if local_finite_fraction is None else round(float(local_finite_fraction), 4)
        ),
        "confidence_notes": (
            "Future plugs: contact-detector quality, ball/racket tracking, "
            "MotionBERT frame confidence, marker_validity_mask."
        ),
    }


def score_contact_event(
    *,
    feature_name: str,
    player_value: float,
    reference_value: float,
    phase: str = "Contact",
    unit: str = "mm",
    value_label: str = "value",
    min_tolerance: float = _DEFAULT_MIN_TOLERANCE,
    player_contact_frame: int | None = None,
    reference_contact_frame: int | None = None,
    player_trial_values: Sequence[float] | None = None,
    joint_confidence_at_contact: float | None = None,
    player_local_series: np.ndarray | None = None,
    reference_local_series: np.ndarray | None = None,
    extra_measurements: Mapping[str, Any] | None = None,
) -> FeatureScoreResult:
    """Score a Contact-phase feature from event values at the contact frame."""
    magnitude_score, direction, magnitude_meta = _magnitude_component(
        float(player_value),
        float(reference_value),
        value_label=value_label,
        min_tolerance=min_tolerance,
    )
    consistency_score, consistency_meta = _consistency_component(player_trial_values)

    local_frac = None
    smoothness_score = None
    smoothness_meta: dict[str, Any] = {}
    if player_local_series is not None and reference_local_series is not None:
        player_local = np.asarray(player_local_series, dtype=float)
        reference_local = np.asarray(reference_local_series, dtype=float)
        local_frac = float(np.mean(np.isfinite(player_local))) if player_local.size else 0.0
        smoothness_score, smoothness_meta = _smoothness_component(
            player_local, reference_local
        )

    confidence, confidence_meta = _confidence_at_contact(
        player_value=float(player_value),
        joint_confidence_at_contact=joint_confidence_at_contact,
        local_finite_fraction=local_frac,
    )

    builder = (
        FeatureScoreBuilder(
            feature_name,
            phase,
            unit=unit,
            confidence=confidence,
        )
        .set_direction(direction)
        .set_values(player_value=float(player_value), reference_value=float(reference_value))
        .add_component("magnitude", magnitude_score, _WEIGHT_MAGNITUDE)
        .add_component("consistency", consistency_score, _WEIGHT_CONSISTENCY)
        .add_measurement("scoring_mode", "biomechanical")
        .add_measurement("player_contact_frame", player_contact_frame)
        .add_measurement("reference_contact_frame", reference_contact_frame)
    )
    if smoothness_score is not None:
        builder.add_component("smoothness", smoothness_score, _WEIGHT_SMOOTHNESS)

    for meta in (magnitude_meta, consistency_meta, smoothness_meta, confidence_meta):
        for key, value in meta.items():
            builder.add_measurement(key, value)
    if extra_measurements:
        for key, value in extra_measurements.items():
            builder.add_measurement(str(key), value)
    return builder.build()


def score_contact_event_from_scalars(
    *,
    feature_name: str,
    player_value: float,
    reference_value: float,
    phase: str = "Contact",
    unit: str = "mm",
    value_label: str = "value",
    min_tolerance: float = _DEFAULT_MIN_TOLERANCE,
    extra_measurements: Mapping[str, Any] | None = None,
) -> FeatureScoreResult:
    """Scalar-only fallback when contact trajectories / frames are unavailable."""
    magnitude_score, direction, magnitude_meta = _magnitude_component(
        float(player_value),
        float(reference_value),
        value_label=value_label,
        min_tolerance=min_tolerance,
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
        .add_component("magnitude", magnitude_score, _WEIGHT_MAGNITUDE)
        .add_component("consistency", NEUTRAL_CONSISTENCY_SCORE, _WEIGHT_CONSISTENCY)
        .add_measurement("scoring_mode", "scalar_fallback")
        .add_measurement("consistency_score", NEUTRAL_CONSISTENCY_SCORE)
        .add_measurement("consistency_mode", "single_serve_neutral")
        .add_measurement("confidence", SCALAR_FALLBACK_CONFIDENCE)
        .add_measurement(
            "confidence_notes",
            "Scalar fallback — pass contact-frame event values for full "
            "biomechanical contact scoring.",
        )
    )
    for key, value in magnitude_meta.items():
        builder.add_measurement(key, value)
    if extra_measurements:
        for key, value in extra_measurements.items():
            builder.add_measurement(str(key), value)
    return builder.build()
