"""Velocity-kernel scorers for Trunk/Hip rotation velocity and Shoulder Deceleration."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from src.scoring_engine.result import FeatureScoreResult
from src.scoring_engine.velocity_kernel import (
    PeakMode,
    score_velocity_feature,
    score_velocity_feature_from_scalars,
)

BiomechanicsFn = Callable[..., FeatureScoreResult]
ScalarsFn = Callable[[float, float], FeatureScoreResult]

_VELOCITY_MIN_TOLERANCE = 30.0  # deg/s
_DECEL_MIN_TOLERANCE = 50.0  # deg/s²


def _make_velocity_scorer(
    *,
    feature_name: str,
    phase: str,
    peak_label: str,
    unit: str,
    min_tolerance: float,
    peak_mode: PeakMode = "max",
) -> tuple[BiomechanicsFn, ScalarsFn]:
    def biomechanics(
        *,
        player_series: np.ndarray,
        reference_series: np.ndarray,
        player_phase_slice: slice | None = None,
        reference_phase_slice: slice | None = None,
        player_trial_peaks: Sequence[float] | None = None,
        player_joint_confidence: np.ndarray | None = None,
        fps: float | None = None,
    ) -> FeatureScoreResult:
        return score_velocity_feature(
            feature_name=feature_name,
            phase=phase,
            player_series=player_series,
            reference_series=reference_series,
            player_phase_slice=player_phase_slice,
            reference_phase_slice=reference_phase_slice,
            player_trial_peaks=player_trial_peaks,
            player_joint_confidence=player_joint_confidence,
            fps=fps,
            unit=unit,
            peak_mode=peak_mode,
            peak_label=peak_label,
            min_tolerance=min_tolerance,
        )

    def from_scalars(player_value: float, reference_value: float) -> FeatureScoreResult:
        return score_velocity_feature_from_scalars(
            feature_name=feature_name,
            phase=phase,
            player_value=player_value,
            reference_value=reference_value,
            unit=unit,
            peak_label=peak_label,
            min_tolerance=min_tolerance,
        )

    return biomechanics, from_scalars


score_trunk_rotation_velocity_biomechanics, score_trunk_rotation_velocity_from_scalars = (
    _make_velocity_scorer(
        feature_name="Trunk Rotation Velocity",
        phase="Acceleration",
        peak_label="velocity",
        unit="deg/s",
        min_tolerance=_VELOCITY_MIN_TOLERANCE,
    )
)

score_hip_rotation_velocity_biomechanics, score_hip_rotation_velocity_from_scalars = (
    _make_velocity_scorer(
        feature_name="Hip Rotation Velocity",
        phase="Acceleration",
        peak_label="velocity",
        unit="deg/s",
        min_tolerance=_VELOCITY_MIN_TOLERANCE,
    )
)

score_shoulder_deceleration_biomechanics, score_shoulder_deceleration_from_scalars = (
    _make_velocity_scorer(
        feature_name="Shoulder Deceleration",
        phase="Deceleration",
        peak_label="deceleration",
        unit="deg/s^2",
        min_tolerance=_DECEL_MIN_TOLERANCE,
    )
)
