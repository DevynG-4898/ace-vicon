"""Angle-kernel scorers for remaining Knowledge Library angle features.

Each scorer only prepares the feature-specific series and labels, then calls
:func:`score_angle_feature` / :func:`score_angle_feature_from_scalars`.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from src.scoring_engine.angle_kernel import (
    PeakMode,
    score_angle_feature,
    score_angle_feature_from_scalars,
)
from src.scoring_engine.result import FeatureScoreResult

BiomechanicsFn = Callable[..., FeatureScoreResult]
ScalarsFn = Callable[[float, float], FeatureScoreResult]


def _make_angle_scorer(
    *,
    feature_name: str,
    phase: str,
    peak_label: str,
    peak_mode: PeakMode = "max",
    unit: str = "deg",
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
        return score_angle_feature(
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
        )

    def from_scalars(player_value: float, reference_value: float) -> FeatureScoreResult:
        return score_angle_feature_from_scalars(
            feature_name=feature_name,
            phase=phase,
            player_value=player_value,
            reference_value=reference_value,
            unit=unit,
            peak_label=peak_label,
        )

    biomechanics.__name__ = f"score_{feature_name.lower().replace(' ', '_').replace('-', '_')}_biomechanics"
    from_scalars.__name__ = f"score_{feature_name.lower().replace(' ', '_').replace('-', '_')}_from_scalars"
    return biomechanics, from_scalars


# --- Loading ---
score_hip_flexion_biomechanics, score_hip_flexion_from_scalars = _make_angle_scorer(
    feature_name="Hip Flexion", phase="Loading", peak_label="flexion"
)
score_shoulder_tilt_biomechanics, score_shoulder_tilt_from_scalars = _make_angle_scorer(
    feature_name="Shoulder Tilt", phase="Loading", peak_label="tilt"
)
score_toss_arm_extension_biomechanics, score_toss_arm_extension_from_scalars = (
    _make_angle_scorer(
        feature_name="Toss Arm Extension", phase="Loading", peak_label="extension"
    )
)
score_trunk_rotation_biomechanics, score_trunk_rotation_from_scalars = _make_angle_scorer(
    feature_name="Trunk Rotation", phase="Loading", peak_label="rotation"
)
score_pelvis_rotation_biomechanics, score_pelvis_rotation_from_scalars = _make_angle_scorer(
    feature_name="Pelvis Rotation", phase="Loading", peak_label="rotation"
)

# --- Cocking ---
score_right_elbow_flexion_biomechanics, score_right_elbow_flexion_from_scalars = (
    _make_angle_scorer(
        feature_name="Right Elbow Flexion", phase="Cocking", peak_label="flexion"
    )
)
score_left_elbow_flexion_biomechanics, score_left_elbow_flexion_from_scalars = (
    _make_angle_scorer(
        feature_name="Left Elbow Flexion", phase="Cocking", peak_label="flexion"
    )
)
score_shoulder_external_rotation_biomechanics, score_shoulder_external_rotation_from_scalars = (
    _make_angle_scorer(
        feature_name="Shoulder External Rotation",
        phase="Cocking",
        peak_label="rotation",
    )
)
score_forearm_angle_biomechanics, score_forearm_angle_from_scalars = _make_angle_scorer(
    feature_name="Forearm Angle", phase="Cocking", peak_label="angle"
)

# --- Acceleration ---
score_shoulder_internal_rotation_biomechanics, score_shoulder_internal_rotation_from_scalars = (
    _make_angle_scorer(
        feature_name="Shoulder Internal Rotation",
        phase="Acceleration",
        peak_label="rotation",
    )
)
score_right_elbow_extension_biomechanics, score_right_elbow_extension_from_scalars = (
    _make_angle_scorer(
        feature_name="Right Elbow Extension",
        phase="Acceleration",
        peak_label="extension",
    )
)
score_left_elbow_extension_biomechanics, score_left_elbow_extension_from_scalars = (
    _make_angle_scorer(
        feature_name="Left Elbow Extension",
        phase="Acceleration",
        peak_label="extension",
    )
)

# --- Contact ---
score_arm_extension_biomechanics, score_arm_extension_from_scalars = _make_angle_scorer(
    feature_name="Arm Extension", phase="Contact", peak_label="extension"
)

# --- Deceleration ---
score_trunk_flexion_biomechanics, score_trunk_flexion_from_scalars = _make_angle_scorer(
    feature_name="Trunk Flexion", phase="Deceleration", peak_label="flexion"
)
