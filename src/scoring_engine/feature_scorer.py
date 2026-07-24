"""Per-feature scoring against a reference serve.

Each Knowledge Library feature builds a score from weighted biomechanical
components via :class:`FeatureScoreBuilder`. Placeholder components currently
mirror the magnitude score so existing pipeline behavior is preserved while
real timing / smoothness / consistency signals are wired in later.
"""

from __future__ import annotations

from typing import Callable

from src.scoring_engine.result import Direction, FeatureScoreResult
from src.scoring_engine.score_builder import DEFAULT_COMPONENT_WEIGHTS, FeatureScoreBuilder

# Relative error below this fraction of |reference| (or absolute floor) counts
# as acceptable / near-match.
_ACCEPTABLE_RELATIVE = 0.05
_ABSOLUTE_FLOOR = 1.0
# Relative error at which magnitude score reaches 0.
_ZERO_SCORE_RELATIVE = 0.50

# Default confidence until MotionBERT / joint-quality signals are available.
_DEFAULT_CONFIDENCE = 1.0


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _magnitude_from_difference(
    player_value: float, reference_value: float
) -> tuple[float, Direction]:
    """Placeholder magnitude score from relative deviation vs reference."""
    difference = player_value - reference_value
    scale = max(abs(reference_value), _ABSOLUTE_FLOOR)
    relative = abs(difference) / scale

    if relative <= _ACCEPTABLE_RELATIVE:
        score = _clamp(100.0 - (relative / _ACCEPTABLE_RELATIVE) * 5.0)
        return score, "acceptable"

    span = _ZERO_SCORE_RELATIVE - _ACCEPTABLE_RELATIVE
    score = _clamp(95.0 * (1.0 - (relative - _ACCEPTABLE_RELATIVE) / span))
    direction: Direction = "too_low" if difference < 0 else "too_high"
    return score, direction


def _score_named(
    name: str,
    phase: str,
    player_value: float,
    reference_value: float,
    *,
    unit: str = "deg",
    confidence: float = _DEFAULT_CONFIDENCE,
) -> FeatureScoreResult:
    """Build a feature score from magnitude + placeholder companion components.

    Timing, smoothness, and consistency currently copy the magnitude score so
    the weighted final score equals magnitude until real signals exist.
    """
    magnitude_score, direction = _magnitude_from_difference(player_value, reference_value)

    builder = (
        FeatureScoreBuilder(name, phase, unit=unit, confidence=confidence)
        .set_direction(direction)
        .set_values(player_value=player_value, reference_value=reference_value)
    )
    for component_name, weight in DEFAULT_COMPONENT_WEIGHTS.items():
        # Placeholders: non-magnitude components mirror magnitude for now.
        component_score = magnitude_score
        builder.add_component(component_name, score=component_score, weight=weight)

    return builder.build()


# --- Loading ---


def _score_angle_entry(
    player_value: float,
    reference_value: float,
    *,
    biomechanics,
    from_scalars,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    """Shared entry: trajectories → angle-kernel biomechanics; else scalar fallback."""
    if player_series is not None and reference_series is not None:
        return biomechanics(
            player_series=player_series,
            reference_series=reference_series,
            player_phase_slice=player_phase_slice,
            reference_phase_slice=reference_phase_slice,
            player_trial_peaks=player_trial_peaks,
            player_joint_confidence=player_joint_confidence,
            fps=fps,
        )
    return from_scalars(player_value, reference_value)


def score_knee_flexion(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_loading_slice: slice | None = None,
    reference_loading_slice: slice | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    """Score Knee Flexion via the Angle Kernel (see knee_flexion module)."""
    from src.scoring_engine.knee_flexion import (
        score_knee_flexion_biomechanics,
        score_knee_flexion_from_scalars,
    )

    # Preserve loading_slice aliases used by the Knee Flexion API.
    player_slice = player_phase_slice if player_phase_slice is not None else player_loading_slice
    reference_slice = (
        reference_phase_slice
        if reference_phase_slice is not None
        else reference_loading_slice
    )
    if player_series is not None and reference_series is not None:
        return score_knee_flexion_biomechanics(
            player_series=player_series,  # type: ignore[arg-type]
            reference_series=reference_series,  # type: ignore[arg-type]
            loading_slice=player_slice,
            reference_loading_slice=reference_slice,
            player_trial_peaks=player_trial_peaks,  # type: ignore[arg-type]
            player_joint_confidence=player_joint_confidence,  # type: ignore[arg-type]
            fps=fps,
        )
    return score_knee_flexion_from_scalars(player_value, reference_value)


def score_hip_flexion(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import angle_scorers as _a

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_a.score_hip_flexion_biomechanics,
        from_scalars=_a.score_hip_flexion_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


def score_shoulder_tilt(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import angle_scorers as _a

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_a.score_shoulder_tilt_biomechanics,
        from_scalars=_a.score_shoulder_tilt_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


def score_toss_arm_extension(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import angle_scorers as _a

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_a.score_toss_arm_extension_biomechanics,
        from_scalars=_a.score_toss_arm_extension_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


def score_center_of_mass(
    player_value: float,
    reference_value: float,
    *,
    player_com: object | None = None,
    reference_com: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_drops: object | None = None,
    joint_confidence: object | None = None,
    vertical_axis: int = 1,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import path_posture_scorers as _p

    if player_com is not None and reference_com is not None:
        return _p.score_center_of_mass_biomechanics(
            player_com=player_com,  # type: ignore[arg-type]
            reference_com=reference_com,  # type: ignore[arg-type]
            player_phase_slice=player_phase_slice,
            reference_phase_slice=reference_phase_slice,
            player_trial_drops=player_trial_drops,  # type: ignore[arg-type]
            joint_confidence=joint_confidence,  # type: ignore[arg-type]
            vertical_axis=vertical_axis,
            fps=fps,
        )
    return _p.score_center_of_mass_from_scalars(player_value, reference_value)


def score_trunk_rotation(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import angle_scorers as _a

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_a.score_trunk_rotation_biomechanics,
        from_scalars=_a.score_trunk_rotation_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


def score_pelvis_rotation(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import angle_scorers as _a

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_a.score_pelvis_rotation_biomechanics,
        from_scalars=_a.score_pelvis_rotation_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


# --- Cocking ---


def score_right_elbow_flexion(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import angle_scorers as _a

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_a.score_right_elbow_flexion_biomechanics,
        from_scalars=_a.score_right_elbow_flexion_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


def score_left_elbow_flexion(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import angle_scorers as _a

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_a.score_left_elbow_flexion_biomechanics,
        from_scalars=_a.score_left_elbow_flexion_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


def score_shoulder_external_rotation(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import angle_scorers as _a

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_a.score_shoulder_external_rotation_biomechanics,
        from_scalars=_a.score_shoulder_external_rotation_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


def score_forearm_angle(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import angle_scorers as _a

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_a.score_forearm_angle_biomechanics,
        from_scalars=_a.score_forearm_angle_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


# --- Acceleration ---


def score_shoulder_internal_rotation(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import angle_scorers as _a

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_a.score_shoulder_internal_rotation_biomechanics,
        from_scalars=_a.score_shoulder_internal_rotation_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


def score_right_elbow_extension(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import angle_scorers as _a

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_a.score_right_elbow_extension_biomechanics,
        from_scalars=_a.score_right_elbow_extension_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


def score_left_elbow_extension(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import angle_scorers as _a

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_a.score_left_elbow_extension_biomechanics,
        from_scalars=_a.score_left_elbow_extension_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


def score_trunk_rotation_velocity(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import velocity_scorers as _v

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_v.score_trunk_rotation_velocity_biomechanics,
        from_scalars=_v.score_trunk_rotation_velocity_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


def score_hip_rotation_velocity(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import velocity_scorers as _v

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_v.score_hip_rotation_velocity_biomechanics,
        from_scalars=_v.score_hip_rotation_velocity_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


# --- Contact ---


def score_contact_height(
    player_value: float,
    reference_value: float,
    *,
    player_hand_positions: object | None = None,
    reference_hand_positions: object | None = None,
    player_contact_frame: int | None = None,
    reference_contact_frame: int | None = None,
    player_trial_heights: object | None = None,
    joint_confidence_at_contact: float | None = None,
    vertical_axis: int = 1,
    unit: str = "mm",
) -> FeatureScoreResult:
    from src.scoring_engine import contact_scorers as _c

    if (
        player_hand_positions is not None
        and reference_hand_positions is not None
        and player_contact_frame is not None
        and reference_contact_frame is not None
    ):
        return _c.score_contact_height_biomechanics(
            player_hand_positions=player_hand_positions,  # type: ignore[arg-type]
            reference_hand_positions=reference_hand_positions,  # type: ignore[arg-type]
            player_contact_frame=player_contact_frame,
            reference_contact_frame=reference_contact_frame,
            vertical_axis=vertical_axis,
            player_trial_heights=player_trial_heights,  # type: ignore[arg-type]
            joint_confidence_at_contact=joint_confidence_at_contact,
            unit=unit,
        )
    return _c.score_contact_height_from_scalars(player_value, reference_value)


def score_contact_position(
    player_value: float,
    reference_value: float,
    *,
    player_hand_positions: object | None = None,
    player_pelvis_positions: object | None = None,
    reference_hand_positions: object | None = None,
    reference_pelvis_positions: object | None = None,
    player_contact_frame: int | None = None,
    reference_contact_frame: int | None = None,
    player_trial_forward: object | None = None,
    joint_confidence_at_contact: float | None = None,
    unit: str = "mm",
) -> FeatureScoreResult:
    from src.scoring_engine import contact_scorers as _c

    if (
        player_hand_positions is not None
        and player_pelvis_positions is not None
        and reference_hand_positions is not None
        and reference_pelvis_positions is not None
        and player_contact_frame is not None
        and reference_contact_frame is not None
    ):
        return _c.score_contact_position_biomechanics(
            player_hand_positions=player_hand_positions,  # type: ignore[arg-type]
            player_pelvis_positions=player_pelvis_positions,  # type: ignore[arg-type]
            reference_hand_positions=reference_hand_positions,  # type: ignore[arg-type]
            reference_pelvis_positions=reference_pelvis_positions,  # type: ignore[arg-type]
            player_contact_frame=player_contact_frame,
            reference_contact_frame=reference_contact_frame,
            player_trial_forward=player_trial_forward,  # type: ignore[arg-type]
            joint_confidence_at_contact=joint_confidence_at_contact,
            unit=unit,
        )
    return _c.score_contact_position_from_scalars(player_value, reference_value)


def score_arm_extension(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import angle_scorers as _a

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_a.score_arm_extension_biomechanics,
        from_scalars=_a.score_arm_extension_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


def score_body_alignment(
    player_value: float,
    reference_value: float,
    *,
    player_shoulder_line_deg: object | None = None,
    player_hip_line_deg: object | None = None,
    player_trunk_rotation_deg: object | None = None,
    reference_shoulder_line_deg: object | None = None,
    reference_hip_line_deg: object | None = None,
    reference_trunk_rotation_deg: object | None = None,
    player_contact_frame: int | None = None,
    reference_contact_frame: int | None = None,
    player_trial_facing: object | None = None,
    joint_confidence_at_contact: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import contact_scorers as _c

    if (
        player_shoulder_line_deg is not None
        and player_hip_line_deg is not None
        and player_trunk_rotation_deg is not None
        and reference_shoulder_line_deg is not None
        and reference_hip_line_deg is not None
        and reference_trunk_rotation_deg is not None
        and player_contact_frame is not None
        and reference_contact_frame is not None
    ):
        return _c.score_body_alignment_biomechanics(
            player_shoulder_line_deg=player_shoulder_line_deg,  # type: ignore[arg-type]
            player_hip_line_deg=player_hip_line_deg,  # type: ignore[arg-type]
            player_trunk_rotation_deg=player_trunk_rotation_deg,  # type: ignore[arg-type]
            reference_shoulder_line_deg=reference_shoulder_line_deg,  # type: ignore[arg-type]
            reference_hip_line_deg=reference_hip_line_deg,  # type: ignore[arg-type]
            reference_trunk_rotation_deg=reference_trunk_rotation_deg,  # type: ignore[arg-type]
            player_contact_frame=player_contact_frame,
            reference_contact_frame=reference_contact_frame,
            player_trial_facing=player_trial_facing,  # type: ignore[arg-type]
            joint_confidence_at_contact=joint_confidence_at_contact,
        )
    return _c.score_body_alignment_from_scalars(player_value, reference_value)


# --- Deceleration ---


def score_follow_through(
    player_value: float,
    reference_value: float,
    *,
    player_hand: object | None = None,
    reference_hand: object | None = None,
    player_contact_frame: int | None = None,
    reference_contact_frame: int | None = None,
    player_phase_end: int | None = None,
    reference_phase_end: int | None = None,
    player_trial_path_lengths: object | None = None,
    joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import path_posture_scorers as _p

    if (
        player_hand is not None
        and reference_hand is not None
        and player_contact_frame is not None
        and reference_contact_frame is not None
    ):
        return _p.score_follow_through_biomechanics(
            player_hand=player_hand,  # type: ignore[arg-type]
            reference_hand=reference_hand,  # type: ignore[arg-type]
            player_contact_frame=player_contact_frame,
            reference_contact_frame=reference_contact_frame,
            player_phase_end=player_phase_end,
            reference_phase_end=reference_phase_end,
            player_trial_path_lengths=player_trial_path_lengths,  # type: ignore[arg-type]
            joint_confidence=joint_confidence,  # type: ignore[arg-type]
            fps=fps,
        )
    return _p.score_follow_through_from_scalars(player_value, reference_value)


def score_shoulder_deceleration(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import velocity_scorers as _v

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_v.score_shoulder_deceleration_biomechanics,
        from_scalars=_v.score_shoulder_deceleration_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


def score_trunk_flexion(
    player_value: float,
    reference_value: float,
    *,
    player_series: object | None = None,
    reference_series: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_peaks: object | None = None,
    player_joint_confidence: object | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    from src.scoring_engine import angle_scorers as _a

    return _score_angle_entry(
        player_value,
        reference_value,
        biomechanics=_a.score_trunk_flexion_biomechanics,
        from_scalars=_a.score_trunk_flexion_from_scalars,
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=player_phase_slice,
        reference_phase_slice=reference_phase_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
    )


# --- Finish ---


def score_balance(
    player_value: float,
    reference_value: float,
    *,
    player_com: object | None = None,
    reference_com: object | None = None,
    player_left_foot: object | None = None,
    player_right_foot: object | None = None,
    reference_left_foot: object | None = None,
    reference_right_foot: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_sway: object | None = None,
    joint_confidence: object | None = None,
    fps: float | None = None,
    vertical_axis: int = 1,
) -> FeatureScoreResult:
    from src.scoring_engine import path_posture_scorers as _p

    if (
        player_com is not None
        and reference_com is not None
        and player_left_foot is not None
        and player_right_foot is not None
        and reference_left_foot is not None
        and reference_right_foot is not None
    ):
        return _p.score_balance_biomechanics(
            player_com=player_com,  # type: ignore[arg-type]
            reference_com=reference_com,  # type: ignore[arg-type]
            player_left_foot=player_left_foot,  # type: ignore[arg-type]
            player_right_foot=player_right_foot,  # type: ignore[arg-type]
            reference_left_foot=reference_left_foot,  # type: ignore[arg-type]
            reference_right_foot=reference_right_foot,  # type: ignore[arg-type]
            player_phase_slice=player_phase_slice,
            reference_phase_slice=reference_phase_slice,
            player_trial_sway=player_trial_sway,  # type: ignore[arg-type]
            joint_confidence=joint_confidence,  # type: ignore[arg-type]
            fps=fps,
            vertical_axis=vertical_axis,
        )
    return _p.score_balance_from_scalars(player_value, reference_value)


def score_weight_transfer(
    player_value: float,
    reference_value: float,
    *,
    player_com: object | None = None,
    reference_com: object | None = None,
    player_back_foot: object | None = None,
    player_front_foot: object | None = None,
    reference_back_foot: object | None = None,
    reference_front_foot: object | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_transfer: object | None = None,
    joint_confidence: object | None = None,
    fps: float | None = None,
    forward_axis: int = 2,
) -> FeatureScoreResult:
    from src.scoring_engine import path_posture_scorers as _p

    if (
        player_com is not None
        and reference_com is not None
        and player_back_foot is not None
        and player_front_foot is not None
        and reference_back_foot is not None
        and reference_front_foot is not None
    ):
        return _p.score_weight_transfer_biomechanics(
            player_com=player_com,  # type: ignore[arg-type]
            reference_com=reference_com,  # type: ignore[arg-type]
            player_back_foot=player_back_foot,  # type: ignore[arg-type]
            player_front_foot=player_front_foot,  # type: ignore[arg-type]
            reference_back_foot=reference_back_foot,  # type: ignore[arg-type]
            reference_front_foot=reference_front_foot,  # type: ignore[arg-type]
            player_phase_slice=player_phase_slice,
            reference_phase_slice=reference_phase_slice,
            player_trial_transfer=player_trial_transfer,  # type: ignore[arg-type]
            joint_confidence=joint_confidence,  # type: ignore[arg-type]
            fps=fps,
            forward_axis=forward_axis,
        )
    return _p.score_weight_transfer_from_scalars(player_value, reference_value)


def score_recovery_position(
    player_value: float,
    reference_value: float,
    *,
    player_com: object | None = None,
    reference_com: object | None = None,
    player_left_foot: object | None = None,
    player_right_foot: object | None = None,
    reference_left_foot: object | None = None,
    reference_right_foot: object | None = None,
    player_contact_frame: int | None = None,
    reference_contact_frame: int | None = None,
    player_phase_slice: slice | None = None,
    reference_phase_slice: slice | None = None,
    player_trial_recovery_times: object | None = None,
    joint_confidence: object | None = None,
    fps: float | None = None,
    vertical_axis: int = 1,
    speed_threshold: float = 0.15,
) -> FeatureScoreResult:
    from src.scoring_engine import path_posture_scorers as _p

    if (
        player_com is not None
        and reference_com is not None
        and player_left_foot is not None
        and player_right_foot is not None
        and reference_left_foot is not None
        and reference_right_foot is not None
        and player_contact_frame is not None
        and reference_contact_frame is not None
    ):
        return _p.score_recovery_position_biomechanics(
            player_com=player_com,  # type: ignore[arg-type]
            reference_com=reference_com,  # type: ignore[arg-type]
            player_left_foot=player_left_foot,  # type: ignore[arg-type]
            player_right_foot=player_right_foot,  # type: ignore[arg-type]
            reference_left_foot=reference_left_foot,  # type: ignore[arg-type]
            reference_right_foot=reference_right_foot,  # type: ignore[arg-type]
            player_contact_frame=player_contact_frame,
            reference_contact_frame=reference_contact_frame,
            player_phase_slice=player_phase_slice,
            reference_phase_slice=reference_phase_slice,
            player_trial_recovery_times=player_trial_recovery_times,  # type: ignore[arg-type]
            joint_confidence=joint_confidence,  # type: ignore[arg-type]
            fps=fps,
            vertical_axis=vertical_axis,
            speed_threshold=speed_threshold,
        )
    return _p.score_recovery_position_from_scalars(player_value, reference_value)


FEATURE_SCORERS: dict[str, Callable[[float, float], FeatureScoreResult]] = {
    "Knee Flexion": score_knee_flexion,
    "Hip Flexion": score_hip_flexion,
    "Shoulder Tilt": score_shoulder_tilt,
    "Toss Arm Extension": score_toss_arm_extension,
    "Center of Mass": score_center_of_mass,
    "Trunk Rotation": score_trunk_rotation,
    "Pelvis Rotation": score_pelvis_rotation,
    "Right Elbow Flexion": score_right_elbow_flexion,
    "Left Elbow Flexion": score_left_elbow_flexion,
    "Shoulder External Rotation": score_shoulder_external_rotation,
    "Forearm Angle": score_forearm_angle,
    "Shoulder Internal Rotation": score_shoulder_internal_rotation,
    "Right Elbow Extension": score_right_elbow_extension,
    "Left Elbow Extension": score_left_elbow_extension,
    "Trunk Rotation Velocity": score_trunk_rotation_velocity,
    "Hip Rotation Velocity": score_hip_rotation_velocity,
    "Contact Height": score_contact_height,
    "Contact Position": score_contact_position,
    "Arm Extension": score_arm_extension,
    "Body Alignment": score_body_alignment,
    "Follow Through": score_follow_through,
    "Shoulder Deceleration": score_shoulder_deceleration,
    "Trunk Flexion": score_trunk_flexion,
    "Balance": score_balance,
    "Weight Transfer": score_weight_transfer,
    "Recovery Position": score_recovery_position,
}


def score_all_features(
    player_values: dict[str, float],
    reference_values: dict[str, float],
) -> tuple[FeatureScoreResult, ...]:
    """Score every registered feature; raise if any measurement is missing."""
    missing_player = [n for n in FEATURE_SCORERS if n not in player_values]
    missing_reference = [n for n in FEATURE_SCORERS if n not in reference_values]
    if missing_player:
        raise ValueError(
            "Missing player measurement(s): " + ", ".join(sorted(missing_player))
        )
    if missing_reference:
        raise ValueError(
            "Missing reference measurement(s): " + ", ".join(sorted(missing_reference))
        )

    return tuple(
        scorer(player_values[name], reference_values[name])
        for name, scorer in FEATURE_SCORERS.items()
    )
