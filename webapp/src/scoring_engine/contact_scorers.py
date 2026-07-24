"""Contact-kernel scorers for Contact Height, Contact Position, Body Alignment."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.scoring_engine.contact_kernel import (
    score_contact_event,
    score_contact_event_from_scalars,
)
from src.scoring_engine.contact_series import (
    alignment_error_measurements,
    body_alignment_angles,
    contact_height,
    contact_position_offsets,
    local_window,
    normalized_height,
)
from src.scoring_engine.result import FeatureScoreResult
from src.scoring_engine.score_builder import FeatureScoreBuilder

_HEIGHT_MIN_TOLERANCE_MM = 40.0
_POSITION_MIN_TOLERANCE_MM = 40.0
_ALIGNMENT_MIN_TOLERANCE = 8.0  # degrees

# Contact Position: forward dominates the coaching cue ("hit in front").
_FORWARD_WEIGHT = 0.70
_LATERAL_WEIGHT = 0.30


def _spatial_min_tolerance(reference: float, *, unit: str, mm_floor: float) -> float:
    """Use mm floors for millimetre inputs; small relative floors for normalized coords."""
    if unit == "mm":
        return mm_floor
    return max(abs(float(reference)) * 0.05, 1e-4)


def score_contact_height_biomechanics(
    *,
    player_hand_positions: np.ndarray,
    reference_hand_positions: np.ndarray,
    player_contact_frame: int,
    reference_contact_frame: int,
    vertical_axis: int = 1,
    player_trial_heights: Sequence[float] | None = None,
    joint_confidence_at_contact: float | None = None,
    unit: str = "mm",
) -> FeatureScoreResult:
    """Score Contact Height from hand trajectories at contact frames."""
    player_h = contact_height(
        player_hand_positions, player_contact_frame, vertical_axis=vertical_axis
    )
    reference_h = contact_height(
        reference_hand_positions, reference_contact_frame, vertical_axis=vertical_axis
    )
    player_series = player_hand_positions[:, vertical_axis]
    reference_series = reference_hand_positions[:, vertical_axis]
    return score_contact_event(
        feature_name="Contact Height",
        player_value=player_h,
        reference_value=reference_h,
        unit=unit,
        value_label="height",
        min_tolerance=_spatial_min_tolerance(
            reference_h, unit=unit, mm_floor=_HEIGHT_MIN_TOLERANCE_MM
        ),
        player_contact_frame=player_contact_frame,
        reference_contact_frame=reference_contact_frame,
        player_trial_values=player_trial_heights,
        joint_confidence_at_contact=joint_confidence_at_contact,
        player_local_series=local_window(player_series, player_contact_frame),
        reference_local_series=local_window(reference_series, reference_contact_frame),
        extra_measurements={
            "normalized_height": round(
                normalized_height(player_h, reference_height=reference_h), 4
            ),
            "vertical_axis": vertical_axis,
        },
    )


def score_contact_height_from_scalars(
    player_value: float, reference_value: float
) -> FeatureScoreResult:
    return score_contact_event_from_scalars(
        feature_name="Contact Height",
        player_value=player_value,
        reference_value=reference_value,
        unit="mm",
        value_label="height",
        min_tolerance=_HEIGHT_MIN_TOLERANCE_MM,
        extra_measurements={
            "normalized_height": round(
                normalized_height(player_value, reference_height=reference_value), 4
            ),
        },
    )


def score_contact_position_biomechanics(
    *,
    player_hand_positions: np.ndarray,
    player_pelvis_positions: np.ndarray,
    reference_hand_positions: np.ndarray,
    reference_pelvis_positions: np.ndarray,
    player_contact_frame: int,
    reference_contact_frame: int,
    player_trial_forward: Sequence[float] | None = None,
    joint_confidence_at_contact: float | None = None,
    unit: str = "mm",
) -> FeatureScoreResult:
    """Score Contact Position from hand–pelvis offsets at contact.

    Magnitude blends forward (70%) and lateral (30%) Gaussian scores.
    Direction follows the forward axis (primary “in front” cue).
    """
    from src.scoring_engine.contact_kernel import (
        _WEIGHT_CONSISTENCY,
        _WEIGHT_MAGNITUDE,
        _consistency_component,
        _confidence_at_contact,
        _magnitude_component,
    )

    player = contact_position_offsets(
        player_hand_positions, player_pelvis_positions, player_contact_frame
    )
    reference = contact_position_offsets(
        reference_hand_positions, reference_pelvis_positions, reference_contact_frame
    )

    fwd_score, direction, fwd_meta = _magnitude_component(
        player["forward"],
        reference["forward"],
        value_label="forward",
        min_tolerance=_spatial_min_tolerance(
            reference["forward"], unit=unit, mm_floor=_POSITION_MIN_TOLERANCE_MM
        ),
    )
    lat_score, _, lat_meta = _magnitude_component(
        player["lateral"],
        reference["lateral"],
        value_label="lateral",
        min_tolerance=_spatial_min_tolerance(
            max(abs(reference["lateral"]), abs(reference["forward"])),
            unit=unit,
            mm_floor=_POSITION_MIN_TOLERANCE_MM,
        ),
    )
    magnitude_score = _FORWARD_WEIGHT * fwd_score + _LATERAL_WEIGHT * lat_score
    consistency_score, consistency_meta = _consistency_component(player_trial_forward)
    confidence, confidence_meta = _confidence_at_contact(
        player_value=player["forward"],
        joint_confidence_at_contact=joint_confidence_at_contact,
        local_finite_fraction=None,
    )

    # Representative scalar for FeatureScoreResult.player_value: forward offset.
    builder = (
        FeatureScoreBuilder(
            "Contact Position",
            "Contact",
            unit=unit,
            confidence=confidence,
        )
        .set_direction(direction)
        .set_values(player_value=player["forward"], reference_value=reference["forward"])
        .add_component("magnitude", magnitude_score, _WEIGHT_MAGNITUDE)
        .add_component("consistency", consistency_score, _WEIGHT_CONSISTENCY)
        .add_measurement("scoring_mode", "biomechanical")
        .add_measurement("player_contact_frame", player_contact_frame)
        .add_measurement("reference_contact_frame", reference_contact_frame)
        .add_measurement("player_forward", player["forward"])
        .add_measurement("reference_forward", reference["forward"])
        .add_measurement("player_lateral", player["lateral"])
        .add_measurement("reference_lateral", reference["lateral"])
        .add_measurement("player_vertical", player["vertical"])
        .add_measurement("reference_vertical", reference["vertical"])
        .add_measurement("forward_score", round(fwd_score, 2))
        .add_measurement("lateral_score", round(lat_score, 2))
        .add_measurement("forward_weight", _FORWARD_WEIGHT)
        .add_measurement("lateral_weight", _LATERAL_WEIGHT)
        .add_measurement("magnitude_score", round(magnitude_score, 2))
    )
    for meta in (fwd_meta, lat_meta, consistency_meta, confidence_meta):
        for key, value in meta.items():
            if key == "magnitude_score":
                continue
            builder.add_measurement(key, value)
    return builder.build()


def score_contact_position_from_scalars(
    player_value: float, reference_value: float
) -> FeatureScoreResult:
    """Scalar fallback treats the scalar as forward contact offset."""
    return score_contact_event_from_scalars(
        feature_name="Contact Position",
        player_value=player_value,
        reference_value=reference_value,
        unit="mm",
        value_label="forward",
        min_tolerance=_POSITION_MIN_TOLERANCE_MM,
    )


def score_body_alignment_biomechanics(
    *,
    player_shoulder_line_deg: np.ndarray,
    player_hip_line_deg: np.ndarray,
    player_trunk_rotation_deg: np.ndarray,
    reference_shoulder_line_deg: np.ndarray,
    reference_hip_line_deg: np.ndarray,
    reference_trunk_rotation_deg: np.ndarray,
    player_contact_frame: int,
    reference_contact_frame: int,
    player_trial_facing: Sequence[float] | None = None,
    joint_confidence_at_contact: float | None = None,
) -> FeatureScoreResult:
    """Score Body Alignment from shoulder/hip/trunk angles at contact.

    Primary magnitude uses trunk rotation (facing). Shoulder and hip line
    errors are stored and lightly blended into the magnitude score.
    """
    from src.scoring_engine.contact_kernel import (
        _WEIGHT_CONSISTENCY,
        _WEIGHT_MAGNITUDE,
        _consistency_component,
        _confidence_at_contact,
        _magnitude_component,
    )

    player = body_alignment_angles(
        shoulder_line_deg=player_shoulder_line_deg,
        hip_line_deg=player_hip_line_deg,
        trunk_rotation_deg=player_trunk_rotation_deg,
        contact_frame=player_contact_frame,
    )
    reference = body_alignment_angles(
        shoulder_line_deg=reference_shoulder_line_deg,
        hip_line_deg=reference_hip_line_deg,
        trunk_rotation_deg=reference_trunk_rotation_deg,
        contact_frame=reference_contact_frame,
    )

    trunk_score, direction, trunk_meta = _magnitude_component(
        player["trunk_rotation_deg"],
        reference["trunk_rotation_deg"],
        value_label="trunk_rotation",
        min_tolerance=_ALIGNMENT_MIN_TOLERANCE,
    )
    shoulder_score, _, shoulder_meta = _magnitude_component(
        player["shoulder_line_deg"],
        reference["shoulder_line_deg"],
        value_label="shoulder_line",
        min_tolerance=_ALIGNMENT_MIN_TOLERANCE,
    )
    hip_score, _, hip_meta = _magnitude_component(
        player["hip_line_deg"],
        reference["hip_line_deg"],
        value_label="hip_line",
        min_tolerance=_ALIGNMENT_MIN_TOLERANCE,
    )
    magnitude_score = 0.50 * trunk_score + 0.25 * shoulder_score + 0.25 * hip_score
    consistency_score, consistency_meta = _consistency_component(player_trial_facing)
    confidence, confidence_meta = _confidence_at_contact(
        player_value=player["trunk_rotation_deg"],
        joint_confidence_at_contact=joint_confidence_at_contact,
        local_finite_fraction=None,
    )
    errors = alignment_error_measurements(player, reference)

    builder = (
        FeatureScoreBuilder(
            "Body Alignment",
            "Contact",
            unit="deg",
            confidence=confidence,
        )
        .set_direction(direction)
        .set_values(
            player_value=player["trunk_rotation_deg"],
            reference_value=reference["trunk_rotation_deg"],
        )
        .add_component("magnitude", magnitude_score, _WEIGHT_MAGNITUDE)
        .add_component("consistency", consistency_score, _WEIGHT_CONSISTENCY)
        .add_measurement("scoring_mode", "biomechanical")
        .add_measurement("player_contact_frame", player_contact_frame)
        .add_measurement("reference_contact_frame", reference_contact_frame)
        .add_measurement("player_shoulder_line_deg", player["shoulder_line_deg"])
        .add_measurement("reference_shoulder_line_deg", reference["shoulder_line_deg"])
        .add_measurement("player_hip_line_deg", player["hip_line_deg"])
        .add_measurement("reference_hip_line_deg", reference["hip_line_deg"])
        .add_measurement("player_trunk_rotation_deg", player["trunk_rotation_deg"])
        .add_measurement("reference_trunk_rotation_deg", reference["trunk_rotation_deg"])
        .add_measurement("trunk_score", round(trunk_score, 2))
        .add_measurement("shoulder_score", round(shoulder_score, 2))
        .add_measurement("hip_score", round(hip_score, 2))
        .add_measurement("magnitude_score", round(magnitude_score, 2))
    )
    for key, value in errors.items():
        builder.add_measurement(key, value)
    for meta in (trunk_meta, shoulder_meta, hip_meta, consistency_meta, confidence_meta):
        for key, value in meta.items():
            if key == "magnitude_score":
                continue
            builder.add_measurement(key, value)
    return builder.build()


def score_body_alignment_from_scalars(
    player_value: float, reference_value: float
) -> FeatureScoreResult:
    """Scalar fallback treats the scalar as trunk/facing angle at contact."""
    return score_contact_event_from_scalars(
        feature_name="Body Alignment",
        player_value=player_value,
        reference_value=reference_value,
        unit="deg",
        value_label="trunk_rotation",
        min_tolerance=_ALIGNMENT_MIN_TOLERANCE,
    )
