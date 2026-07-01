from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np

from src.skeleton.geometry import median_positive_length, orthonormal_body_axes
from src.skeleton.sequence import SkeletonSequence
from src.skeleton.transforms import rotate_positions, scale_positions, translate_positions

REQUIRED_ORIENTATION_JOINTS = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
NORMALIZED_COORDINATE_SYSTEM = "ace_canonical_body"


def _scale_factor(
    left_hip: np.ndarray,
    right_hip: np.ndarray,
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
) -> float:
    shoulder_width = median_positive_length(np.linalg.norm(right_shoulder - left_shoulder, axis=1))
    if shoulder_width is not None:
        return shoulder_width

    hip_width = median_positive_length(np.linalg.norm(right_hip - left_hip, axis=1))
    if hip_width is not None:
        return hip_width

    hip_mid = (left_hip + right_hip) / 2.0
    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    torso_length = median_positive_length(np.linalg.norm(shoulder_mid - hip_mid, axis=1))
    if torso_length is not None:
        return torso_length

    raise ValueError("Cannot normalize scale: shoulder width, hip width, and torso length are all degenerate")


def _normalization_metadata(
    sequence: SkeletonSequence,
    translation: np.ndarray,
    rotation: np.ndarray,
    scale_factor: float,
) -> dict[str, Any]:
    metadata = dict(sequence.metadata)
    metadata["normalization"] = {
        "translation_vector": translation.tolist(),
        "rotation_matrix": rotation.tolist(),
        "scale_factor": float(scale_factor),
        "normalization_timestamp": datetime.now(UTC).isoformat(),
        "source_file": metadata.get("source_file"),
        "motionbert_version": metadata.get("motionbert_version"),
    }
    return metadata


def normalize_skeleton(sequence: SkeletonSequence) -> SkeletonSequence:
    """Translate, rotate, and scale a skeleton into ACE's canonical body frame.

    The input sequence is never modified. Body orientation is estimated from
    left/right shoulders and hips for each frame, so normalization does not rely
    on camera orientation.
    """
    for name in REQUIRED_ORIENTATION_JOINTS:
        sequence.joint_index(name)

    left_hip = sequence.joint("left_hip")
    right_hip = sequence.joint("right_hip")
    left_shoulder = sequence.joint("left_shoulder")
    right_shoulder = sequence.joint("right_shoulder")

    pelvis = (left_hip + right_hip) / 2.0
    translated = translate_positions(sequence.joint_positions, pelvis)
    axes = orthonormal_body_axes(left_hip, right_hip, left_shoulder, right_shoulder)
    rotated = rotate_positions(translated, axes)
    scale_factor = _scale_factor(left_hip, right_hip, left_shoulder, right_shoulder)
    normalized = scale_positions(rotated, scale_factor)

    return sequence.with_positions(
        normalized,
        metadata=_normalization_metadata(sequence, pelvis, axes, scale_factor),
        coordinate_system=NORMALIZED_COORDINATE_SYSTEM,
        source=sequence.source,
    )

