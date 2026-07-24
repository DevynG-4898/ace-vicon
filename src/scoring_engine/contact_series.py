"""Contact-event measurement extractors (hand height, position, body alignment)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.skeleton import SkeletonSequence

EPSILON = 1e-9


def value_at_frame(series: np.ndarray, frame: int) -> float:
    """Return a finite scalar at ``frame`` (clamped to series bounds)."""
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"expected 1D series; got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("series is empty")
    idx = int(np.clip(frame, 0, arr.size - 1))
    value = float(arr[idx])
    if not np.isfinite(value):
        raise ValueError(f"non-finite value at contact frame {idx}")
    return value


def local_window(
    series: np.ndarray,
    frame: int,
    *,
    radius: int = 5,
) -> np.ndarray:
    """Return ``series[frame-radius : frame+radius+1]`` (clipped)."""
    arr = np.asarray(series, dtype=float)
    idx = int(np.clip(frame, 0, max(arr.size - 1, 0)))
    start = max(0, idx - radius)
    end = min(arr.size, idx + radius + 1)
    return arr[start:end].copy()


def contact_height(
    hand_positions: np.ndarray,
    contact_frame: int,
    *,
    vertical_axis: int = 1,
) -> float:
    """Hand height at contact (default vertical = Y in normalized skeleton)."""
    pos = np.asarray(hand_positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] < 3:
        raise ValueError(f"hand_positions must be (N, 3); got {pos.shape}")
    return value_at_frame(pos[:, vertical_axis], contact_frame)


def contact_position_offsets(
    hand_positions: np.ndarray,
    pelvis_positions: np.ndarray,
    contact_frame: int,
    *,
    forward_axis: int = 2,
    lateral_axis: int = 0,
    vertical_axis: int = 1,
) -> dict[str, float]:
    """Hand − pelvis offsets at contact in body/normalized axes.

    Returns forward / lateral / vertical offsets (higher forward = more in front).
    """
    hand = np.asarray(hand_positions, dtype=float)
    pelvis = np.asarray(pelvis_positions, dtype=float)
    if hand.shape != pelvis.shape or hand.ndim != 2 or hand.shape[1] < 3:
        raise ValueError("hand and pelvis positions must both be (N, 3)")
    idx = int(np.clip(contact_frame, 0, hand.shape[0] - 1))
    delta = hand[idx] - pelvis[idx]
    return {
        "forward": float(delta[forward_axis]),
        "lateral": float(delta[lateral_axis]),
        "vertical": float(delta[vertical_axis]),
    }


def body_alignment_angles(
    *,
    shoulder_line_deg: np.ndarray,
    hip_line_deg: np.ndarray,
    trunk_rotation_deg: np.ndarray,
    contact_frame: int,
) -> dict[str, float]:
    """Shoulder line, hip line, and trunk rotation at contact (degrees)."""
    return {
        "shoulder_line_deg": value_at_frame(shoulder_line_deg, contact_frame),
        "hip_line_deg": value_at_frame(hip_line_deg, contact_frame),
        "trunk_rotation_deg": value_at_frame(trunk_rotation_deg, contact_frame),
    }


def body_alignment_from_skeleton(
    sequence: SkeletonSequence,
    contact_frame: int,
) -> dict[str, float]:
    """Derive alignment angles from a skeleton at the contact frame."""
    from src.features.joint_angles import line_angle_xy, trunk_rotation

    return body_alignment_angles(
        shoulder_line_deg=line_angle_xy(sequence, "left_shoulder", "right_shoulder"),
        hip_line_deg=line_angle_xy(sequence, "left_hip", "right_hip"),
        trunk_rotation_deg=trunk_rotation(sequence),
        contact_frame=contact_frame,
    )


def normalized_height(
    height: float,
    *,
    reference_height: float,
) -> float:
    """Height as a fraction of the reference contact height."""
    denom = abs(reference_height) if abs(reference_height) > EPSILON else EPSILON
    return float(height / denom)


def alignment_error_measurements(
    player: dict[str, float],
    reference: dict[str, float],
) -> dict[str, Any]:
    """Signed errors for each alignment channel (player − reference)."""
    return {
        "shoulder_line_error_deg": round(
            player["shoulder_line_deg"] - reference["shoulder_line_deg"], 4
        ),
        "hip_line_error_deg": round(player["hip_line_deg"] - reference["hip_line_deg"], 4),
        "trunk_rotation_error_deg": round(
            player["trunk_rotation_deg"] - reference["trunk_rotation_deg"], 4
        ),
    }
