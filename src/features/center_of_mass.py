from __future__ import annotations

import numpy as np

from src.features.acceleration import acceleration_from_positions
from src.features.velocity import finite_difference, speed_from_velocity
from src.skeleton import SkeletonSequence

SEGMENT_WEIGHTS = {
    "head": 0.081,
    "trunk": 0.497,
    "left_upper_arm": 0.028,
    "right_upper_arm": 0.028,
    "left_forearm_hand": 0.044,
    "right_forearm_hand": 0.044,
    "left_thigh": 0.10,
    "right_thigh": 0.10,
    "left_shank_foot": 0.061,
    "right_shank_foot": 0.061,
}


def _midpoint(sequence: SkeletonSequence, a: str, b: str) -> np.ndarray:
    return (sequence.joint(a) + sequence.joint(b)) / 2.0


def compute_center_of_mass(sequence: SkeletonSequence) -> np.ndarray:
    """Approximate whole-body center of mass from weighted segment centers."""
    pelvis = sequence.joint("pelvis")
    thorax = sequence.joint("thorax")
    centers = {
        "head": _midpoint(sequence, "neck", "head"),
        "trunk": (pelvis + thorax) / 2.0,
        "left_upper_arm": _midpoint(sequence, "left_shoulder", "left_elbow"),
        "right_upper_arm": _midpoint(sequence, "right_shoulder", "right_elbow"),
        "left_forearm_hand": _midpoint(sequence, "left_elbow", "left_wrist"),
        "right_forearm_hand": _midpoint(sequence, "right_elbow", "right_wrist"),
        "left_thigh": _midpoint(sequence, "left_hip", "left_knee"),
        "right_thigh": _midpoint(sequence, "right_hip", "right_knee"),
        "left_shank_foot": _midpoint(sequence, "left_knee", "left_ankle"),
        "right_shank_foot": _midpoint(sequence, "right_knee", "right_ankle"),
    }

    total_weight = float(sum(SEGMENT_WEIGHTS.values()))
    weighted = np.zeros_like(sequence.joint_positions[:, 0, :], dtype=float)
    for name, center in centers.items():
        weighted += SEGMENT_WEIGHTS[name] * center
    return weighted / total_weight


def compute_center_of_mass_features(sequence: SkeletonSequence) -> dict[str, np.ndarray]:
    """Compute center-of-mass position, velocity, and acceleration features."""
    com = compute_center_of_mass(sequence)
    velocity = finite_difference(com, sequence.fps)
    acceleration = acceleration_from_positions(com, sequence.fps)
    return {
        "center_of_mass_x": com[:, 0],
        "center_of_mass_y": com[:, 1],
        "center_of_mass_z": com[:, 2],
        "center_of_mass_velocity_x": velocity[:, 0],
        "center_of_mass_velocity_y": velocity[:, 1],
        "center_of_mass_velocity_z": velocity[:, 2],
        "center_of_mass_velocity": speed_from_velocity(velocity),
        "center_of_mass_acceleration_x": acceleration[:, 0],
        "center_of_mass_acceleration_y": acceleration[:, 1],
        "center_of_mass_acceleration_z": acceleration[:, 2],
        "center_of_mass_acceleration": np.linalg.norm(acceleration, axis=1),
    }

