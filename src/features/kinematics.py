from __future__ import annotations

import numpy as np

from src.features.acceleration import acceleration_from_positions
from src.features.velocity import finite_difference, speed_from_velocity
from src.skeleton import SkeletonSequence

JOINT_FEATURE_ALIASES = {
    "left_wrist": "left_hand",
    "right_wrist": "right_hand",
    "left_ankle": "left_foot",
    "right_ankle": "right_foot",
}
AXES = ("x", "y", "z")


def feature_name_for_joint(joint_name: str) -> str:
    """Return the public feature prefix for a skeleton joint."""
    return JOINT_FEATURE_ALIASES.get(joint_name, joint_name)


def compute_joint_kinematic_features(sequence: SkeletonSequence) -> dict[str, np.ndarray]:
    """Compute velocity, acceleration, and speed features for every joint."""
    velocity = finite_difference(sequence.joint_positions, sequence.fps)
    acceleration = acceleration_from_positions(sequence.joint_positions, sequence.fps)
    features: dict[str, np.ndarray] = {}

    for joint_idx, joint_name in enumerate(sequence.joint_names):
        prefix = feature_name_for_joint(joint_name)
        for axis_idx, axis in enumerate(AXES):
            features[f"{prefix}_velocity_{axis}"] = velocity[:, joint_idx, axis_idx]
            features[f"{prefix}_acceleration_{axis}"] = acceleration[:, joint_idx, axis_idx]
        features[f"{prefix}_speed"] = speed_from_velocity(velocity[:, joint_idx, :])
        features[f"{prefix}_acceleration"] = np.linalg.norm(acceleration[:, joint_idx, :], axis=1)

    return features

