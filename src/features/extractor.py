from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.features.angular_velocity import angular_velocity_from_degrees
from src.features.center_of_mass import compute_center_of_mass_features
from src.features.feature_sequence import FeatureSequence
from src.features.joint_angles import compute_joint_angle_features
from src.features.kinematics import compute_joint_kinematic_features
from src.skeleton import NORMALIZED_COORDINATE_SYSTEM, SkeletonSequence


def _require_normalized(sequence: SkeletonSequence) -> None:
    if sequence.coordinate_system != NORMALIZED_COORDINATE_SYSTEM:
        raise ValueError(
            "Feature extraction requires a normalized SkeletonSequence "
            f"({NORMALIZED_COORDINATE_SYSTEM}); got {sequence.coordinate_system}"
        )


def extract_features(sequence: SkeletonSequence) -> FeatureSequence:
    """Extract canonical biomechanical features from a normalized skeleton."""
    _require_normalized(sequence)

    features: dict[str, np.ndarray] = {}
    angle_features = compute_joint_angle_features(sequence)
    features.update(angle_features)
    features.update(compute_joint_kinematic_features(sequence))
    features.update(compute_center_of_mass_features(sequence))

    for name, values in angle_features.items():
        features[f"{name}_angular_velocity"] = angular_velocity_from_degrees(values, sequence.fps)

    metadata = {
        "source": "SkeletonSequence",
        "source_coordinate_system": sequence.coordinate_system,
        "source_sequence_source": sequence.source,
        "feature_count": len(features),
        "extraction_timestamp": datetime.now(UTC).isoformat(),
        "skeleton_metadata": dict(sequence.metadata),
    }

    return FeatureSequence(
        frames=sequence.frames,
        fps=sequence.fps,
        features=features,
        metadata=metadata,
        source_sequence=sequence,
    )


def plot_features(
    feature_sequence: FeatureSequence,
    feature_names: list[str] | tuple[str, ...],
    *,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot one or more named features over frame number."""
    if not feature_names:
        raise ValueError("At least one feature name is required")

    fig, ax = plt.subplots(figsize=(10, 5))
    for name in feature_names:
        ax.plot(feature_sequence.frames, feature_sequence.feature(name), label=name)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Feature value")
    ax.set_title("ACE biomechanical features")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig

