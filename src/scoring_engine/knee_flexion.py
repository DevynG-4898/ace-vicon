"""Production Knee Flexion scoring (Loading phase).

Mathematics
-----------
Knee flexion is scored as **flexion depth in degrees**:

    flexion(t) = 180° − min(left_knee_interior(t), right_knee_interior(t))

Higher values mean a deeper bend. Peak flexion is the maximum of ``flexion(t)``
inside the Loading window (or the full series if no window is provided).

All magnitude / timing / smoothness / consistency / confidence math lives in
:mod:`src.scoring_engine.angle_kernel`. This module only prepares the
knee-specific series and labels, then delegates to the Angle Kernel.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.scoring_engine.angle_kernel import (
    NEUTRAL_CONSISTENCY_SCORE,
    score_angle_feature,
    score_angle_feature_from_scalars,
)
from src.scoring_engine.result import FeatureScoreResult

# Re-export for existing imports / tests.
__all__ = [
    "NEUTRAL_CONSISTENCY_SCORE",
    "compute_knee_flexion_series",
    "score_knee_flexion_biomechanics",
    "score_knee_flexion_from_scalars",
]


def compute_knee_flexion_series(
    left_knee_interior_deg: np.ndarray,
    right_knee_interior_deg: np.ndarray,
) -> np.ndarray:
    """Convert interior knee angles to flexion depth (higher = more flexed)."""
    left = np.asarray(left_knee_interior_deg, dtype=float)
    right = np.asarray(right_knee_interior_deg, dtype=float)
    if left.shape != right.shape:
        raise ValueError("left and right knee series must share the same shape")
    interior = np.minimum(left, right)
    return 180.0 - interior


def score_knee_flexion_biomechanics(
    *,
    player_series: np.ndarray,
    reference_series: np.ndarray,
    loading_slice: slice | None = None,
    reference_loading_slice: slice | None = None,
    player_trial_peaks: Sequence[float] | None = None,
    player_joint_confidence: np.ndarray | None = None,
    fps: float | None = None,
) -> FeatureScoreResult:
    """Score Knee Flexion from player/reference flexion-depth trajectories."""
    return score_angle_feature(
        feature_name="Knee Flexion",
        phase="Loading",
        player_series=player_series,
        reference_series=reference_series,
        player_phase_slice=loading_slice,
        reference_phase_slice=reference_loading_slice,
        player_trial_peaks=player_trial_peaks,
        player_joint_confidence=player_joint_confidence,
        fps=fps,
        unit="deg",
        peak_mode="max",
        peak_label="flexion",
    )


def score_knee_flexion_from_scalars(
    player_value: float,
    reference_value: float,
) -> FeatureScoreResult:
    """Scalar-only fallback when flexion trajectories are not available."""
    return score_angle_feature_from_scalars(
        feature_name="Knee Flexion",
        phase="Loading",
        player_value=player_value,
        reference_value=reference_value,
        unit="deg",
        peak_label="flexion",
    )
