"""Tests for the Velocity Kernel and velocity-family scorers."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.angular_velocity import angular_velocity_from_degrees
from src.scoring_engine.feature_scorer import (
    score_hip_rotation_velocity,
    score_shoulder_deceleration,
    score_trunk_rotation_velocity,
)
from src.scoring_engine.velocity_kernel import (
    score_velocity_feature,
    score_velocity_feature_from_scalars,
)
from src.scoring_engine.velocity_series import (
    compute_deceleration_magnitude_series,
    compute_hip_rotation_velocity_series,
    compute_shoulder_deceleration_series,
    compute_trunk_rotation_velocity_series,
)


def _angle_ramp(n: int = 40, fps: float = 100.0) -> tuple[np.ndarray, float]:
    """Smooth angle that rises then falls — produces a clear |ω| peak."""
    t = np.arange(n, dtype=float) / fps
    # degrees: smooth pulse
    angle = 40.0 * np.exp(-0.5 * ((t - 0.15) / 0.04) ** 2)
    return angle, fps


class TestVelocitySeries:
    def test_trunk_speed_is_abs_omega(self):
        angle, fps = _angle_ramp()
        speed = compute_trunk_rotation_velocity_series(angle, fps)
        omega = angular_velocity_from_degrees(angle, fps)
        assert np.allclose(speed, np.abs(omega), equal_nan=True)

    def test_hip_speed_from_yaw(self):
        yaw = np.linspace(0.0, 40.0, 30)
        speed = compute_hip_rotation_velocity_series(yaw, fps=60.0)
        assert speed.shape == yaw.shape
        assert np.all(speed[np.isfinite(speed)] >= 0.0)

    def test_deceleration_positive_when_slowing(self):
        # Constant then drop speed → positive deceleration after the drop starts
        speed = np.concatenate([np.full(10, 200.0), np.linspace(200.0, 0.0, 20)])
        decel = compute_deceleration_magnitude_series(
            speed, fps=100.0, input_is_velocity=True
        )
        assert np.nanmax(decel) > 0.0


class TestVelocityKernel:
    def test_matching_velocity_scores_high(self):
        angle, fps = _angle_ramp()
        series = compute_trunk_rotation_velocity_series(angle, fps)
        result = score_velocity_feature(
            feature_name="Trunk Rotation Velocity",
            phase="Acceleration",
            player_series=series,
            reference_series=series,
            player_phase_slice=slice(5, 35),
            reference_phase_slice=slice(5, 35),
            fps=fps,
            peak_label="velocity",
            unit="deg/s",
        )
        assert result.score >= 90.0
        assert result.direction == "acceptable"
        assert result.measurements["scoring_mode"] == "biomechanical"
        assert "player_peak_velocity" in result.measurements
        assert "timing_score" in result.measurements
        assert "smoothness_score" in result.measurements
        assert "consistency_score" in result.measurements

    def test_slow_trunk_is_too_low(self):
        angle, fps = _angle_ramp()
        ref = compute_trunk_rotation_velocity_series(angle, fps)
        player = compute_trunk_rotation_velocity_series(angle * 0.4, fps)
        result = score_velocity_feature(
            feature_name="Trunk Rotation Velocity",
            phase="Acceleration",
            player_series=player,
            reference_series=ref,
            player_phase_slice=slice(5, 35),
            reference_phase_slice=slice(5, 35),
            fps=fps,
        )
        assert result.direction == "too_low"
        assert result.measurements["magnitude_score"] < 85.0

    def test_scalar_fallback(self):
        result = score_velocity_feature_from_scalars(
            feature_name="Hip Rotation Velocity",
            phase="Acceleration",
            player_value=300.0,
            reference_value=400.0,
            unit="deg/s",
            peak_label="velocity",
        )
        assert result.measurements["scoring_mode"] == "scalar_fallback"
        assert result.direction == "too_low"
        assert result.confidence == pytest.approx(0.55)


class TestVelocityFeatureEntries:
    def test_trunk_and_hip_entries(self):
        angle, fps = _angle_ramp()
        series = compute_trunk_rotation_velocity_series(angle, fps)
        trunk = score_trunk_rotation_velocity(
            float(np.nanmax(series)),
            float(np.nanmax(series)),
            player_series=series,
            reference_series=series,
            player_phase_slice=slice(5, 35),
            reference_phase_slice=slice(5, 35),
            fps=fps,
        )
        assert trunk.name == "Trunk Rotation Velocity"
        assert trunk.phase == "Acceleration"
        assert trunk.unit == "deg/s"

        hip = score_hip_rotation_velocity(
            380.0,
            380.0,
            player_series=series,
            reference_series=series,
            player_phase_slice=slice(5, 35),
            reference_phase_slice=slice(5, 35),
            fps=fps,
        )
        assert hip.name == "Hip Rotation Velocity"

    def test_shoulder_deceleration_entry(self):
        angle, fps = _angle_ramp()
        # Amplify then stop to create deceleration
        proxy = np.concatenate([angle, angle[::-1]])
        decel = compute_shoulder_deceleration_series(proxy, fps)
        result = score_shoulder_deceleration(
            float(np.nanmax(decel)),
            float(np.nanmax(decel)),
            player_series=decel,
            reference_series=decel,
            player_phase_slice=slice(10, 60),
            reference_phase_slice=slice(10, 60),
            fps=fps,
        )
        assert result.name == "Shoulder Deceleration"
        assert result.phase == "Deceleration"
        assert result.unit == "deg/s^2"
        assert "player_peak_deceleration" in result.measurements
