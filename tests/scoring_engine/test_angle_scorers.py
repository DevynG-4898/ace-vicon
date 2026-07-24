"""Tests for angle series extractors and angle-kernel feature scorers."""

from __future__ import annotations

import numpy as np
import pytest

from src.scoring_engine import angle_scorers, angle_series
from src.scoring_engine.feature_scorer import (
    score_arm_extension,
    score_hip_flexion,
    score_right_elbow_flexion,
    score_shoulder_external_rotation,
    score_shoulder_internal_rotation,
    score_trunk_flexion,
)


def _peak_series(n: int = 40, peak_at: int = 20, peak: float = 90.0, base: float = 40.0) -> np.ndarray:
    t = np.arange(n, dtype=float)
    return base + (peak - base) * np.exp(-0.5 * ((t - peak_at) / 6.0) ** 2)


class TestAngleSeries:
    def test_elbow_flexion_and_extension(self):
        interior = np.array([180.0, 90.0, 160.0])
        assert angle_series.compute_elbow_flexion_series(interior).tolist() == [0.0, 90.0, 20.0]
        assert angle_series.compute_elbow_extension_series(interior).tolist() == [180.0, 90.0, 160.0]

    def test_shoulder_tilt_is_absolute(self):
        assert angle_series.compute_shoulder_tilt_series(
            np.array([-12.0, 5.0, 0.0])
        ).tolist() == [12.0, 5.0, 0.0]

    def test_trunk_rotation_magnitude(self):
        assert angle_series.compute_trunk_rotation_series(
            np.array([-30.0, 10.0])
        ).tolist() == [30.0, 10.0]

    def test_shoulder_ir_proxy_tracks_er_drop(self):
        er = np.array([10.0, 40.0, 50.0, 35.0, 20.0])
        ir = angle_series.compute_shoulder_ir_proxy_series(er)
        assert ir[2] == pytest.approx(0.0)
        assert ir[4] == pytest.approx(30.0)


class TestAngleScorersViaKernel:
    def test_hip_flexion_biomechanical_path(self):
        ref = _peak_series(peak=50.0)
        player = _peak_series(peak=30.0)
        result = score_hip_flexion(
            30.0,
            50.0,
            player_series=angle_series.compute_hip_flexion_series(player),
            reference_series=angle_series.compute_hip_flexion_series(ref),
            player_phase_slice=slice(5, 35),
            reference_phase_slice=slice(5, 35),
        )
        assert result.name == "Hip Flexion"
        assert result.phase == "Loading"
        assert result.measurements["scoring_mode"] == "biomechanical"
        assert result.direction == "too_low"
        assert "magnitude_score" in result.measurements

    def test_right_elbow_flexion_scalar_fallback(self):
        result = score_right_elbow_flexion(60.0, 95.0)
        assert result.name == "Right Elbow Flexion"
        assert result.phase == "Cocking"
        assert result.measurements["scoring_mode"] == "scalar_fallback"
        assert result.direction == "too_low"

    def test_shoulder_er_and_ir_use_kernel(self):
        ref = _peak_series(peak=110.0)
        player = _peak_series(peak=70.0)
        er = score_shoulder_external_rotation(
            70.0,
            110.0,
            player_series=angle_series.compute_shoulder_er_proxy_series(player),
            reference_series=angle_series.compute_shoulder_er_proxy_series(ref),
            player_phase_slice=slice(5, 35),
            reference_phase_slice=slice(5, 35),
        )
        assert er.name == "Shoulder External Rotation"
        assert er.measurements["scoring_mode"] == "biomechanical"

        ir_player = angle_series.compute_shoulder_ir_proxy_series(player)
        ir_ref = angle_series.compute_shoulder_ir_proxy_series(ref)
        ir = score_shoulder_internal_rotation(
            float(np.nanmax(ir_player)),
            float(np.nanmax(ir_ref)),
            player_series=ir_player,
            reference_series=ir_ref,
            player_phase_slice=slice(5, 35),
            reference_phase_slice=slice(5, 35),
        )
        assert ir.name == "Shoulder Internal Rotation"
        assert ir.phase == "Acceleration"

    def test_arm_extension_and_trunk_flexion_phases(self):
        series = _peak_series(peak=170.0)
        arm = score_arm_extension(
            170.0,
            170.0,
            player_series=angle_series.compute_arm_extension_series(series),
            reference_series=angle_series.compute_arm_extension_series(series),
            player_phase_slice=slice(15, 25),
            reference_phase_slice=slice(15, 25),
        )
        assert arm.phase == "Contact"

        tilt = _peak_series(peak=25.0)
        trunk = score_trunk_flexion(
            25.0,
            25.0,
            player_series=angle_series.compute_trunk_flexion_series(tilt),
            reference_series=angle_series.compute_trunk_flexion_series(tilt),
            player_phase_slice=slice(5, 35),
            reference_phase_slice=slice(5, 35),
        )
        assert trunk.phase == "Deceleration"
        assert trunk.name == "Trunk Flexion"

    def test_all_angle_scorers_expose_biomechanics_entrypoints(self):
        names = [
            "hip_flexion",
            "shoulder_tilt",
            "toss_arm_extension",
            "trunk_rotation",
            "pelvis_rotation",
            "right_elbow_flexion",
            "left_elbow_flexion",
            "shoulder_external_rotation",
            "forearm_angle",
            "shoulder_internal_rotation",
            "right_elbow_extension",
            "left_elbow_extension",
            "arm_extension",
            "trunk_flexion",
        ]
        for name in names:
            assert hasattr(angle_scorers, f"score_{name}_biomechanics")
            assert hasattr(angle_scorers, f"score_{name}_from_scalars")
