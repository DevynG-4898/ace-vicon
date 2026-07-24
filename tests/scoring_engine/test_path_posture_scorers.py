"""Tests for Path / Posture family scorers."""

from __future__ import annotations

import numpy as np
import pytest

from src.scoring_engine.feature_scorer import (
    score_balance,
    score_center_of_mass,
    score_follow_through,
    score_recovery_position,
    score_weight_transfer,
)
from src.scoring_engine.path_posture import path_length


def _com_traj(n: int = 40, *, drop: float = 0.20, forward: float = 0.30) -> np.ndarray:
    """CoM trajectory: dips then rises, moves forward."""
    t = np.linspace(0.0, 1.0, n)
    com = np.zeros((n, 3), dtype=float)
    com[:, 1] = 1.0 - drop * np.exp(-0.5 * ((t - 0.4) / 0.15) ** 2)
    com[:, 2] = forward * t
    return com


def _foot_pair(n: int = 40, *, back_z: float = 0.0, front_z: float = 0.40) -> tuple[np.ndarray, np.ndarray]:
    back = np.zeros((n, 3), dtype=float)
    front = np.zeros((n, 3), dtype=float)
    back[:, 2] = back_z
    front[:, 2] = front_z
    back[:, 0] = -0.15
    front[:, 0] = 0.15
    return back, front


class TestPathPostureScorers:
    def test_center_of_mass_biomechanics(self):
        player = _com_traj(drop=0.10)
        reference = _com_traj(drop=0.25)
        result = score_center_of_mass(
            0.10,
            0.25,
            player_com=player,
            reference_com=reference,
            player_phase_slice=slice(5, 35),
            reference_phase_slice=slice(5, 35),
            fps=100.0,
        )
        assert result.name == "Center of Mass"
        assert result.phase == "Loading"
        assert result.measurements["scoring_mode"] == "biomechanical"
        assert result.direction == "too_low"
        assert "timing_score" in result.measurements
        assert "smoothness_score" in result.measurements

    def test_follow_through_biomechanics(self):
        n = 50
        hand_r = np.zeros((n, 3))
        hand_r[:, 2] = np.linspace(0.0, 1.0, n)
        hand_p = np.zeros((n, 3))
        hand_p[:, 2] = np.linspace(0.0, 0.4, n)
        result = score_follow_through(
            path_length(hand_p[20:]),
            path_length(hand_r[20:]),
            player_hand=hand_p,
            reference_hand=hand_r,
            player_contact_frame=20,
            reference_contact_frame=20,
            fps=100.0,
        )
        assert result.name == "Follow Through"
        assert result.phase == "Deceleration"
        assert result.direction == "too_low"
        assert "path_length_difference" in result.measurements or "player_path_length" in result.measurements

    def test_balance_biomechanics(self):
        n = 40
        # Stable reference CoM; shaky player
        ref = np.zeros((n, 3))
        ref[:, 1] = 1.0
        player = ref.copy()
        player[:, 0] = 0.05 * np.sin(np.linspace(0, 12, n))
        back, front = _foot_pair(n)
        result = score_balance(
            0.0,
            0.0,
            player_com=player,
            reference_com=ref,
            player_left_foot=back,
            player_right_foot=front,
            reference_left_foot=back,
            reference_right_foot=front,
            player_phase_slice=slice(10, 40),
            reference_phase_slice=slice(10, 40),
            fps=100.0,
        )
        assert result.name == "Balance"
        assert result.phase == "Finish"
        assert result.measurements["scoring_mode"] == "biomechanical"
        assert "player_sway_rms" in result.measurements

    def test_weight_transfer_biomechanics(self):
        player = _com_traj(forward=0.15)
        reference = _com_traj(forward=0.40)
        back, front = _foot_pair()
        result = score_weight_transfer(
            0.5,
            0.9,
            player_com=player,
            reference_com=reference,
            player_back_foot=back,
            player_front_foot=front,
            reference_back_foot=back,
            reference_front_foot=front,
            player_phase_slice=slice(0, 40),
            reference_phase_slice=slice(0, 40),
            fps=100.0,
        )
        assert result.name == "Weight Transfer"
        assert result.phase == "Finish"
        assert "player_transfer_fraction" in result.measurements
        assert "timing_score" in result.measurements

    def test_recovery_position_biomechanics(self):
        ref = _com_traj()
        player = _com_traj()
        # Keep player moving longer (harder to stabilize)
        player[:, 0] = 0.2 * np.sin(np.linspace(0, 20, len(player)))
        back, front = _foot_pair()
        result = score_recovery_position(
            0.5,
            0.4,
            player_com=player,
            reference_com=ref,
            player_left_foot=back,
            player_right_foot=front,
            reference_left_foot=back,
            reference_right_foot=front,
            player_contact_frame=15,
            reference_contact_frame=15,
            player_phase_slice=slice(0, 40),
            reference_phase_slice=slice(0, 40),
            fps=100.0,
        )
        assert result.name == "Recovery Position"
        assert result.unit == "s"
        assert "player_recovery_time_s" in result.measurements

    def test_scalar_fallbacks(self):
        assert score_center_of_mass(0.0, 35.0).measurements["scoring_mode"] == "scalar_fallback"
        assert score_follow_through(100.0, 100.0).measurements["scoring_mode"] == "scalar_fallback"
        assert score_balance(100.0, 100.0).direction == "acceptable"
        assert score_weight_transfer(70.0, 70.0).direction == "acceptable"
        rec = score_recovery_position(0.40, 0.40)
        assert rec.measurements["scoring_mode"] == "scalar_fallback"
        assert rec.player_value == pytest.approx(0.40)
