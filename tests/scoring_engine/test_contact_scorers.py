"""Tests for Contact Kernel and contact-family scorers."""

from __future__ import annotations

import numpy as np
import pytest

from src.scoring_engine.contact_kernel import (
    score_contact_event,
    score_contact_event_from_scalars,
)
from src.scoring_engine.contact_series import (
    contact_height,
    contact_position_offsets,
    normalized_height,
)
from src.scoring_engine.feature_scorer import (
    score_body_alignment,
    score_contact_height,
    score_contact_position,
)


def _traj(n: int = 40, *, height: float = 2.0, forward: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic hand and pelvis trajectories (N, 3) in x,y,z."""
    hand = np.zeros((n, 3), dtype=float)
    pelvis = np.zeros((n, 3), dtype=float)
    hand[:, 1] = height
    hand[:, 2] = forward
    pelvis[:, 1] = 1.0
    return hand, pelvis


class TestContactSeries:
    def test_contact_height_and_normalized(self):
        hand, _ = _traj(height=2.45)
        h = contact_height(hand, 20)
        assert h == pytest.approx(2.45)
        assert normalized_height(h, reference_height=2.45) == pytest.approx(1.0)

    def test_contact_position_offsets(self):
        hand, pelvis = _traj(forward=0.40)
        hand[:, 0] = 0.05
        offsets = contact_position_offsets(hand, pelvis, 20)
        assert offsets["forward"] == pytest.approx(0.40)
        assert offsets["lateral"] == pytest.approx(0.05)


class TestContactKernel:
    def test_matching_height_scores_high(self):
        result = score_contact_event(
            feature_name="Contact Height",
            player_value=2450.0,
            reference_value=2450.0,
            value_label="height",
            player_contact_frame=30,
            reference_contact_frame=28,
            unit="mm",
        )
        assert result.score >= 90.0
        assert result.direction == "acceptable"
        assert result.measurements["player_contact_frame"] == 30
        assert "magnitude_score" in result.measurements
        assert "timing_score" not in result.measurements

    def test_low_height_too_low(self):
        result = score_contact_event(
            feature_name="Contact Height",
            player_value=2100.0,
            reference_value=2450.0,
            value_label="height",
            min_tolerance=40.0,
        )
        assert result.direction == "too_low"
        assert result.measurements["magnitude_score"] < 80.0

    def test_scalar_fallback(self):
        result = score_contact_event_from_scalars(
            feature_name="Contact Position",
            player_value=200.0,
            reference_value=350.0,
            value_label="forward",
        )
        assert result.measurements["scoring_mode"] == "scalar_fallback"
        assert result.confidence == pytest.approx(0.55)


class TestContactFeatureEntries:
    def test_contact_height_biomechanics(self):
        hand_p, _ = _traj(height=2.2)
        hand_r, _ = _traj(height=2.45)
        result = score_contact_height(
            2.2,
            2.45,
            player_hand_positions=hand_p,
            reference_hand_positions=hand_r,
            player_contact_frame=20,
            reference_contact_frame=20,
            unit="normalized",
        )
        assert result.name == "Contact Height"
        assert result.phase == "Contact"
        assert result.measurements["scoring_mode"] == "biomechanical"
        assert "normalized_height" in result.measurements
        assert result.direction == "too_low"

    def test_contact_position_biomechanics(self):
        hand_p, pelvis_p = _traj(forward=0.18)
        hand_r, pelvis_r = _traj(forward=0.35)
        result = score_contact_position(
            0.18,
            0.35,
            player_hand_positions=hand_p,
            player_pelvis_positions=pelvis_p,
            reference_hand_positions=hand_r,
            reference_pelvis_positions=pelvis_r,
            player_contact_frame=20,
            reference_contact_frame=20,
            unit="normalized",
        )
        assert result.name == "Contact Position"
        assert "player_forward" in result.measurements
        assert "player_lateral" in result.measurements
        assert result.direction == "too_low"

    def test_body_alignment_biomechanics(self):
        n = 40
        player_trunk = np.full(n, 20.0)
        ref_trunk = np.full(n, 5.0)
        line = np.zeros(n)
        result = score_body_alignment(
            20.0,
            5.0,
            player_shoulder_line_deg=line,
            player_hip_line_deg=line,
            player_trunk_rotation_deg=player_trunk,
            reference_shoulder_line_deg=line,
            reference_hip_line_deg=line,
            reference_trunk_rotation_deg=ref_trunk,
            player_contact_frame=25,
            reference_contact_frame=25,
        )
        assert result.name == "Body Alignment"
        assert result.unit == "deg"
        assert "trunk_rotation_error_deg" in result.measurements
        assert result.direction == "too_high"
