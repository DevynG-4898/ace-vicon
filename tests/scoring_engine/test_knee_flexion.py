"""Tests for production Knee Flexion biomechanical scoring."""

from __future__ import annotations

import numpy as np
import pytest

from src.scoring_engine.feature_scorer import score_knee_flexion
from src.scoring_engine.knee_flexion import (
    NEUTRAL_CONSISTENCY_SCORE,
    compute_knee_flexion_series,
    score_knee_flexion_biomechanics,
)


def _loading_series(
    n: int = 40,
    *,
    peak_at: int = 20,
    peak_flexion: float = 95.0,
    base: float = 40.0,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic knee-flexion depth series (higher = more flexed), peaking in loading."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    # Smooth rise/fall around peak_at
    series = base + (peak_flexion - base) * np.exp(-0.5 * ((t - peak_at) / 6.0) ** 2)
    if noise:
        series = series + rng.normal(0.0, noise, size=n)
    return series


class TestKneeFlexionSeries:
    def test_flexion_depth_from_interior_angles(self):
        # Interior 180° → flexion 0°; interior 90° → flexion 90°
        left = np.array([180.0, 90.0])
        right = np.array([180.0, 100.0])
        flexion = compute_knee_flexion_series(left, right)
        assert flexion[0] == pytest.approx(0.0)
        assert flexion[1] == pytest.approx(90.0)  # deepest leg (min interior)


class TestKneeFlexionBiomechanics:
    def test_matching_trajectories_score_high(self):
        ref = _loading_series(peak_at=20, peak_flexion=95.0)
        player = _loading_series(peak_at=20, peak_flexion=95.0, seed=1)
        result = score_knee_flexion_biomechanics(
            player_series=player,
            reference_series=ref,
            loading_slice=slice(5, 35),
            reference_loading_slice=slice(5, 35),
        )
        assert result.name == "Knee Flexion"
        assert result.phase == "Loading"
        assert result.score >= 85.0
        assert result.direction == "acceptable"
        assert result.confidence >= 0.8
        m = result.measurements
        assert "player_peak_flexion_deg" in m
        assert "reference_peak_flexion_deg" in m
        assert "peak_timing_error_normalized" in m
        assert "smoothness_metric" in m
        assert "magnitude_score" in m
        assert "timing_score" in m
        assert "smoothness_score" in m
        assert "consistency_score" in m

    def test_shallow_bend_is_too_low(self):
        ref = _loading_series(peak_flexion=95.0)
        player = _loading_series(peak_flexion=60.0, seed=2)
        result = score_knee_flexion_biomechanics(
            player_series=player,
            reference_series=ref,
            loading_slice=slice(5, 35),
            reference_loading_slice=slice(5, 35),
        )
        assert result.direction == "too_low"
        assert result.score < 70.0
        assert result.measurements["magnitude_score"] < 70.0

    def test_late_peak_lowers_timing_score(self):
        ref = _loading_series(peak_at=15, peak_flexion=95.0)
        player = _loading_series(peak_at=28, peak_flexion=95.0, seed=3)
        result = score_knee_flexion_biomechanics(
            player_series=player,
            reference_series=ref,
            loading_slice=slice(5, 35),
            reference_loading_slice=slice(5, 35),
        )
        assert result.measurements["timing_score"] < result.measurements["magnitude_score"]
        assert abs(result.measurements["peak_timing_error_normalized"]) > 0.2

    def test_noisy_series_lowers_smoothness(self):
        ref = _loading_series(peak_flexion=95.0, noise=0.0)
        player = _loading_series(peak_flexion=95.0, noise=4.0, seed=4)
        result = score_knee_flexion_biomechanics(
            player_series=player,
            reference_series=ref,
            loading_slice=slice(5, 35),
            reference_loading_slice=slice(5, 35),
        )
        assert result.measurements["smoothness_score"] < 90.0

    def test_single_serve_uses_neutral_consistency(self):
        ref = _loading_series()
        player = _loading_series(seed=5)
        result = score_knee_flexion_biomechanics(
            player_series=player,
            reference_series=ref,
            loading_slice=slice(5, 35),
            reference_loading_slice=slice(5, 35),
        )
        assert result.measurements["consistency_score"] == NEUTRAL_CONSISTENCY_SCORE
        assert result.measurements["consistency_mode"] == "single_serve_neutral"

    def test_multi_serve_consistency_from_peak_spread(self):
        ref = _loading_series()
        player = _loading_series(seed=6)
        result = score_knee_flexion_biomechanics(
            player_series=player,
            reference_series=ref,
            loading_slice=slice(5, 35),
            reference_loading_slice=slice(5, 35),
            player_trial_peaks=(95.0, 96.0, 94.5, 95.5),
        )
        assert result.measurements["consistency_mode"] == "multi_serve"
        assert result.measurements["consistency_score"] >= 80.0

    def test_low_joint_confidence_reduces_confidence(self):
        ref = _loading_series()
        player = _loading_series(seed=7)
        low_conf = np.full(len(player), 0.3)
        result = score_knee_flexion_biomechanics(
            player_series=player,
            reference_series=ref,
            loading_slice=slice(5, 35),
            reference_loading_slice=slice(5, 35),
            player_joint_confidence=low_conf,
        )
        assert result.confidence < 0.6


class TestScoreKneeFlexionPublicEntry:
    def test_scalar_fallback_still_works(self):
        result = score_knee_flexion(player_value=70.0, reference_value=95.0)
        assert result.direction == "too_low"
        assert result.measurements["scoring_mode"] == "scalar_fallback"
        assert "magnitude_score" in result.measurements

    def test_series_kwargs_use_biomechanics(self):
        ref = _loading_series(peak_flexion=95.0)
        player = _loading_series(peak_flexion=70.0, seed=8)
        result = score_knee_flexion(
            player_value=70.0,
            reference_value=95.0,
            player_series=player,
            reference_series=ref,
            player_loading_slice=slice(5, 35),
            reference_loading_slice=slice(5, 35),
        )
        assert result.measurements["scoring_mode"] == "biomechanical"
        assert result.direction == "too_low"
