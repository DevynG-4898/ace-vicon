"""Tests for the reusable Angle Kernel and Knee Flexion parity."""

from __future__ import annotations

import numpy as np
import pytest

from src.scoring_engine.angle_kernel import (
    score_angle_feature,
    score_angle_feature_from_scalars,
)
from src.scoring_engine.knee_flexion import (
    score_knee_flexion_biomechanics,
    score_knee_flexion_from_scalars,
)


def _series(
    n: int = 40,
    *,
    peak_at: int = 20,
    peak_flexion: float = 95.0,
    base: float = 40.0,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    series = base + (peak_flexion - base) * np.exp(-0.5 * ((t - peak_at) / 6.0) ** 2)
    if noise:
        series = series + rng.normal(0.0, noise, size=n)
    return series


def _assert_results_identical(a, b) -> None:
    assert a.name == b.name
    assert a.phase == b.phase
    assert a.score == b.score
    assert a.direction == b.direction
    assert a.impact_score == b.impact_score
    assert a.player_value == b.player_value
    assert a.reference_value == b.reference_value
    assert a.difference == b.difference
    assert a.unit == b.unit
    assert a.confidence == b.confidence
    assert a.measurements == b.measurements


class TestAngleKernelGeneric:
    def test_scores_arbitrary_feature_name(self):
        ref = _series()
        player = _series(peak_flexion=70.0, seed=1)
        result = score_angle_feature(
            feature_name="Hip Flexion",
            phase="Loading",
            player_series=player,
            reference_series=ref,
            player_phase_slice=slice(5, 35),
            reference_phase_slice=slice(5, 35),
            peak_label="flexion",
        )
        assert result.name == "Hip Flexion"
        assert result.phase == "Loading"
        assert result.direction == "too_low"
        assert "player_peak_flexion_deg" in result.measurements

    def test_scalar_fallback_generic(self):
        result = score_angle_feature_from_scalars(
            feature_name="Trunk Flexion",
            phase="Deceleration",
            player_value=20.0,
            reference_value=30.0,
            peak_label="flexion",
        )
        assert result.name == "Trunk Flexion"
        assert result.measurements["scoring_mode"] == "scalar_fallback"


class TestKneeFlexionParityWithKernel:
    def test_biomechanics_matches_direct_kernel_call(self):
        ref = _series()
        player = _series(peak_flexion=60.0, seed=2)
        via_knee = score_knee_flexion_biomechanics(
            player_series=player,
            reference_series=ref,
            loading_slice=slice(5, 35),
            reference_loading_slice=slice(5, 35),
        )
        via_kernel = score_angle_feature(
            feature_name="Knee Flexion",
            phase="Loading",
            player_series=player,
            reference_series=ref,
            player_phase_slice=slice(5, 35),
            reference_phase_slice=slice(5, 35),
            peak_label="flexion",
        )
        _assert_results_identical(via_knee, via_kernel)

    def test_scalar_matches_direct_kernel_call(self):
        via_knee = score_knee_flexion_from_scalars(70.0, 95.0)
        via_kernel = score_angle_feature_from_scalars(
            feature_name="Knee Flexion",
            phase="Loading",
            player_value=70.0,
            reference_value=95.0,
            peak_label="flexion",
        )
        _assert_results_identical(via_knee, via_kernel)

    def test_golden_pre_refactor_scores(self):
        """Pinned values captured from Knee Flexion before the Angle Kernel extract."""
        ref = _series()
        player = _series(seed=1)
        matched = score_knee_flexion_biomechanics(
            player_series=player,
            reference_series=ref,
            loading_slice=slice(5, 35),
            reference_loading_slice=slice(5, 35),
        )
        assert matched.score == pytest.approx(96.25)
        assert matched.direction == "acceptable"
        assert matched.impact_score == pytest.approx(0.0)
        assert matched.confidence == pytest.approx(1.0)

        shallow = score_knee_flexion_biomechanics(
            player_series=_series(peak_flexion=60.0, seed=2),
            reference_series=_series(peak_flexion=95.0),
            loading_slice=slice(5, 35),
            reference_loading_slice=slice(5, 35),
        )
        assert shallow.score == pytest.approx(48.7)
        assert shallow.direction == "too_low"
        assert shallow.impact_score == pytest.approx(0.513)

        scalar = score_knee_flexion_from_scalars(70.0, 95.0)
        assert scalar.score == pytest.approx(48.23)
        assert scalar.direction == "too_low"
        assert scalar.impact_score == pytest.approx(0.5177)
        assert scalar.confidence == pytest.approx(0.55)
