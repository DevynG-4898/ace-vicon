"""Tests for FeatureScoreBuilder weighted component framework."""

from __future__ import annotations

import pytest

from src.scoring_engine.feature_scorer import score_knee_flexion
from src.scoring_engine.score_builder import FeatureScoreBuilder


class TestFeatureScoreBuilder:
    def test_weighted_average_of_components(self):
        result = (
            FeatureScoreBuilder("Knee Flexion", "Loading")
            .add_component("magnitude", score=80.0, weight=0.50)
            .add_component("timing", score=60.0, weight=0.20)
            .add_component("smoothness", score=100.0, weight=0.15)
            .add_component("consistency", score=100.0, weight=0.15)
            .set_direction("too_low")
            .set_values(player_value=70.0, reference_value=95.0)
            .build()
        )
        # 0.5*80 + 0.2*60 + 0.15*100 + 0.15*100 = 40 + 12 + 15 + 15 = 82
        assert result.score == pytest.approx(82.0)
        assert result.direction == "too_low"
        assert result.confidence == 1.0
        assert result.measurements["magnitude_score"] == 80.0
        assert result.measurements["timing_score"] == 60.0
        assert result.measurements["smoothness_score"] == 100.0
        assert result.measurements["consistency_score"] == 100.0
        assert result.measurements["final_score"] == pytest.approx(82.0)

    def test_renormalizes_when_only_some_components_present(self):
        result = (
            FeatureScoreBuilder("Contact Height", "Contact", unit="mm")
            .add_component("magnitude", score=50.0, weight=0.50)
            .set_direction("too_low")
            .set_values(player_value=2000.0, reference_value=2450.0)
            .build()
        )
        assert result.score == pytest.approx(50.0)

    def test_rejects_empty_components(self):
        with pytest.raises(ValueError, match="component"):
            FeatureScoreBuilder("Balance", "Finish").build()

    def test_custom_confidence(self):
        result = (
            FeatureScoreBuilder("Balance", "Finish", confidence=0.75)
            .add_component("magnitude", score=90.0, weight=1.0)
            .set_direction("acceptable")
            .set_values(player_value=100.0, reference_value=100.0)
            .build()
        )
        assert result.confidence == 0.75
        assert result.impact_score == 0.0


class TestFeatureScorersUseBuilder:
    def test_knee_flexion_exposes_component_measurements(self):
        result = score_knee_flexion(player_value=70.0, reference_value=95.0)
        assert "magnitude_score" in result.measurements
        assert "timing_score" in result.measurements
        assert "smoothness_score" in result.measurements
        assert "consistency_score" in result.measurements
        assert "final_score" in result.measurements
        assert result.confidence == 0.55
        assert result.measurements["player_value"] == 70.0
        assert result.measurements["reference_value"] == 95.0
        assert result.measurements["difference"] == pytest.approx(-25.0)
        assert result.measurements["scoring_mode"] == "scalar_fallback"
