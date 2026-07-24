"""Tests for Scoring Engine → ScoringReport → Reader → CoachingEngine."""

from __future__ import annotations

import pytest

from src.coaching_engine import CoachingEngine
from src.knowledge_library import KnowledgeLibrary
from src.scoring_engine import ScoringEngine
from src.scoring_report_reader import ScoringReportReader, validate_scoring_report


# Minimal player/reference scalars covering every Knowledge Library feature.
# Deliberately make Right Elbow Flexion the clear primary weakness.
REFERENCE: dict[str, float] = {
    "Knee Flexion": 95.0,
    "Hip Flexion": 40.0,
    "Shoulder Tilt": 18.0,
    "Toss Arm Extension": 170.0,
    "Center of Mass": 0.0,
    "Trunk Rotation": 25.0,
    "Pelvis Rotation": 20.0,
    "Right Elbow Flexion": 95.0,
    "Left Elbow Flexion": 90.0,
    "Shoulder External Rotation": 110.0,
    "Forearm Angle": 45.0,
    "Shoulder Internal Rotation": 80.0,
    "Right Elbow Extension": 165.0,
    "Left Elbow Extension": 160.0,
    "Trunk Rotation Velocity": 420.0,
    "Hip Rotation Velocity": 380.0,
    "Contact Height": 2450.0,
    "Contact Position": 350.0,
    "Arm Extension": 175.0,
    "Body Alignment": 5.0,
    "Follow Through": 100.0,
    "Shoulder Deceleration": 50.0,
    "Trunk Flexion": 30.0,
    "Balance": 100.0,
    "Weight Transfer": 70.0,
    "Recovery Position": 0.40,
}

PLAYER: dict[str, float] = {
    **REFERENCE,
    # Worst angle-kernel scalar miss → primary weakness
    "Right Elbow Flexion": 40.0,
    # Placeholder secondary weakness — tuned so Contact ranks between elbow and SER
    # under Contact Kernel scalar fallback (mag 0.70 + consistency 0.30).
    "Contact Position": 263.0,
    # Angle-kernel secondary/tertiary weakness
    "Shoulder External Rotation": 75.0,
    # Mild deviations
    "Contact Height": 2300.0,
    "Knee Flexion": 90.0,
    # Strengths (near reference)
    "Balance": 102.0,
    "Follow Through": 103.0,
    "Hip Rotation Velocity": 390.0,
}

_LIBRARY = KnowledgeLibrary.default()
EXPECTED_FEATURES = set(_LIBRARY.features)
EXPECTED_PHASES = {_LIBRARY.get(name).phase for name in _LIBRARY.features}


class TestScoringEngineReportShape:
    def test_produces_valid_scoring_report(self):
        report = ScoringEngine().score(PLAYER, REFERENCE)
        validate_scoring_report(report)
        assert report.schema_version == 1
        assert 0.0 <= report.overall_score <= 100.0
        assert isinstance(report.overall_grade, str) and report.overall_grade

    def test_scores_every_knowledge_library_feature(self):
        report = ScoringEngine().score(PLAYER, REFERENCE)
        names = {f.name for f in report.feature_summaries}
        assert names == EXPECTED_FEATURES

    def test_phases_match_knowledge_library(self):
        report = ScoringEngine().score(PLAYER, REFERENCE)
        phase_names = {p.name for p in report.phase_summaries}
        assert phase_names == EXPECTED_PHASES

    def test_feature_metadata_includes_direction_and_measurements(self):
        report = ScoringEngine().score(PLAYER, REFERENCE)
        elbow = next(f for f in report.feature_summaries if f.name == "Right Elbow Flexion")
        assert elbow.metadata["direction"] == "too_low"
        assert elbow.metadata["player_value"] == 40.0
        assert elbow.metadata["reference_value"] == 95.0
        assert elbow.metadata["difference"] == pytest.approx(-55.0)
        assert elbow.phase == "Cocking"
        assert elbow.category == "weakness"
        assert elbow.score < 60.0
        assert elbow.impact_score > 0.5

    def test_phase_impact_is_zero_so_features_are_coached(self):
        report = ScoringEngine().score(PLAYER, REFERENCE)
        assert all(p.impact_score == 0.0 for p in report.phase_summaries)

    def test_missing_measurement_raises(self):
        incomplete = dict(PLAYER)
        del incomplete["Right Elbow Flexion"]
        with pytest.raises(ValueError, match="Right Elbow Flexion"):
            ScoringEngine().score(incomplete, REFERENCE)


class TestScoringEnginePipeline:
    def test_reader_and_coaching_engine_consume_report(self):
        report = ScoringEngine().score(PLAYER, REFERENCE)
        context = ScoringReportReader(max_secondary=2).read(report)
        coaching = CoachingEngine.with_default_library().generate(context)

        assert context.primary_coaching_target is not None
        assert context.primary_coaching_target.name == "Right Elbow Flexion"
        assert coaching.primary_recommendation is not None
        assert coaching.primary_recommendation.feature == "Right Elbow Flexion"
        assert coaching.primary_recommendation.priority == "High"
        assert (
            coaching.primary_recommendation.correction
            == "Bend your right elbow more during the cocking phase."
        )
        assert len(coaching.secondary_recommendations) == 2
        secondary_names = [r.feature for r in coaching.secondary_recommendations]
        assert "Contact Position" in secondary_names
        assert "Shoulder External Rotation" in secondary_names


class TestFeatureScorers:
    def test_knee_flexion_too_low(self):
        from src.scoring_engine.feature_scorer import score_knee_flexion

        result = score_knee_flexion(player_value=70.0, reference_value=95.0)
        assert result.name == "Knee Flexion"
        assert result.phase == "Loading"
        assert result.direction == "too_low"
        assert 0.0 <= result.score <= 100.0

    def test_near_reference_is_acceptable(self):
        from src.scoring_engine.feature_scorer import score_balance

        result = score_balance(player_value=100.5, reference_value=100.0)
        assert result.direction == "acceptable"
        assert result.score >= 90.0
