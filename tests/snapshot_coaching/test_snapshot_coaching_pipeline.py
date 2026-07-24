"""Snapshot Comparison Scoring Engine → ScoringReport → Coaching Engine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from format.data import ScoringReport, scoring_report_from_snapshot_grade
from format.pipeline import coaching_report_from_scoring_report, run_from_scoring_report
from format.render import CoachingReport
from src.coaching_viewer.pipeline import build_session_from_parts
from src.knowledge_library import KnowledgeLibrary
from src.scoring_report_reader import ScoringReportReader, validate_scoring_report


def _joint(customer: float, reference: float, score: int, tier: str) -> dict:
    return {
        "customer_angle": customer,
        "reference_angle": reference,
        "diff": round(abs(customer - reference), 1),
        "score": score,
        "tier": tier,
    }


def _snapshot_grade_with_weak_elbow() -> dict:
    """Synthetic grade_snapshots.grade_serve output with a clear primary weakness."""
    good = _joint(100.0, 100.0, 95, "Excellent")
    weak_elbow = _joint(40.0, 110.0, 35, "Poor")
    mild_shoulder = _joint(80.0, 110.0, 60, "Fair")

    joints_peak = {
        "racket_elbow": weak_elbow,
        "racket_shoulder": mild_shoulder,
        "toss_elbow": good,
        "left_knee": good,
        "right_knee": good,
    }
    joints_ok = {
        "racket_elbow": good,
        "racket_shoulder": good,
        "toss_elbow": good,
        "left_knee": good,
        "right_knee": good,
    }

    return {
        "customer_racket_side": "right",
        "reference_racket_side": "right",
        "overall_score": 72.5,
        "overall_grade": "C -- Developing serve, focused practice recommended",
        "snapshots": {
            "start_pose": {
                "customer_frame": 1,
                "reference_frame": 1,
                "joints": joints_ok,
                "snapshot_score": 95.0,
            },
            "hand_cross": {
                "customer_frame": 10,
                "reference_frame": 10,
                "joints": joints_ok,
                "snapshot_score": 95.0,
            },
            "flat_racket_arm": {
                "customer_frame": 20,
                "reference_frame": 20,
                "joints": joints_ok,
                "snapshot_score": 95.0,
            },
            "peak_racket_arm": {
                "customer_frame": 40,
                "reference_frame": 40,
                "joints": joints_peak,
                "snapshot_score": 68.0,
            },
            "contact": {
                "customer_frame": 60,
                "reference_frame": 60,
                "joints": joints_ok,
                "snapshot_score": 95.0,
            },
            "hand_cross_2": {
                "customer_frame": 70,
                "reference_frame": 70,
                "joints": joints_ok,
                "snapshot_score": 95.0,
            },
            "racket_deceleration": {
                "customer_frame": 80,
                "reference_frame": 80,
                "joints": joints_ok,
                "snapshot_score": 95.0,
            },
            "finish_pose": {
                "customer_frame": 90,
                "reference_frame": 90,
                "joints": joints_ok,
                "snapshot_score": 95.0,
            },
        },
    }


class TestSnapshotToScoringReport:
    def test_produces_valid_canonical_scoring_report(self):
        report = scoring_report_from_snapshot_grade(_snapshot_grade_with_weak_elbow())
        validate_scoring_report(report)
        assert isinstance(report, ScoringReport)
        assert report.schema_version == 1
        assert report.overall_score == pytest.approx(72.5)
        assert "Developing" in report.overall_grade
        assert report.metadata["source"] == "snapshot_comparison_scoring_engine"

    def test_preserves_feature_scores_and_direction(self):
        report = scoring_report_from_snapshot_grade(_snapshot_grade_with_weak_elbow())
        elbow = next(f for f in report.feature_summaries if f.name == "Right Elbow Flexion")
        assert elbow.score == 35.0
        assert elbow.tier == "Poor"
        assert elbow.category == "weakness"
        assert elbow.metadata["direction"] == "too_low"
        assert elbow.metadata["player_value"] == 40.0
        assert elbow.metadata["reference_value"] == 110.0
        assert elbow.impact_score > 0.5
        assert "measurements" in elbow.metadata

    def test_phase_scores_present_with_zero_impact(self):
        report = scoring_report_from_snapshot_grade(_snapshot_grade_with_weak_elbow())
        assert report.phase_summaries
        assert all(p.impact_score == 0.0 for p in report.phase_summaries)
        phase_names = {p.name for p in report.phase_summaries}
        assert "Cocking" in phase_names

    def test_feature_names_are_knowledge_library_keys(self):
        library = KnowledgeLibrary.default()
        report = scoring_report_from_snapshot_grade(_snapshot_grade_with_weak_elbow())
        for feature in report.feature_summaries:
            assert feature.name in library


class TestSnapshotCoachingPipeline:
    def test_coaching_engine_consumes_snapshot_scoring_report(self):
        scoring = scoring_report_from_snapshot_grade(_snapshot_grade_with_weak_elbow())
        coaching = coaching_report_from_scoring_report(scoring)

        assert isinstance(coaching, CoachingReport)
        assert coaching.overall_score == pytest.approx(scoring.overall_score)
        assert coaching.primary_recommendation is not None
        assert coaching.primary_recommendation.feature == "Right Elbow Flexion"
        assert coaching.primary_recommendation.priority == "High"
        assert (
            coaching.primary_recommendation.correction
            == "Bend your right elbow more during the cocking phase."
        )
        assert coaching.primary_recommendation.coach_quotes
        assert coaching.primary_recommendation.practice_drills

    def test_reader_priorities_match_impact(self):
        scoring = scoring_report_from_snapshot_grade(_snapshot_grade_with_weak_elbow())
        context = ScoringReportReader(max_secondary=2).read(scoring)
        assert context.primary_coaching_target is not None
        assert context.primary_coaching_target.name == "Right Elbow Flexion"
        assert context.overall_score == pytest.approx(72.5)

    def test_pipeline_helper_preserves_scores(self):
        scoring = scoring_report_from_snapshot_grade(_snapshot_grade_with_weak_elbow())
        result = run_from_scoring_report(scoring)
        assert result.scoring_report.overall_score == pytest.approx(72.5)
        assert result.coaching_report.overall_score == pytest.approx(72.5)

    def test_coaching_viewer_consumes_coaching_report_unchanged(self):
        scoring = scoring_report_from_snapshot_grade(_snapshot_grade_with_weak_elbow())
        coaching = coaching_report_from_scoring_report(scoring)
        session = build_session_from_parts(
            markers_player={"frames": []},
            markers_reference={"frames": []},
            comparison=MagicMock(),
            scoring_report=scoring,
            coaching_report=coaching,
        )

        assert session.coaching_report is coaching
        assert session.scoring_report is scoring
        assert session.recommendations[0].feature == "Right Elbow Flexion"
        assert session.recommendations[0].coach_quotes == coaching.primary_recommendation.coach_quotes
        assert session.recommendations[0].practice_drills == coaching.primary_recommendation.practice_drills
        assert session.recommendations[0].correction == coaching.primary_recommendation.correction
