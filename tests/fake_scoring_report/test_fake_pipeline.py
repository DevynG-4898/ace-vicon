"""Tests for the fake scoring report → coaching pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.coaching_engine import CoachingEngine, CoachingReport
from src.fake_scoring_report import FakeScoringReportGenerator, generate_fake_scoring_report
from src.scoring_report_reader import ScoringReportReader, validate_scoring_report

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_SCRIPT = REPO_ROOT / "demo_fake_coaching_pipeline.py"


class TestFakeScoringReport:
    def test_fake_report_is_valid(self):
        report = generate_fake_scoring_report()
        validate_scoring_report(report)
        assert report.overall_score == 78.0
        assert report.schema_version == 1
        assert len(report.phase_summaries) == 6
        assert len(report.feature_summaries) >= 15

    def test_phase_scores_match_scenario(self):
        report = generate_fake_scoring_report()
        by_name = {p.name: p.score for p in report.phase_summaries}
        assert by_name == {
            "Loading": 82.0,
            "Cocking": 63.0,
            "Acceleration": 75.0,
            "Contact": 58.0,
            "Deceleration": 80.0,
            "Finish": 90.0,
        }

    def test_features_include_direction_metadata(self):
        report = generate_fake_scoring_report()
        elbow = next(f for f in report.feature_summaries if f.name == "Right Elbow Flexion")
        assert elbow.metadata["direction"] == "too_low"
        assert elbow.metadata["player_value"] < elbow.metadata["reference_value"]
        assert "impact_score" in elbow.metadata
        assert "confidence" in elbow.metadata
        assert elbow.category == "weakness"

    def test_reader_accepts_fake_report(self):
        report = generate_fake_scoring_report()
        context = ScoringReportReader(max_secondary=2).read(report)
        assert context.overall_score == 78.0
        assert context.primary_coaching_target is not None
        assert context.primary_coaching_target.name == "Right Elbow Flexion"
        assert [t.name for t in context.secondary_coaching_targets] == [
            "Contact Position",
            "Shoulder External Rotation",
        ]
        strength_names = {t.name for t in context.ordered_strengths}
        assert "Balance" in strength_names
        assert "Follow Through" in strength_names
        assert "Hip Rotation Velocity" in strength_names

    def test_coaching_engine_produces_valid_report(self):
        report = generate_fake_scoring_report()
        context = ScoringReportReader(max_secondary=2).read(report)
        coaching = CoachingEngine.with_default_library().generate(context)

        assert isinstance(coaching, CoachingReport)
        assert coaching.overall_score == 78.0
        assert coaching.primary_recommendation is not None
        assert coaching.primary_recommendation.feature == "Right Elbow Flexion"
        assert coaching.primary_recommendation.priority == "High"
        assert (
            coaching.primary_recommendation.correction
            == "Bend your right elbow more during the cocking phase."
        )
        assert "Let the elbow fold naturally." in coaching.primary_recommendation.coach_quotes
        assert len(coaching.secondary_recommendations) == 2
        assert coaching.secondary_recommendations[0].feature == "Contact Position"
        assert coaching.secondary_recommendations[1].feature == "Shoulder External Rotation"
        assert any("Balance" in s for s in coaching.strengths)

    def test_generator_is_deterministic(self):
        a = FakeScoringReportGenerator().generate()
        b = FakeScoringReportGenerator().generate()
        assert a.overall_score == b.overall_score
        assert [f.name for f in a.feature_summaries] == [f.name for f in b.feature_summaries]
        assert [f.score for f in a.feature_summaries] == [f.score for f in b.feature_summaries]

    def test_demo_runs_without_errors(self):
        result = subprocess.run(
            [sys.executable, str(DEMO_SCRIPT)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "ACE COACHING REPORT" in result.stdout
        assert "Overall Score" in result.stdout
        assert "78" in result.stdout
        assert "Primary Focus" in result.stdout
        assert "Right Elbow Flexion" in result.stdout
        assert "Bend your right elbow more during the cocking phase." in result.stdout
        assert "Secondary Focus" in result.stdout
        assert "Contact Position" in result.stdout
        assert "Strengths" in result.stdout
        assert "Balance" in result.stdout
