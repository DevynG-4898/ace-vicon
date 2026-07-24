"""Unit tests for the Scoring Report Reader (no scoring / DTW / MotionBERT)."""

from __future__ import annotations

import pytest

from src.scoring_report_reader import (
    CoachingContext,
    FeatureSummary,
    PhaseSummary,
    ScoringReport,
    ScoringReportReader,
    ScoringReportValidationError,
    read_scoring_report,
    resolve_category,
    validate_scoring_report,
)


def _report(**overrides) -> ScoringReport:
    base = dict(
        overall_score=72.5,
        overall_grade="C — Developing serve, focused practice recommended",
        phase_summaries=(
            PhaseSummary(name="Cocking", score=35.0, impact_score=0.90, tier="Poor"),
            PhaseSummary(name="Acceleration", score=55.0, impact_score=0.70, tier="Fair"),
            PhaseSummary(name="Contact", score=95.0, impact_score=0.10, tier="Excellent"),
            PhaseSummary(name="Start_Stance", score=88.0, impact_score=0.0, tier="Good"),
        ),
        feature_summaries=(
            FeatureSummary(
                name="right_elbow_angle",
                score=40.0,
                impact_score=0.85,
                tier="Poor",
                phase="Cocking",
            ),
            FeatureSummary(
                name="right_shoulder_flexion",
                score=60.0,
                impact_score=0.55,
                tier="Fair",
                phase="Acceleration",
            ),
            FeatureSummary(
                name="trunk_tilt",
                score=92.0,
                impact_score=0.05,
                tier="Excellent",
            ),
            FeatureSummary(
                name="hip_flexion",
                score=78.0,
                impact_score=0.0,
                category="neutral",
            ),
        ),
        warnings=("marker dropout on left_foot",),
        confidence=0.81,
        metadata={"serve_name": "firstserve"},
        schema_version=1,
    )
    base.update(overrides)
    return ScoringReport(**base)


class TestResolveCategory:
    def test_explicit_category_wins(self):
        assert resolve_category(category="strength", tier="Poor") == "strength"

    def test_tier_mapping(self):
        assert resolve_category(category=None, tier="Excellent") == "strength"
        assert resolve_category(category=None, tier="Fair") == "weakness"

    def test_unknown_tier_is_neutral(self):
        assert resolve_category(category=None, tier="Mystery") == "neutral"

    def test_missing_fields_are_neutral(self):
        assert resolve_category(category=None, tier=None) == "neutral"


class TestValidation:
    def test_valid_report_passes(self):
        report = _report()
        assert validate_scoring_report(report) is report

    def test_rejects_wrong_type(self):
        with pytest.raises(ScoringReportValidationError, match="expected ScoringReport"):
            validate_scoring_report({"overall_score": 1})  # type: ignore[arg-type]

    def test_rejects_unsupported_schema(self):
        with pytest.raises(ScoringReportValidationError, match="schema_version"):
            validate_scoring_report(_report(schema_version=99))

    def test_rejects_empty_grade(self):
        with pytest.raises(ScoringReportValidationError, match="overall_grade"):
            validate_scoring_report(_report(overall_grade="  "))

    def test_rejects_nan_score(self):
        with pytest.raises(ScoringReportValidationError, match="overall_score"):
            validate_scoring_report(_report(overall_score=float("nan")))

    def test_rejects_duplicate_phases(self):
        phases = (
            PhaseSummary(name="Cocking", score=1.0, impact_score=1.0),
            PhaseSummary(name="Cocking", score=2.0, impact_score=2.0),
        )
        with pytest.raises(ScoringReportValidationError, match="duplicate phase"):
            validate_scoring_report(_report(phase_summaries=phases))

    def test_rejects_duplicate_features_same_phase(self):
        features = (
            FeatureSummary(name="elbow", score=1.0, impact_score=1.0, phase="Cocking"),
            FeatureSummary(name="elbow", score=2.0, impact_score=2.0, phase="Cocking"),
        )
        with pytest.raises(ScoringReportValidationError, match="duplicate feature"):
            validate_scoring_report(_report(feature_summaries=features))

    def test_allows_same_feature_name_in_different_phases(self):
        features = (
            FeatureSummary(name="elbow", score=1.0, impact_score=1.0, phase="Cocking"),
            FeatureSummary(name="elbow", score=2.0, impact_score=2.0, phase="Contact"),
        )
        validate_scoring_report(_report(feature_summaries=features))


class TestReaderPriorities:
    def test_primary_is_highest_impact_weakness(self):
        ctx = read_scoring_report(_report())
        assert ctx.primary_coaching_target is not None
        # Cocking phase impact 0.90 beats elbow feature 0.85 among weaknesses
        assert ctx.primary_coaching_target.name == "Cocking"
        assert ctx.primary_coaching_target.kind == "phase"

    def test_secondary_follow_impact_order(self):
        ctx = ScoringReportReader(max_secondary=2).read(_report())
        assert [t.name for t in ctx.secondary_coaching_targets] == [
            "right_elbow_angle",
            "Acceleration",
        ]

    def test_max_secondary_respected(self):
        ctx = ScoringReportReader(max_secondary=1).read(_report())
        assert len(ctx.secondary_coaching_targets) == 1
        assert ctx.secondary_coaching_targets[0].name == "right_elbow_angle"

    def test_ordered_priorities_by_impact_descending(self):
        ctx = read_scoring_report(_report())
        impacts = [t.impact_score for t in ctx.ordered_coaching_priorities]
        assert impacts == sorted(impacts, reverse=True)
        assert all(t.impact_score > 0 for t in ctx.ordered_coaching_priorities)

    def test_strengths_ordered_by_score(self):
        ctx = read_scoring_report(_report())
        names = [t.name for t in ctx.ordered_strengths]
        assert names[0] == "Contact"
        assert "trunk_tilt" in names
        scores = [t.score for t in ctx.ordered_strengths]
        assert scores == sorted(scores, reverse=True)

    def test_weaknesses_ordered_by_impact(self):
        ctx = read_scoring_report(_report())
        names = [t.name for t in ctx.ordered_weaknesses]
        assert names[0] == "Cocking"
        assert "right_elbow_angle" in names
        assert "Acceleration" in names

    def test_not_worth_coaching_includes_zero_impact_and_neutral(self):
        ctx = read_scoring_report(_report())
        keys = {t.key for t in ctx.not_worth_coaching}
        assert "phase:Start_Stance" in keys
        assert "feature:hip_flexion" in keys

    def test_phase_and_feature_extrema(self):
        ctx = read_scoring_report(_report())
        assert ctx.weakest_phase == "Cocking"
        assert ctx.strongest_phase == "Contact"
        assert ctx.highest_impact_feature == "right_elbow_angle"
        assert ctx.strongest_feature == "trunk_tilt"

    def test_overall_and_warnings_passthrough(self):
        ctx = read_scoring_report(_report())
        assert ctx.overall_score == 72.5
        assert "Developing" in ctx.overall_grade
        assert ctx.warnings == ("marker dropout on left_foot",)
        assert ctx.confidence == 0.81
        assert ctx.metadata["serve_name"] == "firstserve"
        assert ctx.metadata["reader"]["weakness_count"] == 4

    def test_fallback_to_priorities_when_no_weaknesses(self):
        report = ScoringReport(
            overall_score=95.0,
            overall_grade="A — Pro-level serve",
            phase_summaries=(
                PhaseSummary(name="Contact", score=95.0, impact_score=0.4, tier="Excellent"),
            ),
            feature_summaries=(
                FeatureSummary(name="trunk_tilt", score=92.0, impact_score=0.6, tier="Excellent"),
            ),
        )
        ctx = read_scoring_report(report)
        assert ctx.primary_coaching_target is not None
        assert ctx.primary_coaching_target.name == "trunk_tilt"
        assert ctx.ordered_weaknesses == ()

    def test_empty_summaries_yield_empty_context_targets(self):
        report = ScoringReport(
            overall_score=50.0,
            overall_grade="Incomplete",
            phase_summaries=(),
            feature_summaries=(),
        )
        ctx = read_scoring_report(report)
        assert ctx.primary_coaching_target is None
        assert ctx.secondary_coaching_targets == ()
        assert ctx.weakest_phase is None
        assert ctx.strongest_feature is None

    def test_invalid_report_raises(self):
        with pytest.raises(ScoringReportValidationError):
            read_scoring_report(_report(overall_grade=""))

    def test_deterministic_tie_break_by_key(self):
        report = ScoringReport(
            overall_score=50.0,
            overall_grade="C",
            phase_summaries=(
                PhaseSummary(name="Beta", score=40.0, impact_score=0.5, tier="Poor"),
                PhaseSummary(name="Alpha", score=40.0, impact_score=0.5, tier="Poor"),
            ),
            feature_summaries=(),
        )
        ctx = read_scoring_report(report)
        assert [t.name for t in ctx.ordered_weaknesses] == ["Alpha", "Beta"]

    def test_to_dict_is_json_friendly(self):
        ctx = read_scoring_report(_report())
        payload = ctx.to_dict()
        assert payload["primary_coaching_target"]["name"] == "Cocking"
        assert isinstance(payload["secondary_coaching_targets"], list)
        assert payload["confidence"] == 0.81

    def test_reader_rejects_negative_max_secondary(self):
        with pytest.raises(ValueError, match="max_secondary"):
            ScoringReportReader(max_secondary=-1)

    def test_context_type(self):
        assert isinstance(read_scoring_report(_report()), CoachingContext)

    def test_does_not_mutate_report_values(self):
        report = _report()
        original_impact = report.phase_summaries[0].impact_score
        read_scoring_report(report)
        assert report.phase_summaries[0].impact_score == original_impact
