"""Unit tests for Coaching Engine Version 1."""

from __future__ import annotations

import pytest

from src.coaching_engine import (
    CoachingDirectionError,
    CoachingEngine,
    CoachingEngineError,
    CoachingRecommendation,
    CoachingReport,
    generate_coaching_report,
    resolve_direction,
)
from src.knowledge_library import KnowledgeLibrary
from src.scoring_report_reader.context import CoachingContext, CoachingTarget


def _target(
    name: str,
    *,
    direction: str = "too_low",
    kind: str = "feature",
    phase: str | None = "Cocking",
    score: float = 40.0,
    impact: float = 0.9,
    key: str | None = None,
) -> CoachingTarget:
    return CoachingTarget(
        key=key or f"feature:{name}",
        name=name,
        kind=kind,  # type: ignore[arg-type]
        score=score,
        impact_score=impact,
        category="weakness",
        phase=phase,
        metadata={"direction": direction},
    )


def _context(
    primary: CoachingTarget | None,
    secondary: tuple[CoachingTarget, ...] = (),
    strengths: tuple[CoachingTarget, ...] = (),
) -> CoachingContext:
    return CoachingContext(
        overall_score=68.0,
        overall_grade="C",
        primary_coaching_target=primary,
        secondary_coaching_targets=secondary,
        weakest_phase="Cocking",
        strongest_phase="Contact",
        highest_impact_feature=primary.name if primary else None,
        strongest_feature=None,
        ordered_coaching_priorities=((primary,) if primary else ()) + secondary,
        ordered_strengths=strengths,
        ordered_weaknesses=((primary,) if primary else ()) + secondary,
        not_worth_coaching=(),
        feature_summaries=(),
        phase_summaries=(),
        warnings=("low confidence on contact",),
        confidence=0.7,
        metadata={"serve_name": "demo"},
    )


class TestDirection:
    def test_resolve_too_low(self):
        assert resolve_direction(_target("Right Elbow Flexion", direction="too_low")) == "too_low"

    def test_resolve_too_high(self):
        assert resolve_direction(_target("Right Elbow Flexion", direction="too_high")) == "too_high"

    def test_missing_direction_errors(self):
        target = CoachingTarget(
            key="feature:x",
            name="Right Elbow Flexion",
            kind="feature",
            score=40.0,
            impact_score=0.9,
            category="weakness",
            metadata={},
        )
        with pytest.raises(CoachingDirectionError, match="missing direction"):
            resolve_direction(target)


class TestRecommendationAssembly:
    def test_too_low_returns_correct_correction(self):
        engine = CoachingEngine.with_default_library()
        report = engine.generate(
            _context(_target("Right Elbow Flexion", direction="too_low"))
        )
        assert report.primary_recommendation is not None
        assert (
            report.primary_recommendation.correction
            == "Bend your right elbow more during the cocking phase."
        )
        assert report.primary_recommendation.direction == "too_low"

    def test_too_high_returns_correct_correction(self):
        engine = CoachingEngine.with_default_library()
        report = engine.generate(
            _context(_target("Right Elbow Flexion", direction="too_high"))
        )
        assert report.primary_recommendation is not None
        assert (
            report.primary_recommendation.correction
            == "Reduce right elbow bend during the cocking phase."
        )

    def test_coach_quotes_returned_correctly(self):
        report = generate_coaching_report(
            _context(_target("Right Elbow Flexion", direction="too_low"))
        )
        assert report.primary_recommendation is not None
        assert report.primary_recommendation.coach_quotes == (
            "Let the elbow fold naturally.",
            "Don't keep the arm too straight.",
            "Relax into the trophy position.",
        )

    def test_practice_drills_returned_correctly(self):
        report = generate_coaching_report(
            _context(_target("Right Elbow Flexion", direction="too_low"))
        )
        assert report.primary_recommendation is not None
        assert report.primary_recommendation.practice_drills == (
            "Serve without a ball.",
            "Start directly from the trophy position.",
            "Hold the trophy position before accelerating.",
        )

    def test_recommendation_fields_assembled(self):
        report = generate_coaching_report(
            _context(_target("Right Elbow Flexion", direction="too_low"))
        )
        rec = report.primary_recommendation
        assert isinstance(rec, CoachingRecommendation)
        assert rec.feature == "Right Elbow Flexion"
        assert rec.phase == "Cocking"
        assert rec.priority == "High"

    def test_missing_feature_produces_clear_error(self):
        engine = CoachingEngine.with_default_library()
        with pytest.raises(CoachingEngineError, match="Failed to look up"):
            engine.generate(_context(_target("Unknown Feature", direction="too_low")))


class TestReportAssembly:
    def test_report_assembled_with_secondary(self):
        primary = _target("Right Elbow Flexion", direction="too_low", impact=0.9)
        secondary = (
            _target(
                "Shoulder External Rotation",
                direction="too_low",
                impact=0.7,
                phase="Cocking",
            ),
            _target(
                "Contact Height",
                direction="too_high",
                impact=0.5,
                phase="Contact",
            ),
        )
        strengths = (
            CoachingTarget(
                key="phase:Contact",
                name="Contact",
                kind="phase",
                score=95.0,
                impact_score=0.1,
                category="strength",
                phase="Contact",
            ),
        )
        report = generate_coaching_report(_context(primary, secondary, strengths))

        assert isinstance(report, CoachingReport)
        assert report.overall_score == 68.0
        assert report.primary_recommendation is not None
        assert report.primary_recommendation.feature == "Right Elbow Flexion"
        assert report.primary_recommendation.priority == "High"

        assert len(report.secondary_recommendations) == 2
        assert report.secondary_recommendations[0].feature == "Shoulder External Rotation"
        assert report.secondary_recommendations[0].priority == "Medium"
        assert report.secondary_recommendations[0].correction.startswith(
            "Increase shoulder external rotation"
        )
        assert report.secondary_recommendations[1].feature == "Contact Height"
        assert report.secondary_recommendations[1].priority == "Low"
        assert report.secondary_recommendations[1].direction == "too_high"

        assert report.strengths == ("Contact",)
        assert report.warnings == ("low confidence on contact",)
        assert report.metadata["serve_name"] == "demo"
        assert report.metadata["coaching_engine"]["version"] == 1

    def test_no_primary_yields_none_recommendation(self):
        report = generate_coaching_report(_context(None))
        assert report.primary_recommendation is None
        assert report.secondary_recommendations == ()

    def test_to_dict(self):
        report = generate_coaching_report(
            _context(_target("Right Elbow Flexion", direction="too_low"))
        )
        payload = report.to_dict()
        assert payload["primary_recommendation"]["feature"] == "Right Elbow Flexion"
        assert payload["primary_recommendation"]["priority"] == "High"

    def test_engine_contains_no_hardcoded_correction_strings(self):
        """Sanity: corrections come from the library entry, not engine constants."""
        library = KnowledgeLibrary(
            entries=(
                __import__("src.knowledge_library.models", fromlist=["KnowledgeEntry"]).KnowledgeEntry(
                    feature="Right Elbow Flexion",
                    phase="Cocking",
                    too_low="CUSTOM LOW",
                    too_high="CUSTOM HIGH",
                    coach_quotes=("Q",),
                    practice_drills=("D",),
                ),
            )
        )
        engine = CoachingEngine(library=library)
        report = engine.generate(_context(_target("Right Elbow Flexion", direction="too_low")))
        assert report.primary_recommendation is not None
        assert report.primary_recommendation.correction == "CUSTOM LOW"
        assert report.primary_recommendation.coach_quotes == ("Q",)
        assert report.primary_recommendation.practice_drills == ("D",)
