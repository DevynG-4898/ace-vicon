"""Construct a realistic fake ScoringReport for coaching pipeline demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.scoring_report_reader.report import (
    Category,
    FeatureSummary,
    PhaseSummary,
    ScoringReport,
)

Direction = Literal["too_low", "too_high"]


def _tier_for_score(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Poor"


def _category_for_score(score: float) -> Category:
    if score >= 80:
        return "strength"
    if score >= 60:
        return "neutral"
    return "weakness"


def _grade_for_overall(score: float) -> str:
    if score >= 90:
        return "A — Pro-level serve"
    if score >= 75:
        return "B — Strong serve, minor adjustments needed"
    if score >= 60:
        return "C — Developing serve, focused practice recommended"
    return "D — Fundamentals need significant work"


def _priority_label(impact_score: float, category: Category) -> str:
    if category != "weakness" or impact_score <= 0:
        return "none"
    if impact_score >= 0.85:
        return "high"
    if impact_score >= 0.6:
        return "medium"
    return "low"


@dataclass(frozen=True)
class _FeatureSpec:
    name: str
    phase: str
    score: float
    impact_score: float
    direction: Direction
    reference_value: float
    player_value: float
    unit: str = "deg"
    confidence: float = 0.9


# Believable demo scenario:
#   primary weakness   → Right Elbow Flexion
#   secondary          → Contact Position, Shoulder External Rotation
#   strengths          → Balance, Follow Through, Hip Rotation Velocity, …
_FEATURE_SPECS: tuple[_FeatureSpec, ...] = (
    _FeatureSpec("Knee Flexion", "Loading", 91, 0.05, "too_high", 95.0, 108.0, confidence=0.93),
    _FeatureSpec("Hip Flexion", "Loading", 87, 0.08, "too_high", 40.0, 48.0, confidence=0.91),
    _FeatureSpec("Shoulder Tilt", "Loading", 83, 0.10, "too_low", 18.0, 14.0, confidence=0.88),
    _FeatureSpec("Center of Mass", "Loading", 80, 0.12, "too_high", 0.0, 35.0, unit="mm", confidence=0.86),
    # Major weakness — highest impact among weaknesses
    _FeatureSpec("Right Elbow Flexion", "Cocking", 42, 0.95, "too_low", 95.0, 62.0, confidence=0.94),
    _FeatureSpec("Left Elbow Flexion", "Cocking", 88, 0.06, "too_high", 90.0, 102.0, confidence=0.90),
    # Secondary weakness
    _FeatureSpec("Shoulder External Rotation", "Cocking", 54, 0.82, "too_low", 110.0, 78.0, confidence=0.89),
    _FeatureSpec("Shoulder Internal Rotation", "Acceleration", 74, 0.25, "too_low", 80.0, 68.0, confidence=0.87),
    _FeatureSpec("Right Elbow Extension", "Acceleration", 61, 0.40, "too_low", 165.0, 148.0, confidence=0.88),
    _FeatureSpec("Trunk Rotation Velocity", "Acceleration", 72, 0.30, "too_low", 420.0, 360.0, unit="deg/s", confidence=0.85),
    # Strength
    _FeatureSpec("Hip Rotation Velocity", "Acceleration", 86, 0.07, "too_high", 380.0, 410.0, unit="deg/s", confidence=0.92),
    _FeatureSpec("Contact Height", "Contact", 55, 0.55, "too_low", 2450.0, 2280.0, unit="mm", confidence=0.90),
    # Secondary weakness — second-highest impact
    _FeatureSpec("Contact Position", "Contact", 49, 0.88, "too_low", 350.0, 180.0, unit="mm", confidence=0.91),
    _FeatureSpec("Arm Extension", "Contact", 70, 0.28, "too_low", 175.0, 162.0, confidence=0.89),
    _FeatureSpec("Body Alignment", "Contact", 84, 0.09, "too_high", 5.0, 12.0, confidence=0.90),
    # Strengths
    _FeatureSpec("Follow Through", "Deceleration", 92, 0.04, "too_high", 100.0, 112.0, unit="%", confidence=0.93),
    _FeatureSpec("Balance", "Finish", 95, 0.03, "too_high", 100.0, 108.0, unit="%", confidence=0.96),
    _FeatureSpec("Weight Transfer", "Finish", 81, 0.11, "too_low", 70.0, 58.0, unit="%", confidence=0.88),
    _FeatureSpec("Recovery Position", "Finish", 89, 0.05, "too_high", 0.40, 0.55, unit="s", confidence=0.87),
)

_PHASE_SCORES: tuple[tuple[str, float], ...] = (
    ("Loading", 82.0),
    ("Cocking", 63.0),
    ("Acceleration", 75.0),
    ("Contact", 58.0),
    ("Deceleration", 80.0),
    ("Finish", 90.0),
)


@dataclass(frozen=True)
class FakeScoringReportGenerator:
    """Build a deterministic, realistic fake ScoringReport for demos and tests."""

    overall_score: float = 78.0
    serve_name: str = "fake_demo_serve"
    confidence: float = 0.87

    def generate(self) -> ScoringReport:
        """Return a complete ScoringReport accepted by ScoringReportReader."""
        phase_summaries = self._build_phases()
        feature_summaries = self._build_features()
        return ScoringReport(
            overall_score=float(self.overall_score),
            overall_grade=_grade_for_overall(self.overall_score),
            phase_summaries=phase_summaries,
            feature_summaries=feature_summaries,
            warnings=(
                "Fake ScoringReport — simulated for coaching pipeline demo.",
                "Contact height confidence slightly reduced due to toss variability.",
            ),
            confidence=float(self.confidence),
            metadata={
                "source": "fake_scoring_report",
                "serve_name": self.serve_name,
                "scenario": "right_elbow_flexion_primary",
                "generator_version": 1,
            },
            schema_version=1,
        )

    def _build_phases(self) -> tuple[PhaseSummary, ...]:
        # Phase impact is kept at 0 so coaching targets stay feature-level
        # (phases are not Knowledge Library keys).
        phases: list[PhaseSummary] = []
        for name, score in _PHASE_SCORES:
            tier = _tier_for_score(score)
            category = _category_for_score(score)
            phases.append(
                PhaseSummary(
                    name=name,
                    score=float(score),
                    impact_score=0.0,
                    tier=tier,
                    category=category,
                    metadata={
                        "priority": "none",
                        "confidence": 0.9,
                        "note": "Phase rollup for display; coach features instead.",
                    },
                )
            )
        return tuple(phases)

    def _build_features(self) -> tuple[FeatureSummary, ...]:
        features: list[FeatureSummary] = []
        for spec in _FEATURE_SPECS:
            category = _category_for_score(spec.score)
            # Force known weakness / strength labels for the demo scenario.
            if spec.name == "Right Elbow Flexion":
                category = "weakness"
            elif spec.name in ("Contact Position", "Shoulder External Rotation", "Contact Height"):
                category = "weakness"
            elif spec.name in ("Balance", "Follow Through", "Hip Rotation Velocity"):
                category = "strength"

            tier = _tier_for_score(spec.score)
            difference = round(spec.player_value - spec.reference_value, 2)
            metadata: dict[str, Any] = {
                "feature_name": spec.name,
                "score": spec.score,
                "reference_value": spec.reference_value,
                "player_value": spec.player_value,
                "difference": difference,
                "unit": spec.unit,
                "impact_score": spec.impact_score,
                "priority": _priority_label(spec.impact_score, category),
                "category": category,
                "direction": spec.direction,
                "confidence": spec.confidence,
            }
            features.append(
                FeatureSummary(
                    name=spec.name,
                    score=float(spec.score),
                    impact_score=float(spec.impact_score),
                    tier=tier,
                    category=category,
                    phase=spec.phase,
                    metadata=metadata,
                )
            )
        return tuple(features)


def generate_fake_scoring_report(
    *,
    overall_score: float = 78.0,
    serve_name: str = "fake_demo_serve",
) -> ScoringReport:
    """Convenience helper that returns the standard demo ScoringReport."""
    return FakeScoringReportGenerator(
        overall_score=overall_score,
        serve_name=serve_name,
    ).generate()
