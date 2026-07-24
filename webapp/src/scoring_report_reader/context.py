"""Coaching-oriented output of the Scoring Report Reader."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.scoring_report_reader.report import (
    Category,
    FeatureSummary,
    ItemKind,
    PhaseSummary,
)


@dataclass(frozen=True)
class CoachingTarget:
    """One coachable (or noted) item extracted from a ScoringReport."""

    key: str
    name: str
    kind: ItemKind
    score: float
    impact_score: float
    category: Category
    tier: str | None = None
    phase: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Human-facing short label."""
        if self.kind == "phase":
            return self.name.replace("_", " ")
        if self.phase:
            return f"{self.name} ({self.phase.replace('_', ' ')})"
        return self.name


@dataclass(frozen=True)
class CoachingContext:
    """Everything the Coaching Engine needs — optimized for coaching, not scoring.

    The Coaching Engine should consume this object and never parse a raw
    :class:`~src.scoring_report_reader.report.ScoringReport` directly.
    """

    overall_score: float
    overall_grade: str
    primary_coaching_target: CoachingTarget | None
    secondary_coaching_targets: tuple[CoachingTarget, ...]
    weakest_phase: str | None
    strongest_phase: str | None
    highest_impact_feature: str | None
    strongest_feature: str | None
    ordered_coaching_priorities: tuple[CoachingTarget, ...]
    ordered_strengths: tuple[CoachingTarget, ...]
    ordered_weaknesses: tuple[CoachingTarget, ...]
    not_worth_coaching: tuple[CoachingTarget, ...]
    feature_summaries: tuple[FeatureSummary, ...]
    phase_summaries: tuple[PhaseSummary, ...]
    warnings: tuple[str, ...]
    confidence: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging or handoff without coaching-engine coupling."""

        def _target(t: CoachingTarget | None) -> dict[str, Any] | None:
            if t is None:
                return None
            return {
                "key": t.key,
                "name": t.name,
                "kind": t.kind,
                "score": t.score,
                "impact_score": t.impact_score,
                "category": t.category,
                "tier": t.tier,
                "phase": t.phase,
                "label": t.label,
                "metadata": dict(t.metadata),
            }

        return {
            "overall_score": self.overall_score,
            "overall_grade": self.overall_grade,
            "primary_coaching_target": _target(self.primary_coaching_target),
            "secondary_coaching_targets": [_target(t) for t in self.secondary_coaching_targets],
            "weakest_phase": self.weakest_phase,
            "strongest_phase": self.strongest_phase,
            "highest_impact_feature": self.highest_impact_feature,
            "strongest_feature": self.strongest_feature,
            "ordered_coaching_priorities": [_target(t) for t in self.ordered_coaching_priorities],
            "ordered_strengths": [_target(t) for t in self.ordered_strengths],
            "ordered_weaknesses": [_target(t) for t in self.ordered_weaknesses],
            "not_worth_coaching": [_target(t) for t in self.not_worth_coaching],
            "warnings": list(self.warnings),
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }
