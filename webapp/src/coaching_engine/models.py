"""Structured coaching outputs produced by the Coaching Engine (Version 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Priority = Literal["High", "Medium", "Low"]
Direction = Literal["too_low", "too_high"]


@dataclass(frozen=True)
class CoachingRecommendation:
    """One actionable coaching item assembled from the Knowledge Library."""

    feature: str
    phase: str
    priority: Priority
    correction: str
    coach_quotes: tuple[str, ...] = ()
    practice_drills: tuple[str, ...] = ()
    direction: Direction | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoachingReport:
    """Version 1 coaching product — structured recommendations only (no prose)."""

    overall_score: float
    primary_recommendation: CoachingRecommendation | None
    secondary_recommendations: tuple[CoachingRecommendation, ...]
    strengths: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging or downstream consumers."""

        def _rec(r: CoachingRecommendation | None) -> dict[str, Any] | None:
            if r is None:
                return None
            return {
                "feature": r.feature,
                "phase": r.phase,
                "priority": r.priority,
                "correction": r.correction,
                "coach_quotes": list(r.coach_quotes),
                "practice_drills": list(r.practice_drills),
                "direction": r.direction,
                "metadata": dict(r.metadata),
            }

        return {
            "overall_score": self.overall_score,
            "primary_recommendation": _rec(self.primary_recommendation),
            "secondary_recommendations": [_rec(r) for r in self.secondary_recommendations],
            "strengths": list(self.strengths),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }
