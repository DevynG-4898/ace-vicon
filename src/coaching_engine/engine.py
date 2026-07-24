"""Coaching Engine Version 1 — CoachingContext → CoachingReport.

Contains NO tennis knowledge. All corrections, quotes, and drills come from
the Knowledge Library via feature-name lookup.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.coaching_engine.direction import resolve_direction
from src.coaching_engine.models import (
    CoachingRecommendation,
    CoachingReport,
    Priority,
)
from src.knowledge_library import KnowledgeLibrary, KnowledgeLibraryError
from src.scoring_report_reader.context import CoachingContext, CoachingTarget

_SECONDARY_PRIORITIES: tuple[Priority, ...] = ("Medium", "Low")


class CoachingEngineError(RuntimeError):
    """Raised when the Coaching Engine cannot assemble a recommendation."""


@dataclass(frozen=True)
class CoachingEngine:
    """Generic assembler: context targets + knowledge lookup → report."""

    library: KnowledgeLibrary

    @classmethod
    def with_default_library(cls) -> CoachingEngine:
        return cls(library=KnowledgeLibrary.default())

    def generate(self, context: CoachingContext) -> CoachingReport:
        """Build a :class:`CoachingReport` from a completed :class:`CoachingContext`."""
        primary = None
        if context.primary_coaching_target is not None:
            primary = self._recommend(context.primary_coaching_target, priority="High")

        secondary: list[CoachingRecommendation] = []
        for index, target in enumerate(context.secondary_coaching_targets):
            priority = (
                _SECONDARY_PRIORITIES[index]
                if index < len(_SECONDARY_PRIORITIES)
                else "Low"
            )
            secondary.append(self._recommend(target, priority=priority))

        strengths = tuple(t.label for t in context.ordered_strengths)

        return CoachingReport(
            overall_score=float(context.overall_score),
            primary_recommendation=primary,
            secondary_recommendations=tuple(secondary),
            strengths=strengths,
            warnings=tuple(context.warnings),
            metadata={
                **dict(context.metadata),
                "coaching_engine": {
                    "version": 1,
                    "library_feature_count": len(self.library),
                },
            },
        )

    def _recommend(self, target: CoachingTarget, *, priority: Priority) -> CoachingRecommendation:
        feature_name = target.name
        try:
            entry = self.library.get(feature_name)
        except KnowledgeLibraryError as exc:
            raise CoachingEngineError(
                f"Failed to look up coaching knowledge for feature {feature_name!r}: {exc}"
            ) from exc

        direction = resolve_direction(target)
        correction = entry.correction_for(direction)

        return CoachingRecommendation(
            feature=entry.feature,
            phase=entry.phase,
            priority=priority,
            correction=correction,
            coach_quotes=tuple(entry.coach_quotes),
            practice_drills=tuple(entry.practice_drills),
            direction=direction,
            metadata={
                "target_key": target.key,
                "target_kind": target.kind,
                "score": target.score,
                "impact_score": target.impact_score,
            },
        )


def generate_coaching_report(
    context: CoachingContext,
    *,
    library: KnowledgeLibrary | None = None,
) -> CoachingReport:
    """Convenience wrapper around :class:`CoachingEngine`."""
    engine = (
        CoachingEngine.with_default_library()
        if library is None
        else CoachingEngine(library=library)
    )
    return engine.generate(context)
