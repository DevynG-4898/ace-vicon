"""Scoring Report Reader — interpret a completed ScoringReport for coaching.

Pipeline (read-only):

    ScoringReport
      → validate integrity
      → extract overall / phase / feature summaries
      → rank priorities from engine ``impact_score``
      → identify primary / secondary targets and strengths
      → CoachingContext

This module never runs DTW, scoring, feature extraction, or biomechanical math.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.scoring_report_reader.context import CoachingContext
from src.scoring_report_reader.priorities import (
    extract_targets,
    highest_impact_feature_name,
    not_worth_coaching,
    ordered_coaching_priorities,
    ordered_strengths,
    ordered_weaknesses,
    select_primary_and_secondary,
    strongest_feature_name,
    strongest_phase_name,
    weakest_phase_name,
)
from src.scoring_report_reader.report import ScoringReport
from src.scoring_report_reader.validation import validate_scoring_report


@dataclass(frozen=True)
class ScoringReportReader:
    """Read-only interpreter between Scoring Engine and Coaching Engine.

    Parameters
    ----------
    max_secondary:
        Maximum number of secondary coaching targets to include after the
        primary target. Selection count only — does not alter impact scores.
    """

    max_secondary: int = 2

    def __post_init__(self) -> None:
        if self.max_secondary < 0:
            raise ValueError("max_secondary must be >= 0")

    def read(self, report: ScoringReport) -> CoachingContext:
        """Validate ``report`` and produce a :class:`CoachingContext`."""
        validated = validate_scoring_report(report)
        return self._build_context(validated)

    def _build_context(self, report: ScoringReport) -> CoachingContext:
        targets = extract_targets(report)
        priorities = ordered_coaching_priorities(targets)
        weaknesses = ordered_weaknesses(targets)
        strengths = ordered_strengths(targets)
        skipped = not_worth_coaching(targets)
        primary, secondary = select_primary_and_secondary(
            weaknesses=weaknesses,
            priorities=priorities,
            max_secondary=self.max_secondary,
        )

        metadata = {
            **dict(report.metadata),
            "reader": {
                "max_secondary": self.max_secondary,
                "schema_version": report.schema_version,
                "target_count": len(targets),
                "priority_count": len(priorities),
                "weakness_count": len(weaknesses),
                "strength_count": len(strengths),
            },
        }

        return CoachingContext(
            overall_score=float(report.overall_score),
            overall_grade=str(report.overall_grade),
            primary_coaching_target=primary,
            secondary_coaching_targets=secondary,
            weakest_phase=weakest_phase_name(report.phase_summaries),
            strongest_phase=strongest_phase_name(report.phase_summaries),
            highest_impact_feature=highest_impact_feature_name(report.feature_summaries),
            strongest_feature=strongest_feature_name(report.feature_summaries),
            ordered_coaching_priorities=priorities,
            ordered_strengths=strengths,
            ordered_weaknesses=weaknesses,
            not_worth_coaching=skipped,
            feature_summaries=report.feature_summaries,
            phase_summaries=report.phase_summaries,
            warnings=tuple(report.warnings),
            confidence=None if report.confidence is None else float(report.confidence),
            metadata=metadata,
        )


def read_scoring_report(
    report: ScoringReport,
    *,
    max_secondary: int = 2,
) -> CoachingContext:
    """Convenience wrapper around :class:`ScoringReportReader`."""
    return ScoringReportReader(max_secondary=max_secondary).read(report)
