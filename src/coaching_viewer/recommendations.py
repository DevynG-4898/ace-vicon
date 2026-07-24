"""Assemble the viewer Top-N list from a CoachingReport (no scoring logic)."""

from __future__ import annotations

from src.coaching_engine.models import CoachingRecommendation, CoachingReport


def top_recommendations(
    report: CoachingReport,
    *,
    limit: int = 3,
) -> list[CoachingRecommendation]:
    """Return up to ``limit`` recommendations: primary first, then secondaries."""
    if limit < 0:
        raise ValueError("limit must be >= 0")
    out: list[CoachingRecommendation] = []
    if report.primary_recommendation is not None:
        out.append(report.primary_recommendation)
    for rec in report.secondary_recommendations:
        if len(out) >= limit:
            break
        out.append(rec)
    return out[:limit]
