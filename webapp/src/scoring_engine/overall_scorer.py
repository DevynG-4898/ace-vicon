"""Overall serve score from phase summaries."""

from __future__ import annotations

from src.scoring_report_reader.report import PhaseSummary


def grade_for_overall(score: float) -> str:
    if score >= 90:
        return "A — Pro-level serve"
    if score >= 75:
        return "B — Strong serve, minor adjustments needed"
    if score >= 60:
        return "C — Developing serve, focused practice recommended"
    return "D — Fundamentals need significant work"


def score_overall(phase_summaries: tuple[PhaseSummary, ...]) -> tuple[float, str]:
    """Mean of phase scores → overall score and grade label."""
    if not phase_summaries:
        return 0.0, grade_for_overall(0.0)
    overall = round(sum(p.score for p in phase_summaries) / len(phase_summaries), 2)
    return overall, grade_for_overall(overall)
