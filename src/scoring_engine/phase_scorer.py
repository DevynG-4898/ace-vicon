"""Phase score rollups from per-feature results."""

from __future__ import annotations

from collections import defaultdict

from src.scoring_engine.result import FeatureScoreResult
from src.scoring_report_reader.report import Category, PhaseSummary


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


def score_phases(feature_results: tuple[FeatureScoreResult, ...]) -> tuple[PhaseSummary, ...]:
    """Average feature scores within each phase.

    Phase ``impact_score`` is always 0 so the Scoring Report Reader coaches
    features (Knowledge Library keys) rather than phase names.
    """
    by_phase: dict[str, list[float]] = defaultdict(list)
    for result in feature_results:
        by_phase[result.phase].append(result.score)

    # Stable phase order matching the Knowledge Library / serve timeline.
    phase_order = (
        "Loading",
        "Cocking",
        "Acceleration",
        "Contact",
        "Deceleration",
        "Finish",
    )
    summaries: list[PhaseSummary] = []
    for name in phase_order:
        scores = by_phase.get(name)
        if not scores:
            continue
        score = round(sum(scores) / len(scores), 2)
        summaries.append(
            PhaseSummary(
                name=name,
                score=score,
                impact_score=0.0,
                tier=_tier_for_score(score),
                category=_category_for_score(score),
                metadata={
                    "feature_count": len(scores),
                    "priority": "none",
                    "note": "Phase rollup for display; coach features instead.",
                },
            )
        )
    return tuple(summaries)
