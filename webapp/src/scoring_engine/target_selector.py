"""Convert feature score results into FeatureSummary rows with coaching metadata.

Assigns category / impact so ScoringReportReader can pick primary and secondary
targets without any Scoring Engine fields beyond the existing ScoringReport.
"""

from __future__ import annotations

from typing import Any

from src.scoring_engine.result import FeatureScoreResult
from src.scoring_report_reader.report import Category, FeatureSummary


def _tier_for_score(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 80:
        return "Good"
    if score >= 60:
        return "Fair"
    return "Poor"


def _category_for_result(result: FeatureScoreResult) -> Category:
    if result.direction == "acceptable" or result.score >= 80:
        return "strength"
    if result.score >= 60:
        return "neutral"
    return "weakness"


def _priority_label(impact_score: float, category: Category) -> str:
    if category != "weakness" or impact_score <= 0:
        return "none"
    if impact_score >= 0.85:
        return "high"
    if impact_score >= 0.6:
        return "medium"
    return "low"


def _coaching_direction(result: FeatureScoreResult) -> str:
    """Direction stored for the Coaching Engine (never ``acceptable``)."""
    if result.direction in ("too_low", "too_high"):
        return result.direction
    # Acceptable / strengths still need a legal direction if ever coached.
    if result.difference < 0:
        return "too_low"
    return "too_high"


def _impact_for_category(result: FeatureScoreResult, category: Category) -> float:
    """Weaknesses keep computed impact; strengths/neutrals get low/zero impact."""
    if category == "weakness":
        return float(result.impact_score)
    if category == "neutral":
        return round(min(result.impact_score, 0.25), 4)
    return round(min(result.impact_score, 0.1), 4)


def build_feature_summaries(
    feature_results: tuple[FeatureScoreResult, ...],
) -> tuple[FeatureSummary, ...]:
    """Build FeatureSummary rows with direction metadata for the Coaching Engine."""
    summaries: list[FeatureSummary] = []
    for result in feature_results:
        category = _category_for_result(result)
        impact = _impact_for_category(result, category)
        direction = _coaching_direction(result)
        metadata: dict[str, Any] = {
            "feature_name": result.name,
            "score": result.score,
            "reference_value": result.reference_value,
            "player_value": result.player_value,
            "difference": result.difference,
            "unit": result.unit,
            "impact_score": impact,
            "priority": _priority_label(impact, category),
            "category": category,
            "direction": direction,
            "raw_direction": result.direction,
            "confidence": result.confidence,
            "measurements": dict(result.measurements),
        }
        summaries.append(
            FeatureSummary(
                name=result.name,
                score=float(result.score),
                impact_score=float(impact),
                tier=_tier_for_score(result.score),
                category=category,
                phase=result.phase,
                metadata=metadata,
            )
        )
    return tuple(summaries)


def select_coaching_targets(
    feature_summaries: tuple[FeatureSummary, ...],
    *,
    max_secondary: int = 2,
) -> tuple[str | None, tuple[str, ...]]:
    """Pick primary / secondary feature names (highest-impact weaknesses first).

    Mirrors ScoringReportReader selection so report metadata can record the
    engine's intended targets. The reader still derives targets from impact.
    """
    weaknesses = [
        f
        for f in feature_summaries
        if f.category == "weakness" and f.impact_score > 0
    ]
    weaknesses.sort(key=lambda f: (-f.impact_score, f.name))
    if not weaknesses:
        return None, ()
    primary = weaknesses[0].name
    secondary = tuple(f.name for f in weaknesses[1 : 1 + max_secondary])
    return primary, secondary
