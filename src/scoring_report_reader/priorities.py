"""Priority extraction — sort and interpret ScoringReport fields only.

No score recomputation. Ranking uses the engine-provided ``impact_score`` and
``score`` fields, plus category/tier labels already on the report.
"""

from __future__ import annotations

from src.scoring_report_reader.context import CoachingTarget
from src.scoring_report_reader.report import (
    FeatureSummary,
    PhaseSummary,
    ScoringReport,
    resolve_category,
)


def phase_target(phase: PhaseSummary) -> CoachingTarget:
    category = resolve_category(category=phase.category, tier=phase.tier)
    return CoachingTarget(
        key=f"phase:{phase.name}",
        name=phase.name,
        kind="phase",
        score=float(phase.score),
        impact_score=float(phase.impact_score),
        category=category,
        tier=phase.tier,
        phase=phase.name,
        metadata=dict(phase.metadata),
    )


def feature_target(feature: FeatureSummary) -> CoachingTarget:
    category = resolve_category(category=feature.category, tier=feature.tier)
    phase_part = feature.phase or ""
    key = f"feature:{phase_part}:{feature.name}" if phase_part else f"feature:{feature.name}"
    return CoachingTarget(
        key=key,
        name=feature.name,
        kind="feature",
        score=float(feature.score),
        impact_score=float(feature.impact_score),
        category=category,
        tier=feature.tier,
        phase=feature.phase,
        metadata=dict(feature.metadata),
    )


def extract_targets(report: ScoringReport) -> tuple[CoachingTarget, ...]:
    """Flatten phase and feature summaries into coaching targets."""
    phases = tuple(phase_target(p) for p in report.phase_summaries)
    features = tuple(feature_target(f) for f in report.feature_summaries)
    return phases + features


def _by_impact_desc(target: CoachingTarget) -> tuple[float, str]:
    # Negative impact → descending; name for deterministic ties.
    return (-target.impact_score, target.key)


def _by_score_desc(target: CoachingTarget) -> tuple[float, str]:
    return (-target.score, target.key)


def _by_score_asc(target: CoachingTarget) -> tuple[float, str]:
    return (target.score, target.key)


def ordered_coaching_priorities(targets: tuple[CoachingTarget, ...]) -> tuple[CoachingTarget, ...]:
    """Coachable items ordered by engine ``impact_score`` (highest first).

    Items with ``impact_score <= 0`` are treated as not worth coaching and are
    excluded from the priority list (the engine signals skip via non-positive impact).
    """
    coachable = [t for t in targets if t.impact_score > 0]
    return tuple(sorted(coachable, key=_by_impact_desc))


def ordered_weaknesses(targets: tuple[CoachingTarget, ...]) -> tuple[CoachingTarget, ...]:
    """Weaknesses ordered by impact (highest-impact weakness first)."""
    weaknesses = [t for t in targets if t.category == "weakness" and t.impact_score > 0]
    return tuple(sorted(weaknesses, key=_by_impact_desc))


def ordered_strengths(targets: tuple[CoachingTarget, ...]) -> tuple[CoachingTarget, ...]:
    """Strengths ordered by score (strongest first)."""
    strengths = [t for t in targets if t.category == "strength"]
    return tuple(sorted(strengths, key=_by_score_desc))


def not_worth_coaching(targets: tuple[CoachingTarget, ...]) -> tuple[CoachingTarget, ...]:
    """Items the Scoring Engine marked as not coaching priorities.

    Includes non-positive ``impact_score`` and explicit ``neutral`` category items.
    Sorted by impact ascending so the least relevant appear first.
    """
    skipped = [
        t
        for t in targets
        if t.impact_score <= 0 or t.category == "neutral"
    ]
    return tuple(sorted(skipped, key=lambda t: (t.impact_score, t.key)))


def select_primary_and_secondary(
    *,
    weaknesses: tuple[CoachingTarget, ...],
    priorities: tuple[CoachingTarget, ...],
    max_secondary: int,
) -> tuple[CoachingTarget | None, tuple[CoachingTarget, ...]]:
    """Pick primary / secondary coaching targets without recomputing scores.

    Prefer the highest-impact weakness. If the engine reported no weaknesses,
    fall back to the highest-impact coachable priority.
    """
    if max_secondary < 0:
        raise ValueError("max_secondary must be >= 0")

    pool = weaknesses if weaknesses else priorities
    if not pool:
        return None, ()

    primary = pool[0]
    secondary = tuple(t for t in pool[1 : 1 + max_secondary] if t.key != primary.key)
    return primary, secondary


def weakest_phase_name(phases: tuple[PhaseSummary, ...]) -> str | None:
    if not phases:
        return None
    return min(phases, key=lambda p: (p.score, p.name)).name


def strongest_phase_name(phases: tuple[PhaseSummary, ...]) -> str | None:
    if not phases:
        return None
    return max(phases, key=lambda p: (p.score, p.name)).name


def highest_impact_feature_name(features: tuple[FeatureSummary, ...]) -> str | None:
    if not features:
        return None
    return max(features, key=lambda f: (f.impact_score, f.name)).name


def strongest_feature_name(features: tuple[FeatureSummary, ...]) -> str | None:
    if not features:
        return None
    return max(features, key=lambda f: (f.score, f.name)).name
