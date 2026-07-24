"""ScoringReport contract — the only input the Scoring Report Reader accepts.

The Scoring Engine owns how these values are computed. This module defines the
stable schema the reader trusts. If the engine's internals change but it still
emits this shape, the reader requires no changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

Category = Literal["weakness", "strength", "neutral"]
ItemKind = Literal["phase", "feature"]

# Interpret tier labels already assigned by the Scoring Engine (no score math).
TIER_TO_CATEGORY: Mapping[str, Category] = {
    "Excellent": "strength",
    "Good": "strength",
    "Fair": "weakness",
    "Poor": "weakness",
}

SUPPORTED_SCHEMA_VERSIONS = frozenset({1})


@dataclass(frozen=True)
class PhaseSummary:
    """Per-phase scores already computed by the Scoring Engine."""

    name: str
    score: float
    impact_score: float
    tier: str | None = None
    category: Category | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureSummary:
    """Per-feature scores already computed by the Scoring Engine."""

    name: str
    score: float
    impact_score: float
    tier: str | None = None
    category: Category | None = None
    phase: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScoringReport:
    """Completed analysis product of the Scoring Engine.

    ``impact_score`` is the engine's coaching-priority signal: higher means the
    item should be coached sooner. The reader only sorts and interprets; it
    never recomputes impact or scores.
    """

    overall_score: float
    overall_grade: str
    phase_summaries: tuple[PhaseSummary, ...]
    feature_summaries: tuple[FeatureSummary, ...]
    warnings: tuple[str, ...] = ()
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1


def resolve_category(
    *,
    category: Category | None,
    tier: str | None,
) -> Category:
    """Resolve an item category from engine-provided fields only.

    Preference order:
      1. Explicit ``category`` from the Scoring Engine
      2. Known ``tier`` label mapped via :data:`TIER_TO_CATEGORY`
      3. ``neutral`` when neither is available
    """
    if category is not None:
        return category
    if tier is None:
        return "neutral"
    return TIER_TO_CATEGORY.get(tier, "neutral")
