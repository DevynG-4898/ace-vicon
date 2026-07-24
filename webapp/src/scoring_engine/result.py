"""Shared scoring result types for the Scoring Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Direction = Literal["too_low", "too_high", "acceptable"]


@dataclass(frozen=True)
class FeatureScoreResult:
    """Result of scoring one coaching feature."""

    name: str
    phase: str
    score: float
    direction: Direction
    impact_score: float
    player_value: float
    reference_value: float
    difference: float
    unit: str = "deg"
    confidence: float = 1.0
    measurements: dict[str, Any] = field(default_factory=dict)
