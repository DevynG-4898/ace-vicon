"""FeatureScoreBuilder — combine weighted biomechanical component scores.

Each Knowledge Library feature can aggregate magnitude, timing, smoothness,
consistency (and later DTW similarity, etc.) into one final feature score
without changing the ScoringReport / Coaching Engine pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.scoring_engine.result import Direction, FeatureScoreResult

# Placeholder default weights (sum to 1.0). Individual scorers may override.
DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "magnitude": 0.50,
    "timing": 0.20,
    "smoothness": 0.15,
    "consistency": 0.15,
}


@dataclass
class _Component:
    name: str
    score: float
    weight: float


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _impact_from_score(score: float, direction: Direction) -> float:
    if direction == "acceptable":
        return 0.0
    return round(_clamp((100.0 - score) / 100.0, 0.0, 1.0), 4)


@dataclass
class FeatureScoreBuilder:
    """Fluent builder for a single coaching feature score.

    Example::

        result = (
            FeatureScoreBuilder("Knee Flexion", "Loading")
            .add_component("magnitude", score=74.0, weight=0.50)
            .add_component("timing", score=80.0, weight=0.20)
            .add_component("smoothness", score=90.0, weight=0.15)
            .add_component("consistency", score=85.0, weight=0.15)
            .set_direction("too_low")
            .set_values(player_value=71.0, reference_value=82.0)
            .build()
        )
    """

    name: str
    phase: str
    unit: str = "deg"
    confidence: float = 1.0
    _components: list[_Component] = field(default_factory=list, init=False, repr=False)
    _direction: Direction | None = field(default=None, init=False, repr=False)
    _player_value: float | None = field(default=None, init=False, repr=False)
    _reference_value: float | None = field(default=None, init=False, repr=False)
    _extra_measurements: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def add_component(self, name: str, score: float, weight: float) -> FeatureScoreBuilder:
        """Add a weighted component score in ``[0, 100]``."""
        if weight < 0:
            raise ValueError(f"component weight must be >= 0; got {weight}")
        self._components.append(
            _Component(name=str(name), score=_clamp(float(score)), weight=float(weight))
        )
        return self

    def set_direction(self, direction: Direction) -> FeatureScoreBuilder:
        self._direction = direction
        return self

    def set_values(self, *, player_value: float, reference_value: float) -> FeatureScoreBuilder:
        self._player_value = float(player_value)
        self._reference_value = float(reference_value)
        return self

    def set_confidence(self, confidence: float) -> FeatureScoreBuilder:
        self.confidence = float(confidence)
        return self

    def add_measurement(self, key: str, value: Any) -> FeatureScoreBuilder:
        """Attach an extra debug / explanation measurement."""
        self._extra_measurements[str(key)] = value
        return self

    def build(self) -> FeatureScoreResult:
        """Combine components into a :class:`FeatureScoreResult`."""
        if not self._components:
            raise ValueError("FeatureScoreBuilder requires at least one component")
        if self._direction is None:
            raise ValueError("FeatureScoreBuilder requires a direction")
        if self._player_value is None or self._reference_value is None:
            raise ValueError("FeatureScoreBuilder requires player and reference values")

        total_weight = sum(c.weight for c in self._components)
        if total_weight <= 0:
            raise ValueError("total component weight must be > 0")

        final_score = sum(c.score * c.weight for c in self._components) / total_weight
        final_score = round(_clamp(final_score), 2)
        difference = round(self._player_value - self._reference_value, 4)
        impact = _impact_from_score(final_score, self._direction)

        measurements: dict[str, Any] = {
            "player_value": self._player_value,
            "reference_value": self._reference_value,
            "difference": difference,
            "final_score": final_score,
            "confidence": self.confidence,
            "component_weights": {c.name: c.weight for c in self._components},
        }
        for component in self._components:
            measurements[f"{component.name}_score"] = round(component.score, 2)
        measurements.update(self._extra_measurements)

        return FeatureScoreResult(
            name=self.name,
            phase=self.phase,
            score=final_score,
            direction=self._direction,
            impact_score=impact,
            player_value=self._player_value,
            reference_value=self._reference_value,
            difference=difference,
            unit=self.unit,
            confidence=self.confidence,
            measurements=measurements,
        )
