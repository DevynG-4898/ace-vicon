"""Coaching Engine Version 1 — CoachingContext → CoachingReport.

The engine is intentionally generic. Tennis-specific corrections, quotes, and
drills live exclusively in the Knowledge Library.
"""

from src.coaching_engine.direction import CoachingDirectionError, resolve_direction
from src.coaching_engine.engine import CoachingEngine, CoachingEngineError, generate_coaching_report
from src.coaching_engine.models import CoachingRecommendation, CoachingReport

__all__ = [
    "CoachingDirectionError",
    "CoachingEngine",
    "CoachingEngineError",
    "CoachingRecommendation",
    "CoachingReport",
    "generate_coaching_report",
    "resolve_direction",
]
