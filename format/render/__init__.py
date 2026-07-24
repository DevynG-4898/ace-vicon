"""Canonical coaching render models (CoachingReport contract).

Re-exports the existing Coaching Engine render types — do not duplicate these classes.
"""

from __future__ import annotations

from src.coaching_engine.models import CoachingRecommendation, CoachingReport

__all__ = [
    "CoachingRecommendation",
    "CoachingReport",
]
