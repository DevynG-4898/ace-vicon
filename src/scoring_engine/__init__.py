"""Scoring Engine — player + reference measurements → ScoringReport.

Compares serve measurements against a reference, scores Knowledge Library
features, rolls up phase and overall scores, and emits a ScoringReport that
ScoringReportReader / CoachingEngine already consume.
"""

from __future__ import annotations

from src.scoring_engine.angle_kernel import score_angle_feature
from src.scoring_engine.contact_kernel import score_contact_event
from src.scoring_engine.engine import ScoringEngine
from src.scoring_engine.result import FeatureScoreResult
from src.scoring_engine.score_builder import FeatureScoreBuilder
from src.scoring_engine.velocity_kernel import score_velocity_feature

__all__ = [
    "FeatureScoreBuilder",
    "FeatureScoreResult",
    "ScoringEngine",
    "score_angle_feature",
    "score_contact_event",
    "score_velocity_feature",
]
