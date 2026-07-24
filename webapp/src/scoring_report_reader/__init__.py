"""Scoring Report Reader — ScoringReport → CoachingContext.

This package is intentionally independent of MotionBERT, segmentation, DTW,
and scoring algorithms. It only validates and interprets a completed
:class:`~src.scoring_report_reader.report.ScoringReport`.
"""

from src.scoring_report_reader.context import CoachingContext, CoachingTarget
from src.scoring_report_reader.reader import ScoringReportReader, read_scoring_report
from src.scoring_report_reader.report import (
    SUPPORTED_SCHEMA_VERSIONS,
    TIER_TO_CATEGORY,
    FeatureSummary,
    PhaseSummary,
    ScoringReport,
    resolve_category,
)
from src.scoring_report_reader.validation import (
    ScoringReportValidationError,
    validate_scoring_report,
)

__all__ = [
    "SUPPORTED_SCHEMA_VERSIONS",
    "TIER_TO_CATEGORY",
    "CoachingContext",
    "CoachingTarget",
    "FeatureSummary",
    "PhaseSummary",
    "ScoringReport",
    "ScoringReportReader",
    "ScoringReportValidationError",
    "read_scoring_report",
    "resolve_category",
    "validate_scoring_report",
]
