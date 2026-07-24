"""Canonical scoring data models (ScoringReport contract).

Re-exports the existing ScoringReport schema — do not duplicate these classes.
"""

from __future__ import annotations

from format.data.ace_to_formatted import ace_markers_to_formatted_csv
from format.data.snapshot_report import (
    scoring_report_from_snapshot_grade,
    scoring_report_from_snapshot_paths,
)
from src.scoring_report_reader.report import (
    SUPPORTED_SCHEMA_VERSIONS,
    TIER_TO_CATEGORY,
    FeatureSummary,
    PhaseSummary,
    ScoringReport,
    resolve_category,
)

__all__ = [
    "SUPPORTED_SCHEMA_VERSIONS",
    "TIER_TO_CATEGORY",
    "FeatureSummary",
    "PhaseSummary",
    "ScoringReport",
    "ace_markers_to_formatted_csv",
    "resolve_category",
    "scoring_report_from_snapshot_grade",
    "scoring_report_from_snapshot_paths",
]
