"""Fake Scoring Report Generator — simulate Scoring Engine output for pipeline demos.

Produces a valid :class:`~src.scoring_report_reader.report.ScoringReport` that the
Scoring Report Reader and Coaching Engine can consume end-to-end. This package
does not implement real scoring.
"""

from __future__ import annotations

from src.fake_scoring_report.fake_report_generator import (
    FakeScoringReportGenerator,
    generate_fake_scoring_report,
)

__all__ = [
    "FakeScoringReportGenerator",
    "generate_fake_scoring_report",
]
