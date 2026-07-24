"""Integrity checks for ScoringReport — no scoring, only structural validation."""

from __future__ import annotations

from typing import Iterable

from src.scoring_report_reader.report import (
    SUPPORTED_SCHEMA_VERSIONS,
    FeatureSummary,
    PhaseSummary,
    ScoringReport,
)


class ScoringReportValidationError(ValueError):
    """Raised when a ScoringReport fails integrity checks."""


def _require_finite(name: str, value: float) -> None:
    if value != value:  # NaN
        raise ScoringReportValidationError(f"{name} must be a finite number; got NaN")
    if value in (float("inf"), float("-inf")):
        raise ScoringReportValidationError(f"{name} must be a finite number; got {value}")


def _validate_phases(phases: Iterable[PhaseSummary]) -> None:
    seen: set[str] = set()
    for phase in phases:
        if not phase.name or not str(phase.name).strip():
            raise ScoringReportValidationError("phase summary name must be non-empty")
        if phase.name in seen:
            raise ScoringReportValidationError(f"duplicate phase summary: {phase.name}")
        seen.add(phase.name)
        _require_finite(f"phase[{phase.name}].score", float(phase.score))
        _require_finite(f"phase[{phase.name}].impact_score", float(phase.impact_score))
        if phase.category is not None and phase.category not in ("weakness", "strength", "neutral"):
            raise ScoringReportValidationError(
                f"phase[{phase.name}].category must be weakness|strength|neutral; got {phase.category}"
            )


def _validate_features(features: Iterable[FeatureSummary]) -> None:
    seen: set[str] = set()
    for feature in features:
        if not feature.name or not str(feature.name).strip():
            raise ScoringReportValidationError("feature summary name must be non-empty")
        key = feature.name if feature.phase is None else f"{feature.phase}:{feature.name}"
        if key in seen:
            raise ScoringReportValidationError(f"duplicate feature summary: {key}")
        seen.add(key)
        _require_finite(f"feature[{key}].score", float(feature.score))
        _require_finite(f"feature[{key}].impact_score", float(feature.impact_score))
        if feature.category is not None and feature.category not in ("weakness", "strength", "neutral"):
            raise ScoringReportValidationError(
                f"feature[{key}].category must be weakness|strength|neutral; got {feature.category}"
            )


def validate_scoring_report(report: ScoringReport) -> ScoringReport:
    """Validate report integrity and return the same report on success.

    Checks schema version, required overall fields, uniqueness, and finiteness
    of scores / impact scores. Does not recompute or alter any values.
    """
    if not isinstance(report, ScoringReport):
        raise ScoringReportValidationError(
            f"expected ScoringReport, got {type(report).__name__}"
        )
    if report.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ScoringReportValidationError(
            f"unsupported schema_version {report.schema_version}; "
            f"supported={sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    if report.overall_grade is None or not str(report.overall_grade).strip():
        raise ScoringReportValidationError("overall_grade must be a non-empty string")
    _require_finite("overall_score", float(report.overall_score))
    if report.confidence is not None:
        _require_finite("confidence", float(report.confidence))

    if not isinstance(report.phase_summaries, tuple):
        raise ScoringReportValidationError("phase_summaries must be a tuple")
    if not isinstance(report.feature_summaries, tuple):
        raise ScoringReportValidationError("feature_summaries must be a tuple")
    if not isinstance(report.warnings, tuple):
        raise ScoringReportValidationError("warnings must be a tuple")

    _validate_phases(report.phase_summaries)
    _validate_features(report.feature_summaries)
    return report
