"""End-to-end Coaching Viewer pipeline: DTW → scalars → score → coach."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from format.pipeline import coaching_report_from_scoring_report
from src.coaching_engine.models import CoachingRecommendation, CoachingReport
from src.coaching_viewer.recommendations import top_recommendations
from src.coaching_viewer.scalars import extract_feature_scalars
from src.dtw.alignment import ServeComparison
from src.dtw.phase_dtw import segment_and_compare
from src.scoring_engine import ScoringEngine
from src.scoring_report_reader.report import FeatureSummary, ScoringReport


@dataclass(frozen=True)
class CoachingViewerSession:
    """View-model bundle consumed by the matplotlib Coaching Viewer."""

    markers_player: dict
    markers_reference: dict
    comparison: ServeComparison
    scoring_report: ScoringReport
    coaching_report: CoachingReport
    recommendations: list[CoachingRecommendation]
    feature_by_name: dict[str, FeatureSummary]


def build_session_from_parts(
    *,
    markers_player: dict,
    markers_reference: dict,
    comparison: ServeComparison,
    scoring_report: ScoringReport,
    coaching_report: CoachingReport,
) -> CoachingViewerSession:
    """Assemble a session from already-computed pipeline products."""
    feature_by_name = {f.name: f for f in scoring_report.feature_summaries}
    return CoachingViewerSession(
        markers_player=markers_player,
        markers_reference=markers_reference,
        comparison=comparison,
        scoring_report=scoring_report,
        coaching_report=coaching_report,
        recommendations=top_recommendations(coaching_report),
        feature_by_name=feature_by_name,
    )


def run(
    player_source: str | Path,
    reference_source: str | Path,
    *,
    name_player: str | None = None,
    name_reference: str | None = None,
    use_v2: bool = True,
    handedness: str = "right",
) -> CoachingViewerSession:
    """Load two serves, align, score, coach, and return a viewer session.

    Panel convention: reference is DTW serve A (left), player is serve B (right).
    """
    comparison, bundle = segment_and_compare(
        str(reference_source),
        str(player_source),
        name_a=name_reference or Path(reference_source).name,
        name_b=name_player or Path(player_source).name,
        use_v2=use_v2,
    )
    markers_reference = bundle["markers_a"]
    markers_player = bundle["markers_b"]

    player_scalars = extract_feature_scalars(
        markers_player,
        comparison.segmentation_b,
        handedness=handedness,
    )
    reference_scalars = extract_feature_scalars(
        markers_reference,
        comparison.segmentation_a,
        handedness=handedness,
    )

    scoring_report = ScoringEngine().score(player_scalars, reference_scalars)
    coaching_report = coaching_report_from_scoring_report(scoring_report, max_secondary=2)

    return build_session_from_parts(
        markers_player=markers_player,
        markers_reference=markers_reference,
        comparison=comparison,
        scoring_report=scoring_report,
        coaching_report=coaching_report,
    )


def run_from_scoring_report(
    *,
    markers_player: dict,
    markers_reference: dict,
    comparison: ServeComparison,
    scoring_report: ScoringReport,
    max_secondary: int = 2,
) -> CoachingViewerSession:
    """Build a viewer session from an existing canonical ScoringReport.

    Used when scoring comes from the Snapshot Comparison Scoring Engine
    (``format.data.ScoringReport``) instead of the DTW scalar ScoringEngine.
    Coaching Engine and CoachingReport render models are unchanged.
    """
    coaching_report = coaching_report_from_scoring_report(
        scoring_report, max_secondary=max_secondary
    )
    return build_session_from_parts(
        markers_player=markers_player,
        markers_reference=markers_reference,
        comparison=comparison,
        scoring_report=scoring_report,
        coaching_report=coaching_report,
    )
