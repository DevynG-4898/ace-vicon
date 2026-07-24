"""Scoring Engine orchestrator: measurements → ScoringReport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.scoring_engine.feature_scorer import score_all_features
from src.scoring_engine.overall_scorer import score_overall
from src.scoring_engine.phase_scorer import score_phases
from src.scoring_engine.target_selector import build_feature_summaries, select_coaching_targets
from src.scoring_report_reader.report import ScoringReport


@dataclass(frozen=True)
class ScoringEngine:
    """Compare player vs reference scalars and emit a ScoringReport.

    Parameters
    ----------
    max_secondary:
        How many secondary coaching target names to record in report metadata.
        Selection mirrors ScoringReportReader; the reader still derives targets
        from feature ``impact_score`` / ``category``.
    """

    max_secondary: int = 2

    def score(
        self,
        player_values: Mapping[str, float],
        reference_values: Mapping[str, float],
    ) -> ScoringReport:
        """Score all Knowledge Library features and return a ScoringReport."""
        player = {str(k): float(v) for k, v in player_values.items()}
        reference = {str(k): float(v) for k, v in reference_values.items()}

        feature_results = score_all_features(player, reference)
        feature_summaries = build_feature_summaries(feature_results)
        phase_summaries = score_phases(feature_results)
        overall_score, overall_grade = score_overall(phase_summaries)
        primary, secondary = select_coaching_targets(
            feature_summaries, max_secondary=self.max_secondary
        )

        return ScoringReport(
            overall_score=float(overall_score),
            overall_grade=overall_grade,
            phase_summaries=phase_summaries,
            feature_summaries=feature_summaries,
            warnings=(),
            confidence=0.9,
            metadata={
                "source": "scoring_engine",
                "generator_version": 1,
                "primary_coaching_target": primary,
                "secondary_coaching_targets": list(secondary),
            },
            schema_version=1,
        )
