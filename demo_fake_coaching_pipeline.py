#!/usr/bin/env python3
"""End-to-end demo: Fake ScoringReport → Reader → CoachingEngine → CoachingReport."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.coaching_engine import CoachingEngine, CoachingReport
from src.fake_scoring_report import generate_fake_scoring_report
from src.scoring_report_reader import ScoringReportReader


def format_coaching_report(report: CoachingReport) -> str:
    """Render a CoachingReport as a clean human-readable block."""
    lines: list[str] = []
    sep = "=" * 48
    thin = "-" * 48

    lines.append(sep)
    lines.append("")
    lines.append("ACE COACHING REPORT")
    lines.append("")
    lines.append("Overall Score")
    lines.append(f"{report.overall_score:g}")
    lines.append("")
    lines.append(thin)

    if report.primary_recommendation is not None:
        lines.extend(_format_recommendation_block("Primary Focus", report.primary_recommendation))
        lines.append(thin)

    for rec in report.secondary_recommendations:
        lines.extend(_format_recommendation_block("Secondary Focus", rec))
        lines.append(thin)

    lines.append("")
    lines.append("Strengths")
    lines.append("")
    if report.strengths:
        preferred = ("Balance", "Follow Through", "Hip Rotation Velocity")
        shown: list[str] = []
        for name in preferred:
            if any(name == s or s.startswith(f"{name} ") or s.startswith(f"{name}(") for s in report.strengths):
                shown.append(name)
        for s in report.strengths:
            base = s.split(" (")[0]
            if base not in shown:
                shown.append(base)
        for name in shown[:6]:
            lines.append(f"• {name}")
    else:
        lines.append("• (none)")
    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def _format_recommendation_block(title: str, rec) -> list[str]:
    lines = [
        "",
        title,
        "",
        rec.feature,
        "",
        "Correction",
        "",
        rec.correction,
        "",
    ]
    if rec.coach_quotes:
        lines.append("Coach Quotes")
        lines.append("")
        for quote in rec.coach_quotes:
            lines.append(f"• {quote}")
        lines.append("")
    if rec.practice_drills:
        lines.append("Practice Drills")
        lines.append("")
        for drill in rec.practice_drills:
            lines.append(f"• {drill}")
        lines.append("")
    return lines


def run_pipeline() -> CoachingReport:
    """Execute the full fake coaching pipeline and return the report."""
    scoring_report = generate_fake_scoring_report()
    context = ScoringReportReader(max_secondary=2).read(scoring_report)
    return CoachingEngine.with_default_library().generate(context)


def main() -> None:
    report = run_pipeline()
    print(format_coaching_report(report))


if __name__ == "__main__":
    main()
