"""End-to-end pipelines: videos or CSVs → ScoringReport → CoachingReport.

Uses the existing Coaching Engine public API unchanged.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from format.data import ScoringReport, scoring_report_from_snapshot_grade
from format.data.ace_to_formatted import ace_markers_to_formatted_csv
from format.data.snapshot_report import scoring_report_from_snapshot_paths
from format.render import CoachingReport
from src.coaching_engine import CoachingEngine
from src.motionbert.run_pipeline import process_video
from src.scoring_report_reader import ScoringReportReader


@dataclass(frozen=True)
class SnapshotCoachingPipelineResult:
    """Products of the snapshot → coaching pipeline."""

    scoring_report: ScoringReport
    coaching_report: CoachingReport
    snapshot_grade: dict | None = None
    customer_formatted_csv: Path | None = None
    reference_formatted_csv: Path | None = None
    customer_output_dir: Path | None = None
    reference_output_dir: Path | None = None


def coaching_report_from_scoring_report(
    scoring_report: ScoringReport,
    *,
    max_secondary: int = 2,
) -> CoachingReport:
    """Pass a canonical ScoringReport through Reader → Coaching Engine."""
    context = ScoringReportReader(max_secondary=max_secondary).read(scoring_report)
    return CoachingEngine.with_default_library().generate(context)


def run_snapshot_coaching_pipeline(
    customer_path: str | Path,
    reference_path: str | Path,
    *,
    max_secondary: int = 2,
) -> SnapshotCoachingPipelineResult:
    """Snapshot Comparison Engine → format/data ScoringReport → CoachingReport."""
    from format.data.snapshot_report import _load_grade_snapshots_module

    grade_snapshots = _load_grade_snapshots_module()
    snapshot_grade = grade_snapshots.grade_serve(str(customer_path), str(reference_path))
    scoring_report = scoring_report_from_snapshot_grade(
        snapshot_grade, max_secondary=max_secondary
    )
    coaching_report = coaching_report_from_scoring_report(
        scoring_report, max_secondary=max_secondary
    )
    return SnapshotCoachingPipelineResult(
        scoring_report=scoring_report,
        coaching_report=coaching_report,
        snapshot_grade=snapshot_grade,
        customer_formatted_csv=Path(customer_path),
        reference_formatted_csv=Path(reference_path),
    )


def run_from_scoring_report(
    scoring_report: ScoringReport,
    *,
    max_secondary: int = 2,
) -> SnapshotCoachingPipelineResult:
    """Coaching-only half of the pipeline when a ScoringReport already exists."""
    coaching_report = coaching_report_from_scoring_report(
        scoring_report, max_secondary=max_secondary
    )
    return SnapshotCoachingPipelineResult(
        scoring_report=scoring_report,
        coaching_report=coaching_report,
        snapshot_grade=None,
    )


def _resolve_pose_backend(backend: str) -> str:
    """Use geometric lift when MotionBERT repo/checkpoint are unavailable."""
    if backend != "auto":
        return backend
    try:
        from src.motionbert.motionbert_runner import resolve_motionbert_command

        resolve_motionbert_command()
        return "external"
    except FileNotFoundError as exc:
        print(
            f"MotionBERT unavailable ({exc}); falling back to --backend geometric.\n"
            "Install external/MotionBERT + checkpoint for full 3D refinement."
        )
        return "geometric"


def _video_to_formatted_csv(
    video_path: str | Path,
    *,
    output_root: str | Path,
    backend: str,
    pose_model_path: str | Path | None = None,
    reuse_markers: bool = True,
) -> tuple[Path, Path, dict]:
    """video → MediaPipe/MotionBERT → ACE markers → snapshot-ready CSV."""
    from src.markers.io import SERVE_MARKERS_NPZ, load_serve_markers
    from src.video_discovery import output_dir_for_video

    video = Path(video_path)
    output_dir = output_dir_for_video(video, output_root)
    markers_path = output_dir / SERVE_MARKERS_NPZ

    if reuse_markers and markers_path.is_file():
        print(f"Reusing existing markers: {markers_path}")
        markers = load_serve_markers(markers_path)
    else:
        selected = _resolve_pose_backend(backend)
        output_dir = process_video(
            video,
            output_root=output_root,
            backend=selected,
            pose_model_path=pose_model_path,
        )
        markers = load_serve_markers(output_dir / SERVE_MARKERS_NPZ)

    formatted = ace_markers_to_formatted_csv(
        markers,
        output_dir / f"{video.stem}_formatted.csv",
        fill=True,
    )
    return output_dir, formatted, markers

def run_video_coaching_pipeline(
    customer_video: str | Path,
    reference_video: str | Path,
    *,
    output_root: str | Path = "generated_motionbert",
    backend: str = "auto",
    max_secondary: int = 2,
    pose_model_path: str | Path | None = None,
    reuse_markers: bool = True,
) -> SnapshotCoachingPipelineResult:
    """Two serve videos → pose → snapshots → grade → CoachingReport."""
    customer_dir, customer_csv, _ = _video_to_formatted_csv(
        customer_video,
        output_root=output_root,
        backend=backend,
        pose_model_path=pose_model_path,
        reuse_markers=reuse_markers,
    )
    reference_dir, reference_csv, _ = _video_to_formatted_csv(
        reference_video,
        output_root=output_root,
        backend=backend,
        pose_model_path=pose_model_path,
        reuse_markers=reuse_markers,
    )
    result = run_snapshot_coaching_pipeline(
        customer_csv, reference_csv, max_secondary=max_secondary
    )
    return SnapshotCoachingPipelineResult(
        scoring_report=result.scoring_report,
        coaching_report=result.coaching_report,
        snapshot_grade=result.snapshot_grade,
        customer_formatted_csv=customer_csv,
        reference_formatted_csv=reference_csv,
        customer_output_dir=customer_dir,
        reference_output_dir=reference_dir,
    )

def run_customer_video_vs_reference_csv_pipeline(
    customer_video: str | Path,
    reference_csv: str | Path,
    *,
    output_root: str | Path = "generated_motionbert",
    backend: str = "auto",
    max_secondary: int = 2,
    pose_model_path: str | Path | None = None,
    reuse_markers: bool = True,
) -> SnapshotCoachingPipelineResult:
    """Customer serve video vs. an already-formatted reference CSV → CoachingReport."""
    customer_dir, customer_csv, _ = _video_to_formatted_csv(
        customer_video,
        output_root=output_root,
        backend=backend,
        pose_model_path=pose_model_path,
        reuse_markers=reuse_markers,
    )
    result = run_snapshot_coaching_pipeline(
        customer_csv, reference_csv, max_secondary=max_secondary
    )
    return SnapshotCoachingPipelineResult(
        scoring_report=result.scoring_report,
        coaching_report=result.coaching_report,
        snapshot_grade=result.snapshot_grade,
        customer_formatted_csv=customer_csv,
        reference_formatted_csv=Path(reference_csv),
        customer_output_dir=customer_dir,
        reference_output_dir=None,
    )

def _format_coaching_text(report: CoachingReport) -> str:
    lines: list[str] = []
    sep = "=" * 48
    thin = "-" * 48
    lines.extend([sep, "", "ACE COACHING REPORT", "", "Overall Score", f"{report.overall_score:g}", "", thin])

    def _block(title: str, rec) -> list[str]:
        out = ["", title, "", rec.feature, "", "Correction", "", rec.correction, ""]
        if rec.coach_quotes:
            out.extend(["Coach Quotes", ""])
            out.extend(f"• {q}" for q in rec.coach_quotes)
            out.append("")
        if rec.practice_drills:
            out.extend(["Practice Drills", ""])
            out.extend(f"• {d}" for d in rec.practice_drills)
            out.append("")
        return out

    if report.primary_recommendation is not None:
        lines.extend(_block("Primary Focus", report.primary_recommendation))
        lines.append(thin)
    for rec in report.secondary_recommendations:
        lines.extend(_block("Secondary Focus", rec))
        lines.append(thin)

    lines.extend(["", "Strengths", ""])
    if report.strengths:
        for name in report.strengths[:6]:
            lines.append(f"• {name}")
    else:
        lines.append("• (none)")
    lines.extend(["", sep])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run two tennis serve videos through pose estimation, snapshot grading, "
            "and the Coaching Engine."
        )
    )
    parser.add_argument("customer", help="Player serve video (.mp4) or formatted CSV")
    parser.add_argument("reference", help="Reference serve video (.mp4) or formatted CSV")
    parser.add_argument(
        "--output-root",
        default="generated_motionbert",
        help="Where MotionBERT / formatted CSV outputs are written",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "external", "geometric"),
        default="auto",
        help="MotionBERT backend (auto falls back to geometric if MotionBERT is missing)",
    )
    parser.add_argument("--pose-model", default=None, help="Optional MediaPipe .task model path")
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Ignore existing ace_markers.npz and re-run pose estimation",
    )
    parser.add_argument("--json", action="store_true", help="Print CoachingReport as JSON")
    parser.add_argument("--max-secondary", type=int, default=2)
    args = parser.parse_args(argv)

    customer = Path(args.customer)
    reference = Path(args.reference)
    for path in (customer, reference):
        if not path.exists():
            raise SystemExit(f"File not found: {path}")

    if customer.suffix.lower() == ".csv" and reference.suffix.lower() == ".csv":
        result = run_snapshot_coaching_pipeline(
            customer, reference, max_secondary=args.max_secondary
        )
    else:
        result = run_video_coaching_pipeline(
            customer,
            reference,
            output_root=args.output_root,
            backend=args.backend,
            max_secondary=args.max_secondary,
            pose_model_path=args.pose_model,
            reuse_markers=not args.force_reprocess,
        )

    if args.json:
        print(json.dumps(result.coaching_report.to_dict(), indent=2))
    else:
        if result.customer_formatted_csv is not None:
            print(f"Customer CSV:  {result.customer_formatted_csv}")
        if result.reference_formatted_csv is not None:
            print(f"Reference CSV: {result.reference_formatted_csv}")
        print(f"Overall score: {result.scoring_report.overall_score}")
        print(f"Overall grade: {result.scoring_report.overall_grade}")
        print()
        print(_format_coaching_text(result.coaching_report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SnapshotCoachingPipelineResult",
    "coaching_report_from_scoring_report",
    "run_customer_video_vs_reference_csv_pipeline",   # add this
    "run_from_scoring_report",
    "run_snapshot_coaching_pipeline",
    "run_video_coaching_pipeline",
    "scoring_report_from_snapshot_paths",
    "main",
]