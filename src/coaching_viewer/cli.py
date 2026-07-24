"""CLI for the ACE Coaching Viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLOTTING_DIR = _REPO_ROOT / "plotting"
_SRC_DIR = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PLOTTING_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOTTING_DIR))

from src.coaching_viewer.app import run_coaching_viewer_app
from src.coaching_viewer.pipeline import run
from src.dtw.cli import resolve_serve_path


def _print_summary(session) -> None:
    report = session.coaching_report
    print(f"Overall score: {report.overall_score:.1f}")
    print(f"Overall grade: {session.scoring_report.overall_grade}")
    if report.warnings:
        print("Warnings:")
        for w in report.warnings:
            print(f"  - {w}")
    print("Top recommendations:")
    if not session.recommendations:
        print("  (none)")
        return
    for i, rec in enumerate(session.recommendations, start=1):
        print(f"  {i}. [{rec.priority}] {rec.feature} ({rec.phase})")
        print(f"     {rec.correction}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ACE Coaching Viewer")
    parser.add_argument("player", help="Player serve (name, folder, or ace_markers.npz)")
    parser.add_argument("reference", help="Reference serve")
    parser.add_argument("--no-animate", action="store_true", help="Print coaching summary only")
    parser.add_argument("--interval-ms", type=int, default=50, help="Base animation interval in ms")
    parser.add_argument(
        "--legacy-segmentation",
        action="store_true",
        help="Use legacy 8-phase segmentation",
    )
    args = parser.parse_args(argv)

    player = resolve_serve_path(args.player)
    reference = resolve_serve_path(args.reference)
    session = run(
        player,
        reference,
        name_player=Path(player).name if Path(player).is_dir() else args.player,
        name_reference=Path(reference).name if Path(reference).is_dir() else args.reference,
        use_v2=not args.legacy_segmentation,
    )
    _print_summary(session)

    if not args.no_animate:
        run_coaching_viewer_app(session, interval_ms=args.interval_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
