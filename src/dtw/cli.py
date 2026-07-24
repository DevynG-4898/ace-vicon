"""CLI for phase-aware serve comparison."""

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

from src.dtw.animation import run_synchronized_animation
from src.dtw.phase_dtw import format_comparison_report, segment_and_compare

INDIVIDUAL_DIR = _PLOTTING_DIR / "markers" / "individual"


def resolve_serve_path(name: str) -> str:
    """Resolve shorthand serve names to paths."""
    individual = INDIVIDUAL_DIR / name
    if individual.is_dir():
        return str(individual)
    candidate = Path(name)
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        f"Serve not found: {name}. "
        f"Use a folder under {INDIVIDUAL_DIR}, generated_motionbert/<name>, or ace_markers.npz."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-aware DTW comparison of two ACE serves")
    parser.add_argument("serve_a", help="First serve (name, folder, or ace_markers.npz path)")
    parser.add_argument("serve_b", help="Second serve")
    parser.add_argument("--no-animate", action="store_true", help="Print DTW report only")
    parser.add_argument("--legacy-segmentation", action="store_true", help="Use legacy 8-phase segmentation")
    parser.add_argument("--interval-ms", type=int, default=50, help="Animation interval in ms")
    args = parser.parse_args()

    path_a = resolve_serve_path(args.serve_a)
    path_b = resolve_serve_path(args.serve_b)
    name_a = Path(path_a).name if Path(path_a).is_dir() else args.serve_a
    name_b = Path(path_b).name if Path(path_b).is_dir() else args.serve_b

    comparison, bundle = segment_and_compare(
        path_a,
        path_b,
        name_a=name_a,
        name_b=name_b,
        use_v2=not args.legacy_segmentation,
    )
    print(format_comparison_report(comparison))

    if not args.no_animate:
        run_synchronized_animation(
            comparison,
            bundle["markers_a"],
            bundle["markers_b"],
            interval_ms=args.interval_ms,
        )


if __name__ == "__main__":
    main()
