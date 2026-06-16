"""
Compatibility facade for ACE serve phase segmentation.

Legacy 8-phase detection remains the default (use_legacy_detection=True).
Biomechanics-first v2 segmentation is available via SegmentationConfig(use_legacy_detection=False)
or segment_serve_v2().
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_DEPS_HINT = (
    "Missing Python dependencies. From the project root, use the project venv:\n"
    "  python -m venv .venv\n"
    "  .venv/bin/pip install -r requirements.txt\n"
    "  .venv/bin/python src/serve_segmentation.py --v2 --frames firstserve\n"
    "Or activate it: source .venv/bin/activate"
)

try:
    from segmentation.config import SegmentationConfig  # noqa: E402
    from segmentation.io import load_serve_from_folder  # noqa: E402
    from segmentation.legacy import segment_serve_legacy  # noqa: E402
    from segmentation.pipeline import (  # noqa: E402
        segment_serve,
        segment_serve_folder,
        segment_serve_v2,
        validate_individual_serves,
    )
    from segmentation.result import (  # noqa: E402
        EVENT_LABELS,
        EVENT_NAMES,
        PHASE_COLORS,
        PHASE_NAMES,
        V2_EVENT_LABELS,
        V2_EVENT_NAMES,
        V2_PHASE_NAMES,
        VIEW_OPTIONS,
        SegmentationResult,
        phase_at_index,
        phase_to_index_range,
        vicon_frame_to_index,
        view_index_range,
    )
except ModuleNotFoundError as exc:
    if exc.name in {"pandas", "numpy", "scipy", "matplotlib"}:
        print(_DEPS_HINT, file=sys.stderr)
        raise SystemExit(1) from exc
    raise

__all__ = [
    "EVENT_LABELS",
    "EVENT_NAMES",
    "PHASE_COLORS",
    "PHASE_NAMES",
    "V2_EVENT_LABELS",
    "V2_EVENT_NAMES",
    "V2_PHASE_NAMES",
    "VIEW_OPTIONS",
    "SegmentationConfig",
    "SegmentationResult",
    "format_phase_frames",
    "load_serve_from_folder",
    "phase_at_index",
    "phase_to_index_range",
    "print_phase_frames",
    "segment_serve",
    "segment_serve_folder",
    "segment_serve_legacy",
    "segment_serve_v2",
    "validate_individual_serves",
    "vicon_frame_to_index",
    "view_index_range",
]


def format_phase_frames(phases: dict[str, tuple[int, int]]) -> str:
    lines = []
    for name in phases:
        start, end = phases[name]
        if name == "Contact" or start == end:
            lines.append(f"{name}: {start}")
        else:
            lines.append(f"{name}: {start}-{end}")
    return "\n".join(lines)


def print_phase_frames(serve_name: str, config: SegmentationConfig | None = None) -> None:
    serve_dir = (
        Path(__file__).resolve().parent.parent
        / "plotting"
        / "markers"
        / "individual"
        / serve_name
    )
    if not serve_dir.is_dir():
        raise FileNotFoundError(f"Serve folder not found: {serve_dir}")
    result = segment_serve_folder(serve_dir, config)
    print(format_phase_frames(result.phases))


def _print_validation_report(rows: list[dict[str, Any]]) -> None:
    print("=" * 80)
    print("ACE SERVE PHASE SEGMENTATION — VALIDATION (individual/)")
    print("=" * 80)
    for row in rows:
        print(f"\n--- {row['serve']} --- (schema v{row.get('schema_version', 1)})")
        if row["warnings"]:
            for w in row["warnings"]:
                print(f"  WARNING: {w}")
        print("  Events (Vicon frame):")
        for name, frame in row["events"].items():
            conf = row["confidence"].get(name, 0.0)
            print(f"    {name:40s} frame {frame:5d}  conf={conf:.2f}")
        print("  Phases (start, end):")
        for name, (a, b) in row["phases"].items():
            dur = b - a + 1 if b >= a else 0
            print(f"    {name:20s} ({a:5d}, {b:5d})  duration={dur} frames")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ACE serve phase segmentation")
    parser.add_argument(
        "--frames",
        metavar="SERVE",
        help="Print compact Vicon frame ranges for one serve (e.g. firstserve)",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use biomechanics-first v2 segmentation",
    )
    args = parser.parse_args()
    cfg = SegmentationConfig(use_legacy_detection=not args.v2)

    if args.frames:
        print_phase_frames(args.frames, cfg)
    else:
        report = validate_individual_serves(config=cfg)
        _print_validation_report(report)
