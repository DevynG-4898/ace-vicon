from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from segmentation.config import SegmentationConfig
from segmentation.pipeline import segment_serve_folder, validate_individual_serves


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
        Path(__file__).resolve().parent.parent.parent
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
            for warning in row["warnings"]:
                print(f"  WARNING: {warning}")
        print("  Events (Vicon frame):")
        for name, frame in row["events"].items():
            conf = row["confidence"].get(name, 0.0)
            print(f"    {name:40s} frame {frame:5d}  conf={conf:.2f}")
        print("  Phases (start, end):")
        for name, (start, end) in row["phases"].items():
            duration = end - start + 1 if end >= start else 0
            print(f"    {name:20s} ({start:5d}, {end:5d})  duration={duration} frames")


def main() -> None:
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
        _print_validation_report(validate_individual_serves(config=cfg))
