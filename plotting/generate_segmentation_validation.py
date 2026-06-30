"""
Batch-generate segmentation validation assets for all individual serves.

Outputs per serve under plotting/segmentation_validation/<serve_name>/:
  - segmentation.json
  - timeline.png
  - debug_signals.png
  - phase_summary.txt
  - animation_Full_Serve.gif (+ one GIF per phase when --gifs)

Usage:
    python plotting/generate_segmentation_validation.py
    python plotting/generate_segmentation_validation.py --no-gifs
    python plotting/generate_segmentation_validation.py --serve firstserve
"""

from __future__ import annotations

import argparse
import os
import sys

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
if PLOT_DIR not in sys.path:
    sys.path.insert(0, PLOT_DIR)

from pathlib import Path

from playback import ALLOWED_SPEEDS  # noqa: E402
from segmentation_viz import (  # noqa: E402
    OUTPUT_ROOT,
    generate_all_individual_serves,
    generate_serve_validation_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate segmentation validation visuals for individual serves"
    )
    parser.add_argument(
        "--serve",
        help="Process only this serve folder name (default: all in individual/)",
    )
    parser.add_argument(
        "--no-gifs",
        action="store_true",
        help="Skip GIF generation (faster; still writes JSON and PNGs)",
    )
    parser.add_argument("--fps", type=int, default=30, help="GIF frames per second")
    parser.add_argument(
        "--gif-stride",
        type=int,
        default=2,
        help="Use every Nth frame in GIF exports (1 = all frames)",
    )
    parser.add_argument(
        "--speed",
        type=int,
        choices=ALLOWED_SPEEDS,
        default=1,
        help="GIF playback speed multiplier (1, 2, 4, or 8)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_ROOT),
        help="Output root directory",
    )
    args = parser.parse_args()
    save_gifs = not args.no_gifs
    base = Path(PLOT_DIR) / "markers" / "individual"
    out_root = Path(args.output)

    if args.serve:
        serve_dir = base / args.serve
        if not serve_dir.is_dir():
            print(f"Not found: {serve_dir}")
            sys.exit(1)
        paths = generate_serve_validation_assets(
            serve_dir,
            out_root / args.serve,
            save_gifs=save_gifs,
            fps=args.fps,
            gif_stride=args.gif_stride,
            speed=args.speed,
        )
        print(f"Generated assets for {args.serve} -> {paths['segmentation'].parent}")
        for key, path in paths.items():
            print(f"  {key}: {path}")
        return

    print(f"Generating validation assets (gifs={save_gifs}) -> {out_root}")
    report = generate_all_individual_serves(
        base,
        save_gifs=save_gifs,
        gif_stride=args.gif_stride,
        fps=args.fps,
        speed=args.speed,
    )
    for row in report:
        print(f"\n{row['serve']} -> {row['output_dir']}")
        for key, path in row["files"].items():
            print(f"  {key}: {path}")
    print(f"\nDone. {len(report)} serves processed.")


if __name__ == "__main__":
    main()
