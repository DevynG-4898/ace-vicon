"""
Interactive serve phase viewer with timeline, overlay, and phase selection.

Usage (from repo root or plotting/):
    python plotting/view_serve_phases.py firstserve
    python plotting/view_serve_phases.py firstserve --speed 4

Select view via radio buttons: Full Serve or any of the eight phases.
Keys: + / = faster, - slower, R reset speed.
"""

from __future__ import annotations

import argparse
import os
import sys

from playback import ALLOWED_SPEEDS
from segmentation_viz import run_interactive_viewer

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
INDIVIDUAL_DIR = os.path.join(PLOT_DIR, "markers", "individual")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive ACE serve phase 3D viewer")
    parser.add_argument(
        "serve",
        nargs="?",
        default="firstserve",
        help="Serve folder name under plotting/markers/individual/",
    )
    parser.add_argument(
        "--speed",
        type=int,
        choices=ALLOWED_SPEEDS,
        default=None,
        help="Playback multiplier for all views (1, 2, 4, or 8)",
    )
    args = parser.parse_args()
    serve_dir = os.path.join(INDIVIDUAL_DIR, args.serve)
    if not os.path.isdir(serve_dir):
        print(f"Serve folder not found: {serve_dir}")
        sys.exit(1)
    run_interactive_viewer(serve_dir, args.serve, cli_speed=args.speed)


if __name__ == "__main__":
    main()
