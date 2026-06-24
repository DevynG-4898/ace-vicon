"""
Non-interactive check: playback timing stays aligned with Vicon frames at each speed.

Usage:
    python plotting/demo_playback_speed.py firstserve
"""

from __future__ import annotations

import argparse
import os
import sys

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
if PLOT_DIR not in sys.path:
    sys.path.insert(0, PLOT_DIR)

_SRC = os.path.join(PLOT_DIR, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from playback import (  # noqa: E402
    ALLOWED_SPEEDS,
    PHASE_DEFAULT_SPEED,
    build_play_sequence,
    playback_interval_ms,
    resolve_view_speed,
)
from segmentation_viz import load_serve_from_dir  # noqa: E402
from serve_segmentation import segment_serve, view_index_range  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("serve", nargs="?", default="firstserve")
    args = parser.parse_args()
    serve_dir = os.path.join(PLOT_DIR, "markers", "individual", args.serve)
    markers = load_serve_from_dir(serve_dir)
    result = segment_serve(markers)
    frames = result.frames

    print(f"Playback sync check — {args.serve}")
    print(f"Vicon range: {int(frames[0])}-{int(frames[-1])}  ({len(frames)} samples)\n")

    for speed in ALLOWED_SPEEDS:
        print(f"=== Global speed {speed}x ===")
        for view in ("Full Serve", "Cocking", "Contact"):
            i0, i1 = view_index_range(frames, result.phases, view)
            n_play = i1 - i0 + 1
            eff = resolve_view_speed(view, cli_speed=float(speed) if speed > 1 else None)
            seq = build_play_sequence(n_play, eff, view)
            interval = playback_interval_ms(eff)
            first_v = int(frames[i0 + seq[0]])
            last_v = int(frames[i0 + seq[-1]])
            if view == "Full Serve":
                phase_range = (int(frames[0]), int(frames[-1]))
            else:
                phase_range = result.phases[view]
            print(
                f"  {view:16s} eff={eff}x  interval={interval:2d}ms  "
                f"steps={len(seq):4d}  Vicon {first_v}-{last_v}  "
                f"phase_bounds={phase_range}"
            )
        print()

    print("Per-phase defaults (speed=1, no CLI override):")
    for phase, default in PHASE_DEFAULT_SPEED.items():
        print(f"  {phase}: {default}x")


if __name__ == "__main__":
    main()
