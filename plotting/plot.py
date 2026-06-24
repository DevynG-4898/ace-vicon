"""
Full-serve 3D skeleton animation (original ACE behavior).

Usage:
    python plotting/plot.py firstserve
    python plotting/plot.py firstserve --speed 4

Keys: + / = faster, - slower, R reset speed.

For phase segmentation, timeline, and per-phase views use:
    python plotting/view_serve_phases.py firstserve
    python plotting/generate_segmentation_validation.py
"""

from __future__ import annotations

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from playback import (
    ALLOWED_SPEEDS,
    build_play_sequence,
    format_playback_label,
    playback_interval_ms,
    snap_speed,
    speed_down,
    speed_up,
)
from skeleton_viz import (
    compute_axis_limits,
    draw_skeleton,
    load_serve_from_dir,
    marker_color_map,
    marker_names,
)

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
INDIVIDUAL_DIR = os.path.join(PLOT_DIR, "markers", "individual")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-serve 3D skeleton animation")
    parser.add_argument("serve", nargs="?", default="firstserve")
    parser.add_argument("--speed", type=int, choices=ALLOWED_SPEEDS, default=1)
    args = parser.parse_args()

    serve_dir = os.path.join(INDIVIDUAL_DIR, args.serve)
    if not os.path.isdir(serve_dir):
        print(f"Serve folder not found: {serve_dir}")
        raise SystemExit(1)

    markers = load_serve_from_dir(serve_dir)
    frames = markers["frames"]
    n_frames = len(frames)
    names = marker_names(markers)
    x_lim, y_lim, z_lim = compute_axis_limits(markers)
    colors = marker_color_map(names)

    state = {"speed": snap_speed(args.speed), "ani": None}
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw_overlay(vicon_frame: int) -> None:
        if hasattr(fig, "_speed_overlay") and fig._speed_overlay:
            fig._speed_overlay.remove()
        fig._speed_overlay = fig.text(
            0.02,
            0.02,
            f"Serve: {args.serve}\nFrame: {vicon_frame}\nPlayback: {format_playback_label(state['speed'])}",
            transform=fig.transFigure,
            fontsize=10,
            va="bottom",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
        )

    def start_animation() -> None:
        if state["ani"] is not None:
            state["ani"].event_source.stop()
        spd = state["speed"]
        play_sequence = build_play_sequence(n_frames, spd, "Full Serve")
        interval = playback_interval_ms(spd)

        def update(play_pos: int) -> None:
            frame_idx = play_sequence[play_pos % len(play_sequence)]
            ax.cla()
            draw_skeleton(ax, markers, frame_idx, x_lim, y_lim, z_lim, colors)
            vf = int(frames[frame_idx])
            ax.set_title(f"{args.serve}  —  Frame {vf}")
            draw_overlay(vf)

        state["ani"] = animation.FuncAnimation(
            fig,
            update,
            frames=max(len(play_sequence), 1),
            interval=interval,
            repeat=True,
        )
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key in ("+", "="):
            state["speed"] = speed_up(state["speed"])
            start_animation()
        elif event.key == "-":
            state["speed"] = speed_down(state["speed"])
            start_animation()
        elif event.key in ("r", "R"):
            state["speed"] = snap_speed(args.speed)
            start_animation()

    fig.canvas.mpl_connect("key_press_event", on_key)
    start_animation()
    plt.show()


if __name__ == "__main__":
    main()
