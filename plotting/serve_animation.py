"""Shared full-serve animation using the canonical ACE marker dictionary."""

from __future__ import annotations

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
    marker_color_map,
    marker_names,
)


def run_full_serve_animation(
    markers: dict,
    title: str,
    *,
    speed: int = 1,
) -> None:
    """Animate a serve using the same renderer as ``plotting/plot.py``."""
    if speed not in ALLOWED_SPEEDS:
        speed = snap_speed(speed)

    frames = markers["frames"]
    n_frames = len(frames)
    names = marker_names(markers)
    x_lim, y_lim, z_lim = compute_axis_limits(markers)
    colors = marker_color_map(names)
    state = {"speed": snap_speed(speed), "ani": None}

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw_overlay(vicon_frame: int) -> None:
        if hasattr(fig, "_speed_overlay") and fig._speed_overlay:
            fig._speed_overlay.remove()
        fig._speed_overlay = fig.text(
            0.02,
            0.02,
            f"Serve: {title}\nFrame: {vicon_frame}\nPlayback: {format_playback_label(state['speed'])}",
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
            ax.set_title(f"{title}  —  Frame {vf}")
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
            state["speed"] = snap_speed(speed)
            start_animation()

    fig.canvas.mpl_connect("key_press_event", on_key)
    start_animation()
    plt.show()
