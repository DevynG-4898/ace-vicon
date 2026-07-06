from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PLOTTING_DIR = Path(__file__).resolve().parents[2] / "plotting"
if str(PLOTTING_DIR) not in sys.path:
    sys.path.insert(0, str(PLOTTING_DIR))

from playback import (  # noqa: E402
    ALLOWED_SPEEDS,
    format_playback_label,
    snap_speed,
    speed_down,
    speed_up,
)
from skeleton_viz import DEFAULT_INTERVAL_MS, compute_axis_limits, draw_skeleton, marker_color_map, marker_names  # noqa: E402

from src.motionbert.ace_adapter import load_ace_markers  # noqa: E402


def _fallback_fps() -> float:
    return 1000.0 / DEFAULT_INTERVAL_MS


def _load_source_fps(ace_markers_path: str | Path) -> float:
    metadata_path = Path(ace_markers_path).parent / "video_metadata.json"
    if not metadata_path.is_file():
        return _fallback_fps()
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fps = float(metadata.get("fps") or 0.0)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return _fallback_fps()
    return fps if fps > 0 else _fallback_fps()


def _timer_interval_ms(fps: float) -> int:
    return max(1, int(round(1000.0 / max(float(fps), 1e-6))))


def _frame_index_for_elapsed(elapsed_seconds: float, *, fps: float, speed: float, n_frames: int) -> int:
    if n_frames <= 0:
        return 0
    frame = int(max(0.0, elapsed_seconds) * max(float(fps), 1e-6) * max(float(speed), 1.0))
    return frame % n_frames


def run_ace_animation(ace_markers_path: str | Path, *, title: str | None = None, speed: int = 1) -> None:
    """Animate MotionBERT output through the existing ACE/Vicon skeleton renderer."""
    path = Path(ace_markers_path)
    markers = load_ace_markers(path)
    frames = markers["frames"]
    n_frames = len(frames)
    source_fps = _load_source_fps(path)
    interval = _timer_interval_ms(source_fps)
    names = marker_names(markers)
    x_lim, y_lim, z_lim = compute_axis_limits(markers)
    colors = marker_color_map(names)
    label = title or path.parent.name

    state = {"speed": snap_speed(speed), "ani": None, "started_at": time.monotonic()}
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    def draw_overlay(frame_value: int) -> None:
        if hasattr(fig, "_speed_overlay") and fig._speed_overlay:
            fig._speed_overlay.remove()
        fig._speed_overlay = fig.text(
            0.02,
            0.02,
            f"Serve: {label}\nFrame: {frame_value}\nPlayback: {format_playback_label(state['speed'])}",
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
        state["started_at"] = time.monotonic()

        def update(_: int) -> None:
            elapsed = time.monotonic() - state["started_at"]
            frame_idx = _frame_index_for_elapsed(elapsed, fps=source_fps, speed=spd, n_frames=n_frames)
            ax.cla()
            draw_skeleton(ax, markers, frame_idx, x_lim, y_lim, z_lim, colors)
            ax.view_init(elev=10, azim=-90)
            frame_value = int(frames[frame_idx])
            ax.set_title(f"{label} - MotionBERT via ACE animation - Frame {frame_value}")
            draw_overlay(frame_value)

        state["ani"] = animation.FuncAnimation(
            fig,
            update,
            frames=max(n_frames, 1),
            interval=interval,
            repeat=True,
            cache_frame_data=False,
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="View MotionBERT output through ACE's Vicon animation renderer.")
    parser.add_argument("ace_markers", help="Path to generated ace_markers.npz")
    parser.add_argument("--speed", type=int, choices=ALLOWED_SPEEDS, default=1)
    args = parser.parse_args()
    run_ace_animation(args.ace_markers, speed=args.speed)


if __name__ == "__main__":
    main()
