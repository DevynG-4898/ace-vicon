"""Side-by-side synchronized DTW animation for two ACE serves."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLOTTING_DIR = _REPO_ROOT / "plotting"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PLOTTING_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOTTING_DIR))

from skeleton_viz import compute_axis_limits, draw_skeleton, marker_color_map, marker_names  # noqa: E402

from src.dtw.alignment import ServeComparison
from src.segmentation.result import PHASE_COLORS


def _phase_color(phase: str) -> str:
    if phase == "Deceleration_Finish":
        return PHASE_COLORS["Deceleration"]
    return PHASE_COLORS.get(phase, "#999999")


def _combined_limits(markers_a: dict, markers_b: dict) -> tuple[tuple[float, float], ...]:
    ax_a = compute_axis_limits(markers_a)
    ax_b = compute_axis_limits(markers_b)
    return (
        (min(ax_a[0][0], ax_b[0][0]), max(ax_a[0][1], ax_b[0][1])),
        (min(ax_a[1][0], ax_b[1][0]), max(ax_a[1][1], ax_b[1][1])),
        (min(ax_a[2][0], ax_b[2][0]), max(ax_a[2][1], ax_b[2][1])),
    )


def _shared_marker_colors(markers_a: dict, markers_b: dict) -> dict[str, tuple]:
    """One color per anatomical marker, shared by both skeleton panels."""
    shared_names = sorted(set(marker_names(markers_a)) | set(marker_names(markers_b)))
    return marker_color_map(shared_names)


def _phase_step_spans(steps) -> list[tuple[int, int, str]]:
    """Return ``(start_step_inclusive, end_step_exclusive, phase)`` for the progress bar."""
    if not steps:
        return []
    spans: list[tuple[int, int, str]] = []
    start = 0
    current = steps[0].phase
    for idx in range(1, len(steps)):
        if steps[idx].phase != current:
            spans.append((start, idx, current))
            start = idx
            current = steps[idx].phase
    spans.append((start, len(steps), current))
    return spans


def run_synchronized_animation(
    comparison: ServeComparison,
    markers_a: dict,
    markers_b: dict,
    *,
    interval_ms: int = 50,
) -> None:
    """Animate both serves advancing together along the phase DTW path."""
    steps = comparison.synchronized_steps
    if not steps:
        raise ValueError("No synchronized steps to animate")

    x_lim, y_lim, z_lim = _combined_limits(markers_a, markers_b)
    colors = _shared_marker_colors(markers_a, markers_b)
    phase_spans = _phase_step_spans(steps)
    total_steps = len(steps)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[10, 1], width_ratios=[1, 1], hspace=0.25)
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    ax_bar = fig.add_subplot(gs[1, :])

    state = {"step": 0, "paused": False}

    def draw_progress(step_idx: int) -> None:
        ax_bar.cla()
        ax_bar.set_xlim(0, total_steps)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("DTW synchronized step (colored by phase)")

        bar_y, bar_h = 0.2, 0.6
        for start, end, phase in phase_spans:
            width = max(end - start, 1)
            ax_bar.add_patch(
                Rectangle(
                    (start, bar_y),
                    width,
                    bar_h,
                    facecolor=_phase_color(phase),
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.55,
                )
            )
            label = phase.replace("_", " ")
            ax_bar.text(
                start + width / 2,
                bar_y + bar_h / 2,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if phase in ("Start_Stance", "Contact", "Cocking") else "black",
                fontweight="bold",
            )

        if step_idx >= 0:
            ax_bar.add_patch(
                Rectangle(
                    (0, bar_y),
                    step_idx + 1,
                    bar_h,
                    facecolor="#333333",
                    alpha=0.25,
                )
            )
            ax_bar.axvline(step_idx + 0.5, color="#E45756", linewidth=2.5, zorder=5)

    def update(_frame: int) -> None:
        if state["paused"]:
            return
        step = steps[state["step"]]
        ax_a.cla()
        ax_b.cla()
        draw_skeleton(ax_a, markers_a, step.global_index_a, x_lim, y_lim, z_lim, colors, show_legend=False)
        draw_skeleton(ax_b, markers_b, step.global_index_b, x_lim, y_lim, z_lim, colors, show_legend=False)
        ax_a.set_title(f"{comparison.name_a}\nFrame {step.vicon_frame_a}")
        ax_b.set_title(f"{comparison.name_b}\nFrame {step.vicon_frame_b}")
        draw_progress(step.step_index)
        fig.suptitle(
            f"Phase: {step.phase.replace('_', ' ')}  |  DTW step {step.step_index + 1}/{total_steps}  |  "
            f"A[{step.global_index_a}] ↔ B[{step.global_index_b}]",
            fontsize=11,
        )
        state["step"] = (state["step"] + 1) % total_steps

    def on_key(event) -> None:
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "left" and state["step"] > 0:
            state["step"] -= 1
        elif event.key == "right":
            state["step"] = min(state["step"] + 1, total_steps - 1)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_steps * 20,
        interval=interval_ms,
        repeat=True,
        cache_frame_data=False,
    )
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.text(0.02, 0.02, "Space: pause  |  ←/→: step", fontsize=9)
    plt.show()
    return ani
