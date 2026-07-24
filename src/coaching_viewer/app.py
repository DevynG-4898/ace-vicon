"""Matplotlib ACE Coaching Viewer — DTW Viewer visuals + compact coaching UI."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLOTTING_DIR = _REPO_ROOT / "plotting"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PLOTTING_DIR) not in sys.path:
    sys.path.insert(0, str(_PLOTTING_DIR))

from skeleton_viz import draw_skeleton, get_pos  # noqa: E402

from src.coaching_engine.models import CoachingRecommendation
from src.coaching_viewer.display_orient import orient_ace_markers_for_display
from src.coaching_viewer.joint_map import highlight_joints, resolve_highlight_markers
from src.coaching_viewer.phase_map import coaching_phase_to_dtw
from src.coaching_viewer.pipeline import CoachingViewerSession
from src.dtw.alignment import SynchronizedStep
from src.dtw.animation import (  # reuse DTW Viewer timeline helpers
    _combined_limits,
    _phase_color,
    _phase_step_spans,
    _shared_marker_colors,
)
from src.scoring_report_reader.report import FeatureSummary

_SPEEDS = (0.5, 1.0, 2.0)
# Shared matplotlib 3D camera for both panels after orientation normalization
_VIEW_ELEV = 20.0
_VIEW_AZIM = -60.0


def phase_color(phase: str) -> str:
    """Public alias of the DTW Viewer phase-color helper."""
    return _phase_color(phase)


def steps_for_phase(
    steps: list[SynchronizedStep],
    dtw_phase: str,
) -> list[SynchronizedStep]:
    return [s for s in steps if s.phase == dtw_phase]


def format_details_text(
    rec: CoachingRecommendation,
    feature: FeatureSummary | None,
) -> str:
    lines = [
        f"{rec.feature}  [{rec.priority}]  {rec.phase}  ({rec.direction or '—'})",
        f"Correction: {rec.correction}",
    ]
    if rec.coach_quotes:
        lines.append("Quotes: " + " | ".join(rec.coach_quotes))
    if rec.practice_drills:
        lines.append("Drills: " + " | ".join(rec.practice_drills))
    if feature is None:
        lines.append("Measurements: unavailable")
    else:
        meta = feature.metadata
        lines.append(
            "Meas: "
            f"player={meta.get('player_value', '—')}  "
            f"ref={meta.get('reference_value', '—')}  "
            f"diff={meta.get('difference', '—')}  "
            f"score={feature.score:.1f}  "
            f"conf={meta.get('confidence', '—')}  "
            f"dir={meta.get('direction', '—')}"
        )
    return "\n".join(lines)


def format_scores_text(session: CoachingViewerSession) -> str:
    report = session.scoring_report
    parts = [f"Overall {report.overall_score:.1f} ({report.overall_grade})"]
    for phase in report.phase_summaries:
        parts.append(f"{phase.name} {phase.score:.0f}")
    return "  ·  ".join(parts)


def _emphasize_joints(
    ax,
    markers: dict,
    frame_idx: int,
    highlight: tuple[str, ...],
    colors: dict,
) -> None:
    """Overlay emphasis on top of a normal ``draw_skeleton`` render."""
    if not highlight:
        return
    for joint in highlight:
        if joint not in markers:
            continue
        x, y, z = get_pos(markers, joint, frame_idx)
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            continue
        color = colors.get(joint, "#E45756")
        ax.scatter(
            x,
            y,
            z,
            s=90,
            facecolors=color,
            edgecolors="black",
            linewidths=1.4,
            alpha=1.0,
            zorder=10,
        )


def run_coaching_viewer_app(
    session: CoachingViewerSession,
    *,
    interval_ms: int = 50,
) -> None:
    """DTW Viewer layout + playback, with a compact coaching strip beneath."""
    all_steps = list(session.comparison.synchronized_steps)
    if not all_steps:
        raise ValueError("No synchronized steps to animate")

    # Same panel convention as DTW Viewer: A = left (reference), B = right (player)
    # Source markers stay untouched for scoring; display copies are oriented only.
    markers_a = orient_ace_markers_for_display(session.markers_reference)
    markers_b = orient_ace_markers_for_display(session.markers_player)
    comparison = session.comparison

    # After body-centered orientation (+ uniform scale), share identical limits/camera.
    x_lim, y_lim, z_lim = _combined_limits(markers_a, markers_b)
    colors = _shared_marker_colors(markers_a, markers_b)
    full_spans = _phase_step_spans(all_steps)
    total_steps = len(all_steps)

    # DTW Viewer baseline is figsize=(14, 7) with height_ratios=[10, 1].
    # Add a short coaching strip under the same skeleton + timeline stack.
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[10, 1, 2.2],
        width_ratios=[1, 1],
        hspace=0.28,
        wspace=0.18,
        left=0.04,
        right=0.98,
        top=0.93,
        bottom=0.07,
    )
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    ax_bar = fig.add_subplot(gs[1, :])

    coach_gs = gs[2, :].subgridspec(2, 2, height_ratios=[1.0, 2.2], hspace=0.35, wspace=0.12)
    ax_scores = fig.add_subplot(coach_gs[0, :])
    ax_scores.axis("off")
    ax_recs = fig.add_subplot(coach_gs[1, 0])
    ax_recs.axis("off")
    ax_details = fig.add_subplot(coach_gs[1, 1])
    ax_details.axis("off")

    state: dict = {
        "play_steps": list(all_steps),
        "play_i": 0,
        "paused": False,
        "loop": True,
        "speed": 1.0,
        "rec_i": 0,
        "highlight": (),
        "mode": "whole",
        "active_phase": None,  # DTW phase name to emphasize on timeline
        "status": "",
    }

    scores_artist = ax_scores.text(
        0.0,
        0.5,
        format_scores_text(session),
        va="center",
        ha="left",
        fontsize=8,
        transform=ax_scores.transAxes,
    )
    detail_artist = ax_details.text(
        0.0,
        1.0,
        "",
        va="top",
        ha="left",
        fontsize=7.5,
        family="monospace",
        transform=ax_details.transAxes,
    )
    hint = fig.text(0.02, 0.01, "Space: pause  |  ←/→: step", fontsize=9)

    def current_interval() -> int:
        return max(1, int(interval_ms / float(state["speed"])))

    def draw_progress(global_step_idx: int) -> None:
        """DTW Viewer timeline over the full synchronized path."""
        ax_bar.cla()
        ax_bar.set_xlim(0, total_steps)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_yticks([])
        ax_bar.set_xlabel("DTW synchronized step (colored by phase)")

        bar_y, bar_h = 0.2, 0.6
        active = state["active_phase"]
        for start, end, phase in full_spans:
            width = max(end - start, 1)
            is_active = active is not None and phase == active
            ax_bar.add_patch(
                Rectangle(
                    (start, bar_y),
                    width,
                    bar_h,
                    facecolor=_phase_color(phase),
                    edgecolor="#222222" if is_active else "white",
                    linewidth=2.0 if is_active else 0.8,
                    alpha=0.85 if is_active else 0.55,
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

        if global_step_idx >= 0:
            ax_bar.add_patch(
                Rectangle(
                    (0, bar_y),
                    global_step_idx + 1,
                    bar_h,
                    facecolor="#333333",
                    alpha=0.25,
                )
            )
            ax_bar.axvline(global_step_idx + 0.5, color="#E45756", linewidth=2.5, zorder=5)

    def refresh_details(*, apply_highlight: bool) -> None:
        recs = session.recommendations
        if not recs:
            detail_artist.set_text("No coaching recommendations.")
            if apply_highlight:
                state["highlight"] = ()
            return
        idx = int(np.clip(state["rec_i"], 0, len(recs) - 1))
        rec = recs[idx]
        feature = session.feature_by_name.get(rec.feature)
        detail_artist.set_text(format_details_text(rec, feature))
        if apply_highlight:
            symbolic = highlight_joints(rec.feature)
            state["highlight"] = resolve_highlight_markers(symbolic, markers_b)

    def select_recommendation(idx: int, *, play: bool = True) -> None:
        recs = session.recommendations
        if not recs:
            return
        state["rec_i"] = int(np.clip(idx, 0, len(recs) - 1))
        rec = recs[state["rec_i"]]
        refresh_details(apply_highlight=True)
        dtw_phase = coaching_phase_to_dtw(rec.phase)
        if dtw_phase is None:
            state["play_steps"] = list(all_steps)
            state["mode"] = "whole"
            state["active_phase"] = None
            state["status"] = f"Unknown phase '{rec.phase}'"
        else:
            filtered = steps_for_phase(all_steps, dtw_phase)
            if not filtered:
                state["play_steps"] = list(all_steps)
                state["mode"] = "whole"
                state["active_phase"] = None
                state["status"] = f"No DTW steps for {dtw_phase}"
            else:
                state["play_steps"] = filtered
                state["mode"] = "recommendation"
                state["active_phase"] = dtw_phase
                state["status"] = f"{rec.feature} · {dtw_phase}"
        state["play_i"] = 0
        state["paused"] = not play
        _redraw_rec_labels()
        render_frame()

    def play_whole(_event=None) -> None:
        state["play_steps"] = list(all_steps)
        state["play_i"] = 0
        state["mode"] = "whole"
        state["active_phase"] = None
        state["highlight"] = ()
        state["paused"] = False
        state["status"] = "Whole serve"
        if session.recommendations:
            refresh_details(apply_highlight=False)
        _redraw_rec_labels()
        render_frame()

    def render_frame() -> None:
        play_steps = state["play_steps"]
        if not play_steps:
            return
        i = int(np.clip(state["play_i"], 0, len(play_steps) - 1))
        step = play_steps[i]

        ax_a.cla()
        ax_b.cla()
        # Display-oriented skeletons (source session markers unchanged)
        draw_skeleton(
            ax_a,
            markers_a,
            step.global_index_a,
            x_lim,
            y_lim,
            z_lim,
            colors,
            show_legend=False,
        )
        draw_skeleton(
            ax_b,
            markers_b,
            step.global_index_b,
            x_lim,
            y_lim,
            z_lim,
            colors,
            show_legend=False,
        )
        ax_a.view_init(elev=_VIEW_ELEV, azim=_VIEW_AZIM)
        ax_b.view_init(elev=_VIEW_ELEV, azim=_VIEW_AZIM)
        # Optional coaching emphasis only (does not replace DTW draw)
        _emphasize_joints(ax_a, markers_a, step.global_index_a, state["highlight"], colors)
        _emphasize_joints(ax_b, markers_b, step.global_index_b, state["highlight"], colors)

        # Same title pattern as DTW Viewer
        ax_a.set_title(f"{comparison.name_a}\nFrame {step.vicon_frame_a}")
        ax_b.set_title(f"{comparison.name_b}\nFrame {step.vicon_frame_b}")

        draw_progress(step.step_index)
        scores_artist.set_text(format_scores_text(session))

        pause_tag = "  |  PAUSED" if state["paused"] else ""
        fig.suptitle(
            f"Phase: {step.phase.replace('_', ' ')}  |  "
            f"DTW step {step.step_index + 1}/{total_steps}  |  "
            f"A[{step.global_index_a}] ↔ B[{step.global_index_b}]"
            f"{pause_tag}",
            fontsize=11,
        )
        hint.set_text(
            f"Space: pause  |  ←/→: step  |  {state['status']}  |  "
            f"speed {state['speed']}x  |  loop {state['loop']}"
        )

    def update(_frame: int) -> None:
        # Match DTW Viewer: skip drawing while paused
        if state["paused"]:
            return
        play_steps = state["play_steps"]
        if not play_steps:
            return
        render_frame()
        nxt = state["play_i"] + 1
        if nxt >= len(play_steps):
            if state["loop"]:
                state["play_i"] = 0
            else:
                state["play_i"] = len(play_steps) - 1
                state["paused"] = True
        else:
            state["play_i"] = nxt

    def on_key(event) -> None:
        if event.key == " ":
            state["paused"] = not state["paused"]
            render_frame()
            fig.canvas.draw_idle()
        elif event.key == "left":
            state["play_i"] = max(0, state["play_i"] - 1)
            state["paused"] = True
            render_frame()
            fig.canvas.draw_idle()
        elif event.key == "right":
            state["play_i"] = min(len(state["play_steps"]) - 1, state["play_i"] + 1)
            state["paused"] = True
            render_frame()
            fig.canvas.draw_idle()

    # Compact recommendation buttons in the coaching strip (not over skeletons)
    rec_buttons: list[Button] = []
    # Bottom coaching row occupies roughly y∈[0.07, 0.22]; place buttons there.
    for i in range(3):
        ax_btn = fig.add_axes([0.05, 0.095 + (2 - i) * 0.032, 0.28, 0.028])
        btn = Button(ax_btn, f"Rec {i + 1}")
        btn.label.set_fontsize(7)
        btn.on_clicked(lambda _e, idx=i: select_recommendation(idx))
        rec_buttons.append(btn)

    def _redraw_rec_labels() -> None:
        ax_recs.cla()
        ax_recs.axis("off")
        ax_recs.set_title("Top 3", loc="left", fontsize=9, pad=2)
        for i, btn in enumerate(rec_buttons):
            if i < len(session.recommendations):
                rec = session.recommendations[i]
                mark = "► " if i == state["rec_i"] and state["mode"] == "recommendation" else ""
                btn.label.set_text(f"{mark}{i + 1}. {rec.feature}")
                btn.ax.set_visible(True)
            else:
                btn.label.set_text("")
                btn.ax.set_visible(False)

    def _toggle_pause(_event=None) -> None:
        state["paused"] = not state["paused"]
        render_frame()
        fig.canvas.draw_idle()

    def _replay(_event=None) -> None:
        state["play_i"] = 0
        state["paused"] = False
        render_frame()
        fig.canvas.draw_idle()

    def _toggle_loop(_event=None) -> None:
        state["loop"] = not state["loop"]
        render_frame()
        fig.canvas.draw_idle()

    def _cycle_speed(_event=None) -> None:
        idx = _SPEEDS.index(state["speed"]) if state["speed"] in _SPEEDS else 1
        state["speed"] = _SPEEDS[(idx + 1) % len(_SPEEDS)]
        ani.event_source.interval = current_interval()
        render_frame()
        fig.canvas.draw_idle()

    def _prev_rec(_event=None) -> None:
        if session.recommendations:
            select_recommendation((state["rec_i"] - 1) % len(session.recommendations))

    def _next_rec(_event=None) -> None:
        if session.recommendations:
            select_recommendation((state["rec_i"] + 1) % len(session.recommendations))

    # Compact transport controls along the bottom edge
    ax_whole = fig.add_axes([0.36, 0.015, 0.08, 0.028])
    ax_pause = fig.add_axes([0.45, 0.015, 0.09, 0.028])
    ax_replay = fig.add_axes([0.55, 0.015, 0.07, 0.028])
    ax_loop = fig.add_axes([0.63, 0.015, 0.06, 0.028])
    ax_speed = fig.add_axes([0.70, 0.015, 0.06, 0.028])
    ax_prev = fig.add_axes([0.77, 0.015, 0.05, 0.028])
    ax_next = fig.add_axes([0.83, 0.015, 0.05, 0.028])

    b_whole = Button(ax_whole, "Whole")
    b_pause = Button(ax_pause, "Play/Pause")
    b_replay = Button(ax_replay, "Replay")
    b_loop = Button(ax_loop, "Loop")
    b_speed = Button(ax_speed, "Speed")
    b_prev = Button(ax_prev, "Prev")
    b_next = Button(ax_next, "Next")
    for btn in (b_whole, b_pause, b_replay, b_loop, b_speed, b_prev, b_next):
        btn.label.set_fontsize(7)

    b_whole.on_clicked(lambda _e: (play_whole(), fig.canvas.draw_idle()))
    b_pause.on_clicked(_toggle_pause)
    b_replay.on_clicked(_replay)
    b_loop.on_clicked(_toggle_loop)
    b_speed.on_clicked(_cycle_speed)
    b_prev.on_clicked(_prev_rec)
    b_next.on_clicked(_next_rec)
    _controls = (b_whole, b_pause, b_replay, b_loop, b_speed, b_prev, b_next, *rec_buttons)

    # Start like the DTW Viewer: whole-serve playback, no highlight
    play_whole()
    if session.recommendations:
        refresh_details(apply_highlight=False)
    else:
        detail_artist.set_text("No coaching recommendations.")
    _redraw_rec_labels()

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_steps * 20,
        interval=current_interval(),
        repeat=True,
        cache_frame_data=False,
    )
    fig.canvas.mpl_connect("key_press_event", on_key)
    render_frame()
    plt.show()
    return ani
