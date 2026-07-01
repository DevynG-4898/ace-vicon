"""Timeline, debug signals, and phase-aware 3D animations for serve segmentation."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from segmentation import (  # noqa: E402
    EVENT_LABELS,
    EVENT_NAMES,
    PHASE_COLORS,
    PHASE_NAMES,
    VIEW_OPTIONS,
    V2_EVENT_LABELS,
    V2_EVENT_NAMES,
    V2_PHASE_NAMES,
    SegmentationConfig,
    SegmentationResult,
    phase_at_index,
    phase_to_index_range,
    segment_serve,
    view_index_range,
)

from playback import (  # noqa: E402
    build_play_sequence,
    format_playback_label,
    gif_fps_and_stride,
    playback_interval_ms,
    resolve_view_speed,
    speed_down,
    speed_up,
)
from skeleton_viz import (  # noqa: E402
    apply_axes,
    compute_axis_limits,
    draw_skeleton,
    load_serve_from_dir,
    marker_color_map,
    marker_names,
)

OUTPUT_ROOT = Path(__file__).resolve().parent / "segmentation_validation"

V2_VIEW_OPTIONS = ("Full Serve",) + V2_PHASE_NAMES


def segment_for_viz(markers: dict) -> SegmentationResult:
    """Biomechanics-first segmentation for visualization."""
    return segment_serve(markers, SegmentationConfig(use_legacy_detection=False))


def _phase_names(result: SegmentationResult) -> tuple[str, ...]:
    return V2_PHASE_NAMES if result.schema_version == 2 else PHASE_NAMES


def _event_names(result: SegmentationResult) -> tuple[str, ...]:
    return V2_EVENT_NAMES if result.schema_version == 2 else EVENT_NAMES


def _event_label(result: SegmentationResult, key: str) -> str:
    if result.schema_version == 2:
        return V2_EVENT_LABELS[key]
    return EVENT_LABELS[key]


def _view_options(result: SegmentationResult) -> tuple[str, ...]:
    return V2_VIEW_OPTIONS if result.schema_version == 2 else VIEW_OPTIONS


def _phase_color(name: str) -> str:
    if name == "Deceleration_Finish":
        return PHASE_COLORS["Deceleration"]
    return PHASE_COLORS[name]


def segmentation_to_dict(serve_name: str, result: SegmentationResult) -> dict[str, Any]:
    return {
        "serve": serve_name,
        "phases": {k: list(v) for k, v in result.phases.items()},
        "events": result.events,
        "event_indices": result.event_indices,
        "event_confidence": result.event_confidence,
        "warnings": result.warnings,
    }


def save_segmentation_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_timeline(
    serve_name: str,
    result: SegmentationResult,
    ax: plt.Axes | None = None,
    *,
    show_confidence: bool = True,
) -> plt.Figure:
    """Horizontal phase timeline with event markers and confidence annotations."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 3.2))
    else:
        fig = ax.figure

    frames = result.frames.astype(int)
    t0, t1 = int(frames[0]), int(frames[-1])
    span = max(t1 - t0, 1)

    y_phase = 0.55
    y_event = 0.22
    bar_h = 0.35

    for name in _phase_names(result):
        start_v, end_v = result.phases[name]
        left = (start_v - t0) / span
        width = max((end_v - start_v + 1) / span, 0.002)
        ax.barh(
            y_phase,
            width,
            left=left,
            height=bar_h,
            color=_phase_color(name),
            edgecolor="white",
            linewidth=0.5,
        )
        if width > 0.04:
            ax.text(
                left + width / 2,
                y_phase,
                name.replace("_", "\n"),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )

    event_colors = plt.colormaps["Dark2"].resampled(len(_event_names(result)))
    for i, key in enumerate(_event_names(result)):
        vf = result.events[key]
        x = (vf - t0) / span
        ax.axvline(x, color=event_colors(i), linestyle="--", linewidth=1, alpha=0.85)
        ax.plot(x, y_event, "v", color=event_colors(i), markersize=8)
        label = _event_label(result, key)
        conf = result.event_confidence.get(key, 0.0)
        txt = f"{label}\nF{vf}"
        if show_confidence:
            txt += f"\nconf={conf:.2f}"
        ax.text(
            x,
            y_event - 0.12,
            txt,
            ha="center",
            va="top",
            fontsize=6,
            rotation=0,
            color=event_colors(i),
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Normalized trial timeline (Vicon frames)")
    ax.set_title(f"{serve_name} — Serve phase timeline & detected events")
    phase_legend = [
        Patch(facecolor=_phase_color(n), label=n.replace("_", " ")) for n in _phase_names(result)
    ]
    ax.legend(handles=phase_legend, loc="upper right", fontsize=6, ncol=2, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_debug_signals(
    serve_name: str,
    result: SegmentationResult,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Stacked kinematic signals with vertical lines at detected events."""
    if ax is None:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    else:
        fig = ax.figure
        axes = [ax]

    frames = result.frames.astype(int)
    sig = result.signals
    panels = [
        ("Total body velocity", sig["body_velocity"], "#4C78A8"),
        ("Hand velocity (contact proxy)", sig["hand_velocity"], "#F58518"),
        ("Knee flexion (deg, min L/R)", sig["knee_flexion_deg"], "#54A24B"),
        ("Shoulder rotation proxy (deg)", sig["shoulder_er_proxy_deg"], "#B279A2"),
    ]

    if len(axes) == 1:
        axes = [axes]

    for ax_i, (title, series, color) in zip(axes, panels):
        if series is None:
            ax_i.text(0.5, 0.5, "Signal unavailable", ha="center", va="center", transform=ax_i.transAxes)
            ax_i.set_title(title)
            continue
        ax_i.plot(frames, series, color=color, linewidth=1.2)
        ax_i.set_ylabel(title, fontsize=9)
        ax_i.grid(True, alpha=0.3)
        for key in _event_names(result):
            idx = result.event_indices[key]
            vf = int(frames[idx])
            ax_i.axvline(vf, color="crimson", linestyle="--", linewidth=1.2, alpha=0.7)
            if ax_i is axes[0]:
                ax_i.text(
                    vf,
                    ax_i.get_ylim()[1],
                    _event_label(result, key).split()[0],
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=6,
                    color="crimson",
                )

    axes[-1].set_xlabel("Vicon frame")
    fig.suptitle(f"{serve_name} — Segmentation debug (detected events)", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _overlay_text(
    fig: plt.Figure,
    serve_name: str,
    view_name: str,
    frame_idx: int,
    vicon_frame: int,
    phase_name: str,
    phase_bounds: tuple[int, int],
    playback_speed: float = 1.0,
) -> None:
    start_v, end_v = phase_bounds
    lines = [
        f"Serve: {serve_name}",
        f"Phase: {phase_name.replace('_', ' ')}",
        f"Frame: {vicon_frame}",
        f"Phase range: {start_v}-{end_v}",
        f"Playback: {format_playback_label(playback_speed)}",
    ]
    if hasattr(fig, "_phase_overlay") and fig._phase_overlay:
        fig._phase_overlay.remove()
    fig._phase_overlay = fig.text(
        0.02,
        0.02,
        "\n".join(lines),
        transform=fig.transFigure,
        fontsize=10,
        va="bottom",
        ha="left",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
    )


def build_animation(
    markers: dict,
    result: SegmentationResult,
    serve_name: str,
    view_name: str,
    *,
    speed: float = 1.0,
    fig: plt.Figure | None = None,
    show_timeline: bool = True,
    gif_stride: int = 1,
) -> tuple[plt.Figure, animation.FuncAnimation]:
    """3D skeleton animation for full serve or a single phase."""
    frames = result.frames
    i0, i1 = view_index_range(frames, result.phases, view_name)
    n_play = i1 - i0 + 1
    if speed > 1:
        view_speed = resolve_view_speed(view_name, cli_speed=speed)
    else:
        view_speed = resolve_view_speed(view_name, cli_speed=None)
    play_sequence = build_play_sequence(
        n_play, view_speed, view_name, extra_stride=max(1, int(gif_stride))
    )
    interval_ms = playback_interval_ms(view_speed)
    x_lim, y_lim, z_lim = compute_axis_limits(markers)
    colors = marker_color_map(marker_names(markers))

    if fig is None:
        if show_timeline:
            fig = plt.figure(figsize=(12, 9))
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.22], hspace=0.25)
            ax3d = fig.add_subplot(gs[0], projection="3d")
            ax_tl = fig.add_subplot(gs[1])
            plot_timeline(serve_name, result, ax=ax_tl, show_confidence=False)
            cursor = [None]

            def _highlight_playhead(local_i: int) -> None:
                fi = i0 + local_i
                vf = int(frames[fi])
                t0, t1 = int(frames[0]), int(frames[-1])
                span = max(t1 - t0, 1)
                x = (vf - t0) / span
                if cursor[0] is not None:
                    cursor[0].remove()
                (cursor[0],) = ax_tl.plot([x, x], [0, 1], color="black", linewidth=2)

            playhead_fn = _highlight_playhead
        else:
            fig = plt.figure(figsize=(10, 7))
            ax3d = fig.add_subplot(111, projection="3d")
            playhead_fn = None
    else:
        ax3d = fig.gca() if fig.axes else fig.add_subplot(111, projection="3d")
        playhead_fn = None

    phase_bounds = (
        (int(frames[0]), int(frames[-1]))
        if view_name == "Full Serve"
        else result.phases[view_name]
    )

    def update(play_pos: int) -> None:
        local_i = play_sequence[play_pos]
        frame_idx = i0 + local_i
        ax3d.cla()
        draw_skeleton(ax3d, markers, frame_idx, x_lim, y_lim, z_lim, colors)
        pname = phase_at_index(frames, result.phases, frame_idx)
        vf = int(frames[frame_idx])
        ax3d.set_title(
            f"{serve_name} — {view_name}  ({play_pos + 1}/{len(play_sequence)})  Vicon frame {vf}"
        )
        _overlay_text(
            fig, serve_name, view_name, frame_idx, vf, pname, result.phases[pname], view_speed
        )
        if playhead_fn is not None:
            playhead_fn(local_i)

    ani = animation.FuncAnimation(
        fig, update, frames=len(play_sequence), interval=interval_ms, repeat=True
    )
    ani._playback_speed = view_speed  # type: ignore[attr-defined]
    return fig, ani


def save_animation_gif(
    fig: plt.Figure,
    ani: animation.FuncAnimation,
    path: Path,
    *,
    fps: int = 30,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = animation.PillowWriter(fps=fps)
        ani.save(str(path), writer=writer, dpi=100)
    except Exception:
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(str(path), writer=writer, dpi=100)
    plt.close(fig)


def run_interactive_viewer(
    serve_dir: str | Path,
    serve_name: str | None = None,
    *,
    cli_speed: float | None = None,
) -> None:
    """Interactive 3D viewer: radio buttons select Full Serve or any phase."""
    serve_dir = Path(serve_dir)
    serve_name = serve_name or serve_dir.name
    markers = load_serve_from_dir(str(serve_dir))
    result = segment_for_viz(markers)

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 0.25], hspace=0.3, wspace=0.2)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_tl = fig.add_subplot(gs[1, :])
    ax_radio = fig.add_subplot(gs[0, 1])
    plot_timeline(serve_name, result, ax=ax_tl)

    from matplotlib.widgets import RadioButtons

    cli_fixed = cli_speed is not None and cli_speed > 1
    state: dict[str, Any] = {
        "view": "Full Serve",
        "ani": None,
        "cursor": None,
        "cli_speed": cli_speed,
        "speed": resolve_view_speed("Full Serve", cli_speed if cli_fixed else None),
        "user_adjusted": cli_fixed,
        "play_pos": 0,
    }

    x_lim, y_lim, z_lim = compute_axis_limits(markers)
    colors = marker_color_map(marker_names(markers))
    frames = result.frames

    def current_speed() -> int:
        if cli_fixed:
            return resolve_view_speed(state["view"], state["cli_speed"])
        return resolve_view_speed(
            state["view"],
            None,
            user_speed=state["speed"],
            user_adjusted=state["user_adjusted"],
        )

    def get_range() -> tuple[int, int]:
        return view_index_range(frames, result.phases, state["view"])

    def update_playhead(local_i: int, i0: int) -> None:
        fi = i0 + local_i
        vf = int(frames[fi])
        t0, t1 = int(frames[0]), int(frames[-1])
        span = max(t1 - t0, 1)
        x = (vf - t0) / span
        if state["cursor"] is not None:
            state["cursor"].remove()
        (state["cursor"],) = ax_tl.plot([x, x], [0, 1], color="black", linewidth=2)

    def stop_animation() -> None:
        if state["ani"] is not None:
            state["ani"].event_source.stop()
            state["ani"] = None

    def start_animation() -> None:
        stop_animation()
        i0, i1 = get_range()
        n_play = i1 - i0 + 1
        spd = current_speed()
        play_sequence = build_play_sequence(n_play, spd, state["view"])
        interval = playback_interval_ms(spd)
        def update(play_pos: int) -> None:
            local_i = play_sequence[play_pos % len(play_sequence)]
            frame_idx = i0 + local_i
            ax3d.cla()
            draw_skeleton(ax3d, markers, frame_idx, x_lim, y_lim, z_lim, colors)
            pname = phase_at_index(frames, result.phases, frame_idx)
            vf = int(frames[frame_idx])
            ax3d.set_title(f"{serve_name} — {state['view']}  Vicon {vf}")
            _overlay_text(
                fig,
                serve_name,
                state["view"],
                frame_idx,
                vf,
                pname,
                result.phases[pname],
                spd,
            )
            update_playhead(local_i, i0)

        state["ani"] = animation.FuncAnimation(
            fig,
            update,
            frames=max(len(play_sequence), 1),
            interval=interval,
            repeat=True,
        )
        fig.canvas.draw_idle()

    def on_select(label: str) -> None:
        state["view"] = label
        if not cli_fixed:
            state["user_adjusted"] = False
            state["speed"] = resolve_view_speed(label, None)
        start_animation()

    def on_key(event) -> None:
        if event.key in ("+", "="):
            state["user_adjusted"] = True
            state["speed"] = speed_up(current_speed())
            start_animation()
        elif event.key == "-":
            state["user_adjusted"] = True
            state["speed"] = speed_down(current_speed())
            start_animation()
        elif event.key in ("r", "R"):
            state["user_adjusted"] = False
            state["speed"] = resolve_view_speed(
                state["view"], state["cli_speed"] if cli_fixed else None
            )
            start_animation()

    fig.canvas.mpl_connect("key_press_event", on_key)
    radio = RadioButtons(ax_radio, _view_options(result), active=0)
    radio.on_clicked(on_select)

    start_animation()
    help_txt = "Keys: + faster, - slower, R reset speed"
    if cli_fixed:
        help_txt += f"  (CLI locked at {format_playback_label(state['cli_speed'])})"
    fig.suptitle(f"ACE Serve Phase Viewer — {serve_name}  |  {help_txt}", fontsize=11)
    plt.show()


def generate_serve_validation_assets(
    serve_dir: Path,
    output_dir: Path | None = None,
    *,
    save_gifs: bool = True,
    fps: int = 30,
    gif_stride: int = 2,
    speed: float = 1.0,
) -> dict[str, Path]:
    """Write JSON, timeline PNG, debug PNG, and animations for one serve."""
    serve_name = serve_dir.name
    out = output_dir or (OUTPUT_ROOT / serve_name)
    out.mkdir(parents=True, exist_ok=True)

    markers = load_serve_from_dir(str(serve_dir))
    result = segment_for_viz(markers)
    paths: dict[str, Path] = {}

    json_path = out / "segmentation.json"
    save_segmentation_json(json_path, segmentation_to_dict(serve_name, result))
    paths["segmentation"] = json_path

    fig_tl = plot_timeline(serve_name, result)
    tl_path = out / "timeline.png"
    fig_tl.savefig(tl_path, dpi=150, bbox_inches="tight")
    plt.close(fig_tl)
    paths["timeline"] = tl_path

    fig_db = plot_debug_signals(serve_name, result)
    db_path = out / "debug_signals.png"
    fig_db.savefig(db_path, dpi=150, bbox_inches="tight")
    plt.close(fig_db)
    paths["debug"] = db_path

    if save_gifs:
        for view in _view_options(result):
            safe = view.replace(" ", "_")
            eff_speed = speed if speed > 1 else resolve_view_speed(view, cli_speed=None)
            export_fps, export_stride = gif_fps_and_stride(
                eff_speed, base_fps=fps, base_stride=gif_stride
            )
            fig, ani = build_animation(
                markers,
                result,
                serve_name,
                view,
                speed=eff_speed,
                show_timeline=False,
                gif_stride=export_stride,
            )
            gif_path = out / f"animation_{safe}.gif"
            save_animation_gif(fig, ani, gif_path, fps=export_fps)
            paths[f"animation_{safe}"] = gif_path

    summary_path = out / "phase_summary.txt"
    lines = [f"Serve: {serve_name}", ""]
    for name in _phase_names(result):
        a, b = result.phases[name]
        lines.append(f"{name}: Vicon frames {a}-{b}  ({b - a + 1} frames)")
    lines.append("")
    lines.append("Events:")
    for key in _event_names(result):
        lines.append(
            f"  {_event_label(result, key)}: frame {result.events[key]}  conf={result.event_confidence[key]:.2f}"
        )
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    paths["summary"] = summary_path

    return paths


def generate_all_individual_serves(
    base_dir: Path | None = None,
    *,
    save_gifs: bool = True,
    gif_stride: int = 2,
    fps: int = 30,
    speed: float = 1.0,
) -> list[dict[str, Any]]:
    base = base_dir or Path(__file__).resolve().parent / "markers" / "individual"
    report = []
    for serve_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        paths = generate_serve_validation_assets(
            serve_dir,
            save_gifs=save_gifs,
            gif_stride=gif_stride,
            fps=fps,
            speed=speed,
        )
        report.append({"serve": serve_dir.name, "output_dir": str(paths["segmentation"].parent), "files": {k: str(v) for k, v in paths.items()}})
    return report
