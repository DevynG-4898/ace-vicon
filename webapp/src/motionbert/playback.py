"""Playback speed helpers for serve animations (interactive + GIF export)."""

from __future__ import annotations

from src.motionbert.skeleton_viz import DEFAULT_INTERVAL_MS

# Allowed interactive / CLI speed multipliers
ALLOWED_SPEEDS: tuple[int, ...] = (1, 2, 4, 8)

# Per-phase default speeds when reviewing (CLI --speed 1 or unset)
PHASE_DEFAULT_SPEED: dict[str, float] = {
    "Start_Stance": 4,
    "Release": 2,
    "Loading": 2,
    "Cocking": 2,
    "Acceleration": 1,
    "Contact": 1,
    "Deceleration": 2,
    "Finish": 4,
}

# Hold contact frame ~0.75s at 1x (33ms * 23)
CONTACT_PAUSE_DUPES: int = 23


def snap_speed(speed: float) -> int:
    """Snap to nearest allowed speed (1, 2, 4, 8)."""
    s = max(1.0, float(speed))
    best = ALLOWED_SPEEDS[0]
    for v in ALLOWED_SPEEDS:
        if s >= v:
            best = v
    return best


def speed_up(current: float) -> int:
    cur = snap_speed(current)
    idx = ALLOWED_SPEEDS.index(cur)
    return ALLOWED_SPEEDS[min(idx + 1, len(ALLOWED_SPEEDS) - 1)]


def speed_down(current: float) -> int:
    cur = snap_speed(current)
    idx = ALLOWED_SPEEDS.index(cur)
    return ALLOWED_SPEEDS[max(idx - 1, 0)]


def playback_interval_ms(speed: float) -> int:
    """Timer interval between animation frames (lower = faster)."""
    s = max(1.0, float(speed))
    return max(1, int(DEFAULT_INTERVAL_MS / s))


def resolve_view_speed(
    view_name: str,
    cli_speed: float | None,
    *,
    user_speed: float | None = None,
    user_adjusted: bool = False,
) -> int:
    """
    Effective speed for a view.

    - CLI --speed > 1 fixes speed for all views.
    - Keyboard adjustment overrides phase defaults until view changes.
    - Otherwise per-phase defaults apply (Full Serve stays 1x).
    """
    if cli_speed is not None and cli_speed > 1:
        return snap_speed(cli_speed)
    if user_adjusted and user_speed is not None:
        return snap_speed(user_speed)
    if view_name == "Full Serve":
        return 1
    return snap_speed(PHASE_DEFAULT_SPEED.get(view_name, 1))


def build_play_sequence(
    n_play: int,
    speed: float,
    view_name: str = "Full Serve",
    *,
    extra_stride: int = 1,
) -> list[int]:
    """
    Local frame indices (0 .. n_play-1) for FuncAnimation.

    Uses frame skipping for speed > 1. Contact view holds the contact frame.
    """
    if n_play <= 0:
        return [0]

    s = max(1, snap_speed(speed))
    stride = max(1, s, max(1, int(extra_stride)))
    seq = list(range(0, n_play, stride))
    if seq[-1] != n_play - 1:
        seq.append(n_play - 1)

    if view_name == "Contact":
        hold = CONTACT_PAUSE_DUPES
        contact_local = 0 if n_play == 1 else seq[len(seq) // 2]
        seq = [contact_local] * hold

    return seq


def gif_fps_and_stride(
    speed: float,
    base_fps: int = 30,
    base_stride: int = 1,
) -> tuple[int, int]:
    """GIF export: faster playback via higher FPS and optional frame skip."""
    s = max(1, snap_speed(speed))
    view_stride = s
    stride = max(1, base_stride, view_stride)
    fps = int(base_fps * s)
    return fps, stride


def format_playback_label(speed: float) -> str:
    s = snap_speed(speed)
    return f"{s}x"
