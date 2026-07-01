from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from segmentation.anchors import detect_toss_apex
from segmentation.config import SegmentationConfig
from segmentation.result import V2_PHASE_NAMES
from segmentation.signals import clip_index


@dataclass
class CoachingPhaseBoundaries:
    """Index boundaries for the seven coaching phases (inclusive)."""

    release_start: int = 0
    loading_start: int = 0
    cocking_start: int = 0
    acceleration_start: int = 0
    contact: int = 0
    finish: int = 0
    meta: dict[str, int] = field(default_factory=dict)


def _baseline_threshold(series: np.ndarray, cfg: SegmentationConfig) -> float:
    baseline_n = min(cfg.baseline_frames, max(10, len(series) // 8))
    baseline = float(np.nanmedian(series[:baseline_n]))
    peak = float(np.nanmax(series))
    return baseline + cfg.stance_velocity_threshold_ratio * max(peak - baseline, 1e-6)


def detect_release_start(signals: dict, cfg: SegmentationConfig, n: int) -> int:
    """
    Phase 1 → 2: first sustained serve motion.

    left_hand_velocity OR body_velocity above baseline for N consecutive frames.
    """
    body_v = signals["body_velocity"]
    hand_v = signals["left_hand_velocity"]
    body_thresh = _baseline_threshold(body_v, cfg)
    hand_thresh = _baseline_threshold(hand_v, cfg)
    persist = cfg.stance_motion_persist_frames

    run = 0
    for i in range(n):
        if body_v[i] > body_thresh or hand_v[i] > hand_thresh:
            run += 1
            if run >= persist:
                return i - persist + 1
        else:
            run = 0
    return 0


def _toss_hand_at_head_level(
    signals: dict, cfg: SegmentationConfig, release_start: int, contact: int, n: int
) -> int | None:
    """First sustained frame where toss-hand trunk elevation reaches head level."""
    hand_h = signals.get("left_hand_height")
    head_h = signals.get("head_height")
    if hand_h is None or head_h is None:
        return None
    persist = cfg.release_head_level_persist_frames
    search_end = min(n, contact - cfg.min_event_gap_frames)
    for i in range(release_start + 1, search_end - persist + 1):
        segment_hand = hand_h[i : i + persist]
        segment_head = head_h[i : i + persist]
        if np.any(np.isnan(segment_head)):
            continue
        if np.all(segment_hand >= segment_head):
            return i
    return None


def detect_loading_start(
    signals: dict,
    cfg: SegmentationConfig,
    release_start: int,
    contact: int,
    n: int,
) -> int:
    """
    Phase 2 → 3: leg loading begins once the toss hand reaches head level.

    Release covers ball bounce and toss initiation up to head height; loading
    starts when the toss arm has reached the trophy/toss position at the head.
    """
    head_level = _toss_hand_at_head_level(signals, cfg, release_start, contact, n)
    if head_level is not None:
        return max(head_level, release_start + cfg.min_event_gap_frames)

    # Fallback when head marker is unavailable: end release at main toss apex.
    apex = detect_toss_apex(signals, cfg, release_start, contact, n)
    return max(apex, release_start + cfg.min_event_gap_frames)


def _trophy_posture_entry(
    signals: dict, cfg: SegmentationConfig, loading_start: int, search_end: int, n: int
) -> int | None:
    """First sustained trophy posture while building into the loaded position."""
    ser = signals["shoulder_er_proxy_deg"]
    tilt = signals["trunk_tilt_deg"]
    elbow = signals.get("elbow_extension_angle")
    persist = cfg.cocking_entry_persist_frames
    for i in range(loading_start + cfg.min_event_gap_frames, min(search_end, n - persist) + 1):
        er_ok = all(ser[i : i + persist] >= cfg.cocking_shoulder_er_threshold_deg)
        tilt_ok = all(tilt[i : i + persist] >= cfg.cocking_trunk_tilt_threshold_deg)
        laid_back_ok = True
        if elbow is not None:
            laid_back_ok = all(elbow[i : i + persist] <= cfg.cocking_elbow_extension_max_deg)
        if er_ok and tilt_ok and laid_back_ok:
            return i
    return None


def detect_cocking_start(
    signals: dict, cfg: SegmentationConfig, loading_start: int, contact: int, n: int
) -> int:
    """
    Phase 3 → 4: final stretch of the loaded position begins.

    Loading builds the general coil (leg drive into trophy posture). Cocking
    starts once most of the knee flexion is established, then covers the
    deeper stretch through max knee and shoulder ER until acceleration.
    """
    max_knee = _max_knee_index(signals, loading_start, contact, cfg)
    knee = signals.get("knee_flexion_min_lr_deg")
    if knee is None:
        knee = signals.get("knee_flexion_deg")
    if knee is None:
        trophy = _trophy_posture_entry(
            signals, cfg, loading_start, contact - cfg.shoulder_pre_contact_offset, n
        )
        return trophy if trophy is not None else max(loading_start + cfg.min_event_gap_frames, max_knee)

    k0 = float(knee[loading_start])
    k_min = float(knee[max_knee])
    depth = k0 - k_min
    min_loading_end = loading_start + cfg.loading_min_frames

    if depth >= cfg.cocking_min_knee_depth_deg:
        target = k0 - cfg.cocking_knee_depth_fraction * depth
        for i in range(loading_start + cfg.min_event_gap_frames, max_knee + 1):
            if knee[i] <= target and i >= min_loading_end:
                return i

    trophy = _trophy_posture_entry(signals, cfg, min_loading_end, max_knee, n)
    if trophy is not None:
        return trophy
    return max(min_loading_end, max_knee)


def _max_knee_index(
    signals: dict, loading_start: int, contact: int, cfg: SegmentationConfig
) -> int:
    knee = signals.get("knee_flexion_min_lr_deg")
    if knee is None:
        knee = signals.get("knee_flexion_deg")
    if knee is None:
        return loading_start
    lo = loading_start
    hi = max(lo, contact - cfg.knee_pre_contact_offset)
    return lo + int(np.nanargmin(knee[lo : hi + 1]))


def _racket_stretch_minimum(
    signals: dict, cfg: SegmentationConfig, cocking_start: int, contact: int
) -> int:
    """Deepest racket drop during cocking — bottom of the stretch before upswing."""
    hand_tz = signals["hand_tz"]
    lo = cocking_start + cfg.acceleration_min_frames_after_cocking
    hi = max(lo, contact - cfg.min_event_gap_frames)
    if hi <= cocking_start:
        lo = cocking_start
        hi = max(lo, contact - cfg.min_event_gap_frames)
    return lo + int(np.nanargmin(hand_tz[lo : hi + 1]))


def _detect_upswing_start(
    signals: dict, cfg: SegmentationConfig, stretch_min: int, contact: int, n: int
) -> int | None:
    """Racket leaves max stretch and moves upward toward the ball."""
    hand_tz = signals["hand_tz"]
    hand_v = signals["racket_hand_velocity"]
    min_val = float(hand_tz[stretch_min])
    persist = cfg.acceleration_upswing_persist_frames
    rise = cfg.acceleration_hand_rise_mm
    min_v = cfg.acceleration_min_hand_velocity
    search_end = max(stretch_min + 1, contact - cfg.min_event_gap_frames)

    for i in range(stretch_min + 1, min(search_end, n - persist) + 1):
        diffs = np.diff(hand_tz[i - 1 : i + persist])
        if not np.all(diffs > 0):
            continue
        if hand_tz[i + persist - 1] >= min_val + rise:
            return i
        if hand_v[i + persist - 1] >= min_v:
            return i
    return None


def _detect_er_forward_swing(
    signals: dict, cfg: SegmentationConfig, cocking_start: int, contact: int, n: int
) -> int | None:
    """Fallback: shoulder ER stops increasing (IR / forward swing onset)."""
    ser = signals["shoulder_er_proxy_deg"]
    dser = signals["shoulder_er_derivative"]
    persist = cfg.acceleration_er_derivative_persist_frames
    search_end = max(cocking_start + 1, contact - cfg.min_event_gap_frames)

    for i in range(cocking_start + 1, min(search_end, n - persist)):
        if ser[i] < cfg.acceleration_min_shoulder_er_deg:
            continue
        was_increasing = dser[i - 1] > 0 if i > 0 else False
        now_decreasing = all(dser[i : i + persist] < 0)
        if was_increasing and now_decreasing:
            return i
    for i in range(cocking_start + 1, min(search_end, n - persist)):
        if ser[i] >= cfg.acceleration_min_shoulder_er_deg and all(dser[i : i + persist] < 0):
            return i
    return None


def detect_acceleration_start(
    signals: dict,
    cfg: SegmentationConfig,
    cocking_start: int,
    contact: int,
    n: int,
    loading_start: int = 0,
) -> int:
    """
    Phase 4 → 5: upswing from max stretch toward contact.

    Cocking ends when the racket leaves the stretch bottom; acceleration
    continues through the forward swing until the contact frame.
    """
    stretch_min = _racket_stretch_minimum(signals, cfg, cocking_start, contact)
    upswing = _detect_upswing_start(signals, cfg, stretch_min, contact, n)
    if upswing is not None:
        return upswing

    er_swing = _detect_er_forward_swing(signals, cfg, cocking_start, contact, n)
    if er_swing is not None:
        return er_swing

    return min(n - 1, max(stretch_min + 1, cocking_start + cfg.min_event_gap_frames))


def detect_coaching_boundaries(
    signals: dict,
    cfg: SegmentationConfig,
    n: int,
    contact: int,
    finish: int,
) -> CoachingPhaseBoundaries:
    """Detect posture-driven phase transition indices."""
    t_release = detect_release_start(signals, cfg, n)
    t_loading = detect_loading_start(signals, cfg, t_release, contact, n)
    t_cocking = detect_cocking_start(signals, cfg, t_loading, contact, n)
    t_accel = detect_acceleration_start(signals, cfg, t_cocking, contact, n, t_loading)

    # Enforce monotonic ordering before contact
    ordered = [t_release, t_loading, t_cocking, t_accel, contact]
    gap = cfg.min_event_gap_frames
    for i in range(1, len(ordered)):
        if ordered[i] <= ordered[i - 1]:
            ordered[i] = min(n - 1, ordered[i - 1] + gap)
    t_release, t_loading, t_cocking, t_accel, contact = ordered
    t_accel = min(t_accel, max(t_cocking + 1, contact - 1))

    return CoachingPhaseBoundaries(
        release_start=clip_index(t_release, n),
        loading_start=clip_index(t_loading, n),
        cocking_start=clip_index(t_cocking, n),
        acceleration_start=clip_index(t_accel, n),
        contact=clip_index(contact, n),
        finish=clip_index(finish, n),
        meta={
            "release_start": clip_index(t_release, n),
            "loading_start": clip_index(t_loading, n),
            "cocking_start": clip_index(t_cocking, n),
            "acceleration_start": clip_index(t_accel, n),
        },
    )


def assign_coaching_phases(
    boundaries: CoachingPhaseBoundaries, n: int, frame_ids: np.ndarray
) -> dict[str, tuple[int, int]]:
    """Map coaching transition indices to seven phase Vicon frame bounds."""
    t0 = 0
    t1 = boundaries.release_start
    t2 = boundaries.loading_start
    t3 = boundaries.cocking_start
    t4 = boundaries.acceleration_start
    t5 = boundaries.contact
    t6 = boundaries.finish

    def f(i: int) -> int:
        return int(frame_ids[min(max(i, 0), n - 1)])

    phases_idx = {
        "Start_Stance": (t0, max(0, t1 - 1)),
        "Release": (t1, max(t1, t2 - 1)),
        "Loading": (t2, max(t2, t3 - 1)),
        "Cocking": (t3, max(t3, t4 - 1)),
        "Acceleration": (t4, max(t4, t5 - 1)),
        "Contact": (t5, t5),
        "Deceleration_Finish": (min(n - 1, t5 + 1), max(t6, t5 + 1)),
    }
    return {name: (f(phases_idx[name][0]), f(phases_idx[name][1])) for name in V2_PHASE_NAMES}


def validate_phases(
    phases: dict[str, tuple[int, int]], warnings: list[str], contact_single_frame_ok: bool = False
) -> None:
    names = list(phases.keys())
    for name in names:
        a, b = phases[name]
        if a > b:
            warnings.append(f"{name}: invalid range ({a}, {b})")
    for i in range(len(names) - 1):
        cur = phases[names[i]]
        nxt = phases[names[i + 1]]
        if cur[1] >= nxt[0] and not (contact_single_frame_ok and names[i] == "Contact"):
            if names[i] == "Contact" and names[i + 1] == "Deceleration_Finish":
                if cur[1] + 1 >= nxt[0]:
                    continue
            warnings.append(f"overlap or disorder between {names[i]} and {names[i + 1]}")
