from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks

from segmentation.config import SegmentationConfig
from segmentation import signals as sig


@dataclass
class AnchorResult:
    """Metric extrema and contact/finish anchors (annotations, not phase boundaries)."""

    indices: dict[str, int] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)


def _normalize_segment(arr: np.ndarray, lo: int, hi: int) -> np.ndarray:
    segment = np.asarray(arr[lo : hi + 1], dtype=float)
    if segment.size == 0:
        return segment
    fill = float(np.nanmedian(segment)) if np.any(np.isfinite(segment)) else 0.0
    segment = np.nan_to_num(segment, nan=fill)
    lo_v, hi_v = float(segment.min()), float(segment.max())
    if hi_v - lo_v < 1e-9:
        return np.zeros_like(segment)
    return (segment - lo_v) / (hi_v - lo_v)


def _hand_velocity_peak_index(hand_v: np.ndarray, cfg: SegmentationConfig) -> int:
    peak_max = float(np.nanmax(hand_v))
    if peak_max < 1e-9:
        return int(np.nanargmax(hand_v))
    prominence = cfg.contact_prominence_fraction * peak_max
    peaks, _ = find_peaks(
        hand_v,
        distance=cfg.contact_peak_distance,
        prominence=prominence,
    )
    if len(peaks) == 0:
        return int(np.nanargmax(hand_v))
    return int(peaks[np.argmax(hand_v[peaks])])


def detect_contact(signals: dict, cfg: SegmentationConfig, n: int) -> int:
    """
    Contact: racket-hand height apex in the upswing window before hand-velocity peak.

    Ball strike occurs near maximum reach, not on the downswing when elbow extension
    and shoulder angular velocity peak later. Refine among frames within a height band
    of the window maximum using shoulder velocity, upper-body rotation, and elbow angle.
    """
    hand_v = sig.racket_hand_velocity_series(signals)
    hand_peak = _hand_velocity_peak_index(hand_v, cfg)

    hand_tz = signals.get("hand_tz")
    sh_v = signals.get("shoulder_velocity")
    upper_ang = signals.get("upper_body_angular_velocity")
    elbow = signals.get("elbow_extension_angle")
    if hand_tz is None or sh_v is None or upper_ang is None or elbow is None:
        return hand_peak

    window = cfg.contact_search_window_frames
    lo = max(0, hand_peak - window)
    hi = min(n - 1, hand_peak)

    tz_seg = hand_tz[lo : hi + 1]
    max_tz = float(np.nanmax(tz_seg))
    if max_tz < 1e-9:
        return hand_peak

    height_thresh = max_tz * cfg.contact_height_band_fraction
    candidates = np.where(tz_seg >= height_thresh)[0]
    if len(candidates) == 0:
        return lo + int(np.argmax(tz_seg))

    score = (
        cfg.contact_hand_height_weight * _normalize_segment(hand_tz, lo, hi)
        + cfg.contact_shoulder_velocity_weight * _normalize_segment(sh_v, lo, hi)
        + cfg.contact_upper_body_angular_weight * _normalize_segment(upper_ang, lo, hi)
        + cfg.contact_elbow_extension_weight * _normalize_segment(elbow, lo, hi)
    )
    best_rel = max(candidates, key=lambda c: score[c])
    return lo + int(best_rel)


def detect_finish(signals: dict, cfg: SegmentationConfig, e5: int, n: int) -> int:
    """Finish: sustained low body and hand velocity after contact."""
    body_v = signals["body_velocity"]
    hand_v = sig.racket_hand_velocity_series(signals)
    baseline_n = min(cfg.baseline_frames, max(10, n // 8))
    baseline = float(np.nanmedian(body_v[:baseline_n]))
    peak_body = float(np.nanmax(body_v))
    stab = baseline + cfg.finish_body_velocity_ratio * max(peak_body - baseline, 1e-6)
    peak_hand = float(hand_v[e5])
    hand_thresh = cfg.finish_racket_velocity_fraction * peak_hand
    start = min(e5 + cfg.finish_min_frames_after_contact, n - 1)

    for i in range(start, n - cfg.finish_body_persist_frames):
        body_run = all(body_v[i : i + cfg.finish_body_persist_frames] < stab)
        hand_run = all(hand_v[i : i + cfg.finish_hand_persist_frames] < hand_thresh)
        if body_run and hand_run:
            return i
    return n - 1


def detect_toss_apex(
    signals: dict, cfg: SegmentationConfig, release_start: int, contact: int, n: int
) -> int:
    """Annotation: main toss-hand peak after release (excludes early bounce micro-peaks)."""
    toss_h = signals["toss_hand_height"]
    search_end = max(
        release_start + cfg.min_event_gap_frames,
        min(n - 1, contact - cfg.min_event_gap_frames),
    )
    segment = toss_h[release_start : search_end + 1]
    if len(segment) < 3:
        return release_start + cfg.min_event_gap_frames

    baseline = float(np.nanmedian(toss_h[: max(release_start, 10)]))
    min_h = baseline * cfg.toss_min_height_ratio

    peaks, _ = find_peaks(
        segment,
        distance=cfg.toss_peak_distance,
        prominence=cfg.toss_peak_prominence_mm,
    )
    valid = [int(p) for p in peaks if segment[p] >= min_h]
    if valid:
        return release_start + max(valid, key=lambda p: segment[p])

    above = np.where(segment >= min_h)[0]
    if len(above):
        return release_start + int(above[np.argmax(segment[above])])

    return release_start + int(np.nanargmax(segment))


def detect_max_knee_bend(
    signals: dict, cfg: SegmentationConfig, loading_start: int, contact: int, n: int
) -> tuple[int, str]:
    """Annotation: deepest knee flexion between loading and contact."""
    knee = signals.get("knee_flexion_min_lr_deg")
    if knee is None:
        knee = signals.get("knee_flexion_deg")
    if knee is None:
        return loading_start + cfg.min_event_gap_frames, "unknown"
    lo = loading_start
    hi = max(lo, min(n - 1, contact - cfg.knee_pre_contact_offset))
    segment = knee[lo : hi + 1]
    rel = int(np.nanargmin(segment))
    idx = lo + rel
    knee_r = signals.get("knee_flexion_right_deg")
    knee_l = signals.get("knee_flexion_left_deg")
    leg = "right"
    if knee_r is not None and knee_l is not None:
        leg = "left" if knee_l[idx] < knee_r[idx] else "right"
    elif knee_l is not None:
        leg = "left"
    return idx, leg


def detect_max_shoulder_er(
    signals: dict, cfg: SegmentationConfig, cocking_start: int, contact: int, n: int
) -> int:
    """Annotation: max shoulder ER between cocking entry and contact."""
    ser = signals["shoulder_er_proxy_deg"]
    lo = cocking_start
    hi = max(lo, min(n - 1, contact - cfg.shoulder_pre_contact_offset))
    segment = ser[lo : hi + 1]
    return lo + int(np.nanargmax(segment))


def detect_toss_release_annotation(
    signals: dict, toss_apex: int, contact: int, n: int
) -> int | None:
    """Annotation: first sustained negative toss-hand height slope after apex."""
    toss_h = signals["toss_hand_height"]
    dz = np.diff(toss_h)
    dz = np.concatenate([dz, [dz[-1] if len(dz) else 0.0]])
    for i in range(toss_apex + 1, min(toss_apex + 30, contact)):
        if i + 2 < n and all(dz[i : i + 3] < -0.5):
            return i
    return None


def detect_racket_apex_annotation(
    signals: dict, loading_start: int, contact: int, n: int
) -> int | None:
    hand_tz = signals.get("hand_tz")
    if hand_tz is None:
        return None
    lo = loading_start
    hi = min(contact, n - 1)
    if hi <= lo:
        return None
    return lo + int(np.nanargmax(hand_tz[lo : hi + 1]))


def detect_all_annotations(
    signals: dict,
    cfg: SegmentationConfig,
    n: int,
    phase_meta: dict[str, int] | None = None,
) -> AnchorResult:
    """
    Detect metric annotations (not phase boundaries).

    Uses coaching phase transitions when available for search windows.
    """
    meta = phase_meta or {}
    release_start = meta.get("release_start", 0)
    loading_start = meta.get("loading_start", release_start + cfg.min_event_gap_frames)
    cocking_start = meta.get("cocking_start", loading_start + cfg.min_event_gap_frames)

    contact = detect_contact(signals, cfg, n)
    finish = detect_finish(signals, cfg, contact, n)
    toss_apex = detect_toss_apex(signals, cfg, release_start, contact, n)
    max_knee, knee_leg = detect_max_knee_bend(signals, cfg, loading_start, contact, n)
    max_ser = detect_max_shoulder_er(signals, cfg, cocking_start, contact, n)

    indices = {
        "toss_apex": toss_apex,
        "maximum_knee_bend": max_knee,
        "maximum_shoulder_external_rotation": max_ser,
        "contact": contact,
        "finish": finish,
    }

    ann_meta: dict[str, object] = {"knee_bend_leg": knee_leg}
    rel = detect_toss_release_annotation(signals, toss_apex, contact, n)
    if rel is not None:
        ann_meta["toss_release"] = rel
    racket = detect_racket_apex_annotation(signals, loading_start, contact, n)
    if racket is not None:
        ann_meta["racket_apex"] = racket

    return AnchorResult(indices=indices, meta=ann_meta)


def resolve_anchor_order(
    events: dict[str, int], cfg: SegmentationConfig, n_frames: int
) -> dict[str, int]:
    """Enforce temporal ordering of annotation indices."""
    keys = (
        "toss_apex",
        "maximum_knee_bend",
        "maximum_shoulder_external_rotation",
        "contact",
        "finish",
    )
    ordered = [sig.clip_index(events[k], n_frames) for k in keys]
    gap = cfg.min_event_gap_frames
    for i in range(1, len(ordered)):
        if ordered[i] <= ordered[i - 1] + gap - 1:
            ordered[i] = min(n_frames - 1, ordered[i - 1] + gap)
    return {k: sig.clip_index(ordered[i], n_frames) for i, k in enumerate(keys)}
