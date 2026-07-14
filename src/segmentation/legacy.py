from __future__ import annotations

import numpy as np

from segmentation.confidence import compute_legacy_confidence
from segmentation.config import SegmentationConfig
from segmentation.result import EVENT_NAMES, PHASE_NAMES, SegmentationResult
from segmentation.signals import (
    argmax_in_range,
    clip_index,
    compute_legacy_signals,
    persist_above,
    persist_below,
)
from segmentation.state_machine import validate_phases


def detect_events_legacy(signals: dict, cfg: SegmentationConfig) -> dict[str, int]:
    n = len(signals["body_velocity"])
    body_v = signals["body_velocity"]
    hand_v = signals["hand_velocity"]
    hand_tz = signals["hand_tz"]
    knee = signals["knee_flexion_deg"]
    ser = signals["shoulder_er_proxy_deg"]

    baseline_n = min(cfg.baseline_frames, max(10, n // 8))
    baseline = float(np.nanmedian(body_v[:baseline_n]))
    peak_body = float(np.nanmax(body_v))
    thresh_move = baseline + cfg.body_velocity_threshold_ratio * max(peak_body - baseline, 1e-6)

    e1 = persist_above(body_v, thresh_move, cfg.body_velocity_persist_frames, start=0)
    if e1 is None:
        e1 = int(np.nanargmax(body_v) * 0.15)

    search_end = min(n - 1, e1 + int(n * 0.65))
    e2 = argmax_in_range(hand_tz, e1, search_end, mode="max")

    if knee is not None:
        e3 = argmax_in_range(knee, e2, min(n - 1, e2 + int(n * 0.5)), mode="min")
    else:
        e3 = e2 + cfg.min_event_gap_frames

    e4 = argmax_in_range(ser, e3, min(n - 1, e3 + int(n * 0.45)), mode="max")
    e5 = argmax_in_range(hand_v, e4, n - 1, mode="max")

    peak_v = float(hand_v[e5])
    decel_thresh = peak_v * cfg.post_contact_velocity_fraction
    e6 = persist_below(
        hand_v,
        decel_thresh,
        cfg.velocity_decrease_persist_frames,
        start=min(e5 + 1, n - 1),
    )
    if e6 is None:
        e6 = min(e5 + cfg.min_event_gap_frames, n - 1)

    stab_thresh = baseline + cfg.stabilization_velocity_ratio * max(peak_body - baseline, 1e-6)
    e7 = persist_below(
        body_v,
        stab_thresh,
        cfg.stabilization_persist_frames,
        start=min(e6 + 1, n - 1),
    )
    if e7 is None:
        e7 = min(n - 1, e6 + cfg.stabilization_persist_frames)

    return {
        "first_movement": e1,
        "peak_hand_height": e2,
        "maximum_knee_bend": e3,
        "maximum_shoulder_external_rotation": e4,
        "peak_velocity": e5,
        "sustained_velocity_decrease": e6,
        "stabilization": e7,
    }


def enforce_order_legacy(events: dict[str, int], cfg: SegmentationConfig, n_frames: int) -> dict[str, int]:
    keys = EVENT_NAMES
    ordered = [clip_index(events[k], n_frames) for k in keys]
    gap = cfg.min_event_gap_frames
    for i in range(1, len(ordered)):
        ordered[i] = max(ordered[i], ordered[i - 1] + gap)
    if ordered[-1] >= n_frames:
        ordered[-1] = n_frames - 1
        for i in range(len(ordered) - 2, -1, -1):
            ordered[i] = min(ordered[i], ordered[i + 1] - gap)
        ordered[0] = max(0, ordered[0])
    return {k: clip_index(ordered[i], n_frames) for i, k in enumerate(keys)}


def indices_to_phases_legacy(
    events: dict[str, int], n: int, frame_ids: np.ndarray
) -> dict[str, tuple[int, int]]:
    e1 = events["first_movement"]
    e2 = events["peak_hand_height"]
    e3 = events["maximum_knee_bend"]
    e4 = events["maximum_shoulder_external_rotation"]
    e5 = events["peak_velocity"]
    e6 = events["sustained_velocity_decrease"]
    e7 = events["stabilization"]

    def f(i: int) -> int:
        return int(frame_ids[min(max(i, 0), n - 1)])

    phases_idx = {
        "Start_Stance": (0, max(0, e1 - 1)),
        "Release": (e1, max(e1, e2 - 1)),
        "Loading": (e2, max(e2, e3 - 1)),
        "Cocking": (e3, max(e3, e4 - 1)),
        "Acceleration": (e4, max(e4, e5 - 1)),
        "Contact": (e5, e5),
        "Deceleration": (min(e5 + 1, n - 1), max(e6, e7 - 1)),
        "Finish": (e7, n - 1),
    }

    return {name: (f(phases_idx[name][0]), f(phases_idx[name][1])) for name in PHASE_NAMES}


def segment_serve_legacy(serve: dict, config: SegmentationConfig | None = None) -> SegmentationResult:
    cfg = config or SegmentationConfig()
    warnings: list[str] = []

    required = {cfg.serving_hand, cfg.serving_shoulder, cfg.serving_elbow, "chest"}
    missing = [m for m in required if m not in serve]
    if missing:
        warnings.append(f"missing markers: {missing}")

    signals = compute_legacy_signals(serve, cfg)
    if signals["knee_flexion_deg"] is None:
        warnings.append("knee angles unavailable — using temporal fallback for knee bend event")

    raw_events = detect_events_legacy(signals, cfg)
    n_sig = len(signals["body_velocity"])
    events = enforce_order_legacy(raw_events, cfg, n_sig)

    frames = serve["frames"].astype(int)
    n = len(frames)
    phases = indices_to_phases_legacy(events, n, frames)
    validate_phases(phases, warnings)

    confidence = compute_legacy_confidence(signals, events, cfg)
    event_indices = dict(events)
    events_vicon = {k: int(frames[events[k]]) for k in events}

    return SegmentationResult(
        phases=phases,
        events=events_vicon,
        event_indices=event_indices,
        event_confidence=confidence,
        signals=signals,
        frames=frames,
        warnings=warnings,
        schema_version=1,
    )
