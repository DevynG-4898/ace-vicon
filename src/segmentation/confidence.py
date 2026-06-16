from __future__ import annotations

import numpy as np

from segmentation.config import SegmentationConfig
from segmentation.signals import clip_index, racket_hand_velocity_series


def peak_prominence(series: np.ndarray, idx: int, invert: bool = False) -> float:
    val = series[idx]
    margin = np.nanmax(series) - np.nanmin(series)
    if margin < 1e-9:
        return 0.3
    if invert:
        return float((np.nanmax(series) - val) / margin)
    return float((val - np.nanmin(series)) / margin)


def compute_legacy_confidence(
    signals: dict[str, np.ndarray | None],
    events: dict[str, int],
    cfg: SegmentationConfig,
) -> dict[str, float]:
    body_v = signals["body_velocity"]
    hand_v = signals["hand_velocity"]
    hand_tz = signals["hand_tz"]
    knee = signals["knee_flexion_deg"]
    ser = signals["shoulder_er_proxy_deg"]

    baseline_n = min(cfg.baseline_frames, max(10, len(body_v) // 8))
    baseline = float(np.nanmedian(body_v[:baseline_n]))
    peak_body = float(np.nanmax(body_v))
    n = len(body_v)
    e1 = clip_index(events["first_movement"], n)
    move_strength = (body_v[e1] - baseline) / max(peak_body - baseline, 1e-9)

    return {
        "first_movement": float(np.clip(move_strength, 0, 1)),
        "peak_hand_height": peak_prominence(hand_tz, clip_index(events["peak_hand_height"], n)),
        "maximum_knee_bend": (
            peak_prominence(knee, clip_index(events["maximum_knee_bend"], len(knee)), invert=True)
            if knee is not None
            else 0.4
        ),
        "maximum_shoulder_external_rotation": peak_prominence(
            ser, clip_index(events["maximum_shoulder_external_rotation"], n)
        ),
        "peak_velocity": peak_prominence(hand_v, clip_index(events["peak_velocity"], n)),
        "sustained_velocity_decrease": float(
            np.clip(
                (
                    hand_v[clip_index(events["peak_velocity"], n)]
                    - hand_v[clip_index(events["sustained_velocity_decrease"], n)]
                )
                / max(hand_v[clip_index(events["peak_velocity"], n)], 1e-9),
                0,
                1,
            )
        ),
        "stabilization": float(
            np.clip(
                1.0 - body_v[clip_index(events["stabilization"], n)] / max(peak_body, 1e-9),
                0,
                1,
            )
        ),
    }


def compute_v2_confidence(
    signals: dict[str, np.ndarray | None],
    events: dict[str, int],
    cfg: SegmentationConfig,
    phase_meta: dict[str, int] | None = None,
) -> dict[str, float]:
    body_v = signals["body_velocity"]
    hand_v = racket_hand_velocity_series(signals)
    toss_h = signals["toss_hand_height"]
    knee = signals.get("knee_flexion_min_lr_deg")
    if knee is None:
        knee = signals.get("knee_flexion_deg")
    ser = signals["shoulder_er_proxy_deg"]
    init = signals["initiation_score"]

    baseline_n = min(cfg.baseline_frames, max(10, len(body_v) // 8))
    baseline = float(np.nanmedian(body_v[:baseline_n]))
    peak_body = float(np.nanmax(body_v))
    peak_hand = float(np.nanmax(hand_v))
    n = len(body_v)

    meta = phase_meta or {}
    release = clip_index(meta.get("release_start", events.get("first_movement", 0)), n)
    e2 = clip_index(events["toss_apex"], n)
    e3 = clip_index(events["maximum_knee_bend"], n)
    e5 = clip_index(events["contact"], n)
    e6 = clip_index(events["finish"], n)

    body_thresh = baseline + cfg.stance_velocity_threshold_ratio * max(peak_body - baseline, 1e-9)
    hand_v_toss = signals.get("left_hand_velocity")
    if hand_v_toss is None:
        hand_v_toss = signals["toss_hand_velocity"]
    release_strength = float(
        np.clip(
            max(body_v[release] / max(body_thresh, 1e-9), hand_v_toss[release] / max(body_thresh, 1e-9))
            - 1.0,
            0,
            1,
        )
    )

    knee_depth = 0.5
    loading = clip_index(meta.get("loading_start", release), n)
    if knee is not None and loading < e3:
        knee_depth = float(
            np.clip(
                (knee[loading] - knee[e3]) / max(knee[loading] - np.nanmin(knee[loading:e5 + 1]), 1e-9),
                0,
                1,
            )
        )

    return {
        "first_movement": release_strength,
        "toss_apex": peak_prominence(toss_h, e2),
        "maximum_knee_bend": (
            peak_prominence(knee, e3, invert=True) if knee is not None else 0.4
        ),
        "maximum_shoulder_external_rotation": peak_prominence(
            ser, clip_index(events["maximum_shoulder_external_rotation"], n)
        ),
        "contact": float(np.clip(hand_v[e5] / max(peak_hand, 1e-9), 0, 1)),
        "finish": float(
            np.clip(1.0 - body_v[e6] / max(peak_body, 1e-9), 0, 1)
        ),
    }
