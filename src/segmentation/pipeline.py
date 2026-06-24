from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from segmentation.anchors import detect_all_annotations, detect_contact, detect_finish
from segmentation.confidence import compute_v2_confidence
from segmentation.config import SegmentationConfig
from segmentation.io import load_serve_from_folder
from segmentation.legacy import segment_serve_legacy
from segmentation.result import SegmentationResult
from segmentation.signals import compute_all_signals
from segmentation.state_machine import (
    assign_coaching_phases,
    detect_coaching_boundaries,
    validate_phases,
)


def segment_serve_v2(serve: dict, config: SegmentationConfig | None = None) -> SegmentationResult:
    cfg = config or SegmentationConfig(use_legacy_detection=False)
    warnings: list[str] = []

    required = {cfg.serving_hand, cfg.serving_shoulder, cfg.serving_elbow, cfg.toss_hand, "chest"}
    missing = [m for m in required if m not in serve]
    if missing:
        warnings.append(f"missing markers: {missing}")

    signals = compute_all_signals(serve, cfg)
    if signals.get("knee_flexion_min_lr_deg") is None and signals.get("knee_flexion_deg") is None:
        warnings.append("knee angles unavailable — knee bend detection may be unreliable")

    validity = signals.get("marker_validity_mask")
    if validity is not None and float(np.nanmin(validity)) < (1.0 - cfg.marker_nan_threshold):
        warnings.append("marker dropout detected — review knee/leg signals")

    n = len(signals["body_velocity"])
    frames = serve["frames"].astype(int)

    # Contact and finish detectors (phase boundaries for Contact / Deceleration_Finish)
    contact_idx = detect_contact(signals, cfg, n)
    finish_idx = detect_finish(signals, cfg, contact_idx, n)

    # Posture-driven coaching phase transitions
    boundaries = detect_coaching_boundaries(signals, cfg, n, contact_idx, finish_idx)
    phases = assign_coaching_phases(boundaries, n, frames)
    validate_phases(phases, warnings, contact_single_frame_ok=True)

    # Metric annotations (extrema are NOT phase boundaries)
    anchor_result = detect_all_annotations(signals, cfg, n, boundaries.meta)
    ann_idx = dict(anchor_result.indices)
    ann_idx["contact"] = contact_idx
    ann_idx["finish"] = finish_idx

    annotations: dict[str, int | str | float | bool] = {
        "toss_apex": int(frames[ann_idx["toss_apex"]]),
        "max_knee_bend": int(frames[ann_idx["maximum_knee_bend"]]),
        "max_shoulder_ER": int(frames[ann_idx["maximum_shoulder_external_rotation"]]),
        "contact": int(frames[contact_idx]),
        "finish": int(frames[finish_idx]),
        "release_start": int(frames[boundaries.release_start]),
        "loading_start": int(frames[boundaries.loading_start]),
        "cocking_start": int(frames[boundaries.cocking_start]),
        "acceleration_start": int(frames[boundaries.acceleration_start]),
    }
    if "knee_bend_leg" in anchor_result.meta:
        annotations["knee_bend_leg"] = str(anchor_result.meta["knee_bend_leg"])
    if "toss_release" in anchor_result.meta and isinstance(anchor_result.meta["toss_release"], int):
        annotations["toss_release"] = int(frames[anchor_result.meta["toss_release"]])
    if "racket_apex" in anchor_result.meta and isinstance(anchor_result.meta["racket_apex"], int):
        annotations["racket_apex"] = int(frames[anchor_result.meta["racket_apex"]])

    # events dict: annotations + legacy aliases for downstream consumers
    events_vicon = {
        "toss_apex": annotations["toss_apex"],
        "maximum_knee_bend": annotations["max_knee_bend"],
        "maximum_shoulder_external_rotation": annotations["max_shoulder_ER"],
        "contact": annotations["contact"],
        "finish": annotations["finish"],
        "first_movement": annotations["release_start"],
    }

    confidence = compute_v2_confidence(signals, ann_idx, cfg, boundaries.meta)

    return SegmentationResult(
        phases=phases,
        events=events_vicon,
        event_indices=ann_idx,
        event_confidence=confidence,
        signals=signals,
        frames=frames,
        warnings=warnings,
        annotations=annotations,
        schema_version=2,
    )


def segment_serve(serve: dict, config: SegmentationConfig | None = None) -> SegmentationResult:
    cfg = config or SegmentationConfig()
    if cfg.use_legacy_detection:
        return segment_serve_legacy(serve, cfg)
    return segment_serve_v2(serve, cfg)


def segment_serve_folder(serve_dir: str | Path, config: SegmentationConfig | None = None) -> SegmentationResult:
    return segment_serve(load_serve_from_folder(serve_dir), config)


def validate_individual_serves(
    base_dir: str | Path | None = None,
    config: SegmentationConfig | None = None,
) -> list[dict[str, Any]]:
    base = Path(base_dir or Path(__file__).resolve().parent.parent.parent / "plotting" / "markers" / "individual")
    results = []
    for serve_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        out = segment_serve_folder(serve_dir, config)
        results.append(
            {
                "serve": serve_dir.name,
                "phases": out.phases,
                "events": out.events,
                "annotations": out.annotations,
                "confidence": out.event_confidence,
                "warnings": out.warnings,
                "schema_version": out.schema_version,
            }
        )
    return results
