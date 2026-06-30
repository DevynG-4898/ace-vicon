"""Ensure legacy segmentation output is unchanged after package extraction."""

from segmentation.config import SegmentationConfig
from segmentation.pipeline import segment_serve

LEGACY_FIRSTSERVE_EVENTS = {
    "first_movement": 76,
    "peak_hand_height": 398,
    "maximum_knee_bend": 444,
    "maximum_shoulder_external_rotation": 556,
    "peak_velocity": 561,
    "sustained_velocity_decrease": 572,
    "stabilization": 577,
}


def test_legacy_firstserve_events_unchanged(firstserve_dict):
    cfg = SegmentationConfig(use_legacy_detection=True)
    result = segment_serve(firstserve_dict, cfg)
    for name, frame in LEGACY_FIRSTSERVE_EVENTS.items():
        assert result.events[name] == frame, f"{name}: expected {frame}, got {result.events[name]}"
