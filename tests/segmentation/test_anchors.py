import numpy as np

from segmentation.anchors import detect_all_annotations, detect_contact
from segmentation.config import SegmentationConfig
from segmentation.signals import compute_all_signals, racket_hand_velocity_series


def test_annotation_ordering(firstserve_dict):
    cfg = SegmentationConfig(use_legacy_detection=False)
    signals = compute_all_signals(firstserve_dict, cfg)
    n = len(firstserve_dict["frames"])
    result = detect_all_annotations(signals, cfg, n)
    idx = result.indices
    keys = (
        "toss_apex",
        "maximum_knee_bend",
        "maximum_shoulder_external_rotation",
        "contact",
        "finish",
    )
    for i in range(len(keys) - 1):
        assert idx[keys[i]] < idx[keys[i + 1]]


def test_contact_at_racket_height_apex(firstserve_dict):
    cfg = SegmentationConfig(use_legacy_detection=False)
    signals = compute_all_signals(firstserve_dict, cfg)
    n = len(firstserve_dict["frames"])
    contact = detect_contact(signals, cfg, n)
    hand_v = racket_hand_velocity_series(signals)
    hand_peak = int(np.argmax(hand_v))
    hand_tz = signals["hand_tz"]
    lo = max(0, hand_peak - cfg.contact_search_window_frames)
    hi = hand_peak
    height_peak = lo + int(np.argmax(hand_tz[lo : hi + 1]))
    assert abs(contact - height_peak) <= 3
    assert contact <= hand_peak
