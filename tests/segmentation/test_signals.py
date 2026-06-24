import numpy as np

from segmentation.config import SegmentationConfig
from segmentation.signals import compute_all_signals, compute_legacy_signals


def test_compute_all_signals_length(firstserve_dict):
    cfg = SegmentationConfig()
    signals = compute_all_signals(firstserve_dict, cfg)
    n = len(firstserve_dict["frames"])
    for key in (
        "hand_tz",
        "hand_velocity",
        "body_velocity",
        "shoulder_er_proxy_deg",
        "toss_hand_height",
        "toss_hand_velocity",
        "hip_descent",
        "elbow_extension_angle",
        "marker_validity_mask",
        "initiation_score",
        "racket_hand_velocity",
    ):
        assert key in signals
        assert signals[key] is not None
        assert len(signals[key]) == n


def test_toss_hand_height_differs_from_dominant_hand_tz(firstserve_dict):
    cfg = SegmentationConfig()
    signals = compute_all_signals(firstserve_dict, cfg)
    assert not np.allclose(signals["toss_hand_height"], signals["hand_tz"])


def test_legacy_signals_subset(firstserve_dict):
    cfg = SegmentationConfig()
    legacy = compute_legacy_signals(firstserve_dict, cfg)
    all_sig = compute_all_signals(firstserve_dict, cfg)
    for key in ("hand_tz", "hand_velocity", "body_velocity", "shoulder_er_proxy_deg"):
        assert np.allclose(legacy[key], all_sig[key])


def test_marker_validity_mask_range(firstserve_dict):
    cfg = SegmentationConfig()
    signals = compute_all_signals(firstserve_dict, cfg)
    mask = signals["marker_validity_mask"]
    assert np.nanmin(mask) >= 0.0
    assert np.nanmax(mask) <= 1.0


def test_coaching_signals_present(firstserve_dict):
    cfg = SegmentationConfig(use_legacy_detection=False)
    signals = compute_all_signals(firstserve_dict, cfg)
    n = len(firstserve_dict["frames"])
    for key in (
        "left_hand_height",
        "shoulder_height",
        "head_height",
        "shoulder_velocity",
        "upper_body_angular_velocity",
        "left_hand_velocity",
        "hip_velocity",
        "knee_flexion_derivative",
        "shoulder_er_derivative",
        "trunk_tilt_deg",
    ):
        assert key in signals
        assert signals[key] is not None
        assert len(signals[key]) == n
