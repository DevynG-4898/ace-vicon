from segmentation.config import SegmentationConfig
from segmentation.pipeline import segment_serve, segment_serve_folder
from segmentation.result import EVENT_NAMES, PHASE_NAMES, SegmentationResult


def test_segment_serve_legacy_default(firstserve_dict):
    result = segment_serve(firstserve_dict)
    assert isinstance(result, SegmentationResult)
    assert result.schema_version == 1
    for name in PHASE_NAMES:
        assert name in result.phases
    for name in EVENT_NAMES:
        assert name in result.events


def test_segment_serve_v2_flag(firstserve_dict):
    cfg = SegmentationConfig(use_legacy_detection=False)
    result = segment_serve(firstserve_dict, cfg)
    assert result.schema_version == 2
    assert "toss_apex" in result.events
    assert "contact" in result.events


def test_segment_serve_folder(firstserve_dir):
    result = segment_serve_folder(firstserve_dir)
    assert isinstance(result, SegmentationResult)
    assert len(result.frames) > 0


def test_v2_signals_backward_compat_keys(firstserve_dict, v2_config):
    from segmentation.pipeline import segment_serve_v2

    result = segment_serve_v2(firstserve_dict, v2_config)
    for key in ("body_velocity", "hand_velocity", "knee_flexion_deg", "shoulder_er_proxy_deg", "hand_tz"):
        assert key in result.signals
