import pytest

from segmentation.pipeline import segment_serve_v2

# Metric annotations (not phase boundaries) — biomechanical validation frames
FIRSTSERVE_ANNOTATIONS = {
    "toss_apex": (325, 10),
    "max_knee_bend": (363, 5),
    "max_shoulder_ER": (395, 5),
    "contact": (398, 5),
    "finish": (549, 20),
}

FIRSTSERVE_TRANSITIONS = {
    "release_start": (61, 5),
    "loading_start": (248, 10),
    "cocking_start": (289, 10),
    "acceleration_start": (325, 12),
}


@pytest.mark.parametrize(
    "key,tolerance",
    [(k, v[1]) for k, v in FIRSTSERVE_ANNOTATIONS.items()],
)
def test_firstserve_v2_annotations(firstserve_dict, v2_config, key, tolerance):
    result = segment_serve_v2(firstserve_dict, v2_config)
    expected_frame = FIRSTSERVE_ANNOTATIONS[key][0]
    actual = result.annotations[key]
    assert abs(actual - expected_frame) <= tolerance, (
        f"{key}: expected F{expected_frame} ±{tolerance}, got F{actual}"
    )


@pytest.mark.parametrize(
    "key,tolerance",
    [(k, v[1]) for k, v in FIRSTSERVE_TRANSITIONS.items()],
)
def test_firstserve_v2_transitions(firstserve_dict, v2_config, key, tolerance):
    result = segment_serve_v2(firstserve_dict, v2_config)
    expected_frame = FIRSTSERVE_TRANSITIONS[key][0]
    actual = result.annotations[key]
    assert abs(actual - expected_frame) <= tolerance, (
        f"{key}: expected F{expected_frame} ±{tolerance}, got F{actual}"
    )
