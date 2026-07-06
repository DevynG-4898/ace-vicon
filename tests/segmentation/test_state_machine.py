from segmentation.pipeline import segment_serve_v2
from segmentation.result import V2_PHASE_NAMES


def test_v2_phase_count(firstserve_dict, v2_config):
    result = segment_serve_v2(firstserve_dict, v2_config)
    assert len(result.phases) == len(V2_PHASE_NAMES)
    for name in V2_PHASE_NAMES:
        assert name in result.phases


def test_v2_release_ends_at_head_level(firstserve_dict, v2_config):
    result = segment_serve_v2(firstserve_dict, v2_config)
    rel = result.phases["Release"]
    loading = result.annotations["loading_start"]
    assert rel[1] == loading - 1
    assert loading < result.annotations["toss_apex"]


def test_release_shorter_than_full_toss(firstserve_dict, v2_config):
    result = segment_serve_v2(firstserve_dict, v2_config)
    rel = result.phases["Release"]
    dur = rel[1] - rel[0] + 1
    assert dur < 250
    assert dur >= 100


def test_acceleration_starts_before_max_shoulder_er(firstserve_dict, v2_config):
    result = segment_serve_v2(firstserve_dict, v2_config)
    accel_start = result.annotations["acceleration_start"]
    max_er = result.annotations["max_shoulder_ER"]
    assert accel_start < max_er


def test_cocking_ends_at_upswing(firstserve_dict, v2_config):
    result = segment_serve_v2(firstserve_dict, v2_config)
    cock = result.phases["Cocking"]
    accel = result.annotations["acceleration_start"]
    cock_dur = cock[1] - cock[0] + 1
    assert cock[1] == accel - 1
    assert cock_dur <= 50


def test_acceleration_runs_to_contact(firstserve_dict, v2_config):
    result = segment_serve_v2(firstserve_dict, v2_config)
    accel = result.phases["Acceleration"]
    contact = result.annotations["contact"]
    assert accel[1] == contact - 1
    assert accel[0] < contact


def test_cocking_starts_before_max_knee_bend(firstserve_dict, v2_config):
    result = segment_serve_v2(firstserve_dict, v2_config)
    assert result.annotations["cocking_start"] < result.annotations["max_knee_bend"]


def test_loading_and_cocking_are_bounded(firstserve_dict, v2_config):
    result = segment_serve_v2(firstserve_dict, v2_config)
    load_dur = result.phases["Loading"][1] - result.phases["Loading"][0] + 1
    assert 20 <= load_dur <= 80


def test_release_ends_before_loading(firstserve_dict, v2_config):
    result = segment_serve_v2(firstserve_dict, v2_config)
    release = result.phases["Release"]
    loading = result.phases["Loading"]
    assert release[1] < loading[0]


def test_v2_contact_confidence_high(firstserve_dict, v2_config):
    result = segment_serve_v2(firstserve_dict, v2_config)
    assert result.event_confidence["contact"] > 0.8
