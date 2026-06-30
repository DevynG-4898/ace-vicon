import json

import numpy as np

from src.motionbert.alphapose_adapter import HALPE_26_NAMES, save_alphapose_json


def test_save_alphapose_json_writes_halpe26_keypoints_in_pixels(tmp_path):
    poses_2d = np.zeros((2, 33, 2), dtype=float)
    confidence = np.full((2, 33), 0.8, dtype=float)
    for frame in range(2):
        for landmark in range(33):
            poses_2d[frame, landmark] = [landmark / 32.0, frame / 2.0]

    out = save_alphapose_json(
        tmp_path,
        poses_2d,
        confidence,
        width=640,
        height=480,
    )

    payload = json.loads(out.read_text())
    assert out.name == "alphapose_halpe26.json"
    assert len(payload) == 2
    assert payload[0]["idx"] == 0
    assert len(payload[0]["keypoints"]) == 26 * 3
    assert payload[0]["keypoint_names"] == HALPE_26_NAMES
    assert payload[0]["keypoints"][0] == 0.0
    assert payload[0]["keypoints"][1] == 0.0
    assert payload[0]["keypoints"][2] == 0.8


def test_save_alphapose_json_uses_derived_neck_and_hip_points(tmp_path):
    poses_2d = np.zeros((1, 33, 2), dtype=float)
    confidence = np.ones((1, 33), dtype=float)
    poses_2d[0, 11] = [0.25, 0.2]
    poses_2d[0, 12] = [0.75, 0.4]
    poses_2d[0, 23] = [0.3, 0.8]
    poses_2d[0, 24] = [0.7, 0.6]

    out = save_alphapose_json(tmp_path, poses_2d, confidence, width=100, height=100)
    keypoints = json.loads(out.read_text())[0]["keypoints"]
    neck_idx = HALPE_26_NAMES.index("Neck") * 3
    hip_idx = HALPE_26_NAMES.index("Hip") * 3

    assert keypoints[neck_idx : neck_idx + 3] == [50.0, 30.0, 1.0]
    assert keypoints[hip_idx : hip_idx + 3] == [50.0, 70.0, 1.0]
