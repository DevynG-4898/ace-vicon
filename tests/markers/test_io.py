import json

import numpy as np
import pytest

from src.markers.io import is_marker_dict, load_serve_markers, save_serve_markers
from src.motionbert.ace_adapter import motionbert_to_ace_markers
from src.motionbert.motionbert_runner import MOTIONBERT_JOINT_NAMES


def _sample_markers(frames: int = 4) -> dict:
    pose = np.zeros((frames, len(MOTIONBERT_JOINT_NAMES), 3), dtype=float)
    for frame in range(frames):
        for joint in range(len(MOTIONBERT_JOINT_NAMES)):
            pose[frame, joint] = [joint, frame, joint + frame]
    return motionbert_to_ace_markers(pose, scale=1.0)


def test_is_marker_dict_accepts_canonical_structure():
    assert is_marker_dict(_sample_markers())


def test_is_marker_dict_rejects_incomplete():
    assert not is_marker_dict({"frames": np.array([1, 2])})


def test_save_and_load_round_trip(tmp_path):
    markers = _sample_markers(frames=3)
    path = save_serve_markers(tmp_path, markers)
    loaded = load_serve_markers(path)
    assert np.array_equal(loaded["frames"], markers["frames"])
    assert np.allclose(loaded["head"]["TX"], markers["head"]["TX"])


def test_load_serve_markers_from_output_directory(tmp_path):
    markers = _sample_markers(frames=2)
    save_serve_markers(tmp_path, markers)
    loaded = load_serve_markers(tmp_path)
    assert len(loaded["frames"]) == 2


def test_load_serve_markers_passthrough_dict():
    markers = _sample_markers()
    assert load_serve_markers(markers) is markers


def test_load_serve_markers_from_vicon_folder(firstserve_dir):
    loaded = load_serve_markers(firstserve_dir)
    assert is_marker_dict(loaded)
    assert len(loaded["frames"]) > 0
