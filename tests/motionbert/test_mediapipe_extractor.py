import json
from types import SimpleNamespace

import numpy as np

from src.motionbert.mediapipe_extractor import (
    LANDMARK_COUNT,
    compute_confidence_stats,
    frame_landmarks_to_arrays,
    save_extraction_outputs,
)


def _fake_landmarks(count=LANDMARK_COUNT):
    return SimpleNamespace(
        landmark=[
            SimpleNamespace(x=i / count, y=(count - i) / count, visibility=0.5 + i / (2 * count))
            for i in range(count)
        ]
    )


def test_frame_landmarks_to_arrays_returns_xy_and_confidence():
    xy, confidence, debug = frame_landmarks_to_arrays(_fake_landmarks())

    assert xy.shape == (LANDMARK_COUNT, 2)
    assert confidence.shape == (LANDMARK_COUNT,)
    assert np.isfinite(xy).all()
    assert np.isfinite(confidence).all()
    assert debug[0]["visibility"] == confidence[0]


def test_frame_landmarks_to_arrays_accepts_tasks_landmark_list():
    landmarks = _fake_landmarks().landmark

    xy, confidence, debug = frame_landmarks_to_arrays(landmarks)

    assert xy.shape == (LANDMARK_COUNT, 2)
    assert np.isfinite(xy).all()
    assert np.isfinite(confidence).all()
    assert debug[-1]["visibility"] == confidence[-1]


def test_save_extraction_outputs_writes_arrays_debug_and_metadata(tmp_path):
    poses = np.ones((2, LANDMARK_COUNT, 2), dtype=float)
    confidence = np.full((2, LANDMARK_COUNT), 0.75, dtype=float)
    frames_debug = [[{"x": 1.0, "y": 1.0, "visibility": 0.75}] * LANDMARK_COUNT for _ in range(2)]
    metadata = {
        "video_name": "serve",
        "frame_count": 2,
        "fps": 30.0,
        "resolution": {"width": 640, "height": 480},
    }

    save_extraction_outputs(tmp_path, poses, confidence, frames_debug, metadata)

    assert np.load(tmp_path / "poses_2d.npy").shape == (2, LANDMARK_COUNT, 2)
    assert json.loads((tmp_path / "poses_2d.json").read_text())["frames"][0]["landmarks"][0]["x"] == 1.0
    saved_metadata = json.loads((tmp_path / "video_metadata.json").read_text())
    assert saved_metadata["frame_count"] == 2
    assert saved_metadata["landmark_count"] == LANDMARK_COUNT
    assert saved_metadata["confidence"]["mean"] == 0.75


def test_compute_confidence_stats_counts_missing_frames():
    confidence = np.array(
        [
            [0.9, 0.8, 0.7],
            [np.nan, np.nan, np.nan],
            [0.1, 0.2, 0.3],
        ],
        dtype=float,
    )

    stats = compute_confidence_stats(confidence)

    assert stats["missing_frames"] == 1
    assert stats["min"] == 0.1
    assert stats["max"] == 0.9
