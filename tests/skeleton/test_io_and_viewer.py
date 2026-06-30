import json

import numpy as np

from src.motionbert.motionbert_runner import MOTIONBERT_JOINT_NAMES
from src.motionbert.view_3d import _display_pose_path
from src.skeleton.io import (
    load_motionbert_sequence,
    motionbert_sequence_from_arrays,
    save_skeleton_sequence,
)


def test_load_motionbert_sequence_preserves_shape_and_metadata(tmp_path):
    poses = np.zeros((3, len(MOTIONBERT_JOINT_NAMES), 3), dtype=float)
    confidence = np.full((3, len(MOTIONBERT_JOINT_NAMES)), 0.8, dtype=float)
    pose_path = tmp_path / "poses_3d.npy"
    confidence_path = tmp_path / "motionbert_input_2d.npy"
    metadata_path = tmp_path / "video_metadata.json"
    np.save(pose_path, poses)
    np.save(confidence_path, np.dstack([np.zeros_like(confidence), np.zeros_like(confidence), confidence]))
    metadata_path.write_text(json.dumps({"fps": 59.94, "video_path": "2d_video/serve.mp4"}))

    sequence = load_motionbert_sequence(pose_path)

    assert sequence.joint_positions.shape == (3, 17, 3)
    assert sequence.joint_confidence.shape == (3, 17)
    assert sequence.fps == 59.94
    assert sequence.source == "MotionBERT"
    assert sequence.metadata["source_file"] == str(pose_path)


def test_save_skeleton_sequence_writes_canonical_npz(tmp_path):
    poses = np.zeros((2, len(MOTIONBERT_JOINT_NAMES), 3), dtype=float)
    sequence = motionbert_sequence_from_arrays(
        poses,
        fps=30.0,
        source_file="poses_3d.npy",
    )

    path = save_skeleton_sequence(tmp_path, sequence)
    loaded = np.load(path, allow_pickle=False)

    assert path.name == "skeleton_sequence.npz"
    assert loaded["joint_positions"].shape == (2, 17, 3)
    assert loaded["joint_names"].tolist() == MOTIONBERT_JOINT_NAMES


def test_display_pose_path_uses_normalized_file_when_requested(tmp_path):
    raw = tmp_path / "poses_3d.npy"
    normalized = tmp_path / "skeleton_normalized.npz"
    raw.write_bytes(b"raw")
    normalized.write_bytes(b"normalized")

    assert _display_pose_path(raw, normalized=False) == raw
    assert _display_pose_path(raw, normalized=True) == normalized
