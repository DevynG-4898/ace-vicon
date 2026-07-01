import numpy as np

from src.motionbert.ace_adapter import (
    ACE_MARKER_NAMES,
    motionbert_to_ace_markers,
    save_ace_markers,
)
from src.motionbert.motionbert_runner import MOTIONBERT_JOINT_NAMES


def _sample_pose_3d(frames=5):
    pose = np.zeros((frames, len(MOTIONBERT_JOINT_NAMES), 3), dtype=float)
    for frame in range(frames):
        for joint in range(len(MOTIONBERT_JOINT_NAMES)):
            pose[frame, joint] = [joint, frame, joint + frame]
    return pose


def test_motionbert_to_ace_markers_returns_vicon_animation_contract():
    markers = motionbert_to_ace_markers(_sample_pose_3d())

    assert set(markers) == {"frames", *ACE_MARKER_NAMES}
    assert markers["frames"].tolist() == [1, 2, 3, 4, 5]
    for name in ACE_MARKER_NAMES:
        assert set(markers[name]) == {"TX", "TY", "TZ"}
        assert markers[name]["TX"].shape == (5,)
        assert markers[name]["TY"].shape == (5,)
        assert markers[name]["TZ"].shape == (5,)
        assert np.isfinite(markers[name]["TX"]).all()
        assert np.isfinite(markers[name]["TY"]).all()
        assert np.isfinite(markers[name]["TZ"]).all()


def test_motionbert_to_ace_markers_maps_key_joints_to_video_upright_axes():
    pose = _sample_pose_3d(frames=2)
    markers = motionbert_to_ace_markers(pose)

    head_idx = MOTIONBERT_JOINT_NAMES.index("head")
    right_wrist_idx = MOTIONBERT_JOINT_NAMES.index("right_wrist")

    assert markers["head"]["TX"][0] == pose[0, head_idx, 0]
    assert markers["right_hand"]["TY"][1] == pose[1, right_wrist_idx, 2]
    assert markers["right_hand"]["TZ"][1] == -pose[1, right_wrist_idx, 1]


def test_motionbert_to_ace_markers_puts_head_above_chest_on_z_axis():
    pose = np.zeros((1, len(MOTIONBERT_JOINT_NAMES), 3), dtype=float)
    head_idx = MOTIONBERT_JOINT_NAMES.index("head")
    chest_idx = MOTIONBERT_JOINT_NAMES.index("thorax")
    pose[0, head_idx] = [0.0, -1.0, 0.0]
    pose[0, chest_idx] = [0.0, 0.0, 0.0]

    markers = motionbert_to_ace_markers(pose)

    assert markers["head"]["TZ"][0] > markers["chest"]["TZ"][0]


def test_save_ace_markers_writes_reusable_npz(tmp_path):
    markers = motionbert_to_ace_markers(_sample_pose_3d(frames=3))

    out_path = save_ace_markers(tmp_path, markers)
    loaded = np.load(out_path)

    assert out_path.name == "ace_markers.npz"
    assert loaded["frames"].tolist() == [1, 2, 3]
    assert loaded["head_TX"].shape == (3,)
    assert loaded["right_hand_TZ"].shape == (3,)
