import json

import numpy as np

from src.motionbert.motionbert_runner import MOTIONBERT_JOINT_NAMES, run_motionbert_stage
from src.features.io import load_feature_sequence


def _poses_2d(frames: int = 4) -> tuple[np.ndarray, np.ndarray]:
    poses = np.zeros((frames, 33, 2), dtype=float)
    for frame in range(frames):
        for landmark in range(33):
            poses[frame, landmark] = [landmark / 32.0, 0.2 + frame * 0.01]
    confidence = np.full((frames, 33), 0.9, dtype=float)
    return poses, confidence


def test_motionbert_stage_saves_feature_sequence(tmp_path):
    poses, confidence = _poses_2d()
    np.save(tmp_path / "poses_2d.npy", poses)
    np.save(tmp_path / "poses_2d_confidence.npy", confidence)
    (tmp_path / "video_metadata.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "video_path": "2d_video/serve.mp4",
                "resolution": {"width": 640, "height": 480},
            }
        )
    )

    def fake_generator(_motionbert_2d, **_kwargs):
        pose = np.zeros((poses.shape[0], len(MOTIONBERT_JOINT_NAMES), 3), dtype=float)
        for frame in range(pose.shape[0]):
            pose[frame, MOTIONBERT_JOINT_NAMES.index("left_hip")] = [-0.5, 0.0, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("right_hip")] = [0.5, 0.0, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("pelvis")] = [0.0, 0.0, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("thorax")] = [0.0, 1.0, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("spine")] = [0.0, 0.5, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("neck")] = [0.0, 1.2, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("head")] = [0.0, 1.5, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("left_shoulder")] = [-0.5, 1.0, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("right_shoulder")] = [0.5, 1.0, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("left_elbow")] = [-1.0, 1.0, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("left_wrist")] = [-1.0, 0.5, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("right_elbow")] = [1.0, 1.0, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("right_wrist")] = [1.0 + frame * 0.1, 1.0, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("left_knee")] = [-0.5, -1.0, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("left_ankle")] = [-0.5, -2.0, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("right_knee")] = [0.5, -1.0, 0.0]
            pose[frame, MOTIONBERT_JOINT_NAMES.index("right_ankle")] = [0.5, -2.0, 0.0]
        return pose

    import src.motionbert.motionbert_runner as runner

    original = runner.generate_3d_pose
    runner.generate_3d_pose = fake_generator
    try:
        run_motionbert_stage(tmp_path, backend="geometric")
    finally:
        runner.generate_3d_pose = original

    feature_path = tmp_path / "feature_sequence.npz"
    features = load_feature_sequence(feature_path)

    assert feature_path.is_file()
    assert "right_hand_speed" in features.features
    assert features.metadata["source"] == "SkeletonSequence"
