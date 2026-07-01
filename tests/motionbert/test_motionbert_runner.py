import json
import shlex
from pathlib import Path

import numpy as np

from src.motionbert.motionbert_runner import (
    MOTIONBERT_JOINT_NAMES,
    SKELETON_EDGES,
    _run_external_motionbert,
    convert_mediapipe_to_motionbert,
    generate_3d_pose,
    resolve_motionbert_command,
    save_3d_outputs,
)


def _sample_mediapipe_poses(frames=4):
    poses = np.zeros((frames, 33, 2), dtype=float)
    for frame in range(frames):
        for landmark in range(33):
            poses[frame, landmark, 0] = landmark / 32
            poses[frame, landmark, 1] = frame / max(frames - 1, 1)
    confidence = np.full((frames, 33), 0.8, dtype=float)
    return poses, confidence


def test_convert_mediapipe_to_motionbert_returns_17_joint_input():
    poses, confidence = _sample_mediapipe_poses()

    converted = convert_mediapipe_to_motionbert(poses, confidence)

    assert converted.shape == (4, 17, 3)
    assert np.isfinite(converted).all()
    assert converted[..., 2].min() == 0.8


def test_generate_3d_pose_returns_finite_joint_coordinates():
    poses, confidence = _sample_mediapipe_poses()
    converted = convert_mediapipe_to_motionbert(poses, confidence)

    pose_3d = generate_3d_pose(converted, backend="geometric")

    assert pose_3d.shape == (4, 17, 3)
    assert np.isfinite(pose_3d).all()


def test_save_3d_outputs_writes_numpy_and_debug_json(tmp_path):
    pose_3d = np.zeros((3, 17, 3), dtype=float)
    confidence = np.ones((3, 17), dtype=float)

    save_3d_outputs(tmp_path, pose_3d, confidence, source_backend="geometric")

    assert np.load(tmp_path / "poses_3d.npy").shape == (3, 17, 3)
    payload = json.loads((tmp_path / "poses_3d.json").read_text())
    assert payload["shape"] == [3, 17, 3]
    assert payload["joint_names"] == MOTIONBERT_JOINT_NAMES
    assert payload["source_backend"] == "geometric"


def test_skeleton_topology_references_valid_joint_names():
    valid = set(MOTIONBERT_JOINT_NAMES)

    assert SKELETON_EDGES
    for start, end in SKELETON_EDGES:
        assert start in valid
        assert end in valid


def test_resolve_motionbert_command_requires_checkpoint_for_real_backend(tmp_path):
    motionbert_dir = tmp_path / "MotionBERT"
    motionbert_dir.mkdir()
    (motionbert_dir / "infer_wild.py").write_text("")

    try:
        resolve_motionbert_command(motionbert_dir=motionbert_dir, checkpoint_path=None)
    except FileNotFoundError as exc:
        assert "MotionBERT checkpoint" in str(exc)
    else:
        raise AssertionError("Expected missing checkpoint to fail")


def test_resolve_motionbert_command_builds_official_infer_command(tmp_path):
    motionbert_dir = tmp_path / "MotionBERT"
    motionbert_dir.mkdir()
    (motionbert_dir / "infer_wild.py").write_text("")
    checkpoint = motionbert_dir / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")

    command = resolve_motionbert_command(
        motionbert_dir=motionbert_dir,
        checkpoint_path=checkpoint,
    )

    assert "infer_wild.py" in command
    assert "--evaluate" in command
    assert str(checkpoint) in command
    assert "{alphapose_json}" in command


def test_resolve_motionbert_command_uses_paths_valid_from_motionbert_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    motionbert_dir = Path("external") / "MotionBERT"
    motionbert_dir.mkdir(parents=True)
    (motionbert_dir / "infer_wild.py").write_text("")
    config = motionbert_dir / "configs" / "pose3d" / "MB_ft_h36m_global_lite.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("")
    checkpoint = motionbert_dir / "checkpoint" / "pose3d" / "FT_MB_lite_MB_ft_h36m_global_lite" / "best_epoch.bin"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    command = resolve_motionbert_command()
    parts = shlex.split(command)

    assert Path(parts[1]).is_absolute()
    assert Path(parts[parts.index("--config") + 1]).is_absolute()
    assert Path(parts[parts.index("--evaluate") + 1]).is_absolute()


def test_run_external_motionbert_formats_runtime_paths_as_absolute(tmp_path, monkeypatch):
    calls = []
    monkeypatch.chdir(tmp_path)
    alphapose_json = Path("generated_motionbert") / "clip" / "alphapose.json"
    video = Path("2d_video") / "video.mp4"
    output_dir = Path("generated_motionbert") / "clip"
    alphapose_json.parent.mkdir(parents=True)
    video.parent.mkdir(parents=True)
    alphapose_json.write_text("[]")
    video.write_bytes(b"video")

    def fake_run(args, *, check, cwd):
        calls.append((args, check, cwd))
        np.save(output_dir / "X3D.npy", np.zeros((1, 17, 3), dtype=float))

    monkeypatch.setattr("src.motionbert.motionbert_runner.subprocess.run", fake_run)

    _run_external_motionbert(
        "motionbert --vid_path {video_path} --json_path {alphapose_json} --out_path {output_dir}",
        alphapose_json,
        video,
        output_dir,
    )

    args, check, _cwd = calls[0]
    assert check is True
    assert Path(args[args.index("--vid_path") + 1]).is_absolute()
    assert Path(args[args.index("--json_path") + 1]).is_absolute()
    assert Path(args[args.index("--out_path") + 1]).is_absolute()
