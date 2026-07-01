from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.motionbert.alphapose_adapter import save_alphapose_json

MOTIONBERT_JOINT_NAMES = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]

SKELETON_EDGES = [
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("pelvis", "spine"),
    ("spine", "thorax"),
    ("thorax", "neck"),
    ("neck", "head"),
    ("thorax", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("thorax", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
]

MEDIAPIPE_TO_MOTIONBERT = {
    "right_hip": 24,
    "right_knee": 26,
    "right_ankle": 28,
    "left_hip": 23,
    "left_knee": 25,
    "left_ankle": 27,
    "head": 0,
    "left_shoulder": 11,
    "left_elbow": 13,
    "left_wrist": 15,
    "right_shoulder": 12,
    "right_elbow": 14,
    "right_wrist": 16,
}


def _require_shape(name: str, array: np.ndarray, expected_last_dims: tuple[int, ...]) -> None:
    if array.ndim != len(expected_last_dims) + 1 or array.shape[1:] != expected_last_dims:
        raise ValueError(f"{name} must have shape (frames, {', '.join(map(str, expected_last_dims))}); got {array.shape}")


def _average_points(points: np.ndarray, confidence: np.ndarray, names: list[str], *joint_names: str) -> tuple[np.ndarray, np.ndarray]:
    indices = [names.index(name) for name in joint_names]
    return np.nanmean(points[:, indices, :], axis=1), np.nanmean(confidence[:, indices], axis=1)


def _interpolate_nans(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=float).copy()
    frames = np.arange(arr.shape[0])
    for joint in range(arr.shape[1]):
        for channel in range(arr.shape[2]):
            series = arr[:, joint, channel]
            bad = ~np.isfinite(series)
            if not bad.any():
                continue
            good = ~bad
            if good.sum() >= 2:
                series[bad] = np.interp(frames[bad], frames[good], series[good])
            elif good.sum() == 1:
                series[bad] = series[good][0]
            else:
                series[bad] = 0.0
            arr[:, joint, channel] = series
    return arr


def convert_mediapipe_to_motionbert(
    poses_2d: np.ndarray,
    confidence: np.ndarray | None = None,
) -> np.ndarray:
    """Map MediaPipe's 33-landmark output to MotionBERT's 17-joint 2D input.

    The returned channel order is x, y, confidence.
    """
    poses = np.asarray(poses_2d, dtype=float)
    _require_shape("poses_2d", poses, (33, 2))
    if confidence is None:
        conf = np.ones((poses.shape[0], 33), dtype=float)
    else:
        conf = np.asarray(confidence, dtype=float)
        if conf.shape != (poses.shape[0], 33):
            raise ValueError(f"confidence must have shape ({poses.shape[0]}, 33); got {conf.shape}")

    output = np.full((poses.shape[0], len(MOTIONBERT_JOINT_NAMES), 3), np.nan, dtype=float)
    for joint_name, mp_index in MEDIAPIPE_TO_MOTIONBERT.items():
        out_idx = MOTIONBERT_JOINT_NAMES.index(joint_name)
        output[:, out_idx, :2] = poses[:, mp_index, :]
        output[:, out_idx, 2] = conf[:, mp_index]

    points = output[:, :, :2]
    joint_confidence = output[:, :, 2]
    pelvis_xy, pelvis_conf = _average_points(points, joint_confidence, MOTIONBERT_JOINT_NAMES, "left_hip", "right_hip")
    thorax_xy, thorax_conf = _average_points(points, joint_confidence, MOTIONBERT_JOINT_NAMES, "left_shoulder", "right_shoulder")
    spine_xy = (pelvis_xy + thorax_xy) / 2.0
    neck_xy = (thorax_xy + output[:, MOTIONBERT_JOINT_NAMES.index("head"), :2]) / 2.0

    derived = {
        "pelvis": (pelvis_xy, pelvis_conf),
        "thorax": (thorax_xy, thorax_conf),
        "spine": (spine_xy, (pelvis_conf + thorax_conf) / 2.0),
        "neck": (neck_xy, (thorax_conf + output[:, MOTIONBERT_JOINT_NAMES.index("head"), 2]) / 2.0),
    }
    for name, (xy, conf_values) in derived.items():
        idx = MOTIONBERT_JOINT_NAMES.index(name)
        output[:, idx, :2] = xy
        output[:, idx, 2] = conf_values

    return _interpolate_nans(output)


def _geometric_lift(motionbert_2d: np.ndarray) -> np.ndarray:
    points = np.asarray(motionbert_2d, dtype=float)
    _require_shape("motionbert_2d", points, (17, 3))
    xy = points[:, :, :2]
    pelvis = xy[:, MOTIONBERT_JOINT_NAMES.index("pelvis"), :]
    centered = xy - pelvis[:, None, :]

    pose_3d = np.zeros((points.shape[0], points.shape[1], 3), dtype=float)
    pose_3d[:, :, 0] = centered[:, :, 0]
    pose_3d[:, :, 1] = -centered[:, :, 1]

    shoulder_width = np.linalg.norm(
        xy[:, MOTIONBERT_JOINT_NAMES.index("left_shoulder"), :]
        - xy[:, MOTIONBERT_JOINT_NAMES.index("right_shoulder"), :],
        axis=1,
    )
    depth_scale = np.maximum(shoulder_width, 1e-3)
    for idx, name in enumerate(MOTIONBERT_JOINT_NAMES):
        side = 0.0
        if name.startswith("left_"):
            side = -0.18
        elif name.startswith("right_"):
            side = 0.18
        pose_3d[:, idx, 2] = side * depth_scale

    return _interpolate_nans(pose_3d)


DEFAULT_MOTIONBERT_DIR = Path("external") / "MotionBERT"
DEFAULT_MOTIONBERT_CHECKPOINT = (
    DEFAULT_MOTIONBERT_DIR
    / "checkpoint"
    / "pose3d"
    / "FT_MB_lite_MB_ft_h36m_global_lite"
    / "best_epoch.bin"
)


def resolve_motionbert_command(
    *,
    motionbert_dir: str | Path = DEFAULT_MOTIONBERT_DIR,
    checkpoint_path: str | Path | None = DEFAULT_MOTIONBERT_CHECKPOINT,
) -> str:
    repo = Path(motionbert_dir).resolve()
    infer_script = repo / "infer_wild.py"
    if not infer_script.is_file():
        raise FileNotFoundError(f"MotionBERT infer_wild.py not found: {infer_script}")

    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path is not None else (
        repo / "checkpoint" / "pose3d" / "FT_MB_lite_MB_ft_h36m_global_lite" / "best_epoch.bin"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "MotionBERT checkpoint not found. Download the official pose3d checkpoint "
            f"and place it at {checkpoint}, or pass --checkpoint /path/to/best_epoch.bin."
        )

    config = repo / "configs" / "pose3d" / "MB_ft_h36m_global.yaml"
    return (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(infer_script))} "
        f"--config {shlex.quote(str(config))} "
        f"--evaluate {shlex.quote(str(checkpoint))} "
        "--vid_path {video_path} "
        "--json_path {alphapose_json} "
        "--out_path {output_dir}"
    )


def _run_external_motionbert(
    command_template: str,
    alphapose_json_path: Path,
    video_path: Path,
    output_dir: Path,
) -> np.ndarray:
    x3d_path = output_dir / "X3D.npy"
    command = command_template.format(
        input=str(alphapose_json_path.resolve()),
        alphapose_json=str(alphapose_json_path.resolve()),
        video_path=str(video_path.resolve()),
        output=str(x3d_path),
        output_dir=str(output_dir.resolve()),
    )
    subprocess.run(shlex.split(command), check=True, cwd=str(DEFAULT_MOTIONBERT_DIR))
    if not x3d_path.is_file():
        raise RuntimeError(f"MotionBERT command completed but did not create {x3d_path}")
    pose_3d = np.load(x3d_path)
    _require_shape("poses_3d", pose_3d, (17, 3))
    return pose_3d


def generate_3d_pose(
    motionbert_2d: np.ndarray,
    *,
    backend: str = "auto",
    motionbert_command: str | None = None,
    alphapose_json_path: str | Path | None = None,
    video_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> np.ndarray:
    """Generate 3D joint coordinates from MotionBERT-format 2D input.

    ``backend='external'`` runs a configured MotionBERT command. ``backend='geometric'``
    is a deterministic validation fallback, not a learned MotionBERT model.
    """
    selected = backend
    if selected == "auto":
        selected = "external"

    if selected == "external":
        if not motionbert_command or alphapose_json_path is None or video_path is None or output_dir is None:
            raise ValueError("external backend requires motionbert_command, alphapose_json_path, video_path, and output_dir")
        return _run_external_motionbert(motionbert_command, Path(alphapose_json_path), Path(video_path), Path(output_dir))
    if selected == "geometric":
        return _geometric_lift(motionbert_2d)
    raise ValueError(f"Unknown backend: {backend}")


def _confidence_stats(confidence: np.ndarray | None) -> dict[str, float | None]:
    if confidence is None:
        return {"min": None, "max": None, "mean": None}
    finite = confidence[np.isfinite(confidence)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {"min": float(np.min(finite)), "max": float(np.max(finite)), "mean": float(np.mean(finite))}


def save_3d_outputs(
    output_dir: str | Path,
    poses_3d: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    source_backend: str,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pose = np.asarray(poses_3d, dtype=float)
    _require_shape("poses_3d", pose, (17, 3))

    np.save(out / "poses_3d.npy", pose)
    payload: dict[str, Any] = {
        "shape": list(pose.shape),
        "joint_names": MOTIONBERT_JOINT_NAMES,
        "skeleton_edges": SKELETON_EDGES,
        "coordinate_convention": "Root-centered 3D coordinates. X is image horizontal, Y is image vertical up, Z is approximate depth unless using an external MotionBERT backend.",
        "units": "normalized_video_units",
        "source_backend": source_backend,
        "confidence": _confidence_stats(confidence),
        "frames": [
            {
                "frame_index": frame_idx,
                "joints": {
                    name: [float(v) for v in pose[frame_idx, joint_idx]]
                    for joint_idx, name in enumerate(MOTIONBERT_JOINT_NAMES)
                },
            }
            for frame_idx in range(pose.shape[0])
        ],
    }
    (out / "poses_3d.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_motionbert_stage(
    output_dir: str | Path,
    *,
    backend: str = "auto",
    motionbert_command: str | None = None,
    motionbert_dir: str | Path = DEFAULT_MOTIONBERT_DIR,
    checkpoint_path: str | Path | None = DEFAULT_MOTIONBERT_CHECKPOINT,
    video_path: str | Path | None = None,
) -> Path:
    out = Path(output_dir)
    poses_2d_path = out / "poses_2d.npy"
    if not poses_2d_path.is_file():
        raise FileNotFoundError(f"Missing 2D pose file: {poses_2d_path}")

    poses_2d = np.load(poses_2d_path)
    confidence_path = out / "poses_2d_confidence.npy"
    confidence_33 = np.load(confidence_path) if confidence_path.is_file() else None
    motionbert_2d = convert_mediapipe_to_motionbert(poses_2d, confidence_33)
    np.save(out / "motionbert_input_2d.npy", motionbert_2d)

    metadata_path = out / "video_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    width = int(metadata.get("resolution", {}).get("width") or 0)
    height = int(metadata.get("resolution", {}).get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("video_metadata.json must include positive resolution width and height for MotionBERT input")
    alphapose_json = save_alphapose_json(out, poses_2d, confidence_33, width=width, height=height)

    selected_backend = "external" if backend == "auto" else backend
    command = motionbert_command
    if selected_backend == "external" and command is None:
        command = resolve_motionbert_command(motionbert_dir=motionbert_dir, checkpoint_path=checkpoint_path)
    if selected_backend == "external" and video_path is None:
        video_from_metadata = metadata.get("video_path")
        if not video_from_metadata:
            raise ValueError("external MotionBERT backend requires video_path or video_metadata.json video_path")
        video_path = video_from_metadata

    poses_3d = generate_3d_pose(
        motionbert_2d,
        backend=selected_backend,
        motionbert_command=command,
        alphapose_json_path=alphapose_json,
        video_path=video_path,
        output_dir=out,
    )
    confidence_17 = motionbert_2d[:, :, 2]
    save_3d_outputs(out, poses_3d, confidence_17, source_backend=selected_backend)
    from src.skeleton import NORMALIZED_SKELETON_FILENAME, normalize_skeleton
    from src.skeleton.io import motionbert_sequence_from_arrays, save_skeleton_sequence

    skeleton = motionbert_sequence_from_arrays(
        poses_3d,
        joint_confidence=confidence_17,
        fps=float(metadata["fps"]) if metadata.get("fps") is not None else None,
        metadata={
            **metadata,
            "source_backend": selected_backend,
            "source_file": str(out / "poses_3d.npy"),
        },
        source_file=out / "poses_3d.npy",
    )
    normalized_skeleton = normalize_skeleton(skeleton)
    save_skeleton_sequence(out, skeleton)
    save_skeleton_sequence(out, normalized_skeleton, filename=NORMALIZED_SKELETON_FILENAME)
    from src.features import extract_features, save_feature_sequence

    save_feature_sequence(out, extract_features(normalized_skeleton))
    from src.motionbert.ace_adapter import motionbert_to_ace_markers, save_ace_markers

    save_ace_markers(out, motionbert_to_ace_markers(normalized_skeleton.joint_positions))
    return out / "poses_3d.npy"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert MediaPipe 2D poses into 3D MotionBERT-style output.")
    parser.add_argument("output_dir", help="Directory containing poses_2d.npy")
    parser.add_argument("--backend", choices=("auto", "external", "geometric"), default="auto")
    parser.add_argument("--motionbert-command", default=None)
    parser.add_argument("--motionbert-dir", default=str(DEFAULT_MOTIONBERT_DIR))
    parser.add_argument("--checkpoint", default=str(DEFAULT_MOTIONBERT_CHECKPOINT))
    parser.add_argument("--video-path", default=None)
    args = parser.parse_args()
    out = run_motionbert_stage(
        args.output_dir,
        backend=args.backend,
        motionbert_command=args.motionbert_command,
        motionbert_dir=args.motionbert_dir,
        checkpoint_path=args.checkpoint,
        video_path=args.video_path,
    )
    print(f"Saved 3D pose output to {out}")


if __name__ == "__main__":
    main()
