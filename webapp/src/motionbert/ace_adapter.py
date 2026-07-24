from __future__ import annotations

from pathlib import Path

import numpy as np

from src.motionbert.motionbert_runner import MOTIONBERT_JOINT_NAMES

ACE_MARKER_NAMES = (
    "head",
    "chest",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hand",
    "right_hand",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
)

ACE_TO_MOTIONBERT = {
    "head": "head",
    "chest": "thorax",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
}


def _validate_pose_3d(poses_3d: np.ndarray) -> np.ndarray:
    pose = np.asarray(poses_3d, dtype=float)
    if pose.ndim != 3 or pose.shape[1:] != (len(MOTIONBERT_JOINT_NAMES), 3):
        raise ValueError(
            "poses_3d must have shape "
            f"(frames, {len(MOTIONBERT_JOINT_NAMES)}, 3); got {pose.shape}"
        )
    return pose


def motionbert_to_ace_markers(
    poses_3d: np.ndarray,
    *,
    frame_start: int = 1,
    scale: float = 1.0,
) -> dict[str, dict[str, np.ndarray] | np.ndarray]:
    """Convert 17-joint MotionBERT output into ACE's Vicon marker dictionary.

    The existing Vicon animation code expects marker axes named TX/TY/TZ. MotionBERT
    uses image-like Y coordinates, so map vertical motion onto ACE's Z axis.
    Pass ``scale`` when you want to enlarge normalized model coordinates.
    """
    pose = _validate_pose_3d(poses_3d)
    frames = np.arange(frame_start, frame_start + pose.shape[0], dtype=int)
    markers: dict[str, dict[str, np.ndarray] | np.ndarray] = {"frames": frames}

    for ace_name in ACE_MARKER_NAMES:
        mb_name = ACE_TO_MOTIONBERT[ace_name]
        idx = MOTIONBERT_JOINT_NAMES.index(mb_name)
        coords = pose[:, idx, :] * float(scale)
        markers[ace_name] = {
            "TX": coords[:, 0].astype(float),
            "TY": coords[:, 2].astype(float),
            "TZ": (-coords[:, 1]).astype(float),
        }

    return markers


def save_ace_markers(
    output_dir: str | Path,
    markers: dict[str, dict[str, np.ndarray] | np.ndarray],
    *,
    filename: str = "ace_markers.npz",
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    arrays: dict[str, np.ndarray] = {"frames": np.asarray(markers["frames"])}
    for marker in ACE_MARKER_NAMES:
        marker_axes = markers[marker]
        if not isinstance(marker_axes, dict):
            raise ValueError(f"marker {marker} must contain TX/TY/TZ arrays")
        for axis in ("TX", "TY", "TZ"):
            arrays[f"{marker}_{axis}"] = np.asarray(marker_axes[axis], dtype=float)
    np.savez(path, **arrays)
    return path


def load_ace_markers(npz_path: str | Path) -> dict[str, dict[str, np.ndarray] | np.ndarray]:
    loaded = np.load(npz_path)
    markers: dict[str, dict[str, np.ndarray] | np.ndarray] = {"frames": loaded["frames"]}
    for marker in ACE_MARKER_NAMES:
        markers[marker] = {
            "TX": loaded[f"{marker}_TX"],
            "TY": loaded[f"{marker}_TY"],
            "TZ": loaded[f"{marker}_TZ"],
        }
    return markers


def convert_file_to_ace_markers(
    poses_3d_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    scale: float = 1.0,
) -> Path:
    path = Path(poses_3d_path)
    out = Path(output_dir) if output_dir is not None else path.parent
    pose = np.load(path)
    markers = motionbert_to_ace_markers(pose, scale=scale)
    return save_ace_markers(out, markers)
