from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.json_utils import jsonable
from src.motionbert.motionbert_runner import MOTIONBERT_JOINT_NAMES
from src.skeleton.sequence import SkeletonSequence

DEFAULT_SKELETON_FILENAME = "skeleton_sequence.npz"
NORMALIZED_SKELETON_FILENAME = "skeleton_normalized.npz"
MOTIONBERT_COORDINATE_SYSTEM = "motionbert_root_centered"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_motionbert_version(motionbert_dir: Path = Path("external") / "MotionBERT") -> str | None:
    head = motionbert_dir / ".git" / "HEAD"
    if not head.is_file():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref = motionbert_dir / ".git" / text.removeprefix("ref: ")
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
    return text or None


def motionbert_sequence_from_arrays(
    poses_3d: np.ndarray,
    *,
    joint_confidence: np.ndarray | None = None,
    frames: np.ndarray | None = None,
    fps: float | None = None,
    metadata: dict[str, Any] | None = None,
    source_file: str | Path | None = None,
    coordinate_system: str = MOTIONBERT_COORDINATE_SYSTEM,
) -> SkeletonSequence:
    """Create a canonical `SkeletonSequence` from MotionBERT 17-joint output."""
    positions = np.asarray(poses_3d, dtype=float)
    if frames is None:
        frames = np.arange(1, positions.shape[0] + 1, dtype=int)
    meta = dict(metadata or {})
    if source_file is not None:
        meta["source_file"] = str(source_file)
    meta.setdefault("motionbert_version", _read_motionbert_version())

    return SkeletonSequence(
        frames=np.asarray(frames),
        joint_names=tuple(MOTIONBERT_JOINT_NAMES),
        joint_positions=positions,
        joint_confidence=joint_confidence,
        fps=fps,
        metadata=meta,
        coordinate_system=coordinate_system,
        source="MotionBERT",
    )


def load_motionbert_sequence(poses_3d_path: str | Path) -> SkeletonSequence:
    """Load `poses_3d.npy` and sibling metadata into a `SkeletonSequence`."""
    path = Path(poses_3d_path)
    poses_3d = np.load(path)
    metadata = _load_json(path.parent / "video_metadata.json")
    fps = float(metadata["fps"]) if metadata.get("fps") is not None else None

    confidence = None
    motionbert_input = path.parent / "motionbert_input_2d.npy"
    if motionbert_input.is_file():
        input_2d = np.load(motionbert_input)
        if input_2d.ndim == 3 and input_2d.shape[:2] == poses_3d.shape[:2] and input_2d.shape[2] >= 3:
            confidence = input_2d[:, :, 2]

    return motionbert_sequence_from_arrays(
        poses_3d,
        joint_confidence=confidence,
        fps=fps,
        metadata=metadata,
        source_file=path,
    )


def save_skeleton_sequence(
    output_dir: str | Path,
    sequence: SkeletonSequence,
    *,
    filename: str = DEFAULT_SKELETON_FILENAME,
) -> Path:
    """Persist a `SkeletonSequence` as a portable NPZ file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    metadata_json = json.dumps(jsonable(sequence.metadata), sort_keys=True)
    np.savez(
        path,
        frames=sequence.frames,
        joint_names=np.asarray(sequence.joint_names, dtype=str),
        joint_positions=sequence.joint_positions,
        joint_confidence=(
            np.asarray(sequence.joint_confidence, dtype=float)
            if sequence.joint_confidence is not None
            else np.empty((0, 0), dtype=float)
        ),
        fps=np.asarray(np.nan if sequence.fps is None else sequence.fps, dtype=float),
        metadata=np.asarray(metadata_json),
        coordinate_system=np.asarray(sequence.coordinate_system),
        source=np.asarray(sequence.source),
    )
    return path


def load_skeleton_sequence(path: str | Path) -> SkeletonSequence:
    """Load a `SkeletonSequence` saved by `save_skeleton_sequence()`."""
    loaded = np.load(path, allow_pickle=False)
    confidence = loaded["joint_confidence"]
    fps_value = float(loaded["fps"])
    return SkeletonSequence(
        frames=loaded["frames"],
        joint_names=tuple(str(v) for v in loaded["joint_names"].tolist()),
        joint_positions=loaded["joint_positions"],
        joint_confidence=None if confidence.size == 0 else confidence,
        fps=None if np.isnan(fps_value) else fps_value,
        metadata=json.loads(str(loaded["metadata"])),
        coordinate_system=str(loaded["coordinate_system"]),
        source=str(loaded["source"]),
    )

