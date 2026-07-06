from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HALPE_26_NAMES = [
    "Nose",
    "LEye",
    "REye",
    "LEar",
    "REar",
    "LShoulder",
    "RShoulder",
    "LElbow",
    "RElbow",
    "LWrist",
    "RWrist",
    "LHip",
    "RHip",
    "LKnee",
    "RKnee",
    "LAnkle",
    "RAnkle",
    "Head",
    "Neck",
    "Hip",
    "LBigToe",
    "RBigToe",
    "LSmallToe",
    "RSmallToe",
    "LHeel",
    "RHeel",
]

MEDIAPIPE_BY_HALPE = {
    "Nose": 0,
    "LEye": 2,
    "REye": 5,
    "LEar": 7,
    "REar": 8,
    "LShoulder": 11,
    "RShoulder": 12,
    "LElbow": 13,
    "RElbow": 14,
    "LWrist": 15,
    "RWrist": 16,
    "LHip": 23,
    "RHip": 24,
    "LKnee": 25,
    "RKnee": 26,
    "LAnkle": 27,
    "RAnkle": 28,
    "LBigToe": 31,
    "RBigToe": 32,
    "LSmallToe": 31,
    "RSmallToe": 32,
    "LHeel": 29,
    "RHeel": 30,
}


def _validate_inputs(poses_2d: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(poses_2d, dtype=float)
    conf = np.asarray(confidence, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (33, 2):
        raise ValueError(f"poses_2d must have shape (frames, 33, 2); got {poses.shape}")
    if conf.shape != (poses.shape[0], 33):
        raise ValueError(f"confidence must have shape ({poses.shape[0]}, 33); got {conf.shape}")
    return poses, conf


def _mp_point(
    poses: np.ndarray,
    confidence: np.ndarray,
    frame_idx: int,
    landmark_idx: int,
    width: int,
    height: int,
) -> list[float]:
    x = float(poses[frame_idx, landmark_idx, 0] * width)
    y = float(poses[frame_idx, landmark_idx, 1] * height)
    c = float(confidence[frame_idx, landmark_idx])
    return [x, y, c]


def _average_point(points: list[list[float]]) -> list[float]:
    return [
        float(np.mean([p[0] for p in points])),
        float(np.mean([p[1] for p in points])),
        float(np.mean([p[2] for p in points])),
    ]


def mediapipe_frame_to_halpe26(
    poses_2d: np.ndarray,
    confidence: np.ndarray,
    frame_idx: int,
    *,
    width: int,
    height: int,
) -> list[float]:
    poses, conf = _validate_inputs(poses_2d, confidence)
    points: dict[str, list[float]] = {}
    for halpe_name, mp_idx in MEDIAPIPE_BY_HALPE.items():
        points[halpe_name] = _mp_point(poses, conf, frame_idx, mp_idx, width, height)

    points["Head"] = points["Nose"]
    points["Neck"] = _average_point([points["LShoulder"], points["RShoulder"]])
    points["Hip"] = _average_point([points["LHip"], points["RHip"]])

    flat: list[float] = []
    for name in HALPE_26_NAMES:
        flat.extend(points[name])
    return flat


def save_alphapose_json(
    output_dir: str | Path,
    poses_2d: np.ndarray,
    confidence: np.ndarray,
    *,
    width: int,
    height: int,
    filename: str = "alphapose_halpe26.json",
) -> Path:
    poses, conf = _validate_inputs(poses_2d, confidence)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    payload = [
        {
            "image_id": f"{frame_idx:06d}.jpg",
            "category_id": 1,
            "idx": 0,
            "score": float(np.nanmean(conf[frame_idx])),
            "keypoint_names": HALPE_26_NAMES,
            "keypoints": mediapipe_frame_to_halpe26(
                poses,
                conf,
                frame_idx,
                width=width,
                height=height,
            ),
        }
        for frame_idx in range(poses.shape[0])
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
