from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np

from src.video_discovery import output_dir_for_video

LANDMARK_COUNT = 33
POSE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DEFAULT_MODEL_PATH = Path("generated_motionbert") / "models" / "pose_landmarker_full.task"


def _json_float(value: float) -> float | None:
    if np.isnan(value):
        return None
    return float(value)


def frame_landmarks_to_arrays(pose_landmarks: Any) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | None]]]:
    """Convert one MediaPipe Pose result into 2D points, confidence, and debug JSON."""
    xy = np.full((LANDMARK_COUNT, 2), np.nan, dtype=float)
    confidence = np.full((LANDMARK_COUNT,), np.nan, dtype=float)
    debug: list[dict[str, float | None]] = []

    landmarks = getattr(pose_landmarks, "landmark", pose_landmarks)
    if not landmarks:
        return xy, confidence, [
            {"x": None, "y": None, "z": None, "visibility": None}
            for _ in range(LANDMARK_COUNT)
        ]

    for idx in range(min(LANDMARK_COUNT, len(landmarks))):
        lm = landmarks[idx]
        x = float(getattr(lm, "x", np.nan))
        y = float(getattr(lm, "y", np.nan))
        z = float(getattr(lm, "z", np.nan))
        visibility = float(getattr(lm, "visibility", np.nan))
        xy[idx] = [x, y]
        confidence[idx] = visibility
        debug.append(
            {
                "x": _json_float(x),
                "y": _json_float(y),
                "z": _json_float(z),
                "visibility": _json_float(visibility),
            }
        )

    for _ in range(len(debug), LANDMARK_COUNT):
        debug.append({"x": None, "y": None, "z": None, "visibility": None})

    return xy, confidence, debug


def compute_confidence_stats(confidence: np.ndarray) -> dict[str, float | int | None]:
    conf = np.asarray(confidence, dtype=float)
    missing_frames = int(np.sum(np.all(np.isnan(conf), axis=1))) if conf.ndim == 2 else 0
    finite = conf[np.isfinite(conf)]
    if finite.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "missing_frames": missing_frames,
        }
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "missing_frames": missing_frames,
    }


def save_extraction_outputs(
    output_dir: str | Path,
    poses_2d: np.ndarray,
    confidence: np.ndarray,
    frames_debug: list[list[dict[str, float | None]]],
    metadata: dict[str, Any],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata = dict(metadata)
    metadata["landmark_count"] = int(poses_2d.shape[1]) if poses_2d.ndim == 3 else 0
    metadata["confidence"] = compute_confidence_stats(confidence)

    np.save(out / "poses_2d.npy", poses_2d)
    np.save(out / "poses_2d_confidence.npy", confidence)

    debug_payload = {
        "shape": list(poses_2d.shape),
        "coordinate_convention": "MediaPipe normalized image coordinates: x and y in [0, 1].",
        "frames": [
            {"frame_index": idx, "landmarks": landmarks}
            for idx, landmarks in enumerate(frames_debug)
        ],
    }
    (out / "poses_2d.json").write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
    (out / "video_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required. Install it with: pip install opencv-python") from exc
    return cv2


def _import_mediapipe():
    try:
        import mediapipe as mp  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("MediaPipe is required. Install it with: pip install mediapipe") from exc
    return mp


def ensure_pose_model(model_path: str | Path | None = None) -> Path:
    """Return a local MediaPipe Tasks pose model, downloading the default if needed."""
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(POSE_LANDMARKER_MODEL_URL, path)
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe Tasks PoseLandmarker requires a .task model. "
            f"Download {POSE_LANDMARKER_MODEL_URL} to {path}, or pass --pose-model /path/to/model.task."
        ) from exc
    return path


def extract_video(
    video_path: str | Path,
    output_root: str | Path = "generated_motionbert",
    *,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 2,
    pose_model_path: str | Path | None = None,
) -> Path:
    """Run MediaPipe Pose on every video frame and save reusable 2D outputs."""
    cv2 = _import_cv2()
    mp = _import_mediapipe()

    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    poses: list[np.ndarray] = []
    confidence: list[np.ndarray] = []
    frames_debug: list[list[dict[str, float | None]]] = []

    try:
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            mp_pose = mp.solutions.pose
            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            ) as pose:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame.flags.writeable = False
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)
                    xy, conf, debug = frame_landmarks_to_arrays(getattr(results, "pose_landmarks", None))
                    poses.append(xy)
                    confidence.append(conf)
                    frames_debug.append(debug)
        else:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options as base_options_module

            model_path = ensure_pose_model(pose_model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options_module.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_segmentation_masks=False,
            )
            frame_idx = 0
            frame_ms = 1000.0 / fps if fps > 0 else 33.0
            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(image, int(frame_idx * frame_ms))
                    landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                    xy, conf, debug = frame_landmarks_to_arrays(landmarks)
                    poses.append(xy)
                    confidence.append(conf)
                    frames_debug.append(debug)
                    frame_idx += 1
    finally:
        cap.release()

    poses_array = np.stack(poses, axis=0) if poses else np.empty((0, LANDMARK_COUNT, 2), dtype=float)
    confidence_array = (
        np.stack(confidence, axis=0) if confidence else np.empty((0, LANDMARK_COUNT), dtype=float)
    )
    out = output_dir_for_video(video, output_root)
    metadata = {
        "video_name": video.name,
        "video_path": str(video),
        "frame_count": int(poses_array.shape[0]),
        "expected_frame_count": expected_frames,
        "fps": fps,
        "resolution": {"width": width, "height": height},
    }
    save_extraction_outputs(out, poses_array, confidence_array, frames_debug, metadata)
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract MediaPipe 2D pose landmarks from a video.")
    parser.add_argument("video", help="Path to an .mp4 or other OpenCV-readable video.")
    parser.add_argument("--output-root", default="generated_motionbert")
    parser.add_argument("--pose-model", default=None, help="Optional MediaPipe PoseLandmarker .task model path.")
    args = parser.parse_args()
    out = extract_video(args.video, args.output_root, pose_model_path=args.pose_model)
    print(f"Saved 2D pose outputs to {out}")


if __name__ == "__main__":
    main()
