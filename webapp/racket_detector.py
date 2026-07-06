"""
Lightweight relevance filter: checks whether an uploaded video actually
contains a tennis racket before it gets passed into the (expensive)
MediaPipe pose-extraction pipeline.

Uses the yolov8n.pt weights already in the project — no extra download.
"""

import cv2
from ultralytics import YOLO

TENNIS_RACKET_CLASS_ID = 38  # COCO class id for "tennis racket"

_model = None


def _get_model():
    """Load the YOLO model once and reuse it across requests."""
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def detect_racket(video_path, sample_every_n_frames=15, min_hits=2, conf_threshold=0.35):
    """
    Samples frames from the video and checks for a tennis racket.

    Returns:
        (passed: bool, details: dict)
    """
    model = _get_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return False, {"error": "Could not open video file."}

    frame_idx = 0
    checked = 0
    hits = 0
    best_conf = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n_frames == 0:
            results = model(frame, classes=[TENNIS_RACKET_CLASS_ID], conf=conf_threshold, verbose=False)
            checked += 1
            boxes = results[0].boxes
            if len(boxes) > 0:
                hits += 1
                best_conf = max(best_conf, float(boxes.conf.max()))

        frame_idx += 1

    cap.release()

    passed = checked > 0 and hits >= min_hits
    return passed, {
        "frames_checked": checked,
        "frames_with_racket": hits,
        "best_confidence": round(best_conf, 3),
    }