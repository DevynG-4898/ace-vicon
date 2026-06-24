"""
utils.py
Shared utilities for the Tennis Serve Analyzer.
Centralizes POSE_CONNECTIONS, drawing functions, and orientation handling
so nothing is duplicated across files.
"""

import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Origin / Local Coordinate Helpers ──────────────────────────────────────

LEFT_HIP  = 23
RIGHT_HIP = 24

def get_origin(landmarks):
    """Mid-hip centre point — stable base for all measurements."""
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]
    return {'x': (lh.x + rh.x) / 2,
            'y': (lh.y + rh.y) / 2,
            'z': (lh.z + rh.z) / 2}

def to_local(landmark, origin):
    """Convert any landmark to origin-relative coordinates."""
    return {'x': landmark.x - origin['x'],
            'y': landmark.y - origin['y'],
            'z': landmark.z - origin['z']}

def compute_angular_velocity(angle_now, angle_prev, dt):
    """Degrees per second between two frames."""
    if dt <= 0:
        return 0.0
    return (angle_now - angle_prev) / dt

def compute_angular_acceleration(vel_now, vel_prev, dt):
    """Degrees per second² between two frames."""
    if dt <= 0:
        return 0.0
    return (vel_now - vel_prev) / dt

# ---------------------------------------------------------------------------
# Skeleton definition
# ---------------------------------------------------------------------------

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

# ---------------------------------------------------------------------------
# Detector factory
# ---------------------------------------------------------------------------

def create_detector(mode: str = 'image'):
    """
    Create and return a MediaPipe PoseLandmarker.

    Args:
        mode: 'image' for single images, 'video' for frame-by-frame video.
    """
    _DIR = os.path.dirname(os.path.abspath(__file__))
    base_options = python.BaseOptions(model_asset_path=os.path.join(_DIR, 'pose_landmarker_heavy.task'))

    if mode == 'video':
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.VIDEO
        )
    else:
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.IMAGE
        )

    return vision.PoseLandmarker.create_from_options(options)

# ---------------------------------------------------------------------------
# Orientation correction (images only)
# ---------------------------------------------------------------------------

def _upright_score(pose_landmarks) -> float:
    """Score how upright a detected pose is (head above feet = high score)."""
    nose_y     = pose_landmarks[0].y
    shoulder_y = (pose_landmarks[11].y + pose_landmarks[12].y) / 2
    hip_y      = (pose_landmarks[23].y + pose_landmarks[24].y) / 2
    ankle_y    = (pose_landmarks[27].y + pose_landmarks[28].y) / 2

    score = 0.0
    score += 1.0 if nose_y     < shoulder_y else -1.0
    score += 1.0 if shoulder_y < hip_y      else -1.0
    score += 1.0 if hip_y      < ankle_y    else -1.0

    height = abs(ankle_y - nose_y)
    width  = abs(pose_landmarks[11].x - pose_landmarks[12].x)
    if height > width:
        score += 0.5

    return score


def _brute_force_rotation(image_path: str, detector):
    """Try all 4 rotations and pick the one that gives the most upright pose."""
    import os

    rotations = [
        (0,   None),
        (90,  cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]

    best_score    = -1
    best_rotation = None
    original      = cv2.imread(image_path)
    tmp_path      = "temp_orientation_test.jpg"

    for angle, rot_code in rotations:
        test_img = cv2.rotate(original, rot_code) if rot_code else original.copy()
        cv2.imwrite(tmp_path, test_img)
        try:
            mp_img = mp.Image.create_from_file(tmp_path)
            result = detector.detect(mp_img)
            if result.pose_landmarks:
                score = _upright_score(result.pose_landmarks[0])
                print(f"Rotation {angle}°: score = {score:.3f}")
                if score > best_score:
                    best_score    = score
                    best_rotation = rot_code
        except Exception:
            continue

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return best_rotation


def get_image_rotation(image_path: str, detector):
    """
    Determine the correct rotation for an image using brute-force pose scoring.
    Skips EXIF as phone EXIF tags are often unreliable.
    """
    print("Trying all rotations to find best orientation...")
    return _brute_force_rotation(image_path, detector)

# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_landmarks(frame: np.ndarray, pose_landmarks, angles: dict = None) -> np.ndarray:
    """
    Draw the pose skeleton and optional angle overlay onto a frame/image.

    Args:
        frame:          BGR numpy array (as returned by OpenCV).
        pose_landmarks: List of MediaPipe landmarks for one person.
        angles:         Dict of angle values from TennisServeAnalyzer.get_all_angles().

    Returns:
        Annotated BGR numpy array.
    """
    out    = frame.copy()
    h, w   = out.shape[:2]

    # --- skeleton lines ---
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
            s = (int(pose_landmarks[start_idx].x * w), int(pose_landmarks[start_idx].y * h))
            e = (int(pose_landmarks[end_idx].x   * w), int(pose_landmarks[end_idx].y   * h))
            cv2.line(out, s, e, (0, 255, 0), 2)

    # --- landmark dots ---
    for lm in pose_landmarks:
        cv2.circle(out, (int(lm.x * w), int(lm.y * h)), 5, (0, 0, 255), -1)

    # --- angle overlay ---
    if angles:
        overlay = out.copy()
        cv2.rectangle(overlay, (10, 10), (350, 210), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

        font   = cv2.FONT_HERSHEY_SIMPLEX
        y      = 30
        labels = [
            ("SERVE ANALYSIS",                   0.7, (255, 255, 255)),
            (f"Shoulder:   {angles['shoulder_angle']:.1f} deg", 0.6, (0, 255, 255)),
            (f"Elbow:      {angles['elbow_angle']:.1f} deg",    0.6, (0, 255, 255)),
            (f"Wrist:      {angles['wrist_angle']:.1f} deg",    0.6, (0, 255, 255)),
            (f"Hip Rot:    {angles['hip_rotation']:.1f} deg",   0.6, (0, 255, 255)),
            (f"Knee:       {angles['knee_angle']:.1f} deg",     0.6, (0, 255, 255)),
            (f"Trunk Lean: {angles['trunk_lean']:.1f} deg",     0.6, (0, 255, 255)),
        ]
        for text, scale, color in labels:
            cv2.putText(out, text, (20, y), font, scale, color, 2)
            y += 25 if scale < 0.7 else 30

    return out