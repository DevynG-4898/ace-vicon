"""
video_pose.py
Thin wrapper around the existing tennis_video_analysis.process_video pipeline.

Runs MediaPipe pose + angle extraction on an uploaded video, then reduces
the 6 per-frame joint angles into a single 1D trajectory (mean angle per
frame) so it can be scored against the CSV-derived reference model using
the same scoring function in model.py.

Outputs (per-frame angle CSV, and optionally annotated video) are saved
persistently under analysis_outputs/, instead of a temp dir, so they can
be inspected or reused later (e.g. for DTW alignment against reference
serves).

UPDATED:
- Added video_to_reference_format_csv(), which runs the same MediaPipe
  detection loop as video_to_world_landmarks_csv() but writes the 14
  "nthserve" points (head, chest, left/right shoulder/elbow/hand/hip/
  knee/foot) in the same CSV format as the mocap reference dataset
  (2_labeled.csv): 3 header rows (point names / TX-TY-TZ / units) then
  Frame, Sub Frame, and 14x3 tx/ty/tz columns in mm, blank where a point
  is occluded. See mediapipe_pose/reference_format_writer.py for the
  extraction + write logic and the exact point mapping.
"""

import os
import sys
import csv
import datetime
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


from mediapipe_pose.tennis_video_analysis import process_video
from mediapipe_pose.utils import create_detector
from mediapipe_pose.reference_format_writer import ReferenceFormatWriter, extract_reference_point_values

ANGLE_COLS = [
    'shoulder_angle', 'elbow_angle', 'wrist_angle',
    'hip_rotation', 'knee_angle', 'trunk_lean',
]

# Standard MediaPipe Pose landmark order (0-32)
LANDMARK_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]

# Where persistent per-upload analysis outputs (angle CSV + optional
# annotated video) go. Lives inside webapp/, next to this file.
ANALYSIS_OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_outputs")

# Where reference-format coordinate CSVs from webapp uploads go — inside
# mediapipe_pose/serve_recs/, alongside your other recorded/processed
# serves, so they're all browsable in one place instead of split across
# webapp/analysis_outputs/.
SERVE_RECS_DIR = os.path.join(BASE_DIR, "mediapipe_pose", "serve_recs")


def video_to_marker_trajectory(video_path, hand: str = 'right', frame_step: int = 1,
                                return_angles_df: bool = False,
                                write_video: bool = False):
    """
    Run the MediaPipe pipeline on an uploaded video and return a 1D
    trajectory (mean of the 6 joint angles per frame) for scoring.

    Args:
        video_path:       path to the uploaded video file.
        hand:              'right' or 'left' serving hand.
        frame_step:        analyse every Nth frame (1 = every frame).
        return_angles_df:  if True, also return the full per-frame angle
                            DataFrame (all 6 angles, not just the mean) —
                            useful for DTW, which needs per-joint signals.
        write_video:       if True, also render and save the annotated
                            video. Defaults to False since the webapp
                            doesn't currently show it to users — skipping
                            this saves a meaningful chunk of processing
                            time (landmark drawing + video encoding).

    Returns:
        np.ndarray trajectory, or (trajectory, angles_df) if return_angles_df=True.
    """
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ANALYSIS_OUTPUT_DIR, f"{base_name}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    out_video = os.path.join(out_dir, "annotated.mp4") if write_video else None
    out_csv = os.path.join(out_dir, "angles.csv")

    process_video(
        input_path=video_path,
        output_path=out_video,
        csv_path=out_csv,
        hand=hand,
        frame_step=frame_step,
        write_video=write_video,
    )

    df_raw = pd.read_csv(out_csv)

    # Drop frames where no pose was detected ('N/A' rows)
    df = df_raw[ANGLE_COLS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    if df.empty:
        raise ValueError(
            "No pose could be detected in the uploaded video. "
            "Try a clearer, well-lit video with the full body in frame."
        )

    # Reduce the 6 joint angles to a single scalar per frame
    trajectory = df[ANGLE_COLS].mean(axis=1).values

    if len(trajectory) < 2:
        raise ValueError(
            "Not enough valid pose frames in the uploaded video to analyse."
        )

    if return_angles_df:
        return trajectory, df
    return trajectory


def video_to_world_landmarks_csv(video_path, frame_step: int = 1, scale: float = 1000.0) -> str:
    """
    Run MediaPipe pose detection on an uploaded video and write the raw
    pose_world_landmarks (x/y/z per landmark, plus per-landmark visibility)
    straight to CSV — no angle math involved.

    MediaPipe's pose_world_landmarks are metric-scale, hip-centered, and in
    METERS. `scale` (default 1000) converts to millimeters before writing,
    since format_data_mediapipe.py (the next step in the grading pipeline)
    assumes its input coordinates are already in mm — its fixed thresholds
    (e.g. PARALLEL_THRESHOLD = 5.0mm in find_snapshots.py) are only
    meaningful at that scale. Pass scale=1.0 if you need raw meters for
    some other purpose.

    This is the "give me coordinates instead of angles" path: it bypasses
    TennisServeAnalyzer entirely, unlike video_to_marker_trajectory().

    Returns:
        Path to the written CSV (saved under ANALYSIS_OUTPUT_DIR so it
        persists alongside the angle-based outputs).
    """
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ANALYSIS_OUTPUT_DIR, f"{base_name}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "world_landmarks.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    detector = create_detector(mode='video')
    frame_count = 0
    poses_found = 0

    header = ['frame', 'time_s']
    for name in LANDMARK_NAMES:
        header += [f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_visibility']

    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_step != 0:
                    continue

                time_sec = frame_count / fps
                timestamp_ms = int(frame_count * (1000 / fps))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(mp_image, timestamp_ms)

                row = [frame_count, f"{time_sec:.3f}"]
                if result.pose_world_landmarks:
                    poses_found += 1
                    world_landmarks = result.pose_world_landmarks[0]
                    for lm in world_landmarks:
                        row += [f"{lm.x * scale:.6f}", f"{lm.y * scale:.6f}", f"{lm.z * scale:.6f}", f"{lm.visibility:.3f}"]
                else:
                    row += ["nan"] * (len(LANDMARK_NAMES) * 4)

                writer.writerow(row)
    finally:
        cap.release()
        detector.close()

    if poses_found == 0:
        raise ValueError(
            "No pose could be detected in the uploaded video. "
            "Try a clearer, well-lit video with the full body in frame."
        )

    return csv_path


def video_to_reference_format_csv(video_path, frame_step: int = 1,
                                   visibility_threshold: float = 0.3,
                                   scale: float = 1000.0) -> str:
    """
    Run MediaPipe pose detection on an uploaded video and write the 14
    "nthserve" reference points (head, chest, left/right shoulder/elbow/
    hand/hip/knee/foot) straight to CSV, in the SAME layout as the mocap
    reference dataset (2_labeled.csv):

        Row 1: point names
        Row 2: TX/TY/TZ axis labels
        Row 3: units (mm)
        Data:  Frame, Sub Frame, <14 points x tx,ty,tz>

    Point mapping and occlusion handling are defined in
    mediapipe_pose/reference_format_writer.py. In short:
        - head <- nose, chest <- midpoint(L/R shoulder), and the rest
          map 1:1 (shoulder/elbow/hip/knee direct; hand <- wrist;
          foot <- ankle).
        - x/y/z are scaled by `scale` (default 1000: MediaPipe's
          meter-scale world landmarks -> mm, matching the reference
          units). This fixes units only, not the coordinate-frame
          origin/orientation -- see the module docstring for details.
        - Any point whose landmark visibility is below
          `visibility_threshold` is written blank for that frame,
          matching how the reference CSV leaves occluded points blank.

    Returns:
        Path to the written CSV. Saved under SERVE_RECS_DIR
        (mediapipe_pose/serve_recs/<video_name>_<timestamp>/reference_format.csv),
        alongside your other recorded/processed serves — not under
        webapp/analysis_outputs/.
    """
    os.makedirs(SERVE_RECS_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SERVE_RECS_DIR, f"{base_name}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "reference_format.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    detector = create_detector(mode='video')
    frame_count = 0
    poses_found = 0

    ref_writer = ReferenceFormatWriter()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_step != 0:
                continue

            timestamp_ms = int(frame_count * (1000 / fps))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if result.pose_world_landmarks:
                poses_found += 1
                world_landmarks = result.pose_world_landmarks[0]
                point_values = extract_reference_point_values(
                    world_landmarks,
                    visibility_threshold=visibility_threshold,
                    scale=scale,
                )
                ref_writer.add_frame(frame_count, 0, point_values)
            else:
                ref_writer.add_blank_frame(frame_count, 0)
    finally:
        cap.release()
        detector.close()

    if poses_found == 0:
        raise ValueError(
            "No pose could be detected in the uploaded video. "
            "Try a clearer, well-lit video with the full body in frame."
        )

    ref_writer.write(csv_path)
    return csv_path