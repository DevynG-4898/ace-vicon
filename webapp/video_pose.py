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
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd

# mediapipe_pose/ lives one level up, as a sibling of this webapp/ folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mediapipe_pose.tennis_video_analysis import process_video

ANGLE_COLS = [
    'shoulder_angle', 'elbow_angle', 'wrist_angle',
    'hip_rotation', 'knee_angle', 'trunk_lean',
]

# Where persistent per-upload analysis outputs (angle CSV + optional
# annotated video) go. Lives inside webapp/, next to this file.
ANALYSIS_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_outputs")


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