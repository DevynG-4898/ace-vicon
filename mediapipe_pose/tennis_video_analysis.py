"""
tennis_video_analysis.py
Video pipeline for the Tennis Serve Analyzer.
Processes every N-th frame, annotates the video, writes a CSV timeline,
and identifies key serve moments (trophy position & ball contact).

CSV outputs per-frame joint angles, per-angle visibility (min landmark
visibility used for that angle), and angular velocities/accelerations.

UPDATED:
- Angle computation now uses pose_world_landmarks (metric-scale, camera-
  distance invariant) instead of normalized pose_landmarks. Normalized
  landmarks are still used for drawing the overlay.
- Frames with no pose at all still write NaN across all columns.
- Frames WITH a pose but occluded landmarks now correctly propagate NaN
  for just the affected angle(s), rather than silently computing a
  distorted value from a guessed low-visibility landmark.
- Per-angle visibility columns are written to the CSV so a downstream
  interpolation step (see interpolate_occlusions.py) can decide which
  gaps to fill and how large a gap is acceptable, and so accuracy can
  later be reported separately for interpolated vs. directly-tracked
  frames.
- Velocity/acceleration in THIS raw CSV are computed directly off the
  (possibly NaN) raw angles and will themselves be NaN across any
  occlusion gap and the frame immediately after it. This is expected —
  recompute velocity/acceleration AFTER interpolating the angle columns
  (see interpolate_occlusions.py) rather than trusting these raw
  derivative columns for analysis.
"""

import sys
import os
import csv
import math
import cv2
import mediapipe as mp
from datetime import datetime

from mediapipe_pose.utils import (create_detector, draw_landmarks,
                                   get_origin, to_local,
                                   compute_angular_velocity,
                                   compute_angular_acceleration)
from mediapipe_pose.serve_angles import TennisServeAnalyzer

# Landmark indices (right side)
WRIST    = 16
ELBOW    = 14
SHOULDER = 12

ANGLE_NAMES = [
    'shoulder_angle', 'elbow_angle', 'wrist_angle',
    'hip_rotation', 'knee_angle', 'trunk_lean',
]

NAN = float('nan')


def _is_nan(x) -> bool:
    return isinstance(x, float) and x != x


# ---------------------------------------------------------------------------
# Core video processor
# ---------------------------------------------------------------------------

def process_video(
    input_path:  str,
    output_path: str = None,
    csv_path:    str = None,
    hand:        str = 'right',
    frame_step:  int = 1,
    write_video: bool = True,
    visibility_threshold: float = 0.5,
) -> str:
    """
    Process a video file frame-by-frame, optionally annotate it, and
    export the angle/angular-velocity timeline.

    Args:
        input_path:             Path to the input video.
        output_path:            Path for the annotated output video. Ignored
                                 when write_video=False.
        csv_path:                Path for the angle timeline CSV.
        hand:                   'right' or 'left' serving hand.
        frame_step:             Analyse every Nth frame (1 = every frame).
        write_video:            If False, skip drawing landmarks and encoding
                                 the annotated output video entirely.
        visibility_threshold:   Minimum MediaPipe landmark visibility
                                 required to trust a landmark in angle math.
                                 Landmarks below this yield NaN angles.

    Returns:
        csv_path (so callers can pass it straight to find_key_serve_moments).
    """
    print(f"\nOpening video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_path}")

    fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution : {frame_width}x{frame_height}")
    print(f"  FPS        : {fps}")
    print(f"  Frames     : {total_frames}")
    print(f"  Frame step : every {frame_step} frame(s)")
    print(f"  Write video: {write_video}")
    print(f"  Vis. thresh: {visibility_threshold}")

    detector = create_detector(mode='video')

    writer = None
    if write_video:
        if not output_path:
            raise ValueError("output_path is required when write_video=True")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps // 2, (frame_width, frame_height))

    frame_count    = 0
    poses_found    = 0
    last_angles    = None
    last_landmarks = None   # normalized landmarks, for drawing only

    # Tracking variables for velocity / acceleration
    prev_angles = {}
    prev_vels   = {}
    prev_time   = None

    # ── CSV column definitions ────────────────────────────────────────────
    # visibility columns record the min landmark visibility used per angle,
    # so downstream code can tell a confident angle from an occlusion-derived
    # (NaN) one, and decide how far a gap can be safely interpolated.
    vis_cols = [f"{name}_visibility" for name in ANGLE_NAMES]
    deriv_cols = [
        'elbow_vel', 'shoulder_vel', 'wrist_vel',
        'elbow_acc', 'shoulder_acc',
    ]
    all_cols = ANGLE_NAMES + vis_cols + deriv_cols

    with open(csv_path, 'w', newline='') as csv_file:
        writer_csv = csv.writer(csv_file)
        writer_csv.writerow(['frame', 'time_s'] + all_cols)

        print("\nProcessing frames...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count  += 1
            time_sec      = frame_count / fps
            timestamp_ms  = int(frame_count * (1000 / fps))

            last_confidences = None

            # ── Pose detection ────────────────────────────────────────
            if frame_count % frame_step == 0:
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = detector.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks and result.pose_world_landmarks:
                    poses_found      += 1
                    image_landmarks   = result.pose_landmarks[0]        # drawing only
                    world_landmarks   = result.pose_world_landmarks[0]  # angle math

                    analyzer          = TennisServeAnalyzer(
                                            world_landmarks, hand=hand,
                                            visibility_threshold=visibility_threshold)
                    last_angles       = analyzer.get_all_angles()
                    last_confidences  = analyzer.get_angle_confidences()
                    last_landmarks    = image_landmarks

                    if write_video:
                        annotated = draw_landmarks(frame, image_landmarks, last_angles)
                else:
                    last_angles      = None
                    last_confidences = None
                    last_landmarks   = None
                    if write_video:
                        annotated = frame
            else:
                if write_video:
                    if last_landmarks is not None and last_angles is not None:
                        annotated = draw_landmarks(frame, last_landmarks, last_angles)
                    else:
                        annotated = frame

            if write_video:
                writer.write(annotated)

            # ── CSV writing ───────────────────────────────────────────
            if frame_count % frame_step == 0:
                if last_angles is not None:

                    dt = (time_sec - prev_time) if prev_time is not None else 0

                    elbow_vel    = compute_angular_velocity(
                                       last_angles['elbow_angle'],
                                       prev_angles.get('elbow_angle',    last_angles['elbow_angle']), dt)
                    shoulder_vel = compute_angular_velocity(
                                       last_angles['shoulder_angle'],
                                       prev_angles.get('shoulder_angle', last_angles['shoulder_angle']), dt)
                    wrist_vel    = compute_angular_velocity(
                                       last_angles['wrist_angle'],
                                       prev_angles.get('wrist_angle',    last_angles['wrist_angle']), dt)

                    elbow_acc    = compute_angular_acceleration(
                                       elbow_vel,    prev_vels.get('elbow_vel',    elbow_vel),    dt)
                    shoulder_acc = compute_angular_acceleration(
                                       shoulder_vel, prev_vels.get('shoulder_vel', shoulder_vel), dt)

                    angle_values = [f"{last_angles[n]:.4f}" if not _is_nan(last_angles[n])
                                     else "nan" for n in ANGLE_NAMES]
                    vis_values   = [f"{last_confidences[n]:.3f}" for n in ANGLE_NAMES]
                    deriv_values = [
                        f"{elbow_vel:.2f}"    if not _is_nan(elbow_vel)    else "nan",
                        f"{shoulder_vel:.2f}" if not _is_nan(shoulder_vel) else "nan",
                        f"{wrist_vel:.2f}"    if not _is_nan(wrist_vel)    else "nan",
                        f"{elbow_acc:.2f}"    if not _is_nan(elbow_acc)    else "nan",
                        f"{shoulder_acc:.2f}" if not _is_nan(shoulder_acc) else "nan",
                    ]

                    writer_csv.writerow(
                        [frame_count, f"{time_sec:.3f}"] + angle_values + vis_values + deriv_values
                    )

                    prev_angles = dict(last_angles)
                    prev_vels   = {
                        'elbow_vel':    elbow_vel,
                        'shoulder_vel': shoulder_vel,
                        'wrist_vel':    wrist_vel,
                    }
                    prev_time = time_sec

                else:
                    # No pose detected at all — NaN across every column,
                    # visibility 0 for all angles.
                    angle_values = ["nan"] * len(ANGLE_NAMES)
                    vis_values   = ["0.000"] * len(ANGLE_NAMES)
                    deriv_values = ["nan"] * len(deriv_cols)
                    writer_csv.writerow(
                        [frame_count, f"{time_sec:.3f}"] + angle_values + vis_values + deriv_values
                    )

            if frame_count % (fps * 2) == 0:
                pct = frame_count / total_frames * 100
                print(f"  {pct:.1f}%  ({frame_count}/{total_frames} frames)")

    cap.release()
    if writer is not None:
        writer.release()
    detector.close()

    detection_rate = poses_found / max(frame_count // frame_step, 1) * 100
    print(f"\n{'='*60}")
    print("VIDEO PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Frames processed : {frame_count}")
    print(f"  Poses detected   : {poses_found}  ({detection_rate:.1f}% of sampled frames)")
    if write_video:
        print(f"  Annotated video  : {output_path}")
    else:
        print(f"  Annotated video  : skipped (write_video=False)")
    print(f"  Angle CSV        : {csv_path}")
    print(f"  NOTE: velocity/acceleration columns in this CSV are computed "
          f"from RAW (pre-interpolation) angles and will contain NaN across "
          f"occlusion gaps. Run interpolate_occlusions.py before using them "
          f"for analysis.")

    return csv_path


# ---------------------------------------------------------------------------
# Key-moment finder
# ---------------------------------------------------------------------------

def find_key_serve_moments(csv_path: str):
    """
    Read the CSV and identify trophy position and estimated ball contact.

    Trophy position  = frame with minimum elbow angle (most bent).
    Ball contact     = frame with maximum elbow angle after trophy.
    Also reports peak angular velocity frames for each joint.

    Rows with NaN elbow_angle (occluded / no pose, not yet interpolated)
    are skipped. For best results, run this on a CSV that has already
    been through interpolate_occlusions.py.
    """
    print(f"\nAnalyzing serve motion from: {csv_path}")

    data = []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            elbow_raw = row.get('elbow_angle', 'nan')
            if elbow_raw not in ('nan', 'N/A', ''):
                try:
                    data.append({
                        'frame':        int(row['frame']),
                        'time':         float(row['time_s']),
                        'elbow':        float(row['elbow_angle']),
                        'shoulder':     float(row['shoulder_angle']),
                        'elbow_vel':    float(row['elbow_vel'])    if row['elbow_vel']    not in ('nan', '') else 0.0,
                        'shoulder_vel': float(row['shoulder_vel']) if row['shoulder_vel'] not in ('nan', '') else 0.0,
                        'wrist_vel':    float(row['wrist_vel'])    if row['wrist_vel']    not in ('nan', '') else 0.0,
                        'elbow_acc':    float(row['elbow_acc'])    if row['elbow_acc']    not in ('nan', '') else 0.0,
                        'shoulder_acc': float(row['shoulder_acc']) if row['shoulder_acc'] not in ('nan', '') else 0.0,
                    })
                except ValueError:
                    continue

    if not data:
        print("No valid pose data found — cannot identify key moments.")
        return None, None

    trophy  = min(data, key=lambda x: x['elbow'])
    t_idx   = next(i for i, d in enumerate(data) if d['frame'] == trophy['frame'])
    contact = max(data[t_idx:], key=lambda x: x['elbow'])

    peak_elbow_vel    = max(data, key=lambda x: abs(x['elbow_vel']))
    peak_shoulder_vel = max(data, key=lambda x: abs(x['shoulder_vel']))

    print(f"\n{'='*60}")
    print("KEY SERVE MOMENTS")
    print(f"{'='*60}")
    for label, moment in [("Trophy Position", trophy), ("Ball Contact (est.)", contact)]:
        print(f"\n{label}:")
        print(f"  Frame    : {moment['frame']}")
        print(f"  Time     : {moment['time']:.2f}s")
        print(f"  Elbow    : {moment['elbow']:.1f}°")
        print(f"  Shoulder : {moment['shoulder']:.1f}°")

    print(f"\nPeak Elbow Angular Velocity:")
    print(f"  Frame    : {peak_elbow_vel['frame']}")
    print(f"  Time     : {peak_elbow_vel['time']:.2f}s")
    print(f"  Velocity : {peak_elbow_vel['elbow_vel']:.1f} deg/s")

    print(f"\nPeak Shoulder Angular Velocity:")
    print(f"  Frame    : {peak_shoulder_vel['frame']}")
    print(f"  Time     : {peak_shoulder_vel['time']:.2f}s")
    print(f"  Velocity : {peak_shoulder_vel['shoulder_vel']:.1f} deg/s")

    return trophy, contact


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    input_video = sys.argv[1] if len(sys.argv) > 1 else "tennis_serve.mp4"
    hand        = sys.argv[2] if len(sys.argv) > 2 else "right"

    # ── create output folder ──────────────────────────────────────────────
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name  = os.path.splitext(os.path.basename(input_video))[0]
    output_dir = os.path.join(
                    os.path.dirname(os.path.abspath(input_video)),
                    "outputs",
                    f"{base_name}_{ts}"
                 )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output folder: {output_dir}")

    output_video = os.path.join(output_dir, f"{base_name}_analyzed.mp4")
    output_csv   = os.path.join(output_dir, f"{base_name}_coords_angles.csv")

    # Standalone CLI usage still produces the annotated video by default.
    csv_out = process_video(input_video, output_video, output_csv, hand=hand, write_video=True)

    try:
        find_key_serve_moments(csv_out)
    except Exception as e:
        print(f"\nCould not identify key moments: {e}")

    print(f"\n{'='*60}\nAll done! 🎾\n{'='*60}")