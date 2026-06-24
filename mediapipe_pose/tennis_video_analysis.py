"""
video_serve_analysis.py
Video pipeline for the Tennis Serve Analyzer.
Processes every N-th frame, annotates the video, writes a CSV timeline,
and identifies key serve moments (trophy position & ball contact).

CSV now outputs raw landmark coordinates (x, y, z) for all key joints
alongside angles and angular velocities.
"""

from email import header
import sys
import os
import csv
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

# ---------------------------------------------------------------------------
# Core video processor
# ---------------------------------------------------------------------------

def process_video(
    input_path:  str,
    output_path: str = None,
    csv_path:    str = None,
    hand:        str = 'right',
    frame_step:  int = 1,
) -> str:
    """
    Process a video file frame-by-frame, annotate it, and export coordinate
    and angle data.

    Args:
        input_path:  Path to the input video.
        output_path: Path for the annotated output video.
        csv_path:    Path for the coordinate/angle timeline CSV.
        hand:        'right' or 'left' serving hand.
        frame_step:  Analyse every Nth frame (1 = every frame).

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

    detector = create_detector(mode='video')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps // 2, (frame_width, frame_height))

    frame_count    = 0
    poses_found    = 0
    last_angles    = None
    last_coords    = None
    last_landmarks = None

    # Tracking variables for velocity / acceleration
    prev_angles = {}
    prev_vels   = {}
    prev_time   = None

    # ── CSV column definitions ────────────────────────────────────────────
    coord_cols = [
        # right side
        'r_shoulder_x', 'r_shoulder_y', 'r_shoulder_z',
        'r_elbow_x',    'r_elbow_y',    'r_elbow_z',
        'r_wrist_x',    'r_wrist_y',    'r_wrist_z',
        'r_hip_x',      'r_hip_y',      'r_hip_z',
        'r_knee_x',     'r_knee_y',     'r_knee_z',
        'r_ankle_x',    'r_ankle_y',    'r_ankle_z',
        # left side
        'l_shoulder_x', 'l_shoulder_y', 'l_shoulder_z',
        'l_elbow_x',    'l_elbow_y',    'l_elbow_z',
        'l_wrist_x',    'l_wrist_y',    'l_wrist_z',
        'l_hip_x',      'l_hip_y',      'l_hip_z',
        'l_knee_x',     'l_knee_y',     'l_knee_z',
        'l_ankle_x',    'l_ankle_y',    'l_ankle_z',
        # visibility
        'r_shoulder_vis', 'r_elbow_vis', 'r_wrist_vis',
        'l_shoulder_vis', 'l_elbow_vis', 'l_wrist_vis',
    ]

    angle_cols = [
        'shoulder_angle', 'elbow_angle', 'wrist_angle',
        'hip_rotation', 'knee_angle', 'trunk_lean',
        'elbow_vel', 'shoulder_vel', 'wrist_vel',
        'elbow_acc', 'shoulder_acc',
    ]

    with open(csv_path, 'w', newline='') as csv_file:
        writer_csv = csv.writer(csv_file)
        writer_csv.writerow(['frame', 'time_s'] + coord_cols + angle_cols)

        print("\nProcessing frames...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count  += 1
            time_sec      = frame_count / fps
            timestamp_ms  = int(frame_count * (1000 / fps))

            # ── Pose detection ────────────────────────────────────────
            if frame_count % frame_step == 0:
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = detector.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    poses_found    += 1
                    landmarks       = result.pose_landmarks[0]
                    analyzer        = TennisServeAnalyzer(landmarks, hand=hand)
                    last_angles     = analyzer.get_all_angles()
                    last_coords     = analyzer.get_all_coordinates()
                    last_landmarks  = landmarks
                    annotated       = draw_landmarks(frame, landmarks, last_angles)
                else:
                    last_angles    = None
                    last_coords    = None
                    last_landmarks = None
                    annotated      = frame
            else:
                if last_landmarks is not None and last_angles is not None:
                    annotated = draw_landmarks(frame, last_landmarks, last_angles)
                else:
                    annotated = frame

            writer.write(annotated)

            # ── CSV writing ───────────────────────────────────────────
            if frame_count % frame_step == 0:
                if last_angles and last_coords and last_landmarks is not None:

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

                    # ── coordinate values ─────────────────────────────
                    coord_values = [
                        # right side
                        f"{last_coords['r_shoulder_x']:.4f}", f"{last_coords['r_shoulder_y']:.4f}", f"{last_coords['r_shoulder_z']:.4f}",
                        f"{last_coords['r_elbow_x']:.4f}",    f"{last_coords['r_elbow_y']:.4f}",    f"{last_coords['r_elbow_z']:.4f}",
                        f"{last_coords['r_wrist_x']:.4f}",    f"{last_coords['r_wrist_y']:.4f}",    f"{last_coords['r_wrist_z']:.4f}",
                        f"{last_coords['r_hip_x']:.4f}",      f"{last_coords['r_hip_y']:.4f}",      f"{last_coords['r_hip_z']:.4f}",
                        f"{last_coords['r_knee_x']:.4f}",     f"{last_coords['r_knee_y']:.4f}",     f"{last_coords['r_knee_z']:.4f}",
                        f"{last_coords['r_ankle_x']:.4f}",    f"{last_coords['r_ankle_y']:.4f}",    f"{last_coords['r_ankle_z']:.4f}",
                        # left side
                        f"{last_coords['l_shoulder_x']:.4f}", f"{last_coords['l_shoulder_y']:.4f}", f"{last_coords['l_shoulder_z']:.4f}",
                        f"{last_coords['l_elbow_x']:.4f}",    f"{last_coords['l_elbow_y']:.4f}",    f"{last_coords['l_elbow_z']:.4f}",
                        f"{last_coords['l_wrist_x']:.4f}",    f"{last_coords['l_wrist_y']:.4f}",    f"{last_coords['l_wrist_z']:.4f}",
                        f"{last_coords['l_hip_x']:.4f}",      f"{last_coords['l_hip_y']:.4f}",      f"{last_coords['l_hip_z']:.4f}",
                        f"{last_coords['l_knee_x']:.4f}",     f"{last_coords['l_knee_y']:.4f}",     f"{last_coords['l_knee_z']:.4f}",
                        f"{last_coords['l_ankle_x']:.4f}",    f"{last_coords['l_ankle_y']:.4f}",    f"{last_coords['l_ankle_z']:.4f}",
                        # visibility
                        f"{last_coords['r_shoulder_vis']:.3f}", f"{last_coords['r_elbow_vis']:.3f}", f"{last_coords['r_wrist_vis']:.3f}",
                        f"{last_coords['l_shoulder_vis']:.3f}", f"{last_coords['l_elbow_vis']:.3f}", f"{last_coords['l_wrist_vis']:.3f}",
                    ]

                    # ── angle values ──────────────────────────────────
                    angle_values = [
                        f"{last_angles['shoulder_angle']:.2f}",
                        f"{last_angles['elbow_angle']:.2f}",
                        f"{last_angles['wrist_angle']:.2f}",
                        f"{last_angles['hip_rotation']:.2f}",
                        f"{last_angles['knee_angle']:.2f}",
                        f"{last_angles['trunk_lean']:.2f}",
                        f"{elbow_vel:.2f}",
                        f"{shoulder_vel:.2f}",
                        f"{wrist_vel:.2f}",
                        f"{elbow_acc:.2f}",
                        f"{shoulder_acc:.2f}",
                    ]

                    writer_csv.writerow(
                        [frame_count, f"{time_sec:.3f}"] + coord_values + angle_values
                    )

                    prev_angles = dict(last_angles)
                    prev_vels   = {
                        'elbow_vel':    elbow_vel,
                        'shoulder_vel': shoulder_vel,
                        'wrist_vel':    wrist_vel,
                    }
                    prev_time = time_sec

                else:
                    # No landmarks — write N/A for all columns
                    n_cols = len(coord_cols) + len(angle_cols)
                    writer_csv.writerow(
                        [frame_count, f"{time_sec:.3f}"] + ['N/A'] * n_cols
                    )

            if frame_count % (fps * 2) == 0:
                pct = frame_count / total_frames * 100
                print(f"  {pct:.1f}%  ({frame_count}/{total_frames} frames)")

    cap.release()
    writer.release()
    detector.close()

    detection_rate = poses_found / max(frame_count // frame_step, 1) * 100
    print(f"\n{'='*60}")
    print("VIDEO PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Frames processed : {frame_count}")
    print(f"  Poses detected   : {poses_found}  ({detection_rate:.1f}% of sampled frames)")
    print(f"  Annotated video  : {output_path}")
    print(f"  Coordinate/Angle CSV : {csv_path}")

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
    """
    print(f"\nAnalyzing serve motion from: {csv_path}")

    data = []
    with open(csv_path, 'r') as f:
        for row in csv.DictReader(f):
            if row.get('elbow_angle', 'N/A') != 'N/A':
                data.append({
                    'frame':        int(row['frame']),
                    'time':         float(row['time_s']),
                    'elbow':        float(row['elbow_angle']),
                    'shoulder':     float(row['shoulder_angle']),
                    'elbow_vel':    float(row['elbow_vel']),
                    'shoulder_vel': float(row['shoulder_vel']),
                    'wrist_vel':    float(row['wrist_vel']),
                    'elbow_acc':    float(row['elbow_acc']),
                    'shoulder_acc': float(row['shoulder_acc']),
                })

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

    csv_out = process_video(input_video, output_video, output_csv, hand=hand)

    try:
        find_key_serve_moments(csv_out)
    except Exception as e:
        print(f"\nCould not identify key moments: {e}")

    print(f"\n{'='*60}\nAll done! 🎾\n{'='*60}")