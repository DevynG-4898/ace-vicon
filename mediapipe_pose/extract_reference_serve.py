"""
extract_reference_serve.py
Standalone terminal tool — no webapp/Flask/Supabase involved.

Runs MediaPipe pose detection on a local video file and writes the 14
"nthserve" reference points (head, chest, left/right shoulder/elbow/
hand/hip/knee/foot) straight to CSV, in the same layout as the mocap
reference dataset (2_labeled.csv / Vicon-style export):

    Row 1: point names
    Row 2: TX/TY/TZ axis labels
    Row 3: units (mm)
    Data:  Frame, Sub Frame, <14 points x tx,ty,tz>

Usage:
    python extract_reference_serve.py <video_path> [output_csv] [--frame-step N] [--vis-threshold T] [--scale S]

Examples:
    python extract_reference_serve.py my_serve.mov
    python extract_reference_serve.py my_serve.mov my_serve_coords.csv
    python extract_reference_serve.py my_serve.mov --frame-step 2 --vis-threshold 0.4

Place this file in the same directory as your mediapipe_pose/ package
(i.e. alongside process_reference.py, process_serves.py, etc.) — it
imports create_detector and the reference writer from there.
"""

import sys
import os
import argparse
import cv2
import mediapipe as mp

from utils import create_detector
from reference_format_writer import ReferenceFormatWriter, extract_reference_point_values

def extract_reference_serve(video_path: str, output_csv: str = None,
                             frame_step: int = 1,
                             visibility_threshold: float = 0.3,
                             scale: float = 1000.0) -> str:
    if output_csv is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_csv = f"{base_name}_reference_format.csv"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Opening video: {video_path}")
    print(f"  FPS        : {fps}")
    print(f"  Frames     : {total_frames}")
    print(f"  Frame step : every {frame_step} frame(s)")
    print(f"  Vis thresh : {visibility_threshold}")
    print(f"  Scale      : {scale} (1000 = meters -> mm)")

    detector = create_detector(mode='video')
    frame_count = 0
    poses_found = 0
    ref_writer = ReferenceFormatWriter()

    try:
        print("\nProcessing frames...")
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

            if frame_count % (fps * 2) == 0 and total_frames:
                pct = frame_count / total_frames * 100
                print(f"  {pct:.1f}%  ({frame_count}/{total_frames} frames)")
    finally:
        cap.release()
        detector.close()

    if poses_found == 0:
        raise ValueError(
            "No pose could be detected in the video. "
            "Try a clearer, well-lit video with the full body in frame."
        )

    ref_writer.write(output_csv)

    detection_rate = poses_found / max(frame_count // frame_step, 1) * 100
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Frames processed : {frame_count}")
    print(f"  Poses detected   : {poses_found} ({detection_rate:.1f}%)")
    print(f"  Output CSV       : {output_csv}")

    return output_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 14-point reference-format coordinates from a video (terminal only, no webapp).")
    parser.add_argument("video_path", help="Path to the serve video")
    parser.add_argument("output_csv", nargs="?", default=None, help="Output CSV path (default: <video_name>_reference_format.csv)")
    parser.add_argument("--frame-step", type=int, default=1, help="Analyse every Nth frame (default 1)")
    parser.add_argument("--vis-threshold", type=float, default=0.3, help="Min landmark visibility to trust a point (default 0.3)")
    parser.add_argument("--scale", type=float, default=1000.0, help="Multiply x/y/z by this (default 1000: meters -> mm)")
    args = parser.parse_args()

    extract_reference_serve(
        args.video_path,
        args.output_csv,
        frame_step=args.frame_step,
        visibility_threshold=args.vis_threshold,
        scale=args.scale,
    )