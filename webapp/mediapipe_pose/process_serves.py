#!/usr/bin/env python3
"""
process_serves.py
Process one or more serve videos and save angles-only CSVs.

Usage:
    python process_serves.py serve1.mov serve2.mov serve3.mov
    python process_serves.py serves/          # process all .mov/.mp4 in a folder
"""

import os
import sys

# Allow running from inside the mediapipe_pose folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from tennis_video_analysis import process_video

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}


def collect_videos(args: list[str]) -> list[str]:
    """Expand file/folder args into a flat list of video paths."""
    videos = []
    for arg in args:
        if os.path.isdir(arg):
            for f in sorted(os.listdir(arg)):
                if os.path.splitext(f)[1].lower() in VIDEO_EXTS:
                    videos.append(os.path.join(arg, f))
        elif os.path.isfile(arg):
            if os.path.splitext(arg)[1].lower() in VIDEO_EXTS:
                videos.append(arg)
            else:
                print(f"  Skipping (unsupported format): {arg}")
        else:
            print(f"  Skipping (not found): {arg}")
    return videos


def main():
    if len(sys.argv) < 2:
        sys.exit(
            "Usage:\n"
            "  python process_serves.py serve1.mov serve2.mov\n"
            "  python process_serves.py serves_folder/"
        )

    hand = 'right'  # change to 'left' if needed

    videos = collect_videos(sys.argv[1:])
    if not videos:
        sys.exit("No valid video files found.")

    print(f"Found {len(videos)} video(s) to process.\n")

    output_csvs = []

    for i, video_path in enumerate(videos, 1):
        base      = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(os.path.abspath(video_path))
        out_dir   = os.path.join(video_dir, "outputs", base)
        os.makedirs(out_dir, exist_ok=True)

        csv_path = os.path.join(out_dir, f"{base}_angles.csv")

        print(f"[{i}/{len(videos)}] Processing: {video_path}")
        print(f"  → CSV: {csv_path}")

        try:
            process_video(
                input_path=video_path,
                output_path=None,   # no annotated video — angles only
                csv_path=csv_path,
                hand=hand,
                frame_step=1,
            )
            output_csvs.append(csv_path)
            print(f"  ✓ Done\n")
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")

    print("=" * 60)
    print(f"Processed {len(output_csvs)}/{len(videos)} videos successfully.")
    print("\nAngles CSVs:")
    for p in output_csvs:
        print(f"  {p}")
    print("=" * 60)
    print("\nNext step: run dtw_avg_serve.py with these CSVs to find your average serve.")


if __name__ == "__main__":
    main()