#!/usr/bin/env python3
"""
analyze_serve.py
Universal entry point for the Tennis Serve Analyzer.
Routes image or video input to the correct pipeline.

Usage:
    python analyze_serve.py <file>           # right-handed serve (default)
    python analyze_serve.py <file> left      # left-handed serve
"""

import os
import sys
from datetime import datetime

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}


def get_file_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return 'image'
    if ext in VIDEO_EXTS:
        return 'video'
    return 'unknown'


def resolve_input() -> str:
    """Return a valid input file path or exit with a helpful message."""
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.exists(path):
            sys.exit(f"Error: file not found — '{path}'")
        return path

    # No argument given: look for a sensible default
    defaults = ['tennis_serve.mp4', 'tennis_serve.jpg', 'serve.mp4', 'serve.jpg']
    for f in defaults:
        if os.path.exists(f):
            print(f"No file specified — using: {f}")
            return f

    sys.exit(
        "No input file found.\n"
        "Usage: python analyze_serve.py <image_or_video> [right|left]"
    )


def main():
    print("=" * 60)
    print("TENNIS SERVE ANALYZER")
    print("=" * 60)

    input_file = resolve_input()
    hand       = sys.argv[2].lower() if len(sys.argv) > 2 else 'right'

    if hand not in ('right', 'left'):
        sys.exit(f"Error: hand must be 'right' or 'left', got '{hand}'")

    file_type = get_file_type(input_file)
    if file_type == 'unknown':
        ext = os.path.splitext(input_file)[1]
        sys.exit(
            f"Unsupported file type '{ext}'.\n"
            f"Images : {', '.join(sorted(IMAGE_EXTS))}\n"
            f"Videos : {', '.join(sorted(VIDEO_EXTS))}"
        )

    print(f"Input : {input_file}  ({file_type.upper()}, {hand}-handed serve)\n")

    if file_type == 'image':
        from pose_estimation import analyze_image
        analyze_image(input_file, hand=hand)

    else:
        from tennis_video_analysis import process_video, find_key_serve_moments

        ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video = f"serve_analysis_{ts}.mp4"
        output_csv   = f"serve_angles_{ts}.csv"

        csv_out = process_video(input_file, output_video, output_csv, hand=hand)

        try:
            find_key_serve_moments(csv_out)
        except Exception as e:
            print(f"\nCould not identify key moments: {e}")

    print(f"\n{'='*60}\nAnalysis complete! 🎾\n{'='*60}")


if __name__ == "__main__":
    main()