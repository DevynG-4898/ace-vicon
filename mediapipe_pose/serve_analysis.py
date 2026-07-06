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

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}


def validate_video(path: str):
    ext = os.path.splitext(path)[1].lower()

    if ext not in VIDEO_EXTS:
        sys.exit(f"Unsupported video format: {ext}")
        
validate_video(input_file)



def resolve_input() -> str:
    """Return a valid input file path or exit with a helpful message."""
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if not os.path.exists(path):
            sys.exit(f"Error: file not found — '{path}'")
        return path

    # No argument given: look for a sensible default
    defaults = ['tennis_serve.mp4', 'serve.mp4']
    for f in defaults:
        if os.path.exists(f):
            print(f"No file specified — using: {f}")
            return f

    sys.exit(
        "No input file found.\n"
        "Usage: python analyze_serve.py <video> [right|left]"
    )


def main():
    print("=" * 60)
    print("TENNIS SERVE ANALYZER")
    print("=" * 60)

    input_file = resolve_input()
    hand       = sys.argv[2].lower() if len(sys.argv) > 2 else 'right'

    if hand not in ('right', 'left'):
        sys.exit(f"Error: hand must be 'right' or 'left', got '{hand}'")

        validate_video(input_file)
        ext = os.path.splitext(input_file)[1]
        sys.exit(
            f"Unsupported file type '{ext}'.\n"
            f"Videos : {', '.join(sorted(VIDEO_EXTS))}"
        )

    print(f"Input : {input_file}  ({file_type.upper()}, {hand}-handed serve)\n")

    from tennis_video_analysis import process_video

    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = f"serve_analysis_{ts}.mp4"
    output_csv   = f"serve_angles_{ts}.csv"

    csv_out = process_video(input_file, output_video, output_csv, hand=hand)


if __name__ == "__main__":
    main()