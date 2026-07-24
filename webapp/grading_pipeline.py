"""
grading_pipeline.py
Orchestrates the full "upload a serve video -> grade it against a
reference serve" flow, by chaining together already-existing pieces:

  1. video_to_world_landmarks_csv()      (video_pose.py)
     -> raw MediaPipe pose landmarks CSV, in mm

  2. format_data_mediapipe.load_and_remap() + detect_toss_and_racket_hand()
     + compute_peaks() + export_formatted_csv()
     -> a _formatted.csv: 14 labeled body parts, floor-referenced,
        PEAK1/PEAK2 metadata, empty SNAPSHOT= placeholders

  3. find_snapshots.find_snapshots() + write_snapshots_back()
     -> fills in the 8 real SNAPSHOT= frame numbers

  4. grade_snapshots.grade_serve()
     -> joint-by-joint + overall score, comparing the customer's
        formatted+snapshotted CSV against a reference one

Requires a pre-built reference CSV (itself the output of steps 1-3 run
on a reference video) to grade against — see build_reference_serve().

NOTE ON FOLDER LOCATION: format_data_mediapipe.py, find_snapshots.py, and
grade_snapshots.py are assumed to live in a folder named exactly
"formatdata and render" at the project root (sibling of webapp/ and
mediapipe_pose/), per your file tree. If the actual folder name differs,
update FORMATDATA_DIR_NAME below.
"""

import os
import sys
import datetime

FORMATDATA_DIR_NAME = "formatdata and render"  # <- adjust if this differs

# --- path setup: project root, the formatdata-and-render folder, and mediapipe_pose ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMATDATA_DIR = os.path.join(PROJECT_ROOT, FORMATDATA_DIR_NAME)

for p in (PROJECT_ROOT, FORMATDATA_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# These are plain top-level-module imports (not dotted package imports)
# specifically because FORMATDATA_DIR_NAME contains spaces and can't be
# used as a Python package/dotted-import path -- adding the directory
# itself to sys.path and importing the .py files directly sidesteps that.
import format_data_mediapipe as fdm
import find_snapshots as fsnap
import grade_snapshots as grader

from video_pose import video_to_world_landmarks_csv, SERVE_RECS_DIR

# Where the standing reference serve's fully-processed CSV lives once built
# via build_reference_serve(). Change this path if you want a different
# location or multiple named references later.
REFERENCE_FORMATTED_CSV = os.path.join(SERVE_RECS_DIR, "reference_serve", "reference_formatted.csv")


def build_formatted_and_snapshotted_csv(mediapipe_csv_path: str, output_csv_path: str) -> str:
    """
    Steps 2-3 of the pipeline on an already-extracted raw MediaPipe
    landmarks CSV (mm-scale, from video_to_world_landmarks_csv):
    format_data_mediapipe -> find_snapshots. Writes the final
    formatted+snapshotted CSV to output_csv_path.
    """
    print("  [format] Converting raw landmarks -> formatted CSV (14 points, floor-referenced, PEAK1/PEAK2)...")
    df = fdm.load_and_remap(mediapipe_csv_path)
    toss_label, racket_label, _ = fdm.detect_toss_and_racket_hand(df)
    peak1, peak2 = fdm.compute_peaks(df, toss_label, racket_label)
    fdm.export_formatted_csv(df, peak1, peak2, output_csv_path)

    print("  [snapshots] Finding the 8 snapshot frames...")
    snap_df, tz_cols, part_names, peaks, _existing_snapshots, lines, meta_end = fsnap.read_formatted_csv(output_csv_path)
    snapshots = fsnap.find_snapshots(snap_df, tz_cols, part_names, peaks)
    fsnap.write_snapshots_back(output_csv_path, lines, meta_end, snapshots)

    return output_csv_path


def build_reference_serve(video_path: str, output_csv_path: str = None) -> str:
    """
    One-time (or whenever you want to swap in a new reference) setup:
    processes a reference video all the way through to a formatted +
    snapshotted CSV, and saves it as the standing reference that
    analyze_and_grade_video() compares uploaded serves against.

    Run this yourself once, e.g. from a terminal:
        python3 -c "from grading_pipeline import build_reference_serve; build_reference_serve('path/to/reference.mov')"
    """
    if output_csv_path is None:
        output_csv_path = REFERENCE_FORMATTED_CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    print(f"Building reference serve from: {video_path}")
    raw_csv = video_to_world_landmarks_csv(video_path)
    build_formatted_and_snapshotted_csv(raw_csv, output_csv_path)
    print(f"Reference serve saved: {output_csv_path}")
    return output_csv_path


def analyze_and_grade_video(video_path: str, reference_csv_path: str = None) -> dict:
    """
    Full pipeline for an uploaded customer video: extract raw landmarks,
    format + floor-reference them, find the 8 snapshot frames, then grade
    against the standing reference serve.

    Returns the results dict from grade_snapshots.grade_serve()
    (overall_score, overall_grade, per-snapshot/per-joint breakdown, etc.),
    plus an extra 'customer_formatted_csv' key with the path to this
    customer's own formatted+snapshotted CSV (kept on disk for records).

    Raises FileNotFoundError if no reference CSV exists yet -- run
    build_reference_serve() first.
    """
    if reference_csv_path is None:
        reference_csv_path = REFERENCE_FORMATTED_CSV
    if not os.path.exists(reference_csv_path):
        raise FileNotFoundError(
            f"Reference serve not found at {reference_csv_path}. "
            f"Run build_reference_serve(<reference_video_path>) once first."
        )

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SERVE_RECS_DIR, f"{base_name}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    formatted_csv_path = os.path.join(out_dir, "formatted.csv")

    print("\n[1/3] Extracting raw landmarks from uploaded video...")
    raw_csv = video_to_world_landmarks_csv(video_path)

    print("[2/3] Formatting + finding snapshots...")
    build_formatted_and_snapshotted_csv(raw_csv, formatted_csv_path)

    print("[3/3] Grading against reference serve...")
    results = grader.grade_serve(formatted_csv_path, reference_csv_path)
    results["customer_formatted_csv"] = formatted_csv_path
    grader.print_report(results)   # <-- add this line

    return results