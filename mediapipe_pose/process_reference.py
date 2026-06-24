"""
process_reference.py
────────────────────
Run this on YOUR machine (where pose_landmarker_lite.task lives).
It processes IMG_5092.mov through the full angle extraction pipeline
and then auto-labels the phases using RELATIVE boundaries (0.0–1.0)
derived from visual inspection of the reference video.

Phase boundaries are stored as proportions of total video length so
they work correctly on ANY video regardless of frame count or FPS.

Observed phases (from visual inspection of IMG_5092.mov, 112 frames):
  0% to 19%  → Preparation       (walking up, racket back, ball toss starting)
  19% to 41% → Trophy Position   (arm raised, elbow bent, ball released)
  41% to 56% → Acceleration      (arm driving upward fast)
  56% to 63% → Contact           (full extension, ball strike)
  63% to 100%→ Follow-through    (arm crossing body, landing)

Usage
─────
    python process_reference.py
    # or specify video path:
    python process_reference.py IMG_5092.mov
"""

import sys
import csv
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ── Phase boundaries as RELATIVE positions (0.0 – 1.0) ───────────────────────
# Derived from IMG_5092.mov (112 frames) but generalise to any video length.
PHASE_BOUNDARIES = {
    0: (0.00, 0.19),   # Preparation
    1: (0.19, 0.41),   # Trophy Position
    2: (0.41, 0.56),   # Acceleration
    3: (0.56, 0.63),   # Contact
    4: (0.63, 1.00),   # Follow-through
}

PHASE_NAMES = {
    0: "Preparation",
    1: "Trophy Position",
    2: "Acceleration",
    3: "Contact",
    4: "Follow-through",
}


def get_phase(frame_num: int, total_frames: int) -> int:
    """
    Map a frame number to a phase using relative position.
    Works for any video length — a 300-frame video gets the
    same proportional phase splits as a 112-frame video.
    """
    position = frame_num / max(total_frames, 1)   # 0.0 → 1.0
    for pid, (start, end) in PHASE_BOUNDARIES.items():
        if start <= position < end:
            return pid
    return 4   # anything at or beyond 100% = follow-through


def add_phase_labels(angle_csv: str, out_path: str = "reference_labeled.csv"):
    """
    Read the angle CSV produced by tennis_video_analysis.py and
    add phase_id / phase_name columns based on relative boundaries.
    """
    # First pass: count total valid frames to compute relative positions
    all_rows = []
    with open(angle_csv, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) + ['phase_id', 'phase_name']
        for row in reader:
            if row.get('Elbow_deg', 'N/A') == 'N/A':
                continue
            all_rows.append(row)

    total_frames = len(all_rows)
    print(f"  Total valid frames: {total_frames}")

    # Second pass: assign phase labels using relative position
    rows = []
    for i, row in enumerate(all_rows):
        pid = get_phase(i, total_frames)
        row['phase_id']   = pid
        row['phase_name'] = PHASE_NAMES[pid]
        rows.append(row)

    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nLabeled CSV saved → {out_path}")
    print(f"Total frames labeled: {len(rows)}")
    print("\nPhase breakdown:")
    for pid, pname in PHASE_NAMES.items():
        count  = sum(1 for r in rows if int(r['phase_id']) == pid)
        pct    = count / total_frames * 100 if total_frames else 0
        print(f"  {pname:<22}: {count:>4} frames  ({pct:.1f}%)")

    return out_path


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "IMG_5092.mov"

    if not os.path.exists(video_path):
        sys.exit(f"Video not found: {video_path}")

    # ── Step 1: Extract angles using existing pipeline ────────────────────────
    print("=" * 60)
    print("STEP 1: Extracting angles from reference video...")
    print("=" * 60)

    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    angle_csv   = f"reference_angles_{ts}.csv"
    output_vid  = f"reference_annotated_{ts}.mp4"

    # Import and run the video pipeline
    from mediapipe_pose_estimation.tennis_video_analysis import process_video
    process_video(
        input_path  = video_path,
        output_path = output_vid,
        csv_path    = angle_csv,
        hand        = 'right',
        frame_step  = 1,    # process every frame — reference video is only 112 frames
    )

    # ── Step 2: Add phase labels ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Adding phase labels...")
    print("=" * 60)
    labeled_csv = add_phase_labels(angle_csv)

    # ── Step 3: Train HMM ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Training HMM on reference data...")
    print("=" * 60)
    from phase_2_hmm import train as hmm_train
    hmm_train(labeled_csv)

    # ── Step 4: Train Bayesian Network ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Training Bayesian Network on reference data...")
    print("=" * 60)
    from phase_3_bayes import train as bayes_train
    bayes_train(labeled_csv)

    print("\n" + "=" * 60)
    print("REFERENCE PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nFiles produced:")
    print(f"  {angle_csv:<40} ← raw angle data")
    print(f"  {labeled_csv:<40} ← labeled with phases")
    print(f"  hmm_model.pkl                            ← trained HMM")
    print(f"  hmm_scaler.pkl                           ← feature scaler")
    print(f"  bayesian_model.pkl                       ← trained Bayesian network")
    print(f"\nNext step — analyse a user's serve:")
    print(f"  python serve_analysis.py <user_video.mp4>")
    print(f"  python phase4_feedback.py full \\")
    print(f"      --ref-csv {labeled_csv} \\")
    print(f"      --user-csv serve_angles_<timestamp>.csv")


if __name__ == "__main__":
    main()