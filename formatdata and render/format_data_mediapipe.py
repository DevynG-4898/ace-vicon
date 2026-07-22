"""
format_data_mediapipe.py
--------------------------
Converts a MediaPipe Pose landmark CSV (frame, time_s, <landmark>_x/_y/_z/
_visibility columns, already in mm) directly into the same _formatted.csv
structure produced by format_data.py -- ready for find_snapshots.py.

Unlike format_data.py (which blindly detects head/hands/body-parts from
anonymous Vicon tracks via TZ-height heuristics), MediaPipe already tells
us exactly which landmark is which. So this script skips all of that
detection and just hardcodes the mapping:

    nose            -> head
    left_shoulder   -> left_shoulder
    right_shoulder  -> right_shoulder
    left_elbow      -> left_elbow
    right_elbow     -> right_elbow
    left_wrist      -> left_hand
    right_wrist     -> right_hand
    left_hip        -> left_hip
    right_hip       -> right_hip
    left_knee       -> left_knee
    right_knee      -> right_knee
    left_foot_index -> left_foot
    right_foot_index-> right_foot
    (left_shoulder + right_shoulder) / 2  -> chest   [synthesized -- MediaPipe
                                                        has no chest landmark]

Axis remap (MediaPipe y increases downward; Vicon TZ increases upward):
    TX = mp_x
    TY = mp_z
    TZ = -mp_y

MediaPipe's landmarks are hip-centered (hip TZ ~= 0), while Vicon is
floor-referenced (floor TZ == 0). After the axis remap, every marker's TZ
is shifted by one rigid offset -- estimated from the low end of the foot
trajectories -- so the floor sits at TZ=0, matching Vicon's convention.
This is a uniform shift only; it does not alter joint angles or distances.

The only things still auto-detected (because they genuinely aren't known
in advance -- depends on which hand tosses vs. swings) are:
    - the floor height, from the low end of the foot TZ trajectories
    - which wrist is the ball-toss hand vs. the racket hand
    - PEAK1 (Ball Toss)      = frame of max TZ for the toss hand
    - PEAK2 (Follow Through) = frame of max TZ for the racket hand

Usage:
    python format_data_mediapipe.py <mediapipe_csv> [output_csv]
"""

import sys
import os
import csv
import numpy as np
import pandas as pd

# Direct landmark -> body-part-label mapping (chest handled separately)
DIRECT_MAP = {
    "nose":             "head",
    "left_shoulder":    "left_shoulder",
    "right_shoulder":   "right_shoulder",
    "left_elbow":       "left_elbow",
    "right_elbow":      "right_elbow",
    "left_hip":         "left_hip",
    "right_hip":        "right_hip",
    "left_knee":        "left_knee",
    "right_knee":       "right_knee",
    "left_foot_index":  "left_foot",
    "right_foot_index": "right_foot",
    "left_wrist":       "left_hand",
    "right_wrist":      "right_hand",
}

# Column order for the output header (order doesn't affect downstream
# tools -- they read by name -- but keeping it consistent/readable)
OUTPUT_ORDER = [
    "head", "chest",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_hand", "right_hand",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_foot", "right_foot",
]

SNAPSHOT_NAMES = [
    'start_pose',
    'hand_cross',
    'flat_racket_arm',
    'peak_racket_arm',
    'contact',
    'hand_cross_2',
    'racket_deceleration',
    'finish_pose',
]


def estimate_floor_z(out, floor_percentile=2.0):
    """
    Estimate where the physical floor sits in MediaPipe's raw TZ (vertical)
    coordinates. MediaPipe landmarks are hip-centered (hip TZ hovers near 0),
    not floor-referenced like Vicon (floor TZ == 0). We estimate the floor
    from the low end of the foot trajectories -- the feet can't go below the
    real floor, so their lowest points approximate ground contact.

    Uses a low percentile instead of a strict min so a single noisy/occluded
    frame can't skew the estimate.
    """
    foot_tz = np.concatenate([
        out["left_foot_TZ"].dropna().values,
        out["right_foot_TZ"].dropna().values,
    ])
    if foot_tz.size == 0:
        raise ValueError("No valid left_foot_TZ/right_foot_TZ samples -- cannot estimate floor.")
    return float(np.percentile(foot_tz, floor_percentile))


def apply_floor_offset(out, floor_z):
    """
    Shift every marker's TZ by the same rigid offset so the floor sits at
    TZ=0, matching Vicon's convention. This is a uniform shift (same amount
    added to every marker, every frame) -- it moves the whole skeleton along
    the vertical axis without altering joint angles or relative distances.
    """
    tz_cols = [c for c in out.columns if c.endswith("_TZ")]
    for col in tz_cols:
        out[col] = out[col] - floor_z
    return out


def load_and_remap(filepath):
    """Read the MediaPipe CSV and build a DataFrame with TX/TY/TZ columns
    for each of the 14 output body-part labels."""
    df = pd.read_csv(filepath)

    required = list(DIRECT_MAP.keys()) + ["left_shoulder", "right_shoulder"]
    missing = [lm for lm in set(required) if f"{lm}_x" not in df.columns]
    if missing:
        raise ValueError(f"Missing expected MediaPipe columns for: {missing}")

    out = pd.DataFrame()
    out["Frame"] = df["frame"].astype(int)
    out["SubFrame"] = 0

    def remap_axes(mp_x, mp_y, mp_z):
        return mp_x, mp_z, -mp_y  # TX, TY, TZ

    for landmark, label in DIRECT_MAP.items():
        tx, ty, tz = remap_axes(df[f"{landmark}_x"], df[f"{landmark}_y"], df[f"{landmark}_z"])
        out[f"{label}_TX"] = tx
        out[f"{label}_TY"] = ty
        out[f"{label}_TZ"] = tz

    # Synthesize chest = shoulder midpoint
    ls_x, ls_y, ls_z = remap_axes(df["left_shoulder_x"], df["left_shoulder_y"], df["left_shoulder_z"])
    rs_x, rs_y, rs_z = remap_axes(df["right_shoulder_x"], df["right_shoulder_y"], df["right_shoulder_z"])
    out["chest_TX"] = (ls_x + rs_x) / 2.0
    out["chest_TY"] = (ls_y + rs_y) / 2.0
    out["chest_TZ"] = (ls_z + rs_z) / 2.0

    # Re-reference TZ from MediaPipe's hip-centered origin to Vicon's
    # floor-referenced origin (floor == 0).
    floor_z = estimate_floor_z(out)
    out = apply_floor_offset(out, floor_z)
    print(f"      Floor estimated at raw TZ={floor_z:.1f}mm (hip-relative) -> shifted so floor=0 (Vicon convention)")

    return out


def detect_toss_and_racket_hand(df):
    """
    Determine which hand is the ball-toss hand vs. the racket hand.
    Same rule as format_data.py's identify_ball_hand(): whichever hand's
    TZ first exceeds the head's TZ, frame by frame, is the toss hand.
    Works for both left- and right-handed servers.
    """
    head_tz = df["head_TZ"].values
    lh_tz = df["left_hand_TZ"].values
    rh_tz = df["right_hand_TZ"].values

    for i in range(len(df)):
        hz = head_tz[i]
        if np.isnan(hz):
            continue
        if not np.isnan(lh_tz[i]) and lh_tz[i] > hz:
            return "left_hand", "right_hand", i
        if not np.isnan(rh_tz[i]) and rh_tz[i] > hz:
            return "right_hand", "left_hand", i

    raise ValueError("Neither hand ever exceeded head TZ -- cannot determine toss hand.")


def compute_peaks(df, toss_label, racket_label):
    toss_tz = df[f"{toss_label}_TZ"].dropna()
    peak1_idx = int(toss_tz.idxmax())
    peak1_val = float(toss_tz[peak1_idx])

    racket_tz = df[f"{racket_label}_TZ"].dropna()
    peak2_idx = int(racket_tz.idxmax())
    peak2_val = float(racket_tz[peak2_idx])

    return (
        {"frame_idx": peak1_idx, "col": f"{toss_label}_TZ", "value": peak1_val, "label": "Ball Toss"},
        {"frame_idx": peak2_idx, "col": f"{racket_label}_TZ", "value": peak2_val, "label": "Follow Through"},
    )


def export_formatted_csv(df, peak1, peak2, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)

        f.write(f"PEAK1={df['Frame'].iloc[peak1['frame_idx']]},{peak1['col']},{peak1['label']}\n")
        f.write(f"PEAK2={df['Frame'].iloc[peak2['frame_idx']]},{peak2['col']},{peak2['label']}\n")
        for name in SNAPSHOT_NAMES:
            f.write(f"SNAPSHOT={name},0\n")

        # Row 0: marker names
        row0 = ["", ""]
        for label in OUTPUT_ORDER:
            row0 += [label, "", ""]
        writer.writerow(row0)

        # Row 1: TX/TY/TZ
        row1 = ["", ""]
        for _ in OUTPUT_ORDER:
            row1 += ["TX", "TY", "TZ"]
        writer.writerow(row1)

        # Row 2: units
        row2 = ["", ""]
        for _ in OUTPUT_ORDER:
            row2 += ["mm", "mm", "mm"]
        writer.writerow(row2)

        # Data rows
        for _, row in df.iterrows():
            out_row = [int(row["Frame"]), int(row["SubFrame"])]
            for label in OUTPUT_ORDER:
                out_row += [
                    f"{row[f'{label}_TX']:.6g}",
                    f"{row[f'{label}_TY']:.6g}",
                    f"{row[f'{label}_TZ']:.6g}",
                ]
            writer.writerow(out_row)

    print(f"  Formatted CSV saved -> {out_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python format_data_mediapipe.py <mediapipe_csv> [output_csv]")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        base = os.path.splitext(os.path.basename(filepath))[0]
        out_path = os.path.join(os.path.dirname(os.path.abspath(filepath)), base + "_formatted.csv")

    print(f"\n[1/3] Reading MediaPipe CSV: {filepath}")
    df = load_and_remap(filepath)
    print(f"      {len(df)} frames | 14 body parts (direct mapping, no detection)")

    print("\n[2/3] Determining toss hand vs. racket hand...")
    toss_label, racket_label, first_frame = detect_toss_and_racket_hand(df)
    print(f"  Toss hand   -> {toss_label}  (first exceeded head at df idx {first_frame})")
    print(f"  Racket hand -> {racket_label}")

    peak1, peak2 = compute_peaks(df, toss_label, racket_label)
    print(f"  Peak 1 (Ball Toss)      -> {peak1['col']}  frame {df['Frame'].iloc[peak1['frame_idx']]}  TZ = {peak1['value']:.1f} mm")
    print(f"  Peak 2 (Follow Through) -> {peak2['col']}  frame {df['Frame'].iloc[peak2['frame_idx']]}  TZ = {peak2['value']:.1f} mm")

    print("\n[3/3] Writing formatted CSV...")
    export_formatted_csv(df, peak1, peak2, out_path)
    print("\nDone. Run find_snapshots.py on the output to fill in snapshot frames.\n")


if __name__ == "__main__":
    main()
