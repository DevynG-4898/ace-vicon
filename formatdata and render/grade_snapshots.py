"""
grade_snapshots.py
--------------------------
Compares two _formatted.csv files (as produced by format_data.py or
format_data_mediapipe.py, then run through find_snapshots.py) directly
against each other -- snapshot name to matching snapshot name, across
several joint angles. No DTW / timeline alignment is used: this assumes
both files already have SNAPSHOT= metadata filled in, and each snapshot
in file A is compared only to the same-named snapshot in file B.

Because both the Vicon path and the MediaPipe path now share the same
14-marker, floor-referenced, 8-snapshot schema, either file can act as
"customer" or "reference" -- a Vicon reference can grade a MediaPipe
customer serve, or vice versa.

Usage:
    python grade_snapshots.py <customer_formatted.csv> <reference_formatted.csv>
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 1. SNAPSHOT NAMES + WEIGHTS
#    Contact and the trophy/cocking position (peak_racket_arm) carry the
#    most weight since they're the power-generating / ball-strike moments.
#    Starting point only -- adjust freely.
# ─────────────────────────────────────────────

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

SNAPSHOT_WEIGHTS = {
    'start_pose':          0.05,
    'hand_cross':          0.05,
    'flat_racket_arm':     0.10,
    'peak_racket_arm':     0.20,
    'contact':             0.30,
    'hand_cross_2':        0.05,
    'racket_deceleration': 0.15,
    'finish_pose':         0.10,
}

SNAPSHOT_CONTEXT = {
    'start_pose':          "your starting stance and initial arm position",
    'hand_cross':          "the toss/racket hand hand-off as your arms swing apart",
    'flat_racket_arm':     "the racket-arm 'flat' position heading into the backswing",
    'peak_racket_arm':     "the trophy/cocking position -- the key power-loading moment",
    'contact':             "ball contact -- extension and alignment at the strike",
    'hand_cross_2':        "the arms crossing again during follow-through",
    'racket_deceleration': "controlled arm slowdown after contact",
    'finish_pose':         "your follow-through and finish position",
}

# ─────────────────────────────────────────────
# 2. SCORING THRESHOLDS (degrees difference)
#    Same tiering as tennis_serve_grader.py, reused for consistency.
# ─────────────────────────────────────────────

ANGLE_THRESHOLDS = [
    (5,   95, "Excellent"),
    (15,  80, "Good"),
    (30,  60, "Fair"),
    (999, 35, "Poor"),
]

TIER_ICON = {"Excellent": "\u2713", "Good": "~", "Fair": "!", "Poor": "\u2717"}

JOINT_LABELS = {
    "racket_elbow":    "Racket elbow",
    "racket_shoulder": "Racket shoulder",
    "toss_elbow":      "Toss elbow",
    "left_knee":       "Left knee",
    "right_knee":      "Right knee",
}

FEEDBACK_TEMPLATES = {
    "Fair": "{joint} deviates noticeably at {snapshot_label}. Focus on {context}.",
    "Poor": "{joint} is significantly misaligned at {snapshot_label}. Prioritize drills for {context}.",
}

# ─────────────────────────────────────────────
# 3. FILE PARSING
#    Mirrors find_snapshots.py's reader so both scripts stay in sync on
#    what the _formatted.csv structure looks like.
# ─────────────────────────────────────────────

def read_formatted_csv(filepath: str):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    peaks = {}
    snapshots = {}
    meta_end = 0

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("PEAK1="):
            parts = s[6:].split(',')
            peaks['peak1'] = {'frame': int(parts[0]), 'col': parts[1], 'label': parts[2]}
        elif s.startswith("PEAK2="):
            parts = s[6:].split(',')
            peaks['peak2'] = {'frame': int(parts[0]), 'col': parts[1], 'label': parts[2]}
        elif s.startswith("SNAPSHOT="):
            parts = s[9:].split(',')
            snapshots[parts[0]] = int(parts[1])
        else:
            meta_end = i
            break

    df_raw = pd.read_csv(filepath, header=None, skiprows=meta_end, dtype=str)
    p_row = df_raw.iloc[0].values
    a_row = df_raw.iloc[1].values

    columns = []
    for col_idx in range(df_raw.shape[1]):
        if col_idx == 0:
            columns.append("Frame")
        elif col_idx == 1:
            columns.append("SubFrame")
        else:
            base_idx = ((col_idx - 2) // 3) * 3 + 2
            base_name = str(p_row[base_idx]).strip()
            axis = str(a_row[col_idx]).strip()
            columns.append(f"{base_name}_{axis}")

    data_rows = df_raw.iloc[3:].copy()
    data_rows.columns = columns
    for col in columns:
        data_rows[col] = pd.to_numeric(data_rows[col], errors='coerce')
    data_rows['Frame'] = data_rows['Frame'].astype(int)
    df = data_rows.reset_index(drop=True)

    return df, peaks, snapshots


def racket_and_toss_side(df: pd.DataFrame, peaks: dict) -> tuple[str, str]:
    """
    Which side ('left'/'right') is the racket arm vs. the toss arm.

    Determined directly from the labeled left_hand_TZ/right_hand_TZ data
    at the PEAK1/PEAK2 frames (whichever hand is higher at the PEAK2 frame
    is the racket hand, by the same definition used to find PEAK2 in the
    first place) -- NOT by parsing the PEAK metadata's 'col' field. That
    field's naming convention isn't consistent across formatters: MediaPipe
    files write semantic labels like 'right_hand_TZ', but Vicon-sourced
    files can still reference the pre-label raw track name (e.g.
    'Track1248_TZ'), which isn't parseable as left/right. Reading the
    actual data sidesteps that inconsistency entirely.
    """
    def side_at(frame_num):
        idx = frame_to_idx(df, frame_num)
        if idx is None:
            raise ValueError(f"Frame {frame_num} not found in data -- cannot determine racket/toss side.")
        lh, rh = df['left_hand_TZ'].iloc[idx], df['right_hand_TZ'].iloc[idx]
        if np.isnan(lh) or np.isnan(rh):
            raise ValueError(f"left_hand_TZ/right_hand_TZ is NaN at frame {frame_num} -- cannot determine side.")
        return 'left' if lh > rh else 'right'

    racket_side = side_at(peaks['peak2']['frame'])
    toss_side = side_at(peaks['peak1']['frame'])
    return racket_side, toss_side


# ─────────────────────────────────────────────
# 4. JOINT ANGLE DEFINITIONS
# ─────────────────────────────────────────────

def build_joint_defs(racket_side: str, toss_side: str) -> dict:
    """
    Each joint angle is defined as (marker_a, vertex_marker, marker_b);
    the angle is computed at vertex_marker. Keys are side-agnostic
    ('racket_elbow', not 'right_elbow') so results from a lefty and a
    righty serve line up under the same joint name.
    """
    return {
        "racket_elbow":    (f"{racket_side}_shoulder", f"{racket_side}_elbow", f"{racket_side}_hand"),
        "racket_shoulder": ("chest", f"{racket_side}_shoulder", f"{racket_side}_elbow"),
        "toss_elbow":      (f"{toss_side}_shoulder", f"{toss_side}_elbow", f"{toss_side}_hand"),
        "left_knee":       ("left_hip", "left_knee", "left_foot"),
        "right_knee":      ("right_hip", "right_knee", "right_foot"),
    }


def compute_joint_angle(df: pd.DataFrame, frame_idx: int, marker_a: str, vertex: str, marker_b: str):
    """
    Angle in degrees at `vertex`, between the vectors to marker_a and
    marker_b. Returns None if any marker is occluded (NaN) at this frame,
    rather than letting NaN silently flow through the arccos math -- an
    occluded marker is a data gap, not evidence of a bad angle match, and
    should be reported as such instead of defaulting to a "Poor" score.
    """
    row = df.iloc[frame_idx]
    a = np.array([row[f"{marker_a}_TX"], row[f"{marker_a}_TY"], row[f"{marker_a}_TZ"]], dtype=float)
    v = np.array([row[f"{vertex}_TX"],   row[f"{vertex}_TY"],   row[f"{vertex}_TZ"]],   dtype=float)
    b = np.array([row[f"{marker_b}_TX"], row[f"{marker_b}_TY"], row[f"{marker_b}_TZ"]], dtype=float)

    if np.isnan(a).any() or np.isnan(v).any() or np.isnan(b).any():
        return None

    v1, v2 = a - v, b - v
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def score_angle(diff_deg: float) -> tuple[int, str]:
    for threshold, points, tier in ANGLE_THRESHOLDS:
        if diff_deg < threshold:
            return points, tier
    return 35, "Poor"


def frame_to_idx(df: pd.DataFrame, frame_num: int):
    matches = df.index[df['Frame'] == frame_num].tolist()
    return matches[0] if matches else None


# ─────────────────────────────────────────────
# 5. MAIN GRADING FUNCTION
# ─────────────────────────────────────────────

def grade_serve(customer_path: str, reference_path: str) -> dict:
    cust_df, cust_peaks, cust_snaps = read_formatted_csv(customer_path)
    ref_df,  ref_peaks,  ref_snaps  = read_formatted_csv(reference_path)

    cust_racket, cust_toss = racket_and_toss_side(cust_df, cust_peaks)
    ref_racket,  ref_toss  = racket_and_toss_side(ref_df, ref_peaks)

    cust_joints = build_joint_defs(cust_racket, cust_toss)
    ref_joints  = build_joint_defs(ref_racket, ref_toss)
    joint_names = list(cust_joints.keys())

    results = {"snapshots": {}, "customer_racket_side": cust_racket, "reference_racket_side": ref_racket}
    weighted_total = 0.0
    weight_used = 0.0

    for snap in SNAPSHOT_NAMES:
        cust_frame = cust_snaps.get(snap, 0)
        ref_frame  = ref_snaps.get(snap, 0)

        if not cust_frame or not ref_frame:
            missing = []
            if not cust_frame:
                missing.append("customer")
            if not ref_frame:
                missing.append("reference")
            results["snapshots"][snap] = {
                "snapshot_score": None,
                "summary": f"not found in {' and '.join(missing)}",
            }
            continue

        cust_idx = frame_to_idx(cust_df, cust_frame)
        ref_idx  = frame_to_idx(ref_df, ref_frame)
        if cust_idx is None or ref_idx is None:
            results["snapshots"][snap] = {"snapshot_score": None, "summary": "frame not found in data"}
            continue

        joint_results = {}
        joint_scores = []
        for jname in joint_names:
            ca, cv, cb = cust_joints[jname]
            ra, rv, rb = ref_joints[jname]
            try:
                cust_angle = compute_joint_angle(cust_df, cust_idx, ca, cv, cb)
                ref_angle  = compute_joint_angle(ref_df, ref_idx, ra, rv, rb)
            except KeyError as e:
                print(f"  WARNING: {jname} at {snap} -- marker column missing ({e}). Skipping this joint.")
                continue

            if cust_angle is None or ref_angle is None:
                joint_results[jname] = {"tier": "No data", "note": "marker occluded at this frame"}
                continue

            diff = abs(cust_angle - ref_angle)
            score, tier = score_angle(diff)
            joint_results[jname] = {
                "customer_angle": round(cust_angle, 1),
                "reference_angle": round(ref_angle, 1),
                "diff": round(diff, 1),
                "score": score,
                "tier": tier,
            }
            joint_scores.append(score)

        if not joint_scores:
            results["snapshots"][snap] = {"snapshot_score": None, "summary": "no joints computable"}
            continue

        snapshot_score = round(sum(joint_scores) / len(joint_scores), 1)
        results["snapshots"][snap] = {
            "customer_frame": cust_frame,
            "reference_frame": ref_frame,
            "joints": joint_results,
            "snapshot_score": snapshot_score,
        }

        w = SNAPSHOT_WEIGHTS.get(snap, 0)
        weighted_total += snapshot_score * w
        weight_used += w

    if weight_used == 0:
        results["overall_score"] = None
        results["overall_grade"] = "No comparable snapshots found."
        return results

    overall = round(weighted_total / weight_used, 1)
    results["overall_score"] = overall

    if overall >= 90:
        results["overall_grade"] = "A -- Pro-level serve"
    elif overall >= 75:
        results["overall_grade"] = "B -- Strong serve, minor adjustments needed"
    elif overall >= 60:
        results["overall_grade"] = "C -- Developing serve, focused practice recommended"
    else:
        results["overall_grade"] = "D -- Fundamentals need significant work"

    return results


# ─────────────────────────────────────────────
# 6. REPORT PRINTER
# ─────────────────────────────────────────────

def print_report(results: dict) -> None:
    sep = "\u2500" * 78
    print(f"\n{'ACE SERVE COMPARISON — MULTI-JOINT SNAPSHOT REPORT':^78}")
    print(f"  Customer racket side: {results['customer_racket_side']}   "
          f"Reference racket side: {results['reference_racket_side']}")
    print(sep)

    for snap in SNAPSHOT_NAMES:
        s = results["snapshots"].get(snap)
        label = snap.replace("_", " ")

        if not s or s.get("snapshot_score") is None:
            reason = s.get("summary", "no data") if s else "no data"
            print(f"\n  {label}  \u2014 {reason}")
            continue

        weight_pct = int(SNAPSHOT_WEIGHTS.get(snap, 0) * 100)
        print(f"\n  {label}  ({weight_pct}% of grade)   "
              f"customer f{s['customer_frame']}  vs  reference f{s['reference_frame']}")
        print(f"  Snapshot score: {s['snapshot_score']}/100")

        for jname, j in s["joints"].items():
            jlabel = JOINT_LABELS.get(jname, jname.replace("_", " ").title())
            if j["tier"] == "No data":
                print(f"    [?] {jlabel:16s} No data    ({j['note']})")
                continue
            icon = TIER_ICON[j["tier"]]
            print(f"    [{icon}] {jlabel:16s} {j['tier']:9s} Score:{j['score']:3d}  "
                  f"Cust:{j['customer_angle']:6.1f}\u00b0  Ref:{j['reference_angle']:6.1f}\u00b0  Diff:{j['diff']:5.1f}\u00b0")
            if j["tier"] in FEEDBACK_TEMPLATES:
                feedback = FEEDBACK_TEMPLATES[j["tier"]].format(
                    joint=jlabel, snapshot_label=label, context=SNAPSHOT_CONTEXT[snap]
                )
                print(f"        \u2192 {feedback}")

    print(f"\n{sep}")
    print(f"  OVERALL SCORE : {results['overall_score']}/100")
    print(f"  GRADE         : {results['overall_grade']}")
    print(sep)


# ─────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print("Usage: python grade_snapshots.py <customer_formatted.csv> <reference_formatted.csv>")
        sys.exit(1)

    customer_path, reference_path = sys.argv[1], sys.argv[2]
    for p in (customer_path, reference_path):
        if not os.path.exists(p):
            print(f"File not found: {p}")
            sys.exit(1)

    print(f"\nCustomer serve:  {customer_path}")
    print(f"Reference serve: {reference_path}")

    results = grade_serve(customer_path, reference_path)
    print_report(results)


if __name__ == "__main__":
    main()
