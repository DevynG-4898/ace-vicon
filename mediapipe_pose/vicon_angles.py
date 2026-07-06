#!/usr/bin/env python3
"""
vicon_angles.py
Computes the same 6 angles as MediaPipe's serve_angles.py
but from Vicon XYZ marker data.

Angle definitions are identical to TennisServeAnalyzer so that
Vicon and MediaPipe outputs are directly comparable.

Usage:
    python vicon_angles.py my_combined_serve.csv
    python vicon_angles.py my_combined_serve.csv --hand left
"""

import os
import sys
import math
import csv
import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Vicon column mapping → same body parts as MediaPipe (right-handed serve)
# ---------------------------------------------------------------------------
# MediaPipe right-hand serve uses:
#   shoulder     = right shoulder (12)
#   opp_shoulder = left shoulder  (11)
#   elbow        = right elbow    (14)
#   wrist        = right wrist    (16)
#   index        = right index    (20)  ← no finger marker in Vicon, use righthand
#   hip (serve)  = right hip      (24)
#   hip (opp)    = left hip       (23)
#   knee         = right knee     (26)
#   ankle        = right ankle    (28)  ← use rightfoot as proxy

RIGHT_HAND_MARKERS = {
    'shoulder':     'rightshoulder',
    'opp_shoulder': 'leftshoulder',
    'elbow':        'rightelbow',
    'wrist':        'righthand',      # Vicon has no separate wrist; righthand ≈ wrist
    'index':        'righthand',      # no finger marker → same as wrist (wrist angle ~0 or use chest vector)
    'serve_hip':    'righthip',
    'opp_hip':      'lefthip',
    'knee':         'rightknee',
    'ankle':        'rightfoot',
    'left_shoulder': 'leftshoulder',
    'right_shoulder': 'rightshoulder',
    'left_hip':     'lefthip',
    'right_hip':    'righthip',
    'chest':        'chest',
    'head':         'head',
}

LEFT_HAND_MARKERS = {
    'shoulder':     'leftshoulder',
    'opp_shoulder': 'rightshoulder',
    'elbow':        'leftelbow',
    'wrist':        'lefthand',
    'index':        'lefthand',
    'serve_hip':    'lefthip',
    'opp_hip':      'righthip',
    'knee':         'leftknee',
    'ankle':        'leftfoot',
    'left_shoulder': 'leftshoulder',
    'right_shoulder': 'rightshoulder',
    'left_hip':     'lefthip',
    'right_hip':    'righthip',
    'chest':        'chest',
    'head':         'head',
}


# ---------------------------------------------------------------------------
# Math helpers (identical logic to serve_angles.py)
# ---------------------------------------------------------------------------

def angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle at p2 formed by p1-p2-p3 in 3D (degrees). Same as TennisServeAnalyzer._angle."""
    v1 = p1 - p2
    v2 = p3 - p2
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float('nan')
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos))


def trunk_lean(mid_shoulder: np.ndarray, mid_hip: np.ndarray) -> float:
    """Same as TennisServeAnalyzer.get_trunk_lean_angle."""
    d = mid_shoulder - mid_hip
    horizontal = math.sqrt(d[0]**2 + d[2]**2)
    return math.degrees(math.atan2(horizontal, abs(d[1])))


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_vicon_csv(path: str) -> list[dict]:
    """
    Parse Vicon CSV with multi-row header:
      Row 0: body part names (repeated across TX/TY/TZ triplets)
      Row 1: TX / TY / TZ
      Row 2: units (mm)
      Row 3+: data

    Returns list of dicts: {marker_name: np.array([x, y, z]) or None, 'frame': int}
    """
    with open(path, newline='') as f:
        raw = list(csv.reader(f))

    # Row 0: Frame, Sub Frame, leftshoulder,,, head,,, ...
    # Row 1: ,, TX, TY, TZ, TX, TY, TZ, ...
    # Row 2: units
    # Row 3+: data

    header_names = raw[0]   # body part names
    header_axes  = raw[1]   # TX/TY/TZ

    # Build column index: marker_name -> (col_x, col_y, col_z)
    marker_cols = {}
    i = 2  # skip Frame, Sub Frame
    while i < len(header_names):
        name = header_names[i].strip().lower()
        if name and name not in ('frames',):
            # next three cols should be TX, TY, TZ
            marker_cols[name] = (i, i + 1, i + 2)
            i += 3
        else:
            i += 1

    # Parse data rows (skip first 3 header rows)
    rows = []
    for raw_row in raw[3:]:
        if not raw_row or not raw_row[0].strip().isdigit():
            continue
        frame = int(raw_row[0])
        entry = {'frame': frame}
        for name, (cx, cy, cz) in marker_cols.items():
            try:
                x = float(raw_row[cx])
                y = float(raw_row[cy])
                z = float(raw_row[cz])
                entry[name] = np.array([x, y, z])
            except (ValueError, IndexError):
                entry[name] = None  # missing / occluded
        rows.append(entry)

    return rows


# ---------------------------------------------------------------------------
# Angle computation per frame
# ---------------------------------------------------------------------------

def compute_angles(row: dict, markers: dict) -> dict | None:
    """
    Compute all 6 angles for one frame using the same math as serve_angles.py.
    Returns None if any required marker is missing.
    """
    def get(key):
        name = markers[key]
        return row.get(name)

    # Fetch all needed points
    shoulder     = get('shoulder')
    opp_shoulder = get('opp_shoulder')
    elbow        = get('elbow')
    wrist        = get('wrist')
    index        = get('index')       # same as wrist in Vicon
    serve_hip    = get('serve_hip')
    knee         = get('knee')
    ankle        = get('ankle')
    l_shoulder   = get('left_shoulder')
    r_shoulder   = get('right_shoulder')
    l_hip        = get('left_hip')
    r_hip        = get('right_hip')

    # Check required markers for each angle individually
    angles = {}

    # 1. shoulder_angle: opp_shoulder → shoulder → elbow
    if all(v is not None for v in [opp_shoulder, shoulder, elbow]):
        angles['shoulder_angle'] = angle_3d(opp_shoulder, shoulder, elbow)
    else:
        angles['shoulder_angle'] = float('nan')

    # 2. elbow_angle: shoulder → elbow → wrist
    if all(v is not None for v in [shoulder, elbow, wrist]):
        angles['elbow_angle'] = angle_3d(shoulder, elbow, wrist)
    else:
        angles['elbow_angle'] = float('nan')

    # 3. wrist_angle: elbow → wrist → index
    #    Vicon has no finger marker so index == wrist → angle is undefined.
    #    Use chest as a proxy for the forearm extension direction instead.
    chest = row.get('chest')
    if all(v is not None for v in [elbow, wrist, chest]):
        # Direction from wrist toward chest gives an approximate wrist extension
        angles['wrist_angle'] = angle_3d(elbow, wrist, chest)
    else:
        angles['wrist_angle'] = float('nan')

    # 4. hip_rotation: angle between shoulder vector and hip vector
    if all(v is not None for v in [l_shoulder, r_shoulder, l_hip, r_hip]):
        sv = r_shoulder - l_shoulder
        hv = r_hip - l_hip
        n1, n2 = np.linalg.norm(sv), np.linalg.norm(hv)
        if n1 > 1e-6 and n2 > 1e-6:
            cos = np.clip(np.dot(sv, hv) / (n1 * n2), -1.0, 1.0)
            angles['hip_rotation'] = math.degrees(math.acos(cos))
        else:
            angles['hip_rotation'] = float('nan')
    else:
        angles['hip_rotation'] = float('nan')

    # 5. knee_angle: serve_hip → knee → ankle
    if all(v is not None for v in [serve_hip, knee, ankle]):
        angles['knee_angle'] = angle_3d(serve_hip, knee, ankle)
    else:
        angles['knee_angle'] = float('nan')

    # 6. trunk_lean: midpoint(shoulders) → midpoint(hips) deviation from vertical
    if all(v is not None for v in [l_shoulder, r_shoulder, l_hip, r_hip]):
        mid_sh = (l_shoulder + r_shoulder) / 2
        mid_hp = (l_hip + r_hip) / 2
        angles['trunk_lean'] = trunk_lean(mid_sh, mid_hp)
    else:
        angles['trunk_lean'] = float('nan')

    return angles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute MediaPipe-equivalent angles from Vicon CSV")
    parser.add_argument('csv', help="Path to Vicon CSV (e.g. my_combined_serve.csv)")
    parser.add_argument('--hand', default='right', choices=['right', 'left'])
    parser.add_argument('--out', default=None, help="Output CSV path (default: <input>_angles.csv)")
    args = parser.parse_args()

    markers = RIGHT_HAND_MARKERS if args.hand == 'right' else LEFT_HAND_MARKERS

    print(f"Loading: {args.csv}")
    rows = load_vicon_csv(args.csv)
    print(f"  {len(rows)} frames loaded")

    out_path = args.out or os.path.splitext(args.csv)[0] + '_angles.csv'

    ANGLE_COLS = ['shoulder_angle', 'elbow_angle', 'wrist_angle',
                  'hip_rotation', 'knee_angle', 'trunk_lean']

    written = 0
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'time_s'] + ANGLE_COLS)

        for row in rows:
            angles = compute_angles(row, markers)
            fps = 100  # Vicon default; adjust if needed
            time_s = row['frame'] / fps

            if angles:
                writer.writerow([
                    row['frame'],
                    f"{time_s:.4f}",
                    *[f"{angles[c]:.2f}" if not math.isnan(angles[c]) else 'N/A'
                      for c in ANGLE_COLS]
                ])
                written += 1

    print(f"  Angles written: {written} frames")
    print(f"  Output: {out_path}")
    print("\nNote: wrist_angle uses chest as proxy (no finger marker in Vicon).")
    print("This is comparable to MediaPipe wrist_angle only approximately.")


if __name__ == '__main__':
    main()