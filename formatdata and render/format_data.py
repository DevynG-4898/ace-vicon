"""
format_data.py
--------------
Takes a raw unmarked Vicon CSV and produces a fully formatted output CSV
with labeled columns and metadata rows at the top.

Processing steps:
  1. Head         — greatest average TZ across all frames
  2. Ball hand    — first marker (frame by frame) whose TZ exceeds the head's
                    TZ at that same frame
  3. Peak 1 (Ball Toss)      — frame where ball hand TZ is maximum
  4. Racket hand  — marker (excluding head + ball hand) with highest global TZ max
  5. Peak 2 (Follow Through) — frame where racket hand TZ is maximum
  6. Elbow1       — tallest remaining marker at the exact frame of Peak 2
  7. Swing window — Peak 1 frame → Peak 2 frame
  8. Body parts   — remaining markers assigned by TZ ranking within swing window

Output CSV metadata (top of file):
  PEAK1=<frame>,<col>,<label>
  PEAK2=<frame>,<col>,<label>
  SNAPSHOT=start_pose,0
  SNAPSHOT=flat_racket_arm,0
  SNAPSHOT=peak_racket_arm,0
  SNAPSHOT=contact,0
  SNAPSHOT=racket_deceleration,0
  SNAPSHOT=finish_pose,0
  <Vicon data with body part names in header row>

Run find_snapshots.py on the output to fill in the snapshot frames.

Usage:
    python format_data.py <path_to_csv>
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── 1. CSV PARSING ─────────────────────────────────────────────────────────────

def parse_vicon_csv(filepath):
    """
    Parse Vicon CSV structure:
      Row 0: marker names
      Row 1: TX / TY / TZ component labels
      Row 2: units (skipped)
      Row 3+: numeric data

    Returns:
        df      : DataFrame with columns like Track1045_TX, Track1045_TY, Track1045_TZ
        tz_cols : list of TZ column names
        tracks  : list of track base names (e.g. 'Track1045') in column order
    """
    raw = pd.read_csv(filepath, header=None, low_memory=False)

    marker_row    = raw.iloc[0].fillna("").tolist()
    component_row = raw.iloc[1].fillna("").tolist()

    col_labels = []
    current_marker = ""
    for i, (m, c) in enumerate(zip(marker_row, component_row)):
        m = str(m).strip()
        c = str(c).strip()
        if m:
            current_marker = m
        if i == 0:
            col_labels.append("Frame")
        elif i == 1:
            col_labels.append("SubFrame")
        else:
            if "Track" in current_marker and "(" in current_marker:
                track_id = current_marker.split("(")[-1].rstrip(")")
                label = f"Track{track_id}_{c}"
            else:
                label = f"Col{i}_{c}"
            col_labels.append(label)

    df = raw.iloc[3:].reset_index(drop=True)
    df.columns = col_labels
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)

    tz_cols = [
        col_labels[i]
        for i, c in enumerate(component_row)
        if str(c).strip().upper() == 'TZ' and i >= 2
    ]

    # Derive unique track base names in order
    tracks = [col[:-3] for col in tz_cols]  # strip '_TZ'

    return df, tz_cols, tracks


# ── 2. MARKER IDENTIFICATION ───────────────────────────────────────────────────

def identify_head(df, tracks, excluded=None):
    """
    Head = track with greatest average TZ across all frames.
    Skips tracks in excluded set.
    """
    excluded = excluded or set()
    best_track = None
    best_avg = -np.inf

    for track in tracks:
        if track in excluded:
            continue
        tz_col = track + "_TZ"
        if tz_col not in df.columns:
            continue
        series = df[tz_col].dropna()
        if series.empty:
            continue
        avg = series.mean()
        if avg > best_avg:
            best_avg = avg
            best_track = track

    return best_track, best_avg


def identify_ball_hand(df, tracks, head_track):
    """
    Ball hand = first marker (frame by frame from frame 0) whose TZ exceeds
    the head marker's TZ at that exact same frame.
    No exclusions — any marker can be the ball hand.
    Head itself is the only marker skipped.
    """
    candidates = [t for t in tracks if t != head_track]

    head_tz = df[head_track + "_TZ"].values

    for fi in range(len(df)):
        head_z = head_tz[fi]
        if np.isnan(head_z):
            continue
        for track in candidates:
            tz_col = track + "_TZ"
            if tz_col not in df.columns:
                continue
            val = df[tz_col].iloc[fi]
            if np.isnan(val):
                continue
            if val > head_z:
                return track, fi  # first occurrence

    return None, None  # never happened


def identify_racket_hand(df, tracks, excluded=None):
    """
    Racket hand = marker (excluding head + ball hand) with the highest
    global TZ maximum across all frames.
    """
    excluded = excluded or set()
    best_track = None
    best_max = -np.inf

    for track in tracks:
        if track in excluded:
            continue
        tz_col = track + "_TZ"
        if tz_col not in df.columns:
            continue
        series = df[tz_col].dropna()
        if series.empty:
            continue
        tz_max = float(series.max())
        if tz_max > best_max:
            best_max = tz_max
            best_track = track

    return best_track, best_max


def identify_elbow1(df, tracks, excluded, peak2_frame):
    """
    Elbow1 = tallest remaining marker (excluding head + ball hand + racket hand)
    at the exact frame of Peak 2 (Follow Through).
    """
    best_track = None
    best_val = -np.inf

    for track in tracks:
        if track in excluded:
            continue
        tz_col = track + "_TZ"
        if tz_col not in df.columns:
            continue
        val = df.iloc[peak2_frame][tz_col]
        if np.isnan(val):
            continue
        if val > best_val:
            best_val = val
            best_track = track

    return best_track, best_val


# ── 3. BODY PART ASSIGNMENT ────────────────────────────────────────────────────

# Ordered body part labels from highest to lowest Z
BODY_PART_ORDER = [
    'elbow2',
    'shoulder1',
    'shoulder2',
    'chest',
    'hip1',
    'hip2',
    'knee1',
    'knee2',
]

def avg_position_frames(df, track, frame_start, frame_end):
    """Return mean TX, TY, TZ for a track across a given frame range."""
    window = df.iloc[frame_start:frame_end + 1]
    tx = float(window[track + "_TX"].dropna().mean())
    ty = float(window[track + "_TY"].dropna().mean())
    tz = float(window[track + "_TZ"].dropna().mean())
    return np.array([tx, ty, tz])


def determine_forward_axis(df, racket_hand_track, peak1_frame, peak2_frame):
    """
    Forward = whichever horizontal axis (X or Y) the racket hand moves
    most along between Peak 1 (Ball Toss) and Peak 2 (Follow Through),
    signed by the direction of that movement. The swing itself defines
    "forward."

    Snaps to one of the four cardinal directions:
    (1,0,0), (-1,0,0), (0,1,0), or (0,-1,0).
    """
    start = df.iloc[peak1_frame]
    end   = df.iloc[peak2_frame]
    dx = float(end[racket_hand_track + "_TX"] - start[racket_hand_track + "_TX"])
    dy = float(end[racket_hand_track + "_TY"] - start[racket_hand_track + "_TY"])

    if abs(dx) >= abs(dy):
        forward = np.array([1.0, 0.0, 0.0]) if dx >= 0 else np.array([-1.0, 0.0, 0.0])
    else:
        forward = np.array([0.0, 1.0, 0.0]) if dy >= 0 else np.array([0.0, -1.0, 0.0])

    return forward


def right_axis_from_forward(forward):
    """right = forward rotated 90 deg clockwise in the XY plane: (fy, -fx, 0)"""
    return np.array([forward[1], -forward[0], 0.0])


LATE_FRAME_THRESHOLD = 200  # if the first valid frame for a pair is beyond this, fall back to searching backward from the end


def first_valid_frame(df, track_a, track_b):
    """
    First frame index where both tracks have non-NaN TX/TY/TZ (handles
    markers that drop out / go NaN for stretches). Returns None if no
    such frame exists.
    """
    cols = [track_a + "_TX", track_a + "_TY", track_a + "_TZ",
            track_b + "_TX", track_b + "_TY", track_b + "_TZ"]
    valid = df[cols].notna().all(axis=1)
    if not valid.any():
        return None
    return int(valid.idxmax())


def last_valid_frame_in_window(df, track_a, track_b, window):
    """
    Search backward from the very last frame of the file, looking only
    within the last `window` frames, for a frame where both tracks have
    valid (non-NaN) coordinates. Returns the frame closest to the end
    that qualifies (i.e. the first hit when scanning backward), or None
    if no such frame exists in that window.
    """
    cols = [track_a + "_TX", track_a + "_TY", track_a + "_TZ",
            track_b + "_TX", track_b + "_TY", track_b + "_TZ"]
    n = len(df)
    start = max(0, n - window)
    valid = df[cols].iloc[start:n].notna().all(axis=1)
    if not valid.any():
        return None
    return int(valid[valid].index.max())


def side_score_at_frame(df, track, head_track, right, frame_idx):
    """Projection of (track - head) onto `right`, at a single frame. Positive → right side."""
    pos      = df.loc[frame_idx, [track + "_TX", track + "_TY", track + "_TZ"]].to_numpy(dtype=float)
    head_pos = df.loc[frame_idx, [head_track + "_TX", head_track + "_TY", head_track + "_TZ"]].to_numpy(dtype=float)
    rel = pos - head_pos
    return float(rel[0] * right[0] + rel[1] * right[1])


def orient_labels(df, labels_map, body_parts, peak1_frame, peak2_frame):
    """
    Rename paired body part labels to left_/right_.

    Forward = the horizontal axis (X or Y) the racket hand moves most
    along between Peak 1 (Ball Toss) and Peak 2 (Follow Through), signed
    by that movement's direction — the swing itself defines "forward."
    Right = forward rotated 90 deg clockwise in the XY plane.

    For each paired body part (hands, shoulders, elbows, hips, knees,
    feet), find the first frame where both markers in the pair have
    valid (non-NaN) coordinates, and compare their positions relative
    to the head — projected onto the right axis — at that single frame.
    Whichever is further toward "right" → right_, the other → left_.

    If that first valid frame is later than LATE_FRAME_THRESHOLD frames
    in (a marker took an unexpectedly long time to appear), that pair
    instead uses the frame closest to the END of the file, found by
    searching backward within the last LATE_FRAME_THRESHOLD frames.
    Pairs whose first valid frame is already within the threshold are
    left exactly as-is — this only kicks in for the pairs that need it.

    Returns updated labels_map and body_parts dicts with left_/right_ names.
    """
    head_track   = labels_map['head']
    racket_track = labels_map['racket_hand']

    forward = determine_forward_axis(df, racket_track, peak1_frame, peak2_frame)
    right   = right_axis_from_forward(forward)
    print(f"  Forward axis (racket hand swing direction) = {tuple(forward)}")
    print(f"  Right axis                                  = {tuple(right)}")

    new_labels_map = dict(labels_map)
    new_body_parts = dict(body_parts)

    # Pairs to orient: (generic_label_a, generic_label_b, right_name, left_name)
    PAIRS = [
        ('ball_hand', 'racket_hand', 'right_hand',     'left_hand'),
        ('shoulder1', 'shoulder2',   'right_shoulder', 'left_shoulder'),
        ('elbow1',    'elbow2',      'right_elbow',    'left_elbow'),
        ('hip1',      'hip2',        'right_hip',      'left_hip'),
        ('knee1',     'knee2',       'right_knee',     'left_knee'),
        ('foot1',     'foot2',       'right_foot',     'left_foot'),
    ]

    for a_label, b_label, right_name, left_name in PAIRS:
        a_track = new_labels_map.get(a_label) or new_body_parts.get(a_label)
        b_track = new_labels_map.get(b_label) or new_body_parts.get(b_label)
        if not a_track or not b_track:
            continue

        a_in_labels = a_label in new_labels_map
        b_in_labels = b_label in new_labels_map

        fi = first_valid_frame(df, a_track, b_track)

        if fi is not None and fi > LATE_FRAME_THRESHOLD:
            print(f"  NOTE: {a_label}/{b_label} first appear together at frame {fi} "
                  f"(> {LATE_FRAME_THRESHOLD}) — using the last {LATE_FRAME_THRESHOLD} "
                  f"frames instead, traced backward from the end.")
            fallback_fi = last_valid_frame_in_window(df, a_track, b_track, LATE_FRAME_THRESHOLD)
            if fallback_fi is not None:
                fi = fallback_fi
            else:
                print(f"  WARNING: no valid frame in the last {LATE_FRAME_THRESHOLD} frames "
                      f"either — falling back to frame {fi}.")

        if fi is None:
            print(f"  WARNING: no frame where both {a_label} and {b_label} are valid — skipping.")
            continue

        a_score = side_score_at_frame(df, a_track, head_track, right, fi)
        b_score = side_score_at_frame(df, b_track, head_track, right, fi)

        if a_score >= b_score:
            right_track, left_track = a_track, b_track
        else:
            right_track, left_track = b_track, a_track

        for label, src in [(a_label, a_in_labels), (b_label, b_in_labels)]:
            if src:
                new_labels_map.pop(label, None)
            else:
                new_body_parts.pop(label, None)

        if right_name in ('right_hand', 'right_elbow'):
            new_labels_map[right_name] = right_track
            new_labels_map[left_name]  = left_track
        else:
            new_body_parts[right_name] = right_track
            new_body_parts[left_name]  = left_track

    return new_labels_map, new_body_parts


def assign_body_parts(df, tracks, excluded, swing_start, swing_end):
    """
    Step 1 — elbow2 + shoulder1:
        Rank all remaining markers by TZ at the exact swing start frame.
        Highest → elbow2, second highest → shoulder1.

    Step 2 — Rank the rest by avg TZ within swing (highest to lowest):
        Slots 0-1  → temp chest/shoulder2 candidates
        Slots 2-3  → hip1, hip2
        Slots 4-5  → knee1, knee2
        Slots 6+   → unlabeled

    Step 3 — Resolve chest vs shoulder2 via hip midpoint:
        Compute hip midpoint = avg 3D position of hip1 + hip2 across swing.
        Candidate closest to hip midpoint → chest; other → shoulder2.

    Returns:
        assignments : dict of { label: track }
        elbow2      : track assigned as elbow2 (or None)
        elbow2_val  : TZ of elbow2 at swing start frame
        ranked      : list of (track, avg_tz) for markers after elbow2+shoulder1
    """
    swing_df = df.iloc[swing_start:swing_end + 1]
    candidates = [t for t in tracks if t not in excluded]

    # ── Step 1: elbow2 + shoulder1 — top 2 TZ at swing start frame ──
    start_tz_vals = []
    for track in candidates:
        tz_col = track + "_TZ"
        if tz_col not in df.columns:
            continue
        val = df.iloc[swing_start][tz_col]
        if pd.isna(val):
            continue
        start_tz_vals.append((track, float(val)))

    start_tz_vals.sort(key=lambda x: x[1], reverse=True)

    elbow2     = start_tz_vals[0][0] if len(start_tz_vals) > 0 else None
    elbow2_val = start_tz_vals[0][1] if len(start_tz_vals) > 0 else np.nan
    shoulder1  = start_tz_vals[1][0] if len(start_tz_vals) > 1 else None

    assignments = {}
    if elbow2:
        assignments['elbow2'] = elbow2
        candidates = [t for t in candidates if t != elbow2]
    if shoulder1:
        assignments['shoulder1'] = shoulder1
        candidates = [t for t in candidates if t != shoulder1]

    # ── Step 2: rank remaining by avg TZ within swing ──
    ranked = []
    for track in candidates:
        tz_col = track + "_TZ"
        if tz_col not in swing_df.columns:
            continue
        series = swing_df[tz_col].dropna()
        if series.empty:
            continue
        ranked.append((track, float(series.mean())))

    ranked.sort(key=lambda x: x[1], reverse=True)

    # Assign slots: [0,1]=temp chest/shoulder2 candidates, [2]=hip1, [3]=hip2,
    #               [4]=knee1, [5]=knee2, [6]=foot1, [7]=foot2, [8+]=unlabeled
    chest_shoulder_candidates = [ranked[i][0] for i in range(min(2, len(ranked)))]
    if len(ranked) > 2: assignments['hip1']  = ranked[2][0]
    if len(ranked) > 3: assignments['hip2']  = ranked[3][0]
    if len(ranked) > 4: assignments['knee1'] = ranked[4][0]
    if len(ranked) > 5: assignments['knee2'] = ranked[5][0]
    if len(ranked) > 6: assignments['foot1'] = ranked[6][0]
    if len(ranked) > 7: assignments['foot2'] = ranked[7][0]

    # ── Step 3: resolve chest vs shoulder2 via hip midpoint ──
    if len(chest_shoulder_candidates) == 2 and 'hip1' in assignments and 'hip2' in assignments:
        hip1_pos = avg_position_frames(df, assignments['hip1'], swing_start, swing_end)
        hip2_pos = avg_position_frames(df, assignments['hip2'], swing_start, swing_end)
        hip_mid  = (hip1_pos + hip2_pos) / 2.0

        dists = {}
        for track in chest_shoulder_candidates:
            pos = avg_position_frames(df, track, swing_start, swing_end)
            dists[track] = float(np.linalg.norm(pos - hip_mid))

        chest    = min(chest_shoulder_candidates, key=lambda t: dists[t])
        shoulder2 = max(chest_shoulder_candidates, key=lambda t: dists[t])
        assignments['chest']     = chest
        assignments['shoulder2'] = shoulder2
    elif len(chest_shoulder_candidates) >= 1:
        # Fallback: not enough hips identified, just assign by Z rank
        if len(chest_shoulder_candidates) > 0:
            assignments['shoulder2'] = chest_shoulder_candidates[0]
        if len(chest_shoulder_candidates) > 1:
            assignments['chest'] = chest_shoulder_candidates[1]

    return assignments, elbow2, elbow2_val, ranked


# ── 5. CSV EXPORT ──────────────────────────────────────────────────────────────

SNAPSHOT_NAMES = [
    'start_pose',
    'flat_racket_arm',
    'peak_racket_arm',
    'contact',
    'racket_deceleration',
    'finish_pose',
]


def export_annotated_csv(source_filepath, peaks, labels_map, body_parts, out_path=None):
    """
    Write formatted output CSV:
      PEAK1=<frame>,<col>,<label>
      PEAK2=<frame>,<col>,<label>
      SNAPSHOT=start_pose,0
      SNAPSHOT=flat_racket_arm,0
      SNAPSHOT=peak_racket_arm,0
      SNAPSHOT=contact,0
      SNAPSHOT=racket_deceleration,0
      SNAPSHOT=finish_pose,0
      <Vicon data with body part names in header row>

    Snapshot frames are placeholders (0) — run find_snapshots.py to fill them.
    """
    sorted_peaks = sorted(peaks, key=lambda x: x['frame_idx'])

    if out_path is None:
        base = os.path.splitext(os.path.basename(source_filepath))[0]
        out_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(out_dir, base + '_formatted.csv')
        if os.path.exists(out_path):
            counter = 2
            while os.path.exists(os.path.join(out_dir, f"{base}_formatted_{counter}.csv")):
                counter += 1
            out_path = os.path.join(out_dir, f"{base}_formatted_{counter}.csv")

    # Build track → label map
    all_labels = {}
    for role, track in labels_map.items():
        all_labels[track] = role
    for part, track in body_parts.items():
        all_labels[track] = part

    # Read original CSV lines
    with open(source_filepath, 'r') as f:
        lines = f.readlines()

    # Rewrite row 0: replace track names with body part labels
    import csv as _csv, io as _io
    reader = _csv.reader(_io.StringIO(lines[0]))
    marker_row = next(reader)

    new_marker_row = []
    for i, cell in enumerate(marker_row):
        cell_stripped = cell.strip().strip('"')
        if i < 2 or cell_stripped == '':
            new_marker_row.append(cell)
        else:
            if "Track" in cell_stripped and "(" in cell_stripped:
                track_id  = cell_stripped.split("(")[-1].rstrip(")")
                track_key = f"Track{track_id}"
                label = all_labels.get(track_key, cell_stripped)
            else:
                label = cell_stripped
            new_marker_row.append(label)

    out_buf = _io.StringIO()
    _csv.writer(out_buf).writerow(new_marker_row)
    new_header_line = out_buf.getvalue()

    with open(out_path, 'w', newline='') as f:
        # Peak metadata
        for i, p in enumerate(sorted_peaks, 1):
            f.write(f"PEAK{i}={p['frame_idx']},{p['col']},{p['label']}\n")
        # Snapshot placeholders
        for name in SNAPSHOT_NAMES:
            f.write(f"SNAPSHOT={name},0\n")
        # Labeled Vicon data
        f.write(new_header_line)
        f.writelines(lines[1:])

    print(f"  Formatted CSV saved → {out_path}")
    return out_path


# ── 5. CONFIRMATION PLOT ───────────────────────────────────────────────────────

def plot_segment(df, tz_cols, peaks, labels_map, body_parts=None):
    """
    2D confirmation plot:
      - All TZ columns as faint dots
      - Head / ball hand / racket hand TZ traces highlighted
      - Body part traces drawn in the swing segment
      - Two peak points marked
      - Timeline bar at bottom
    """
    peak_a = min(peaks, key=lambda x: x['frame_idx'])
    peak_b = max(peaks, key=lambda x: x['frame_idx'])
    seg_start    = peak_a['frame_idx']
    seg_end      = peak_b['frame_idx']
    total_frames = len(df)

    fig, (ax_main, ax_time) = plt.subplots(
        nrows=2, figsize=(14, 8),
        gridspec_kw={'height_ratios': [7, 1]},
    )
    fig.patch.set_facecolor('#1a1a2e')
    for ax in (ax_main, ax_time):
        ax.set_facecolor('#1a1a2e')

    # All TZ columns — faint background dots
    for col in tz_cols:
        series = df[col].dropna()
        ax_main.scatter(series.index, series.values,
                        s=2, alpha=0.15, color='#4cc9f0', linewidths=0)

    # Highlighted traces for identified markers (uses post-orientation names,
    # since plot_segment() is only ever called after orient_labels())
    highlight = {}
    if 'head' in labels_map:
        highlight[labels_map['head']] = ('#a78bfa', 'Head')
    if 'left_hand' in labels_map:
        highlight[labels_map['left_hand']] = ('#ffd166', 'Left Hand')
    if 'right_hand' in labels_map:
        highlight[labels_map['right_hand']] = ('#06d6a0', 'Right Hand')
    if 'left_elbow' in labels_map:
        highlight[labels_map['left_elbow']] = ('#ff6b6b', 'Left Elbow')
    if 'right_elbow' in labels_map:
        highlight[labels_map['right_elbow']] = ('#f72585', 'Right Elbow')
    for track, (color, name) in highlight.items():
        col = track + "_TZ"
        if col in df.columns:
            series = df[col].dropna()
            ax_main.plot(series.index, series.values,
                         color=color, linewidth=1.4, alpha=0.85, label=name)

    # Body part traces (drawn only, faintly, so they're visible but not dominant)
    BODY_COLORS = [
        '#f72585', '#b5179e', '#7209b7', '#560bad',
        '#480ca8', '#3a0ca3', '#3f37c9', '#4361ee',
    ]
    if body_parts:
        for i, (part, track) in enumerate(body_parts.items()):
            col = track + "_TZ"
            if col not in df.columns:
                continue
            series = df[col].dropna()
            color = BODY_COLORS[i % len(BODY_COLORS)]
            ax_main.plot(series.index, series.values,
                         color=color, linewidth=1.0, alpha=0.6, label=part)

    # Segment shading
    ax_main.axvspan(seg_start, seg_end, alpha=0.08, color='#f72585', zorder=1)
    ax_main.axvline(seg_start, color='#f72585', linewidth=1.2, linestyle='--', alpha=0.7)
    ax_main.axvline(seg_end,   color='#f72585', linewidth=1.2, linestyle='--', alpha=0.7)

    # Peak markers
    peak_styles = [
        ('#ffd166', 'Ball Toss'),
        ('#06d6a0', 'Follow Through'),
    ]
    for peak, (color, plabel) in zip(
        sorted(peaks, key=lambda x: x['frame_idx']), peak_styles
    ):
        val = df.loc[peak['frame_idx'], peak['col']]
        ax_main.scatter(peak['frame_idx'], val,
                        s=140, color=color, zorder=5,
                        edgecolors='white', linewidths=0.8)
        ax_main.annotate(
            f"{plabel}\n{peak['col']}\nFrame {peak['frame_idx']}  Z={val:.1f}",
            xy=(peak['frame_idx'], val),
            xytext=(30, 20), textcoords='offset points',
            fontsize=8, color=color,
            arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
            bbox=dict(boxstyle='round,pad=0.3', fc='#1a1a2e', ec=color, alpha=0.85),
        )

    ax_main.set_title('Vicon TZ — Identified Markers & Serve Segment',
                      color='white', fontsize=13, pad=12)
    ax_main.set_xlabel('Frame', color='#aaaaaa', fontsize=10)
    ax_main.set_ylabel('TZ Position (mm)', color='#aaaaaa', fontsize=10)
    ax_main.tick_params(colors='#aaaaaa')
    for spine in ax_main.spines.values():
        spine.set_edgecolor('#333355')
    ax_main.legend(facecolor='#1a1a2e', edgecolor='#333355',
                   labelcolor='white', fontsize=8)

    # Timeline bar
    ax_time.set_xlim(0, total_frames - 1)
    ax_time.set_ylim(0, 1)
    ax_time.barh(0.5, total_frames - 1, left=0, height=0.4,
                 color='#333355', align='center')
    ax_time.barh(0.5, seg_end - seg_start, left=seg_start, height=0.4,
                 color='#f72585', alpha=0.75, align='center')
    for peak, (color, _) in zip(
        sorted(peaks, key=lambda x: x['frame_idx']), peak_styles
    ):
        ax_time.axvline(peak['frame_idx'], color=color, linewidth=2)

    ax_time.set_yticks([])
    ax_time.set_xlabel('Frame', color='#aaaaaa', fontsize=9)
    ax_time.set_title(
        f'Segment:  Frame {seg_start}  →  Frame {seg_end}  '
        f'({seg_end - seg_start} frames)',
        color='#aaaaaa', fontsize=9, pad=4,
    )
    ax_time.tick_params(colors='#aaaaaa')
    for spine in ax_time.spines.values():
        spine.set_edgecolor('#333355')

    plt.tight_layout(h_pad=1.5)
    out_path = 'serve_segment_peaks.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Plot saved → {out_path}")


# ── 6. MAIN ────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python find_peaks.py <path_to_csv>")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"\n[1/4] Parsing CSV: {filepath}")
    df, tz_cols, tracks = parse_vicon_csv(filepath)
    print(f"      {len(df)} frames | {len(tracks)} tracks | {len(tz_cols)} TZ columns")

    excluded = set()

    print("\n[2/4] Identifying markers...")

    # Step 1 — Head (greatest average TZ across all frames)
    head, head_avg_tz = identify_head(df, tracks, excluded)
    if head is None:
        print("  ERROR: Could not identify head.")
        sys.exit(1)
    excluded.add(head)
    print(f"  Head        → {head}  (avg TZ = {head_avg_tz:.1f} mm)")

    # Step 2 — Ball hand (first marker to clear head TZ, no exclusions except head)
    ball_hand, first_frame = identify_ball_hand(df, tracks, head)
    if ball_hand is None:
        print("  ERROR: No marker ever exceeded the head's TZ. Cannot identify ball hand.")
        sys.exit(1)
    excluded.add(ball_hand)
    print(f"  Ball Hand   → {ball_hand}  (first exceeded head at frame {first_frame})")

    # Step 3 — Peak 1: max TZ of ball hand across all frames
    bh_tz = df[ball_hand + "_TZ"].dropna()
    peak1_fi  = int(bh_tz.idxmax())
    peak1_val = float(bh_tz[peak1_fi])
    print(f"  Peak 1 (Ball Toss)     → {ball_hand}_TZ  frame {peak1_fi}  TZ = {peak1_val:.1f} mm")

    # Step 4 — Racket hand (highest global TZ max, excluding head + ball hand)
    racket_hand, rh_max_tz = identify_racket_hand(df, tracks, excluded)
    if racket_hand is None:
        print("  ERROR: Could not identify racket hand.")
        sys.exit(1)
    excluded.add(racket_hand)
    print(f"  Racket Hand → {racket_hand}  (global TZ max = {rh_max_tz:.1f} mm)")

    # Step 5 — Peak 2: max TZ of racket hand across all frames
    rh_tz = df[racket_hand + "_TZ"].dropna()
    peak2_fi  = int(rh_tz.idxmax())
    peak2_val = float(rh_tz[peak2_fi])
    print(f"  Peak 2 (Follow Through) → {racket_hand}_TZ  frame {peak2_fi}  TZ = {peak2_val:.1f} mm")

    # Step 6 — Elbow1: tallest remaining marker at exact Peak 2 frame
    elbow1, elbow1_val = identify_elbow1(df, tracks, excluded, peak2_fi)
    if elbow1 is None:
        print("  ERROR: Could not identify elbow1.")
        sys.exit(1)
    excluded.add(elbow1)
    print(f"  Elbow1      → {elbow1}  (TZ at Peak 2 frame = {elbow1_val:.1f} mm)")

    peaks = [
        {'col': ball_hand   + "_TZ", 'frame_idx': peak1_fi, 'value': peak1_val, 'label': 'Ball Toss'},
        {'col': racket_hand + "_TZ", 'frame_idx': peak2_fi, 'value': peak2_val, 'label': 'Follow Through'},
    ]
    peaks.sort(key=lambda x: x['frame_idx'])

    labels_map = {
        'head':        head,
        'ball_hand':   ball_hand,
        'racket_hand': racket_hand,
        'elbow1':      elbow1,
    }

    # Step 7 — Swing window
    swing_start = min(p['frame_idx'] for p in peaks)
    swing_end   = max(p['frame_idx'] for p in peaks)
    print(f"  Swing window: frame {swing_start} → {swing_end} ({swing_end - swing_start} frames)")

    # Step 8 — Body parts
    print("\n[3/4] Assigning body parts...")
    body_parts, elbow2, elbow2_val, ranked = assign_body_parts(df, tracks, excluded, swing_start, swing_end)

    if elbow2:
        print(f"  Elbow2    → {elbow2}  (TZ at swing start = {elbow2_val:.1f} mm)")
    if 'shoulder1' in body_parts:
        s1_tz = df.iloc[swing_start][body_parts['shoulder1'] + '_TZ']
        print(f"  Shoulder1 → {body_parts['shoulder1']}  (TZ at swing start = {s1_tz:.1f} mm)")
    print(f"  Remaining {len(ranked)} markers ranked by avg TZ within swing:")
    for track, avg_tz in ranked:
        label = next((k for k, v in body_parts.items() if v == track), '(unlabeled)')
        print(f"    {label:<12} → {track}  (avg TZ = {avg_tz:.1f} mm)")

    print("\n[4/5] Orienting left/right labels...")
    labels_map, body_parts = orient_labels(df, labels_map, body_parts, peaks[0]['frame_idx'], peaks[1]['frame_idx'])
    print("  Labels after orientation:")
    for label, track in {**labels_map, **body_parts}.items():
        if label not in ('head', 'chest'):
            print(f"    {label:<20} → {track}")

    print("\n[5/5] Generating plot + formatted CSV...")
    plot_segment(df, tz_cols, peaks, labels_map, body_parts)
    export_annotated_csv(filepath, peaks, labels_map, body_parts)
    print("\nDone. Run find_snapshots.py on the output to fill in snapshot frames.\n")


if __name__ == '__main__':
    main()
