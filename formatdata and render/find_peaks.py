"""
find_peaks.py
-------------
Identifies key markers from an unmarked Vicon CSV using biomechanical logic,
then finds two serve peaks and exports an annotated CSV for plot_serve.py S mode.

Marker identification order:
  1. Head         — greatest average TZ across all frames
  2. Ball hand    — first marker (frame by frame) whose TZ exceeds the head's
                    TZ at that same frame (no exclusions)
  3. Peak 1 (Ball Toss)      — frame where ball hand TZ is maximum
  4. Racket hand  — marker (excluding head + ball hand) with highest global TZ maximum
  5. Peak 2 (Follow Through) — frame where racket hand TZ is maximum
  6. Elbow1       — tallest remaining marker (excluding head + ball hand + racket hand)
                    at the exact frame of Peak 2
  7. Swing window — Peak 1 frame → Peak 2 frame
  8. Everything else assigned by TZ ranking within swing window

Output:
  <input>_peaks.csv   — original CSV with PEAK1/PEAK2 metadata prepended
  serve_segment_peaks.png — 2D confirmation plot

Usage:
    python find_peaks.py <path_to_csv>
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

def avg_position(df, track, swing_start, swing_end):
    """Return mean TX, TY, TZ for a track across the swing window."""
    swing_df = df.iloc[swing_start:swing_end + 1]
    tx = float(swing_df[track + "_TX"].dropna().mean())
    ty = float(swing_df[track + "_TY"].dropna().mean())
    tz = float(swing_df[track + "_TZ"].dropna().mean())
    return np.array([tx, ty, tz])


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
        hip1_pos = avg_position(df, assignments['hip1'], swing_start, swing_end)
        hip2_pos = avg_position(df, assignments['hip2'], swing_start, swing_end)
        hip_mid  = (hip1_pos + hip2_pos) / 2.0

        dists = {}
        for track in chest_shoulder_candidates:
            pos = avg_position(df, track, swing_start, swing_end)
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

def export_annotated_csv(source_filepath, peaks, labels_map, body_parts, out_path=None):
    """
    Prepend metadata lines to the original CSV:
        PEAK1=<frame_idx>,<col_name>,<label>
        PEAK2=<frame_idx>,<col_name>,<label>
        LABEL=<track>,<body_part_name>
        ...
    labels_map : dict of named markers  e.g. {'head': 'Track1377', ...}
    body_parts : dict of body part assignments e.g. {'elbow1': 'Track1383', ...}
    """
    sorted_peaks = sorted(peaks, key=lambda x: x['frame_idx'])

    if out_path is None:
        base = os.path.splitext(os.path.basename(source_filepath))[0]
        out_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(out_dir, base + '_peaks.csv')
        if os.path.exists(out_path):
            counter = 2
            while os.path.exists(os.path.join(out_dir, f"{base}_peaks_{counter}.csv")):
                counter += 1
            out_path = os.path.join(out_dir, f"{base}_peaks_{counter}.csv")

    # Build full label map: named markers + body parts
    all_labels = {}
    for role, track in labels_map.items():
        all_labels[track] = role          # e.g. Track1377 → head
    for part, track in body_parts.items():
        all_labels[track] = part          # e.g. Track1383 → elbow1

    with open(source_filepath, 'r') as f:
        original = f.read()

    with open(out_path, 'w') as f:
        for i, p in enumerate(sorted_peaks, 1):
            f.write(f"PEAK{i}={p['frame_idx']},{p['col']},{p['label']}\n")
        for track, label in all_labels.items():
            f.write(f"LABEL={track},{label}\n")
        f.write(original)

    print(f"  Annotated CSV saved → {out_path}")
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

    # Highlighted traces for identified markers
    highlight = {
        labels_map['head']:        ('#a78bfa', 'Head'),
        labels_map['ball_hand']:   ('#ffd166', 'Ball Hand'),
        labels_map['racket_hand']: ('#06d6a0', 'Racket Hand'),
        labels_map['elbow1']:      ('#f72585', 'Elbow1'),
    }
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

    print("\n[4/4] Generating plot + annotated CSV...")
    plot_segment(df, tz_cols, peaks, labels_map, body_parts)
    export_annotated_csv(filepath, peaks, labels_map, body_parts)
    print("\nDone.\n")


if __name__ == '__main__':
    main()
