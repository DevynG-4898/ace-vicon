"""
plot_serve.py

3D animated skeleton viewer for serve CSVs.

Usage:
    U mode — verify file is unlabeled, then show raw points:
        python plot_serve.py U <csv>
        Errors if the CSV header contains body part labels.

    L mode — full serve, all points labeled from CSV header:
        python plot_serve.py L <csv>
        Reads body part names directly from the header row of a _peaks.csv.

    P mode — swing segment only, labeled, with timeline bar:
        python plot_serve.py P <csv>
        Reads PEAK1/PEAK2 metadata, slices to segment, labels all markers.

Examples:
    python plot_serve.py U data\\14unL.csv
    python plot_serve.py L data\\14unL_formatted.csv
    python plot_serve.py P data\\14unL_formatted.csv

Controls:
    - Animation plays automatically
    - Click and drag to rotate the 3D view
    - Close the window to exit
"""

import sys
import os
import io
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

MARKER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#f7b733", "#00c957", "#6a0dad", "#e6194b",
]

LABEL_COLORS = {
    'head':          '#a78bfa',
    'right_hand':    '#ffd166',
    'left_hand':     '#06d6a0',
    'right_elbow':   '#f72585',
    'left_elbow':    '#ff6b6b',
    'right_shoulder':'#b5179e',
    'left_shoulder': '#7209b7',
    'chest':         '#4361ee',
    'right_hip':     '#4cc9f0',
    'left_hip':      '#3a86ff',
    'right_knee':    '#8ecae6',
    'left_knee':     '#90e0ef',
    'right_foot':    '#caf0f8',
    'left_foot':     '#ade8f4',
}


# ── SHARED HELPERS ─────────────────────────────────────────────────────────────

def padded_limits(arrays, pad=0.08):
    flat = np.concatenate([a for a in arrays if len(a) > 0])
    lo, hi = np.nanmin(flat), np.nanmax(flat)
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


def make_axes(ax, x_lim, y_lim, z_lim, x_range, y_range, z_range):
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_box_aspect([x_range, y_range, z_range])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (height)")


def is_track_name(name):
    """Return True if the name looks like an original unlabeled Vicon track."""
    return "Reconstruction Position" in name or (
        "Track" in name and "(" in name
    )


def parse_peaks_csv(filepath):
    """
    Parse a _peaks.csv file produced by find_peaks.py.

    Format:
        PEAK1=<frame_idx>,<col_name>,<label>
        PEAK2=<frame_idx>,<col_name>,<label>
        <header row 0: body part names>
        <header row 1: TX/TY/TZ>
        <header row 2: units>
        <data rows>

    Returns:
        df       : full DataFrame with columns like ball_hand_TX, head_TZ, etc.
        tz_cols  : list of _TZ column names
        part_names: list of body part names in column order
        peaks    : list of two dicts {frame_idx, col, label, color} sorted by frame
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    meta_peaks = {}
    data_start = 0

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("PEAK1="):
            parts = s[6:].split(",", 2)
            meta_peaks['peak1'] = {
                'frame_idx': int(parts[0]),
                'col':       parts[1].strip(),
                'label':     parts[2].strip() if len(parts) > 2 else 'Ball Toss',
                'color':     '#ffd166',
            }
        elif s.startswith("PEAK2="):
            parts = s[6:].split(",", 2)
            meta_peaks['peak2'] = {
                'frame_idx': int(parts[0]),
                'col':       parts[1].strip(),
                'label':     parts[2].strip() if len(parts) > 2 else 'Follow Through',
                'color':     '#06d6a0',
            }
        elif s.startswith("SNAPSHOT="):
            continue  # skip snapshot lines
        else:
            data_start = i
            break

    if 'peak1' not in meta_peaks or 'peak2' not in meta_peaks:
        raise ValueError("Missing PEAK1/PEAK2 lines. Was this file produced by find_peaks.py?")

    raw = pd.read_csv(io.StringIO("".join(lines[data_start:])),
                      header=None, low_memory=False)

    marker_row    = raw.iloc[0].fillna("").tolist()
    component_row = raw.iloc[1].fillna("").tolist()

    # Build column labels from the header (body part names already in row 0)
    col_labels = []
    current_name = ""
    for i, (m, c) in enumerate(zip(marker_row, component_row)):
        m = str(m).strip()
        c = str(c).strip()
        if m:
            current_name = m
        if i == 0:
            col_labels.append("Frame")
        elif i == 1:
            col_labels.append("SubFrame")
        else:
            col_labels.append(f"{current_name}_{c}" if current_name else f"Col{i}_{c}")

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

    # Part names in column order (base of each TZ col)
    part_names = [col[:-3] for col in tz_cols]

    # Remap peak col names to new label-based names
    # e.g. Track1370_TZ → ball_hand_TZ
    orig_to_new = {}
    cur_orig = ""
    cur_new  = ""
    for i, (m, c) in enumerate(zip(marker_row, component_row)):
        m = str(m).strip()
        c = str(c).strip()
        if m:
            cur_orig = m
            cur_new  = col_labels[i][:-3] if i < len(col_labels) else m
        if i >= 2 and c.upper() == 'TZ':
            if "Track" in cur_orig and "(" in cur_orig:
                track_id = cur_orig.split("(")[-1].rstrip(")")
                orig_to_new[f"Track{track_id}_TZ"] = col_labels[i]

    for key in ('peak1', 'peak2'):
        old = meta_peaks[key]['col']
        meta_peaks[key]['col'] = orig_to_new.get(old, old)

    peaks = sorted([meta_peaks['peak1'], meta_peaks['peak2']],
                   key=lambda x: x['frame_idx'])

    return df, tz_cols, part_names, peaks


def parse_unlabeled_csv(filepath):
    """
    Parse a plain Vicon CSV (no PEAK lines).
    Returns df, tz_cols, part_names (track IDs), is_labeled (bool).
    """
    raw = pd.read_csv(filepath, header=None, low_memory=False)

    marker_row    = raw.iloc[0].fillna("").tolist()
    component_row = raw.iloc[1].fillna("").tolist()

    col_labels = []
    current_name = ""
    labeled = False

    for i, (m, c) in enumerate(zip(marker_row, component_row)):
        m = str(m).strip()
        c = str(c).strip()
        if m:
            current_name = m
            if i >= 2 and not is_track_name(m):
                labeled = True   # found a human-readable label
        if i == 0:
            col_labels.append("Frame")
        elif i == 1:
            col_labels.append("SubFrame")
        else:
            if "Track" in current_name and "(" in current_name:
                track_id = current_name.split("(")[-1].rstrip(")")
                col_labels.append(f"Track{track_id}_{c}")
            else:
                col_labels.append(f"{current_name}_{c}" if current_name else f"Col{i}_{c}")

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
    part_names = [col[:-3] for col in tz_cols]

    return df, tz_cols, part_names, labeled


def build_axis_limits(df, tz_cols):
    all_tx, all_ty, all_tz = [], [], []
    for col in tz_cols:
        base = col[:-3]
        tx_col, ty_col = base + "_TX", base + "_TY"
        if tx_col in df.columns:
            all_tx.append(df[tx_col].dropna().values)
        if ty_col in df.columns:
            all_ty.append(df[ty_col].dropna().values)
        all_tz.append(df[col].dropna().values)

    x_lim = padded_limits(all_tx) if all_tx else (-1000, 1000)
    y_lim = padded_limits(all_ty) if all_ty else (-1000, 1000)
    z_lim = padded_limits(all_tz) if all_tz else (0, 2000)
    return x_lim, y_lim, z_lim


def draw_timeline_bar(fig, fi, total_frames, peak1_fi, peak2_fi):
    ax = fig.axes[-1]
    ax.cla()
    ax.set_xlim(0, total_frames)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_facecolor('#1a1a2e')

    swing_width = peak2_fi - peak1_fi
    ax.barh(0.5, peak1_fi,                   left=0,        height=0.55, color='#334155', align='center')
    ax.barh(0.5, swing_width,                left=peak1_fi, height=0.55, color='#f72585', alpha=0.75, align='center')
    ax.barh(0.5, total_frames - peak2_fi,    left=peak2_fi, height=0.55, color='#334155', align='center')

    def zone_label(x, text, color='white'):
        ax.text(x, 0.5, text, color=color, fontsize=7.5,
                fontweight='bold', ha='center', va='center', clip_on=True)

    zone_label(peak1_fi / 2,                             'Before Swing')
    zone_label(peak1_fi + swing_width / 2,               'Swing',       color='#1a1a2e')
    zone_label(peak2_fi + (total_frames - peak2_fi) / 2, 'After Swing')

    ax.axvline(peak1_fi, color='#ffd166', linewidth=2, zorder=4)
    ax.axvline(peak2_fi, color='#06d6a0', linewidth=2, zorder=4)
    ax.axvline(fi,       color='white',   linewidth=1.5, linestyle='--', zorder=5)
    ax.set_xlabel(f'Frame {fi} / {total_frames - 1}', color='#aaaaaa', fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')


def draw_phase_bar(fig, fi, total_frames, peak1_fi, peak2_fi):
    """
    TEMP DEBUG WIDGET — light-background version of the timeline bar for
    L mode, labeling Pre-Serve / Serve / Post-Serve zones relative to the
    ball-toss (peak1) and follow-through (peak2) frames.
    """
    ax = fig.axes[-1]
    ax.cla()
    ax.set_xlim(0, total_frames)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_facecolor('white')

    pre_w   = peak1_fi
    serve_w = peak2_fi - peak1_fi
    post_w  = total_frames - peak2_fi

    ax.barh(0.5, pre_w,   left=0,        height=0.55, color='#cbd5e1', align='center')
    ax.barh(0.5, serve_w, left=peak1_fi, height=0.55, color='#f72585', alpha=0.85, align='center')
    ax.barh(0.5, post_w,  left=peak2_fi, height=0.55, color='#cbd5e1', align='center')

    def zone_label(x, text, color='black'):
        ax.text(x, 0.5, text, color=color, fontsize=7.5,
                fontweight='bold', ha='center', va='center', clip_on=True)

    if pre_w > 0:
        zone_label(pre_w / 2, 'Pre-Serve')
    zone_label(peak1_fi + serve_w / 2, 'Serve', color='white')
    if post_w > 0:
        zone_label(peak2_fi + post_w / 2, 'Post-Serve')

    ax.axvline(peak1_fi, color='#ffd166', linewidth=2, zorder=4)
    ax.axvline(peak2_fi, color='#06d6a0', linewidth=2, zorder=4)
    ax.axvline(fi,       color='black',   linewidth=1.5, linestyle='--', zorder=5)

    if fi < peak1_fi:
        phase = 'PRE-SERVE'
    elif fi <= peak2_fi:
        phase = 'SERVE'
    else:
        phase = 'POST-SERVE'
    ax.set_xlabel(f'Frame {fi} / {total_frames - 1}   —   {phase}', color='black', fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')


# ── U MODE ─────────────────────────────────────────────────────────────────────

def plot_unlabeled_mode(filepath):
    """
    U mode: verify the CSV is truly unlabeled, then animate raw points.
    Errors if any body part labels are found in the header.
    """
    # Check for PEAK lines first
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
    if first_line.startswith("PEAK1="):
        print("ERROR: This CSV file is labeled (contains PEAK metadata).")
        print("       Use S mode or L mode instead.")
        sys.exit(1)

    df, tz_cols, part_names, is_labeled = parse_unlabeled_csv(filepath)

    if is_labeled:
        print("ERROR: This CSV file is labeled.")
        print("       Use L mode or S mode to view labeled files.")
        sys.exit(1)

    total_frames = len(df)
    print(f"[U mode] {len(tz_cols)} markers | {total_frames} frames")

    x_lim, y_lim, z_lim = build_axis_limits(df, tz_cols)
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    z_range = z_lim[1] - z_lim[0]

    fig = plt.figure(figsize=(10, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#aaaaaa')

    def update(fi):
        ax.cla()
        ax.set_facecolor('#1a1a2e')
        make_axes(ax, x_lim, y_lim, z_lim, x_range, y_range, z_range)
        ax.set_title(f"Unlabeled — Frame {fi + 1} / {total_frames}",
                     color='white', fontsize=11)
        for i, tz_col in enumerate(tz_cols):
            base = tz_col[:-3]
            tx_col, ty_col = base + "_TX", base + "_TY"
            if tx_col not in df.columns or ty_col not in df.columns:
                continue
            x, y, z = df.loc[fi, tx_col], df.loc[fi, ty_col], df.loc[fi, tz_col]
            if any(pd.isna(v) for v in [x, y, z]):
                continue
            color = MARKER_COLORS[i % len(MARKER_COLORS)]
            ax.scatter(x, y, z, s=50, color=color, zorder=4)

    make_axes(ax, x_lim, y_lim, z_lim, x_range, y_range, z_range)
    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=33, repeat=True)
    plt.tight_layout()
    plt.show()


# ── L MODE ─────────────────────────────────────────────────────────────────────

def plot_labeled_mode(filepath):
    """
    L mode: show the full serve with all markers labeled from the CSV header.
    Black and white markers/labels, yellow bones.
    """
    df, tz_cols, part_names, peaks = parse_peaks_csv(filepath)
    total_frames = len(df)

    print(f"[L mode] {len(tz_cols)} labeled markers | {total_frames} frames")
    for name in part_names:
        print(f"  {name}")

    x_lim, y_lim, z_lim = build_axis_limits(df, tz_cols)
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    z_range = z_lim[1] - z_lim[0]

    # Bone connections using body part label names
    BONES = [
        ('head',           'chest'),
        ('chest',          'right_shoulder'),
        ('chest',          'left_shoulder'),
        ('right_shoulder', 'right_elbow'),
        ('left_shoulder',  'left_elbow'),
        ('right_elbow',    'right_hand'),
        ('left_elbow',     'left_hand'),
        ('chest',          'right_hip'),
        ('chest',          'left_hip'),
        ('right_hip',      'left_hip'),
        ('right_hip',      'right_knee'),
        ('left_hip',       'left_knee'),
        ('right_knee',     'right_foot'),
        ('left_knee',      'left_foot'),
    ]

    def get_xyz(part, fi):
        tz_col = part + "_TZ"
        tx_col = part + "_TX"
        ty_col = part + "_TY"
        if tx_col not in df.columns or ty_col not in df.columns or tz_col not in df.columns:
            return None
        x, y, z = df.loc[fi, tx_col], df.loc[fi, ty_col], df.loc[fi, tz_col]
        if any(pd.isna(v) for v in [x, y, z]):
            return None
        return x, y, z

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0.0, 0.13, 1.0, 0.85], projection='3d')
    ax.set_facecolor('white')

    ax_bar = fig.add_axes([0.08, 0.02, 0.86, 0.07])
    ax_bar.set_facecolor('white')

    peak1_fi = peaks[0]['frame_idx']
    peak2_fi = peaks[1]['frame_idx']

    def update(fi):
        ax.cla()
        ax.set_facecolor('white')
        make_axes(ax, x_lim, y_lim, z_lim, x_range, y_range, z_range)
        ax.set_title(f"Full Serve — Frame {fi + 1} / {total_frames}",
                     color='black', fontsize=11)
        ax.tick_params(colors='black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.zaxis.label.set_color('black')

        # Draw bones first (behind markers)
        for start, end in BONES:
            p0 = get_xyz(start, fi)
            p1 = get_xyz(end, fi)
            if p0 is None or p1 is None:
                continue
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    color='#FFD700', linewidth=2.0, alpha=0.85, zorder=3)

        # Draw markers and labels on top
        for i, tz_col in enumerate(tz_cols):
            base = tz_col[:-3]
            result = get_xyz(base, fi)
            if result is None:
                continue
            x, y, z = result
            ax.scatter(x, y, z, s=60, color='black', zorder=5)
            ax.text(x, y, z + z_range * 0.015,
                    base, color='black', fontsize=7, ha='center')

        draw_phase_bar(fig, fi, total_frames, peak1_fi, peak2_fi)

    make_axes(ax, x_lim, y_lim, z_lim, x_range, y_range, z_range)
    draw_phase_bar(fig, 0, total_frames, peak1_fi, peak2_fi)
    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=33, repeat=True)
    plt.show()


# ── S MODE ─────────────────────────────────────────────────────────────────────

def plot_segment_mode(filepath):
    """
    S mode: show only the swing segment (between the two peaks), labeled,
    with a timeline bar at the bottom.
    """
    print(f"[S mode] Reading: {filepath}")
    df, tz_cols, part_names, peaks = parse_peaks_csv(filepath)

    peak1, peak2 = peaks  # already sorted by frame_idx
    peak1_fi = peak1['frame_idx']
    peak2_fi = peak2['frame_idx']

    # Slice to segment only
    df = df.iloc[peak1_fi:peak2_fi + 1].reset_index(drop=True)
    total_frames = len(df)
    seg_peak1_fi = 0
    seg_peak2_fi = total_frames - 1

    print(f"  Peak 1 ({peak1['label']}): frame {peak1_fi}  →  segment frame 0")
    print(f"  Peak 2 ({peak2['label']}): frame {peak2_fi}  →  segment frame {seg_peak2_fi}")
    print(f"  Segment frames: {total_frames}")

    peak_lookup = {
        peak1['col']: {**peak1, 'frame_idx': seg_peak1_fi},
        peak2['col']: {**peak2, 'frame_idx': seg_peak2_fi},
    }

    x_lim, y_lim, z_lim = build_axis_limits(df, tz_cols)
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    z_range = z_lim[1] - z_lim[0]

    fig = plt.figure(figsize=(11, 8))
    fig.patch.set_facecolor('#1a1a2e')

    ax3d = fig.add_axes([0.0, 0.18, 1.0, 0.80], projection='3d')
    ax3d.set_facecolor('#1a1a2e')
    ax3d.tick_params(colors='#aaaaaa')

    ax_bar = fig.add_axes([0.07, 0.04, 0.88, 0.10])
    ax_bar.set_facecolor('#1a1a2e')

    def update(fi):
        ax3d.cla()
        ax3d.set_facecolor('#1a1a2e')
        make_axes(ax3d, x_lim, y_lim, z_lim, x_range, y_range, z_range)

        on_peak = next((p for p in peak_lookup.values()
                        if fi == p['frame_idx']), None)
        title_color  = on_peak['color'] if on_peak else 'white'
        title_suffix = f"  ★ {on_peak['label']}" if on_peak else ''
        ax3d.set_title(f"Swing Segment — Frame {fi + 1} / {total_frames}{title_suffix}",
                       color=title_color, fontsize=11, pad=6)

        for i, tz_col in enumerate(tz_cols):
            base = tz_col[:-3]
            tx_col, ty_col = base + "_TX", base + "_TY"
            if tx_col not in df.columns or ty_col not in df.columns:
                continue
            x, y, z = df.loc[fi, tx_col], df.loc[fi, ty_col], df.loc[fi, tz_col]
            if any(pd.isna(v) for v in [x, y, z]):
                continue

            is_peak = (tz_col in peak_lookup and
                       fi == peak_lookup[tz_col]['frame_idx'])

            if is_peak:
                p_info = peak_lookup[tz_col]
                color  = p_info['color']
                ax3d.scatter(x, y, z, s=200, color=color,
                             edgecolors='white', linewidths=1.2, zorder=6)
                ax3d.text(x, y, z + z_range * 0.03,
                          f"★ {p_info['label']}",
                          color=color, fontsize=9, fontweight='bold', ha='center')
            else:
                color = LABEL_COLORS.get(base, MARKER_COLORS[i % len(MARKER_COLORS)])
                ax3d.scatter(x, y, z, s=60, color=color,
                             edgecolors='white', linewidths=0.4, zorder=5)
                ax3d.text(x, y, z + z_range * 0.015,
                          base, color=color, fontsize=7, ha='center')

        draw_timeline_bar(fig, fi, total_frames, seg_peak1_fi, seg_peak2_fi)

    make_axes(ax3d, x_lim, y_lim, z_lim, x_range, y_range, z_range)
    draw_timeline_bar(fig, 0, total_frames, seg_peak1_fi, seg_peak2_fi)

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=33, repeat=True)
    plt.show()


# ── S MODE ─────────────────────────────────────────────────────────────────────

def plot_snapshot_mode(filepath):
    """
    S mode: step through the 6 snapshot frames one at a time.
    Shows a labeled 3D view with prev/next buttons.
    Skips any snapshot with frame = 0 (not yet computed).
    """
    print(f"[S mode] Reading: {filepath}")

    # Parse snapshots from metadata
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    snapshots = {}   # name → frame number
    meta_end = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("SNAPSHOT="):
            parts = s[9:].split(',', 1)
            if len(parts) == 2:
                name  = parts[0].strip()
                frame = int(parts[1].strip())
                if frame != 0:
                    snapshots[name] = frame
        elif not s.startswith("PEAK"):
            meta_end = i
            break

    if not snapshots:
        print("ERROR: No snapshot frames found (all are 0). Run find_snapshots.py first.")
        sys.exit(1)

    print(f"  Found {len(snapshots)} snapshots:")
    for name, frame in snapshots.items():
        print(f"    {name:<22} → frame {frame}")

    # Parse the Vicon data
    df, tz_cols, part_names, peaks = parse_peaks_csv(filepath)
    total_frames = len(df)

    # Map snapshot frame numbers → df row indices
    snapshot_list = []  # list of (name, df_row_idx)
    for name, frame_num in snapshots.items():
        matches = df.index[df['Frame'] == frame_num].tolist()
        if matches:
            snapshot_list.append((name, matches[0]))
        else:
            # Fallback: use closest frame index
            snapshot_list.append((name, min(frame_num, total_frames - 1)))

    x_lim, y_lim, z_lim = build_axis_limits(df, tz_cols)
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    z_range = z_lim[1] - z_lim[0]

    # State
    state = {'idx': 0}
    n_snapshots = len(snapshot_list)

    fig = plt.figure(figsize=(11, 8))
    fig.patch.set_facecolor('white')

    ax3d = fig.add_axes([0.0, 0.12, 1.0, 0.86], projection='3d')
    ax3d.set_facecolor('white')
    ax3d.tick_params(colors='black')

    # Prev / Next buttons
    ax_prev = fig.add_axes([0.25, 0.02, 0.18, 0.06])
    ax_next = fig.add_axes([0.57, 0.02, 0.18, 0.06])
    ax_info = fig.add_axes([0.0,  0.02, 1.0,  0.06])
    ax_info.axis('off')

    btn_prev = plt.Button(ax_prev, '◀  Prev', color='#eeeeee', hovercolor='#cccccc')
    btn_next = plt.Button(ax_next, 'Next  ▶', color='#eeeeee', hovercolor='#cccccc')

    BONES = [
        ('head',           'chest'),
        ('chest',          'right_shoulder'),
        ('chest',          'left_shoulder'),
        ('right_shoulder', 'right_elbow'),
        ('left_shoulder',  'left_elbow'),
        ('right_elbow',    'right_hand'),
        ('left_elbow',     'left_hand'),
        ('chest',          'right_hip'),
        ('chest',          'left_hip'),
        ('right_hip',      'left_hip'),
        ('right_hip',      'right_knee'),
        ('left_hip',       'left_knee'),
        ('right_knee',     'right_foot'),
        ('left_knee',      'left_foot'),
    ]

    def get_xyz(part, fi):
        tx_col = part + "_TX"
        ty_col = part + "_TY"
        tz_col = part + "_TZ"
        if tx_col not in df.columns or ty_col not in df.columns or tz_col not in df.columns:
            return None
        x, y, z = df.loc[fi, tx_col], df.loc[fi, ty_col], df.loc[fi, tz_col]
        if any(pd.isna(v) for v in [x, y, z]):
            return None
        return float(x), float(y), float(z)

    def draw_snapshot():
        ax3d.cla()
        ax3d.set_facecolor('white')
        make_axes(ax3d, x_lim, y_lim, z_lim, x_range, y_range, z_range)
        ax3d.tick_params(colors='black')
        ax3d.xaxis.label.set_color('black')
        ax3d.yaxis.label.set_color('black')
        ax3d.zaxis.label.set_color('black')

        name, fi = snapshot_list[state['idx']]
        frame_num = int(df.loc[fi, 'Frame']) if 'Frame' in df.columns else fi

        # Title: snapshot name + position counter
        display_name = name.replace('_', ' ').title()
        ax3d.set_title(
            f"{display_name}   [{state['idx'] + 1} / {n_snapshots}]   frame {frame_num}",
            color='black', fontsize=13, pad=10, fontweight='bold'
        )

        # Draw bones
        for start_part, end_part in BONES:
            p0 = get_xyz(start_part, fi)
            p1 = get_xyz(end_part, fi)
            if p0 and p1:
                ax3d.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                          color='#FFD700', linewidth=2.0, alpha=0.85, zorder=3)

        # Draw markers + labels
        for i, tz_col in enumerate(tz_cols):
            base = tz_col[:-3]
            result = get_xyz(base, fi)
            if result is None:
                continue
            x, y, z = result
            color = LABEL_COLORS.get(base, '#333333')
            ax3d.scatter(x, y, z, s=70, color=color,
                         edgecolors='black', linewidths=0.5, zorder=5)
            ax3d.text(x, y, z + z_range * 0.015,
                      base, color=color, fontsize=7, ha='center')

        # Bottom info bar — current snapshot only
        ax_info.cla()
        ax_info.axis('off')
        display_name = name.replace('_', ' ').title()
        ax_info.text(0.5, 0.5, f"{state['idx'] + 1} / {n_snapshots}  —  {display_name}",
                     ha='center', va='center', fontsize=11,
                     fontweight='bold', color='black',
                     transform=ax_info.transAxes)

        fig.canvas.draw_idle()

    def on_prev(event):
        state['idx'] = (state['idx'] - 1) % n_snapshots
        draw_snapshot()

    def on_next(event):
        state['idx'] = (state['idx'] + 1) % n_snapshots
        draw_snapshot()

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    draw_snapshot()
    plt.show()


# ── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python plot_serve.py U <csv>   — raw points (errors if labeled)")
        print("  python plot_serve.py L <csv>   — full serve, labeled from header")
        print("  python plot_serve.py P <csv>   — swing segment only, labeled")
        print("  python plot_serve.py S <csv>   — step through 6 snapshot frames")
        sys.exit(1)

    mode     = sys.argv[1].upper()
    csv_path = sys.argv[2]

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    if mode == "U":
        plot_unlabeled_mode(csv_path)
    elif mode == "L":
        plot_labeled_mode(csv_path)
    elif mode == "P":
        plot_segment_mode(csv_path)
    elif mode == "S":
        plot_snapshot_mode(csv_path)
    else:
        print(f"Unknown mode '{mode}' — use U, L, P, or S.")
        sys.exit(1)
