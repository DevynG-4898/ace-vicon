"""
find_snapshots.py
-----------------
Takes a formatted CSV produced by format_data.py and finds 6 significant
snapshot frames, writing them back into the SNAPSHOT= metadata rows.
"""

import sys
import os
import numpy as np
import pandas as pd

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


def read_formatted_csv(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    peaks = {}
    snapshots = {}
    meta_end = 0

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("PEAK1="):
            parts = s[6:].split(',')
            peaks['peak1'] = {'frame_idx': int(parts[0]), 'col': parts[1], 'label': parts[2]}
        elif s.startswith("PEAK2="):
            parts = s[6:].split(',')
            peaks['peak2'] = {'frame_idx': int(parts[0]), 'col': parts[1], 'label': parts[2]}
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
    part_names = []
    tz_cols = []

    for col_idx in range(df_raw.shape[1]):
        if col_idx == 0:
            columns.append("Frame")
        elif col_idx == 1:
            columns.append("SubFrame")
        else:
            base_idx = ((col_idx - 2) // 3) * 3 + 2
            base_name = str(p_row[base_idx]).strip()
            axis = str(a_row[col_idx]).strip()
            col_name = f"{base_name}_{axis}"
            columns.append(col_name)

            if base_name not in part_names and base_name.lower() != 'nan' and base_name != '':
                part_names.append(base_name)
            if axis == 'TZ' and col_name not in tz_cols:
                tz_cols.append(col_name)

    data_rows = df_raw.iloc[3:].copy()
    data_rows.columns = columns

    for col in columns:
        data_rows[col] = pd.to_numeric(data_rows[col], errors='coerce')

    data_rows['Frame'] = data_rows['Frame'].astype(int)
    df_clean = data_rows.reset_index(drop=True)

    return df_clean, tz_cols, part_names, peaks, snapshots, lines, meta_end


def find_snapshots(df, tz_cols, part_names, peaks):
    snapshots = {name: 0 for name in SNAPSHOT_NAMES}
    peak1_frame = peaks['peak1']['frame_idx']
    peak2_frame = peaks['peak2']['frame_idx']

    PARALLEL_THRESHOLD = 5.0  # mm

    def frame_to_idx(frame_num):
        matches = df.index[df['Frame'] == frame_num].tolist()
        return matches[0] if matches else min(frame_num, len(df) - 1)

    p1_idx   = frame_to_idx(peak1_frame)
    p2_idx   = frame_to_idx(peak2_frame)
    last_idx = len(df) - 1

    rh_tz = df['right_hand_TZ'].values  if 'right_hand_TZ'  in df.columns else None
    re_tz = df['right_elbow_TZ'].values if 'right_elbow_TZ' in df.columns else None
    lh_tz = df['left_hand_TZ'].values   if 'left_hand_TZ'   in df.columns else None

    if rh_tz is None or re_tz is None:
        print("  WARNING: right_hand_TZ or right_elbow_TZ not found.")
        return snapshots

    # ── 1. start_pose ──────────────────────────────────────────────────────────
    # Frame where left_hand_TZ is at its maximum (peak of ball toss)
    start_idx = 0
    if lh_tz is not None:
        max_val = -np.inf
        for i in range(len(df)):
            if not np.isnan(lh_tz[i]) and lh_tz[i] > max_val:
                max_val = lh_tz[i]
                start_idx = i
    snapshots['start_pose'] = int(df['Frame'].iloc[start_idx])
    print(f"  -> start_pose:          frame {snapshots['start_pose']}  (df idx {start_idx})")

    # ── 2. hand_cross ──────────────────────────────────────────────────────────
    # Frame where left_hand_TZ and right_hand_TZ intercept (cross) — the toss
    # hand descending as the racket hand rises. Must happen before peak2 (contact).
    intercept_idx = None
    if lh_tz is not None:
        for i in range(start_idx + 1, p2_idx + 1):
            if np.isnan(lh_tz[i]) or np.isnan(rh_tz[i]):
                continue
            if np.isnan(lh_tz[i - 1]) or np.isnan(rh_tz[i - 1]):
                continue
            diff_prev = lh_tz[i - 1] - rh_tz[i - 1]
            diff_curr = lh_tz[i]     - rh_tz[i]
            if diff_prev * diff_curr <= 0:
                intercept_idx = i
                break

    if intercept_idx is None:
        snapshots['hand_cross'] = 0
        print(f"  -> hand_cross:          NOT FOUND before contact — skipping hand_cross, flat_racket_arm, and peak_racket_arm")
    else:
        snapshots['hand_cross'] = int(df['Frame'].iloc[intercept_idx])
        print(f"  -> hand_cross:          frame {snapshots['hand_cross']}  (df idx {intercept_idx})")

    # ── 3. flat_racket_arm ─────────────────────────────────────────────────────
    # From hand_cross onward, find |right_hand_TZ - right_elbow_TZ| <= 5mm
    flat_idx = start_idx  # fallback if lh_tz unavailable

    if lh_tz is not None:
        if intercept_idx is None:
            snapshots['flat_racket_arm'] = 0
            snapshots['peak_racket_arm'] = 0
            print(f"  -> flat_racket_arm:     SKIPPED (no hand_cross before contact)")
            print(f"  -> peak_racket_arm:     SKIPPED")
        else:
            # From intercept, find |right_hand_TZ - right_elbow_TZ| <= 1mm
            # Must happen before peak2
            flat_idx = None
            for i in range(intercept_idx + 1, p2_idx + 1):
                if np.isnan(rh_tz[i]) or np.isnan(re_tz[i]):
                    continue
                if abs(rh_tz[i] - re_tz[i]) <= PARALLEL_THRESHOLD:
                    flat_idx = i
                    break

            if flat_idx is None:
                print(f"  -> flat_racket_arm:     RACKET HAND NEVER WENT FLAT before contact (threshold {PARALLEL_THRESHOLD}mm not met)")
                snapshots['flat_racket_arm'] = 0
                snapshots['peak_racket_arm'] = 0
                print(f"  -> peak_racket_arm:     SKIPPED")
            else:
                rh_val = rh_tz[flat_idx]
                re_val = re_tz[flat_idx]
                snapshots['flat_racket_arm'] = int(df['Frame'].iloc[flat_idx])
                print(f"  -> flat_racket_arm:     frame {snapshots['flat_racket_arm']}  (df idx {flat_idx})")
                print(f"     CHECK: right_hand_TZ = {rh_val:.2f} mm  |  right_elbow_TZ = {re_val:.2f} mm  |  diff = {abs(rh_val - re_val):.2f} mm")

                # ── 4. peak_racket_arm ─────────────────────────────────────────
                # Scan forward from flat_idx: right hand descending then turns back up
                # Must happen before peak2
                peak_racket_idx = None
                descending = False
                for i in range(flat_idx + 1, p2_idx + 1):
                    if np.isnan(rh_tz[i]) or np.isnan(rh_tz[i - 1]):
                        continue
                    if rh_tz[i] < rh_tz[i - 1]:
                        descending = True
                    elif descending and rh_tz[i] > rh_tz[i - 1]:
                        peak_racket_idx = i - 1
                        break

                if peak_racket_idx is None:
                    print(f"  -> peak_racket_arm:     NOT FOUND before contact — skipping")
                    snapshots['peak_racket_arm'] = 0
                else:
                    snapshots['peak_racket_arm'] = int(df['Frame'].iloc[peak_racket_idx])
                    print(f"  -> peak_racket_arm:     frame {snapshots['peak_racket_arm']}  (df idx {peak_racket_idx})")

    # Update flat_idx for downstream use (contact search starts after peak_racket_arm if found)
    peak_racket_idx_final = None
    if snapshots['peak_racket_arm'] != 0:
        matches = df.index[df['Frame'] == snapshots['peak_racket_arm']].tolist()
        peak_racket_idx_final = matches[0] if matches else None

    search_from = peak_racket_idx_final if peak_racket_idx_final is not None else p2_idx

    # ── 5. contact ─────────────────────────────────────────────────────────────
    # Scan FORWARD from search_from: frame where right_hand_TZ is maximum
    contact_idx = search_from
    max_val = -np.inf
    for i in range(search_from + 1, last_idx + 1):
        if np.isnan(rh_tz[i]):
            continue
        if rh_tz[i] > max_val:
            max_val = rh_tz[i]
            contact_idx = i
    snapshots['contact'] = int(df['Frame'].iloc[contact_idx])
    print(f"  -> contact:             frame {snapshots['contact']}  (df idx {contact_idx})")

    # ── 6. hand_cross_2 ────────────────────────────────────────────────────────
    # Same LH/RH intercept logic as hand_cross, but searched forward from
    # contact to the end of the file — the hands crossing again during
    # follow-through as the toss arm comes back down and the racket arm
    # continues past contact.
    intercept2_idx = None
    if lh_tz is not None:
        for i in range(contact_idx + 1, last_idx + 1):
            if np.isnan(lh_tz[i]) or np.isnan(rh_tz[i]):
                continue
            if np.isnan(lh_tz[i - 1]) or np.isnan(rh_tz[i - 1]):
                continue
            diff_prev = lh_tz[i - 1] - rh_tz[i - 1]
            diff_curr = lh_tz[i]     - rh_tz[i]
            if diff_prev * diff_curr <= 0:
                intercept2_idx = i
                break

    if intercept2_idx is None:
        snapshots['hand_cross_2'] = 0
        print(f"  -> hand_cross_2:        NOT FOUND after contact — skipping")
    else:
        snapshots['hand_cross_2'] = int(df['Frame'].iloc[intercept2_idx])
        print(f"  -> hand_cross_2:        frame {snapshots['hand_cross_2']}  (df idx {intercept2_idx})")

    # ── 7. racket_deceleration ─────────────────────────────────────────────────
    # Scan FORWARD from contact_idx+1 → last_idx:
    # first frame |right_hand_TZ - right_elbow_TZ| <= threshold
    decel_idx = contact_idx
    for i in range(contact_idx + 1, last_idx + 1):
        if np.isnan(rh_tz[i]) or np.isnan(re_tz[i]):
            continue
        if abs(rh_tz[i] - re_tz[i]) <= PARALLEL_THRESHOLD:
            decel_idx = i
            break
    snapshots['racket_deceleration'] = int(df['Frame'].iloc[decel_idx])
    print(f"  -> racket_deceleration: frame {snapshots['racket_deceleration']}  (df idx {decel_idx})")

    # ── 8. finish_pose ─────────────────────────────────────────────────────────
    # Scan FORWARD from decel_idx+1 → last_idx:
    # right hand descending, then first frame it turns back up (local min)
    finish_idx = decel_idx
    descending = False
    for i in range(decel_idx + 1, last_idx + 1):
        if np.isnan(rh_tz[i]) or np.isnan(rh_tz[i - 1]):
            continue
        if rh_tz[i] < rh_tz[i - 1]:
            descending = True
        elif descending and rh_tz[i] > rh_tz[i - 1]:
            finish_idx = i - 1
            break
    snapshots['finish_pose'] = int(df['Frame'].iloc[finish_idx])
    print(f"  -> finish_pose:         frame {snapshots['finish_pose']}  (df idx {finish_idx})")

    return snapshots


def write_snapshots_back(filepath, lines, meta_end, snapshots):
    output_lines = []
    snapshot_block_written = False

    for i in range(meta_end):
        line = lines[i]
        s = line.strip()
        if s.startswith("SNAPSHOT="):
            if not snapshot_block_written:
                # Emit the full snapshot block here, in canonical SNAPSHOT_NAMES
                # order, replacing whatever set/order of SNAPSHOT= lines the
                # file had before (handles older files missing newer names,
                # e.g. hand_cross, and keeps them in the right sequence).
                for name in SNAPSHOT_NAMES:
                    if name in snapshots:
                        output_lines.append(f"SNAPSHOT={name},{snapshots[name]}\n")
                snapshot_block_written = True
            # skip original SNAPSHOT= lines individually -- already emitted above
            continue
        output_lines.append(line)

    if not snapshot_block_written:
        # File had no SNAPSHOT= lines at all -- insert the block before the data header
        for name in SNAPSHOT_NAMES:
            if name in snapshots:
                output_lines.append(f"SNAPSHOT={name},{snapshots[name]}\n")

    output_lines.extend(lines[meta_end:])
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(output_lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_snapshots.py <path_to_formatted_csv>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    print(f"\n[1/3] Reading formatted CSV: {filepath}")
    df, tz_cols, part_names, peaks, existing_snapshots, lines, meta_end = read_formatted_csv(filepath)

    print("\n[2/3] Finding snapshot frames...")
    snapshots = find_snapshots(df, tz_cols, part_names, peaks)

    print(f"\n[3/3] Saving calculated metadata...")
    write_snapshots_back(filepath, lines, meta_end, snapshots)
    print("      Done!")


if __name__ == "__main__":
    main()