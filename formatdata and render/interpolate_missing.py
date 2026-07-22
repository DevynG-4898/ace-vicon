"""
interpolate_missing.py
-----------------------
Takes a formatted CSV (the output of format_data.py, before or after
find_snapshots.py has been run on it) and linearly interpolates missing
(NaN) marker values across frames -- one marker/axis column at a time.

This is the same technique as interpolate_nans() in dtw/prepare_data.py
(np.interp per column), adapted to work directly on the enhancing/
formatted-CSV structure instead of the dtw pipeline's serve_data dict.

Behavior:
  - Metadata rows (PEAK1=, PEAK2=, SNAPSHOT=) are left untouched.
  - Header rows (marker names, TX/TY/TZ labels, units) are left untouched.
  - For each marker/axis data column, any NaN frame that has valid data
    on BOTH sides gets linearly interpolated.
  - A column that is entirely NaN (marker never tracked at all) is left
    alone -- there's nothing to interpolate from.
  - Leading/trailing NaN runs (marker drops out before the first valid
    frame, or after the last valid frame, and never comes back) are left
    as NaN. There's no reference point on one side, so rather than
    flat-filling with the nearest valid value (np.interp's default
    out-of-range behavior), these are deliberately left missing.

Usage:
    python interpolate_missing.py <path_to_formatted_csv>
"""

import sys
import os
import numpy as np
import pandas as pd


def read_formatted_csv(filepath):
    """Parse metadata + header + data, same convention as find_snapshots.py."""
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    meta_end = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("PEAK1=") or s.startswith("PEAK2=") or s.startswith("SNAPSHOT="):
            continue
        meta_end = i
        break

    df_raw = pd.read_csv(filepath, header=None, skiprows=meta_end, dtype=str)
    p_row = df_raw.iloc[0].values  # marker/body-part names
    a_row = df_raw.iloc[1].values  # TX / TY / TZ axis labels

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
    df_clean = data_rows.reset_index(drop=True)

    return df_clean, columns, lines, meta_end


def interpolate_missing(df, columns):
    """Linearly interpolate NaNs in every marker/axis column, but only
    where a gap has valid data on BOTH sides. Leading/trailing NaN runs
    (no reference on one side) are left as NaN rather than flat-filled.

    Columns that are entirely NaN are left unchanged. Returns the filled
    DataFrame plus a per-column count of values filled in (and a count of
    edge-NaNs left alone), for reporting.
    """
    x = df['Frame'].values.astype(float)
    filled = df.copy()
    fill_counts = {}
    edge_counts = {}

    for col in columns:
        if col in ('Frame', 'SubFrame'):
            continue
        y = df[col].values.astype(float)
        nans = np.isnan(y)
        n_missing = int(nans.sum())
        if n_missing == 0:
            continue
        if nans.all():
            print(f"  SKIPPED {col}: entirely missing ({n_missing} frames) -- can't interpolate")
            continue

        valid_idx = np.where(~nans)[0]
        first_valid, last_valid = valid_idx[0], valid_idx[-1]

        # Only interpolate NaNs strictly between the first and last valid frame
        interior = nans & (np.arange(len(y)) > first_valid) & (np.arange(len(y)) < last_valid)
        n_interior = int(interior.sum())
        n_edge = n_missing - n_interior

        if n_interior > 0:
            y = y.copy()
            y[interior] = np.interp(x[interior], x[~nans], y[~nans])
            filled[col] = y
            fill_counts[col] = n_interior

        if n_edge > 0:
            edge_counts[col] = n_edge

        msg = f"  {col:<22} {n_interior} interior missing frame(s) filled"
        if n_edge > 0:
            msg += f", {n_edge} leading/trailing frame(s) left as NaN"
        print(msg)

    return filled, fill_counts, edge_counts


def write_back(filepath, lines, meta_end, df, columns):
    """Rewrite the file: metadata + header rows untouched verbatim, data
    rows regenerated from the (now-filled) DataFrame. Numeric formatting
    matches the original precision (~6 significant figures) but isn't
    guaranteed byte-identical for cells that weren't missing.
    """
    header_lines = lines[:meta_end + 3]  # metadata + marker/axis/units rows
    out_lines = list(header_lines)

    for _, row in df.iterrows():
        cells = [str(int(row['Frame'])), str(int(row['SubFrame']))]
        for col in columns[2:]:
            val = row[col]
            cells.append('"nan"' if pd.isna(val) else f'"{val:.6g}"')
        out_lines.append(','.join(cells) + '\n')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python interpolate_missing.py <path_to_formatted_csv>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    print(f"\n[1/3] Reading formatted CSV: {filepath}")
    df, columns, lines, meta_end = read_formatted_csv(filepath)
    print(f"      {len(df)} frames | {len(columns) - 2} marker/axis columns")

    print("\n[2/3] Interpolating missing values...")
    df_filled, fill_counts, edge_counts = interpolate_missing(df, columns)

    if not fill_counts and not edge_counts:
        print("  No missing values found -- nothing to do.")
    else:
        total = sum(fill_counts.values())
        print(f"\n  Filled {total} interior missing value(s) across {len(fill_counts)} column(s).")
        if edge_counts:
            total_edge = sum(edge_counts.values())
            print(f"  Left {total_edge} leading/trailing value(s) as NaN across {len(edge_counts)} column(s) "
                  f"(no reference on one side):")
            for col, n in edge_counts.items():
                print(f"    {col:<22} {n} frame(s)")

    print(f"\n[3/3] Saving...")
    write_back(filepath, lines, meta_end, df_filled, columns)
    print("      Done!")


if __name__ == "__main__":
    main()
