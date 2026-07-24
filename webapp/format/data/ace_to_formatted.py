"""Convert ACE marker dictionaries into snapshot-ready `_formatted.csv` files."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.markers.io import ACE_MARKER_NAMES, load_serve_markers

SNAPSHOT_NAMES = [
    "start_pose",
    "hand_cross",
    "flat_racket_arm",
    "peak_racket_arm",
    "contact",
    "hand_cross_2",
    "racket_deceleration",
    "finish_pose",
]

OUTPUT_ORDER = list(ACE_MARKER_NAMES)


def _load_formatdata_module(module_filename: str):
    """Load a script from ``formatdata and render/`` (path contains spaces)."""
    path = Path(__file__).resolve().parents[2] / "formatdata and render" / module_filename
    if not path.is_file():
        raise FileNotFoundError(f"Formatdata module not found: {path}")
    name = f"formatdata_{path.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def markers_to_dataframe(markers: dict[str, Any]) -> pd.DataFrame:
    """Flatten a canonical ACE marker dict into Frame/SubFrame/TX/TY/TZ columns."""
    frames = np.asarray(markers["frames"], dtype=int)
    out = pd.DataFrame({"Frame": frames, "SubFrame": np.zeros(len(frames), dtype=int)})
    for name in OUTPUT_ORDER:
        marker = markers[name]
        out[f"{name}_TX"] = np.asarray(marker["TX"], dtype=float)
        out[f"{name}_TY"] = np.asarray(marker["TY"], dtype=float)
        out[f"{name}_TZ"] = np.asarray(marker["TZ"], dtype=float)
    return out


def detect_toss_and_racket_hand(df: pd.DataFrame) -> tuple[str, str, int]:
    """Whichever hand first exceeds head TZ is the toss hand."""
    head_tz = df["head_TZ"].to_numpy(dtype=float)
    lh_tz = df["left_hand_TZ"].to_numpy(dtype=float)
    rh_tz = df["right_hand_TZ"].to_numpy(dtype=float)

    for i in range(len(df)):
        hz = head_tz[i]
        if np.isnan(hz):
            continue
        if not np.isnan(lh_tz[i]) and lh_tz[i] > hz:
            return "left_hand", "right_hand", i
        if not np.isnan(rh_tz[i]) and rh_tz[i] > hz:
            return "right_hand", "left_hand", i

    # Fallback: taller peak hand is racket, other is toss.
    lh_peak = float(np.nanmax(lh_tz)) if np.isfinite(lh_tz).any() else -np.inf
    rh_peak = float(np.nanmax(rh_tz)) if np.isfinite(rh_tz).any() else -np.inf
    if rh_peak >= lh_peak:
        return "left_hand", "right_hand", 0
    return "right_hand", "left_hand", 0


def compute_peaks(df: pd.DataFrame, toss_label: str, racket_label: str) -> tuple[dict, dict]:
    toss_tz = df[f"{toss_label}_TZ"].dropna()
    racket_tz = df[f"{racket_label}_TZ"].dropna()
    if toss_tz.empty or racket_tz.empty:
        raise ValueError("Cannot compute peaks: toss or racket hand TZ is empty")

    peak1_idx = int(toss_tz.idxmax())
    peak2_idx = int(racket_tz.idxmax())
    return (
        {
            "frame_idx": peak1_idx,
            "col": f"{toss_label}_TZ",
            "value": float(toss_tz[peak1_idx]),
            "label": "Ball Toss",
        },
        {
            "frame_idx": peak2_idx,
            "col": f"{racket_label}_TZ",
            "value": float(racket_tz[peak2_idx]),
            "label": "Follow Through",
        },
    )


def export_formatted_csv(
    df: pd.DataFrame,
    peak1: dict,
    peak2: dict,
    out_path: str | Path,
) -> Path:
    """Write PEAK/SNAPSHOT metadata + marker table matching format_data_mediapipe."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        f.write(
            f"PEAK1={int(df['Frame'].iloc[peak1['frame_idx']])},"
            f"{peak1['col']},{peak1['label']}\n"
        )
        f.write(
            f"PEAK2={int(df['Frame'].iloc[peak2['frame_idx']])},"
            f"{peak2['col']},{peak2['label']}\n"
        )
        for name in SNAPSHOT_NAMES:
            f.write(f"SNAPSHOT={name},0\n")

        row0 = ["", ""]
        for label in OUTPUT_ORDER:
            row0 += [label, "", ""]
        writer.writerow(row0)

        row1 = ["", ""]
        for _ in OUTPUT_ORDER:
            row1 += ["TX", "TY", "TZ"]
        writer.writerow(row1)

        row2 = ["", ""]
        for _ in OUTPUT_ORDER:
            row2 += ["mm", "mm", "mm"]
        writer.writerow(row2)

        for _, row in df.iterrows():
            out_row = [int(row["Frame"]), int(row["SubFrame"])]
            for label in OUTPUT_ORDER:
                out_row += [
                    f"{row[f'{label}_TX']:.6g}",
                    f"{row[f'{label}_TY']:.6g}",
                    f"{row[f'{label}_TZ']:.6g}",
                ]
            writer.writerow(out_row)

    return path


def fill_snapshots(formatted_csv: str | Path) -> Path:
    """Run find_snapshots.py logic in-place on a formatted CSV."""
    path = Path(formatted_csv)
    find_snapshots = _load_formatdata_module("find_snapshots.py")
    df, tz_cols, part_names, peaks, _existing, lines, meta_end = find_snapshots.read_formatted_csv(
        str(path)
    )
    snapshots = find_snapshots.find_snapshots(df, tz_cols, part_names, peaks)
    find_snapshots.write_snapshots_back(str(path), lines, meta_end, snapshots)
    return path


def ace_markers_to_formatted_csv(
    markers: dict[str, Any] | str | Path,
    out_path: str | Path,
    *,
    fill: bool = True,
) -> Path:
    """ACE markers (dict / npz / dir) → snapshot-ready `_formatted.csv`."""
    marker_dict = load_serve_markers(markers)
    df = markers_to_dataframe(marker_dict)
    toss_label, racket_label, _ = detect_toss_and_racket_hand(df)
    peak1, peak2 = compute_peaks(df, toss_label, racket_label)
    path = export_formatted_csv(df, peak1, peak2, out_path)
    if fill:
        fill_snapshots(path)
    return path
