import glob
import os

import numpy as np
import pandas as pd

from constants import MARKER_ORDER
from load_data import FILENAME_TO_MARKER
from load_data import load_multi_serve
from load_data import load_single_serve

_AXES = ("TX", "TY", "TZ")


def _markers(serve_data):
    return [k for k in serve_data if k != "frames"]


def is_valid_serve(serve_data, threshold=0.2):
    """Return True if the serve's overall NaN fraction is at or below threshold.

    Args:
        serve_data: dict from load_data
        threshold: max allowed overall NaN fraction (default 0.2)
    """
    total = nan_count = 0
    for marker in _markers(serve_data):
        for axis in _AXES:
            arr = serve_data[marker][axis].astype(float)
            total += len(arr)
            nan_count += np.isnan(arr).sum()
    return (nan_count / total) <= threshold


def trim_serve(
    serve, marker="right_hand", threshold_fraction=0.05, buffer=10, quiet_run=20
):
    """Trim a serve dictionary to the active motion window.

    Finds the peak speed frame, then scans outward in both directions until
    `quiet_run` consecutive frames are all below the threshold. This avoids
    being fooled by isolated noise spikes before or after the main serve.

    Args:
        serve: dict in the format returned by load_multi_serve
        marker: marker to use for speed detection (default 'right_hand')
        threshold_fraction: fraction of peak smoothed speed used as the
                            activity threshold (default 0.05)
        buffer: extra frames to keep beyond the quiet boundary (default 10)
        quiet_run: number of consecutive below-threshold frames required to
                   mark the boundary (default 20)

    Returns:
        A new dict with the same structure as `serve`, sliced to the trimmed
        frame range.
    """
    TX = serve[marker]["TX"]
    TY = serve[marker]["TY"]
    TZ = serve[marker]["TZ"]

    speed = np.sqrt(np.diff(TX) ** 2 + np.diff(TY) ** 2 + np.diff(TZ) ** 2)
    smoothed = pd.Series(speed).rolling(window=10, center=True).mean().values

    threshold = np.nanmax(smoothed) * threshold_fraction
    peak = int(np.nanargmax(smoothed))

    # Scan backwards from peak: stop when quiet_run consecutive frames are
    # all below threshold. Fall back to index 0 if we never see a quiet run.
    first = 0
    run = 0
    for i in range(peak, -1, -1):
        if np.isnan(smoothed[i]) or smoothed[i] < threshold:
            run += 1
            if run >= quiet_run:
                first = i  # beginning of the quiet run (farthest from peak)
                break
        else:
            run = 0

    # Scan forwards from peak: same idea.
    last = len(smoothed) - 1
    run = 0
    for i in range(peak, len(smoothed)):
        if np.isnan(smoothed[i]) or smoothed[i] < threshold:
            run += 1
            if run >= quiet_run:
                last = i  # end of the quiet run (farthest from peak)
                break
        else:
            run = 0

    # speed-space indices need +1 to convert to frame-space; apply buffer
    first = max(0, first - buffer)
    last = min(len(serve["frames"]) - 1, last + 1 + buffer)

    result = {"frames": serve["frames"][first:last]}
    for name, coords in serve.items():
        if name == "frames":
            continue
        result[name] = {axis: coords[axis][first:last] for axis in _AXES}
    return result


def filter_nan_frames(serve_data, threshold=0.5):
    """Drop frames where the fraction of NaN values across all markers exceeds threshold.

    Args:
        serve_data: dict from load_data
        threshold: max allowed NaN fraction per frame (default 0.5)

    Returns:
        serve_data dict with bad frames removed
    """
    markers = _markers(serve_data)
    n_frames = len(serve_data["frames"])
    total_cols = len(markers) * len(_AXES)

    nan_counts = np.zeros(n_frames)
    for marker in markers:
        for axis in _AXES:
            nan_counts += np.isnan(serve_data[marker][axis].astype(float))

    valid = (nan_counts / total_cols) <= threshold
    result = {"frames": serve_data["frames"][valid]}
    for marker in markers:
        result[marker] = {axis: serve_data[marker][axis][valid] for axis in _AXES}
    return result


def interpolate_nans(serve_data):
    """Linearly interpolate NaN values per marker per axis.

    Columns that are entirely NaN are left unchanged.

    Args:
        serve_data: dict from load_data or filter_nan_frames

    Returns:
        serve_data dict with NaNs filled
    """
    markers = _markers(serve_data)
    x = np.arange(len(serve_data["frames"]))

    result = {"frames": serve_data["frames"]}
    for marker in markers:
        result[marker] = {}
        for axis in _AXES:
            y = serve_data[marker][axis].astype(float).copy()
            nans = np.isnan(y)
            if nans.any() and not nans.all():
                y[nans] = np.interp(x[nans], x[~nans], y[~nans])
            result[marker][axis] = y
    return result


def normalize_serve(serve_data, anchor="right_hip"):
    """Normalize all markers relative to an anchor marker's mean position.

    Computes the mean TX, TY, TZ of the anchor marker across all frames,
    then subtracts those values from every marker so the anchor is centred
    around zero and all other markers are positioned relative to it.

    Args:
        serve_data: dict from load_data (after NaN handling)
        anchor: marker to use as the reference point (default 'right_hip')

    Returns:
        new serve_data dict with all marker positions shifted
    """
    anchor_mean = {
        axis: np.nanmean(serve_data[anchor][axis].astype(float)) for axis in _AXES
    }

    markers = _markers(serve_data)
    result = {"frames": serve_data["frames"]}
    for marker in markers:
        result[marker] = {
            axis: serve_data[marker][axis].astype(float) - anchor_mean[axis]
            for axis in _AXES
        }
    return result


def convert(serve_data):
    """Convert a serve dict to a numpy array ordered by MARKER_ORDER.

    Args:
        serve_data: dict from load_data (after filtering and interpolation)

    Returns:
        np.ndarray of shape (n_frames, n_markers * 3)
        Columns ordered as [head_TX, head_TY, head_TZ, left_shoulder_TX, ...]
    """
    columns = []
    for marker in MARKER_ORDER:
        m = serve_data[marker]
        columns.extend([m["TX"], m["TY"], m["TZ"]])
    return np.column_stack(columns)


def load_prepared_serves(dirpath, multi=True, serve_threshold=0.2, frame_threshold=0.5, skip_trim=False):
    """Load all serves from dirpath, run the full preparation pipeline, and return
    the resulting list of numpy arrays.

    When multi=True, expects dirpath to contain one multi-marker CSV per serve.
    When multi=False, expects dirpath to contain one subdirectory per serve, each
    subdirectory holding one CSV file per marker (e.g. righthand.csv, leftelbow.csv).

    Args:
        dirpath: path to the serve data folder
        multi: True for flat multi-marker CSVs, False for per-serve subdirectories
        serve_threshold: passed through to prepare_all_serves
        frame_threshold: passed through to prepare_all_serves

    Returns:
        list of np.ndarray, one per valid serve
    """
    serves = []
    if multi:
        for p in sorted(glob.glob(os.path.join(dirpath, "*.csv"))):
            serve = load_multi_serve(p)
            serve["_filename"] = os.path.basename(p)
            serves.append(serve)
    else:
        serve_dirs = sorted([
            d for d in glob.glob(os.path.join(dirpath, "*"))
            if os.path.isdir(d)
        ])
        for serve_dir in serve_dirs:
            marker_dict = {}
            for csv_path in glob.glob(os.path.join(serve_dir, "*.csv")):
                stem = os.path.splitext(os.path.basename(csv_path))[0].lower()
                marker_name = FILENAME_TO_MARKER.get(stem)
                if marker_name:
                    marker_dict[marker_name] = csv_path
            if marker_dict:
                serve = load_single_serve(marker_dict)
                serve["_filename"] = os.path.basename(serve_dir)
                serves.append(serve)
    return prepare_all_serves(serves, serve_threshold, frame_threshold, skip_trim=skip_trim)


def filter_length_outliers(named_serves, min_frames=300, max_frames=480):
    """Drop serves whose trimmed frame count falls outside [min_frames, max_frames].

    Args:
        named_serves: list of (filename, serve_dict) tuples
        min_frames: minimum allowed frame count (default 300)
        max_frames: maximum allowed frame count (default 480)

    Returns:
        filtered list of (filename, serve_dict) tuples
    """
    kept = []
    for filename, serve in named_serves:
        n = len(serve["frames"])
        if min_frames <= n <= max_frames:
            kept.append((filename, serve))
        else:
            print(
                f"  {filename}: DROPPED (length {n} outside [{min_frames}, {max_frames}])"
            )
    return kept


def prepare_all_serves(
    serves,
    serve_threshold=0.2,
    frame_threshold=0.5,
    min_frames=300,
    max_frames=480,
    skip_trim=False,
):
    """Filter, clean, and convert a list of serve dicts to numpy arrays.

    Pipeline per serve:
      1. Drop entire serve if overall NaN fraction exceeds serve_threshold
      2. Trim to active motion window (skipped when skip_trim=True)
      3. Drop if NaN fraction after trim exceeds serve_threshold
      4. Drop serves whose trimmed length is outside [min_frames, max_frames]
         (skipped when skip_trim=True)
      5. Drop individual frames exceeding frame_threshold
      6. Interpolate remaining NaNs
      7. Normalize and convert to numpy array

    Args:
        serves: list of dicts from load_data
        serve_threshold: max allowed overall NaN fraction to keep a serve (default 0.2)
        frame_threshold: max allowed per-frame NaN fraction to keep a frame (default 0.5)
        min_frames: minimum trimmed frame count (default 300, ignored when skip_trim=True)
        max_frames: maximum trimmed frame count (default 480, ignored when skip_trim=True)
        skip_trim: if True, skip trimming and length filtering entirely (default False)

    Returns:
        list of np.ndarray, one per valid serve
    """
    # Pass 1: NaN checks and optional trimming
    trimmed = []
    for serve in serves:
        filename = serve.pop("_filename", "unknown")
        if not is_valid_serve(serve, serve_threshold):
            print(f"  {filename}: DROPPED (failed NaN check before trim)")
            continue
        if skip_trim:
            print(f"  {filename}: {len(serve['frames'])} frames (trim skipped)")
            trimmed.append((filename, serve))
        else:
            before = len(serve["frames"])
            serve = trim_serve(serve)
            after = len(serve["frames"])
            print(f"  {filename}: {before} frames -> {after} after trim")
            if not is_valid_serve(serve, serve_threshold):
                print(f"  {filename}: DROPPED (failed NaN check after trim)")
                continue
            trimmed.append((filename, serve))

    # Pass 2: drop length outliers (only when trimming is active)
    if not skip_trim and len(trimmed) > 1:
        trimmed = filter_length_outliers(trimmed, min_frames, max_frames)

    # Pass 3: frame-level cleaning and conversion
    results = []
    for filename, serve in trimmed:
        missing = [m for m in MARKER_ORDER if m not in serve]
        if missing:
            print(f"  {filename}: DROPPED (missing markers: {missing})")
            continue
        serve = filter_nan_frames(serve, frame_threshold)
        serve = interpolate_nans(serve)
        serve = normalize_serve(serve)
        arr = convert(serve)
        if np.isnan(arr).any():
            print(
                f"  {filename}: DROPPED (NaN remains after interpolation — marker fully missing)"
            )
            continue
        results.append(arr)
    return results
