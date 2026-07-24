"""Load and save ACE marker dictionaries.

The canonical ACE marker format matches ``load_single_serve()`` output:

    {
        "frames": np.ndarray[int],          # one Vicon-style frame id per row
        "head":   {"TX": ..., "TY": ..., "TZ": ...},
        "chest":  {"TX": ..., "TY": ..., "TZ": ...},
        ...  # 14 anatomical markers total
    }

Every marker value is a 1D float array aligned with ``frames``.
Sources (Vicon CSV folders and MotionBERT NPZ) must produce this structure so
downstream plotting and segmentation cannot distinguish the origin.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_DTW_DIR = Path(__file__).resolve().parent.parent.parent / "dtw"
if str(_DTW_DIR) not in sys.path:
    sys.path.insert(0, str(_DTW_DIR))

from src.load_data import FILENAME_TO_MARKER, load_single_serve  # noqa: E402

ACE_MARKER_NAMES = (
    "head",
    "chest",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hand",
    "right_hand",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
)

SERVE_MARKERS_NPZ = "ace_markers.npz"


def is_marker_dict(value: Any) -> bool:
    """Return True if ``value`` looks like a canonical ACE marker dictionary."""
    if not isinstance(value, dict) or "frames" not in value:
        return False
    frames = value["frames"]
    if not isinstance(frames, np.ndarray) or frames.ndim != 1:
        return False
    n = len(frames)
    for name in ACE_MARKER_NAMES:
        marker = value.get(name)
        if not isinstance(marker, dict):
            return False
        for axis in ("TX", "TY", "TZ"):
            arr = marker.get(axis)
            if not isinstance(arr, np.ndarray) or arr.shape != (n,):
                return False
    return True


def _load_from_folder(folder: str | Path) -> dict[str, Any]:
    folder = Path(folder)
    marker_paths: dict[str, str] = {}
    for csv_path in glob.glob(str(folder / "*.csv")):
        stem = os.path.splitext(os.path.basename(csv_path))[0].lower()
        marker_name = FILENAME_TO_MARKER.get(stem)
        if marker_name:
            marker_paths[marker_name] = csv_path
    return load_single_serve(marker_paths)


def _load_from_npz(path: Path) -> dict[str, Any]:
    loaded = np.load(path)
    markers: dict[str, Any] = {"frames": np.asarray(loaded["frames"], dtype=int)}
    for marker in ACE_MARKER_NAMES:
        markers[marker] = {
            "TX": np.asarray(loaded[f"{marker}_TX"], dtype=float),
            "TY": np.asarray(loaded[f"{marker}_TY"], dtype=float),
            "TZ": np.asarray(loaded[f"{marker}_TZ"], dtype=float),
        }
    return markers


def load_serve_markers(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load ACE markers from a folder, NPZ file, or pass through an in-memory dict."""
    if is_marker_dict(source):
        return source

    path = Path(source)
    if path.suffix == ".npz":
        return _load_from_npz(path)
    if path.is_dir():
        npz_candidate = path / SERVE_MARKERS_NPZ
        if npz_candidate.is_file():
            return _load_from_npz(npz_candidate)
        return _load_from_folder(path)

    raise FileNotFoundError(
        f"Cannot load ACE markers from {source}. "
        "Expected a marker dict, .npz file, or directory of Vicon CSVs."
    )


def save_serve_markers(
    output_dir: str | Path,
    markers: dict[str, Any],
    *,
    filename: str = SERVE_MARKERS_NPZ,
) -> Path:
    """Persist a marker dictionary as ``ace_markers.npz``."""
    if not is_marker_dict(markers):
        raise ValueError("markers must be a canonical ACE marker dictionary")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    arrays: dict[str, np.ndarray] = {"frames": np.asarray(markers["frames"], dtype=int)}
    for marker in ACE_MARKER_NAMES:
        marker_axes = markers[marker]
        for axis in ("TX", "TY", "TZ"):
            arrays[f"{marker}_{axis}"] = np.asarray(marker_axes[axis], dtype=float)
    np.savez(path, **arrays)
    return path
