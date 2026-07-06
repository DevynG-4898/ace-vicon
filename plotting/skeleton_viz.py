"""Shared 3D skeleton drawing for ACE Vicon marker serves."""

from __future__ import annotations

import glob
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_DTW_DIR = os.path.join(os.path.dirname(__file__), "..", "dtw")
if _DTW_DIR not in sys.path:
    sys.path.insert(0, _DTW_DIR)

from load_data import FILENAME_TO_MARKER, load_single_serve  # noqa: E402

BONES: list[tuple[str, str]] = [
    ("head", "chest"),
    ("chest", "left_shoulder"),
    ("chest", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "left_hand"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
    ("left_hip", "right_hip"),
    ("chest", "left_hip"),
    ("chest", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_foot"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_foot"),
]

DEFAULT_INTERVAL_MS = 33


def load_serve_from_dir(serve_dir: str) -> dict[str, Any]:
    marker_dict = {}
    for csv_path in glob.glob(os.path.join(serve_dir, "*.csv")):
        stem = os.path.splitext(os.path.basename(csv_path))[0].lower()
        marker_name = FILENAME_TO_MARKER.get(stem)
        if marker_name:
            marker_dict[marker_name] = csv_path
    return load_single_serve(marker_dict)


def marker_names(markers: dict) -> list[str]:
    return sorted(k for k in markers if k != "frames")


def get_pos(markers: dict, joint: str, frame_idx: int) -> tuple[float, float, float]:
    m = markers[joint]
    return (
        float(m["TX"][frame_idx]),
        float(m["TY"][frame_idx]),
        float(m["TZ"][frame_idx]),
    )


def padded_limits(data: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


def compute_axis_limits(markers: dict) -> tuple[tuple[float, float], ...]:
    names = marker_names(markers)
    all_x = np.concatenate([markers[m]["TX"] for m in names])
    all_y = np.concatenate([markers[m]["TY"] for m in names])
    all_z = np.concatenate([markers[m]["TZ"] for m in names])
    all_x = all_x[~np.isnan(all_x)]
    all_y = all_y[~np.isnan(all_y)]
    all_z = all_z[~np.isnan(all_z)]
    return padded_limits(all_x), padded_limits(all_y), padded_limits(all_z)


def marker_color_map(marker_names_list: list[str]) -> dict[str, Any]:
    cmap = plt.colormaps["tab20"].resampled(max(len(marker_names_list), 1))
    return {name: cmap(i) for i, name in enumerate(marker_names_list)}


def apply_axes(
    ax,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    z_lim: tuple[float, float],
) -> None:
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    z_range = z_lim[1] - z_lim[0]
    ax.set_box_aspect([x_range, y_range, z_range])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def draw_skeleton(
    ax,
    markers: dict,
    frame_idx: int,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    z_lim: tuple[float, float],
    colors: dict[str, Any] | None = None,
    show_legend: bool = True,
) -> None:
    names = marker_names(markers)
    if colors is None:
        colors = marker_color_map(names)

    apply_axes(ax, x_lim, y_lim, z_lim)

    for joint in names:
        x, y, z = get_pos(markers, joint, frame_idx)
        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
            ax.scatter(x, y, z, s=20, color=colors[joint], label=joint)

    for start, end in BONES:
        if start not in markers or end not in markers:
            continue
        x0, y0, z0 = get_pos(markers, start, frame_idx)
        x1, y1, z1 = get_pos(markers, end, frame_idx)
        if any(np.isnan(v) for v in [x0, y0, z0, x1, y1, z1]):
            continue
        ax.plot([x0, x1], [y0, y1], [z0, z1], color="steelblue", linewidth=1.5)

    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7, framealpha=0.7)
