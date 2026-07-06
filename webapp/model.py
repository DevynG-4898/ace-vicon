"""
Unified serve scoring model.

Both CSV (Vicon motion capture) and video (MediaPipe-derived) uploads
are reduced to a single 1D "marker trajectory" — magnitude of x/y/z
motion over time, resampled to a fixed length — so they can be scored
against the same reference model with the same formula.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

TARGET_LENGTH = 200


# ---------------------------
# CSV loading (Vicon format: TX/TY/TZ marker columns, no header skip)
# ---------------------------
def _load_vicon_marker_csv(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna(axis=1, how="all")
    df = df.dropna()

    marker_cols = [c for c in df.columns if "TX" in c or "TY" in c or "TZ" in c]
    if not marker_cols:
        return None  # not this format

    xyz = df[marker_cols[:3]].values  # first marker's x/y/z
    return xyz


# ---------------------------
# CSV loading (fallback: raw numeric export, 3 metadata header rows)
# ---------------------------
def _load_raw_numeric_csv(filepath):
    df = pd.read_csv(filepath, skiprows=3, header=None)
    df = df.dropna(axis=1, how="all")
    df = df.dropna()
    numeric = df.select_dtypes(include=[np.number]).values

    if numeric.shape[1] < 3:
        raise ValueError(
            "CSV does not contain at least 3 numeric columns of motion data."
        )
    return numeric[:, :3]


def csv_to_marker_trajectory(filepath):
    """
    Load a CSV (either Vicon-labeled-column format or raw numeric export)
    and reduce it to a single normalized magnitude trajectory.
    """
    xyz = _load_vicon_marker_csv(filepath)
    if xyz is None:
        xyz = _load_raw_numeric_csv(filepath)

    if xyz.shape[0] == 0:
        raise ValueError("No usable rows found in uploaded CSV.")

    traj = np.sqrt(np.sum(xyz**2, axis=1))
    return normalize_trajectory(traj)


# ---------------------------
# Normalize trajectory length so different recordings are comparable
# ---------------------------
def normalize_trajectory(traj, target_length=TARGET_LENGTH):
    traj = np.asarray(traj, dtype=float)
    x_old = np.linspace(0, 1, len(traj))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, traj, kind="linear")
    return f(x_new)


# ---------------------------
# Build probabilistic reference model from multiple reference CSVs
# ---------------------------
def build_reference_model(reference_files):
    trajectories = [csv_to_marker_trajectory(f) for f in reference_files]
    stacked = np.vstack(trajectories)

    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    std[std == 0] = 1e-6

    return mean, std


# ---------------------------
# Similarity scoring (shared by CSV and video paths)
# ---------------------------
def score_trajectory(user_traj, mean_traj, std_traj):
    z_scores = (user_traj - mean_traj) / std_traj
    avg_z = np.mean(np.abs(z_scores))
    score = 100 * np.exp(-avg_z)
    return float(score), float(avg_z)


# ---------------------------
# Public entry points used by app.py
# ---------------------------
def compute_similarity_from_csv(user_file, reference_files):
    mean, std = build_reference_model(reference_files)
    user_traj = csv_to_marker_trajectory(user_file)
    score, avg_z = score_trajectory(user_traj, mean, std)
    return score, avg_z, mean, user_traj


def compute_similarity_from_video(video_file, reference_files):
    from video_pose import video_to_marker_trajectory

    mean, std = build_reference_model(reference_files)
    raw_traj = video_to_marker_trajectory(video_file)
    user_traj = normalize_trajectory(raw_traj)
    score, avg_z = score_trajectory(user_traj, mean, std)
    return score, avg_z, mean, user_traj