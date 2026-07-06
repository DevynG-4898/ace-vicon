import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


# ---------------------------
# Load and clean Vicon CSV
# ---------------------------
def load_vicon_csv(filepath):

    df = pd.read_csv(filepath)

    # Drop columns that are completely empty
    df = df.dropna(axis=1, how="all")

    # Remove rows without frame data
    df = df.dropna()

    return df


# ---------------------------
# Extract marker trajectories
# ---------------------------
def extract_markers(df):

    marker_cols = [c for c in df.columns if "TX" in c or "TY" in c or "TZ" in c]

    markers = {}

    for i in range(0, len(marker_cols), 3):
        name = f"marker_{i//3}"

        markers[name] = df[marker_cols[i : i + 3]].values

    return markers


# ---------------------------
# Compute trajectory magnitude
# ---------------------------
def compute_marker_trajectory(marker):

    x = marker[:, 0]
    y = marker[:, 1]
    z = marker[:, 2]

    trajectory = np.sqrt(x**2 + y**2 + z**2)

    return trajectory


# ---------------------------
# Normalize trajectory length
# ---------------------------
def normalize_trajectory(traj, target_length=200):

    x_old = np.linspace(0, 1, len(traj))
    x_new = np.linspace(0, 1, target_length)

    f = interp1d(x_old, traj, kind="linear")

    return f(x_new)


# ---------------------------
# Build probabilistic model
# ---------------------------
def build_reference_model(trajectories):

    stacked = np.vstack(trajectories)

    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)

    return mean, std


# ---------------------------
# Similarity scoring
# ---------------------------
def compute_similarity(user_traj, mean_traj, std_traj):

    # avoid divide by zero
    std_traj[std_traj == 0] = 1e-6

    z_scores = (user_traj - mean_traj) / std_traj

    avg_z = np.mean(np.abs(z_scores))

    score = 100 * np.exp(-avg_z)

    return score, avg_z


# ---------------------------
# Full evaluation pipeline
# ---------------------------
def evaluate_serve(user_file, reference_files):

    ref_trajs = []

    for f in reference_files:

        df = load_vicon_csv(f)
        markers = extract_markers(df)

        marker = list(markers.values())[0]

        traj = compute_marker_trajectory(marker)

        traj = normalize_trajectory(traj)

        ref_trajs.append(traj)

    mean_traj, std_traj = build_reference_model(ref_trajs)

    df_user = load_vicon_csv(user_file)

    markers_user = extract_markers(df_user)

    marker_user = list(markers_user.values())[0]

    user_traj = compute_marker_trajectory(marker_user)

    user_traj = normalize_trajectory(user_traj)

    score, avg_z = compute_similarity(user_traj, mean_traj, std_traj)

    return {
        "similarity_score": score,
        "avg_z_score": avg_z,
        "reference_mean": mean_traj,
        "user_traj": user_traj,
    }