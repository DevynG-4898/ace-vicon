import pandas as pd
import numpy as np
from dtaidistance import dtw


# ---------------------------------------------------------------------------
# Angle helpers
# ---------------------------------------------------------------------------

def calculate_elbow_angle(df, arm):
    S = df[[f"{arm}_shoulder_x", f"{arm}_shoulder_y", f"{arm}_shoulder_z"]].values
    E = df[[f"{arm}_elbow_x", f"{arm}_elbow_y", f"{arm}_elbow_z"]].values
    W = df[[f"{arm}_wrist_x", f"{arm}_wrist_y", f"{arm}_wrist_z"]].values

    angles = []
    for s, e, w in zip(S, E, W):
        v1 = s - e
        v2 = w - e
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angles.append(np.degrees(np.arccos(cos_angle)))
    return np.array(angles)


def detect_serving_arm(df):
    """Pick whichever of l_/r_ has higher mean elbow visibility in this file.

    MediaPipe's l_/r_ labels are camera-relative, not player-relative, so the
    serving arm can show up under either letter depending on which side of
    the body faces that camera. We never assume -- we check.
    """
    r_vis = df["r_elbow_vis"].mean()
    l_vis = df["l_elbow_vis"].mean()
    arm = "r" if r_vis >= l_vis else "l"
    print(f"  serving arm detected: '{arm}' (r_vis={r_vis:.3f}, l_vis={l_vis:.3f})")
    return arm


# Angle columns that are already precomputed per-frame in the CSVs and are
# NOT arm-letter-specific (whole-body / single-value angles).
WHOLE_BODY_ANGLES = ["hip_rotation", "knee_angle", "trunk_lean"]

ALL_ANGLES = ["shoulder_angle", "elbow_angle", "wrist_angle"] + WHOLE_BODY_ANGLES


def load_view(path):
    """Load one view's CSV, detect serving arm, return angle dict + vis dict."""
    df = pd.read_csv(path)
    arm = detect_serving_arm(df)

    angles = {}
    vis = {}

    angles["elbow_angle"] = calculate_elbow_angle(df, arm)
    vis["elbow_angle"] = df[f"{arm}_elbow_vis"].values

    # shoulder/wrist angle columns are already in the CSV (single column,
    # not split by l_/r_), so just pull them directly, paired with the
    # detected arm's vis as the trust signal.
    angles["shoulder_angle"] = df["shoulder_angle"].values
    vis["shoulder_angle"] = df[f"{arm}_shoulder_vis"].values

    angles["wrist_angle"] = df["wrist_angle"].values
    vis["wrist_angle"] = df[f"{arm}_wrist_vis"].values

    # whole-body angles have no per-arm vis column -- use elbow vis (same
    # camera/occlusion conditions) as a proxy trust signal.
    for col in WHOLE_BODY_ANGLES:
        angles[col] = df[col].values
        vis[col] = df[f"{arm}_elbow_vis"].values

    return angles, vis


def align_path(base_signal, other_signal):
    """Return best_path list of (base_idx, other_idx) via DTW on elbow angle."""
    _, paths = dtw.warping_paths(base_signal, other_signal)
    return dtw.best_path(paths)


def bucket_by_base(path, other_angles, other_vis, n_base):
    """Group (value, vis) pairs from `other` by the base index they align to."""
    bucketed = {col: [[] for _ in range(n_base)] for col in ALL_ANGLES}
    for i, j in path:
        for col in ALL_ANGLES:
            bucketed[col][i].append((other_angles[col][j], other_vis[col][j]))
    return bucketed


def fuse(base_angles, base_vis, bucketed_views, base_weight=2.0):
    """Visibility-weighted fuse across base + any number of other views.

    base_weight multiplies the base view's (side_2) trust so it dominates
    unless it is itself occluded at that frame.
    """
    n = len(next(iter(base_angles.values())))
    result = {col: np.zeros(n) for col in ALL_ANGLES}

    for col in ALL_ANGLES:
        for i in range(n):
            values = [base_angles[col][i]]
            weights = [base_vis[col][i] * base_weight]

            for bucketed in bucketed_views:
                for v, vis in bucketed[col][i]:
                    values.append(v)
                    weights.append(vis)

            weights = np.array(weights)
            values = np.array(values)
            if weights.sum() == 0:
                result[col][i] = np.mean(values)
            else:
                result[col][i] = np.average(values, weights=weights)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("Loading front...")
front_angles, front_vis = load_view(
    "../mediapipe_pose/serve_recs/serve_1/outputs/mp_2_front_20260617_154200/mp_2_front_coords_angles.csv"
)

print("Loading side_1...")
side1_angles, side1_vis = load_view(
    "../mediapipe_pose/serve_recs/serve_1/outputs/mp_2_side_1_20260617_155223/mp_2_side_1_coords_angles.csv"
)

print("Loading side_2 (base timeline)...")
side2_angles, side2_vis = load_view(
    "../mediapipe_pose/serve_recs/serve_1/outputs/mp_2_side_2_20260617_155628/mp_2_side_2_coords_angles.csv"
)

n_base = len(side2_angles["elbow_angle"])

print("Aligning front -> side_2 (DTW on elbow angle)...")
path_front = align_path(side2_angles["elbow_angle"], front_angles["elbow_angle"])

print("Aligning side_1 -> side_2 (DTW on elbow angle)...")
path_side1 = align_path(side2_angles["elbow_angle"], side1_angles["elbow_angle"])

bucketed_front = bucket_by_base(path_front, front_angles, front_vis, n_base)
bucketed_side1 = bucket_by_base(path_side1, side1_angles, side1_vis, n_base)

print("Fusing all angles (side_2-dominant, visibility-weighted)...")
reference = fuse(
    side2_angles, side2_vis,
    bucketed_views=[bucketed_front, bucketed_side1],
    base_weight=2.0,
)

# Sanity check: flag any frame-to-frame jump bigger than 20 degrees, which
# usually means a residual DTW-alignment or occlusion artifact slipped
# through rather than real joint motion.
out = {"phase": range(n_base)}
for col in ALL_ANGLES:
    out[f"reference_{col}"] = reference[col]

    jumps = np.abs(np.diff(reference[col]))
    flagged = np.where(jumps > 20)[0]
    if len(flagged) > 0:
        print(f"  WARNING: {col} has {len(flagged)} frame-to-frame jump(s) > 20 deg "
              f"at phases {flagged.tolist()[:10]}{'...' if len(flagged) > 10 else ''}")

pd.DataFrame(out).to_csv("reference_serve.csv", index=False)

print("Reference serve created!")
print("Frames:", n_base)
print("Columns:", list(out.keys()))