import numpy as np
import pandas as pd
from dtaidistance import dtw

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SERVE_PATHS = [
    "../mediapipe_pose/serve_recs/serve_3_bplv/outputs/s1_20260622_210848/s1_coords_angles.csv",
    "../mediapipe_pose/serve_recs/serve_3_bplv/outputs/s2_20260622_210926/s2_coords_angles.csv",
    "../mediapipe_pose/serve_recs/serve_3_bplv/outputs/s3_20260622_211039/s3_coords_angles.csv",
    "../mediapipe_pose/serve_recs/serve_3_bplv/outputs/s4_20260622_211317/s4_coords_angles.csv",
]

# Index into SERVE_PATHS for the serve to use as the initial reference.
# (0 = s1, 1 = s2, 2 = s3, 3 = s4)
REFERENCE_INDEX = 0

N_ITER = 10  # number of DBA refinement passes

ANGLE_COLS = [
    "shoulder_angle",
    "elbow_angle",
    "wrist_angle",
    "hip_rotation",
    "knee_angle",
    "trunk_lean",
]

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

serves = []
for p in SERVE_PATHS:
    df = pd.read_csv(p)
    serves.append(df[ANGLE_COLS].values)  # shape (n_frames, 6)

print("Loaded serves with frame counts:", [s.shape[0] for s in serves])

barycenter = serves[REFERENCE_INDEX].copy().astype(float)
others = [s for i, s in enumerate(serves) if i != REFERENCE_INDEX]

# ---------------------------------------------------------------------------
# DBA: iteratively align every serve to the current barycenter, then
# update each barycenter frame as the mean of everything that aligned to it.
# DTW alignment is computed per-column (angles can move somewhat
# independently, e.g. elbow extending while hips are already rotating back),
# but the same path is reused consistently within a column across iterations.
# ---------------------------------------------------------------------------

def dba_update(barycenter, others):
    n_frames, n_cols = barycenter.shape
    new_bc = np.zeros_like(barycenter)

    for col_idx in range(n_cols):
        accum = [[] for _ in range(n_frames)]
        bc_col = barycenter[:, col_idx]

        for serve in others:
            other_col = serve[:, col_idx]
            _, paths = dtw.warping_paths(bc_col, other_col)
            best = dtw.best_path(paths)
            for i, j in best:
                accum[i].append(other_col[j])

        for i in range(n_frames):
            if accum[i]:
                # include the barycenter's own current value alongside the
                # aligned values from the other serves for a balanced mean
                new_bc[i, col_idx] = np.mean(accum[i] + [bc_col[i]])
            else:
                new_bc[i, col_idx] = bc_col[i]

    return new_bc


for it in range(N_ITER):
    barycenter = dba_update(barycenter, others)
    print(f"  DBA iteration {it + 1}/{N_ITER} complete")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out = pd.DataFrame(np.round(barycenter, 2), columns=[f"reference_{c}" for c in ANGLE_COLS])
out.insert(0, "phase", range(len(out)))

# Flag any frame-to-frame jump > 20 degrees as a possible residual artifact.
for c in ANGLE_COLS:
    jumps = np.abs(np.diff(out[f"reference_{c}"].values))
    flagged = np.where(jumps > 20)[0]
    if len(flagged) > 0:
        print(f"  WARNING: reference_{c} has {len(flagged)} jump(s) > 20 deg "
              f"at phases {flagged.tolist()[:10]}{'...' if len(flagged) > 10 else ''}")

out.to_csv("avg_serve.csv", index=False)
print("avg_serve.csv created!")
print("Frames:", len(out))