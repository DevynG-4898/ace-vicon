import os

import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.barycenters import softdtw_barycenter

from prepare_data import load_raw_serves, prepare_all_serves

MULTI_DIR = os.path.join(
    os.path.dirname(__file__), "..", "plotting", "markers", "multi"
)
INDIVIDUAL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "plotting", "markers", "individual"
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "barycenter1.npz")
CSV_PATH = os.path.join(os.path.dirname(__file__), "barycenter1.csv")


def compute_barycenter(
    multi_dir=MULTI_DIR,
    individual_dir=INDIVIDUAL_DIR,
    out_path=OUT_PATH,
    csv_path=CSV_PATH,
):
    """Load all serves from both data sources, compute the DTW barycenter via tslearn, and save it.

    Args:
        multi_dir: folder of multi-marker Vicon CSVs (unmarked_edited)
        individual_dir: folder of per-serve subdirectories with per-marker CSVs (individual)
        out_path: .npy file to write the barycenter to
        csv_path: .csv file to write the barycenter to

    Returns:
        barycenter as np.ndarray of shape (n_frames, n_features)
    """
    raw = load_raw_serves(multi_dir, mode="multi", skip_trim=False) + load_raw_serves(
        individual_dir, mode="individual", skip_trim=True
    )
    arrays, common_markers = prepare_all_serves(raw)
    print(f"Loaded {len(arrays)} valid serves")

    barycenter = softdtw_barycenter(arrays)
    print(f"Barycenter shape: {barycenter.shape}")

    np.savez(out_path, barycenter=barycenter, markers=common_markers)
    print(f"Saved to {out_path}")

    np.savetxt(csv_path, barycenter, delimiter=",")
    print(f"Saved to {csv_path}")

    return barycenter


if __name__ == "__main__":
    compute_barycenter()
