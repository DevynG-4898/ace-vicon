import os

import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.barycenters import softdtw_barycenter

from prepare_data import load_prepared_serves

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "plotting", "markers", "individual"
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "barycenter1.npy")
CSV_PATH = os.path.join(os.path.dirname(__file__), "barycenter1.csv")


def compute_barycenter(dirpath=DATA_DIR, out_path=OUT_PATH, csv_path=CSV_PATH):
    """Load all serves, compute the DTW barycenter via tslearn, and save it.

    Args:
        dirpath: folder of multi-marker Vicon CSVs (unmarked_edited)
        out_path: .npy file to write the barycenter to
        csv_path: .csv file to write the barycenter to

    Returns:
        barycenter as np.ndarray of shape (n_frames, n_features)
    """
    arrays = load_prepared_serves(dirpath, multi=False, skip_trim=True)
    print(f"Loaded {len(arrays)} valid serves")

    barycenter = softdtw_barycenter(arrays)
    print(f"Barycenter shape: {barycenter.shape}")

    np.save(out_path, barycenter)
    print(f"Saved to {out_path}")

    np.savetxt(csv_path, barycenter, delimiter=",")
    print(f"Saved to {csv_path}")

    return barycenter


if __name__ == "__main__":
    compute_barycenter()
