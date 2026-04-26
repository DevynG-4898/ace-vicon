import os

import numpy as np
from dtw import dtw
from scipy.spatial.distance import cdist

from prepare_data import load_prepared_serves

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "plotting", "markers", "individual"
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "barycenter2.npy")
CSV_PATH = os.path.join(os.path.dirname(__file__), "barycenter2.csv")


def _dba_update(barycenter, arrays):
    """One DBA iteration: align each series to the barycenter and average.

    For each series, computes the DTW warping path against the current
    barycenter, then updates each barycenter frame as the mean of all
    series frames mapped to it.
    """
    n = barycenter.shape[0]
    accum = [[] for _ in range(n)]

    for arr in arrays:
        dist_mat = cdist(arr, barycenter)
        alignment = dtw(dist_mat, distance_only=False)
        for i, j in zip(alignment.index1, alignment.index2):
            accum[j].append(arr[i])

    new_bc = np.zeros_like(barycenter)
    for j in range(n):
        if accum[j]:
            new_bc[j] = np.mean(accum[j], axis=0)
        else:
            new_bc[j] = barycenter[j]
    return new_bc


def compute_barycenter(
    dirpath=DATA_DIR, out_path=OUT_PATH, csv_path=CSV_PATH, n_iter=30
):
    """Load all serves, compute the DTW barycenter via DBA with dtw-python, and save it.

    Args:
        dirpath: folder of multi-marker Vicon CSVs (unmarked_edited)
        out_path: .npy file to write the barycenter to
        csv_path: .csv file to write the barycenter to
        n_iter: number of DBA iterations (default 30)

    Returns:
        barycenter as np.ndarray of shape (n_frames, n_features)
    """
    arrays = load_prepared_serves(dirpath, multi=False, skip_trim=True)
    print(f"Loaded {len(arrays)} valid serves")

    # Initialise with the series closest to the median length to avoid
    # extreme-length bias in the first round of alignments.
    lengths = [a.shape[0] for a in arrays]
    init_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])[len(lengths) // 2]
    barycenter = arrays[init_idx].copy().astype(float)

    for i in range(n_iter):
        barycenter = _dba_update(barycenter, arrays)
        print(f"  DBA iteration {i + 1}/{n_iter}")

    print(f"Barycenter shape: {barycenter.shape}")
    np.save(out_path, barycenter)
    print(f"Saved to {out_path}")

    np.savetxt(csv_path, barycenter, delimiter=",")
    print(f"Saved to {csv_path}")

    return barycenter


if __name__ == "__main__":
    compute_barycenter()
