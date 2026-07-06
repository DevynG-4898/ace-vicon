import os

import numpy as np
from dtw import dtw
from scipy.spatial.distance import cdist

from prepare_data import load_raw_serves, prepare_all_serves

MULTI_DIR = os.path.join(
    os.path.dirname(__file__), "..", "plotting", "markers", "unmarked_edited"
)
INDIVIDUAL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "plotting", "markers", "individual"
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "barycenter.npz")


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


def compute_barycenter(multi_dir=MULTI_DIR, individual_dir=INDIVIDUAL_DIR, out_path=OUT_PATH, n_iter=30):
    """Load all serves from both data sources, compute the DTW barycenter via DBA, and save it.

    Args:
        multi_dir: folder of multi-marker Vicon CSVs (unmarked_edited)
        individual_dir: folder of per-serve subdirectories with per-marker CSVs (individual)
        out_path: .npy file to write the barycenter to
        n_iter: number of DBA iterations (default 30)

    Returns:
        barycenter as np.ndarray of shape (n_frames, n_features)
    """
    raw = (
        load_raw_serves(multi_dir, mode="multi", skip_trim=False)
        + load_raw_serves(individual_dir, mode="individual", skip_trim=True)
    )
    arrays, common_markers = prepare_all_serves(raw)
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
    np.savez(out_path, barycenter=barycenter, markers=common_markers)
    print(f"Saved to {out_path}")

    return barycenter


if __name__ == "__main__":
    compute_barycenter()
