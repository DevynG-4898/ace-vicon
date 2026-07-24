"""DTW distance computation and warping paths."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def dtw_warping_path(
    series_a: np.ndarray,
    series_b: np.ndarray,
    *,
    metric: str = "euclidean",
) -> tuple[list[tuple[int, int]], float]:
    """Compute DTW warping path and total accumulated cost.

    Parameters
    ----------
    series_a, series_b
        Arrays of shape ``(n_frames, n_features)``.

    Returns
    -------
    path
        List of ``(index_a, index_b)`` pairs in chronological order.
    total_cost
        Sum of pairwise frame costs along the path.
    """
    a = np.nan_to_num(np.asarray(series_a, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(np.asarray(series_b, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("series_a and series_b must be 2D (frames, features)")
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return [], 0.0
    if n == 1 and m == 1:
        c = float(cdist(a, b, metric=metric)[0, 0])
        return [(0, 0)], c

    cost = cdist(a, b, metric=metric)
    cost = np.nan_to_num(cost, nan=0.0, posinf=0.0, neginf=0.0)

    d = np.full((n + 1, m + 1), np.inf, dtype=float)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = cost[i - 1, j - 1]
            d[i, j] = c + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])

    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step_costs = (
            (d[i - 1, j - 1], i - 1, j - 1),
            (d[i - 1, j], i - 1, j),
            (d[i, j - 1], i, j - 1),
        )
        _, i, j = min(step_costs, key=lambda item: item[0])

    path.reverse()
    total = float(d[n, m])
    return path, total


def normalized_dtw_distance(total_cost: float, path: list[tuple[int, int]], n_a: int, n_b: int) -> float:
    """Normalize DTW cost by path length and feature scale reference."""
    if not path or not np.isfinite(total_cost):
        return 0.0 if not path else float(total_cost)
    denom = max(len(path), max(n_a, n_b), 1)
    return total_cost / denom
