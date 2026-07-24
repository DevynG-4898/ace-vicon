"""Convert ACE marker dictionaries into feature matrices for DTW."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.markers.io import ACE_MARKER_NAMES


class FeatureExtractor(Protocol):
    """Protocol for pluggable per-frame feature vectors."""

    def __call__(self, markers: dict) -> np.ndarray:
        """Return shape ``(n_frames, n_features)``."""
        ...


class MarkerFeatureExtractor:
    """Stack ACE marker TX/TY/TZ channels with hip-centered normalization.

    Future extractors (joint angles, velocities, COM) can implement the same
    protocol without changing the DTW or animation APIs.
    """

    def __init__(
        self,
        marker_names: tuple[str, ...] = ACE_MARKER_NAMES,
        anchor: str = "right_hip",
        scale_by_shoulder_width: bool = True,
    ) -> None:
        self.marker_names = marker_names
        self.anchor = anchor
        self.scale_by_shoulder_width = scale_by_shoulder_width

    def __call__(self, markers: dict) -> np.ndarray:
        n = len(markers["frames"])
        columns: list[np.ndarray] = []
        for name in self.marker_names:
            m = markers[name]
            columns.extend(
                [
                    np.asarray(m["TX"], dtype=float),
                    np.asarray(m["TY"], dtype=float),
                    np.asarray(m["TZ"], dtype=float),
                ]
            )
        raw = np.column_stack(columns)
        raw = np.nan_to_num(raw, nan=0.0)

        anchor_pos = np.column_stack(
            [
                markers[self.anchor]["TX"].astype(float),
                markers[self.anchor]["TY"].astype(float),
                markers[self.anchor]["TZ"].astype(float),
            ]
        )
        n_markers = len(self.marker_names)
        reshaped = raw.reshape(len(raw), n_markers, 3)
        centered = reshaped - anchor_pos[:, np.newaxis, :]
        flat = centered.reshape(len(raw), n_markers * 3)
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

        if not self.scale_by_shoulder_width:
            return flat

        ls = markers["left_shoulder"]
        rs = markers["right_shoulder"]
        shoulder = np.linalg.norm(
            np.column_stack([rs["TX"] - ls["TX"], rs["TY"] - ls["TY"], rs["TZ"] - ls["TZ"]]),
            axis=1,
        )
        scale = float(np.nanmedian(shoulder[np.isfinite(shoulder) & (shoulder > 1e-6)]))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        return flat / scale
