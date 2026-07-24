"""Convert ACE marker dictionaries into SkeletonSequence for scalar extraction."""

from __future__ import annotations

import numpy as np

from src.skeleton import SkeletonSequence

# ACE name → skeleton joint name used by features / angle_series
_ACE_TO_SKELETON = {
    "head": "head",
    "chest": "thorax",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_hand": "left_wrist",
    "right_hand": "right_wrist",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
}


def ace_markers_to_skeleton(
    markers: dict,
    *,
    fps: float | None = None,
) -> SkeletonSequence:
    """Build a Y-up SkeletonSequence from an ACE ``TX/TY/TZ`` marker dict."""
    frames = np.asarray(markers["frames"])
    n = int(frames.shape[0])
    names: list[str] = []
    cols: list[np.ndarray] = []

    for ace_name, skel_name in _ACE_TO_SKELETON.items():
        if ace_name not in markers:
            continue
        m = markers[ace_name]
        # Inverse of motionbert_to_ace_markers: Y-up from Vicon Z-up
        xyz = np.column_stack(
            [
                np.asarray(m["TX"], dtype=float),
                np.asarray(m["TZ"], dtype=float),
                np.asarray(m["TY"], dtype=float),
            ]
        )
        if xyz.shape[0] != n:
            raise ValueError(f"{ace_name} length {xyz.shape[0]} != frames {n}")
        names.append(skel_name)
        cols.append(xyz)

    if "left_hip" in markers and "right_hip" in markers and "pelvis" not in names:
        lh = cols[names.index("left_hip")]
        rh = cols[names.index("right_hip")]
        names.append("pelvis")
        cols.append(0.5 * (lh + rh))

    if not names:
        raise ValueError("No usable ACE markers found")

    positions = np.stack(cols, axis=1)
    return SkeletonSequence(
        frames=frames,
        joint_names=tuple(names),
        joint_positions=positions,
        joint_confidence=None,
        fps=fps,
        metadata={"source": "ace_markers"},
        coordinate_system="ace_y_up",
        source="ACE",
    )
