"""Source-independent display skeleton for the Coaching Viewer.

Stage 1 — ``to_common_positions``:
    ACE marker dict (Vicon or MotionBERT) → ``(frames, joints, 3)`` in ACE
    ``TX/TY/TZ`` order. No source-specific drawing logic.

Stage 2 — ``orient_common_positions``:
    Center (pelvis), orient (body axes), uniform scale. NaN-safe so a single
    missing Vicon marker cannot wipe an entire frame.

Visualization only — never mutate source markers used by DTW / scoring.
"""

from __future__ import annotations

import copy

import numpy as np

from src.markers.io import ACE_MARKER_NAMES
from src.skeleton.geometry import EPSILON, median_positive_length, orthonormal_body_axes
from src.skeleton.transforms import rotate_positions, scale_positions

_ORIENT_JOINTS = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")


def to_common_positions(markers: dict) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Stage 1: ACE markers → common ``(frames, joints, 3)`` positions + names + frames."""
    names = tuple(n for n in ACE_MARKER_NAMES if n in markers)
    missing = [n for n in _ORIENT_JOINTS if n not in names]
    if missing:
        raise ValueError(f"Cannot build display skeleton; missing joints: {missing}")
    cols = [
        np.column_stack(
            [
                np.asarray(markers[name]["TX"], dtype=float),
                np.asarray(markers[name]["TY"], dtype=float),
                np.asarray(markers[name]["TZ"], dtype=float),
            ]
        )
        for name in names
    ]
    positions = np.stack(cols, axis=1)
    frames = np.asarray(markers["frames"]).copy()
    return positions, names, frames


def _finite_rows(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=1)


def _estimate_pelvis(left_hip: np.ndarray, right_hip: np.ndarray) -> np.ndarray:
    """Pelvis mid-point; fall back to the single available hip when one is NaN."""
    pelvis = np.full_like(left_hip, np.nan)
    both = _finite_rows(left_hip) & _finite_rows(right_hip)
    only_l = _finite_rows(left_hip) & ~_finite_rows(right_hip)
    only_r = _finite_rows(right_hip) & ~_finite_rows(left_hip)
    pelvis[both] = 0.5 * (left_hip[both] + right_hip[both])
    pelvis[only_l] = left_hip[only_l]
    pelvis[only_r] = right_hip[only_r]
    return pelvis


def _impute_hip_pair(
    left_hip: np.ndarray,
    right_hip: np.ndarray,
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill a missing hip for axis estimation only (does not invent marker data)."""
    lh = left_hip.copy()
    rh = right_hip.copy()
    sh_ok = _finite_rows(left_shoulder) & _finite_rows(right_shoulder)
    only_l = _finite_rows(lh) & ~_finite_rows(rh) & sh_ok
    only_r = _finite_rows(rh) & ~_finite_rows(lh) & sh_ok
    # Place missing hip using shoulder lateral direction + existing hip.
    lateral = right_shoulder - left_shoulder
    if np.any(only_l):
        rh[only_l] = lh[only_l] + lateral[only_l]
    if np.any(only_r):
        lh[only_r] = rh[only_r] - lateral[only_r]
    # If still missing, mirror the available hip (degenerate but finite).
    only_l2 = _finite_rows(lh) & ~_finite_rows(rh)
    only_r2 = _finite_rows(rh) & ~_finite_rows(lh)
    rh[only_l2] = lh[only_l2]
    lh[only_r2] = rh[only_r2]
    return lh, rh


def _identity_axes(n_frames: int) -> np.ndarray:
    axes = np.zeros((n_frames, 3, 3), dtype=float)
    axes[:, 0, 0] = 1.0
    axes[:, 1, 1] = 1.0
    axes[:, 2, 2] = 1.0
    return axes


def _shoulder_scale(
    left_hip: np.ndarray,
    right_hip: np.ndarray,
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
) -> float:
    shoulder_width = median_positive_length(np.linalg.norm(right_shoulder - left_shoulder, axis=1))
    if shoulder_width is not None:
        return float(shoulder_width)
    hip_width = median_positive_length(np.linalg.norm(right_hip - left_hip, axis=1))
    if hip_width is not None:
        return float(hip_width)
    return 1.0


def _nan_safe_translate(positions: np.ndarray, pelvis: np.ndarray) -> np.ndarray:
    """Subtract pelvis only on frames with a finite pelvis; never broadcast NaNs."""
    out = positions.copy()
    ok = _finite_rows(pelvis)
    out[ok] = positions[ok] - pelvis[ok][:, None, :]
    return out


def _carry_forward_axes(axes: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace invalid-axis frames with the previous valid axes (or identity)."""
    out = axes.copy()
    last = np.eye(3, dtype=float)
    for i in range(out.shape[0]):
        if valid[i] and np.isfinite(out[i]).all() and abs(np.linalg.det(out[i])) > 0.5:
            last = out[i]
        else:
            out[i] = last
    return out


def orient_common_positions(
    positions: np.ndarray,
    joint_names: tuple[str, ...],
) -> np.ndarray:
    """Stage 2: body-center, orient, and uniformly scale common positions.

    Preserves NaNs on individual missing markers. Does not mirror.
    Output last-axis layout before ACE remap: (right, up, forward).
    """
    name_to_i = {n: i for i, n in enumerate(joint_names)}
    for req in _ORIENT_JOINTS:
        if req not in name_to_i:
            raise ValueError(f"common positions missing joint {req}")

    left_hip = positions[:, name_to_i["left_hip"], :]
    right_hip = positions[:, name_to_i["right_hip"], :]
    left_shoulder = positions[:, name_to_i["left_shoulder"], :]
    right_shoulder = positions[:, name_to_i["right_shoulder"], :]

    pelvis = _estimate_pelvis(left_hip, right_hip)
    # If pelvis still NaN, fall back to nanmean of finite joints in that frame.
    still_bad = ~_finite_rows(pelvis)
    if np.any(still_bad):
        for i in np.flatnonzero(still_bad):
            pts = positions[i]
            finite = pts[np.isfinite(pts).all(axis=1)]
            if finite.size:
                pelvis[i] = np.mean(finite, axis=0)

    translated = _nan_safe_translate(positions, pelvis)

    lh_i, rh_i = _impute_hip_pair(left_hip, right_hip, left_shoulder, right_shoulder)
    # Re-estimate pelvis-relative hips after translate for axis math on translated coords
    lh_t = translated[:, name_to_i["left_hip"], :]
    rh_t = translated[:, name_to_i["right_hip"], :]
    ls_t = translated[:, name_to_i["left_shoulder"], :]
    rs_t = translated[:, name_to_i["right_shoulder"], :]
    lh_u, rh_u = _impute_hip_pair(lh_t, rh_t, ls_t, rs_t)

    axes_valid = (
        _finite_rows(lh_u)
        & _finite_rows(rh_u)
        & _finite_rows(ls_t)
        & _finite_rows(rs_t)
    )
    axes = _identity_axes(positions.shape[0])
    if np.any(axes_valid):
        # Compute axes only on valid frames to avoid NaN pollution inside cross products
        lh_c = np.where(axes_valid[:, None], lh_u, 0.0)
        rh_c = np.where(axes_valid[:, None], rh_u, np.array([1.0, 0.0, 0.0]))
        ls_c = np.where(axes_valid[:, None], ls_t, np.array([0.0, 0.0, 1.0]))
        rs_c = np.where(axes_valid[:, None], rs_t, np.array([1.0, 0.0, 1.0]))
        computed = orthonormal_body_axes(lh_c, rh_c, ls_c, rs_c)
        axes[axes_valid] = computed[axes_valid]
    axes = _carry_forward_axes(axes, axes_valid)

    body = rotate_positions(translated, axes)

    # Scale from finite shoulder widths only (ignore NaN frames)
    scale = _shoulder_scale(left_hip, right_hip, left_shoulder, right_shoulder)
    if not np.isfinite(scale) or scale <= EPSILON:
        scale = 1.0
    body = scale_positions(body, scale)

    # Keep original NaNs: any joint that was NaN stays NaN after transform
    was_nan = ~np.isfinite(positions)
    body = body.copy()
    body[was_nan] = np.nan
    return body


def common_positions_to_ace_markers(
    positions: np.ndarray,
    joint_names: tuple[str, ...],
    frames: np.ndarray,
) -> dict:
    """Map body-frame (right, up, forward) → ACE TX/TY/TZ (right, forward, up)."""
    display = np.empty_like(positions)
    display[:, :, 0] = positions[:, :, 0]  # TX = right
    display[:, :, 1] = positions[:, :, 2]  # TY = forward
    display[:, :, 2] = positions[:, :, 1]  # TZ = up

    out: dict = {"frames": np.asarray(frames).copy()}
    for i, name in enumerate(joint_names):
        out[name] = {
            "TX": display[:, i, 0].copy(),
            "TY": display[:, i, 1].copy(),
            "TZ": display[:, i, 2].copy(),
        }
    return out


def orient_ace_markers_for_display(markers: dict) -> dict:
    """Full display pipeline: common representation → orient → ACE marker dict.

    Source ``markers`` is never modified.
    """
    positions, names, frames = to_common_positions(markers)
    oriented = orient_common_positions(positions, names)
    out = common_positions_to_ace_markers(oriented, names, frames)
    for key, value in markers.items():
        if key == "frames" or key in out:
            continue
        out[key] = copy.deepcopy(value)
    return out


def pairwise_joint_distances(markers: dict, frame_idx: int) -> np.ndarray:
    """Upper-triangle pairwise distances among finite joints at ``frame_idx``."""
    names = [n for n in ACE_MARKER_NAMES if n in markers]
    pts = []
    for n in names:
        p = np.array(
            [
                float(markers[n]["TX"][frame_idx]),
                float(markers[n]["TY"][frame_idx]),
                float(markers[n]["TZ"][frame_idx]),
            ],
            dtype=float,
        )
        if np.isfinite(p).all():
            pts.append(p)
    if len(pts) < 2:
        return np.array([], dtype=float)
    pts_a = np.asarray(pts, dtype=float)
    diff = pts_a[:, None, :] - pts_a[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(len(pts_a), k=1)
    return dist[iu]
