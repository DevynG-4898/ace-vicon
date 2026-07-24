"""Tests for Coaching Viewer display-only orientation (Vicon + MotionBERT)."""

from __future__ import annotations

import numpy as np

from src.coaching_viewer.display_orient import (
    orient_ace_markers_for_display,
    pairwise_joint_distances,
    to_common_positions,
)
from src.markers.io import ACE_MARKER_NAMES, load_serve_markers


def _synthetic_markers(*, yaw_deg: float = 0.0, scale: float = 1.0, n: int = 3) -> dict:
    """Simple standing pose in ACE TX/TY/TZ (Z up), optionally yawed about Z."""
    base = {
        "head": (0.0, 0.1, 1.7),
        "chest": (0.0, 0.05, 1.2),
        "left_shoulder": (-0.2, 0.05, 1.25),
        "right_shoulder": (0.2, 0.05, 1.25),
        "left_elbow": (-0.35, 0.05, 1.0),
        "right_elbow": (0.35, 0.05, 1.0),
        "left_hand": (-0.4, 0.05, 0.75),
        "right_hand": (0.4, 0.05, 0.75),
        "left_hip": (-0.1, 0.0, 0.9),
        "right_hip": (0.1, 0.0, 0.9),
        "left_knee": (-0.1, 0.0, 0.45),
        "right_knee": (0.1, 0.0, 0.45),
        "left_foot": (-0.1, 0.05, 0.0),
        "right_foot": (0.1, 0.05, 0.0),
    }
    theta = np.deg2rad(yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    frames = np.arange(1, n + 1)
    markers: dict = {"frames": frames}
    for name in ACE_MARKER_NAMES:
        p = np.asarray(base[name], dtype=float) * scale
        world = (rot @ p) + np.array([10.0, -5.0, 2.0])
        markers[name] = {
            "TX": np.full(n, world[0]),
            "TY": np.full(n, world[1]),
            "TZ": np.full(n, world[2]),
        }
    return markers


def test_does_not_mutate_source():
    src = _synthetic_markers(yaw_deg=40.0, scale=2.0)
    tx0 = src["head"]["TX"].copy()
    orient_ace_markers_for_display(src)
    assert np.array_equal(src["head"]["TX"], tx0)


def test_pelvis_at_origin():
    oriented = orient_ace_markers_for_display(_synthetic_markers(yaw_deg=55.0, scale=3.0))
    for i in range(len(oriented["frames"])):
        pelvis = 0.5 * (
            np.array(
                [
                    oriented["left_hip"]["TX"][i],
                    oriented["left_hip"]["TY"][i],
                    oriented["left_hip"]["TZ"][i],
                ]
            )
            + np.array(
                [
                    oriented["right_hip"]["TX"][i],
                    oriented["right_hip"]["TY"][i],
                    oriented["right_hip"]["TZ"][i],
                ]
            )
        )
        assert np.allclose(pelvis, 0.0, atol=1e-8)


def test_yawed_poses_align():
    a = orient_ace_markers_for_display(_synthetic_markers(yaw_deg=0.0))
    b = orient_ace_markers_for_display(_synthetic_markers(yaw_deg=73.0))
    for name in ACE_MARKER_NAMES:
        for axis in ("TX", "TY", "TZ"):
            assert np.allclose(a[name][axis], b[name][axis], atol=1e-6)


def test_left_right_not_mirrored():
    oriented = orient_ace_markers_for_display(_synthetic_markers(yaw_deg=90.0))
    for i in range(len(oriented["frames"])):
        assert oriented["right_shoulder"]["TX"][i] > oriented["left_shoulder"]["TX"][i]
        assert oriented["right_hip"]["TX"][i] > oriented["left_hip"]["TX"][i]


def test_pairwise_distances_scale_invariantly():
    src = _synthetic_markers(yaw_deg=20.0, scale=2.5)
    oriented = orient_ace_markers_for_display(src)
    d0 = pairwise_joint_distances(src, 0)
    d1 = pairwise_joint_distances(oriented, 0)
    ratio = d1 / np.maximum(d0, 1e-12)
    assert np.allclose(ratio, ratio[0], atol=1e-5)


def test_missing_hip_does_not_poison_frame():
    """Regression: one NaN Vicon hip must not wipe all other joints."""
    src = _synthetic_markers(n=2)
    for ax in ("TX", "TY", "TZ"):
        src["right_hip"][ax][0] = np.nan
    oriented = orient_ace_markers_for_display(src)
    # right_hip stays NaN
    assert not np.isfinite(oriented["right_hip"]["TX"][0])
    # other joints remain drawable
    for name in ("head", "left_hip", "left_shoulder", "right_shoulder"):
        assert np.isfinite(oriented[name]["TX"][0])
        assert np.isfinite(oriented[name]["TY"][0])
        assert np.isfinite(oriented[name]["TZ"][0])


def test_vicon_secondserve_early_frames_renderable():
    markers = load_serve_markers("plotting/markers/individual/secondserve")
    oriented = orient_ace_markers_for_display(markers)
    # Frame 0 has NaN right_hip in source but 13 other joints — must stay drawable
    finite = sum(
        1
        for name in ACE_MARKER_NAMES
        if all(np.isfinite(oriented[name][ax][0]) for ax in ("TX", "TY", "TZ"))
    )
    assert finite >= 12


def test_vicon_and_motionbert_both_orient():
    vicon = orient_ace_markers_for_display(
        load_serve_markers("plotting/markers/individual/firstserve")
    )
    mb = orient_ace_markers_for_display(load_serve_markers("generated_motionbert/andy"))
    for markers in (vicon, mb):
        # pelvis ~ origin on a fully finite mid frame
        n = len(markers["frames"])
        for i in range(n):
            if not all(
                np.isfinite(markers[j][ax][i])
                for j in ("left_hip", "right_hip")
                for ax in ("TX", "TY", "TZ")
            ):
                continue
            pelvis = 0.5 * (
                np.array([markers["left_hip"][a][i] for a in ("TX", "TY", "TZ")])
                + np.array([markers["right_hip"][a][i] for a in ("TX", "TY", "TZ")])
            )
            assert np.allclose(pelvis, 0.0, atol=1e-5)
            break


def test_stage1_common_positions_shape():
    markers = _synthetic_markers()
    pos, names, frames = to_common_positions(markers)
    assert pos.shape == (3, 14, 3)
    assert len(names) == 14
    assert len(frames) == 3
