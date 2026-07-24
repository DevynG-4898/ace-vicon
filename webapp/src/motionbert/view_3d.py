from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.motionbert.motionbert_runner import MOTIONBERT_JOINT_NAMES, SKELETON_EDGES
from src.skeleton.io import NORMALIZED_SKELETON_FILENAME, load_skeleton_sequence


def _load_debug(path: Path) -> dict[str, Any]:
    debug_path = path.with_suffix(".json")
    if debug_path.is_file():
        return json.loads(debug_path.read_text(encoding="utf-8"))
    return {"joint_names": MOTIONBERT_JOINT_NAMES, "skeleton_edges": SKELETON_EDGES}


def _limits(values: np.ndarray, pad: float = 0.08) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if lo == hi:
        return lo - 0.5, hi + 0.5
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


def _display_pose_path(poses_3d_path: str | Path, *, normalized: bool = False) -> Path:
    """Return the raw or normalized display file for a MotionBERT output."""
    path = Path(poses_3d_path)
    if normalized:
        return path.parent / NORMALIZED_SKELETON_FILENAME
    return path


def _load_pose_data(path: Path) -> tuple[np.ndarray, list[str], list[tuple[str, str]], str]:
    if path.suffix == ".npz":
        sequence = load_skeleton_sequence(path)
        return (
            sequence.joint_positions,
            list(sequence.joint_names),
            SKELETON_EDGES,
            sequence.coordinate_system,
        )

    pose = np.load(path)
    if pose.ndim != 3 or pose.shape[2] != 3:
        raise ValueError(f"poses_3d must have shape (frames, joints, 3); got {pose.shape}")
    debug = _load_debug(path)
    return (
        pose,
        list(debug.get("joint_names", MOTIONBERT_JOINT_NAMES)),
        debug.get("skeleton_edges", SKELETON_EDGES),
        "raw_motionbert",
    )


def run_viewer(poses_3d_path: str | Path, *, speed: int = 1, normalized: bool = False) -> None:
    path = Path(poses_3d_path)
    display_path = _display_pose_path(path, normalized=normalized)
    if not display_path.is_file():
        raise FileNotFoundError(f"Skeleton display file not found: {display_path}")

    pose, joint_names, skeleton_edges, coordinate_system = _load_pose_data(display_path)
    joint_index = {name: idx for idx, name in enumerate(joint_names)}
    interval_ms = max(1, int(33 / max(speed, 1)))

    x_lim = _limits(pose[:, :, 0])
    y_lim = _limits(pose[:, :, 1])
    z_lim = _limits(pose[:, :, 2])

    state = {
        "paused": False,
        "frame": 0,
        "normalized": normalized,
        "pose": pose,
        "joint_names": joint_names,
        "skeleton_edges": skeleton_edges,
        "joint_index": joint_index,
        "coordinate_system": coordinate_system,
        "x_lim": x_lim,
        "y_lim": y_lim,
        "z_lim": z_lim,
    }
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    def load_mode(use_normalized: bool) -> None:
        next_path = _display_pose_path(path, normalized=use_normalized)
        if not next_path.is_file():
            return
        next_pose, next_names, next_edges, next_coordinate_system = _load_pose_data(next_path)
        state["normalized"] = use_normalized
        state["pose"] = next_pose
        state["joint_names"] = next_names
        state["skeleton_edges"] = next_edges
        state["joint_index"] = {name: idx for idx, name in enumerate(next_names)}
        state["coordinate_system"] = next_coordinate_system
        state["x_lim"] = _limits(next_pose[:, :, 0])
        state["y_lim"] = _limits(next_pose[:, :, 1])
        state["z_lim"] = _limits(next_pose[:, :, 2])
        state["frame"] = min(state["frame"], next_pose.shape[0] - 1)

    def draw(frame_idx: int) -> None:
        pose_data = state["pose"]
        x_lim_state = state["x_lim"]
        y_lim_state = state["y_lim"]
        z_lim_state = state["z_lim"]
        ax.cla()
        ax.set_xlim(*x_lim_state)
        ax.set_ylim(*y_lim_state)
        ax.set_zlim(*z_lim_state)
        ax.set_box_aspect(
            [
                x_lim_state[1] - x_lim_state[0],
                y_lim_state[1] - y_lim_state[0],
                z_lim_state[1] - z_lim_state[0],
            ]
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        mode = "normalized" if state["normalized"] else "raw"
        ax.set_title(
            f"Frame {frame_idx + 1}/{pose_data.shape[0]} | Joints: {pose_data.shape[1]} | "
            f"{mode} | {state['coordinate_system']}"
        )

        points = pose_data[frame_idx]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=24, color="steelblue")
        current_joint_index = state["joint_index"]
        for start, end in state["skeleton_edges"]:
            if start not in current_joint_index or end not in current_joint_index:
                continue
            a = points[current_joint_index[start]]
            b = points[current_joint_index[end]]
            if np.isfinite(a).all() and np.isfinite(b).all():
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="black", linewidth=1.5)

        ax.text2D(
            0.02,
            0.02,
            "Space: pause/resume | Left/Right: step | N: raw/normalized",
            transform=ax.transAxes,
            fontsize=9,
        )

    def update(_: int) -> None:
        if not state["paused"]:
            state["frame"] = (state["frame"] + 1) % max(state["pose"].shape[0], 1)
        draw(state["frame"])

    def on_key(event) -> None:
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "right":
            state["paused"] = True
            state["frame"] = min(state["frame"] + 1, state["pose"].shape[0] - 1)
            draw(state["frame"])
            fig.canvas.draw_idle()
        elif event.key == "left":
            state["paused"] = True
            state["frame"] = max(state["frame"] - 1, 0)
            draw(state["frame"])
            fig.canvas.draw_idle()
        elif event.key in ("n", "N"):
            state["paused"] = True
            load_mode(not bool(state["normalized"]))
            draw(state["frame"])
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    animation.FuncAnimation(fig, update, frames=max(state["pose"].shape[0], 1), interval=interval_ms, repeat=True)
    draw(0)
    plt.show()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="View standalone MotionBERT-style 3D skeleton output.")
    parser.add_argument("poses_3d", help="Path to generated poses_3d.npy")
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--normalized", action="store_true", help="Display skeleton_normalized.npz beside poses_3d.npy.")
    args = parser.parse_args()
    run_viewer(args.poses_3d, speed=args.speed, normalized=args.normalized)


if __name__ == "__main__":
    main()
