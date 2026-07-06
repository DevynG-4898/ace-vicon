from __future__ import annotations

import numpy as np

from src.skeleton import SkeletonSequence

EPSILON = 1e-9
Y_AXIS = np.array([0.0, 1.0, 0.0], dtype=float)
NEG_Y_AXIS = np.array([0.0, -1.0, 0.0], dtype=float)


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = np.maximum(a_norm * b_norm, EPSILON)
    cosang = np.clip(np.sum(a * b, axis=1) / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def joint_angle(sequence: SkeletonSequence, proximal: str, joint: str, distal: str) -> np.ndarray:
    """Return the interior angle at `joint` in degrees."""
    center = sequence.joint(joint)
    return _angle_between(sequence.joint(proximal) - center, sequence.joint(distal) - center)


def _project(values: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    out = np.zeros_like(values)
    out[:, axes[0]] = values[:, axes[0]]
    out[:, axes[1]] = values[:, axes[1]]
    return out


def _angle_to_axis(values: np.ndarray, axis: np.ndarray) -> np.ndarray:
    target = np.repeat(axis[None, :], values.shape[0], axis=0)
    good = np.linalg.norm(values, axis=1) > EPSILON
    result = np.zeros(values.shape[0], dtype=float)
    if np.any(good):
        result[good] = _angle_between(values[good], target[good])
    return result


def shoulder_flexion(sequence: SkeletonSequence, side: str) -> np.ndarray:
    """Shoulder flexion from the upper arm projected onto the sagittal Y/Z plane."""
    upper_arm = sequence.joint(f"{side}_elbow") - sequence.joint(f"{side}_shoulder")
    return _angle_to_axis(_project(upper_arm, (1, 2)), NEG_Y_AXIS)


def shoulder_abduction(sequence: SkeletonSequence, side: str) -> np.ndarray:
    """Shoulder abduction from the upper arm projected onto the frontal X/Y plane."""
    upper_arm = sequence.joint(f"{side}_elbow") - sequence.joint(f"{side}_shoulder")
    return _angle_to_axis(_project(upper_arm, (0, 1)), NEG_Y_AXIS)


def line_angle_xy(sequence: SkeletonSequence, left: str, right: str) -> np.ndarray:
    """Angle of a left-to-right body line in the X/Y plane."""
    line = sequence.joint(right) - sequence.joint(left)
    return np.degrees(np.arctan2(line[:, 1], line[:, 0]))


def trunk_tilt(sequence: SkeletonSequence) -> np.ndarray:
    """Angle between pelvis-to-thorax spine vector and canonical vertical."""
    trunk = sequence.joint("thorax") - sequence.joint("pelvis")
    return _angle_to_axis(trunk, Y_AXIS)


def trunk_rotation(sequence: SkeletonSequence) -> np.ndarray:
    """Signed shoulder rotation relative to pelvis line around the vertical axis."""
    hips = _project(sequence.joint("right_hip") - sequence.joint("left_hip"), (0, 2))
    shoulders = _project(sequence.joint("right_shoulder") - sequence.joint("left_shoulder"), (0, 2))
    cross_y = hips[:, 2] * shoulders[:, 0] - hips[:, 0] * shoulders[:, 2]
    dot = np.sum(hips * shoulders, axis=1)
    return np.degrees(np.arctan2(cross_y, dot))


def hip_flexion(sequence: SkeletonSequence) -> np.ndarray:
    """Mean left/right thigh flexion from canonical downward vertical."""
    values = []
    for side in ("left", "right"):
        thigh = sequence.joint(f"{side}_knee") - sequence.joint(f"{side}_hip")
        values.append(_angle_to_axis(_project(thigh, (1, 2)), NEG_Y_AXIS))
    return np.mean(np.stack(values, axis=0), axis=0)


def hip_abduction(sequence: SkeletonSequence) -> np.ndarray:
    """Mean left/right thigh abduction from canonical downward vertical."""
    values = []
    for side in ("left", "right"):
        thigh = sequence.joint(f"{side}_knee") - sequence.joint(f"{side}_hip")
        values.append(_angle_to_axis(_project(thigh, (0, 1)), NEG_Y_AXIS))
    return np.mean(np.stack(values, axis=0), axis=0)


def compute_joint_angle_features(sequence: SkeletonSequence) -> dict[str, np.ndarray]:
    """Compute all canonical per-frame joint angle features in degrees."""
    return {
        "left_elbow_angle": joint_angle(sequence, "left_shoulder", "left_elbow", "left_wrist"),
        "right_elbow_angle": joint_angle(sequence, "right_shoulder", "right_elbow", "right_wrist"),
        "left_shoulder_flexion": shoulder_flexion(sequence, "left"),
        "right_shoulder_flexion": shoulder_flexion(sequence, "right"),
        "left_shoulder_abduction": shoulder_abduction(sequence, "left"),
        "right_shoulder_abduction": shoulder_abduction(sequence, "right"),
        "shoulder_line_angle": line_angle_xy(sequence, "left_shoulder", "right_shoulder"),
        "trunk_tilt": trunk_tilt(sequence),
        "trunk_rotation": trunk_rotation(sequence),
        "left_knee_angle": joint_angle(sequence, "left_hip", "left_knee", "left_ankle"),
        "right_knee_angle": joint_angle(sequence, "right_hip", "right_knee", "right_ankle"),
        "hip_flexion": hip_flexion(sequence),
        "hip_abduction": hip_abduction(sequence),
        "spine_angle": trunk_tilt(sequence),
        "pelvis_tilt": line_angle_xy(sequence, "left_hip", "right_hip"),
    }

