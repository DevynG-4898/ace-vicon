from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Legacy (8-phase) public constants — unchanged for visualization consumers
PHASE_NAMES = (
    "Start_Stance",
    "Release",
    "Loading",
    "Cocking",
    "Acceleration",
    "Contact",
    "Deceleration",
    "Finish",
)

EVENT_NAMES = (
    "first_movement",
    "peak_hand_height",
    "maximum_knee_bend",
    "maximum_shoulder_external_rotation",
    "peak_velocity",
    "sustained_velocity_decrease",
    "stabilization",
)

EVENT_LABELS = {
    "first_movement": "First Movement",
    "peak_hand_height": "Peak Hand Height",
    "maximum_knee_bend": "Maximum Knee Bend",
    "maximum_shoulder_external_rotation": "Maximum Shoulder Rotation",
    "peak_velocity": "Peak Velocity (Contact Proxy)",
    "sustained_velocity_decrease": "Sustained Velocity Decrease",
    "stabilization": "Stabilization",
}

VIEW_OPTIONS = ("Full Serve",) + PHASE_NAMES

PHASE_COLORS = {
    "Start_Stance": "#4C78A8",
    "Release": "#72B7B2",
    "Loading": "#54A24B",
    "Cocking": "#EECA3B",
    "Acceleration": "#F58518",
    "Contact": "#E45756",
    "Deceleration": "#B279A2",
    "Finish": "#9D755D",
}

# V2 biomechanics-first model (6 anchors, 7 phases)
V2_PHASE_NAMES = (
    "Start_Stance",
    "Release",
    "Loading",
    "Cocking",
    "Acceleration",
    "Contact",
    "Deceleration_Finish",
)

V2_EVENT_NAMES = (
    "first_movement",
    "toss_apex",
    "maximum_knee_bend",
    "maximum_shoulder_external_rotation",
    "contact",
    "finish",
)

V2_EVENT_LABELS = {
    "first_movement": "First Movement",
    "toss_apex": "Toss Apex",
    "maximum_knee_bend": "Maximum Knee Bend",
    "maximum_shoulder_external_rotation": "Maximum Shoulder ER",
    "contact": "Contact",
    "finish": "Finish",
}

# Coaching metric annotations (not phase boundaries)
COACHING_ANNOTATION_NAMES = (
    "toss_apex",
    "toss_release",
    "max_knee_bend",
    "racket_apex",
    "max_shoulder_ER",
    "contact",
    "finish",
)

COACHING_TRANSITION_NAMES = (
    "release_start",
    "loading_start",
    "cocking_start",
    "acceleration_start",
)


@dataclass
class SegmentationResult:
    phases: dict[str, tuple[int, int]]
    events: dict[str, int]
    event_confidence: dict[str, float]
    signals: dict[str, np.ndarray]
    frames: np.ndarray
    event_indices: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    annotations: dict[str, int | str | float | bool] = field(default_factory=dict)
    schema_version: int = 1


def vicon_frame_to_index(frames: np.ndarray, vicon_frame: int) -> int:
    f = frames.astype(int)
    return int(np.clip(np.searchsorted(f, int(vicon_frame), side="left"), 0, len(f) - 1))


def phase_to_index_range(
    frames: np.ndarray, phase_bounds: tuple[int, int]
) -> tuple[int, int]:
    start_v, end_v = phase_bounds
    f = frames.astype(int)
    i0 = int(np.searchsorted(f, int(start_v), side="left"))
    i1 = int(np.searchsorted(f, int(end_v), side="right") - 1)
    i0 = int(np.clip(i0, 0, len(f) - 1))
    i1 = int(np.clip(i1, i0, len(f) - 1))
    return i0, i1


def view_index_range(
    frames: np.ndarray,
    phases: dict[str, tuple[int, int]],
    view_name: str,
) -> tuple[int, int]:
    if view_name == "Full Serve":
        return 0, len(frames) - 1
    if view_name not in phases:
        raise ValueError(f"Unknown view: {view_name}")
    return phase_to_index_range(frames, phases[view_name])


def phase_at_index(
    frames: np.ndarray, phases: dict[str, tuple[int, int]], frame_idx: int
) -> str:
    vicon = int(frames[frame_idx])
    for name in phases:
        start_v, end_v = phases[name]
        if start_v <= vicon <= end_v:
            return name
    return list(phases.keys())[-1]
