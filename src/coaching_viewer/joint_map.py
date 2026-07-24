"""Static feature → ACE marker highlight map (no biomechanics)."""

from __future__ import annotations

_STATIC: dict[str, tuple[str, ...]] = {
    "Knee Flexion": ("left_knee", "right_knee"),
    "Hip Flexion": ("left_hip", "right_hip", "pelvis"),
    "Shoulder Tilt": ("left_shoulder", "right_shoulder", "chest"),
    "Center of Mass": ("pelvis", "chest"),
    "Trunk Rotation": ("chest", "left_shoulder", "right_shoulder", "pelvis"),
    "Pelvis Rotation": ("pelvis", "left_hip", "right_hip"),
    "Body Alignment": ("chest", "pelvis", "left_shoulder", "right_shoulder"),
    "Balance": ("pelvis", "left_foot", "right_foot"),
    "Weight Transfer": ("pelvis", "left_foot", "right_foot"),
    "Recovery Position": ("pelvis", "left_foot", "right_foot"),
}

_HITTING_CHAIN = ("{side}_shoulder", "{side}_elbow", "{side}_hand")
_SIDE_FEATURES: dict[str, str] = {
    "Toss Arm Extension": "toss",
    "Right Elbow Flexion": "hit",
    "Left Elbow Flexion": "toss",
    "Shoulder External Rotation": "hit",
    "Forearm Angle": "hit",
    "Shoulder Internal Rotation": "hit",
    "Right Elbow Extension": "hit",
    "Left Elbow Extension": "toss",
    "Trunk Rotation Velocity": "hit",
    "Hip Rotation Velocity": "hit",
    "Contact Height": "hit",
    "Contact Position": "hit",
    "Arm Extension": "hit",
    "Follow Through": "hit",
    "Shoulder Deceleration": "hit",
    "Trunk Flexion": "hit",
}


def _sides(handedness: str) -> tuple[str, str]:
    """Return (hitting_side, toss_side)."""
    if handedness == "left":
        return "left", "right"
    return "right", "left"


def highlight_joints(feature: str, *, handedness: str = "right") -> tuple[str, ...]:
    """Return ACE marker names to emphasize for ``feature``."""
    if feature in _STATIC:
        return _STATIC[feature]

    role = _SIDE_FEATURES.get(feature)
    if role is None:
        return ()

    hit, toss = _sides(handedness)
    side = hit if role == "hit" else toss
    if feature == "Toss Arm Extension":
        return tuple(p.format(side=side) for p in _HITTING_CHAIN)
    if feature in ("Trunk Rotation Velocity", "Hip Rotation Velocity", "Trunk Flexion"):
        return ("chest", "pelvis", f"{side}_shoulder", f"{side}_hip")
    return tuple(p.format(side=side) for p in _HITTING_CHAIN)


def resolve_highlight_markers(
    symbolic: tuple[str, ...],
    markers: dict,
) -> tuple[str, ...]:
    """Expand symbolic names (e.g. pelvis) to markers present in ``markers``."""
    out: list[str] = []
    for name in symbolic:
        if name == "pelvis":
            for hip in ("left_hip", "right_hip"):
                if hip in markers and hip not in out:
                    out.append(hip)
            continue
        if name in markers and name not in out:
            out.append(name)
    return tuple(out)
