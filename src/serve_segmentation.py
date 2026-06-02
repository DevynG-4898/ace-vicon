"""
Deterministic serve phase segmentation for ACE Vicon 14-marker captures.

Detects ordered biomechanical events from smoothed kinematic signals, enforces
monotonic event order, and maps events to eight serve phases.
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow imports from dtw/ when run as script or from src/
_DTW_DIR = Path(__file__).resolve().parent.parent / "dtw"
if str(_DTW_DIR) not in sys.path:
    sys.path.insert(0, str(_DTW_DIR))

from constants import MARKER_ORDER  # noqa: E402
from load_data import FILENAME_TO_MARKER, load_single_serve  # noqa: E402

_AXES = ("TX", "TY", "TZ")

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


@dataclass
class SegmentationConfig:
    """Tunable thresholds — all detection uses smoothed signals."""

    smooth_window: int = 11
    min_phase_frames: int = 8
    min_event_gap_frames: int = 5

    # Event 1 — first movement (body velocity)
    baseline_frames: int = 40
    body_velocity_threshold_ratio: float = 0.12
    body_velocity_persist_frames: int = 6

    # Event 6 — sustained decrease after contact
    post_contact_velocity_fraction: float = 0.55
    velocity_decrease_persist_frames: int = 10

    # Event 7 — stabilization
    stabilization_velocity_ratio: float = 0.15
    stabilization_persist_frames: int = 20

    serving_hand: str = "right_hand"
    serving_side_knee: str = "right_knee"
    serving_side_hip: str = "right_hip"
    serving_side_foot: str = "right_foot"
    serving_shoulder: str = "right_shoulder"
    serving_elbow: str = "right_elbow"


@dataclass
class SegmentationResult:
    phases: dict[str, tuple[int, int]]
    events: dict[str, int]
    event_confidence: dict[str, float]
    signals: dict[str, np.ndarray]
    frames: np.ndarray
    event_indices: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def vicon_frame_to_index(frames: np.ndarray, vicon_frame: int) -> int:
    """Map a Vicon frame number to the closest row index in the capture."""
    f = frames.astype(int)
    return int(np.clip(np.searchsorted(f, int(vicon_frame), side="left"), 0, len(f) - 1))


def phase_to_index_range(
    frames: np.ndarray, phase_bounds: tuple[int, int]
) -> tuple[int, int]:
    """Convert Vicon (start, end) phase bounds to inclusive array index range."""
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
    """Index range for 'Full Serve' or a named phase."""
    if view_name == "Full Serve":
        return 0, len(frames) - 1
    if view_name not in phases:
        raise ValueError(f"Unknown view: {view_name}")
    return phase_to_index_range(frames, phases[view_name])


def phase_at_index(
    frames: np.ndarray, phases: dict[str, tuple[int, int]], frame_idx: int
) -> str:
    """Return the phase name containing this array index (Vicon frame lookup)."""
    vicon = int(frames[frame_idx])
    for name in PHASE_NAMES:
        start_v, end_v = phases[name]
        if start_v <= vicon <= end_v:
            return name
    return PHASE_NAMES[-1]


def _marker_names(serve: dict) -> list[str]:
    return [k for k in serve if k != "frames"]


def _position(serve: dict, marker: str) -> np.ndarray:
    m = serve[marker]
    return np.column_stack(
        [
            m["TX"].astype(float),
            m["TY"].astype(float),
            m["TZ"].astype(float),
        ]
    )


def _smooth(series: np.ndarray, window: int) -> np.ndarray:
    w = max(3, window | 1)  # odd window
    filled = pd.Series(series, dtype=float).interpolate(limit_direction="both").bfill().ffill()
    return filled.rolling(window=w, center=True, min_periods=1).mean().values


def _clip_index(idx: int, n: int) -> int:
    return int(min(max(idx, 0), max(n - 1, 0)))


def _speed(pos: np.ndarray) -> np.ndarray:
    """Per-frame speed aligned to frame index (length n-1, pad last)."""
    d = np.diff(pos, axis=0)
    sp = np.linalg.norm(d, axis=1)
    return np.concatenate([sp, [sp[-1] if len(sp) else 0.0]])


def _angle_at_joint(hip: np.ndarray, knee: np.ndarray, foot: np.ndarray) -> np.ndarray:
    """Interior angle at knee (degrees). Lower angle = more flexion."""
    v1 = hip - knee
    v2 = foot - knee
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = np.maximum(n1 * n2, 1e-9)
    cosang = np.sum(v1 * v2, axis=1) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def _shoulder_external_rotation_proxy(
    shoulder: np.ndarray, elbow: np.ndarray, chest: np.ndarray
) -> np.ndarray:
    """
  Proxy for shoulder external rotation: angle between upper arm and trunk vectors.
  Higher values indicate more arm laid back (cocked).
    """
    upper_arm = elbow - shoulder
    trunk = chest - shoulder
    n1 = np.linalg.norm(upper_arm, axis=1)
    n2 = np.linalg.norm(trunk, axis=1)
    denom = np.maximum(n1 * n2, 1e-9)
    cosang = np.sum(upper_arm * trunk, axis=1) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def _body_velocity(serve: dict, cfg: SegmentationConfig) -> np.ndarray:
    speeds = []
    for name in _marker_names(serve):
        if name in MARKER_ORDER:
            speeds.append(_speed(_position(serve, name)))
    return np.nanmean(np.stack(speeds, axis=0), axis=0)


def _compute_signals(serve: dict, cfg: SegmentationConfig) -> dict[str, np.ndarray]:
    hand_pos = _position(serve, cfg.serving_hand)
    hand_tz = hand_pos[:, 2]

    knee_angles = []
    for hip, knee, foot in (
        ("right_hip", "right_knee", "right_foot"),
        ("left_hip", "left_knee", "left_foot"),
    ):
        if all(m in serve for m in (hip, knee, foot)):
            knee_angles.append(
                _angle_at_joint(
                    _position(serve, hip),
                    _position(serve, knee),
                    _position(serve, foot),
                )
            )
    knee_flexion = np.min(np.stack(knee_angles, axis=0), axis=0) if knee_angles else None

    ser_proxy = _shoulder_external_rotation_proxy(
        _position(serve, cfg.serving_shoulder),
        _position(serve, cfg.serving_elbow),
        _position(serve, "chest"),
    )

    body_v = _body_velocity(serve, cfg)
    hand_v = _speed(hand_pos)

    w = cfg.smooth_window
    return {
        "hand_tz": _smooth(hand_tz, w),
        "knee_flexion_deg": _smooth(knee_flexion, w) if knee_flexion is not None else None,
        "shoulder_er_proxy_deg": _smooth(ser_proxy, w),
        "body_velocity": _smooth(body_v, w),
        "hand_velocity": _smooth(hand_v, w),
    }


def _persist_above(series: np.ndarray, threshold: float, persist: int, start: int = 0) -> int | None:
    run = 0
    for i in range(start, len(series)):
        if series[i] > threshold:
            run += 1
            if run >= persist:
                return i - persist + 1
        else:
            run = 0
    return None


def _persist_below(series: np.ndarray, threshold: float, persist: int, start: int = 0) -> int | None:
    run = 0
    for i in range(start, len(series)):
        if series[i] < threshold:
            run += 1
            if run >= persist:
                return i - persist + 1
        else:
            run = 0
    return None


def _argmax_in_range(series: np.ndarray, low: int, high: int, mode: str = "max") -> int:
    low = max(0, low)
    high = min(high, len(series) - 1)
    if high < low:
        return low
    segment = np.asarray(series[low : high + 1], dtype=float)
    if np.all(np.isnan(segment)):
        return low + len(segment) // 2
    if mode == "max":
        return low + int(np.nanargmax(segment))
    return low + int(np.nanargmin(segment))


def _detect_events(signals: dict[str, np.ndarray], cfg: SegmentationConfig) -> dict[str, int]:
    n = len(signals["body_velocity"])
    body_v = signals["body_velocity"]
    hand_v = signals["hand_velocity"]
    hand_tz = signals["hand_tz"]
    knee = signals["knee_flexion_deg"]
    ser = signals["shoulder_er_proxy_deg"]

    baseline_n = min(cfg.baseline_frames, max(10, n // 8))
    baseline = float(np.nanmedian(body_v[:baseline_n]))
    peak_body = float(np.nanmax(body_v))
    thresh_move = baseline + cfg.body_velocity_threshold_ratio * max(peak_body - baseline, 1e-6)

    e1 = _persist_above(body_v, thresh_move, cfg.body_velocity_persist_frames, start=0)
    if e1 is None:
        e1 = int(np.nanargmax(body_v) * 0.15)  # fallback: early motion

    # Event 2 — peak hand height (TZ) after first movement
    search_end = min(n - 1, e1 + int(n * 0.65))
    e2 = _argmax_in_range(hand_tz, e1, search_end, mode="max")

    # Event 3 — maximum knee bend (minimum knee angle)
    if knee is not None:
        e3 = _argmax_in_range(knee, e2, min(n - 1, e2 + int(n * 0.5)), mode="min")
    else:
        e3 = e2 + cfg.min_event_gap_frames

    # Event 4 — max shoulder ER proxy after knee bend
    e4 = _argmax_in_range(ser, e3, min(n - 1, e3 + int(n * 0.45)), mode="max")

    # Event 5 — peak hand velocity (contact proxy)
    e5 = _argmax_in_range(hand_v, e4, n - 1, mode="max")

    peak_v = float(hand_v[e5])
    decel_thresh = peak_v * cfg.post_contact_velocity_fraction
    e6 = _persist_below(
        hand_v,
        decel_thresh,
        cfg.velocity_decrease_persist_frames,
        start=min(e5 + 1, n - 1),
    )
    if e6 is None:
        e6 = min(e5 + cfg.min_event_gap_frames, n - 1)

    stab_thresh = baseline + cfg.stabilization_velocity_ratio * max(peak_body - baseline, 1e-6)
    e7 = _persist_below(
        body_v,
        stab_thresh,
        cfg.stabilization_persist_frames,
        start=min(e6 + 1, n - 1),
    )
    if e7 is None:
        e7 = min(n - 1, e6 + cfg.stabilization_persist_frames)

    return {
        "first_movement": e1,
        "peak_hand_height": e2,
        "maximum_knee_bend": e3,
        "maximum_shoulder_external_rotation": e4,
        "peak_velocity": e5,
        "sustained_velocity_decrease": e6,
        "stabilization": e7,
    }


def _enforce_order(events: dict[str, int], cfg: SegmentationConfig, n_frames: int) -> dict[str, int]:
    """Enforce strictly increasing event indices with minimum gaps, clamped to [0, n-1]."""
    keys = EVENT_NAMES
    ordered = [_clip_index(events[k], n_frames) for k in keys]
    gap = cfg.min_event_gap_frames
    for i in range(1, len(ordered)):
        ordered[i] = max(ordered[i], ordered[i - 1] + gap)
    if ordered[-1] >= n_frames:
        ordered[-1] = n_frames - 1
        for i in range(len(ordered) - 2, -1, -1):
            ordered[i] = min(ordered[i], ordered[i + 1] - gap)
        ordered[0] = max(0, ordered[0])
    return {k: _clip_index(ordered[i], n_frames) for i, k in enumerate(keys)}


def _indices_to_phases(
    events: dict[str, int], n: int, frame_ids: np.ndarray
) -> dict[str, tuple[int, int]]:
    e1 = events["first_movement"]
    e2 = events["peak_hand_height"]
    e3 = events["maximum_knee_bend"]
    e4 = events["maximum_shoulder_external_rotation"]
    e5 = events["peak_velocity"]
    e6 = events["sustained_velocity_decrease"]
    e7 = events["stabilization"]

    def f(i: int) -> int:
        return int(frame_ids[min(max(i, 0), n - 1)])

    phases_idx = {
        "Start_Stance": (0, max(0, e1 - 1)),
        "Release": (e1, max(e1, e2 - 1)),
        "Loading": (e2, max(e2, e3 - 1)),
        "Cocking": (e3, max(e3, e4 - 1)),
        "Acceleration": (e4, max(e4, e5 - 1)),
        "Contact": (e5, e5),
        "Deceleration": (min(e5 + 1, n - 1), max(e6, e7 - 1)),
        "Finish": (e7, n - 1),
    }

    return {name: (f(phases_idx[name][0]), f(phases_idx[name][1])) for name in PHASE_NAMES}


def _event_confidence(
    signals: dict[str, np.ndarray], events: dict[str, int], cfg: SegmentationConfig
) -> dict[str, float]:
    body_v = signals["body_velocity"]
    hand_v = signals["hand_velocity"]
    hand_tz = signals["hand_tz"]
    knee = signals["knee_flexion_deg"]
    ser = signals["shoulder_er_proxy_deg"]

    baseline_n = min(cfg.baseline_frames, max(10, len(body_v) // 8))
    baseline = float(np.nanmedian(body_v[:baseline_n]))
    peak_body = float(np.nanmax(body_v))

    def peak_prominence(series: np.ndarray, idx: int, invert: bool = False) -> float:
        val = series[idx]
        margin = np.nanmax(series) - np.nanmin(series)
        if margin < 1e-9:
            return 0.3
        if invert:
            return float((np.nanmax(series) - val) / margin)
        return float((val - np.nanmin(series)) / margin)

    n = len(body_v)
    e1 = _clip_index(events["first_movement"], n)
    move_strength = (body_v[e1] - baseline) / max(peak_body - baseline, 1e-9)

    return {
        "first_movement": float(np.clip(move_strength, 0, 1)),
        "peak_hand_height": peak_prominence(hand_tz, _clip_index(events["peak_hand_height"], n)),
        "maximum_knee_bend": (
            peak_prominence(knee, _clip_index(events["maximum_knee_bend"], len(knee)), invert=True)
            if knee is not None
            else 0.4
        ),
        "maximum_shoulder_external_rotation": peak_prominence(
            ser, _clip_index(events["maximum_shoulder_external_rotation"], n)
        ),
        "peak_velocity": peak_prominence(hand_v, _clip_index(events["peak_velocity"], n)),
        "sustained_velocity_decrease": float(
            np.clip(
                (
                    hand_v[_clip_index(events["peak_velocity"], n)]
                    - hand_v[_clip_index(events["sustained_velocity_decrease"], n)]
                )
                / max(hand_v[_clip_index(events["peak_velocity"], n)], 1e-9),
                0,
                1,
            )
        ),
        "stabilization": float(
            np.clip(
                1.0 - body_v[_clip_index(events["stabilization"], n)] / max(peak_body, 1e-9),
                0,
                1,
            )
        ),
    }


def _validate_phases(phases: dict[str, tuple[int, int]], warnings: list[str]) -> None:
    for name in PHASE_NAMES:
        a, b = phases[name]
        if a > b:
            warnings.append(f"{name}: invalid range ({a}, {b})")
    for i in range(len(PHASE_NAMES) - 1):
        cur = phases[PHASE_NAMES[i]]
        nxt = phases[PHASE_NAMES[i + 1]]
        if cur[1] >= nxt[0] and not (PHASE_NAMES[i] == "Contact"):
            warnings.append(
                f"overlap or disorder between {PHASE_NAMES[i]} and {PHASE_NAMES[i + 1]}"
            )


def segment_serve(
    serve: dict, config: SegmentationConfig | None = None
) -> SegmentationResult:
    """
    Segment a loaded serve dict into eight phases.

    Args:
        serve: output of load_single_serve / load_multi_serve
        config: optional SegmentationConfig

    Returns:
        SegmentationResult with phases keyed by phase name, values (start_frame, end_frame)
        using Vicon frame numbers from serve['frames'].
    """
    cfg = config or SegmentationConfig()
    warnings: list[str] = []

    required = {cfg.serving_hand, cfg.serving_shoulder, cfg.serving_elbow, "chest"}
    missing = [m for m in required if m not in serve]
    if missing:
        warnings.append(f"missing markers: {missing}")

    signals = _compute_signals(serve, cfg)
    if signals["knee_flexion_deg"] is None:
        warnings.append("knee angles unavailable — using temporal fallback for knee bend event")

    raw_events = _detect_events(signals, cfg)
    n_sig = len(signals["body_velocity"])
    events = _enforce_order(raw_events, cfg, n_sig)

    frames = serve["frames"].astype(int)
    n = len(frames)
    phases = _indices_to_phases(events, n, frames)
    _validate_phases(phases, warnings)

    confidence = _event_confidence(signals, events, cfg)

    event_indices = dict(events)
    events_vicon = {k: int(frames[events[k]]) for k in events}

    return SegmentationResult(
        phases=phases,
        events=events_vicon,
        event_indices=event_indices,
        event_confidence=confidence,
        signals=signals,
        frames=frames,
        warnings=warnings,
    )


def load_serve_from_folder(serve_dir: str | Path) -> dict:
    serve_dir = Path(serve_dir)
    marker_dict = {}
    for csv_path in glob.glob(str(serve_dir / "*.csv")):
        stem = os.path.splitext(os.path.basename(csv_path))[0].lower()
        marker_name = FILENAME_TO_MARKER.get(stem)
        if marker_name:
            marker_dict[marker_name] = csv_path
    return load_single_serve(marker_dict)


def segment_serve_folder(serve_dir: str | Path, config: SegmentationConfig | None = None):
    return segment_serve(load_serve_from_folder(serve_dir), config)


def validate_individual_serves(
    base_dir: str | Path | None = None,
    config: SegmentationConfig | None = None,
) -> list[dict[str, Any]]:
    """Run segmentation on every serve under plotting/markers/individual/."""
    base = Path(base_dir or Path(__file__).resolve().parent.parent / "plotting" / "markers" / "individual")
    results = []
    for serve_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        out = segment_serve_folder(serve_dir, config)
        results.append(
            {
                "serve": serve_dir.name,
                "phases": out.phases,
                "events": out.events,
                "confidence": out.event_confidence,
                "warnings": out.warnings,
            }
        )
    return results


def format_phase_frames(phases: dict[str, tuple[int, int]]) -> str:
    """Compact phase boundaries using Vicon frame numbers (one line per phase)."""
    lines = []
    for name in PHASE_NAMES:
        start, end = phases[name]
        if name == "Contact" or start == end:
            lines.append(f"{name}: {start}")
        else:
            lines.append(f"{name}: {start}-{end}")
    return "\n".join(lines)


def print_phase_frames(serve_name: str, config: SegmentationConfig | None = None) -> None:
    """Segment one serve under individual/ and print phase frame ranges only."""
    serve_dir = (
        Path(__file__).resolve().parent.parent
        / "plotting"
        / "markers"
        / "individual"
        / serve_name
    )
    if not serve_dir.is_dir():
        raise FileNotFoundError(f"Serve folder not found: {serve_dir}")
    result = segment_serve_folder(serve_dir, config)
    print(format_phase_frames(result.phases))


def _print_validation_report(rows: list[dict[str, Any]]) -> None:
    print("=" * 80)
    print("ACE SERVE PHASE SEGMENTATION — VALIDATION (individual/)")
    print("=" * 80)
    for row in rows:
        print(f"\n--- {row['serve']} ---")
        if row["warnings"]:
            for w in row["warnings"]:
                print(f"  WARNING: {w}")
        print("  Events (Vicon frame):")
        for name in EVENT_NAMES:
            conf = row["confidence"].get(name, 0.0)
            print(f"    {name:40s} frame {row['events'][name]:5d}  conf={conf:.2f}")
        print("  Phases (start, end):")
        for name in PHASE_NAMES:
            a, b = row["phases"][name]
            dur = b - a + 1 if b >= a else 0
            print(f"    {name:16s} ({a:5d}, {b:5d})  duration={dur} frames")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ACE serve phase segmentation")
    parser.add_argument(
        "--frames",
        metavar="SERVE",
        help="Print compact Vicon frame ranges for one serve (e.g. firstserve)",
    )
    args = parser.parse_args()

    if args.frames:
        print_phase_frames(args.frames)
    else:
        report = validate_individual_serves()
        _print_validation_report(report)
