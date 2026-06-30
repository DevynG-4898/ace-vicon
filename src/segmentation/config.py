from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SegmentationConfig:
    """Tunable thresholds for legacy and v2 segmentation."""

    # Detection mode (Phase B: default legacy until Phase C flip)
    use_legacy_detection: bool = True

    smooth_window: int = 11
    min_event_gap_frames: int = 5

    # Legacy — first movement (body velocity)
    baseline_frames: int = 40
    body_velocity_threshold_ratio: float = 0.12
    body_velocity_persist_frames: int = 6

    # Legacy — post-contact
    post_contact_velocity_fraction: float = 0.55
    velocity_decrease_persist_frames: int = 10
    stabilization_velocity_ratio: float = 0.15
    stabilization_persist_frames: int = 20

    # Legacy only (unused in v2 phase mapping)
    min_phase_frames: int = 8

    # Handedness
    handedness: str = "right"  # "right" | "left"

    # Marker roles (derived from handedness when not set explicitly)
    serving_hand: str = "right_hand"
    serving_side_knee: str = "right_knee"
    serving_side_hip: str = "right_hip"
    serving_side_foot: str = "right_foot"
    serving_shoulder: str = "right_shoulder"
    serving_elbow: str = "right_elbow"
    toss_hand: str = "left_hand"

    # V2 — E1 first movement
    initiation_weights: tuple[float, float, float] = (0.35, 0.40, 0.25)
    initiation_threshold_ratio: float = 0.08
    initiation_persist_frames: int = 5

    # V2 — E2 toss apex
    toss_smooth_window: int = 7
    toss_peak_distance: int = 20
    toss_peak_prominence_mm: float = 20.0
    toss_search_fraction: float = 0.35
    toss_min_height_ratio: float = 1.12

    # V2 — E3 knee flexion
    knee_smooth_window: int = 9
    knee_flexion_min_depth_deg: float = 10.0
    knee_pre_contact_offset: int = 10

    # V2 — E4 shoulder ER
    shoulder_er_min_increase_deg: float = 15.0
    shoulder_pre_contact_offset: int = 3
    shoulder_max_hand_v_fraction: float = 0.40

    # V2 — E5 contact (racket height apex in pre-velocity-peak window)
    contact_search_window_frames: int = 30
    contact_hand_height_weight: float = 0.45
    contact_shoulder_velocity_weight: float = 0.20
    contact_upper_body_angular_weight: float = 0.15
    contact_elbow_extension_weight: float = 0.20
    contact_height_band_fraction: float = 0.97
    contact_min_elbow_extension_deg: float = 125.0
    contact_elbow_fraction_of_max: float = 0.90
    contact_smooth_window: int = 5
    contact_prominence_fraction: float = 0.25
    contact_peak_distance: int = 15
    contact_max_frames_after_shoulder: int = 0  # 0 => use fraction
    contact_max_frames_after_shoulder_fraction: float = 0.15
    contact_min_frames_after_shoulder: int = 3

    # V2 — E6 finish
    finish_min_frames_after_contact: int = 10
    finish_body_velocity_ratio: float = 0.15
    finish_racket_velocity_fraction: float = 0.20
    finish_body_persist_frames: int = 20
    finish_hand_persist_frames: int = 15

    # Marker quality
    marker_nan_threshold: float = 0.15

    # Coaching phase model — Start Stance → Release
    stance_velocity_threshold_ratio: float = 0.08
    stance_motion_persist_frames: int = 5

    # Coaching — Release → Loading (toss hand reaches head level)
    release_head_level_persist_frames: int = 3
    release_min_post_apex_frames: int = 20
    release_max_post_apex_frames: int = 60
    release_toss_drop_mm: float = 25.0
    release_toss_drop_persist_frames: int = 3
    loading_knee_derivative_threshold: float = -0.08
    loading_knee_derivative_persist_frames: int = 5
    release_loading_hand_persist_frames: int = 3

    # Coaching — Loading → Cocking
    loading_min_frames: int = 20
    cocking_knee_depth_fraction: float = 0.70
    cocking_min_knee_depth_deg: float = 8.0
    cocking_shoulder_er_threshold_deg: float = 95.0
    cocking_elbow_extension_max_deg: float = 95.0
    cocking_trunk_tilt_threshold_deg: float = 8.0
    cocking_entry_persist_frames: int = 3

    # Coaching — Cocking → Acceleration (upswing from max stretch)
    acceleration_upswing_persist_frames: int = 3
    acceleration_hand_rise_mm: float = 35.0
    acceleration_min_hand_velocity: float = 8.0
    acceleration_min_frames_after_cocking: int = 5
    acceleration_er_derivative_persist_frames: int = 3
    acceleration_min_shoulder_er_deg: float = 105.0

    def __post_init__(self) -> None:
        if self.handedness == "left":
            self.serving_hand = "left_hand"
            self.serving_side_knee = "left_knee"
            self.serving_side_hip = "left_hip"
            self.serving_side_foot = "left_foot"
            self.serving_shoulder = "left_shoulder"
            self.serving_elbow = "left_elbow"
            self.toss_hand = "right_hand"

    @property
    def toss_shoulder(self) -> str:
        """Shoulder on the toss-arm side (opposite serving shoulder for standard grips)."""
        if self.handedness == "right":
            return "left_shoulder"
        return "right_shoulder"
