"""ACE biomechanics-first serve segmentation (v2) and legacy pipeline."""

from segmentation.config import SegmentationConfig
from segmentation.io import load_serve_from_folder
from segmentation.legacy import segment_serve_legacy
from segmentation.pipeline import segment_serve, segment_serve_folder, segment_serve_v2, validate_individual_serves
from segmentation.result import (
    EVENT_LABELS,
    EVENT_NAMES,
    PHASE_COLORS,
    PHASE_NAMES,
    V2_EVENT_LABELS,
    V2_EVENT_NAMES,
    V2_PHASE_NAMES,
    VIEW_OPTIONS,
    SegmentationResult,
    phase_at_index,
    phase_to_index_range,
    vicon_frame_to_index,
    view_index_range,
)

__all__ = [
    "EVENT_LABELS",
    "EVENT_NAMES",
    "PHASE_COLORS",
    "PHASE_NAMES",
    "V2_EVENT_LABELS",
    "V2_EVENT_NAMES",
    "V2_PHASE_NAMES",
    "VIEW_OPTIONS",
    "SegmentationConfig",
    "SegmentationResult",
    "load_serve_from_folder",
    "phase_at_index",
    "phase_to_index_range",
    "segment_serve",
    "segment_serve_folder",
    "segment_serve_legacy",
    "segment_serve_v2",
    "validate_individual_serves",
    "vicon_frame_to_index",
    "view_index_range",
]
