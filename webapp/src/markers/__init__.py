"""Canonical ACE marker dictionary — shared interface for Vicon and MotionBERT sources."""

from src.markers.io import (
    ACE_MARKER_NAMES,
    is_marker_dict,
    load_serve_markers,
    save_serve_markers,
)

__all__ = [
    "ACE_MARKER_NAMES",
    "is_marker_dict",
    "load_serve_markers",
    "save_serve_markers",
]
