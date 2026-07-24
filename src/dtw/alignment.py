"""Data structures for phase-aware serve alignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from segmentation.result import SegmentationResult


@dataclass(frozen=True)
class PhaseAlignment:
    """DTW alignment for one biomechanical phase."""

    phase: str
    frames_a: tuple[int, int]
    frames_b: tuple[int, int]
    index_range_a: tuple[int, int]
    index_range_b: tuple[int, int]
    dtw_path: tuple[tuple[int, int], ...]
    global_path: tuple[tuple[int, int], ...]
    distance: float
    normalized_distance: float
    n_frames_a: int
    n_frames_b: int


@dataclass(frozen=True)
class SynchronizedStep:
    """One synchronized playback step spanning a single phase DTW pair."""

    phase: str
    global_index_a: int
    global_index_b: int
    vicon_frame_a: int
    vicon_frame_b: int
    step_index: int
    phase_step_index: int


@dataclass
class ServeComparison:
    """Full phase-by-phase DTW comparison between two serves."""

    name_a: str
    name_b: str
    segmentation_a: SegmentationResult
    segmentation_b: SegmentationResult
    phase_alignments: list[PhaseAlignment] = field(default_factory=list)
    synchronized_steps: list[SynchronizedStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_path_length(self) -> int:
        return len(self.synchronized_steps)

    def phase_alignment(self, phase: str) -> PhaseAlignment | None:
        for alignment in self.phase_alignments:
            if alignment.phase == phase:
                return alignment
        return None
