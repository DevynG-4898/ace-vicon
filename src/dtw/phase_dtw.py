"""Phase-bounded DTW between two ACE serves."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from segmentation import SegmentationConfig, SegmentationResult, segment_serve
from segmentation.result import V2_PHASE_NAMES, phase_to_index_range

from src.dtw.alignment import PhaseAlignment, ServeComparison, SynchronizedStep
from src.dtw.features import FeatureExtractor, MarkerFeatureExtractor
from src.dtw.metrics import dtw_warping_path, normalized_dtw_distance
from src.markers.io import load_serve_markers


def _phase_list(result: SegmentationResult) -> tuple[str, ...]:
    if result.schema_version == 2:
        return V2_PHASE_NAMES
    return tuple(result.phases.keys())


def _extract_phase_series(
    features: np.ndarray,
    index_range: tuple[int, int],
) -> np.ndarray:
    i0, i1 = index_range
    return features[i0 : i1 + 1]


def align_phase(
    phase: str,
    features_a: np.ndarray,
    features_b: np.ndarray,
    frames_a: np.ndarray,
    frames_b: np.ndarray,
    bounds_a: tuple[int, int],
    bounds_b: tuple[int, int],
    index_a: tuple[int, int],
    index_b: tuple[int, int],
) -> PhaseAlignment:
    """Run DTW on one phase slice; path indices are local to the phase."""
    series_a = _extract_phase_series(features_a, index_a)
    series_b = _extract_phase_series(features_b, index_b)

    local_path, total_cost = dtw_warping_path(series_a, series_b)
    i0_a, _ = index_a
    i0_b, _ = index_b
    global_path = tuple((i0_a + ia, i0_b + ib) for ia, ib in local_path)
    norm = normalized_dtw_distance(total_cost, local_path, series_a.shape[0], series_b.shape[0])

    return PhaseAlignment(
        phase=phase,
        frames_a=bounds_a,
        frames_b=bounds_b,
        index_range_a=index_a,
        index_range_b=index_b,
        dtw_path=tuple(local_path),
        global_path=global_path,
        distance=float(total_cost),
        normalized_distance=float(norm),
        n_frames_a=int(series_a.shape[0]),
        n_frames_b=int(series_b.shape[0]),
    )


def _build_synchronized_steps(
    markers_a: dict,
    markers_b: dict,
    phase_alignments: list[PhaseAlignment],
) -> list[SynchronizedStep]:
    steps: list[SynchronizedStep] = []
    step_idx = 0
    for alignment in phase_alignments:
        for phase_step, (ga, gb) in enumerate(alignment.global_path):
            steps.append(
                SynchronizedStep(
                    phase=alignment.phase,
                    global_index_a=ga,
                    global_index_b=gb,
                    vicon_frame_a=int(markers_a["frames"][ga]),
                    vicon_frame_b=int(markers_b["frames"][gb]),
                    step_index=step_idx,
                    phase_step_index=phase_step,
                )
            )
            step_idx += 1
    return steps


def compare_serves(
    markers_a: dict,
    markers_b: dict,
    segmentation_a: SegmentationResult,
    segmentation_b: SegmentationResult,
    *,
    name_a: str = "serve_a",
    name_b: str = "serve_b",
    feature_extractor: FeatureExtractor | None = None,
) -> ServeComparison:
    """Segmentation must already be computed; runs DTW independently per phase."""
    extractor = feature_extractor or MarkerFeatureExtractor()
    features_a = extractor(markers_a)
    features_b = extractor(markers_b)

    phases = _phase_list(segmentation_a)
    if set(phases) != set(_phase_list(segmentation_b)):
        missing = set(phases) ^ set(_phase_list(segmentation_b))
        raise ValueError(f"Phase name mismatch between serves: {missing}")

    alignments: list[PhaseAlignment] = []
    for phase in phases:
        if phase not in segmentation_a.phases or phase not in segmentation_b.phases:
            continue
        bounds_a = segmentation_a.phases[phase]
        bounds_b = segmentation_b.phases[phase]
        index_a = phase_to_index_range(segmentation_a.frames, bounds_a)
        index_b = phase_to_index_range(segmentation_b.frames, bounds_b)
        if index_a[1] < index_a[0] or index_b[1] < index_b[0]:
            continue
        alignments.append(
            align_phase(
                phase,
                features_a,
                features_b,
                segmentation_a.frames,
                segmentation_b.frames,
                bounds_a,
                bounds_b,
                index_a,
                index_b,
            )
        )

    steps = _build_synchronized_steps(markers_a, markers_b, alignments)
    return ServeComparison(
        name_a=name_a,
        name_b=name_b,
        segmentation_a=segmentation_a,
        segmentation_b=segmentation_b,
        phase_alignments=alignments,
        synchronized_steps=steps,
    )


def segment_and_compare(
    source_a: str | dict,
    source_b: str | dict,
    *,
    name_a: str | None = None,
    name_b: str | None = None,
    use_v2: bool = True,
    feature_extractor: FeatureExtractor | None = None,
) -> tuple[ServeComparison, dict[str, Any]]:
    """Load markers, segment both serves, then run phase DTW."""
    started = time.perf_counter()
    markers_a = load_serve_markers(source_a)
    markers_b = load_serve_markers(source_b)

    if name_a is None:
        name_a = source_a if isinstance(source_a, str) else "serve_a"
    if name_b is None:
        name_b = source_b if isinstance(source_b, str) else "serve_b"

    cfg = SegmentationConfig(use_legacy_detection=not use_v2)
    seg_a = segment_serve(markers_a, cfg)
    seg_b = segment_serve(markers_b, cfg)

    comparison = compare_serves(
        markers_a,
        markers_b,
        seg_a,
        seg_b,
        name_a=str(name_a),
        name_b=str(name_b),
        feature_extractor=feature_extractor,
    )
    elapsed = time.perf_counter() - started
    timing = {"elapsed_seconds": elapsed, "total_path_length": comparison.total_path_length}
    comparison.metadata["timing"] = timing
    return comparison, {"markers_a": markers_a, "markers_b": markers_b, "timing": timing}


def format_comparison_report(comparison: ServeComparison) -> str:
    lines = [
        f"Phase-aware DTW: {comparison.name_a}  vs  {comparison.name_b}",
        "=" * 72,
    ]
    for alignment in comparison.phase_alignments:
        fa0, fa1 = alignment.frames_a
        fb0, fb1 = alignment.frames_b
        lines.append(f"\n{alignment.phase}")
        lines.append(f"  Frames A: {fa0}-{fa1}  ({alignment.n_frames_a} frames)")
        lines.append(f"  Frames B: {fb0}-{fb1}  ({alignment.n_frames_b} frames)")
        lines.append(f"  Warping path length: {len(alignment.dtw_path)}")
        lines.append(f"  DTW distance: {alignment.distance:.2f}")
        lines.append(f"  Normalized distance: {alignment.normalized_distance:.4f}")
    timing = comparison.metadata.get("timing", {})
    if timing:
        lines.append(f"\nTotal synchronized steps: {comparison.total_path_length}")
        lines.append(f"Execution time: {timing.get('elapsed_seconds', 0.0):.3f}s")
    return "\n".join(lines)
