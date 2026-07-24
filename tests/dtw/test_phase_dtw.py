import numpy as np
import pytest

from segmentation import SegmentationConfig, segment_serve
from src.dtw.features import MarkerFeatureExtractor
from src.dtw.metrics import dtw_warping_path
from src.dtw.phase_dtw import compare_serves, segment_and_compare
from src.markers.io import load_serve_markers


def test_dtw_path_same_series_is_near_diagonal():
    features = np.stack([np.arange(20, dtype=float), np.arange(20, dtype=float) * 0.5], axis=1)
    path, cost = dtw_warping_path(features, features)
    assert len(path) >= 19
    assert cost < 1e-6
    assert sum(1 for ia, ib in path if ia == ib) >= len(path) - 2


def test_phase_dtw_does_not_cross_phase_boundaries(firstserve_dict, v2_config):
    markers = firstserve_dict
    cfg = v2_config
    seg = segment_serve(markers, cfg)
    comparison = compare_serves(markers, markers, seg, seg, name_a="a", name_b="b")

    for alignment in comparison.phase_alignments:
        i0_a, i1_a = alignment.index_range_a
        i0_b, i1_b = alignment.index_range_b
        for ga, gb in alignment.global_path:
            assert i0_a <= ga <= i1_a
            assert i0_b <= gb <= i1_b


def test_firstserve_vs_itself_has_low_normalized_distance(firstserve_dir):
    comparison, _ = segment_and_compare(str(firstserve_dir), str(firstserve_dir), name_a="a", name_b="b")
    finite = [a.normalized_distance for a in comparison.phase_alignments if np.isfinite(a.normalized_distance)]
    assert finite
    assert np.nanmedian(finite) < 0.05


def test_segment_and_compare_produces_synchronized_steps(firstserve_dir):
    comparison, bundle = segment_and_compare(
        str(firstserve_dir),
        str(firstserve_dir),
        name_a="firstserve",
        name_b="firstserve",
    )
    assert comparison.total_path_length > 0
    assert len(bundle["markers_a"]["frames"]) == len(bundle["markers_b"]["frames"])
    step = comparison.synchronized_steps[0]
    assert step.vicon_frame_a == step.vicon_frame_b


def test_marker_feature_extractor_shape(firstserve_dict):
    extractor = MarkerFeatureExtractor()
    features = extractor(firstserve_dict)
    assert features.ndim == 2
    assert features.shape[0] == len(firstserve_dict["frames"])
    assert features.shape[1] == 14 * 3


@pytest.mark.skipif(
    not __import__("pathlib").Path("generated_motionbert/andy/ace_markers.npz").is_file(),
    reason="andy motionbert output not present",
)
def test_vicon_vs_motionbert_runs():
    comparison, _ = segment_and_compare(
        "plotting/markers/individual/firstserve",
        "generated_motionbert/andy",
        name_a="firstserve",
        name_b="andy",
    )
    assert len(comparison.phase_alignments) >= 5
    assert comparison.total_path_length > 0
