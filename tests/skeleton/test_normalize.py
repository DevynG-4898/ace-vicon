import numpy as np

from src.skeleton import SkeletonSequence, normalize_skeleton


JOINT_NAMES = (
    "left_hip",
    "right_hip",
    "left_shoulder",
    "right_shoulder",
    "head",
)


def _base_positions(scale: float = 1.0) -> np.ndarray:
    frame = np.array(
        [
            [-0.4, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            [-0.5, 1.0, 0.2],
            [0.5, 1.0, 0.2],
            [0.0, 1.7, 0.25],
        ],
        dtype=float,
    )
    return np.stack([frame * scale, frame * scale + np.array([0.2, 0.1, -0.1])], axis=0)


def _sequence(positions: np.ndarray) -> SkeletonSequence:
    return SkeletonSequence(
        frames=np.arange(1, positions.shape[0] + 1),
        joint_names=JOINT_NAMES,
        joint_positions=positions,
        joint_confidence=np.ones(positions.shape[:2], dtype=float),
        fps=30.0,
        metadata={"source_file": "synthetic.npy"},
        coordinate_system="test_world",
        source="unit-test",
    )


def _rotate_y(points: np.ndarray, degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    rot = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ],
        dtype=float,
    )
    return points @ rot.T


def test_translation_places_pelvis_at_origin():
    normalized = normalize_skeleton(_sequence(_base_positions()))
    left_hip = normalized.joint_index("left_hip")
    right_hip = normalized.joint_index("right_hip")

    pelvis = (
        normalized.joint_positions[:, left_hip, :]
        + normalized.joint_positions[:, right_hip, :]
    ) / 2.0

    assert np.allclose(pelvis, 0.0, atol=1e-8)


def test_rotation_makes_rotated_copies_match():
    base = normalize_skeleton(_sequence(_base_positions()))
    rotated = normalize_skeleton(_sequence(_rotate_y(_base_positions(), 37.0)))

    assert np.allclose(rotated.joint_positions, base.joint_positions, atol=1e-8)


def test_scale_makes_large_and_small_skeletons_match():
    small = normalize_skeleton(_sequence(_base_positions(scale=0.5)))
    large = normalize_skeleton(_sequence(_base_positions(scale=3.0)))

    assert np.allclose(small.joint_positions, large.joint_positions, atol=1e-8)
    assert np.isclose(small.metadata["normalization"]["scale_factor"], 0.5)
    assert np.isclose(large.metadata["normalization"]["scale_factor"], 3.0)


def test_normalization_does_not_modify_input_sequence():
    positions = _base_positions(scale=2.0)
    original_positions = positions.copy()
    sequence = _sequence(positions)
    original_metadata = dict(sequence.metadata)

    normalize_skeleton(sequence)

    assert np.array_equal(sequence.joint_positions, original_positions)
    assert sequence.metadata == original_metadata


def test_normalization_is_idempotent():
    normalized_once = normalize_skeleton(_sequence(_base_positions(scale=2.0)))
    normalized_twice = normalize_skeleton(normalized_once)

    assert np.allclose(normalized_twice.joint_positions, normalized_once.joint_positions, atol=1e-8)
    assert np.isclose(normalized_twice.metadata["normalization"]["scale_factor"], 1.0)
