import numpy as np

from src.features import FeatureSequence, extract_features
from src.features.io import load_feature_sequence, save_feature_sequence
from src.skeleton import NORMALIZED_COORDINATE_SYSTEM, SkeletonSequence


JOINT_NAMES = (
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)


def _frame() -> np.ndarray:
    points = {name: np.zeros(3, dtype=float) for name in JOINT_NAMES}
    points.update(
        {
            "pelvis": np.array([0.0, 0.0, 0.0]),
            "left_hip": np.array([-0.5, 0.0, 0.0]),
            "right_hip": np.array([0.5, 0.0, 0.0]),
            "spine": np.array([0.0, 0.6, 0.0]),
            "thorax": np.array([0.0, 1.0, 0.0]),
            "neck": np.array([0.0, 1.25, 0.0]),
            "head": np.array([0.0, 1.6, 0.0]),
            "left_shoulder": np.array([-0.5, 1.0, 0.0]),
            "right_shoulder": np.array([0.5, 1.0, 0.0]),
            "left_elbow": np.array([-1.0, 1.0, 0.0]),
            "left_wrist": np.array([-1.0, 0.5, 0.0]),
            "right_elbow": np.array([1.0, 1.0, 0.0]),
            "right_wrist": np.array([1.5, 1.0, 0.0]),
            "left_knee": np.array([-0.5, -1.0, 0.0]),
            "left_ankle": np.array([-0.5, -2.0, 0.0]),
            "right_knee": np.array([0.5, -1.0, 0.0]),
            "right_ankle": np.array([1.0, -1.0, 0.0]),
        }
    )
    return np.stack([points[name] for name in JOINT_NAMES], axis=0)


def _sequence(positions: np.ndarray, *, fps: float = 10.0) -> SkeletonSequence:
    return SkeletonSequence(
        frames=np.arange(1, positions.shape[0] + 1),
        joint_names=JOINT_NAMES,
        joint_positions=positions,
        joint_confidence=np.ones(positions.shape[:2], dtype=float),
        fps=fps,
        metadata={"fixture": "features"},
        coordinate_system=NORMALIZED_COORDINATE_SYSTEM,
        source="unit-test",
    )


def _constant_sequence(n_frames: int = 5, *, fps: float = 10.0) -> SkeletonSequence:
    return _sequence(np.repeat(_frame()[None, :, :], n_frames, axis=0), fps=fps)


def test_extract_features_returns_canonical_feature_sequence():
    sequence = _constant_sequence()

    features = extract_features(sequence)

    assert isinstance(features, FeatureSequence)
    assert np.array_equal(features.frames, sequence.frames)
    assert features.fps == sequence.fps
    assert features.source_sequence is sequence
    assert features.metadata["source_coordinate_system"] == NORMALIZED_COORDINATE_SYSTEM


def test_known_joint_angle_geometries_are_degrees():
    features = extract_features(_constant_sequence())

    assert np.allclose(features.feature("left_elbow_angle"), 90.0)
    assert np.allclose(features.feature("right_elbow_angle"), 180.0)
    assert np.allclose(features.feature("left_knee_angle"), 180.0)
    assert np.allclose(features.feature("right_knee_angle"), 90.0)
    assert np.allclose(features.feature("spine_angle"), 0.0)
    assert np.allclose(features.feature("trunk_tilt"), 0.0)
    assert np.allclose(features.feature("shoulder_line_angle"), 0.0)
    assert np.allclose(features.feature("pelvis_tilt"), 0.0)


def test_velocity_and_speed_use_fps():
    base = _frame()
    frames = np.stack([base, base.copy(), base.copy()], axis=0)
    right_wrist = JOINT_NAMES.index("right_wrist")
    frames[:, right_wrist, 0] = [0.0, 0.5, 1.0]
    sequence = _sequence(frames, fps=20.0)

    features = extract_features(sequence)

    assert np.allclose(features.feature("right_hand_velocity_x"), 10.0)
    assert np.allclose(features.feature("right_hand_velocity_y"), 0.0)
    assert np.allclose(features.feature("right_hand_velocity_z"), 0.0)
    assert np.allclose(features.feature("right_hand_speed"), 10.0)


def test_acceleration_is_zero_for_constant_velocity_motion():
    base = _frame()
    frames = np.stack([base, base.copy(), base.copy(), base.copy()], axis=0)
    right_wrist = JOINT_NAMES.index("right_wrist")
    frames[:, right_wrist, 0] = [0.0, 0.25, 0.5, 0.75]

    features = extract_features(_sequence(frames, fps=4.0))

    assert np.allclose(features.feature("right_hand_acceleration_x"), 0.0)
    assert np.allclose(features.feature("right_hand_acceleration"), 0.0)


def test_constant_sequence_has_zero_joint_and_com_motion():
    features = extract_features(_constant_sequence())

    assert np.allclose(features.feature("left_hand_speed"), 0.0)
    assert np.allclose(features.feature("center_of_mass_velocity"), 0.0)
    assert np.allclose(features.feature("center_of_mass_acceleration"), 0.0)


def test_center_of_mass_translates_with_skeleton():
    base = _frame()
    shifted = base + np.array([2.0, -1.0, 0.5])
    base_features = extract_features(_sequence(base[None, :, :]))
    shifted_features = extract_features(_sequence(shifted[None, :, :]))

    assert np.isclose(
        shifted_features.feature("center_of_mass_x")[0] - base_features.feature("center_of_mass_x")[0],
        2.0,
    )
    assert np.isclose(
        shifted_features.feature("center_of_mass_y")[0] - base_features.feature("center_of_mass_y")[0],
        -1.0,
    )
    assert np.isclose(
        shifted_features.feature("center_of_mass_z")[0] - base_features.feature("center_of_mass_z")[0],
        0.5,
    )


def test_feature_names_are_consistent_and_no_placeholders():
    names = set(extract_features(_constant_sequence()).features)
    expected = {
        "left_elbow_angle",
        "right_elbow_angle",
        "left_shoulder_flexion",
        "right_shoulder_flexion",
        "left_shoulder_abduction",
        "right_shoulder_abduction",
        "shoulder_line_angle",
        "trunk_tilt",
        "trunk_rotation",
        "left_knee_angle",
        "right_knee_angle",
        "hip_flexion",
        "hip_abduction",
        "spine_angle",
        "pelvis_tilt",
        "left_hand_speed",
        "right_hand_speed",
        "center_of_mass_x",
        "center_of_mass_y",
        "center_of_mass_z",
        "center_of_mass_velocity",
        "center_of_mass_acceleration",
    }

    assert expected.issubset(names)
    assert all("todo" not in name.lower() and "placeholder" not in name.lower() for name in names)


def test_feature_sequence_serialization_round_trips(tmp_path):
    features = extract_features(_constant_sequence())

    path = save_feature_sequence(tmp_path, features)
    loaded = load_feature_sequence(path)

    assert path.name == "feature_sequence.npz"
    assert np.array_equal(loaded.frames, features.frames)
    assert loaded.fps == features.fps
    assert loaded.metadata["source_coordinate_system"] == NORMALIZED_COORDINATE_SYSTEM
    assert np.allclose(loaded.feature("left_elbow_angle"), features.feature("left_elbow_angle"))
    assert loaded.source_sequence is None


def test_extract_features_rejects_raw_skeleton_sequences():
    sequence = _constant_sequence()
    raw = SkeletonSequence(
        frames=sequence.frames,
        joint_names=sequence.joint_names,
        joint_positions=sequence.joint_positions,
        joint_confidence=sequence.joint_confidence,
        fps=sequence.fps,
        metadata=sequence.metadata,
        coordinate_system="motionbert_root_centered",
        source=sequence.source,
    )

    try:
        extract_features(raw)
    except ValueError as exc:
        assert "normalized" in str(exc)
    else:
        raise AssertionError("Expected raw skeleton feature extraction to fail")
