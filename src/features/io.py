from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.features.feature_sequence import FeatureSequence
from src.json_utils import jsonable

FEATURE_SEQUENCE_FILENAME = "feature_sequence.npz"


def save_feature_sequence(
    output_dir: str | Path,
    feature_sequence: FeatureSequence,
    *,
    filename: str = FEATURE_SEQUENCE_FILENAME,
) -> Path:
    """Persist a FeatureSequence as `feature_sequence.npz`."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    names = np.asarray(feature_sequence.names, dtype=str)
    matrix = np.column_stack([feature_sequence.feature(name) for name in feature_sequence.names])
    metadata = json.dumps(jsonable(feature_sequence.metadata), sort_keys=True)
    np.savez(
        path,
        frames=feature_sequence.frames,
        fps=np.asarray(np.nan if feature_sequence.fps is None else feature_sequence.fps, dtype=float),
        feature_names=names,
        feature_values=matrix,
        metadata=np.asarray(metadata),
    )
    return path


def load_feature_sequence(path: str | Path) -> FeatureSequence:
    """Load a FeatureSequence saved by `save_feature_sequence()`."""
    loaded = np.load(path, allow_pickle=False)
    names = [str(name) for name in loaded["feature_names"].tolist()]
    values = loaded["feature_values"]
    features = {name: values[:, idx] for idx, name in enumerate(names)}
    fps_value = float(loaded["fps"])
    return FeatureSequence(
        frames=loaded["frames"],
        fps=None if np.isnan(fps_value) else fps_value,
        features=features,
        metadata=json.loads(str(loaded["metadata"])),
        source_sequence=None,
    )

