from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.skeleton import SkeletonSequence


@dataclass(frozen=True)
class FeatureSequence:
    """Canonical per-frame biomechanical feature representation."""

    frames: np.ndarray
    fps: float | None
    features: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    source_sequence: SkeletonSequence | None = None

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames)
        if frames.ndim != 1:
            raise ValueError(f"frames must be 1D; got {frames.shape}")

        copied: dict[str, np.ndarray] = {}
        for name, values in self.features.items():
            arr = np.asarray(values, dtype=float)
            if arr.shape != frames.shape:
                raise ValueError(f"feature {name} must have shape {frames.shape}; got {arr.shape}")
            copied[str(name)] = arr.copy()

        object.__setattr__(self, "frames", frames.copy())
        object.__setattr__(self, "fps", None if self.fps is None else float(self.fps))
        object.__setattr__(self, "features", copied)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def feature(self, name: str) -> np.ndarray:
        """Return a named per-frame feature array."""
        try:
            return self.features[name]
        except KeyError as exc:
            raise KeyError(f"Feature not found: {name}") from exc

    @property
    def names(self) -> tuple[str, ...]:
        """Feature names in deterministic sorted order."""
        return tuple(sorted(self.features))

