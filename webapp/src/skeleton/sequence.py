from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SkeletonSequence:
    """Canonical representation of a full motion sequence.

    `joint_positions` always has shape `(frames, joints, 3)`. Confidence is
    stored separately with shape `(frames, joints)` so downstream modules do not
    need to infer score channels from coordinate arrays.
    """

    frames: np.ndarray
    joint_names: tuple[str, ...]
    joint_positions: np.ndarray
    joint_confidence: np.ndarray | None
    fps: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
    coordinate_system: str = "unknown"
    source: str = "unknown"

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames)
        positions = np.asarray(self.joint_positions, dtype=float)
        names = tuple(self.joint_names)
        confidence = None if self.joint_confidence is None else np.asarray(self.joint_confidence, dtype=float)

        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError(f"joint_positions must have shape (frames, joints, 3); got {positions.shape}")
        if frames.ndim != 1 or frames.shape[0] != positions.shape[0]:
            raise ValueError(
                "frames must be a 1D array with one entry per frame; "
                f"got frames {frames.shape} and positions {positions.shape}"
            )
        if len(names) != positions.shape[1]:
            raise ValueError(
                f"joint_names length must match joint count {positions.shape[1]}; got {len(names)}"
            )
        if confidence is not None and confidence.shape != positions.shape[:2]:
            raise ValueError(
                "joint_confidence must have shape (frames, joints); "
                f"got {confidence.shape} for positions {positions.shape}"
            )

        object.__setattr__(self, "frames", frames.copy())
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "joint_positions", positions.copy())
        object.__setattr__(self, "joint_confidence", None if confidence is None else confidence.copy())
        object.__setattr__(self, "fps", None if self.fps is None else float(self.fps))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_frames(self) -> int:
        """Number of frames in the sequence."""
        return int(self.joint_positions.shape[0])

    @property
    def n_joints(self) -> int:
        """Number of joints in each frame."""
        return int(self.joint_positions.shape[1])

    def joint_index(self, name: str) -> int:
        """Return the integer index for a named joint."""
        try:
            return self.joint_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Joint not found: {name}") from exc

    def joint(self, name: str) -> np.ndarray:
        """Return `(frames, 3)` positions for a named joint."""
        return self.joint_positions[:, self.joint_index(name), :]

    def with_positions(
        self,
        joint_positions: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
        coordinate_system: str | None = None,
        source: str | None = None,
    ) -> "SkeletonSequence":
        """Return a copy with replaced positions and optional metadata changes."""
        return SkeletonSequence(
            frames=self.frames,
            joint_names=self.joint_names,
            joint_positions=joint_positions,
            joint_confidence=self.joint_confidence,
            fps=self.fps,
            metadata=self.metadata if metadata is None else metadata,
            coordinate_system=self.coordinate_system if coordinate_system is None else coordinate_system,
            source=self.source if source is None else source,
        )

