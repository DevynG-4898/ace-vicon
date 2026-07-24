"""Knowledge Library entry schema.

All tennis-specific coaching copy lives here (or in entry data). The Coaching
Engine must not hard-code these strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class KnowledgeEntry:
    """Coaching knowledge for one feature, keyed by feature name."""

    feature: str
    phase: str
    too_low: str
    too_high: str
    coach_quotes: tuple[str, ...] = ()
    practice_drills: tuple[str, ...] = field(default_factory=tuple)

    def correction_for(self, direction: str) -> str:
        """Return the correction message for ``too_low`` or ``too_high``."""
        if direction == "too_low":
            return self.too_low
        if direction == "too_high":
            return self.too_high
        raise ValueError(f"direction must be 'too_low' or 'too_high'; got {direction!r}")
