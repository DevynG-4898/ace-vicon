"""ACE Coaching Viewer — DTW + scoring + coaching visualization."""

from __future__ import annotations

__all__ = ["CoachingViewerSession", "run"]


def __getattr__(name: str):
    if name in {"CoachingViewerSession", "run"}:
        from src.coaching_viewer.pipeline import CoachingViewerSession, run

        return {"CoachingViewerSession": CoachingViewerSession, "run": run}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
