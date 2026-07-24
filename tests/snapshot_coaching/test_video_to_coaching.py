"""Tests for ACE markers → formatted CSV → snapshot coaching path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from format.data.ace_to_formatted import ace_markers_to_formatted_csv, markers_to_dataframe
from format.pipeline import run_snapshot_coaching_pipeline
from src.markers.io import ACE_MARKER_NAMES


def _synthetic_serve(*, racket_side: str = "right", frames: int = 120, seed: int = 0) -> dict:
    """Build a simple ACE marker dict with toss/racket peaks and knee bend."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, frames)
    frames_arr = np.arange(1, frames + 1, dtype=int)
    markers: dict = {"frames": frames_arr}

    # Base standing pose (mm-ish units).
    base = {
        "head": (0.0, 0.0, 1700.0),
        "chest": (0.0, 0.0, 1400.0),
        "left_shoulder": (-200.0, 0.0, 1450.0),
        "right_shoulder": (200.0, 0.0, 1450.0),
        "left_elbow": (-350.0, 50.0, 1200.0),
        "right_elbow": (350.0, 50.0, 1200.0),
        "left_hand": (-400.0, 80.0, 1000.0),
        "right_hand": (400.0, 80.0, 1000.0),
        "left_hip": (-100.0, 0.0, 900.0),
        "right_hip": (100.0, 0.0, 900.0),
        "left_knee": (-110.0, 40.0, 450.0),
        "right_knee": (110.0, 40.0, 450.0),
        "left_foot": (-120.0, 60.0, 50.0),
        "right_foot": (120.0, 60.0, 50.0),
    }

    toss = "left_hand" if racket_side == "right" else "right_hand"
    racket = "right_hand" if racket_side == "right" else "left_hand"

    for name in ACE_MARKER_NAMES:
        x0, y0, z0 = base[name]
        noise = rng.normal(0.0, 2.0, size=(frames, 3))
        xyz = np.column_stack(
            [
                np.full(frames, x0) + noise[:, 0],
                np.full(frames, y0) + noise[:, 1],
                np.full(frames, z0) + noise[:, 2],
            ]
        )
        # Toss hand rises early then drops.
        if name == toss:
            xyz[:, 2] += 900.0 * np.exp(-((t - 0.25) ** 2) / 0.01)
        # Racket hand rises later (contact / follow-through).
        if name == racket:
            xyz[:, 2] += 1100.0 * np.exp(-((t - 0.65) ** 2) / 0.008)
            # Elbow / shoulder move with racket roughly.
        if name == f"{racket_side}_elbow":
            xyz[:, 2] += 700.0 * np.exp(-((t - 0.55) ** 2) / 0.01)
        if name == f"{racket_side}_shoulder":
            xyz[:, 2] += 200.0 * np.exp(-((t - 0.55) ** 2) / 0.015)
        # Knees dip during loading.
        if name in ("left_knee", "right_knee"):
            xyz[:, 2] -= 180.0 * np.exp(-((t - 0.35) ** 2) / 0.02)
        markers[name] = {
            "TX": xyz[:, 0],
            "TY": xyz[:, 1],
            "TZ": xyz[:, 2],
        }
    return markers


class TestAceToFormatted:
    def test_writes_peaks_and_snapshots(self, tmp_path: Path):
        markers = _synthetic_serve()
        out = ace_markers_to_formatted_csv(markers, tmp_path / "serve_formatted.csv")
        text = out.read_text(encoding="utf-8")
        assert "PEAK1=" in text
        assert "PEAK2=" in text
        assert "SNAPSHOT=contact," in text
        # Snapshots should be filled (not all zeros).
        snap_lines = [line for line in text.splitlines() if line.startswith("SNAPSHOT=")]
        values = [int(line.split(",")[1]) for line in snap_lines]
        assert any(v > 0 for v in values)

    def test_markers_to_dataframe_shape(self):
        markers = _synthetic_serve(frames=30)
        df = markers_to_dataframe(markers)
        assert len(df) == 30
        assert "right_hand_TZ" in df.columns


class TestMarkersToCoaching:
    def test_two_marker_serves_produce_coaching_report(self, tmp_path: Path):
        player = _synthetic_serve(seed=1)
        # Exaggerate elbow difference for a clear weakness signal.
        player["right_elbow"]["TX"] = player["right_elbow"]["TX"] + 250.0
        reference = _synthetic_serve(seed=2)

        player_csv = ace_markers_to_formatted_csv(player, tmp_path / "player_formatted.csv")
        ref_csv = ace_markers_to_formatted_csv(reference, tmp_path / "ref_formatted.csv")

        result = run_snapshot_coaching_pipeline(player_csv, ref_csv)
        assert result.scoring_report.overall_score is not None
        assert 0.0 <= result.scoring_report.overall_score <= 100.0
        assert result.coaching_report.overall_score == pytest.approx(
            result.scoring_report.overall_score
        )
        assert result.customer_formatted_csv == player_csv
        assert result.scoring_report.feature_summaries
