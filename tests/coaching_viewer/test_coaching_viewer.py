"""Unit tests for coaching viewer helpers and pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from segmentation.result import SegmentationResult
from src.coaching_engine.models import CoachingRecommendation, CoachingReport
from src.coaching_viewer.app import (
    format_details_text,
    phase_color,
    steps_for_phase,
)
from src.coaching_viewer.joint_map import highlight_joints, resolve_highlight_markers
from src.coaching_viewer.markers_to_skeleton import ace_markers_to_skeleton
from src.coaching_viewer.phase_map import coaching_phase_to_dtw
from src.coaching_viewer.pipeline import build_session_from_parts
from src.coaching_viewer.recommendations import top_recommendations
from src.coaching_viewer.scalars import extract_feature_scalars
from src.dtw.alignment import ServeComparison, SynchronizedStep
from src.scoring_engine.feature_scorer import FEATURE_SCORERS
from src.scoring_report_reader.report import FeatureSummary, ScoringReport


def test_phase_map():
    assert coaching_phase_to_dtw("Loading") == "Loading"
    assert coaching_phase_to_dtw("Deceleration") == "Deceleration_Finish"
    assert coaching_phase_to_dtw("Finish") == "Deceleration_Finish"
    assert coaching_phase_to_dtw("Start_Stance") is None


def test_joint_map_and_pelvis_alias():
    assert highlight_joints("Knee Flexion") == ("left_knee", "right_knee")
    assert highlight_joints("Contact Height") == (
        "right_shoulder",
        "right_elbow",
        "right_hand",
    )
    markers = {"left_hip": {}, "right_hip": {}, "chest": {}}
    assert set(resolve_highlight_markers(("pelvis", "chest"), markers)) == {
        "left_hip",
        "right_hip",
        "chest",
    }


def test_top_recommendations():
    def rec(name: str) -> CoachingRecommendation:
        return CoachingRecommendation(
            feature=name, phase="Cocking", priority="High", correction="x"
        )

    report = CoachingReport(
        overall_score=70.0,
        primary_recommendation=rec("A"),
        secondary_recommendations=(rec("B"), rec("C"), rec("D")),
        strengths=(),
        warnings=(),
    )
    assert [r.feature for r in top_recommendations(report)] == ["A", "B", "C"]


def test_markers_to_skeleton_y_up():
    n = 5
    frames = np.arange(1, n + 1)
    markers: dict = {"frames": frames}
    for name in (
        "head",
        "chest",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_hand",
        "right_hand",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_foot",
        "right_foot",
    ):
        markers[name] = {
            "TX": np.zeros(n),
            "TY": np.zeros(n),
            "TZ": np.linspace(0, 100, n),
        }
    markers["right_hand"]["TZ"][-1] = 200.0
    seq = ace_markers_to_skeleton(markers)
    assert "thorax" in seq.joint_names
    assert "right_wrist" in seq.joint_names
    assert "pelvis" in seq.joint_names
    assert float(seq.joint("right_wrist")[-1, 1]) == 200.0


def _synthetic_serve(n: int = 60) -> tuple[dict, SegmentationResult]:
    frames = np.arange(1, n + 1)
    t = np.linspace(0, 1, n)
    markers: dict = {"frames": frames}
    for name in (
        "head",
        "chest",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_hand",
        "right_hand",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_foot",
        "right_foot",
    ):
        markers[name] = {
            "TX": np.zeros(n) + (10.0 if "right" in name else -10.0),
            "TY": np.zeros(n),
            "TZ": 1000.0 + 200.0 * np.sin(2 * np.pi * t),
        }
    markers["left_knee"]["TZ"][20:30] = 700.0
    markers["right_knee"]["TZ"][20:30] = 700.0
    markers["right_hand"]["TZ"][40] = 2500.0
    markers["right_hand"]["TY"][40] = 400.0

    phases = {
        "Start_Stance": (1, 5),
        "Release": (5, 10),
        "Loading": (10, 25),
        "Cocking": (25, 35),
        "Acceleration": (35, 40),
        "Contact": (40, 42),
        "Deceleration_Finish": (42, 60),
    }
    events = {"contact": 40, "finish": 60}
    seg = SegmentationResult(
        phases=phases,
        events=events,
        event_confidence={k: 1.0 for k in events},
        signals={},
        frames=frames,
        event_indices={"contact": 39, "finish": 59},
        schema_version=2,
    )
    return markers, seg


def test_extract_returns_all_feature_scorer_keys():
    markers, seg = _synthetic_serve()
    values = extract_feature_scalars(markers, seg)
    assert set(values.keys()) == set(FEATURE_SCORERS.keys())
    assert all(np.isfinite(v) for v in values.values())


def test_build_session_from_parts():
    scoring = ScoringReport(
        overall_score=70.0,
        overall_grade="C",
        phase_summaries=(),
        feature_summaries=(
            FeatureSummary(name="Knee Flexion", score=50.0, impact_score=0.9, phase="Loading"),
        ),
    )
    coaching = CoachingReport(
        overall_score=70.0,
        primary_recommendation=CoachingRecommendation(
            feature="Knee Flexion",
            phase="Loading",
            priority="High",
            correction="Bend more",
        ),
        secondary_recommendations=(),
        strengths=(),
        warnings=(),
    )
    empty_seg = SegmentationResult(
        phases={},
        events={},
        event_confidence={},
        signals={},
        frames=np.array([1]),
    )
    comparison = ServeComparison("ref", "player", empty_seg, empty_seg)
    session = build_session_from_parts(
        markers_player={"frames": np.array([1])},
        markers_reference={"frames": np.array([1])},
        comparison=comparison,
        scoring_report=scoring,
        coaching_report=coaching,
    )
    assert session.feature_by_name["Knee Flexion"].score == 50.0
    assert session.recommendations[0].feature == "Knee Flexion"


def test_app_helpers():
    assert phase_color("Deceleration_Finish")
    assert phase_color("Loading")
    steps = [
        SynchronizedStep("Loading", 0, 0, 1, 1, 0, 0),
        SynchronizedStep("Cocking", 1, 1, 2, 2, 1, 0),
        SynchronizedStep("Cocking", 2, 2, 3, 3, 2, 1),
    ]
    assert [s.step_index for s in steps_for_phase(steps, "Cocking")] == [1, 2]


def test_details_format():
    rec = CoachingRecommendation(
        feature="Knee Flexion",
        phase="Loading",
        priority="High",
        correction="Bend more",
        coach_quotes=("Load.",),
        practice_drills=("Squat toss.",),
        direction="too_low",
    )
    feature = FeatureSummary(
        name="Knee Flexion",
        score=40.0,
        impact_score=0.9,
        metadata={
            "player_value": 70.0,
            "reference_value": 95.0,
            "difference": -25.0,
            "unit": "deg",
            "confidence": 0.9,
        },
    )
    text = format_details_text(rec, feature)
    assert "Bend more" in text
    assert "70.0" in text
    assert "unavailable" in format_details_text(rec, None).lower()


def test_cli_no_animate(capsys):
    from src.coaching_viewer.cli import main

    session = MagicMock()
    session.coaching_report.overall_score = 72.5
    session.coaching_report.warnings = ()
    session.recommendations = []
    session.scoring_report.overall_grade = "C"
    with (
        patch("src.coaching_viewer.cli.resolve_serve_path", side_effect=lambda x: x),
        patch("src.coaching_viewer.cli.run", return_value=session) as run_mock,
        patch("src.coaching_viewer.cli.run_coaching_viewer_app") as app_mock,
    ):
        code = main(["player", "reference", "--no-animate"])
    assert code == 0
    run_mock.assert_called_once()
    app_mock.assert_not_called()
    assert "72.5" in capsys.readouterr().out
