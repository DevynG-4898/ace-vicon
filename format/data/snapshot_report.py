"""Adapt Snapshot Comparison Scoring Engine output → canonical ScoringReport.

Does not re-implement scoring math. Consumes ``grade_snapshots.grade_serve``
results and populates the existing ``format.data.ScoringReport`` schema so the
Coaching Engine / ScoringReportReader pipeline can run unchanged.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from src.knowledge_library import KnowledgeLibrary
from src.scoring_engine.result import Direction, FeatureScoreResult
from src.scoring_engine.target_selector import build_feature_summaries, select_coaching_targets
from src.scoring_engine.phase_scorer import score_phases
from src.scoring_report_reader.report import ScoringReport

# Snapshot → serve phase used when rolling up PhaseSummary rows.
# Acceleration has no dedicated snapshot; contact-adjacent racket features
# that map to Acceleration-phase Knowledge Library entries fill that phase.
SNAPSHOT_TO_PHASE: dict[str, str] = {
    "start_pose": "Loading",
    "hand_cross": "Loading",
    "flat_racket_arm": "Loading",
    "peak_racket_arm": "Cocking",
    "contact": "Contact",
    "hand_cross_2": "Deceleration",
    "racket_deceleration": "Deceleration",
    "finish_pose": "Finish",
}

# (snapshot, joint) → Knowledge Library feature name.
# Phase for each feature comes from the Knowledge Library entry.
SNAPSHOT_JOINT_TO_FEATURE: dict[tuple[str, str], str] = {
    # Loading
    ("start_pose", "left_knee"): "Knee Flexion",
    ("start_pose", "right_knee"): "Knee Flexion",
    ("start_pose", "toss_elbow"): "Toss Arm Extension",
    ("start_pose", "racket_shoulder"): "Shoulder Tilt",
    ("start_pose", "racket_elbow"): "Right Elbow Flexion",
    ("hand_cross", "toss_elbow"): "Toss Arm Extension",
    ("hand_cross", "racket_elbow"): "Right Elbow Flexion",
    ("hand_cross", "racket_shoulder"): "Shoulder Tilt",
    ("hand_cross", "left_knee"): "Knee Flexion",
    ("hand_cross", "right_knee"): "Knee Flexion",
    ("flat_racket_arm", "racket_elbow"): "Right Elbow Flexion",
    ("flat_racket_arm", "racket_shoulder"): "Shoulder Tilt",
    ("flat_racket_arm", "toss_elbow"): "Toss Arm Extension",
    ("flat_racket_arm", "left_knee"): "Knee Flexion",
    ("flat_racket_arm", "right_knee"): "Knee Flexion",
    # Cocking
    ("peak_racket_arm", "racket_elbow"): "Right Elbow Flexion",
    ("peak_racket_arm", "racket_shoulder"): "Shoulder External Rotation",
    ("peak_racket_arm", "toss_elbow"): "Left Elbow Flexion",
    ("peak_racket_arm", "left_knee"): "Knee Flexion",
    ("peak_racket_arm", "right_knee"): "Knee Flexion",
    # Contact / Acceleration (KL Acceleration features sampled at contact)
    ("contact", "racket_elbow"): "Right Elbow Extension",
    ("contact", "racket_shoulder"): "Shoulder Internal Rotation",
    ("contact", "toss_elbow"): "Left Elbow Extension",
    ("contact", "left_knee"): "Knee Flexion",
    ("contact", "right_knee"): "Knee Flexion",
    # Deceleration
    ("hand_cross_2", "racket_elbow"): "Follow Through",
    ("hand_cross_2", "racket_shoulder"): "Shoulder Deceleration",
    ("hand_cross_2", "toss_elbow"): "Left Elbow Extension",
    ("hand_cross_2", "left_knee"): "Balance",
    ("hand_cross_2", "right_knee"): "Balance",
    ("racket_deceleration", "racket_elbow"): "Follow Through",
    ("racket_deceleration", "racket_shoulder"): "Shoulder Deceleration",
    ("racket_deceleration", "toss_elbow"): "Follow Through",
    ("racket_deceleration", "left_knee"): "Weight Transfer",
    ("racket_deceleration", "right_knee"): "Weight Transfer",
    # Finish
    ("finish_pose", "racket_elbow"): "Follow Through",
    ("finish_pose", "racket_shoulder"): "Shoulder Deceleration",
    ("finish_pose", "toss_elbow"): "Recovery Position",
    ("finish_pose", "left_knee"): "Balance",
    ("finish_pose", "right_knee"): "Balance",
}


def _load_grade_snapshots_module():
    """Load ``formatdata and render/grade_snapshots.py`` (path has spaces)."""
    path = (
        Path(__file__).resolve().parents[2]
        / "formatdata and render"
        / "grade_snapshots.py"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot Comparison Scoring Engine not found: {path}")
    spec = importlib.util.spec_from_file_location("grade_snapshots", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load grade_snapshots from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("grade_snapshots", module)
    spec.loader.exec_module(module)
    return module


def _direction_from_joint(customer_angle: float, reference_angle: float, tier: str) -> Direction:
    if tier == "Excellent":
        return "acceptable"
    difference = float(customer_angle) - float(reference_angle)
    if difference < 0:
        return "too_low"
    if difference > 0:
        return "too_high"
    return "acceptable"


def _impact_from_score(score: float, direction: Direction) -> float:
    if direction == "acceptable":
        return 0.0
    return round(max(0.0, min(1.0, (100.0 - float(score)) / 100.0)), 4)


def _candidates_from_snapshot_grade(
    results: dict[str, Any],
    *,
    library: KnowledgeLibrary,
) -> list[FeatureScoreResult]:
    """Flatten scored joints into FeatureScoreResult rows (Knowledge Library keys)."""
    candidates: list[FeatureScoreResult] = []
    snapshots = results.get("snapshots") or {}

    for snap_name, snap in snapshots.items():
        if not isinstance(snap, dict):
            continue
        joints = snap.get("joints") or {}
        for joint_name, joint in joints.items():
            if not isinstance(joint, dict):
                continue
            if joint.get("tier") == "No data" or joint.get("score") is None:
                continue
            feature_name = SNAPSHOT_JOINT_TO_FEATURE.get((snap_name, joint_name))
            if feature_name is None or feature_name not in library:
                continue

            entry = library.get(feature_name)
            customer = float(joint["customer_angle"])
            reference = float(joint["reference_angle"])
            score = float(joint["score"])
            tier = str(joint["tier"])
            direction = _direction_from_joint(customer, reference, tier)
            difference = customer - reference
            impact = _impact_from_score(score, direction)

            candidates.append(
                FeatureScoreResult(
                    name=feature_name,
                    phase=entry.phase,
                    score=score,
                    direction=direction,
                    impact_score=impact,
                    player_value=customer,
                    reference_value=reference,
                    difference=difference,
                    unit="deg",
                    confidence=1.0,
                    measurements={
                        "snapshot": snap_name,
                        "joint": joint_name,
                        "tier": tier,
                        "diff_deg": float(joint.get("diff", abs(difference))),
                        "snapshot_phase": SNAPSHOT_TO_PHASE.get(snap_name),
                        "snapshot_score": snap.get("snapshot_score"),
                        "customer_frame": snap.get("customer_frame"),
                        "reference_frame": snap.get("reference_frame"),
                        "magnitude_score": score,
                    },
                )
            )
    return candidates


def _aggregate_by_feature(
    candidates: list[FeatureScoreResult],
) -> tuple[FeatureScoreResult, ...]:
    """One row per Knowledge Library feature — keep the worst (lowest) score."""
    best: dict[str, FeatureScoreResult] = {}
    extras: dict[str, list[dict[str, Any]]] = {}

    for result in candidates:
        extras.setdefault(result.name, []).append(dict(result.measurements))
        current = best.get(result.name)
        if current is None:
            best[result.name] = result
            continue
        # Prefer lower score; tie-break on larger absolute deviation.
        if result.score < current.score or (
            result.score == current.score
            and abs(result.difference) > abs(current.difference)
        ):
            best[result.name] = result

    aggregated: list[FeatureScoreResult] = []
    for name, result in sorted(best.items(), key=lambda item: item[0]):
        measurements = dict(result.measurements)
        measurements["snapshot_observations"] = extras.get(name, [])
        aggregated.append(
            FeatureScoreResult(
                name=result.name,
                phase=result.phase,
                score=result.score,
                direction=result.direction,
                impact_score=result.impact_score,
                player_value=result.player_value,
                reference_value=result.reference_value,
                difference=result.difference,
                unit=result.unit,
                confidence=result.confidence,
                measurements=measurements,
            )
        )
    return tuple(aggregated)


def scoring_report_from_snapshot_grade(
    results: dict[str, Any],
    *,
    max_secondary: int = 2,
    library: KnowledgeLibrary | None = None,
) -> ScoringReport:
    """Build a canonical ScoringReport from ``grade_snapshots.grade_serve`` output."""
    lib = library if library is not None else KnowledgeLibrary.default()
    candidates = _candidates_from_snapshot_grade(results, library=lib)
    feature_results = _aggregate_by_feature(candidates)
    feature_summaries = build_feature_summaries(feature_results)
    phase_summaries = score_phases(feature_results)

    overall_score = results.get("overall_score")
    overall_grade = results.get("overall_grade")
    if overall_score is None:
        if phase_summaries:
            overall_score = round(
                sum(p.score for p in phase_summaries) / len(phase_summaries), 1
            )
        else:
            overall_score = 0.0
    if not overall_grade:
        overall_grade = "No comparable snapshots found."

    primary, secondary = select_coaching_targets(
        feature_summaries, max_secondary=max_secondary
    )

    warnings: list[str] = []
    for snap_name, snap in (results.get("snapshots") or {}).items():
        if isinstance(snap, dict) and snap.get("snapshot_score") is None:
            reason = snap.get("summary") or "unavailable"
            warnings.append(f"snapshot {snap_name}: {reason}")

    return ScoringReport(
        overall_score=float(overall_score),
        overall_grade=str(overall_grade),
        phase_summaries=phase_summaries,
        feature_summaries=feature_summaries,
        warnings=tuple(warnings),
        confidence=0.9,
        metadata={
            "source": "snapshot_comparison_scoring_engine",
            "generator_version": 1,
            "primary_coaching_target": primary,
            "secondary_coaching_targets": list(secondary),
            "customer_racket_side": results.get("customer_racket_side"),
            "reference_racket_side": results.get("reference_racket_side"),
            "snapshot_overall_score": results.get("overall_score"),
            "snapshot_overall_grade": results.get("overall_grade"),
        },
        schema_version=1,
    )


def scoring_report_from_snapshot_paths(
    customer_path: str | Path,
    reference_path: str | Path,
    *,
    max_secondary: int = 2,
) -> ScoringReport:
    """Run Snapshot Comparison Scoring Engine and emit a canonical ScoringReport."""
    grade_snapshots = _load_grade_snapshots_module()
    results = grade_snapshots.grade_serve(str(customer_path), str(reference_path))
    return scoring_report_from_snapshot_grade(results, max_secondary=max_secondary)
