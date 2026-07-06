"""
ACE Tennis Serve Grader — Phase-Based
--------------------------------------
Grades a customer's serve phase by phase, using the 8-phase legacy
segmentation from serve_segmentation.py / result.py:

    Start_Stance, Release, Loading, Cocking,
    Acceleration, Contact, Deceleration, Finish

Your partner's code computes joint angles per phase (from segmented
Vicon marker data) and passes them in as a dict. This module compares
those angles against a professional reference, scores each phase, and
produces feedback.

Expected input shape from your partner:

    customer_data = {
        "Start_Stance":  {"elbow": 170.0, "shoulder": 20.0, "hip": 175.0, "knee": 170.0, "wrist": 180.0},
        "Release":       {"elbow": 160.0, "shoulder": 45.0, "hip": 170.0, "knee": 165.0, "wrist": 175.0},
        "Loading":       {"elbow": 130.0, "shoulder": 90.0, "hip": 160.0, "knee": 130.0, "wrist": 170.0},
        "Cocking":       {"elbow": 95.0,  "shoulder": 150.0,"hip": 150.0, "knee": 140.0, "wrist": 160.0},
        "Acceleration":  {"elbow": 120.0, "shoulder": 120.0,"hip": 140.0, "knee": 150.0, "wrist": 150.0},
        "Contact":       {"elbow": 175.0, "shoulder": 100.0,"hip": 130.0, "knee": 160.0, "wrist": 178.0},
        "Deceleration":  {"elbow": 140.0, "shoulder": 60.0, "hip": 120.0, "knee": 155.0, "wrist": 140.0},
        "Finish":        {"elbow": 90.0,  "shoulder": 30.0, "hip": 110.0, "knee": 165.0, "wrist": 100.0},
    }

Any joint missing for a phase is simply skipped in scoring — no need
to provide all 5 angles for every phase if your partner doesn't have them.

File location: src/tennis_serve_grader.py
(sibling to src/serve_analysis.py and src/serve_segmentation.py)
"""

from segmentation.result import PHASE_NAMES

# ─────────────────────────────────────────────
# 1. PHASE DEFINITIONS
#    Pulled directly from Max's segmentation.result module so this
#    grader always stays in sync with the actual phase names used
#    by the segmentation pipeline — no duplicated/hardcoded tuple.
# ─────────────────────────────────────────────

# How much each phase contributes to the overall serve grade.
# Cocking, Acceleration, and Contact matter most biomechanically.
PHASE_WEIGHTS = {
    "Start_Stance":  0.05,
    "Release":       0.10,
    "Loading":       0.15,
    "Cocking":       0.20,
    "Acceleration":  0.20,
    "Contact":       0.20,
    "Deceleration":  0.05,
    "Finish":        0.05,
}

JOINTS = ("elbow", "shoulder", "hip", "knee", "wrist")

# ─────────────────────────────────────────────
# 2. PROFESSIONAL REFERENCE
#    Replace with real pro angles per phase once available.
# ─────────────────────────────────────────────

PRO_REFERENCE = {
    "Start_Stance":  {"elbow": 170.0, "shoulder": 20.0,  "hip": 175.0, "knee": 170.0, "wrist": 180.0},
    "Release":       {"elbow": 160.0, "shoulder": 45.0,  "hip": 170.0, "knee": 165.0, "wrist": 175.0},
    "Loading":       {"elbow": 130.0, "shoulder": 90.0,  "hip": 160.0, "knee": 130.0, "wrist": 170.0},
    "Cocking":       {"elbow": 95.0,  "shoulder": 150.0, "hip": 150.0, "knee": 140.0, "wrist": 160.0},
    "Acceleration":  {"elbow": 120.0, "shoulder": 120.0, "hip": 140.0, "knee": 150.0, "wrist": 150.0},
    "Contact":       {"elbow": 175.0, "shoulder": 100.0, "hip": 130.0, "knee": 160.0, "wrist": 178.0},
    "Deceleration":  {"elbow": 140.0, "shoulder": 60.0,  "hip": 120.0, "knee": 155.0, "wrist": 140.0},
    "Finish":        {"elbow": 90.0,  "shoulder": 30.0,  "hip": 110.0, "knee": 165.0, "wrist": 100.0},
}

# ─────────────────────────────────────────────
# 3. SCORING THRESHOLDS (degrees difference)
# ─────────────────────────────────────────────

ANGLE_THRESHOLDS = [
    (5,   95, "Excellent"),
    (15,  80, "Good"),
    (30,  60, "Fair"),
    (999, 35, "Poor"),
]


def score_angle(diff_deg: float) -> tuple:
    for threshold, points, tier in ANGLE_THRESHOLDS:
        if diff_deg < threshold:
            return points, tier
    return 35, "Poor"


# ─────────────────────────────────────────────
# 4. PHASE-SPECIFIC FEEDBACK
#    Generic per-joint feedback, applied within each phase context.
# ─────────────────────────────────────────────

PHASE_FOCUS = {
    "Start_Stance":  "balanced, stable base stance",
    "Release":       "smooth toss release and early weight transfer",
    "Loading":       "leg loading and shoulder turn",
    "Cocking":       "maximum shoulder external rotation (the power-loading position)",
    "Acceleration":  "rapid arm acceleration toward the ball",
    "Contact":       "full extension at ball contact",
    "Deceleration":  "controlled arm deceleration after contact",
    "Finish":        "balanced landing and follow-through",
}

JOINT_FEEDBACK = {
    "Excellent": "{joint} angle is right in line with the pro at this phase.",
    "Good":      "{joint} angle is slightly off — minor adjustment needed.",
    "Fair":      "{joint} angle deviates noticeably here, affecting {focus}.",
    "Poor":      "{joint} angle is significantly off, which is likely hurting {focus}.",
}


# ─────────────────────────────────────────────
# 5. GRADING ENGINE
# ─────────────────────────────────────────────

def grade_phase(phase_name: str, customer_angles: dict, pro_angles: dict) -> dict:
    """Score a single phase by comparing each available joint angle."""
    joint_results = {}

    for joint in JOINTS:
        c_val = customer_angles.get(joint)
        p_val = pro_angles.get(joint)
        if c_val is None or p_val is None:
            continue

        diff = abs(c_val - p_val)
        score, tier = score_angle(diff)
        feedback = JOINT_FEEDBACK[tier].format(
            joint=joint.capitalize(), focus=PHASE_FOCUS[phase_name]
        )

        joint_results[joint] = {
            "customer": c_val,
            "pro": p_val,
            "diff": round(diff, 1),
            "score": score,
            "tier": tier,
            "feedback": feedback,
        }

    if not joint_results:
        return {
            "phase_score": None,
            "joints": {},
            "summary": f"No joint data provided for {phase_name}.",
        }

    phase_score = round(
        sum(j["score"] for j in joint_results.values()) / len(joint_results), 1
    )

    return {
        "phase_score": phase_score,
        "joints": joint_results,
        "summary": _phase_summary(phase_name, phase_score),
    }


def _phase_summary(phase_name: str, score: float) -> str:
    if score >= 90:
        return f"{phase_name.replace('_', ' ')}: Excellent — matches pro mechanics closely."
    elif score >= 75:
        return f"{phase_name.replace('_', ' ')}: Good — solid form with minor refinements needed."
    elif score >= 60:
        return f"{phase_name.replace('_', ' ')}: Fair — noticeable technique gaps in this phase."
    else:
        return f"{phase_name.replace('_', ' ')}: Poor — this phase needs focused practice."


def grade_serve(customer_data: dict, pro_data: dict = None) -> dict:
    """
    Grade a full serve across all 8 phases.

    Parameters
    ----------
    customer_data : dict — { phase_name: {joint: angle, ...}, ... }
    pro_data      : dict — optional override of PRO_REFERENCE

    Returns
    -------
    dict with per-phase results, overall score, and overall grade
    """
    if pro_data is None:
        pro_data = PRO_REFERENCE

    results = {"phases": {}}
    weighted_total = 0.0
    weight_used = 0.0

    for phase in PHASE_NAMES:
        customer_angles = customer_data.get(phase, {})
        pro_angles = pro_data.get(phase, {})

        phase_result = grade_phase(phase, customer_angles, pro_angles)
        results["phases"][phase] = phase_result

        if phase_result["phase_score"] is not None:
            w = PHASE_WEIGHTS[phase]
            weighted_total += phase_result["phase_score"] * w
            weight_used += w

    if weight_used == 0:
        results["overall_score"] = None
        results["overall_grade"] = "No data provided."
        return results

    overall = round(weighted_total / weight_used, 1)
    results["overall_score"] = overall

    if overall >= 90:
        results["overall_grade"] = "A — Pro-level serve"
    elif overall >= 75:
        results["overall_grade"] = "B — Strong serve, minor adjustments needed"
    elif overall >= 60:
        results["overall_grade"] = "C — Developing serve, focused practice recommended"
    else:
        results["overall_grade"] = "D — Fundamentals need significant work"

    return results


# ─────────────────────────────────────────────
# 6. REPORT PRINTER
# ─────────────────────────────────────────────

TIER_ICON = {"Excellent": "✓", "Good": "~", "Fair": "!", "Poor": "✗"}


def print_report(results: dict) -> None:
    sep = "─" * 64
    print(f"\n{'ACE TENNIS SERVE — PHASE-BY-PHASE REPORT':^64}")
    print(sep)

    for phase in PHASE_NAMES:
        phase_result = results["phases"].get(phase)
        if not phase_result or phase_result["phase_score"] is None:
            print(f"\n  {phase.replace('_', ' ')}  — no data")
            continue

        weight_pct = int(PHASE_WEIGHTS[phase] * 100)
        print(f"\n  {phase.replace('_', ' ')}  ({weight_pct}% of grade)")
        print(f"  Phase score: {phase_result['phase_score']}/100")

        for joint, j in phase_result["joints"].items():
            icon = TIER_ICON[j["tier"]]
            print(f"    [{icon}] {joint.capitalize():9s} {j['customer']:>6.1f}°  "
                  f"(pro: {j['pro']:.1f}°, diff: {j['diff']}°)  — {j['tier']}")

        print(f"  → {phase_result['summary']}")

    print(f"\n{sep}")
    print(f"  OVERALL SCORE : {results['overall_score']}/100")
    print(f"  GRADE         : {results['overall_grade']}")
    print(sep)


# ─────────────────────────────────────────────
# 7. EXAMPLE — replace with real values from your partner
# ─────────────────────────────────────────────

if __name__ == "__main__":

    customer_data = {
        "Start_Stance":  {"elbow": 168.0, "shoulder": 22.0, "hip": 172.0, "knee": 168.0, "wrist": 178.0},
        "Release":       {"elbow": 155.0, "shoulder": 50.0, "hip": 168.0, "knee": 160.0, "wrist": 172.0},
        "Loading":       {"elbow": 125.0, "shoulder": 95.0, "hip": 155.0, "knee": 122.0, "wrist": 165.0},
        "Cocking":       {"elbow": 110.0, "shoulder": 132.0,"hip": 148.0, "knee": 138.0, "wrist": 158.0},
        "Acceleration":  {"elbow": 128.0, "shoulder": 110.0,"hip": 142.0, "knee": 148.0, "wrist": 148.0},
        "Contact":       {"elbow": 172.0, "shoulder": 96.0, "hip": 128.0, "knee": 158.0, "wrist": 175.0},
        "Deceleration":  {"elbow": 145.0, "shoulder": 65.0, "hip": 118.0, "knee": 150.0, "wrist": 135.0},
        "Finish":        {"elbow": 95.0,  "shoulder": 35.0, "hip": 108.0, "knee": 160.0, "wrist": 105.0},
    }

    results = grade_serve(customer_data)
    print_report(results)