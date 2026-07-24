"""Default Knowledge Library entries (tennis serve coaching copy).

Adding a new feature to the product only requires appending an entry here.
"""

from __future__ import annotations

from src.knowledge_library.models import KnowledgeEntry

DEFAULT_ENTRIES: tuple[KnowledgeEntry, ...] = (
    # Loading
    KnowledgeEntry(
        feature="Knee Flexion",
        phase="Loading",
        too_low="Bend your knees more during the loading phase.",
        too_high="Reduce knee bend during the loading phase.",
    ),
    KnowledgeEntry(
        feature="Hip Flexion",
        phase="Loading",
        too_low="Bend more at the hips during the loading phase.",
        too_high="Reduce hip bend during the loading phase.",
    ),
    KnowledgeEntry(
        feature="Shoulder Tilt",
        phase="Loading",
        too_low="Increase shoulder tilt during the loading phase.",
        too_high="Reduce shoulder tilt during the loading phase.",
    ),
    KnowledgeEntry(
        feature="Toss Arm Extension",
        phase="Loading",
        too_low="Extend your tossing arm more during the loading phase.",
        too_high="Lower the tossing arm slightly during the loading phase.",
        practice_drills=("Toss the ball and let it fall before swinging.",),
    ),
    KnowledgeEntry(
        feature="Center of Mass",
        phase="Loading",
        too_low="Lower your center of mass more during the loading phase.",
        too_high="Raise your center of mass slightly during the loading phase.",
        coach_quotes=("You load.", "You take energy."),
        practice_drills=(
            "Separate the loading and acceleration into two distinct movements.",
            'Feel "one... two..." instead of one continuous motion.',
        ),
    ),
    KnowledgeEntry(
        feature="Trunk Rotation",
        phase="Loading",
        too_low="Rotate your torso more during the loading phase.",
        too_high="Reduce torso rotation during the loading phase.",
        coach_quotes=("This shoulder is going to be replaced by the other one.",),
        practice_drills=("Practice the serving motion without a ball.",),
    ),
    KnowledgeEntry(
        feature="Pelvis Rotation",
        phase="Loading",
        too_low="Rotate your hips more during the loading phase.",
        too_high="Reduce hip rotation during the loading phase.",
    ),
    # Cocking
    KnowledgeEntry(
        feature="Right Elbow Flexion",
        phase="Cocking",
        too_low="Bend your right elbow more during the cocking phase.",
        too_high="Reduce right elbow bend during the cocking phase.",
        coach_quotes=(
            "Let the elbow fold naturally.",
            "Don't keep the arm too straight.",
            "Relax into the trophy position.",
        ),
        practice_drills=(
            "Serve without a ball.",
            "Start directly from the trophy position.",
            "Hold the trophy position before accelerating.",
        ),
    ),
    KnowledgeEntry(
        feature="Left Elbow Flexion",
        phase="Cocking",
        too_low="Bend your left elbow more during the cocking phase.",
        too_high="Reduce left elbow bend during the cocking phase.",
    ),
    KnowledgeEntry(
        feature="Shoulder External Rotation",
        phase="Cocking",
        too_low="Increase shoulder external rotation before acceleration.",
        too_high="Reduce shoulder external rotation before acceleration.",
        coach_quotes=("Load the shoulder.", "Stay relaxed and let it rotate."),
        practice_drills=(
            "Let the racket drop before accelerating.",
            "Keep the shoulder relaxed during the loading phase.",
        ),
    ),
    KnowledgeEntry(
        feature="Forearm Angle",
        phase="Cocking",
        too_low="Increase the forearm angle during the cocking phase.",
        too_high="Reduce the forearm angle during the cocking phase.",
    ),
    # Acceleration
    KnowledgeEntry(
        feature="Shoulder Internal Rotation",
        phase="Acceleration",
        too_low="Increase shoulder internal rotation during acceleration.",
        too_high="Reduce shoulder internal rotation during acceleration.",
    ),
    KnowledgeEntry(
        feature="Right Elbow Extension",
        phase="Acceleration",
        too_low="Extend your right elbow more during acceleration.",
        too_high="Reduce right elbow extension during acceleration.",
        coach_quotes=("The arm follows.", "Don't accelerate with the arm."),
        practice_drills=(
            "Throw a tennis ball using only the wrist.",
            "Swing using only the wrist.",
            "Accelerate with the wrist and let the arm follow.",
            "Start halfway through the serve and accelerate only with the wrist.",
        ),
    ),
    KnowledgeEntry(
        feature="Left Elbow Extension",
        phase="Acceleration",
        too_low="Extend your left elbow more during acceleration.",
        too_high="Reduce left elbow extension during acceleration.",
    ),
    KnowledgeEntry(
        feature="Trunk Rotation Velocity",
        phase="Acceleration",
        too_low="Rotate your torso faster during acceleration.",
        too_high="Slow torso rotation slightly during acceleration.",
        coach_quotes=("Push toward the target.", "This is where you get the power."),
        practice_drills=("Push toward the target instead of pushing upward.",),
    ),
    KnowledgeEntry(
        feature="Hip Rotation Velocity",
        phase="Acceleration",
        too_low="Rotate your hips faster during acceleration.",
        too_high="Slow hip rotation slightly during acceleration.",
        coach_quotes=("This is where you get the power.",),
    ),
    # Contact
    KnowledgeEntry(
        feature="Contact Height",
        phase="Contact",
        too_low="Contact the ball higher.",
        too_high="Lower your contact point slightly.",
        coach_quotes=("Hit as high as possible.", "Hit it as close to the net as possible."),
        practice_drills=("Focus on contacting the ball at maximum reach.",),
    ),
    KnowledgeEntry(
        feature="Contact Position",
        phase="Contact",
        too_low="Contact the ball farther in front of your body.",
        too_high="Contact the ball closer to your body.",
        coach_quotes=(
            "Everything in front.",
            "Touch in front.",
            "Hit in front.",
            "Keep the ball in front of you.",
        ),
        practice_drills=(
            "Let the toss fall before swinging.",
            "Keep the toss in front.",
            "Aim every serve toward the T.",
        ),
    ),
    KnowledgeEntry(
        feature="Arm Extension",
        phase="Contact",
        too_low="Extend your hitting arm more at contact.",
        too_high="Reduce arm extension slightly at contact.",
        coach_quotes=("Reach up to the ball.",),
    ),
    KnowledgeEntry(
        feature="Body Alignment",
        phase="Contact",
        too_low="Improve body alignment at contact.",
        too_high="Reduce body rotation at contact.",
        coach_quotes=("Stay in front.", "Don't turn again."),
        practice_drills=("Keep the head and chest facing forward throughout the serve.",),
    ),
    # Deceleration
    KnowledgeEntry(
        feature="Follow Through",
        phase="Deceleration",
        too_low="Continue the follow-through further after contact.",
        too_high="Shorten the follow-through slightly.",
        coach_quotes=("Let the racket continue.", "Let it go all the way through."),
        practice_drills=("Allow the racket to continue naturally after contact.",),
    ),
    KnowledgeEntry(
        feature="Shoulder Deceleration",
        phase="Deceleration",
        too_low="Slow the shoulder more gradually after contact.",
        too_high="Avoid slowing the shoulder too early.",
    ),
    KnowledgeEntry(
        feature="Trunk Flexion",
        phase="Deceleration",
        too_low="Bend forward more during deceleration.",
        too_high="Reduce forward trunk bend during deceleration.",
    ),
    # Finish
    KnowledgeEntry(
        feature="Balance",
        phase="Finish",
        too_low="Maintain better balance after the serve.",
        too_high="Reduce unnecessary body movement after the serve.",
        coach_quotes=("Stay balanced.",),
        practice_drills=("Stay balanced and hold your finish.",),
    ),
    KnowledgeEntry(
        feature="Weight Transfer",
        phase="Finish",
        too_low="Transfer more weight onto your front foot after contact.",
        too_high="Reduce excessive forward weight transfer after contact.",
        coach_quotes=("Lean into it.",),
        practice_drills=("Avoid letting your body weight fall backward after contact.",),
    ),
    KnowledgeEntry(
        feature="Recovery Position",
        phase="Finish",
        too_low="Recover to a ready position more quickly.",
        too_high="Delay your recovery slightly after the follow-through.",
    ),
)
