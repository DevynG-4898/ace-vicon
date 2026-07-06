"""
pose_detection_with_angles.py
Image pipeline for the Tennis Serve Analyzer.
Accepts an image path, corrects orientation, runs pose detection,
computes serve angles, and saves an annotated output image.

UPDATED:
- Angle computation now uses pose_world_landmarks (metric-scale, hip-centered,
  camera-distance invariant) instead of the normalized pose_landmarks.
  Normalized landmarks are still used for drawing the overlay on the image,
  since draw_landmarks needs pixel-space coordinates.
- Angles that could not be computed (landmark visibility below threshold)
  are now written as 'N/A' in the exported text file instead of crashing
  on a float format spec.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import mediapipe as mp
from datetime import datetime

from mediapipe_pose.utils import create_detector, get_image_rotation, draw_landmarks
from serve_angles import TennisServeAnalyzer


def analyze_image(image_path: str, hand: str = 'right',
                   visibility_threshold: float = 0.5) -> dict | None:
    """
    Full image analysis pipeline.

    Args:
        image_path:            Path to the input image.
        hand:                  'right' or 'left' serving hand.
        visibility_threshold:  Minimum MediaPipe landmark visibility required
                                for a landmark to be trusted in angle math.

    Returns:
        Dict of angles (values may be float('nan') where occluded), or None
        if no pose was detected at all.
    """
    print(f"\nAnalyzing image: {image_path}")

    # ── create output folder ──────────────────────────────────────────────
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name  = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(
                    os.path.dirname(os.path.abspath(image_path)),
                    "outputs",
                    f"{base_name}_{ts}"
                 )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output folder: {output_dir}")

    # ── create detector ───────────────────────────────────────────────────
    detector = create_detector(mode='image')

    # ── correct orientation ───────────────────────────────────────────────
    rotation = get_image_rotation(image_path, detector)

    img = cv2.imread(image_path)
    if rotation is not None:
        img = cv2.rotate(img, rotation)

    # ── run pose detection ────────────────────────────────────────────────
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    detector.close()

    if not result.pose_landmarks:
        print("No pose detected in image.")
        return None

    if not result.pose_world_landmarks:
        # Should not normally happen if pose_landmarks exists, but guard anyway.
        print("Pose landmarks found but world landmarks unavailable — cannot "
              "compute metric-scale angles.")
        return None

    print(f"Detected {len(result.pose_landmarks)} pose(s).")

    # ── calculate angles (metric-scale, camera-distance invariant) ────────
    image_landmarks = result.pose_landmarks[0]        # for drawing overlay
    world_landmarks  = result.pose_world_landmarks[0]  # for angle math

    analyzer = TennisServeAnalyzer(world_landmarks, hand=hand,
                                    visibility_threshold=visibility_threshold)
    angles   = analyzer.get_all_angles()
    analyzer.print_analysis(angles)

    # ── annotate & save ───────────────────────────────────────────────────
    annotated = draw_landmarks(img, image_landmarks, angles)

    out_path = os.path.join(output_dir, f"{base_name}_analyzed.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"\nAnnotated image saved: {out_path}")

    cv2.imshow('Tennis Serve Analysis', annotated)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ── export angles to text ─────────────────────────────────────────────
    txt_path = os.path.join(output_dir, f"{base_name}_angles.txt")
    with open(txt_path, 'w') as f:
        f.write("TENNIS SERVE ANGLE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        for name, value in angles.items():
            if value is None or (isinstance(value, float) and value != value):  # NaN check
                f.write(f"{name}: N/A (occluded — below visibility threshold)\n")
            else:
                f.write(f"{name}: {value:.2f}°\n")
    print(f"Angles exported: {txt_path}")

    return angles


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    image_file = sys.argv[1] if len(sys.argv) > 1 else "tennis_serve.jpg"
    hand       = sys.argv[2] if len(sys.argv) > 2 else "right"
    analyze_image(image_file, hand=hand)