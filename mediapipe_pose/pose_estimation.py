"""
pose_detection_with_angles.py
Image pipeline for the Tennis Serve Analyzer.
Accepts an image path, corrects orientation, runs pose detection,
computes serve angles, and saves an annotated output image.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import mediapipe as mp
from datetime import datetime

from mediapipe_pose.utils import create_detector, get_image_rotation, draw_landmarks
from serve_angles import TennisServeAnalyzer


def analyze_image(image_path: str, hand: str = 'right') -> dict | None:
    """
    Full image analysis pipeline.

    Args:
        image_path: Path to the input image.
        hand:       'right' or 'left' serving hand.

    Returns:
        Dict of angles, or None if no pose was detected.
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

    print(f"Detected {len(result.pose_landmarks)} pose(s).")

    # ── calculate angles ──────────────────────────────────────────────────
    analyzer = TennisServeAnalyzer(result.pose_landmarks[0], hand=hand)
    angles   = analyzer.print_analysis()

    # ── annotate & save ───────────────────────────────────────────────────
    annotated = draw_landmarks(img, result.pose_landmarks[0], angles)

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