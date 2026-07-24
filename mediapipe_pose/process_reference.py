import sys
from datetime import datetime
from tennis_video_analysis import process_video

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "IMG_5092.mov"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"reference_angles_{ts}.csv"

    process_video(
        input_path=video_path,
        output_path=None,      # or leave empty if you don't want an annotated video
        csv_path=csv_path,
        hand="right",
        frame_step=1
    )

    print(f"\nReference CSV saved: {csv_path}")

if __name__ == "__main__":
    main()