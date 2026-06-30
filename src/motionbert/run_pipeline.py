from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.motionbert.mediapipe_extractor import extract_video  # noqa: E402
from src.motionbert.motionbert_runner import (  # noqa: E402
    DEFAULT_MOTIONBERT_CHECKPOINT,
    DEFAULT_MOTIONBERT_DIR,
    run_motionbert_stage,
)
from src.motionbert.view_3d import run_viewer  # noqa: E402
from src.motionbert.view_ace_animation import run_ace_animation  # noqa: E402
from src.video_discovery import find_videos, latest_video  # noqa: E402


def process_video(
    video_path: str | Path,
    *,
    output_root: str | Path = "generated_motionbert",
    backend: str = "auto",
    motionbert_command: str | None = None,
    pose_model_path: str | Path | None = None,
    motionbert_dir: str | Path = DEFAULT_MOTIONBERT_DIR,
    checkpoint_path: str | Path | None = DEFAULT_MOTIONBERT_CHECKPOINT,
) -> Path:
    output_dir = extract_video(video_path, output_root, pose_model_path=pose_model_path)
    run_motionbert_stage(
        output_dir,
        backend=backend,
        motionbert_command=motionbert_command,
        motionbert_dir=motionbert_dir,
        checkpoint_path=checkpoint_path,
        video_path=video_path,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run video -> MediaPipe -> MotionBERT-style 3D skeleton pipeline.")
    parser.add_argument("--video-dir", default="2d_video", help="Directory to scan for video files.")
    parser.add_argument("--output-root", default="generated_motionbert", help="Directory for generated outputs.")
    parser.add_argument("--view", action="store_true", help="Process the latest video and open the ACE/Vicon animation viewer.")
    parser.add_argument("--raw-view", action="store_true", help="Open the standalone 17-joint 3D viewer instead of the ACE viewer.")
    parser.add_argument("--normalized", action="store_true", help="With --raw-view, open the normalized SkeletonSequence viewer.")
    parser.add_argument("--backend", choices=("auto", "external", "geometric"), default="auto")
    parser.add_argument("--pose-model", default=None, help="Optional MediaPipe PoseLandmarker .task model path.")
    parser.add_argument("--motionbert-dir", default=str(DEFAULT_MOTIONBERT_DIR), help="Official MotionBERT checkout path.")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_MOTIONBERT_CHECKPOINT),
        help="MotionBERT pose3d checkpoint path.",
    )
    parser.add_argument(
        "--motionbert-command",
        default=None,
        help="External MotionBERT command template. Supports {input}, {output}, and {output_dir}.",
    )
    args = parser.parse_args()

    if args.view:
        videos = [latest_video(args.video_dir)]
    else:
        videos = find_videos(args.video_dir)
        if not videos:
            raise FileNotFoundError(f"No videos found in {args.video_dir}. Add an .mp4 file such as 2d_video/serve.mp4.")

    outputs: list[Path] = []
    try:
        for video in videos:
            print(f"Processing {video}")
            outputs.append(
                process_video(
                    video,
                    output_root=args.output_root,
                    backend=args.backend,
                    motionbert_command=args.motionbert_command,
                    pose_model_path=args.pose_model,
                    motionbert_dir=args.motionbert_dir,
                    checkpoint_path=args.checkpoint,
                )
            )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    if args.view and outputs:
        if args.raw_view:
            run_viewer(outputs[-1] / "poses_3d.npy", normalized=args.normalized)
        else:
            run_ace_animation(outputs[-1] / "ace_markers.npz")


if __name__ == "__main__":
    main()
