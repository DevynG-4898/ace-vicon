from __future__ import annotations

from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")


def find_videos(video_dir: str | Path = "2d_video") -> list[Path]:
    root = Path(video_dir)
    if not root.is_dir():
        return []
    return sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS],
        key=lambda p: p.stat().st_mtime,
    )


def latest_video(video_dir: str | Path = "2d_video") -> Path:
    videos = find_videos(video_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found in {video_dir}. Add an .mp4 file such as 2d_video/serve.mp4.")
    return videos[-1]


def output_dir_for_video(video_path: str | Path, output_root: str | Path) -> Path:
    return Path(output_root) / Path(video_path).stem
