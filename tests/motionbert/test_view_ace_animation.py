import json

from src.motionbert.view_ace_animation import (
    _frame_index_for_elapsed,
    _load_source_fps,
    _timer_interval_ms,
)


def test_load_source_fps_reads_video_metadata_next_to_markers(tmp_path):
    (tmp_path / "video_metadata.json").write_text(json.dumps({"fps": 59.94005994005994}))

    assert _load_source_fps(tmp_path / "ace_markers.npz") == 59.94005994005994


def test_timer_interval_uses_source_video_fps():
    assert _timer_interval_ms(59.94005994005994) == 17


def test_frame_index_for_elapsed_matches_video_duration():
    assert _frame_index_for_elapsed(3.0, fps=60.0, speed=1, n_frames=360) == 180
    assert _frame_index_for_elapsed(6.0, fps=60.0, speed=1, n_frames=360) == 0
