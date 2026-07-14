from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

_DTW_DIR = Path(__file__).resolve().parent.parent.parent / "dtw"
if str(_DTW_DIR) not in sys.path:
    sys.path.insert(0, str(_DTW_DIR))

from load_data import FILENAME_TO_MARKER, load_single_serve  # noqa: E402


def load_serve_from_folder(serve_dir: str | Path) -> dict:
    serve_dir = Path(serve_dir)
    marker_dict = {}
    for csv_path in glob.glob(str(serve_dir / "*.csv")):
        stem = os.path.splitext(os.path.basename(csv_path))[0].lower()
        marker_name = FILENAME_TO_MARKER.get(stem)
        if marker_name:
            marker_dict[marker_name] = csv_path
    return load_single_serve(marker_dict)
