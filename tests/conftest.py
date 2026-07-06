import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DTW = ROOT / "dtw"
INDIVIDUAL = ROOT / "plotting" / "markers" / "individual"

for p in (str(SRC), str(DTW)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def firstserve_dir():
    return INDIVIDUAL / "firstserve"


@pytest.fixture
def firstserve_dict(firstserve_dir):
    from segmentation.io import load_serve_from_folder

    return load_serve_from_folder(firstserve_dir)


@pytest.fixture
def default_config():
    from segmentation.config import SegmentationConfig

    return SegmentationConfig()


@pytest.fixture
def v2_config():
    from segmentation.config import SegmentationConfig

    return SegmentationConfig(use_legacy_detection=False)
