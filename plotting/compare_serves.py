#!/usr/bin/env python3
"""Compare two ACE serves with phase-aware DTW — see ``src.dtw.cli``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from src.dtw.cli import main

if __name__ == "__main__":
    main()
