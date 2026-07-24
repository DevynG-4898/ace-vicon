"""Allow ``python -m format player.mp4 reference.mp4``."""

from __future__ import annotations

from format.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
