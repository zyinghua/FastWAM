#!/usr/bin/env python3
"""Measure recorded action boundaries offline; requires NumPy only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.evaluation.smoothness.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
