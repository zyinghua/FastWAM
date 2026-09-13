#!/usr/bin/env python3
"""Plot recorded RoboTwin second-difference profiles; requires NumPy/Matplotlib."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.evaluation.smoothness.plotting import main


if __name__ == "__main__":
    raise SystemExit(main())
