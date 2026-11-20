"""Canonical filesystem paths shared by runtime modules."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DIVERGENCE_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
