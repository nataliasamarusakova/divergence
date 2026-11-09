"""Canonical Divergence application version."""
from __future__ import annotations

import re
from pathlib import Path

_VERSION_FILE = Path(__file__).resolve().parents[1] / "VERSION"
try:
    APP_VERSION = _VERSION_FILE.read_text(encoding="utf-8").strip()
except OSError as exc:
    raise RuntimeError(f"Cannot read canonical version file: {_VERSION_FILE}") from exc

if not re.fullmatch(r"divergence-v\d+\.\d+", APP_VERSION):
    raise RuntimeError(
        f"Invalid canonical Divergence version {APP_VERSION!r} in {_VERSION_FILE}; "
        "expected divergence-vMAJOR.MINOR"
    )
