"""JSON-safe conversion helpers shared by runtime persistence paths.

Non-finite numbers are represented as JSON null rather than non-standard
NaN/Infinity tokens. Unknown objects are bounded strings, and recursive
containers are depth-limited to keep diagnostics from breaking persistence.
"""
from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from typing import Any


def json_safe(value: Any, *, max_depth: int = 8, _depth: int = 0) -> Any:
    if _depth > max_depth:
        return "<max-depth>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item, max_depth=max_depth, _depth=_depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item, max_depth=max_depth, _depth=_depth + 1) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return {
                field.name: json_safe(getattr(value, field.name), max_depth=max_depth, _depth=_depth + 1)
                for field in fields(value)
            }
        except Exception:
            pass
    if hasattr(value, "_asdict"):
        try:
            return json_safe(value._asdict(), max_depth=max_depth, _depth=_depth + 1)
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return json_safe(value.item(), max_depth=max_depth, _depth=_depth + 1)
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    # Handle scalar NA sentinels without imposing pandas on every importer.
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    return str(value)[:500]
