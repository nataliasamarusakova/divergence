from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from math import isfinite
from typing import Any


def _decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        d = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if not d.is_finite():
        return None
    return d


def format_number(value: Any, *, decimals: int = 8) -> str:
    d = _decimal(value)
    if d is None:
        return "—"
    decimals = max(0, int(decimals))
    try:
        quantum = Decimal(1).scaleb(-decimals)
        d = d.quantize(quantum, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return str(value)
    text = format(d, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"-0", "-0.0"}:
        text = "0"
    return text or "0"


def format_price(value: Any, *, decimals: int = 8) -> str:
    return format_number(value, decimals=decimals)


def format_percent(value: Any, *, decimals: int = 2, signed: bool = False) -> str:
    d = _decimal(value)
    if d is None:
        return "—"
    text = format_number(d, decimals=decimals)
    if signed and not text.startswith(("-", "+")):
        text = "+" + text
    return text + "%"


def format_rr(value: Any, *, decimals: int = 2) -> str:
    d = _decimal(value)
    if d is None:
        return "—"
    return format_number(d, decimals=decimals) + "R"


def tp_price_from_pnl(entry_price: Any, direction: str, pnl_pct: Any) -> float | None:
    entry = _decimal(entry_price)
    pnl = _decimal(pnl_pct)
    if entry is None or pnl is None or entry <= 0:
        return None
    d = str(direction).upper()
    if d == "LONG":
        out = entry * (Decimal("1") + pnl / Decimal("100"))
    elif d == "SHORT":
        out = entry * (Decimal("1") - pnl / Decimal("100"))
    else:
        return None
    try:
        result = float(out)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isfinite(result) and result > 0 else None
