#!/usr/bin/env python3
"""Display Ajay R5.41 Demand/Supply and horizontal SR zones only."""
from __future__ import annotations

import argparse
import math
import sys
from decimal import Decimal
from typing import Any

from event_engine.sr_context import get_cached_sr_snapshot
from event_engine.sr_context import resolve_futures_symbol
from event_engine.sr_context import _build_zones

DIAGNOSTIC_SOURCE_1H_LIMIT = 1500


def format_pine_price(value: float | None) -> str:
    """Render Pine-style prices without applying an exchange-tick rounding."""
    if value is None:
        return "None"
    number = float(value)
    if not math.isfinite(number):
        return "None"
    rendered = format(Decimal(str(number)).normalize(), "f")
    return rendered or "0"


def _display_zones(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Combine visible zones from both Pine subsystems, sorted high to low."""
    zones: list[dict[str, Any]] = []

    # Subsystem 1: the production Demand/Supply boxes.
    for raw in _build_zones(state):
        try:
            lower = float(raw["lower"])
            upper = float(raw["upper"])
            level = float(raw.get("center", (lower + upper) / 2.0))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(value) for value in (lower, upper, level)):
            continue
        if lower > upper:
            lower, upper = upper, lower
        kind = str(raw.get("zone_type") or ("SUPPLY" if raw.get("kind") == "H" else "DEMAND")).upper()
        if kind not in {"SUPPLY", "DEMAND"}:
            kind = "DEMAND" if raw.get("kind") == "L" else "SUPPLY"
        zones.append({"kind": kind, "level": level, "lower": lower, "upper": upper})

    # Subsystem 2: horizontal clusters and Pine's High/Low bands.
    horizontal = state.get("horizontal_sr")
    if isinstance(horizontal, dict):
        for record in horizontal.get("level_records", []):
            if not isinstance(record, dict):
                continue
            try:
                level = float(record["level"])
                lower = float(record["band_lower"])
                upper = float(record["band_upper"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(value) for value in (level, lower, upper)):
                continue
            if lower > upper:
                lower, upper = upper, lower
            zones.append({"kind": "HORIZONTAL_SR", "level": level, "lower": lower, "upper": upper})

        half_width_raw = horizontal.get("zone_half_width")
        try:
            half_width = float(half_width_raw)
            if not math.isfinite(half_width) or half_width < 0:
                half_width = None
        except (TypeError, ValueError, OverflowError):
            half_width = None

        for name, key in (("HIGH_LEVEL", "high_level"), ("LOW_LEVEL", "low_level")):
            raw_level = horizontal.get(key)
            try:
                level = float(raw_level)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(level):
                continue
            lower = level - half_width if half_width is not None else level
            upper = level + half_width if half_width is not None else level
            zones.append({"kind": name, "level": level, "lower": lower, "upper": upper})

    # At equal levels, keep Pine's named High label first and Low label last.
    tie_priority = {"HIGH_LEVEL": 2, "LOW_LEVEL": 0}
    zones.sort(
        key=lambda item: (
            item["level"],
            item["upper"],
            tie_priority.get(item["kind"], 1),
            item["kind"],
        ),
        reverse=True,
    )
    return zones


def main() -> int:
    parser = argparse.ArgumentParser(description="Display Ajay R5.41 Pine SR zones")
    parser.add_argument("--symbol", required=True, help="TradingView Binance perpetual ticker, e.g. BTCUSDT.P")
    args = parser.parse_args()

    resolved = resolve_futures_symbol(args.symbol)
    state = get_cached_sr_snapshot(
        args.symbol, limit=DIAGNOSTIC_SOURCE_1H_LIMIT, include_horizontal_sr=True
    )
    if resolved.get("source_symbol") != state.get("source_symbol"):
        raise RuntimeError("Resolved Binance Futures contract does not match the SR snapshot")
    horizontal = state.get("horizontal_sr")
    if not isinstance(horizontal, dict):
        raise RuntimeError("Missing Ajay R5.41 horizontal SR calculation")
    horizontal_status = str(horizontal.get("status") or "UNKNOWN")
    if horizontal_status != "OK":
        raise RuntimeError(f"Insufficient history for Ajay R5.41 horizontal SR: {horizontal_status}")

    zones = _display_zones(state)
    for zone in zones:
        lower = format_pine_price(zone["lower"])
        upper = format_pine_price(zone["upper"])
        level = format_pine_price(zone["level"])
        if zone["kind"] in {"HIGH_LEVEL", "LOW_LEVEL"}:
            print(f"{zone['kind']}={level} | zone=[{lower}, {upper}]")
        else:
            print(f"{zone['kind']} | level={level} | zone=[{lower}, {upper}]")
    if not zones:
        print("NO_ZONES")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
