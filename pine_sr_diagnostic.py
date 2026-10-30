#!/usr/bin/env python3
"""Read-only diagnostic for the visible Ajay R5.41 Pine S/R values.

The diagnostic intentionally uses the exact production S/R provider and symbol
resolver: Binance Spot 1H source candles aggregated into UTC-aligned 3H bars.
This prevents the manual tool from reintroducing a symbol-specific assumption.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from event_engine.sr_context import get_cached_sr_snapshot
from event_engine.sr_context import get_spot_current_price
from event_engine.sr_context import resolve_spot_symbol
from event_engine.sr_context import _build_zones
import event_engine.sr_context as sr_context


def format_price(value: float | None, tick_size: float | None) -> str:
    if value is None:
        return "None"
    if tick_size is not None and tick_size > 0:
        try:
            quantum = Decimal(str(tick_size))
            # Decimal.quantize(tick) controls decimal places, not tick multiples.
            # Round the number of ticks first, then convert back to a price.
            tick_count = (Decimal(str(value)) / quantum).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
            rounded = tick_count * quantum
            places = max(0, -quantum.normalize().as_tuple().exponent)
            return f"{rounded:.{places}f}"
        except (InvalidOperation, ValueError, ZeroDivisionError):
            pass
    text = f"{float(value):.12f}".rstrip("0").rstrip(".")
    return text or "0"


def _source_tick_size(resolved: dict[str, Any]) -> float | None:
    """Read the source Spot tick size without creating another symbol resolver."""
    try:
        # Keep this import private so this diagnostic still treats sr_context as
        # the single source of truth for symbol resolution and transport.
        import event_engine.sr_context as sr
        row = sr._spot_symbol_catalog().get(str(resolved["source_symbol"]).upper())
        for item in (row or {}).get("filters", []):
            if isinstance(item, dict) and item.get("filterType") == "PRICE_FILTER":
                tick = float(item.get("tickSize", 0) or 0)
                if tick > 0:
                    return tick * float(resolved.get("source_price_scale", 1.0))
    except Exception:
        return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Ajay R5.41 visible Pine SR value diagnostic")
    parser.add_argument("--symbol", required=True, help="e.g. BTC-USDT or 1000SHIB")
    parser.add_argument("--limit", type=int, default=1000, help="Binance Spot 1H source bars aggregated into 3H (900-1000)")
    parser.add_argument("--output", default="pine-sr-zones.csv", help="CSV path for calculated Demand/Supply bands")
    args = parser.parse_args()
    if not 900 <= args.limit <= 1000:
        parser.error("--limit must be between 900 and 1000 source 1H bars to reliably build at least 296 closed 3H bars")

    resolved = resolve_spot_symbol(args.symbol)
    state = get_cached_sr_snapshot(args.symbol, limit=args.limit)
    tick_size = _source_tick_size(resolved)
    try:
        spot_price = get_spot_current_price(args.symbol, snapshot=state)
    except Exception as exc:
        spot_price = None
        print(f"SPOT_TICKER_WARNING={exc}")

    print(f"AJAY R5.41 | REQUESTED={resolved['requested_symbol']} | SOURCE=BINANCE_SPOT")
    print(f"SOURCE_SYMBOL={resolved['source_symbol']}")
    print("TIMEFRAME=3h | SOURCE_KLINE_INTERVAL=1h_AGGREGATED")
    print(f"ALIAS_KIND={resolved['source_alias_kind']}")
    print(f"PRICE_SCALE={resolved['source_price_scale']}")
    print(f"VOLUME_SCALE={resolved['source_volume_scale']}")
    print(f"REQUESTED_SOURCE_1H_BARS={args.limit}")
    print(f"PINE_PARAMETERS=rb:{sr_context.SR_RB},prd:{sr_context.SR_PRD},channelW:{sr_context.SR_CHANNEL_W},strengthSR:{sr_context.SR_STRENGTH}")
    print(f"CLOSED_3H_BARS={state.get('closed_candle_count_3h', 'not-reported')}")
    latest_ts = state.get("latest_closed_timestamp")
    if latest_ts is not None:
        print(f"LATEST_CLOSED_3H_OPEN_UTC={datetime.fromtimestamp(int(latest_ts) / 1000, tz=timezone.utc).isoformat()}")
    print(f"LATEST_CLOSED_3H_AGE_MIN={state.get('latest_closed_age_min')}")
    print(f"CURRENT_BINANCE_SPOT_PRICE={format_price(spot_price, tick_size)}")
    print(f"HIGH LEVEL: {format_price(state['highestph'], tick_size)}")
    print("SR LEVELS:")
    for level in state["levels"]:
        print(format_price(float(level), tick_size))
    print(f"LOW LEVEL: {format_price(state['lowestpl'], tick_size)}")
    print(f"CWIDTH: {state.get('cwidth')}")

    zones = _build_zones(state)
    output_path = Path(args.output)
    fieldnames = ["symbol", "source_symbol", "timeframe", "zone_type", "kind", "center", "lower", "upper", "cluster_width", "points", "countpp", "current_spot_price", "distance_to_band_pct", "latest_closed_3h_open_utc"]
    latest_utc = datetime.fromtimestamp(int(latest_ts) / 1000, tz=timezone.utc).isoformat() if latest_ts is not None else ""
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for zone in zones:
            lower, upper = float(zone["lower"]), float(zone["upper"])
            distance = ""
            if spot_price is not None:
                gap = lower - spot_price if spot_price < lower else spot_price - upper if spot_price > upper else 0.0
                distance = round(gap / spot_price * 100.0, 8)
            row = {
                "symbol": resolved["requested_symbol"],
                "source_symbol": resolved["source_symbol"],
                "timeframe": "3h",
                "zone_type": "DEMAND" if zone["kind"] == "L" else "SUPPLY" if zone["kind"] == "H" else "UNKNOWN",
                "kind": zone["kind"],
                "center": zone["center"],
                "lower": lower,
                "upper": upper,
                "cluster_width": zone["cluster_width"],
                "points": zone["points"],
                "countpp": zone["countpp"],
                "current_spot_price": spot_price if spot_price is not None else "",
                "distance_to_band_pct": distance,
                "latest_closed_3h_open_utc": latest_utc,
            }
            writer.writerow(row)
            print(
                f"{row['zone_type']} | center={format_price(zone['center'], tick_size)} "
                f"| band=[{format_price(lower, tick_size)}, {format_price(upper, tick_size)}] "
                f"| points={zone['points']} | countpp={zone['countpp']} "
                f"| distance_pct={distance}"
            )
    print(f"ZONE_COUNT={len(zones)}")
    print(f"ZONES_CSV={output_path}")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(f"## Ajay R5.41 S/R diagnostic — {resolved['requested_symbol']} (3H)\n\n")
            summary.write(f"- Binance Spot source: `{resolved['source_symbol']}`\n")
            summary.write(f"- Closed 3H bars: `{state.get('closed_candle_count_3h', 'not-reported')}`\n")
            summary.write(f"- Latest closed 3H open (UTC): `{latest_utc}`\n")
            summary.write(f"- Current Spot price: `{format_price(spot_price, tick_size)}`\n")
            summary.write(f"- Pine levels: `{', '.join(format_price(float(v), tick_size) for v in state.get('levels', []))}`\n\n")
            summary.write("| Type | Center | Lower | Upper | Points | Distance % |\n|---|---:|---:|---:|---:|---:|\n")
            for zone in zones:
                lower, upper = float(zone["lower"]), float(zone["upper"])
                gap_pct = ""
                if spot_price is not None:
                    gap = lower - spot_price if spot_price < lower else spot_price - upper if spot_price > upper else 0.0
                    gap_pct = f"{gap / spot_price * 100.0:.4f}%"
                kind = "DEMAND" if zone["kind"] == "L" else "SUPPLY" if zone["kind"] == "H" else "UNKNOWN"
                summary.write(f"| {kind} | {format_price(zone['center'], tick_size)} | {format_price(lower, tick_size)} | {format_price(upper, tick_size)} | {zone['points']} | {gap_pct} |\n")
            if not zones:
                summary.write("\nNo accepted Demand/Supply zones were returned by the Pine SR calculation.\n")
            summary.write(f"\nCSV artifact: `{output_path}`\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
