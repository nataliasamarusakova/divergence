#!/usr/bin/env python3
"""Read-only diagnostic for the visible Ajay R5.41 Pine S/R values.

The diagnostic intentionally uses the exact production S/R provider and symbol
resolver: Binance Spot 1H source candles aggregated into UTC-aligned 3H bars.
This prevents the manual tool from reintroducing a symbol-specific assumption.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from event_engine.sr_context import get_cached_sr_snapshot
from event_engine.sr_context import get_spot_current_price
from event_engine.sr_context import resolve_spot_symbol
from event_engine.sr_context import evaluate_sr_directional_zone
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


def _zone_identity(zone: dict[str, Any]) -> tuple[str, float, int, int]:
    return (
        str(zone.get("kind", "")).upper(),
        float(zone.get("center", 0.0) or 0.0),
        int(zone.get("points", 0) or 0),
        int(zone.get("countpp", 0) or 0),
    )


def _nearest_zones(result: dict[str, Any]) -> list[dict[str, Any]]:
    nearest = result.get("nearest_zone")
    if isinstance(nearest, dict):
        return [nearest]
    if isinstance(nearest, list):
        return [zone for zone in nearest if isinstance(zone, dict)]
    return []


def _gate_columns(result: dict[str, Any]) -> dict[str, Any]:
    nearest = _nearest_zones(result)
    selected_zone = nearest[0] if len(nearest) == 1 else None
    allowed = not bool(result.get("reject"))
    return {
        "gate_decision": "ALLOW" if allowed else "REJECT",
        "gate_reason": "ALLOWED" if allowed else str(result.get("reject_reason") or "UNKNOWN_REJECTION"),
        "gate_zone_type": (selected_zone or {}).get("zone_type", "AMBIGUOUS" if nearest else ""),
        "gate_allowed_direction": (selected_zone or {}).get("allowed_direction", ""),
        "gate_distance_pct": result.get("distance_pct") if result.get("distance_pct") is not None else "",
        "gate_nearest_centers": ";".join(str(zone.get("center", "")) for zone in nearest),
    }


def _format_gate_line(direction: str, columns: dict[str, Any], tick_size: float | None) -> str:
    center = columns.get("gate_nearest_centers") or "none"
    if ";" in center:
        center = ",".join(format_price(float(value), tick_size) for value in center.split(";") if value)
    elif center != "none":
        center = format_price(float(center), tick_size)
    distance = columns.get("gate_distance_pct")
    distance_text = f"{float(distance):.6f}%" if distance not in (None, "") else "n/a"
    return (
        f"GATE_{direction}={columns['gate_decision']} "
        f"REASON={columns['gate_reason']} "
        f"ZONE={columns.get('gate_zone_type') or 'none'} "
        f"CENTER={center} DISTANCE={distance_text}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ajay R5.41 visible Pine SR value diagnostic")
    parser.add_argument("--symbol", required=True, help="e.g. BTC-USDT or 1000SHIB")
    parser.add_argument("--limit", type=int, default=1000, help="Must be 1000 Binance Spot 1H source bars to match the production bot")
    parser.add_argument("--output", default="pine-sr-zones.csv", help="CSV path for calculated Demand/Supply bands")
    args = parser.parse_args()
    if args.limit != 1000:
        parser.error("--limit must be exactly 1000 source 1H bars to match the production bot")

    resolved = resolve_spot_symbol(args.symbol)
    state = get_cached_sr_snapshot(args.symbol, limit=args.limit)
    tick_size = _source_tick_size(resolved)
    # Match the bot's fail-closed entry gate: a zone-only report without a live
    # Binance Spot price is not a valid directional-entry diagnostic.
    spot_price = get_spot_current_price(args.symbol, snapshot=state)
    try:
        spot_price = float(spot_price)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Binance Spot ticker price is unavailable; refusing to publish a zone-only diagnostic") from exc
    if not math.isfinite(spot_price) or spot_price <= 0:
        raise RuntimeError("Binance Spot ticker price is unavailable; refusing to publish a zone-only diagnostic")

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
    gate_results = {
        direction: evaluate_sr_directional_zone(state, current_price=float(spot_price), direction=direction)
        for direction in ("LONG", "SHORT")
    }
    gate_columns = {
        direction: _gate_columns(result)
        for direction, result in gate_results.items()
    }
    print(_format_gate_line("LONG", gate_columns["LONG"], tick_size))
    print(_format_gate_line("SHORT", gate_columns["SHORT"], tick_size))

    output_path = Path(args.output)
    gate_fields = [
        f"{prefix}_{field}"
        for prefix in ("long", "short")
        for field in ("gate_decision", "gate_reason", "gate_zone_type", "gate_allowed_direction", "gate_distance_pct", "gate_nearest_centers", "selected")
    ]
    fieldnames = ["symbol", "source_symbol", "timeframe", "zone_type", "kind", "center", "lower", "upper", "cluster_width", "points", "countpp", "current_spot_price", "distance_to_band_pct", "latest_closed_3h_open_utc", *gate_fields]
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
                **{f"long_{key}": value for key, value in gate_columns["LONG"].items()},
                **{f"short_{key}": value for key, value in gate_columns["SHORT"].items()},
            }
            row["long_selected"] = _zone_identity(zone) in {
                _zone_identity(item) for item in _nearest_zones(gate_results["LONG"])
            }
            row["short_selected"] = _zone_identity(zone) in {
                _zone_identity(item) for item in _nearest_zones(gate_results["SHORT"])
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
            summary.write("### Exact bot directional gate\n\n")
            summary.write("| Direction | Decision | Reason | Nearest zone | Allowed direction | Distance |\n|---|---|---|---|---|---:|\n")
            for direction in ("LONG", "SHORT"):
                gate = gate_results[direction]
                nearest = _nearest_zones(gate)
                nearest_label = ", ".join(
                    f"{z.get('zone_type')} {format_price(float(z.get('center', 0)), tick_size)}" for z in nearest
                ) or "None"
                distance_value = gate.get("distance_pct")
                distance_text = "" if distance_value is None else f"{float(distance_value):.6f}%"
                summary.write(
                    f"| {direction} | {'REJECT' if gate.get('reject') else 'ALLOW'} | "
                    f"{gate.get('reject_reason') or 'ALLOWED'} | {nearest_label} | "
                    f"{gate.get('allowed_direction') or ('AMBIGUOUS' if len(nearest) > 1 else '')} | "
                    f"{distance_text} |\n"
                )
            summary.write("\n### Calculated zones\n\n")
            summary.write("| Type | Center | Lower | Upper | Points | Geometric distance % |\n|---|---:|---:|---:|---:|---:|\n")
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
