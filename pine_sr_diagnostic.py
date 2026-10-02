#!/usr/bin/env python3
"""Read-only diagnostic for the visible Ajay R5.41 Pine S/R values.

The diagnostic intentionally uses the exact production S/R provider and symbol
resolver: Binance SPOT 1H. This prevents the manual tool from reintroducing a
symbol-specific assumption that is absent from production.
"""
from __future__ import annotations

import argparse
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

from event_engine.sr_context import get_cached_sr_snapshot
from event_engine.sr_context import resolve_spot_symbol


def format_price(value: float | None, tick_size: float | None) -> str:
    if value is None:
        return "None"
    if tick_size is not None and tick_size > 0:
        try:
            quantum = Decimal(str(tick_size))
            rounded = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP)
            places = max(0, -quantum.as_tuple().exponent)
            return f"{rounded:.{places}f}"
        except (InvalidOperation, ValueError):
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
    parser.add_argument("--limit", type=int, default=1000, help="closed 1H bars, 300-1000")
    args = parser.parse_args()

    resolved = resolve_spot_symbol(args.symbol)
    state = get_cached_sr_snapshot(args.symbol, limit=args.limit)
    tick_size = _source_tick_size(resolved)

    print(f"AJAY R5.41 | REQUESTED={resolved['requested_symbol']} | SOURCE=BINANCE_SPOT")
    print(f"SOURCE_SYMBOL={resolved['source_symbol']}")
    print(f"ALIAS_KIND={resolved['source_alias_kind']}")
    print(f"PRICE_SCALE={resolved['source_price_scale']}")
    print(f"VOLUME_SCALE={resolved['source_volume_scale']}")
    print(f"HIGH LEVEL: {format_price(state['highestph'], tick_size)}")
    print("SR LEVELS:")
    for level in state["levels"]:
        print(format_price(float(level), tick_size))
    print(f"LOW LEVEL: {format_price(state['lowestpl'], tick_size)}")
    print(f"CWIDTH: {state.get('cwidth')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
