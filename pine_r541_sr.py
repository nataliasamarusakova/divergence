from __future__ import annotations

"""Python port of the DEMAND / SUPPLY block in the supplied Ajay R5.41 Pine v5.

Only the first Pine subsystem (the swing-based DEMAND and SUPPLY boxes) is
implemented here. The later ``enableSR``/``prd=284`` support-resistance cluster
indicator is intentionally not part of this module.

The engine consumes closed candles in chronological order. Pivot values become
available on the confirmation bar, ATR(50) is Pine-style RMA(True Range), boxes
use 0.25 * ATR by default, and broken zones leave the active set immediately.
"""

from dataclasses import dataclass
from typing import Any
import math


@dataclass(frozen=True)
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def _is_pivot_high(candles: list[Candle], pivot_idx: int, swing_length: int) -> bool:
    """TradingView ta.pivothigh tie rule: left ties allowed, right ties reject."""
    left = pivot_idx - swing_length
    right = pivot_idx + swing_length
    if left < 0 or right >= len(candles):
        return False
    value = float(candles[pivot_idx].high)
    # Older/left bars must not be strictly higher. Equal older highs are allowed.
    if any(float(candles[i].high) > value for i in range(left, pivot_idx)):
        return False
    # Newer/right bars must be strictly lower. An equal newer high wins the tie.
    if any(float(candles[i].high) >= value for i in range(pivot_idx + 1, right + 1)):
        return False
    return True


def _is_pivot_low(candles: list[Candle], pivot_idx: int, swing_length: int) -> bool:
    """TradingView ta.pivotlow tie rule: left ties allowed, right ties reject."""
    left = pivot_idx - swing_length
    right = pivot_idx + swing_length
    if left < 0 or right >= len(candles):
        return False
    value = float(candles[pivot_idx].low)
    # Older/left bars must not be strictly lower. Equal older lows are allowed.
    if any(float(candles[i].low) < value for i in range(left, pivot_idx)):
        return False
    # Newer/right bars must be strictly higher. An equal newer low wins the tie.
    if any(float(candles[i].low) <= value for i in range(pivot_idx + 1, right + 1)):
        return False
    return True


def build_confirmed_pivots(
    candles: list[Candle], swing_length: int = 10
) -> tuple[list[float | None], list[float | None]]:
    """Return pivot series indexed by confirmation bar, matching Pine timing."""
    if swing_length < 1:
        raise ValueError("swing_length must be >= 1")
    count = len(candles)
    pivot_highs: list[float | None] = [None] * count
    pivot_lows: list[float | None] = [None] * count
    for confirmation_idx in range(count):
        pivot_idx = confirmation_idx - swing_length
        if pivot_idx < swing_length:
            continue
        if _is_pivot_high(candles, pivot_idx, swing_length):
            pivot_highs[confirmation_idx] = float(candles[pivot_idx].high)
        if _is_pivot_low(candles, pivot_idx, swing_length):
            pivot_lows[confirmation_idx] = float(candles[pivot_idx].low)
    return pivot_highs, pivot_lows


def calculate_true_range(candles: list[Candle]) -> list[float]:
    """Pine ta.tr(true): first bar is high-low, then gap-aware true range."""
    result: list[float] = []
    previous_close: float | None = None
    for candle in candles:
        high = float(candle.high)
        low = float(candle.low)
        if previous_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - previous_close), abs(low - previous_close))
        result.append(float(tr))
        previous_close = float(candle.close)
    return result


def calculate_atr(candles: list[Candle], length: int = 50) -> list[float | None]:
    """Pine-style ta.atr(length): RMA(True Range), seeded with the first SMA."""
    if length < 1:
        raise ValueError("ATR length must be >= 1")
    true_ranges = calculate_true_range(candles)
    result: list[float | None] = [None] * len(candles)
    if len(true_ranges) < length:
        return result
    seed = sum(true_ranges[:length]) / length
    result[length - 1] = seed
    previous = seed
    for idx in range(length, len(true_ranges)):
        previous = (previous * (length - 1) + true_ranges[idx]) / length
        result[idx] = previous
    return result


def _zone_record(
    *, kind: str, pivot_idx: int, confirm_idx: int, candle: Candle,
    confirmation_candle: Candle, top: float, bottom: float,
    points: int = 1, countpp: int = 1,
) -> dict[str, Any]:
    center = (top + bottom) / 2.0
    zone_type = "SUPPLY" if kind == "H" else "DEMAND"
    return {
        "kind": kind,
        "zone_type": zone_type,
        "pivot_index": int(pivot_idx),
        "confirmation_index": int(confirm_idx),
        "pivot_timestamp": int(candle.ts),
        "created_timestamp": int(confirmation_candle.ts),
        "top": float(top),
        "bottom": float(bottom),
        "upper": float(top),
        "lower": float(bottom),
        "center": float(center),
        "poi": float(center),
        "points": int(points),
        "countpp": int(countpp),
        "cluster_width": float(top - bottom),
        "box_width": float(top - bottom),
        "broken": False,
        "bos_timestamp": None,
    }


def _overlaps_existing_poi(
    new_poi: float, slots: list[dict[str, Any] | None], atr: float
) -> bool:
    threshold = float(atr) * 2.0
    for existing in slots:
        if not isinstance(existing, dict) or existing.get("broken"):
            continue
        poi = float(existing["poi"])
        if poi - threshold <= new_poi <= poi + threshold:
            return True
    return False


def compute_current_sr(
    candles: list[Candle],
    swing_length: int = 10,
    history_to_keep: int = 20,
    box_width: float = 2.5,
    atr_length: int = 50,
) -> dict[str, Any]:
    """Compute the latest active DEMAND/SUPPLY zones from closed OHLCV bars.

    ``history_to_keep`` mirrors the fixed-size Pine arrays. Broken zones are
    represented by ``None`` slots, so they stop participating in overlap checks
    while still preserving the queue's eviction/history semantics.
    """
    if swing_length < 1:
        raise ValueError("swing_length must be >= 1")
    if history_to_keep < 1:
        raise ValueError("history_to_keep must be >= 1")
    if not math.isfinite(float(box_width)) or box_width <= 0:
        raise ValueError("box_width must be finite and > 0")
    if atr_length < 1:
        raise ValueError("atr_length must be >= 1")
    minimum = max(atr_length, 2 * swing_length + 1)
    if len(candles) < minimum:
        raise ValueError(f"Need at least {minimum} candles, got {len(candles)}")
    for candle in candles:
        values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Candles must contain finite OHLCV values")
        if min(float(candle.open), float(candle.high), float(candle.low), float(candle.close)) <= 0:
            raise ValueError("Candle prices must be positive")
        if float(candle.high) < max(float(candle.open), float(candle.close), float(candle.low)):
            raise ValueError("Candle high is inconsistent with OHLC values")
        if float(candle.low) > min(float(candle.open), float(candle.close), float(candle.high)):
            raise ValueError("Candle low is inconsistent with OHLC values")

    pivot_highs, pivot_lows = build_confirmed_pivots(candles, swing_length=swing_length)
    atr_values = calculate_atr(candles, length=atr_length)
    supply_slots: list[dict[str, Any] | None] = [None] * history_to_keep
    demand_slots: list[dict[str, Any] | None] = [None] * history_to_keep
    supply_bos: list[dict[str, Any]] = []
    demand_bos: list[dict[str, Any]] = []
    last_pivot_high = None
    last_pivot_low = None
    last_event_idx = None

    for idx, candle in enumerate(candles):
        atr = atr_values[idx]
        if atr is not None and atr > 0:
            # Pine uses `if not na(swing_high) ... else if not na(swing_low)`.
            # A simultaneous high/low pivot therefore creates Supply only.
            if pivot_highs[idx] is not None:
                pivot_idx = idx - swing_length
                top = float(pivot_highs[idx])
                bottom = top - float(atr) * (float(box_width) / 10.0)
                poi = (top + bottom) / 2.0
                if not _overlaps_existing_poi(poi, supply_slots, float(atr)):
                    zone = _zone_record(
                        kind="H", pivot_idx=pivot_idx, confirm_idx=idx,
                        candle=candles[pivot_idx], confirmation_candle=candle,
                        top=top, bottom=bottom,
                    )
                    supply_slots.insert(0, zone)
                    supply_slots.pop()
                    last_event_idx = idx
                    last_pivot_high = float(pivot_highs[idx])
            elif pivot_lows[idx] is not None:
                pivot_idx = idx - swing_length
                bottom = float(pivot_lows[idx])
                top = bottom + float(atr) * (float(box_width) / 10.0)
                poi = (top + bottom) / 2.0
                if not _overlaps_existing_poi(poi, demand_slots, float(atr)):
                    zone = _zone_record(
                        kind="L", pivot_idx=pivot_idx, confirm_idx=idx,
                        candle=candles[pivot_idx], confirmation_candle=candle,
                        top=top, bottom=bottom,
                    )
                    demand_slots.insert(0, zone)
                    demand_slots.pop()
                    last_event_idx = idx
                    last_pivot_low = float(pivot_lows[idx])

        # Pine checks BOS on every bar using CLOSE, not high/low wicks.
        for slots, bos_records, is_supply in (
            (supply_slots, supply_bos, True),
            (demand_slots, demand_bos, False),
        ):
            for slot_idx, zone in enumerate(slots):
                if not isinstance(zone, dict):
                    continue
                broken = (
                    float(candle.close) >= float(zone["top"])
                    if is_supply else float(candle.close) <= float(zone["bottom"])
                )
                if not broken:
                    continue
                bos_mid = (float(zone["top"]) + float(zone["bottom"])) / 2.0
                bos_records.insert(0, {
                    **zone,
                    "bos_price": bos_mid,
                    "bos_timestamp": int(candle.ts),
                    "broken": True,
                    "bos_type": "SUPPLY_BROKEN" if is_supply else "DEMAND_BROKEN",
                })
                del bos_records[5:]
                # Deleted Pine box IDs remain in its fixed array. The Python slot
                # becomes None so it cannot incorrectly block later valid zones.
                slots[slot_idx] = None

    active_supply = [dict(z) for z in supply_slots if isinstance(z, dict) and not z.get("broken")]
    active_demand = [dict(z) for z in demand_slots if isinstance(z, dict) and not z.get("broken")]
    active_zones = active_supply + active_demand
    latest_atr = atr_values[-1]
    latest_pivot_high_confirmed = pivot_highs[-1]
    latest_pivot_low_confirmed = pivot_lows[-1]
    levels = [float(z["center"]) for z in active_zones]
    return {
        "algorithm": "AJAY_R541_DEMAND_SUPPLY",
        "swing_length": int(swing_length),
        "history_to_keep": int(history_to_keep),
        "box_width": float(box_width),
        "atr_length": int(atr_length),
        "atr": latest_atr,
        "atr_last": latest_atr,
        "cwidth": (float(latest_atr) * float(box_width) / 10.0) if latest_atr is not None else None,
        "zones": active_zones,
        "active_zones": active_zones,
        "supply_zones": active_supply,
        "demand_zones": active_demand,
        "supply_slots": supply_slots,
        "demand_slots": demand_slots,
        "supply_bos": supply_bos,
        "demand_bos": demand_bos,
        "levels": levels,
        "accepted": [
            {"price": z["center"], "kind": z["kind"], "points": z["points"], "countpp": z["countpp"],
             "cluster_width": z["cluster_width"], "top": z["top"], "bottom": z["bottom"]}
            for z in active_zones
        ],
        "highestph": max((float(z["top"]) for z in active_supply), default=None),
        "lowestpl": min((float(z["bottom"]) for z in active_demand), default=None),
        "event_idx": last_event_idx,
        "event_timestamp": candles[last_event_idx].ts if last_event_idx is not None else None,
        "pivot_high_confirmed_at_latest": latest_pivot_high_confirmed,
        "pivot_low_confirmed_at_latest": latest_pivot_low_confirmed,
        "latest_closed_timestamp": int(candles[-1].ts),
        "latest_closed_close": float(candles[-1].close),
        "closed_candle_count": len(candles),
    }
