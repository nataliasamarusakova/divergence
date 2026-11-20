from __future__ import annotations

"""Python port of the DEMAND / SUPPLY block in the supplied Ajay R5.41 Pine v5.

Both Pine SR subsystems are implemented independently: (1) swing-based DEMAND /
SUPPLY boxes and (2) the ``enableSR`` / ``prd=284`` horizontal pivot-cluster
levels. The latter reproduces the Pine script's ``aas`` clustering, persistent
``sr_levs`` update behavior, High/Low labels, and 300-bar zone bands.

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
    include_horizontal_sr: bool = False,
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
    result = {
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
    if include_horizontal_sr:
        result["horizontal_sr"] = compute_ajay_horizontal_sr(candles)
    return result



def compute_ajay_horizontal_sr(
    candles: list[Candle],
    *,
    pivot_length: int = 10,
    period: int = 284,
    channel_width_pct: float = 10.0,
    strength: int = 2,
    zone_lookback: int = 300,
    zone_width_pct: float = 2.0,
    max_pivots: int = 40,
    max_levels: int = 20,
) -> dict[str, Any]:
    """Port the second ``enableSR`` block in Ajay R5.41 Pine v5.

    This is deliberately separate from ``compute_current_sr``'s Demand/Supply
    boxes. It mirrors the Pine script's pivot-event ordering and clustering:

    * ``ta.pivothigh/low(rb, rb)`` events are indexed by their confirmation bar.
    * On every bar with a new pivot event, the last ``period`` bars determine
      ``cwidth``; the Pine loop itself scans offsets ``0..period`` inclusive.
    * The local ``aas`` array starts all-true on each calculation. Accepted
      clusters mark their member pivot-event ordinals false.
    * ``sr_levs`` persists between pivot events and is replaced (including by
      an all-empty set) whenever a new pivot event occurs.
    * ``highestph`` / ``lowestpl`` follow the exact assignments in the supplied
      Pine code: High replaces missing pivots with ``prdlowest``; Low replaces
      missing pivots with ``prdhighest``.
    * Visible line-zone bands use ``percWidth(300, zoneWidth)`` on the latest
      closed bar, separately from the clustering tolerance ``cwidth``.

    ``status`` warns when the fetched history cannot reconstruct Pine's current
    persistent state or the full rolling windows. The function does not silently
    claim an exact match in those cases.
    """
    if pivot_length < 1 or period < 1 or zone_lookback < 1:
        raise ValueError("pivot_length, period and zone_lookback must be >= 1")
    if strength < 1:
        raise ValueError("strength must be >= 1, matching Pine input.int minval=1")
    if max_pivots < 1 or max_pivots > 40:
        raise ValueError("max_pivots must be between 1 and Pine's fixed limit of 40")
    if max_levels < 1 or max_levels > 20:
        raise ValueError("max_levels must be between 1 and Pine's fixed limit of 20")
    numeric_params = (channel_width_pct, zone_width_pct)
    if not all(math.isfinite(float(value)) and float(value) >= 0 for value in numeric_params):
        raise ValueError("channel_width_pct and zone_width_pct must be finite and >= 0")

    count = len(candles)
    pivot_highs, pivot_lows = build_confirmed_pivots(candles, swing_length=pivot_length)
    sr_levels: list[float | None] = [None] * (max_levels + 1)
    sr_records: list[dict[str, Any] | None] = [None] * (max_levels + 1)
    high_level: float | None = None
    low_level: float | None = None
    latest_update_idx: int | None = None
    latest_update_event_count = 0
    latest_update_cluster_width: float | None = None
    latest_update_range_high: float | None = None
    latest_update_range_low: float | None = None

    for current_idx in range(count):
        if pivot_highs[current_idx] is None and pivot_lows[current_idx] is None:
            continue

        # Pine ta.highest(period) / ta.lowest(period) are current-inclusive and
        # use exactly `period` bars, while its explicit x-loop uses 0..period.
        range_start = max(0, current_idx - period + 1)
        range_bars = candles[range_start: current_idx + 1]
        prd_highest = max(float(bar.high) for bar in range_bars)
        prd_lowest = min(float(bar.low) for bar in range_bars)
        cluster_width = (prd_highest - prd_lowest) * float(channel_width_pct) / 100.0

        # Recreate Pine's reverse scan: offset 0 is the current confirmation
        # bar, then older confirmation bars. The x-loop is inclusive at `period`.
        events: list[dict[str, Any]] = []
        event_start = max(0, current_idx - period)
        for confirmation_idx in range(current_idx, event_start - 1, -1):
            high_value = pivot_highs[confirmation_idx]
            low_value = pivot_lows[confirmation_idx]
            if high_value is None and low_value is None:
                continue
            pivot_idx = confirmation_idx - pivot_length
            event = {
                "ordinal": len(events) + 1,
                "confirmation_index": confirmation_idx,
                "pivot_index": pivot_idx,
                "confirmation_timestamp": int(candles[confirmation_idx].ts),
                "pivot_timestamp": int(candles[pivot_idx].ts) if pivot_idx >= 0 else None,
                "high": float(high_value) if high_value is not None else None,
                "low": float(low_value) if low_value is not None else None,
            }
            events.append(event)
            # Pine updates High/Low before incrementing/checking countpp > 40,
            # so the 41st event contributes to these two labels but not clusters.
            if len(events) >= max_pivots + 1:
                break

        # Exact Pine assignments inside `if ph or pl`.
        high_level = prd_lowest
        low_level = prd_highest
        for event in events:
            # Keep the two Pine nz() fallbacks distinct. The highest label
            # substitutes prd_lowest for missing pivots, while the lowest
            # label substitutes prd_highest. Reusing prd_lowest for both
            # incorrectly pins LOW_LEVEL to the bottom of the whole window.
            high_from_ph = event["high"] if event["high"] is not None else prd_lowest
            high_from_pl = event["low"] if event["low"] is not None else prd_lowest
            low_from_ph = event["high"] if event["high"] is not None else prd_highest
            low_from_pl = event["low"] if event["low"] is not None else prd_highest
            high_level = max(high_level, float(high_from_ph), float(high_from_pl))
            low_level = min(low_level, float(low_from_ph), float(low_from_pl))

        # The source Pine declares `aas = array.new_bool(41, true)` without var:
        # it is recreated on each chart bar, not persisted between updates.
        aas = [True] * (max_pivots + 1)
        sr_levels = [None] * (max_levels + 1)
        sr_records = [None] * (max_levels + 1)

        cluster_events = events[:max_pivots]
        for candidate in cluster_events:
            countpp = int(candidate["ordinal"])
            if countpp >= len(aas) or not aas[countpp]:
                continue
            # Pine's ternary selects pivot HIGH if both ph and pl are true.
            # Pine's cluster tolerance uses the ternary high-first candidate
            # (`ph[x] ? high[x+rb] : low[x+rb]`). However, its two subsequent
            # `if ph[x]` / `if pl[x]` writes target the same sr_levels slot;
            # when both pivots exist, the LOW write happens last and wins.
            cluster_seed_price = (
                float(candidate["high"])
                if candidate["high"] is not None
                else float(candidate["low"])
            )
            stored_level_price = (
                float(candidate["low"])
                if candidate["low"] is not None
                else float(candidate["high"])
            )
            upper_boundary = cluster_seed_price + cluster_width
            lower_boundary = cluster_seed_price - cluster_width
            tmp = [True] * (max_pivots + 1)
            tpoint = 0
            included: list[dict[str, Any]] = []

            for event in cluster_events:
                ordinal = int(event["ordinal"])
                if ordinal >= len(aas):
                    break
                if not aas[ordinal]:
                    continue
                changed = False
                for pivot_kind in ("high", "low"):
                    pivot_value = event[pivot_kind]
                    if pivot_value is not None and lower_boundary <= float(pivot_value) <= upper_boundary:
                        tpoint += 1
                        changed = True
                        included.append({
                            "ordinal": ordinal,
                            "kind": pivot_kind.upper(),
                            "price": float(pivot_value),
                            "pivot_timestamp": event["pivot_timestamp"],
                            "confirmation_timestamp": event["confirmation_timestamp"],
                        })
                if changed and ordinal < len(tmp):
                    tmp[ordinal] = False

            if tpoint >= strength:
                for ordinal in range(len(aas)):
                    if not tmp[ordinal]:
                        aas[ordinal] = False
                if countpp < len(sr_levels):
                    sr_levels[countpp] = stored_level_price
                    sr_records[countpp] = {
                        "slot": countpp,
                        "level": stored_level_price,
                        "candidate_kind": "LOW" if candidate["low"] is not None else "HIGH",
                        "cluster_seed_kind": "HIGH" if candidate["high"] is not None else "LOW",
                        "cluster_seed_price": cluster_seed_price,
                        "candidate_pivot_timestamp": candidate["pivot_timestamp"],
                        "candidate_confirmation_timestamp": candidate["confirmation_timestamp"],
                        "candidate_pivot_index": candidate["pivot_index"],
                        "points": int(tpoint),
                        "pivot_event_count": len({item["ordinal"] for item in included}),
                        "cluster_half_width": float(cluster_width),
                        "cluster_lower": float(lower_boundary),
                        "cluster_upper": float(upper_boundary),
                        "members": included,
                    }

        latest_update_idx = current_idx
        latest_update_event_count = len(events)
        latest_update_cluster_width = float(cluster_width)
        latest_update_range_high = float(prd_highest)
        latest_update_range_low = float(prd_lowest)

    # The displayed Pine zonePerc is recalculated every bar even when no pivot
    # event occurs; it is based on the latest closed bar's 300-bar range.
    zone_start = max(0, count - zone_lookback)
    zone_bars = candles[zone_start:]
    zone_range_high = max((float(bar.high) for bar in zone_bars), default=None)
    zone_range_low = min((float(bar.low) for bar in zone_bars), default=None)
    zone_half_width = (
        (zone_range_high - zone_range_low) * float(zone_width_pct) / 100.0
        if zone_range_high is not None and zone_range_low is not None else None
    )

    records: list[dict[str, Any]] = []
    for slot, record in enumerate(sr_records):
        if slot == 0 or not isinstance(record, dict):
            continue
        item = dict(record)
        level = float(item["level"])
        if zone_half_width is not None:
            item["band_lower"] = level - zone_half_width
            item["band_upper"] = level + zone_half_width
            item["band_half_width"] = zone_half_width
        else:
            item["band_lower"] = None
            item["band_upper"] = None
            item["band_half_width"] = None
        records.append(item)

    latest_update_bars_ago = (
        count - 1 - latest_update_idx if latest_update_idx is not None else None
    )
    # The Pine event scan is inclusive (`for x = 0 to prd`), i.e. prd+1
    # confirmation bars. A refresh at index `period` is the first complete scan.
    complete_prd_scan = latest_update_idx is not None and latest_update_idx >= period
    # Every pivot event in that scan must itself have had full left/right
    # confirmation history. The oldest inspected confirmation index is
    # latest_update_idx-period; its pivot needs another 2*rb bars before it.
    complete_pivot_warmup = (
        latest_update_idx is not None
        and latest_update_idx >= period + 2 * pivot_length
    )
    complete_prd_at_update = complete_prd_scan and complete_pivot_warmup
    complete_zone_window = count >= zone_lookback
    if latest_update_idx is None:
        status = "UNRECOVERABLE_NO_PIVOT_EVENT_IN_FETCHED_HISTORY"
    elif not complete_prd_scan:
        status = "INCOMPLETE_PRD_SCAN_WINDOW_AT_LAST_REFRESH"
    elif not complete_pivot_warmup:
        status = "INCOMPLETE_PIVOT_WARMUP_AT_LAST_REFRESH"
    elif not complete_zone_window:
        status = "INCOMPLETE_300_BAR_ZONE_WIDTH_WINDOW"
    else:
        status = "OK"

    return {
        "algorithm": "AJAY_R541_HORIZONTAL_SR_CLUSTERS",
        "status": status,
        "pivot_length": int(pivot_length),
        "period": int(period),
        "channel_width_pct": float(channel_width_pct),
        "strength": int(strength),
        "zone_lookback": int(zone_lookback),
        "zone_width_pct": float(zone_width_pct),
        "max_pivots": int(max_pivots),
        "max_levels": int(max_levels),
        "high_level": high_level,
        "low_level": low_level,
        "levels": [float(value) for value in sr_levels[1:] if value is not None],
        "level_records": records,
        "sr_level_slots": sr_levels,
        "prd_range_high_at_update": latest_update_range_high,
        "prd_range_low_at_update": latest_update_range_low,
        "cluster_half_width_at_update": latest_update_cluster_width,
        "zone_range_high": zone_range_high,
        "zone_range_low": zone_range_low,
        "zone_half_width": zone_half_width,
        "zone_width": (zone_half_width * 2.0) if zone_half_width is not None else None,
        "latest_update_index": latest_update_idx,
        "latest_update_timestamp": (
            int(candles[latest_update_idx].ts) if latest_update_idx is not None else None
        ),
        "latest_update_bars_ago": latest_update_bars_ago,
        "latest_update_pivot_event_count": latest_update_event_count,
        "closed_candle_count": count,
        "prd_scan_window_complete_at_update": complete_prd_scan,
        "prd_window_complete_at_update": complete_prd_at_update,
        "pivot_warmup_complete_at_update": complete_pivot_warmup,
        "zone_window_complete": complete_zone_window,
        "latest_closed_timestamp": int(candles[-1].ts) if candles else None,
        "latest_closed_close": float(candles[-1].close) if candles else None,
    }
