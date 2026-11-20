from __future__ import annotations

"""Binance USDⓈ-M Futures 3H Demand/Supply context for strict pre-order validation.

S/R levels and the live directional price are sourced exclusively from the same
active Binance USDⓈ-M USDT perpetual contract used by the signal engine. Fully
closed 1H Futures candles are aggregated into exact UTC-aligned 3H candles.
All HTTP transport, pacing, retries, exchange metadata and VPN proxy routing are
owned by ``event_engine.binance``; this module does not create a second HTTP path.
"""

import logging
import math
import os
import threading
import time
from typing import Any

from event_engine.binance import (
    BinanceSymbolUnavailableError,
    fetch_klines as fetch_futures_klines,
    fetch_price as fetch_futures_price,
    resolve_symbol as resolve_binance_futures_symbol,
)
from pine_r541_sr import Candle, compute_current_sr

log = logging.getLogger("event_engine")

SR_TIMEFRAME = "3h"
SR_INTERVAL_MS = 3 * 60 * 60 * 1000
SR_SOURCE_INTERVAL_MS = 60 * 60 * 1000
# Binance Futures does not provide a native 3h kline interval. Use its native 1h
# contract candles and aggregate only complete UTC-aligned groups of three.
SR_SOURCE_KLINE_LIMIT_1H = max(300, min(1500, int(os.environ.get("AJAY_SR_KLINE_LIMIT_1H", "1000"))))
# User-approved hard rule: candidate must be within 5% of nearest actionable zone.
SR_DIRECTIONAL_MAX_DISTANCE_PCT = 5.0
SR_SWING_LENGTH = max(1, int(os.environ.get("AJAY_SR_SWING_LENGTH", "10")))
SR_HISTORY_TO_KEEP = max(5, min(50, int(os.environ.get("AJAY_SR_HISTORY_TO_KEEP", "20"))))
SR_BOX_WIDTH = float(os.environ.get("AJAY_SR_BOX_WIDTH", "2.5"))
SR_ATR_LENGTH = max(1, int(os.environ.get("AJAY_SR_ATR_LENGTH", "50")))
SR_MIN_CANDLES = max(SR_ATR_LENGTH, 2 * SR_SWING_LENGTH + 1)
SR_ENTRY_BUFFER_PCT = max(0.0, float(os.environ.get("AJAY_SR_ENTRY_BUFFER_PCT", "0.05")))
SR_TARGET_BUFFER_PCT = max(0.0, float(os.environ.get("AJAY_SR_TARGET_BUFFER_PCT", "0.05")))
SR_TARGET_BUFFER_R = max(0.0, float(os.environ.get("AJAY_SR_TARGET_BUFFER_R", "0.10")))
SR_SUPPORT_CONTEXT_MAX_R = max(0.0, float(os.environ.get("AJAY_SR_SUPPORT_CONTEXT_MAX_R", "3.0")))
SR_MAX_DATA_AGE_MIN = max(30.0, float(os.environ.get("AJAY_SR_MAX_DATA_AGE_MIN", "190")))

_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


class SRSymbolUnavailableError(RuntimeError):
    """The requested symbol has no active Binance USDⓈ-M USDT perpetual contract."""


def normalize_futures_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper().replace("/", "").replace("-", "").replace(" ", "")
    if value.endswith("_PERP"):
        value = value[:-5]
    if value.endswith("_USDT"):
        value = value[:-5] + "USDT"
    if not value.endswith("USDT"):
        value = f"{value}USDT"
    return value


def resolve_futures_symbol(symbol: str) -> dict[str, Any]:
    """Resolve a logical symbol only against active Binance USDⓈ-M perpetuals."""
    requested = normalize_futures_symbol(symbol)
    try:
        resolved = resolve_binance_futures_symbol(requested)
    except BinanceSymbolUnavailableError as exc:
        raise SRSymbolUnavailableError(str(exc)) from exc
    if not resolved:
        raise SRSymbolUnavailableError(
            f"No active Binance USDⓈ-M Futures USDT perpetual contract for {requested}"
        )
    base = resolved[:-4] if resolved.endswith("USDT") else resolved
    return {
        "requested_symbol": requested,
        "source_symbol": resolved,
        "underlying_asset": base,
        "source_alias_kind": "EXACT_FUTURES" if resolved == requested else "NORMALIZED_FUTURES",
        "contract_type": "PERPETUAL",
        "quote_asset": "USDT",
    }


def _fetch_closed_1h_futures(
    symbol: str,
    *,
    resolved: dict[str, Any] | None = None,
    limit: int | None = None,
) -> tuple[list[Candle], dict[str, Any]]:
    """Fetch fully closed 1h klines from the resolved Futures perpetual contract."""
    resolved = dict(resolved or resolve_futures_symbol(symbol))
    kline_limit = SR_SOURCE_KLINE_LIMIT_1H if limit is None else max(1, min(1500, int(limit)))
    try:
        rows = fetch_futures_klines(resolved["source_symbol"], "1h", limit=kline_limit)
    except BinanceSymbolUnavailableError as exc:
        raise SRSymbolUnavailableError(str(exc)) from exc
    if not isinstance(rows, list):
        raise ValueError(f"Binance Futures 1H klines response is not a list for {resolved['source_symbol']}")

    candles: list[Candle] = []
    previous_open_time: int | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            open_time = int(row["open_time"])
            candle = Candle(
                ts=open_time,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
        if (
            open_time % SR_SOURCE_INTERVAL_MS != 0
            or not all(math.isfinite(value) for value in values)
            or min(candle.open, candle.high, candle.low, candle.close) <= 0
            or candle.volume < 0
            or candle.high < max(candle.open, candle.close, candle.low)
            or candle.low > min(candle.open, candle.close, candle.high)
        ):
            continue
        if previous_open_time is not None and open_time <= previous_open_time:
            raise ValueError(
                f"Binance Futures 1H klines are duplicated or out of order for {resolved['source_symbol']}"
            )
        candles.append(candle)
        previous_open_time = open_time

    if len(candles) < SR_MIN_CANDLES * 3:
        raise ValueError(
            f"Only {len(candles)} closed Binance Futures 1H candles available from "
            f"{resolved['source_symbol']}; need at least {SR_MIN_CANDLES * 3} to build {SR_MIN_CANDLES} 3H bars"
        )
    return candles, resolved


def _aggregate_closed_3h_candles(hourly_candles: list[Candle]) -> list[Candle]:
    """Aggregate closed 1H Futures candles into exact UTC-aligned 3H candles.

    A 3h bar is emitted only when all three expected 1h open timestamps exist.
    OHLCV aggregation is first-open, max-high, min-low, last-close and summed base volume.
    """
    groups: dict[int, dict[int, Candle]] = {}
    previous_ts: int | None = None
    for candle in hourly_candles:
        ts = int(candle.ts)
        if previous_ts is not None and ts <= previous_ts:
            raise ValueError("Binance Futures 1H candles are duplicated or out of order before 3H aggregation")
        previous_ts = ts
        bucket_open = (ts // SR_INTERVAL_MS) * SR_INTERVAL_MS
        groups.setdefault(bucket_open, {})[ts] = candle

    aggregated: list[Candle] = []
    for bucket_open in sorted(groups):
        members = groups[bucket_open]
        expected = (bucket_open, bucket_open + SR_SOURCE_INTERVAL_MS, bucket_open + 2 * SR_SOURCE_INTERVAL_MS)
        if any(ts not in members for ts in expected):
            continue
        bars = [members[ts] for ts in expected]
        aggregated.append(Candle(
            ts=bucket_open,
            open=float(bars[0].open),
            high=max(float(bar.high) for bar in bars),
            low=min(float(bar.low) for bar in bars),
            close=float(bars[-1].close),
            volume=sum(float(bar.volume) for bar in bars),
        ))
    return aggregated


def _fetch_closed_3h_futures(
    symbol: str,
    *,
    resolved: dict[str, Any] | None = None,
    limit: int | None = None,
) -> tuple[list[Candle], dict[str, Any]]:
    """Build the production 3h S/R series from Binance Futures 1h klines."""
    hourly_candles, meta = _fetch_closed_1h_futures(symbol, resolved=resolved, limit=limit)
    minimum = SR_MIN_CANDLES
    if limit is not None and int(limit) < minimum * 3:
        raise ValueError(
            f"Requested {limit} source 1H candles cannot build {minimum} closed 3H bars; "
            f"at least {minimum * 3} source 1H candles are required"
        )
    candles_3h = _aggregate_closed_3h_candles(hourly_candles)
    if len(candles_3h) < minimum:
        raise ValueError(
            f"Only {len(candles_3h)} complete closed Futures 3H candles can be aggregated from "
            f"{len(hourly_candles)} 1H candles for {meta.get('source_symbol')}; need {minimum}"
        )
    gaps = [
        (left.ts, right.ts)
        for left, right in zip(candles_3h, candles_3h[1:])
        if right.ts - left.ts != SR_INTERVAL_MS
    ]
    if gaps:
        first_left, first_right = gaps[0]
        raise ValueError(
            "Incomplete Futures 3H history: missing one or more UTC-aligned 3H buckets "
            f"between {first_left} and {first_right} for {meta.get('source_symbol')}"
        )
    return candles_3h, meta


def get_futures_current_price(symbol: str, *, snapshot: dict[str, Any] | None = None) -> float:
    """Return the current Binance USDⓈ-M Futures last-traded price."""
    target = (
        str(snapshot.get("source_symbol")).strip().upper()
        if isinstance(snapshot, dict) and snapshot.get("source_symbol")
        else normalize_futures_symbol(symbol)
    )
    try:
        price = float(fetch_futures_price(target))
    except BinanceSymbolUnavailableError as exc:
        raise SRSymbolUnavailableError(str(exc)) from exc
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"Invalid Binance Futures ticker price for {target}: {price!r}")
    return price


def get_cached_sr_snapshot(symbol: str, *, limit: int | None = None) -> dict[str, Any]:
    """Fetch/cache the current Pine SR state; a custom history limit is uncached."""
    key = normalize_futures_symbol(symbol)
    use_cache = limit is None
    if use_cache:
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
            if cached is not None:
                cached_copy = dict(cached)
                cached_ts = int(cached_copy.get("latest_closed_timestamp") or 0)
                cached_age = max(
                    0.0,
                    (time.time() * 1000 - (cached_ts + SR_INTERVAL_MS - 1)) / 60_000.0,
                )
                if cached_ts > 0 and cached_age <= SR_MAX_DATA_AGE_MIN:
                    cached_copy["latest_closed_age_min"] = cached_age
                    return cached_copy
                _CACHE.pop(key, None)

    candles, resolved = _fetch_closed_3h_futures(symbol, limit=limit)
    latest_ts = candles[-1].ts
    state = compute_current_sr(
        candles,
        swing_length=SR_SWING_LENGTH,
        history_to_keep=SR_HISTORY_TO_KEEP,
        box_width=SR_BOX_WIDTH,
        atr_length=SR_ATR_LENGTH,
    )
    latest_age_min = max(0.0, (time.time() * 1000 - (latest_ts + SR_INTERVAL_MS - 1)) / 60_000.0)
    if latest_age_min > SR_MAX_DATA_AGE_MIN:
        raise ValueError(f"SR Futures data stale: latest_closed_age={latest_age_min:.2f}m > {SR_MAX_DATA_AGE_MIN:.2f}m")

    snapshot = dict(state)
    snapshot.update({
        "source": "binance_usds_m_futures",
        "timeframe": SR_TIMEFRAME,
        "interval_ms": SR_INTERVAL_MS,
        "source_interval": "1h_aggregated",
        "closed_candle_count_3h": len(candles),
        "source_symbol": resolved["source_symbol"],
        "source_symbol_requested": resolved["requested_symbol"],
        "source_alias_kind": resolved["source_alias_kind"],
        "contract_type": resolved["contract_type"],
        "quote_asset": resolved["quote_asset"],
        "underlying_asset": resolved["underlying_asset"],
        "latest_closed_timestamp": latest_ts,
        "latest_closed_age_min": latest_age_min,
        "zone_scale": 1.0,
    })
    if use_cache:
        with _CACHE_LOCK:
            _CACHE[key] = dict(snapshot)
            if len(_CACHE) > 128:
                for old_key in list(_CACHE)[:-64]:
                    _CACHE.pop(old_key, None)
    return snapshot


def clear_sr_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def _safe_price(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _build_zones(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Return active zone geometry; legacy snapshots remain accepted by tests/tools.

    Production snapshots come from the DEMAND/SUPPLY port and contain explicit
    top/bottom boundaries.  No cwidth or zone_scale is applied to those boxes.
    The accepted/cwidth path is retained only for older persisted diagnostic
    fixtures; the new production calculator always supplies active_zones.
    """
    active = snapshot.get("active_zones")
    if isinstance(active, list):
        zones: list[dict[str, Any]] = []
        for raw in active:
            if not isinstance(raw, dict):
                continue
            top = _safe_price(raw.get("top", raw.get("upper")))
            bottom = _safe_price(raw.get("bottom", raw.get("lower")))
            center = _safe_price(raw.get("center", raw.get("poi")))
            if top is None or bottom is None or center is None or top < bottom:
                continue
            kind = str(raw.get("kind", "")).upper()
            if kind not in {"H", "L"}:
                continue
            zones.append({
                **raw,
                "center": center,
                "lower": bottom,
                "upper": top,
                "top": top,
                "bottom": bottom,
                "kind": kind,
                "zone_type": "SUPPLY" if kind == "H" else "DEMAND",
                "points": int(raw.get("points", 1) or 1),
                "countpp": int(raw.get("countpp", 0) or 0),
                "box_width": top - bottom,
                "cluster_width": top - bottom,  # backward-compatible diagnostic alias
            })
        return zones

    # Backward-compatible parsing for historic test fixtures only.
    cwidth = _safe_price(snapshot.get("cwidth"))
    if cwidth is None:
        return []
    scale = max(0.0, float(snapshot.get("zone_scale", 1.0)))
    width = cwidth * scale
    zones = []
    for accepted in snapshot.get("accepted", []) or []:
        if not isinstance(accepted, dict):
            continue
        center = _safe_price(accepted.get("price"))
        if center is None:
            continue
        zones.append({
            "center": center,
            "lower": max(0.0, center - width),
            "upper": center + width,
            "kind": str(accepted.get("kind", "")).upper(),
            "points": int(accepted.get("points", 0) or 0),
            "countpp": int(accepted.get("countpp", 0) or 0),
            "cluster_width": cwidth,
        })
    return zones


def _room_buffer(entry_price: float, risk_distance: float) -> float:
    return max(
        entry_price * SR_TARGET_BUFFER_PCT / 100.0,
        risk_distance * SR_TARGET_BUFFER_R,
    )



def evaluate_sr_directional_zone(
    snapshot: dict[str, Any],
    *,
    current_price: float,
    direction: str,
) -> dict[str, Any]:
    """Hard directional S/R gate for a closed 3H Binance USDⓈ-M Futures snapshot.

    A pivot-low zone (``L`` / Demand) permits LONG only; a pivot-high zone
    (``H`` / Supply) permits SHORT only. Only unbroken zones on the actionable
    side of price are candidates. The nearest candidate zone decides the current
    directional regime. Mixed-type ties are ambiguous and fail closed.

    Distance is measured from the live Futures last price to the nearest zone boundary,
    divided by the live Futures price. Being inside the zone has distance zero.
    """
    d = str(direction or "").strip().upper()
    if d not in {"LONG", "SHORT"}:
        raise ValueError(f"Invalid direction={direction!r}")
    price = _safe_price(current_price)
    if price is None:
        raise ValueError(f"Invalid Binance Futures current price={current_price!r}")
    timeframe = str(snapshot.get("timeframe") or "").strip().lower()
    if timeframe != "3h":
        return {
            "reject": True,
            "reject_reason": "SR_TIMEFRAME_MISMATCH",
            "expected_timeframe": "3h",
            "actual_timeframe": timeframe or None,
            "current_price": price,
            "direction": d,
        }
    # The user-specified 5% rule is a hard invariant; callers cannot widen it.
    limit_pct = SR_DIRECTIONAL_MAX_DISTANCE_PCT

    eligible: list[dict[str, Any]] = []
    for zone in _build_zones(snapshot):
        lower = float(zone["lower"])
        upper = float(zone["upper"])
        kind = str(zone.get("kind", "")).upper()
        if kind == "L":
            # A Demand zone only remains actionable while price is in or above it.
            # If price has broken below the entire zone, do not treat it as support.
            if price < lower:
                continue
            distance = 0.0 if price <= upper else (price - upper) / price * 100.0
            zone_type = "DEMAND"
            allowed_direction = "LONG"
        elif kind == "H":
            # A Supply zone only remains actionable while price is in or below it.
            # If price has broken above the entire zone, do not treat it as resistance.
            if price > upper:
                continue
            distance = 0.0 if price >= lower else (lower - price) / price * 100.0
            zone_type = "SUPPLY"
            allowed_direction = "SHORT"
        else:
            continue
        eligible.append({
            **zone,
            "zone_type": zone_type,
            "allowed_direction": allowed_direction,
            "distance_pct": max(0.0, float(distance)),
        })

    base = {
        "source": snapshot.get("source"),
        "timeframe": timeframe,
        "source_symbol": snapshot.get("source_symbol"),
        "snapshot_latest_closed_timestamp": snapshot.get("latest_closed_timestamp"),
        "snapshot_latest_closed_age_min": snapshot.get("latest_closed_age_min"),
        "current_price": price,
        "direction": d,
        "max_distance_pct": limit_pct,
    }
    if not eligible:
        return {
            **base,
            "reject": True,
            "reject_reason": "NO_VALID_DEMAND_OR_SUPPLY_ZONE",
            "nearest_zone": None,
            "allowed_direction": None,
            "distance_pct": None,
        }

    eligible.sort(key=lambda z: (
        float(z["distance_pct"]),
        -int(z.get("points", 0)),
        -int(z.get("countpp", 0)),
        float(z["center"]),
    ))
    nearest_distance = float(eligible[0]["distance_pct"])
    tied = [z for z in eligible if abs(float(z["distance_pct"]) - nearest_distance) <= 1e-9]
    tied_types = {str(z["zone_type"]) for z in tied}
    if len(tied_types) > 1:
        return {
            **base,
            "reject": True,
            "reject_reason": "AMBIGUOUS_NEAREST_DEMAND_SUPPLY_ZONE",
            "nearest_zone": tied,
            "allowed_direction": None,
            "distance_pct": nearest_distance,
        }

    nearest = eligible[0]
    allowed = str(nearest["allowed_direction"])
    if nearest_distance > limit_pct + 1e-9:
        return {
            **base,
            "reject": True,
            "reject_reason": "NO_DIRECTIONAL_ZONE_WITHIN_DISTANCE",
            "nearest_zone": nearest,
            "zone_type": nearest["zone_type"],
            "allowed_direction": allowed,
            "distance_pct": nearest_distance,
        }
    if d != allowed:
        return {
            **base,
            "reject": True,
            "reject_reason": "DIRECTION_CONTRADICTS_CURRENT_ZONE",
            "nearest_zone": nearest,
            "zone_type": nearest["zone_type"],
            "allowed_direction": allowed,
            "distance_pct": nearest_distance,
        }
    return {
        **base,
        "reject": False,
        "reject_reason": None,
        "nearest_zone": nearest,
        "zone_type": nearest["zone_type"],
        "allowed_direction": allowed,
        "distance_pct": nearest_distance,
        "alignment": "LONG_IN_DEMAND" if allowed == "LONG" else "SHORT_IN_SUPPLY",
    }


def evaluate_sr_room(
    snapshot: dict[str, Any],
    *,
    entry_price: float,
    direction: str,
    risk_pct: float,
    target_rrs: tuple[float, ...],
) -> dict[str, Any]:
    """Evaluate opposing-zone room and supporting-zone context.

    Rule set:
    - Opposing zone before/inside TP1 => reject.
    - Opposing zone after TP1 => allow; it never mutates the TP ladder.
    - Supporting zone inside the entry is a directional confirmation only.
    - Supporting context never adds arbitrary legacy-score points and never blocks entry.
    - If entry overlaps an opposing zone => reject.
    """
    d = str(direction).upper()
    if d not in {"LONG", "SHORT"}:
        raise ValueError(f"Invalid direction={direction}")
    if entry_price <= 0 or risk_pct <= 0:
        raise ValueError("Invalid entry/risk for SR room")
    if len(target_rrs) < 2 or any(not math.isfinite(float(rr)) or float(rr) <= 0 for rr in target_rrs):
        raise ValueError("SR room requires at least two positive finite TP targets")

    risk_distance = entry_price * risk_pct / 100.0
    buffer = _room_buffer(entry_price, risk_distance)
    entry_buffer = entry_price * SR_ENTRY_BUFFER_PCT / 100.0
    target_prices = [
        entry_price + risk_distance * float(rr) if d == "LONG" else entry_price - risk_distance * float(rr)
        for rr in target_rrs
    ]
    zones = _build_zones(snapshot)

    opposing: list[dict[str, Any]] = []
    supporting: list[dict[str, Any]] = []
    for zone in zones:
        lower = float(zone["lower"])
        upper = float(zone["upper"])
        center = float(zone["center"])
        kind = str(zone.get("kind", "")).upper()
        if d == "LONG":
            # Ajay semantics: H is resistance, L is support. Geometry decides
            # whether that level is actionable relative to entry; kind decides
            # direction. An H below LONG entry is not silently re-labelled
            # as support, and unknown kinds are neutral.
            if kind == "H":
                if lower <= entry_price <= upper:
                    zone["entry_overlap"] = "opposing"
                    opposing.append(zone)
                elif lower > entry_price:
                    zone["distance_from_entry"] = lower - entry_price
                    if zone["distance_from_entry"] <= entry_buffer:
                        zone["near_entry"] = "opposing"
                    opposing.append(zone)
            elif kind == "L":
                if lower <= entry_price <= upper:
                    zone["entry_overlap"] = "supporting"
                    supporting.append(zone)
                elif upper < entry_price:
                    zone["distance_from_entry"] = entry_price - upper
                    if zone["distance_from_entry"] <= entry_buffer:
                        zone["near_entry"] = "supporting"
                    supporting.append(zone)
        else:
            if kind == "L":
                if lower <= entry_price <= upper:
                    zone["entry_overlap"] = "opposing"
                    opposing.append(zone)
                elif upper < entry_price:
                    zone["distance_from_entry"] = entry_price - upper
                    if zone["distance_from_entry"] <= entry_buffer:
                        zone["near_entry"] = "opposing"
                    opposing.append(zone)
            elif kind == "H":
                if lower <= entry_price <= upper:
                    zone["entry_overlap"] = "supporting"
                    supporting.append(zone)
                elif lower > entry_price:
                    zone["distance_from_entry"] = lower - entry_price
                    if zone["distance_from_entry"] <= entry_buffer:
                        zone["near_entry"] = "supporting"
                    supporting.append(zone)

    if d == "LONG":
        opposing.sort(key=lambda z: float(z["lower"]) - entry_price if float(z["lower"]) > entry_price else 0.0)
        supporting.sort(key=lambda z: (
            0 if z.get("entry_overlap") == "supporting" else 1,
            abs(entry_price - float(z["upper"])),
        ))
    else:
        opposing.sort(key=lambda z: entry_price - float(z["upper"]) if float(z["upper"]) < entry_price else 0.0)
        supporting.sort(key=lambda z: (
            0 if z.get("entry_overlap") == "supporting" else 1,
            abs(float(z["lower"]) - entry_price),
        ))

    nearest_opp = opposing[0] if opposing else None
    nearest_support = supporting[0] if supporting else None
    result: dict[str, Any] = {
        "source": snapshot.get("source", "binance_usds_m_futures"),
        "source_symbol": snapshot.get("source_symbol"),
        "source_symbol_requested": snapshot.get("source_symbol_requested"),
        "source_alias_kind": snapshot.get("source_alias_kind"),
        "underlying_asset": snapshot.get("underlying_asset"),
        "snapshot_latest_closed_timestamp": snapshot.get("latest_closed_timestamp"),
        "snapshot_latest_closed_age_min": snapshot.get("latest_closed_age_min"),
        "cwidth": snapshot.get("cwidth"),
        "zone_scale": snapshot.get("zone_scale", 1.0),
        "entry_price": entry_price,
        "direction": d,
        "risk_pct": risk_pct,
        "risk_distance": risk_distance,
        "tp_rrs": list(target_rrs),
        "tp_prices": target_prices,
        "buffer_price": buffer,
        "entry_buffer_price": entry_buffer,
        "nearest_opposing_zone": nearest_opp,
        "nearest_supporting_zone": nearest_support,
        "room_status": "FULL_ROOM",
        "reject": False,
        "reject_reason": None,
        "tp3_capped": False,  # legacy diagnostic field; new ladders contain only TP1/TP2
        "effective_tp2_price": target_prices[1],
        "effective_final_tp_price": target_prices[-1],
        "effective_tp3_price": target_prices[2] if len(target_prices) >= 3 else None,
        "supporting_zone_confirmation": False,
        "directional_zone_alignment": None,
    }

    if nearest_support is not None:
        if nearest_support.get("distance_from_entry") is None and nearest_support.get("entry_overlap") == "supporting":
            result["supporting_zone_distance_r"] = 0.0
            result["supporting_zone_near"] = True
            result["supporting_zone_context"] = "SUPPORTIVE_INSIDE"
            result["supporting_zone_confirmation"] = True
            result["directional_zone_alignment"] = "LONG_IN_DEMAND" if d == "LONG" else "SHORT_IN_SUPPLY"
        elif nearest_support.get("distance_from_entry") is not None:
            support_distance_r = float(nearest_support["distance_from_entry"]) / risk_distance
            result["supporting_zone_distance_r"] = support_distance_r
            result["supporting_zone_near"] = support_distance_r <= SR_SUPPORT_CONTEXT_MAX_R
            result["supporting_zone_context"] = "SUPPORTIVE_NEAR" if result["supporting_zone_near"] else "SUPPORTIVE_FAR"
            result["supporting_zone_confirmation"] = False
            result["directional_zone_alignment"] = None
        else:
            result["supporting_zone_distance_r"] = None
            result["supporting_zone_near"] = False
            result["supporting_zone_context"] = "NONE"
    else:
        result["supporting_zone_distance_r"] = None
        result["supporting_zone_near"] = False
        result["supporting_zone_context"] = "NONE"
        result["supporting_zone_confirmation"] = False
        result["directional_zone_alignment"] = None

    if nearest_opp is None:
        return result

    if nearest_opp.get("entry_overlap") == "opposing":
        result.update({
            "room_status": "ENTRY_IN_OPPOSING_ZONE",
            "reject": True,
            "reject_reason": "ENTRY_IN_OPPOSING_ZONE",
        })
        return result

    near_edge = float(nearest_opp["lower"] if d == "LONG" else nearest_opp["upper"])
    distance_r = (near_edge - entry_price) / risk_distance if d == "LONG" else (entry_price - near_edge) / risk_distance
    result["opposing_zone_distance_r"] = distance_r
    result["room_margin_to_tp1_r"] = distance_r - float(target_rrs[0])
    result["room_margin_to_tp2_r"] = distance_r - float(target_rrs[1])
    if len(target_rrs) >= 3:
        result["room_margin_to_tp3_r"] = distance_r - float(target_rrs[2])

    # The trading rule is intentionally literal: only an opposing zone that reaches
    # the FIRST take-profit invalidates the entry. A safety buffer is reported for
    # diagnostics but does not move the hard TP1 boundary. A zone after TP1 is not
    # an entry veto and does not mutate the TP ladder.
    tp1 = target_prices[0]
    blocks_tp1 = (d == "LONG" and near_edge <= tp1) or (d == "SHORT" and near_edge >= tp1)
    if blocks_tp1:
        result.update({
            "room_status": "OPPOSING_ZONE_BEFORE_TP1",
            "reject": True,
            "reject_reason": "OPPOSING_ZONE_BEFORE_TP1",
        })
    else:
        result["room_status"] = "POST_TP1_OPPOSING_ZONE"

    return result
