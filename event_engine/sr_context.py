from __future__ import annotations

"""Lazy Ajay R5.41 S/R context for final entry validation.

The trading engine requests S/R ONLY for already-selected candidates immediately
before an order. The current source is Binance SPOT 1H, deliberately configurable
and isolated from the Binance Futures signal client. Futures can be added later
without changing the room/gate logic.
"""

import logging
import math
import os
import threading
import time
from typing import Any

import requests

from pine_r541_sr import Candle, compute_current_sr

log = logging.getLogger("event_engine")

SR_SPOT_BASE_URL = os.environ.get("AJAY_SR_SPOT_BASE_URL", "https://data-api.binance.vision").rstrip("/")
SR_SPOT_KLINES_URL = f"{SR_SPOT_BASE_URL}/api/v3/klines"
SR_KLINE_LIMIT = max(300, min(1000, int(os.environ.get("AJAY_SR_KLINE_LIMIT_1H", "1000"))))
SR_RB = int(os.environ.get("AJAY_SR_RB", "10"))
SR_PRD = int(os.environ.get("AJAY_SR_PRD", "284"))
SR_CHANNEL_W = float(os.environ.get("AJAY_SR_CHANNEL_W", "10"))
SR_STRENGTH = int(os.environ.get("AJAY_SR_STRENGTH", "2"))
SR_ZONE_SCALE = max(0.0, float(os.environ.get("AJAY_SR_ZONE_SCALE", "1.0")))
SR_ENTRY_BUFFER_PCT = max(0.0, float(os.environ.get("AJAY_SR_ENTRY_BUFFER_PCT", "0.05")))
SR_TARGET_BUFFER_PCT = max(0.0, float(os.environ.get("AJAY_SR_TARGET_BUFFER_PCT", "0.05")))
SR_TARGET_BUFFER_R = max(0.0, float(os.environ.get("AJAY_SR_TARGET_BUFFER_R", "0.10")))
SR_MIN_PARTIAL_TP3_R = max(0.0, float(os.environ.get("AJAY_SR_MIN_PARTIAL_TP3_R", "0.25")))
SR_SUPPORT_CONTEXT_MAX_R = max(0.0, float(os.environ.get("AJAY_SR_SUPPORT_CONTEXT_MAX_R", "3.0")))
SR_HTTP_TIMEOUT_SEC = max(2.0, float(os.environ.get("AJAY_SR_HTTP_TIMEOUT_SEC", "5")))
SR_REQUEST_MIN_INTERVAL_SEC = max(0.0, float(os.environ.get("AJAY_SR_REQUEST_MIN_INTERVAL_SEC", "0.10")))
SR_MAX_DATA_AGE_MIN = max(30.0, float(os.environ.get("AJAY_SR_MAX_DATA_AGE_MIN", "130")))

_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()
_SESSION = requests.Session()
_LAST_REQUEST_MONOTONIC = 0.0
_REQUEST_LOCK = threading.Lock()
_SESSION.trust_env = False
_SR_HTTP_PROXY = os.environ.get("AJAY_SR_HTTP_PROXY", os.environ.get("BINANCE_HTTP_PROXY", "")).strip()
if _SR_HTTP_PROXY:
    _SESSION.proxies.update({"http": _SR_HTTP_PROXY, "https": _SR_HTTP_PROXY})


def normalize_spot_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper().replace("/", "").replace("-", "").replace(" ", "")
    if value.endswith("_PERP"):
        value = value[:-5]
    if not value.endswith("USDT"):
        value = f"{value}USDT"
    return value


def _fetch_closed_1h_spot(symbol: str) -> list[Candle]:
    global _LAST_REQUEST_MONOTONIC
    with _REQUEST_LOCK:
        now = time.monotonic()
        wait = SR_REQUEST_MIN_INTERVAL_SEC - (now - _LAST_REQUEST_MONOTONIC)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_MONOTONIC = time.monotonic()
    payload = _SESSION.get(
        SR_SPOT_KLINES_URL,
        params={"symbol": normalize_spot_symbol(symbol), "interval": "1h", "limit": SR_KLINE_LIMIT},
        timeout=SR_HTTP_TIMEOUT_SEC,
    )
    payload.raise_for_status()
    rows = payload.json()
    if not isinstance(rows, list):
        raise ValueError("Binance SPOT klines response is not a list")
    now_ms = int(time.time() * 1000)
    candles: list[Candle] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        open_time = int(row[0])
        close_time = int(row[6]) if len(row) > 6 else open_time + 3_599_999
        if close_time > now_ms:
            continue
        try:
            candles.append(
                Candle(
                    ts=open_time,
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )
        except (TypeError, ValueError):
            continue
    if len(candles) < SR_PRD + SR_RB + 2:
        raise ValueError(f"Only {len(candles)} closed SPOT 1H candles available")
    return candles


def get_cached_sr_snapshot(symbol: str) -> dict[str, Any]:
    """Fetch and cache the current Pine SR state once per symbol in one engine cycle."""
    key = normalize_spot_symbol(symbol)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return dict(cached)

    candles = _fetch_closed_1h_spot(symbol)
    latest_ts = candles[-1].ts
    state = compute_current_sr(
        candles,
        rb=SR_RB,
        prd=SR_PRD,
        channel_w=SR_CHANNEL_W,
        strength_sr=SR_STRENGTH,
    )
    latest_age_min = max(0.0, (time.time() * 1000 - (latest_ts + 3_599_999)) / 60_000.0)
    if latest_age_min > SR_MAX_DATA_AGE_MIN:
        raise ValueError(f"SR SPOT data stale: latest_closed_age={latest_age_min:.2f}m > {SR_MAX_DATA_AGE_MIN:.2f}m")

    snapshot = dict(state)
    snapshot.update({
        "source": "binance_spot",
        "source_symbol": normalize_spot_symbol(symbol),
        "latest_closed_timestamp": latest_ts,
        "latest_closed_age_min": latest_age_min,
        "zone_scale": SR_ZONE_SCALE,
    })
    with _CACHE_LOCK:
        _CACHE[key] = dict(snapshot)
        # Keep only recent cache rows to avoid accidental growth if reused by tests.
        if len(_CACHE) > 128:
            for old_key in list(_CACHE)[:-64]:
                _CACHE.pop(old_key, None)
    return snapshot


def clear_sr_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def _safe_price(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) and x > 0 else None
    except (TypeError, ValueError):
        return None


def _build_zones(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    cwidth = _safe_price(snapshot.get("cwidth"))
    if cwidth is None:
        return []
    scale = max(0.0, float(snapshot.get("zone_scale", SR_ZONE_SCALE)))
    width = cwidth * scale
    zones: list[dict[str, Any]] = []
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


def evaluate_sr_room(
    snapshot: dict[str, Any],
    *,
    entry_price: float,
    direction: str,
    risk_pct: float,
    target_rrs: tuple[float, float, float],
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
        if d == "LONG":
            if lower <= entry_price <= upper:
                # If price is inside a pivot-derived zone, use the zone origin
                # (H=resistance/supply, L=support/demand) rather than the zone
                # center. The center can legitimately sit on the opposite side
                # of entry when the cluster is wide. Unknown kind falls back to
                # geometry and is never silently treated as a supporting zone.
                kind = str(zone.get("kind", "")).upper()
                if kind == "H" or (kind not in {"H", "L"} and center >= entry_price):
                    zone["entry_overlap"] = "opposing"
                    opposing.append(zone)
                else:
                    zone["entry_overlap"] = "supporting"
                    supporting.append(zone)
            elif center > entry_price and lower > entry_price:
                zone["distance_from_entry"] = lower - entry_price
                if zone["distance_from_entry"] <= entry_buffer:
                    zone["near_entry"] = "opposing"
                opposing.append(zone)
            elif center < entry_price and upper < entry_price:
                zone["distance_from_entry"] = entry_price - upper
                if zone["distance_from_entry"] <= entry_buffer:
                    zone["near_entry"] = "supporting"
                supporting.append(zone)
        else:
            if lower <= entry_price <= upper:
                kind = str(zone.get("kind", "")).upper()
                if kind == "L" or (kind not in {"H", "L"} and center <= entry_price):
                    zone["entry_overlap"] = "opposing"
                    opposing.append(zone)
                else:
                    zone["entry_overlap"] = "supporting"
                    supporting.append(zone)
            elif center < entry_price and upper < entry_price:
                zone["distance_from_entry"] = entry_price - upper
                if zone["distance_from_entry"] <= entry_buffer:
                    zone["near_entry"] = "opposing"
                opposing.append(zone)
            elif center > entry_price and lower > entry_price:
                zone["distance_from_entry"] = lower - entry_price
                if zone["distance_from_entry"] <= entry_buffer:
                    zone["near_entry"] = "supporting"
                supporting.append(zone)

    if d == "LONG":
        opposing.sort(key=lambda z: float(z["lower"]) - entry_price if float(z["lower"]) > entry_price else 0.0)
        supporting.sort(key=lambda z: abs(entry_price - float(z["upper"])))
    else:
        opposing.sort(key=lambda z: entry_price - float(z["upper"]) if float(z["upper"]) < entry_price else 0.0)
        supporting.sort(key=lambda z: abs(float(z["lower"]) - entry_price))

    nearest_opp = opposing[0] if opposing else None
    nearest_support = supporting[0] if supporting else None
    result: dict[str, Any] = {
        "source": snapshot.get("source", "binance_spot"),
        "source_symbol": snapshot.get("source_symbol"),
        "snapshot_latest_closed_timestamp": snapshot.get("latest_closed_timestamp"),
        "snapshot_latest_closed_age_min": snapshot.get("latest_closed_age_min"),
        "cwidth": snapshot.get("cwidth"),
        "zone_scale": snapshot.get("zone_scale", SR_ZONE_SCALE),
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
        "tp3_capped": False,
        "effective_tp3_price": target_prices[2],
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


def apply_sr_tp3_cap(
    setup: dict[str, Any],
    *,
    direction: str,
    tp_levels: list[dict[str, Any]],
    sr_result: dict[str, Any],
    actual_entry_price: float,
) -> tuple[list[dict[str, Any]], float]:
    """Apply a previously validated partial-room TP3 cap after the real fill."""
    if not sr_result.get("tp3_capped"):
        return tp_levels, float(setup.get("target_rr", 2.5) or 2.5)
    if len(tp_levels) < 3:
        raise ValueError("SR TP3 cap requires three TP levels")
    cap_price = _safe_price(sr_result.get("effective_tp3_price"))
    if cap_price is None or actual_entry_price <= 0:
        raise ValueError("invalid SR TP3 cap price")
    risk_pct = float(setup.get("risk_pct", 0) or 0)
    if risk_pct <= 0:
        raise ValueError("invalid risk_pct for SR TP3 cap")
    if str(direction).upper() == "LONG":
        pnl_pct = (cap_price - actual_entry_price) / actual_entry_price * 100.0
    else:
        pnl_pct = (actual_entry_price - cap_price) / actual_entry_price * 100.0
    tp2_pnl = float(tp_levels[1].get("pnl_pct", 0) or 0)
    if pnl_pct <= tp2_pnl:
        raise ValueError("SR TP3 cap became unreachable after fill")
    adjusted = [dict(x) for x in tp_levels]
    adjusted[2]["pnl_pct"] = round(pnl_pct, 6)
    effective_target_rr = round(pnl_pct / risk_pct, 9)
    adjusted_weighted_rr = round(sum(float(x.get("close_fraction", 0) or 0) * (float(x.get("pnl_pct", 0) or 0) / risk_pct) for x in adjusted), 9)
    setup["target_price"] = cap_price
    setup["target_rr"] = effective_target_rr
    setup["planned_weighted_rr"] = adjusted_weighted_rr
    setup["sr_tp3_capped"] = True
    return adjusted, effective_target_rr
