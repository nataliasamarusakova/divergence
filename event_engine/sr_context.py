from __future__ import annotations

"""Binance Spot 3H Demand/Supply context for strict pre-order validation.

S/R levels are calculated only from fully closed Binance Spot 3H candles. The
entry gate uses the current Binance Spot ticker price and allows LONG only near
a valid Demand (pivot-low) zone, SHORT only near a valid Supply (pivot-high)
zone. No fallback to 1H, futures klines, or stale snapshots is permitted.
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
SR_SPOT_TICKER_PRICE_URL = f"{SR_SPOT_BASE_URL}/api/v3/ticker/price"
SR_TIMEFRAME = "3h"
SR_INTERVAL_MS = 3 * 60 * 60 * 1000
# Binance Spot REST does not expose a 3h kline interval. We request up to 1000
# closed 1h Spot candles, then aggregate exact UTC-aligned groups of three into 3h.
SR_SOURCE_KLINE_LIMIT_1H = max(300, min(1000, int(os.environ.get("AJAY_SR_KLINE_LIMIT_1H", "1000"))))
SR_SOURCE_INTERVAL_MS = 60 * 60 * 1000
# User-approved hard rule: a candidate must be within 5.0% of the nearest
# actionable 3H Demand/Supply zone. Changing it requires a code/release change.
SR_DIRECTIONAL_MAX_DISTANCE_PCT = 5.0
SR_RB = int(os.environ.get("AJAY_SR_RB", "10"))
SR_PRD = int(os.environ.get("AJAY_SR_PRD", "284"))
SR_CHANNEL_W = float(os.environ.get("AJAY_SR_CHANNEL_W", "10"))
SR_STRENGTH = int(os.environ.get("AJAY_SR_STRENGTH", "2"))
SR_ZONE_SCALE = max(0.0, float(os.environ.get("AJAY_SR_ZONE_SCALE", "1.0")))
SR_ENTRY_BUFFER_PCT = max(0.0, float(os.environ.get("AJAY_SR_ENTRY_BUFFER_PCT", "0.05")))
SR_TARGET_BUFFER_PCT = max(0.0, float(os.environ.get("AJAY_SR_TARGET_BUFFER_PCT", "0.05")))
SR_TARGET_BUFFER_R = max(0.0, float(os.environ.get("AJAY_SR_TARGET_BUFFER_R", "0.10")))
SR_SUPPORT_CONTEXT_MAX_R = max(0.0, float(os.environ.get("AJAY_SR_SUPPORT_CONTEXT_MAX_R", "3.0")))
SR_HTTP_TIMEOUT_SEC = max(2.0, float(os.environ.get("AJAY_SR_HTTP_TIMEOUT_SEC", "5")))
SR_REQUEST_MIN_INTERVAL_SEC = max(0.0, float(os.environ.get("AJAY_SR_REQUEST_MIN_INTERVAL_SEC", "0.10")))
SR_MAX_DATA_AGE_MIN = max(30.0, float(os.environ.get("AJAY_SR_MAX_DATA_AGE_MIN", "190")))
SR_SPOT_EXCHANGE_INFO_TTL_SEC = max(30.0, float(os.environ.get("AJAY_SR_SPOT_EXCHANGE_INFO_TTL_SEC", "900")))
SR_SPOT_EXCHANGE_INFO_URL = f"{SR_SPOT_BASE_URL}/api/v3/exchangeInfo"
# Quantity-prefixed Binance perpetuals (e.g. 1000SHIBUSDT) can use the
# underlying Spot pair for the 1H source candles aggregated into 3H S/R. Keep the prefix vocabulary deliberately
# narrow so ordinary symbols such as 1INCH are never treated as multipliers.
SR_NUMERIC_PREFIXES: tuple[int, ...] = (1_000_000, 100_000, 10_000, 1_000)

_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()
_SPOT_INFO_CACHE: dict[str, Any] = {"ts": 0.0, "symbols": {}}
_SPOT_INFO_LOCK = threading.Lock()
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


class SRSymbolUnavailableError(RuntimeError):
    """The requested logical symbol has no unambiguous active Binance Spot source."""


def _spot_symbol_catalog(*, force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    global _LAST_REQUEST_MONOTONIC
    now = time.monotonic()
    with _SPOT_INFO_LOCK:
        if not force_refresh and _SPOT_INFO_CACHE["symbols"] and now - float(_SPOT_INFO_CACHE["ts"]) < SR_SPOT_EXCHANGE_INFO_TTL_SEC:
            return dict(_SPOT_INFO_CACHE["symbols"])

    with _REQUEST_LOCK:
        now_req = time.monotonic()
        wait = SR_REQUEST_MIN_INTERVAL_SEC - (now_req - _LAST_REQUEST_MONOTONIC)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_MONOTONIC = time.monotonic()
    response = _SESSION.get(SR_SPOT_EXCHANGE_INFO_URL, timeout=SR_HTTP_TIMEOUT_SEC)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("symbols") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Binance SPOT exchangeInfo response has no symbols list")

    symbols: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol", "")).strip().upper()
        if not sym:
            continue
        if str(row.get("status", "")).strip().upper() != "TRADING":
            continue
        if str(row.get("quoteAsset", "")).strip().upper() != "USDT":
            continue
        if row.get("isSpotTradingAllowed") is False:
            continue
        symbols[sym] = row

    with _SPOT_INFO_LOCK:
        _SPOT_INFO_CACHE["ts"] = time.monotonic()
        _SPOT_INFO_CACHE["symbols"] = dict(symbols)
    return symbols


def clear_spot_symbol_cache() -> None:
    with _SPOT_INFO_LOCK:
        _SPOT_INFO_CACHE["ts"] = 0.0
        _SPOT_INFO_CACHE["symbols"] = {}


def _logical_base(symbol: str) -> str:
    normalized = normalize_spot_symbol(symbol)
    return normalized[:-4]


def _numeric_prefix_candidate(base: str) -> tuple[int, str] | None:
    # Longest first, so a future 1000000TOKEN cannot be mistaken for 1000TOKEN.
    for multiplier in sorted(SR_NUMERIC_PREFIXES, reverse=True):
        prefix = str(multiplier)
        if base.startswith(prefix) and len(base) > len(prefix):
            underlying = base[len(prefix):]
            return multiplier, underlying
    return None


def resolve_spot_symbol(symbol: str, *, force_refresh: bool = False) -> dict[str, Any]:
    """Resolve a logical engine symbol to an active Binance Spot USDT symbol.

    Resolution is venue-local and data-driven:
    1. exact Spot symbol wins;
    2. a known quantity-prefixed logical symbol may use its unprefixed Spot pair;
    3. an unprefixed logical asset may use exactly one matching numeric-prefixed
       Spot source; a numeric prefix on Binance USDⓈ-M contracts changes the
       contract's underlying quantity/index scaling, so the Spot source must be
       converted into the logical futures price unit;
    4. ambiguity or absence fails closed.

    For quantity-prefixed Binance perpetuals such as 1000SHIBUSDT, the contract
    represents 1,000 SHIB and its quoted/index price is 1,000x the SHIB/USDT
    Spot index. Therefore a Spot alias such as SHIBUSDT uses price_scale=1000
    and volume_scale=0.001 to express the snapshot in the logical 1000SHIB unit.
    No BingX contract is consulted here and no cross-venue alias is inferred.
    """
    requested = normalize_spot_symbol(symbol)
    base = requested[:-4]
    catalog = _spot_symbol_catalog(force_refresh=force_refresh)

    exact = catalog.get(requested)
    if exact is not None:
        return {
            "requested_symbol": requested,
            "source_symbol": requested,
            "underlying_asset": str(exact.get("baseAsset") or base).upper(),
            "source_price_scale": 1.0,
            "source_volume_scale": 1.0,
            "source_contract_multiplier": 1.0,
            "source_alias_kind": "EXACT",
        }

    pref = _numeric_prefix_candidate(base)
    if pref is not None:
        multiplier, underlying = pref
        source = f"{underlying}USDT"
        row = catalog.get(source)
        if row is not None:
            scale = float(multiplier)
            return {
                "requested_symbol": requested,
                "source_symbol": source,
                "underlying_asset": str(row.get("baseAsset") or underlying).upper(),
                "source_price_scale": scale,
                "source_volume_scale": 1.0 / scale,
                "source_contract_multiplier": scale,
                "source_alias_kind": "NUMERIC_PREFIX_UNDERLYING",
            }

    prefixed_sources: list[tuple[int, str, dict[str, Any]]] = []
    for multiplier in SR_NUMERIC_PREFIXES:
        source = f"{multiplier}{base}USDT"
        row = catalog.get(source)
        if row is not None:
            prefixed_sources.append((multiplier, source, row))
    if len(prefixed_sources) == 1:
        multiplier, source, row = prefixed_sources[0]
        scale = 1.0 / float(multiplier)
        return {
            "requested_symbol": requested,
            "source_symbol": source,
            "underlying_asset": str(row.get("baseAsset") or base).upper(),
            "source_price_scale": scale,
            "source_volume_scale": float(multiplier),
            "source_contract_multiplier": float(multiplier),
            "source_alias_kind": "PLAIN_NUMERIC_PREFIX_SOURCE",
        }
    if len(prefixed_sources) > 1:
        names = ", ".join(item[1] for item in prefixed_sources)
        raise SRSymbolUnavailableError(
            f"Ambiguous Binance Spot source for {requested}: multiple numeric-prefix pairs: {names}"
        )

    raise SRSymbolUnavailableError(f"No active Binance Spot USDT source for logical symbol {requested}")


def _rate_limit_before_request() -> None:
    global _LAST_REQUEST_MONOTONIC
    with _REQUEST_LOCK:
        now = time.monotonic()
        wait = SR_REQUEST_MIN_INTERVAL_SEC - (now - _LAST_REQUEST_MONOTONIC)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_MONOTONIC = time.monotonic()


def _fetch_closed_spot(
    symbol: str,
    *,
    interval: str,
    resolved: dict[str, Any] | None = None,
    limit: int | None = None,
) -> tuple[list[Candle], dict[str, Any]]:
    """Fetch fully closed Spot klines for a supported interval.

    Binance Spot's supported raw source interval is ``1h``. The trading snapshot
    aggregates these rows into exact, closed 3h bars in ``_fetch_closed_3h_spot``.
    """
    interval_key = str(interval or "").strip().lower()
    interval_ms = {"1h": SR_SOURCE_INTERVAL_MS}.get(interval_key)
    if interval_ms is None:
        raise ValueError(f"Unsupported raw Binance Spot S/R source interval: {interval!r}; use 1h and aggregate to 3h")

    resolved = dict(resolved or resolve_spot_symbol(symbol))
    kline_limit = SR_SOURCE_KLINE_LIMIT_1H if limit is None else max(300, min(1000, int(limit)))
    for attempt in range(2):
        _rate_limit_before_request()
        payload_response = _SESSION.get(
            SR_SPOT_KLINES_URL,
            params={"symbol": resolved["source_symbol"], "interval": interval_key, "limit": kline_limit},
            timeout=SR_HTTP_TIMEOUT_SEC,
        )
        if payload_response.status_code == 400 and attempt == 0:
            clear_spot_symbol_cache()
            resolved = resolve_spot_symbol(symbol, force_refresh=True)
            continue
        payload_response.raise_for_status()
        rows = payload_response.json()
        if not isinstance(rows, list):
            raise ValueError("Binance SPOT klines response is not a list")
        now_ms = int(time.time() * 1000)
        candles: list[Candle] = []
        price_scale = float(resolved.get("source_price_scale", 1.0))
        volume_scale = float(resolved.get("source_volume_scale", 1.0))
        if not math.isfinite(price_scale) or price_scale <= 0 or not math.isfinite(volume_scale) or volume_scale <= 0:
            raise ValueError(f"Invalid Spot source scales for {resolved}")
        previous_open_time: int | None = None
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue
            try:
                open_time = int(row[0])
                close_time = int(row[6]) if len(row) > 6 else open_time + interval_ms - 1
            except (TypeError, ValueError, OverflowError):
                continue
            if close_time > now_ms:
                continue
            if previous_open_time is not None and open_time <= previous_open_time:
                raise ValueError(
                    f"Binance SPOT {interval_key} klines are duplicated or out of order for {resolved['source_symbol']}"
                )
            try:
                candle = Candle(
                    ts=open_time,
                    open=float(row[1]) * price_scale,
                    high=float(row[2]) * price_scale,
                    low=float(row[3]) * price_scale,
                    close=float(row[4]) * price_scale,
                    volume=float(row[5]) * volume_scale,
                )
            except (TypeError, ValueError, OverflowError):
                continue
            values = (candle.open, candle.high, candle.low, candle.close, candle.volume)
            if (
                not all(math.isfinite(value) for value in values)
                or min(candle.open, candle.high, candle.low, candle.close) <= 0
                or candle.volume < 0
                or candle.high < max(candle.open, candle.close, candle.low)
                or candle.low > min(candle.open, candle.close, candle.high)
            ):
                continue
            candles.append(candle)
            previous_open_time = open_time
        if len(candles) < SR_PRD + SR_RB + 2:
            raise ValueError(
                f"Only {len(candles)} closed SPOT {interval_key.upper()} candles available from {resolved['source_symbol']}"
            )
        if any(b.ts <= a.ts for a, b in zip(candles, candles[1:])):
            raise ValueError(f"Non-monotonic Binance SPOT {interval_key} candles for {resolved['source_symbol']}")
        return candles, resolved
    raise RuntimeError(f"Spot kline retry exhausted for {symbol}/{interval_key}")


def _aggregate_closed_3h_candles(hourly_candles: list[Candle]) -> list[Candle]:
    """Aggregate closed 1h Spot candles into exact UTC-aligned 3h candles.

    A 3h bar is emitted only when all three expected 1h open timestamps exist.
    Incomplete groups are discarded; OHLCV aggregation is first-open, max-high,
    min-low, last-close and summed base volume.
    """
    groups: dict[int, dict[int, Candle]] = {}
    previous_ts: int | None = None
    for candle in hourly_candles:
        ts = int(candle.ts)
        if previous_ts is not None and ts <= previous_ts:
            raise ValueError("Binance Spot 1h candles are duplicated or out of order before 3h aggregation")
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


def _fetch_closed_3h_spot(
    symbol: str,
    *,
    resolved: dict[str, Any] | None = None,
    limit: int | None = None,
) -> tuple[list[Candle], dict[str, Any]]:
    """Build the trading 3h S/R series from supported Binance Spot 1h klines."""
    hourly_candles, meta = _fetch_closed_spot(
        symbol, interval="1h", resolved=resolved, limit=limit
    )
    minimum = SR_PRD + SR_RB + 2
    minimum_hourly = minimum * 3
    if limit is not None and int(limit) < minimum_hourly:
        raise ValueError(
            f"Requested {limit} source 1H candles cannot build {minimum} closed 3H bars; "
            f"at least {minimum_hourly} source 1H candles are required"
        )
    candles_3h = _aggregate_closed_3h_candles(hourly_candles)
    if len(candles_3h) < minimum:
        raise ValueError(
            f"Only {len(candles_3h)} complete closed SPOT 3H candles can be aggregated from "
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
            "Incomplete 3H Spot history: missing one or more UTC-aligned 3H buckets "
            f"between {first_left} and {first_right} for {meta.get('source_symbol')}"
        )
    return candles_3h, meta


def _fetch_closed_1h_spot(
    symbol: str,
    *,
    resolved: dict[str, Any] | None = None,
    limit: int | None = None,
) -> tuple[list[Candle], dict[str, Any]]:
    """Legacy diagnostic helper; the trading S/R snapshot does not call this."""
    return _fetch_closed_spot(symbol, interval="1h", resolved=resolved, limit=limit)


def get_spot_current_price(symbol: str, *, snapshot: dict[str, Any] | None = None) -> float:
    """Get the current Binance Spot ticker price in the logical symbol's unit."""
    if isinstance(snapshot, dict) and snapshot.get("source_symbol"):
        resolved = {
            "source_symbol": str(snapshot["source_symbol"]),
            "source_price_scale": snapshot.get("source_price_scale", 1.0),
        }
    else:
        resolved = resolve_spot_symbol(symbol)
    _rate_limit_before_request()
    response = _SESSION.get(
        SR_SPOT_TICKER_PRICE_URL,
        params={"symbol": resolved["source_symbol"]},
        timeout=SR_HTTP_TIMEOUT_SEC,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Binance SPOT ticker response is not an object")
    raw_price = _safe_price(payload.get("price"))
    scale = float(resolved.get("source_price_scale", 1.0))
    if raw_price is None or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid Binance SPOT ticker price for {resolved['source_symbol']}: {payload!r}")
    return raw_price * scale


def get_cached_sr_snapshot(symbol: str, *, limit: int | None = None) -> dict[str, Any]:
    """Fetch/cache the current Pine SR state; a custom limit is diagnostic-only and uncached."""
    key = normalize_spot_symbol(symbol)
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

    fetched = _fetch_closed_3h_spot(symbol, limit=limit) if limit is not None else _fetch_closed_3h_spot(symbol)
    if isinstance(fetched, tuple) and len(fetched) == 2:
        candles, resolved = fetched
    else:
        # Backward-compatible test seam: older tests may stub the fetcher with
        # a candle list. Production always returns (candles, resolver metadata).
        candles = fetched
        resolved = {
            "requested_symbol": normalize_spot_symbol(symbol),
            "source_symbol": normalize_spot_symbol(symbol),
            "underlying_asset": _logical_base(symbol),
            "source_price_scale": 1.0,
            "source_volume_scale": 1.0,
            "source_contract_multiplier": 1.0,
            "source_alias_kind": "EXACT_TEST_STUB",
        }
    latest_ts = candles[-1].ts
    state = compute_current_sr(
        candles,
        rb=SR_RB,
        prd=SR_PRD,
        channel_w=SR_CHANNEL_W,
        strength_sr=SR_STRENGTH,
    )
    latest_age_min = max(0.0, (time.time() * 1000 - (latest_ts + SR_INTERVAL_MS - 1)) / 60_000.0)
    if latest_age_min > SR_MAX_DATA_AGE_MIN:
        raise ValueError(f"SR SPOT data stale: latest_closed_age={latest_age_min:.2f}m > {SR_MAX_DATA_AGE_MIN:.2f}m")

    snapshot = dict(state)
    snapshot.update({
        "source": "binance_spot",
        "timeframe": SR_TIMEFRAME,
        "interval_ms": SR_INTERVAL_MS,
        "closed_candle_count_3h": len(candles),
        "source_symbol": resolved["source_symbol"],
        "source_symbol_requested": resolved["requested_symbol"],
        "source_price_scale": resolved["source_price_scale"],
        "source_volume_scale": resolved["source_volume_scale"],
        "source_contract_multiplier": resolved["source_contract_multiplier"],
        "source_alias_kind": resolved["source_alias_kind"],
        "underlying_asset": resolved["underlying_asset"],
        "latest_closed_timestamp": latest_ts,
        "latest_closed_age_min": latest_age_min,
        "zone_scale": SR_ZONE_SCALE,
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



def evaluate_sr_directional_zone(
    snapshot: dict[str, Any],
    *,
    current_price: float,
    direction: str,
) -> dict[str, Any]:
    """Hard directional S/R gate for a closed 3H Binance Spot snapshot.

    A pivot-low zone (``L`` / Demand) permits LONG only; a pivot-high zone
    (``H`` / Supply) permits SHORT only. Only unbroken zones on the actionable
    side of price are candidates. The nearest candidate zone decides the current
    directional regime. Mixed-type ties are ambiguous and fail closed.

    Distance is measured from the live Spot price to the nearest zone boundary,
    divided by the live Spot price. Being inside the zone has distance zero.
    """
    d = str(direction or "").strip().upper()
    if d not in {"LONG", "SHORT"}:
        raise ValueError(f"Invalid direction={direction!r}")
    price = _safe_price(current_price)
    if price is None:
        raise ValueError(f"Invalid Binance Spot current price={current_price!r}")
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
        "source": snapshot.get("source", "binance_spot"),
        "source_symbol": snapshot.get("source_symbol"),
        "source_symbol_requested": snapshot.get("source_symbol_requested"),
        "source_price_scale": snapshot.get("source_price_scale", 1.0),
        "source_volume_scale": snapshot.get("source_volume_scale", 1.0),
        "source_contract_multiplier": snapshot.get("source_contract_multiplier", 1.0),
        "source_alias_kind": snapshot.get("source_alias_kind"),
        "underlying_asset": snapshot.get("underlying_asset"),
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
