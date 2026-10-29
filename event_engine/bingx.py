# bingx.py

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from decimal import (
    Decimal,
    ROUND_DOWN,
    ROUND_FLOOR,
)
from typing import Any
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("event_engine.bingx")


class BingXRateLimitError(RuntimeError):
    """A BingX rate-limit response that must not be blind-retried.

    ``retry_after_ms`` is the server-provided absolute retry timestamp when
    BingX includes one in the error message.
    """

    def __init__(self, message: str, *, code: int | str | None = None, retry_after_ms: int | None = None):
        super().__init__(message)
        self.code = code
        self.retry_after_ms = retry_after_ms

API_KEY = os.environ.get("BINGX_API_KEY", "").strip()
SECRET_KEY = os.environ.get("BINGX_SECRET_KEY", "").strip()
BASE_URL = os.environ.get("BINGX_BASE_URL", "https://open-api-vst.bingx.com").rstrip("/")
MARGIN_USDT = float(os.environ.get("BINGX_MARGIN_USDT", "1"))
REQUIRE_MULTI_TP = os.environ.get("REQUIRE_MULTI_TP", os.environ.get("REJECT_SINGLE_TP", "false")).strip().lower() == "true"
LEVERAGE = int(os.environ.get("BINGX_LEVERAGE", "10"))
MAX_LEVERAGE = int(os.environ.get("BINGX_MAX_LEVERAGE", "10"))

SYMBOL_MAP = {}
try:
    SYMBOL_MAP = json.loads(os.environ.get("BINGX_SYMBOL_MAP", "{}"))
except Exception:
    SYMBOL_MAP = {}

CONTRACTS_PATH = "/openApi/swap/v2/quote/contracts"
KLINE_PATH = "/openApi/swap/v3/quote/klines"
LIVE_PRICE_PATH = "/openApi/swap/v1/ticker/price"
ORDER_PATH = "/openApi/swap/v2/trade/order"
POSITION_PATH = os.environ.get("BINGX_POSITIONS_PATH", "/openApi/swap/v2/user/positions")
LEVERAGE_PATH = "/openApi/swap/v2/trade/leverage"
OPEN_ORDERS_PATH = "/openApi/swap/v2/trade/openOrders"
OPEN_ORDERS_COOLDOWN_STATE_PATH = Path(
    os.environ.get(
        "BINGX_OPEN_ORDERS_COOLDOWN_PATH",
        str(Path(os.environ.get("DIVERGENCE_DATA_DIR", Path(__file__).resolve().parents[1] / "data")) / "bingx_open_orders_cooldown.json"),
    )
)
OPEN_ORDERS_COOLDOWN_FALLBACK_SEC = 900.0
_OPEN_ORDERS_COOLDOWN_UNTIL_MS = 0
_OPEN_ORDERS_COOLDOWN_REASON = ""
FILL_HISTORY_PATH = "/openApi/swap/v2/trade/fillHistory"
INCOME_HISTORY_PATH = "/openApi/swap/v2/user/income"
POSITION_MODE_PATH = "/openApi/swap/v1/positionSide/dual"
POSITION_MODE = os.environ.get("BINGX_POSITION_MODE", "HEDGE").strip().upper()

CACHE = {
    "ts": 0.0,
    "data": {},
    "by_display_name": {},
}
TTL = 3600
SERVER_TIME_OFFSET_MS = 0

# Live-price requests are used by the tracker and by execution sizing. A bad
# symbol must never be allowed to fan out into repeated invalid requests that
# trigger BingX's 109429 temporary restriction. Keep this protection local to
# the current process; a fresh engine cycle re-validates the live endpoint.
LIVE_PRICE_INVALID_SYMBOL_COOLDOWN_SEC = max(30.0, float(os.environ.get("BINGX_LIVE_PRICE_INVALID_SYMBOL_COOLDOWN_SEC", "900")))
LIVE_PRICE_RATE_LIMIT_FALLBACK_SEC = max(30.0, float(os.environ.get("BINGX_LIVE_PRICE_RATE_LIMIT_FALLBACK_SEC", "900")))
_LIVE_PRICE_INVALID_UNTIL: dict[str, float] = {}
_LIVE_PRICE_COOLDOWN_UNTIL = 0.0

SESSION = requests.Session()
# Fast-fail session used by reconciliation/health-sensitive GETs. It deliberately
# has no urllib3 retry adapter so a single stalled request cannot consume several
# consecutive 10-second retry windows before the reconciliation loop can continue.
FAST_SESSION = requests.Session()

_adapter = HTTPAdapter(
    pool_connections=20,
    pool_maxsize=20,
    max_retries=Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    ),
)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Parse a finite numeric value without allowing malformed API fields to raise.

    ``open_market`` uses this while resolving an acknowledgement whose order ID is
    missing. Keeping the helper local to the exchange client prevents a malformed
    executed-quantity field from raising NameError before the live-position safety
    path can run.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _sign(params: dict[str, Any]) -> str:
    qs = urlencode(params)
    return hmac.new(SECRET_KEY.encode(), qs.encode(), hashlib.sha256).hexdigest()


def _apply_request_timestamp(params: dict[str, Any]) -> None:
    params.pop("signature", None)
    params["timestamp"] = str(int(time.time() * 1000) + SERVER_TIME_OFFSET_MS)
    params["signature"] = _sign(params)


def _update_server_time_offset(response: requests.Response) -> bool:
    global SERVER_TIME_OFFSET_MS
    date_header = response.headers.get("Date")
    if not date_header:
        return False
    try:
        server_ms = int(parsedate_to_datetime(date_header).timestamp() * 1000)
    except (TypeError, ValueError, OverflowError):
        return False

    local_ms = int(time.time() * 1000)
    SERVER_TIME_OFFSET_MS = server_ms - local_ms
    log.warning("[BINGX] Time sync offset_ms=%d", SERVER_TIME_OFFSET_MS)
    return True


def _parse_open_orders_unblock_ms(message: object) -> int | None:
    """Parse BingX's absolute endpoint cooldown timestamp when supplied."""
    text = str(message or "")
    match = re.search(
        r"(?:unblocked\s+after|retry\s+after\s+time\s*:)[\s:=]*(\d{13})",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError, OverflowError):
        return None


def _read_open_orders_cooldown(now_ms: int | None = None) -> tuple[int, str] | None:
    """Load the openOrders cooldown across workflow processes; never query while active."""
    global _OPEN_ORDERS_COOLDOWN_UNTIL_MS, _OPEN_ORDERS_COOLDOWN_REASON
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    until_ms = int(_OPEN_ORDERS_COOLDOWN_UNTIL_MS or 0)
    reason = _OPEN_ORDERS_COOLDOWN_REASON
    try:
        if OPEN_ORDERS_COOLDOWN_STATE_PATH.exists():
            state = json.loads(OPEN_ORDERS_COOLDOWN_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(state, dict) and state.get("base_url", BASE_URL) == BASE_URL:
                stored_until = int(state.get("cooldown_until_ms", 0) or 0)
                if stored_until > until_ms:
                    until_ms = stored_until
                    reason = str(state.get("reason") or reason or "BingX code 100410")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # Cooldown persistence is best-effort; the in-process deadline still applies.
        pass
    if until_ms > now_ms:
        _OPEN_ORDERS_COOLDOWN_UNTIL_MS = until_ms
        _OPEN_ORDERS_COOLDOWN_REASON = reason
        return until_ms, reason
    if until_ms:
        _OPEN_ORDERS_COOLDOWN_UNTIL_MS = 0
        _OPEN_ORDERS_COOLDOWN_REASON = ""
        try:
            state = json.loads(OPEN_ORDERS_COOLDOWN_STATE_PATH.read_text(encoding="utf-8")) if OPEN_ORDERS_COOLDOWN_STATE_PATH.exists() else {}
            if not isinstance(state, dict) or state.get("base_url", BASE_URL) == BASE_URL:
                OPEN_ORDERS_COOLDOWN_STATE_PATH.unlink(missing_ok=True)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return None


def _persist_open_orders_cooldown(until_ms: int, reason: str) -> None:
    """Persist a rate-limit deadline atomically so the next workflow run respects it."""
    global _OPEN_ORDERS_COOLDOWN_UNTIL_MS, _OPEN_ORDERS_COOLDOWN_REASON
    _OPEN_ORDERS_COOLDOWN_UNTIL_MS = max(int(until_ms), int(_OPEN_ORDERS_COOLDOWN_UNTIL_MS or 0))
    _OPEN_ORDERS_COOLDOWN_REASON = str(reason or "BingX code 100410")
    path = OPEN_ORDERS_COOLDOWN_STATE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump({
                "base_url": BASE_URL,
                "endpoint": OPEN_ORDERS_PATH,
                "cooldown_until_ms": _OPEN_ORDERS_COOLDOWN_UNTIL_MS,
                "reason": _OPEN_ORDERS_COOLDOWN_REASON,
                "updated_at_ms": int(time.time() * 1000),
            }, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        try:
            directory_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except OSError as exc:
        log.warning("[BINGX] Could not persist openOrders rate-limit cooldown: %s", exc)


def _clear_open_orders_cooldown() -> None:
    global _OPEN_ORDERS_COOLDOWN_UNTIL_MS, _OPEN_ORDERS_COOLDOWN_REASON
    _OPEN_ORDERS_COOLDOWN_UNTIL_MS = 0
    _OPEN_ORDERS_COOLDOWN_REASON = ""
    try:
        state = json.loads(OPEN_ORDERS_COOLDOWN_STATE_PATH.read_text(encoding="utf-8")) if OPEN_ORDERS_COOLDOWN_STATE_PATH.exists() else {}
        if not isinstance(state, dict) or state.get("base_url", BASE_URL) == BASE_URL:
            OPEN_ORDERS_COOLDOWN_STATE_PATH.unlink(missing_ok=True)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass


def _request(
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    signed: bool = True,
    *,
    timeout_sec: float | None = None,
    retryable: bool = True,
):
    is_open_orders = method.upper() == "GET" and path == OPEN_ORDERS_PATH
    if is_open_orders:
        cooldown = _read_open_orders_cooldown()
        if cooldown is not None:
            until_ms, reason = cooldown
            return {
                "code": 100410,
                "msg": f"The endpoint trigger frequency limit rule is currently in the disabled period and will be unblocked after {until_ms}",
                "retry_after_ms": until_ms,
                "local_cooldown": True,
                "cooldown_reason": reason,
            }

    base_params = dict(params or {})
    try:
        request_timeout = float(timeout_sec) if timeout_sec is not None else float(os.environ.get("BINGX_HTTP_TIMEOUT_SEC", "10"))
    except (TypeError, ValueError):
        request_timeout = 10.0
    request_timeout = max(1.0, min(request_timeout, 60.0))
    session = SESSION if retryable else FAST_SESSION
    headers = {}
    max_timestamp_retries = 1 if signed else 0

    if signed:
        if not API_KEY or not SECRET_KEY:
            return {"code": -1, "msg": "missing BingX credentials"}
        headers["X-BX-APIKEY"] = API_KEY

    for attempt in range(max_timestamp_retries + 1):
        request_params = dict(base_params)
        if signed:
            _apply_request_timestamp(request_params)

        try:
            response = session.request(
                method=method,
                url=BASE_URL + path,
                params=request_params,
                headers=headers,
                timeout=request_timeout,
            )
            payload = response.json()
        except Exception as exc:
            return {"code": -1, "msg": str(exc)}

        try:
            code = int(payload.get("code"))
        except (TypeError, ValueError, AttributeError):
            code = None

        if is_open_orders and code == 100410:
            now_ms = int(time.time() * 1000)
            retry_after_ms = _parse_open_orders_unblock_ms(payload.get("msg") if isinstance(payload, dict) else "")
            if retry_after_ms is None or retry_after_ms <= now_ms:
                retry_after_ms = now_ms + int(OPEN_ORDERS_COOLDOWN_FALLBACK_SEC * 1000)
            _persist_open_orders_cooldown(retry_after_ms, str(payload.get("msg") or "BingX code 100410") if isinstance(payload, dict) else "BingX code 100410")
            if isinstance(payload, dict):
                payload = dict(payload)
                payload["retry_after_ms"] = retry_after_ms
                payload["rate_limited"] = True
        elif is_open_orders and code == 0:
            _clear_open_orders_cooldown()

        if signed and code == 109400 and attempt < max_timestamp_retries and _update_server_time_offset(response):
            continue

        return payload

    return {"code": -1, "msg": "request retry exhausted"}


def refresh_contracts() -> dict[str, Any]:
    resp = _request("GET", CONTRACTS_PATH, signed=False)
    if resp.get("code") not in (0, "0"):
        raise RuntimeError(f"[BINGX] Contracts error: {resp.get('msg')}")

    data = {}
    by_name = {}

    for c in resp.get("data", []) or []:
        sym = str(c.get("symbol", "")).strip().upper()
        name = str(c.get("displayName", "")).strip().upper()
        if sym:
            data[sym] = c
        if name:
            by_name[name] = c

    CACHE.update(ts=time.time(), data=data, by_display_name=by_name)
    log.info("[BINGX] Active contracts=%d", len(data))
    return data


def contracts(*, require_fresh: bool = False) -> dict[str, dict]:
    cache_fresh = bool(CACHE["data"]) and time.time() - CACHE["ts"] < TTL
    if cache_fresh and not require_fresh:
        return CACHE["data"]
    try:
        return refresh_contracts()
    except Exception:
        return {} if require_fresh else CACHE["data"]


def get_contract(symbol: str, *, require_fresh: bool = False) -> dict | None:
    s = (symbol or "").strip().upper().replace("_", "-").replace("/", "-")
    if not s:
        return None
    # Compact quote symbols (e.g. BTCUSDT) are common in logs and external data.
    # Normalize them before catalog lookup so fallback cannot form BTCUSDT-USDT.
    if "-" not in s and s.endswith("USDT") and len(s) > 4:
        s = f"{s[:-4]}-USDT"

    catalog = contracts(require_fresh=require_fresh)
    mapped = SYMBOL_MAP.get(s)
    if mapped:
        c = catalog.get(str(mapped).strip().upper())
        if c:
            return c

    direct = s if s.endswith("-USDT") else f"{s.replace('-', '')}-USDT"
    c = catalog.get(direct)
    if c:
        return c

    base = s.replace("-USDT", "").replace("-", "")
    for c in catalog.values():
        cs = str(c.get("symbol", "")).upper()
        if cs == f"{base}-USDT" or cs == base:
            return c

    norm_base = base.replace("-", "").replace("/", "").replace(" ", "")
    for c in catalog.values():
        name = str(c.get("displayName", "")).upper().replace("-", "").replace("/", "").replace(" ", "")
        if name == f"{norm_base}USDT" or name == norm_base:
            return c

    return {str(c.get("displayName", "")).strip().upper(): c for c in catalog.values() if c.get("displayName")}.get(f"{base}-USDT")



def to_bx_symbol(symbol: str) -> str | None:
    c = get_contract(symbol)
    if not c:
        return None
    return str(c.get("symbol", "")).upper()


def contract_exists(symbol: str) -> bool:
    c = get_contract(symbol)
    if not isinstance(c, dict):
        return False
    status_ok = str(c.get("status", "")).strip() in {"1", "1.0"}
    api_open = str(c.get("apiStateOpen", "")).strip().lower() == "true"
    return bool(status_ok and api_open)


def fetch_klines(
    symbol: str,
    interval: str,
    limit: int = 250,
    *,
    timeout_sec: float | None = None,
    retryable: bool = True,
) -> list[dict]:
    bx = to_bx_symbol(symbol)
    if not bx:
        raise ValueError(f"[BINGX] No contract found for {symbol}")

    resp = _request(
        "GET",
        KLINE_PATH,
        {"symbol": bx, "interval": interval, "limit": limit},
        signed=False,
        timeout_sec=timeout_sec,
        retryable=retryable,
    )
    code = resp.get("code")
    if code not in (0, "0"):
        msg = str(resp.get("msg") or "")
        try:
            code_int = int(code)
        except (TypeError, ValueError):
            code_int = None

        # BingX Kline limits can surface as a 109429 wrapper whose message
        # contains the backend 109415 counter (for example, ">5 ... requests
        # within 900000 ms"). Treat both forms as a hard cooldown condition.
        rate_limited = (
            code_int == 109429
            or ("109415" in msg and "requests within" in msg.lower())
        )
        if rate_limited:
            import re

            retry_after_ms = None
            match = re.search(r"retry\s+after\s+time\s*:\s*(\d{13})", msg, flags=re.IGNORECASE)
            if match:
                try:
                    retry_after_ms = int(match.group(1))
                except (TypeError, ValueError):
                    retry_after_ms = None
            raise BingXRateLimitError(
                f"[BINGX] Klines rate limited {bx}/{interval}: code={code} msg={msg}",
                code=code_int if code_int is not None else code,
                retry_after_ms=retry_after_ms,
            )

        raise RuntimeError(f"[BINGX] Klines error {bx}/{interval}: code={code} msg={msg}")

    rows = resp.get("data") or []
    out: list[dict] = []
    now_ms = int(time.time() * 1000) + SERVER_TIME_OFFSET_MS

    duration_ms = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
        "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000,
    }.get(interval)

    for row in rows:
        if isinstance(row, (list, tuple)):
            if len(row) < 6:
                continue
            try:
                open_time = int(row[0])
                open_price = float(row[1])
                high = float(row[2])
                low = float(row[3])
                close = float(row[4])
                volume = float(row[5])
                close_time = int(row[6]) if (len(row) >= 7 and row[6] is not None) else (open_time + duration_ms - 1 if duration_ms else open_time)
                quote_volume = float(row[7]) if (len(row) >= 8 and row[7] is not None) else None
                taker_buy_base = float(row[9]) if (len(row) >= 10 and row[9] is not None) else None
                taker_buy_quote = float(row[10]) if (len(row) >= 11 and row[10] is not None) else None
            except (TypeError, ValueError, IndexError):
                continue
        elif isinstance(row, dict):
            def pick(*names):
                for name in names:
                    if name in row and row[name] is not None:
                        return row[name]
                return None
            try:
                open_time = int(pick("openTime", "open_time", "time"))
                open_price = float(pick("open"))
                high = float(pick("high"))
                low = float(pick("low"))
                close = float(pick("close"))
                volume = float(pick("volume"))
                raw_close_time = pick("closeTime", "close_time")
                close_time = int(raw_close_time) if raw_close_time is not None else (open_time + duration_ms - 1 if duration_ms else open_time)
                quote_volume_raw = pick("quoteAssetVolume", "quoteVolume", "quote_volume")
                taker_base_raw = pick("takerBuyBaseVolume", "taker_buy_base", "takerBuyBase", "buyVolume")
                taker_quote_raw = pick("takerBuyQuoteVolume", "taker_buy_quote", "takerBuyQuote", "buyQuoteVolume")
                quote_volume = float(quote_volume_raw) if quote_volume_raw is not None else None
                taker_buy_base = float(taker_base_raw) if taker_base_raw is not None else None
                taker_buy_quote = float(taker_quote_raw) if taker_quote_raw is not None else None
            except (TypeError, ValueError, KeyError):
                continue
        else:
            continue

        if close_time > now_ms:
            continue
        if open_price <= 0 or high <= 0 or low <= 0 or close <= 0 or volume < 0:
            continue
        if high < low or high < open_price or high < close or low > open_price or low > close:
            continue

        taker_flow_valid = (
            quote_volume is not None and taker_buy_base is not None and taker_buy_quote is not None
            and quote_volume >= 0 and taker_buy_base >= 0 and taker_buy_quote >= 0
            and taker_buy_base <= volume * 1.001 + 1e-8
            and taker_buy_quote <= quote_volume * 1.001 + 1e-8
        )
        bar_delta_usdt = 2.0 * taker_buy_quote - quote_volume if taker_flow_valid else None

        out.append(
            {
                "open_time": open_time,
                "close_time": close_time,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "quote_volume": quote_volume,
                "taker_buy_base": taker_buy_base,
                "taker_buy_quote": taker_buy_quote,
                "taker_flow_valid": taker_flow_valid,
                "bar_delta_usdt": bar_delta_usdt,
            }
        )

    out.sort(key=lambda x: x["close_time"])
    deduped = []
    seen_close_times = set()

    for bar in out:
        ct = bar["close_time"]
        if ct in seen_close_times:
            continue
        seen_close_times.add(ct)
        deduped.append(bar)

    return deduped


def _set_leverage(bx_symbol: str, leverage: int, direction: str = "LONG") -> bool:
    d = direction.upper() if direction else "LONG"
    sides = (d, "BOTH") if d in {"LONG", "SHORT"} else ("BOTH",)

    for side in sides:
        resp = _request("POST", LEVERAGE_PATH, {"symbol": bx_symbol, "side": side, "leverage": str(leverage)})
        if resp.get("code") in (0, "0"):
            return True
    return False


def _extract_response_rows(
    resp: dict,
    *,
    collection_keys: tuple[str, ...],
    single_row_markers: tuple[str, ...],
    context: str,
) -> list[dict]:
    """Extract a documented collection shape without treating schema drift as empty.

    An empty list is accepted only when the response explicitly represents a
    collection. An unexpected ``data`` object, scalar, or malformed row raises
    so callers cannot mistake an unreadable API response for no positions/orders.
    """
    if not isinstance(resp, dict):
        raise ValueError(f"{context}: response is not an object")
    data = resp.get("data")
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = None
        for key in collection_keys:
            if key not in data:
                continue
            value = data.get(key)
            if isinstance(value, list):
                rows = value
                break
            if isinstance(value, dict):
                rows = [value]
                break
            raise ValueError(f"{context}: data.{key} is not a list/object")
        if rows is None and any(key in data for key in single_row_markers):
            rows = [data]
        if rows is None:
            raise ValueError(f"{context}: unrecognized data shape")
    else:
        raise ValueError(f"{context}: missing or invalid data collection")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{context}: collection contains non-object row")
    return rows


def _normalize_orders_list(resp: dict) -> list[dict]:
    """Compatibility wrapper; strict callers should handle malformed shapes."""
    return _extract_response_rows(
        resp,
        collection_keys=("orders", "order", "list", "rows"),
        single_row_markers=("orderId", "orderID", "clientOrderId", "type", "stopPrice"),
        context="openOrders",
    )


def get_positions(*, timeout_sec: float | None = None, retryable: bool = True) -> list[dict]:
    resp = _request(
        "GET",
        POSITION_PATH,
        {},
        signed=True,
        timeout_sec=timeout_sec,
        retryable=retryable,
    )
    if not isinstance(resp, dict) or resp.get("code") not in (0, "0"):
        code = resp.get("code") if isinstance(resp, dict) else "invalid_response"
        msg = resp.get("msg") if isinstance(resp, dict) else repr(resp)
        raise RuntimeError(f"[BINGX] get_positions failed: code={code} msg={msg}")
    try:
        rows = _extract_response_rows(
            resp,
            collection_keys=("positions", "position", "list", "rows"),
            single_row_markers=("positionAmt", "positionSide", "avgPrice", "entryPrice"),
            context="get_positions",
        )
        for index, row in enumerate(rows):
            if not str(row.get("symbol") or "").strip():
                raise ValueError(f"get_positions: row {index} has no symbol")
            try:
                qty = float(row.get("positionAmt"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"get_positions: row {index} has invalid positionAmt") from exc
            if not math.isfinite(qty):
                raise ValueError(f"get_positions: row {index} has non-finite positionAmt")
            if abs(qty) > 0 and str(row.get("positionSide") or "").upper() not in {"LONG", "SHORT", "BOTH"}:
                raise ValueError(f"get_positions: active row {index} has no valid positionSide")
        return rows
    except ValueError as exc:
        raise RuntimeError(f"[BINGX] get_positions response is ambiguous: {exc}") from exc


def get_open_orders_snapshot(
    *, timeout_sec: float | None = None, retryable: bool = True,
) -> dict:
    """Fetch one account-wide open-order snapshot for reconciliation.

    BingX documents ``symbol`` as optional on openOrders. This helper is stricter
    than the symbol-scoped reader: an unexpected response shape or an unidentifiable
    protection order is reported as an error so callers can fall back to the known
    per-symbol query rather than treating unknown data as missing protection.
    """
    try:
        resp = _request(
            "GET",
            OPEN_ORDERS_PATH,
            {},
            timeout_sec=timeout_sec,
            retryable=retryable,
        )
    except Exception as exc:
        return {"status": "error", "error": f"openOrders snapshot request failed: {exc}", "orders": []}

    if not isinstance(resp, dict) or resp.get("code") not in (0, "0"):
        error = resp.get("msg", "openOrders snapshot failed") if isinstance(resp, dict) else "invalid openOrders response"
        code = resp.get("code") if isinstance(resp, dict) else None
        try:
            code_int = int(code)
        except (TypeError, ValueError, OverflowError):
            code_int = None
        return {
            "status": "error", "error": f"openOrders snapshot failed: code={code} {error}", "orders": [],
            "code": code_int if code_int is not None else code,
            "retry_after_ms": resp.get("retry_after_ms") if isinstance(resp, dict) else None,
            "rate_limited": bool(isinstance(resp, dict) and (resp.get("rate_limited") or resp.get("local_cooldown") or code_int == 100410)),
            "local_cooldown": bool(isinstance(resp, dict) and resp.get("local_cooldown")),
        }

    data = resp.get("data")
    if isinstance(data, list):
        orders = data
    elif isinstance(data, dict) and isinstance(data.get("orders"), list):
        orders = data["orders"]
    else:
        return {"status": "error", "error": "openOrders snapshot has unexpected data shape", "orders": []}

    if not all(isinstance(order, dict) for order in orders):
        return {"status": "error", "error": "openOrders snapshot contains a non-object order", "orders": []}

    protection_types = {"TAKE_PROFIT", "TAKE_PROFIT_MARKET", "STOP", "STOP_MARKET"}
    for order in orders:
        order_type = str(order.get("type") or "").strip().upper()
        order_symbol = str(order.get("symbol") or "").strip()
        if not order_type or not order_symbol:
            return {
                "status": "error",
                "error": "openOrders snapshot has an unidentifiable protection/order row: missing type or symbol; protection state is unknown",
                "orders": [],
            }
        if order_type not in protection_types:
            continue
        if str(order.get("positionSide") or "").upper() not in {"LONG", "SHORT", "BOTH"}:
            return {
                "status": "error",
                "error": "openOrders snapshot contains an unidentifiable protection order",
                "orders": [],
            }

    return {"status": "ok", "orders": [dict(order) for order in orders], "source": "account_open_orders_snapshot"}


def _query_order(symbol: str, *, order_id: str | int | None = None, client_order_id: str | None = None) -> dict:
    bx = to_bx_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "contract_not_found"}
    if order_id is None and not client_order_id:
        return {"status": "error", "error": "order identifier required"}

    params = {"symbol": bx}
    if order_id is not None:
        params["orderId"] = str(order_id)
    else:
        params["clientOrderId"] = str(client_order_id)

    resp = _request("GET", ORDER_PATH, params, signed=True, retryable=False)
    if not isinstance(resp, dict) or resp.get("code") not in (0, "0"):
        return {"status": "error", "error": resp.get("msg") if isinstance(resp, dict) else str(resp), "code": resp.get("code") if isinstance(resp, dict) else None}

    data = resp.get("data") or {}
    order = data.get("order") or data
    if not isinstance(order, dict):
        return {"status": "error", "error": "order response missing order object", "code": resp.get("code")}

    def _first_float(*keys: str) -> float:
        for key in keys:
            raw = order.get(key)
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
        return 0.0

    def _first_positive_float(*keys: str) -> float:
        for key in keys:
            raw = order.get(key)
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                return value
        return 0.0

    executed_qty = _first_positive_float("executedQty", "cumQty", "stopExecutedQty", "executedQuantity", "cumulatedQuantity")
    avg_price = _first_positive_float("avgPrice", "avgFillPrice", "fillAvgPrice", "averagePrice", "executedAvgPrice")
    trigger_price = _first_float("stopPrice", "triggerPrice", "activationPrice")
    orig_qty = _first_positive_float("origQty", "quantity", "qty")

    return {
        "status": "ok",
        "order_id": str(order.get("orderId") or order.get("orderID") or order_id or ""),
        "order_status": str(order.get("status", "")).upper(),
        "avg_price": avg_price,
        "trigger_price": trigger_price,
        "executed_qty": executed_qty,
        "orig_qty": orig_qty,
        "client_order_id": str(order.get("clientOrderId", "")),
        "raw_order": order,
    }


def get_order(symbol: str, order_id: str | int) -> dict:
    return _query_order(symbol, order_id=order_id)


def get_order_by_client_order_id(symbol: str, client_order_id: str) -> dict:
    return _query_order(symbol, client_order_id=client_order_id)


def _extract_known_rows(
    payload: Any,
    row_keys: tuple[str, ...],
    *,
    depth: int = 0,
    recursive_wrapper_keys: set[str] | frozenset[str] | None = None,
) -> tuple[list[dict] | None, Any]:
    """Extract rows from explicitly named API wrappers; never guess arbitrary list keys.

    Callers can expand the wrapper set for one endpoint without changing parsing
    behavior for other endpoints that use this shared helper.
    """
    if isinstance(payload, list):
        if all(isinstance(row, dict) for row in payload):
            return payload, None
        return None, "non_object_row"
    if not isinstance(payload, dict) or depth > 4:
        return None, "unexpected_shape"
    if recursive_wrapper_keys is None:
        recursive_wrapper_keys = {"result", "data", "resultData", "response"}
    for key in row_keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(value, list):
            if all(isinstance(row, dict) for row in value):
                return value, payload.get("total")
            return None, "non_object_row"
        # Some gateway responses may nest fund-flow rows under these documented-
        # meaning names. Recurse only through this endpoint's named wrappers.
        if isinstance(value, dict) and key in recursive_wrapper_keys:
            rows, total = _extract_known_rows(
                value, row_keys, depth=depth + 1, recursive_wrapper_keys=recursive_wrapper_keys,
            )
            if rows is not None:
                return rows, total
    return None, "unexpected_shape"


def _shape_fingerprint(payload: Any) -> tuple[str, str]:
    """Return a stable, non-sensitive fingerprint and a brief shape description."""
    def shape(value: Any, depth: int = 0) -> Any:
        if isinstance(value, list):
            sample = shape(value[0], depth + 1) if value and depth < 3 else None
            return {"type": "list", "length": len(value), "sample": sample}
        if isinstance(value, dict):
            if depth >= 3:
                return {"type": "dict", "keys": sorted(map(str, value.keys()))[:30]}
            return {"type": "dict", "fields": {str(k): shape(v, depth + 1) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))[:30]}}
        return {"type": type(value).__name__}
    rendered = json.dumps(shape(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    fingerprint = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:12]
    if isinstance(payload, dict):
        description = ",".join(f"{key}:{type(value).__name__}" for key, value in sorted(payload.items(), key=lambda kv: str(kv[0]))[:20])
    else:
        description = type(payload).__name__
    return fingerprint, description


def _normalize_history_symbol(symbol: str) -> str | None:
    """Normalize base, dashed, and compact USDT symbols for history endpoints.

    History APIs are contract-scoped. A compact market symbol such as BTCUSDT
    must become BTC-USDT before contract lookup/fallback; appending another quote
    suffix would query the invalid BTCUSDT-USDT contract.
    """
    raw = str(symbol or "").strip().upper().replace("_", "-").replace("/", "-")
    if not raw:
        return None
    # A no-dash symbol ending in USDT is a compact quote symbol, not a base asset.
    if not raw.endswith("-USDT") and raw.endswith("USDT") and len(raw) > 4:
        raw = f"{raw[:-4]}-USDT"
    try:
        bx = to_bx_symbol(raw)
    except Exception:
        bx = None
    if bx:
        return str(bx).strip().upper().replace("_", "-").replace("/", "-")
    # Keep the legacy fallback for already contract-shaped symbols and bare base
    # tickers, but only after compact-quote normalization above.
    if raw.endswith("-USDT") and raw[:-5].replace("-", "").isalnum():
        return raw
    if raw.replace("-", "").isalnum() and 1 <= len(raw) <= 24:
        return raw.replace("-", "") + "-USDT"
    return None


def get_fill_history(
    symbol: str,
    start_ts: int,
    end_ts: int,
    *,
    order_id: str | int | None = None,
    page_index: int = 1,
    page_size: int = 1000,
    currency: str | None = None,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    """Query actual swap fills and normalize common BingX field aliases.

    Unknown payload shapes fail closed, with a shape fingerprint logged for
    diagnosis. A market/trigger price is never promoted to an execution price.
    """
    bx = _normalize_history_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "invalid_or_unknown_historical_symbol", "fills": []}
    try:
        start_ts, end_ts = int(start_ts), int(end_ts)
        page_index, page_size = int(page_index), int(page_size)
    except (TypeError, ValueError):
        return {"status": "error", "error": "invalid_fill_history_parameters", "fills": []}
    if start_ts <= 0 or end_ts <= start_ts or page_index < 1 or page_size < 1:
        return {"status": "error", "error": "invalid_fill_history_range_or_pagination", "fills": []}
    page_size = min(page_size, 1000)
    params: dict[str, Any] = {
        "symbol": bx, "startTs": start_ts, "endTs": end_ts,
        "pageIndex": page_index, "pageSize": page_size,
    }
    if order_id not in (None, ""):
        params["orderId"] = str(order_id)
    requested_currency = str(currency or "").strip().upper()
    if requested_currency in {"USDT", "USDC"}:
        params["currency"] = requested_currency
    else:
        requested_currency = ""
    try:
        resp = _request("GET", FILL_HISTORY_PATH, params, signed=True, timeout_sec=timeout_sec, retryable=False)
    except Exception as exc:
        return {"status": "error", "error": f"request_exception:{exc}", "fills": []}
    if not isinstance(resp, dict) or resp.get("code") not in (0, "0"):
        error = resp.get("msg", resp) if isinstance(resp, dict) else resp
        return {"status": "error", "error": str(error), "code": resp.get("code") if isinstance(resp, dict) else None, "fills": []}

    data = resp.get("data")
    row_keys = (
        "fill_orders", "fillOrders", "fills", "list", "orders", "rows", "records",
        "tradeFills", "trade_fills", "fillList", "fill_list", "result", "data",
        "resultData", "response",
    )
    rows, total = _extract_known_rows(data, row_keys)
    if rows is None:
        fingerprint, description = _shape_fingerprint(data)
        reason = "fill_history_non_object_row" if total == "non_object_row" else "fill_history_unexpected_data_shape"
        log.warning("[BINGX] %s fingerprint=%s shape=%s", reason, fingerprint, description)
        return {"status": "error", "error": reason, "shape_fingerprint": fingerprint, "fills": []}

    def first_number(row: dict, names: tuple[str, ...]) -> float | None:
        for key in names:
            value = row.get(key)
            if value in (None, ""):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(number):
                return number
        return None

    def first_text(row: dict, names: tuple[str, ...]) -> str:
        for key in names:
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
        return ""

    fills: list[dict[str, Any]] = []
    for row in rows:
        fill_ts = first_number(row, (
            "time", "tradeTime", "trade_time", "fillTime", "closeTime", "tradeTimestamp",
            "timestamp", "transactTime", "ts", "createTime",
        ))
        if fill_ts is not None and 0 < fill_ts < 100_000_000_000:
            fill_ts *= 1000.0
        qty = first_number(row, (
            "qty", "quantity", "volume", "tradeVolume", "executedQty", "fillQty", "dealVolume",
        ))
        price = first_number(row, ("price", "tradePrice", "fillPrice", "avgPrice", "dealPrice"))
        reported_fee_asset = first_text(row, ("feeAsset", "commissionAsset", "feeCurrency", "currency", "asset")).upper()
        # Official fillHistory rows may expose `fee` without a per-row asset
        # field. Treat it as the requested settlement currency only when the
        # caller explicitly filtered the endpoint by that currency.
        normalized_fee_asset = reported_fee_asset or requested_currency or None
        fills.append({
            "trade_id": first_text(row, ("tradeId", "tradeID", "trade_id", "fillId", "fillID", "fill_id", "id")),
            "order_id": first_text(row, ("orderId", "orderID", "order_id")),
            "symbol": first_text(row, ("symbol", "contract", "instrument")) or bx,
            "side": first_text(row, ("side", "tradeSide", "orderSide")).upper(),
            "position_side": first_text(row, ("positionSide", "positionSideName", "position_side")).upper(),
            "price": price,
            "qty": abs(qty) if qty is not None else None,
            "time": int(fill_ts) if fill_ts is not None else None,
            "realized_pnl": first_number(row, (
                "realizedPnl", "realizedPNL", "realizedProfit", "realized_profit", "realized_pnl",
            )),
            "fee": first_number(row, (
                "fee", "commission", "tradingFee", "tradeFee", "fillFee", "commissionFee", "transactionFee",
            )),
            "fee_asset": normalized_fee_asset,
            "raw_fill": row,
        })
    # If the endpoint supplies a total record count, it is authoritative. A full
    # final page (returned == page_size == total) is complete, not truncated. When
    # total is absent/unparseable, retain the conservative full-page fallback.
    total_count = None
    if total not in (None, ""):
        try:
            parsed_total = float(total)
            if math.isfinite(parsed_total) and parsed_total >= 0:
                total_count = parsed_total
        except (TypeError, ValueError, OverflowError):
            total_count = None
    has_more = (page_index * page_size < total_count) if total_count is not None else (len(fills) >= page_size)
    return {
        "status": "ok", "fills": fills, "page_index": page_index,
        "page_size": page_size, "returned": len(fills), "total": total,
        "has_more": has_more,
    }


def get_income_history(
    symbol: str,
    start_ts: int,
    end_ts: int,
    *,
    limit: int = 1000,
    timeout_sec: float | None = 5.0,
    max_range_requests: int = 64,
) -> dict[str, Any]:
    """Read signed account cashflows (realized PnL, trading fees, funding fees).

    The endpoint returns actual account ledger amounts. Full-history completeness
    is required: a truncated response or an unknown row shape is an error, never
    treated as a zero cost. The API retains only a bounded recent history.
    """
    bx = _normalize_history_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "invalid_or_unknown_income_symbol", "records": [], "complete": False}
    try:
        start_ts, end_ts, limit = int(start_ts), int(end_ts), int(limit)
    except (TypeError, ValueError):
        return {"status": "error", "error": "invalid_income_history_parameters", "records": [], "complete": False}
    if start_ts <= 0 or end_ts <= start_ts or limit < 1:
        return {"status": "error", "error": "invalid_income_history_range_or_limit", "records": [], "complete": False}
    limit = min(limit, 1000)
    seen: set[str] = set()
    request_count = 0
    last_shape_diagnostics: dict[str, str] = {}
    # Bound recursive time-range splitting so a dense account history cannot
    # turn a single recovery into thousands of signed API requests. Callers
    # doing secondary/background reconciliation may choose a smaller budget;
    # exhausting it is an explicit incomplete-data error, never a partial total.
    try:
        max_range_requests = max(1, min(64, int(max_range_requests)))
    except (TypeError, ValueError, OverflowError):
        return {"status": "error", "error": "invalid_income_history_request_budget", "records": [], "complete": False}

    def fetch_range(lo: int, hi: int, depth: int = 0) -> tuple[list[dict] | None, str | None]:
        nonlocal request_count
        if request_count >= max_range_requests:
            return None, "income_history_query_budget_exhausted"
        request_count += 1
        params = {"symbol": bx, "startTime": lo, "endTime": hi, "limit": limit}
        try:
            response = _request("GET", INCOME_HISTORY_PATH, params, signed=True, timeout_sec=timeout_sec, retryable=False)
        except Exception as exc:
            return None, f"request_exception:{exc}"
        if not isinstance(response, dict) or response.get("code") not in (0, "0"):
            err = response.get("msg", response) if isinstance(response, dict) else response
            return None, str(err)
        data = response.get("data")
        income_row_keys = (
            "list", "rows", "records", "income", "incomeList", "income_list",
            "incomeRecords", "income_records", "incomeHistory", "income_history",
            "incomeData", "income_data", "incomeHistoryList", "income_history_list",
            "incomeRecordsList", "income_records_list", "dataList", "data_list",
            "result", "data", "resultData", "response",
        )
        income_recursive_wrappers = frozenset({
            "result", "data", "resultData", "response",
            "income", "incomeHistory", "income_history", "incomeData", "income_data",
            "incomeRecords", "income_records", "incomeList", "income_list",
            "incomeHistoryList", "income_history_list", "records", "rows",
            "dataList", "data_list", "incomeRecordsList", "income_records_list",
        })
        rows, meta = _extract_known_rows(
            data, income_row_keys, recursive_wrapper_keys=income_recursive_wrappers,
        )
        if rows is None:
            fp, desc = _shape_fingerprint(data)
            log.warning("[BINGX] income history unexpected shape fingerprint=%s shape=%s", fp, desc)
            # Keep the accounting result UNKNOWN, but propagate only structural
            # diagnostics (keys/types), never raw account values, to health/alerts.
            last_shape_diagnostics.update({"shape_fingerprint": fp, "shape_summary": desc, "data_type": type(data).__name__})
            return None, "income_history_unexpected_data_shape"
        if len(rows) < limit:
            return rows, None
        # `user/income` has no offset cursor. Split saturated ranges by time rather
        # than assuming the first limit rows represent the whole ledger.
        if depth >= 12 or hi - lo <= 1:
            return None, "income_history_range_still_truncated_after_time_split"
        mid = (lo + hi) // 2
        left, left_error = fetch_range(lo, mid, depth + 1)
        if left_error:
            return None, left_error
        right, right_error = fetch_range(mid + 1, hi, depth + 1)
        if right_error:
            return None, right_error
        return (left or []) + (right or []), None

    rows, error = fetch_range(start_ts, end_ts)
    if error or rows is None:
        result = {"status": "error", "error": error or "income_history_unavailable", "records": [], "complete": False}
        result.update(last_shape_diagnostics)
        return result

    records: list[dict[str, Any]] = []
    for row in rows:
        def first_number(names: tuple[str, ...]) -> float | None:
            for key in names:
                value = row.get(key)
                if value in (None, ""):
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(number):
                    return number
            return None
        def first_text(names: tuple[str, ...]) -> str:
            for key in names:
                value = row.get(key)
                if value not in (None, ""):
                    return str(value)
            return ""
        amount = first_number(("income", "amount", "cashFlow", "cashflow"))
        timestamp = first_number(("time", "timestamp", "ts", "createTime", "transactTime"))
        if timestamp is not None and 0 < timestamp < 100_000_000_000:
            timestamp *= 1000.0
        if amount is None or timestamp is None or timestamp <= 0:
            return {"status": "error", "error": "income_history_record_missing_amount_or_time", "records": [], "complete": False}
        record = {
            "record_id": first_text(("tranId", "transactionId", "transId", "id")),
            "trade_id": first_text(("tradeId", "tradeID", "trade_id")),
            "order_id": first_text(("orderId", "orderID", "order_id")),
            # Do not infer a missing row symbol from the query filter; an unidentifiable
            # ledger row must not be silently attributed to the requested contract.
            "symbol": first_text(("symbol", "contract", "instrument")).upper(),
            "income_type": first_text(("incomeType", "income_type", "type")).upper(),
            "amount": amount,
            "asset": first_text(("asset", "currency")).upper(),
            "time": int(timestamp),
            "info": first_text(("info", "remark", "description")),
            "raw_income": row,
        }
        identity = record["record_id"] or hashlib.sha256(
            json.dumps({k: record[k] for k in ("symbol", "income_type", "amount", "asset", "time", "trade_id")}, sort_keys=True).encode()
        ).hexdigest()
        if identity in seen:
            continue
        seen.add(identity)
        records.append(record)
    records.sort(key=lambda item: (item["time"], item["record_id"]))
    return {
        "status": "ok", "records": records, "complete": True,
        "symbol": bx, "start_ts": start_ts, "end_ts": end_ts,
        "returned": len(records), "source": "BINGX_USER_INCOME", "query_count": request_count,
    }

def cancel_order(symbol: str, order_id: str | int) -> dict:
    bx = to_bx_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "contract_not_found"}

    return _request("DELETE", ORDER_PATH, {"symbol": bx, "orderId": str(order_id)}, signed=True)


def has_open_position(symbol: str, direction: str) -> bool:
    bx = to_bx_symbol(symbol)
    if not bx:
        return False

    want = "LONG" if direction.upper() == "LONG" else "SHORT"
    positions = get_positions()

    for p in positions:
        if str(p.get("symbol", "")).upper() != bx:
            continue
        position_side = str(p.get("positionSide", "")).strip().upper()
        try:
            amt = float(p.get("positionAmt", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            amt = 0.0

        if amt == 0:
            continue

        # In hedge mode, BingX reports positionAmt as a magnitude for both
        # LONG and SHORT positions. Trust a recognized positionSide instead of
        # inferring direction from the amount's sign. Fall back to signed amount
        # only for one-way/legacy payloads that do not expose LONG/SHORT.
        if position_side in {"LONG", "SHORT"}:
            if position_side == want:
                return True
            continue

        if (want == "LONG" and amt > 0) or (want == "SHORT" and amt < 0):
            return True
    return False


def _trade_digest(trade_id: str) -> str:
    return hashlib.sha256(str(trade_id).upper().encode()).hexdigest().upper()[:16]


def _new_open_client_order_id(bx_symbol: str, trade_id: str, retry_index: int = 0) -> str:
    # BingX clientOrderId is constrained to alphanumeric identifiers.
    # The first attempt keeps the historical deterministic ID; a retry is a
    # distinct deterministic ID so a confirmed CANCELED/EXPIRED order is never
    # reused as a new order.
    suffix = "" if retry_index <= 0 else f":RETRY:{int(retry_index)}"
    digest = hashlib.sha256(f"{bx_symbol}:{trade_id}{suffix}".encode()).hexdigest().upper()[:24]
    return f"EVTOPEN{digest}"


def open_market(symbol: str, direction: str, price: float, trade_id: str) -> dict:
    direction = str(direction).upper()
    if direction not in {"LONG", "SHORT"}:
        return {"status": "error", "error": f"invalid direction={direction}"}

    bx = to_bx_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "contract_not_found"}

    c = get_contract(symbol) or {}
    if not contract_exists(symbol):
        return {"status": "error", "error": "contract_unavailable", "symbol": bx}

    try:
        if has_open_position(symbol, direction):
            return {"status": "existing_position", "symbol": bx, "direction": direction}
    except Exception as exc:
        return {"status": "error", "error": f"position_check_failed: {exc}", "symbol": bx}

    try:
        prec = int(c.get("quantityPrecision") or 0)
        min_qty = float(c.get("tradeMinQuantity") or c.get("minQty") or 0)
        min_usdt = float(c.get("tradeMinUSDT") or c.get("minNotional") or c.get("minSizeUsd") or 0)
        max_lev = int(c.get("maxShortLeverage" if direction == "SHORT" else "maxLongLeverage") or c.get("maxLeverage") or MAX_LEVERAGE)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": f"invalid contract parameters: {exc}", "symbol": bx}

    sizing_price = _current_close_price(symbol)
    if sizing_price is None or sizing_price <= 0:
        return {"status": "error", "error": "live sizing price unavailable", "symbol": bx}

    leverage = min(LEVERAGE, MAX_LEVERAGE, max_lev)
    # Swap order quantity is expressed in the base coin. Contract ``size`` is
    # informational here; do not multiply quantity by undocumented legacy fields.
    target_notional_usdt = MARGIN_USDT * leverage
    qty = target_notional_usdt / max(sizing_price, 1e-12)
    q = Decimal(str(qty)).quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
    qty = float(q)

    # TP count is selected after the confirmed fill from the actual position
    # quantity. New entries use two TP legs when executable, otherwise one
    # terminal TP2 leg; no new position may install three TP orders.

    if qty <= 0 or qty < min_qty:
        return {
            "status": "error",
            "error": f"qty={qty} < min_qty={min_qty} at configured leverage={leverage}",
            "symbol": bx, "qty": qty, "min_qty": min_qty,
            "leverage": leverage, "sizing_price": sizing_price,
        }

    notional_usdt = qty * sizing_price
    if min_usdt > 0 and notional_usdt + 1e-12 < min_usdt:
        return {
            "status": "error",
            "error": f"notional={notional_usdt:.8g} < min_notional={min_usdt:.8g} at configured leverage={leverage}",
            "symbol": bx, "qty": qty, "min_qty": min_qty,
            "min_notional": min_usdt, "notional_usdt": notional_usdt,
            "leverage": leverage, "sizing_price": sizing_price,
        }

    if not _set_leverage(bx, leverage, direction):
        return {"status": "error", "error": f"failed to set leverage={leverage} for {direction}", "symbol": bx, "leverage": leverage}

    side = "BUY" if direction == "LONG" else "SELL"
    client_order_id = _new_open_client_order_id(bx, trade_id)

    params = {
        "symbol": bx,
        "side": side,
        "positionSide": direction,
        "type": "MARKET",
        "quantity": f"{qty:.{prec}f}",
        "clientOrderId": client_order_id,
    }

    # A confirmed CANCELED/EXPIRED order is no longer active and BingX forbids
    # reusing its clientOrderId. One deterministic fresh-ID retry is safe.
    # NEW/unknown or failed lookup states remain UNKNOWN and are never blindly
    # retried from inside this call.
    for open_attempt in range(2):
        # Per-attempt state must not leak from a previously canceled order into
        # the result of a later retry.
        order_status = ""
        response = _request("POST", ORDER_PATH, params)

        if isinstance(response, dict) and response.get("code") not in (0, "0"):
            # Audit P1-4 (order idempotency): a transport-level failure (-1) leaves
            # the outcome unknown -- the order may have been created even though we
            # did not receive an ack. Never blindly retry a POST; verify the result
            # via the position instead. The pre-flight has_open_position check above
            # guarantees any position present now was opened by THIS order.
            transport_error = (
                response.get("code") == -1
                and "missing bingx credentials" not in str(response.get("msg", "")).lower()
            )
            if transport_error:
                log.warning("[BINGX] Order POST transport error for %s (%s); querying clientOrderId before retry/reconciliation...", bx, response.get("msg"))
                try:
                    queried = get_order_by_client_order_id(symbol, client_order_id)
                except Exception as exc:
                    queried = {"status": "error", "error": str(exc)}

                if queried.get("status") == "ok":
                    order_status = str(queried.get("order_status", "")).upper()
                    executed_qty = float(queried.get("executed_qty", 0) or 0)
                    if executed_qty > 0 or order_status in {"FILLED", "PARTIALLY_FILLED"}:
                        log.warning(
                            "[BINGX] Order %s found after transport error with status=%s executed_qty=%s; treating entry as accepted.",
                            client_order_id, order_status, executed_qty,
                        )
                        return {
                            "status": "opened",
                            "symbol": bx,
                            "qty": qty,
                            "leverage": leverage,
                            "sizing_price": sizing_price,
                            "signal_price": float(price),
                            "order_reference_price": sizing_price,
                            "order_id": queried.get("order_id"),
                            "client_order_id": client_order_id,
                            "idempotency": "order_queried_after_transport_error",
                            "order_status": order_status,
                            "executed_qty": executed_qty,
                            "response": response,
                        }
                    if order_status in {"CANCELED", "EXPIRED"}:
                        if open_attempt == 0:
                            retry_client_order_id = _new_open_client_order_id(bx, trade_id, retry_index=1)
                            log.warning(
                                "[BINGX] Entry order %s is %s after transport error; retrying once with fresh clientOrderId=%s.",
                                client_order_id, order_status, retry_client_order_id,
                            )
                            client_order_id = retry_client_order_id
                            params["clientOrderId"] = client_order_id
                            continue
                        return {
                            "status": "error",
                            "error": f"market entry retry order {client_order_id} is {order_status} after transport error",
                            "symbol": bx, "clientOrderId": client_order_id,
                            "order_id": queried.get("order_id"),
                            "order_status": order_status,
                            "retry_exhausted": True,
                            "response": response,
                        }
                    # NEW/unknown exchange states are not safe to retry inside this call.
                    return {
                        "status": "unknown",
                        "error": f"market order outcome remains pending after transport error: status={order_status or 'unknown'}",
                        "symbol": bx, "clientOrderId": client_order_id,
                        "order_id": queried.get("order_id"),
                        "order_status": order_status,
                        "response": response,
                    }

                try:
                    if has_open_position(symbol, direction):
                        log.warning("[BINGX] Position found after transport error; treating deterministic clientOrderId entry as accepted.")
                        return {
                            "status": "opened",
                            "symbol": bx,
                            "qty": qty,
                            "leverage": leverage,
                            "sizing_price": sizing_price,
                            "signal_price": float(price),
                            "order_reference_price": sizing_price,
                            "order_id": None,
                            "client_order_id": client_order_id,
                            "entry_ack_missing_order_id": True,
                            "idempotency": "position_verified_after_transport_error",
                            "response": response,
                        }
                except Exception as exc:
                    log.error("[BINGX] Post-error position verification failed: %s", exc)

                return {
                    "status": "unknown",
                    "error": f"market order outcome unknown after transport error: {response.get('msg', '')}",
                    "symbol": bx, "clientOrderId": client_order_id,
                    "response": response,
                }

            return {"status": "error", "error": str(response.get("msg", "")), "symbol": bx, "clientOrderId": client_order_id, "response": response}

        data = response.get("data") or {}
        order = data.get("order") or {} if isinstance(data, dict) else {}
        order_id = (order.get("orderId") or order.get("orderID") or data.get("orderId") or data.get("orderID")) if isinstance(data, dict) else None
        linked_client_order_id = order.get("clientOrderId") or client_order_id if isinstance(order, dict) else client_order_id
        if not order_id:
            # A success code without an order ID is not a complete execution ack.
            # Try resolving the deterministic client id; if the exchange still
            # cannot provide a direct ID, return UNKNOWN so main() terminalizes the
            # event and reconciliation adopts any live position rather than losing
            # the audit link or blindly opening again.
            try:
                queried = get_order_by_client_order_id(symbol, str(linked_client_order_id))
            except Exception as exc:
                queried = {"status": "error", "error": str(exc)}
            if not isinstance(queried, dict):
                queried = {"status": "error", "error": "order lookup returned a non-object response"}
            queried_id = str(queried.get("order_id") or "").strip()
            queried_status = str(queried.get("order_status") or "").upper()
            queried_executed_qty = _safe_float(queried.get("executed_qty"), 0.0)

            # A successful HTTP/API envelope plus an order id does not prove that
            # a MARKET order filled. In particular, NEW with zero executed quantity
            # must not enter the normal protection path while the entry order can
            # still execute later. Cancel and verify any still-active order first.
            if queried.get("status") == "ok" and queried_id and queried_status in {"NEW", "PARTIALLY_FILLED"}:
                cancel_response = None
                latest_query = queried
                terminal_order_states = {"FILLED", "CANCELED", "EXPIRED", "REJECTED"}
                for cancel_attempt in range(3):
                    try:
                        cancel_response = cancel_order(symbol, queried_id)
                    except Exception as exc:
                        cancel_response = {"status": "error", "error": str(exc)}
                    try:
                        latest_query = get_order(symbol, queried_id)
                    except Exception as exc:
                        latest_query = {"status": "error", "error": str(exc)}
                    latest_status = str(latest_query.get("order_status") or "").upper() if isinstance(latest_query, dict) else ""
                    if isinstance(latest_query, dict) and latest_query.get("status") == "ok" and latest_status in terminal_order_states:
                        queried = latest_query
                        queried_status = latest_status
                        queried_executed_qty = _safe_float(queried.get("executed_qty"), 0.0)
                        break
                    if cancel_attempt < 2:
                        time.sleep(0.2 * (cancel_attempt + 1))
                else:
                    # Cancellation was attempted but not verified. Do not claim
                    # success or allow a new entry retry while the old order may
                    # still execute; expose the unresolved order for escalation.
                    try:
                        live_position = get_position_directional(symbol, direction)
                    except Exception as exc:
                        live_position = {"status": "error", "error": str(exc)}
                    if isinstance(live_position, dict) and str(live_position.get("status", "")).lower() == "found":
                        try:
                            live_qty = abs(float(live_position.get("positionAmt", 0) or 0))
                            live_avg = float(live_position.get("avgPrice") or live_position.get("entryPrice") or 0)
                        except (TypeError, ValueError, OverflowError):
                            live_qty = live_avg = 0.0
                        if live_qty > 0 and math.isfinite(live_qty) and live_avg > 0 and math.isfinite(live_avg):
                            log.critical(
                                "[BINGX] Entry order %s remains non-terminal after cancel attempts; a live %s %s position exists, proceeding to protect observed quantity",
                                queried_id, direction, bx,
                            )
                            return {
                                "status": "opened", "symbol": bx, "qty": live_qty,
                                "leverage": leverage, "sizing_price": sizing_price,
                                "signal_price": float(price), "order_reference_price": sizing_price,
                                "order_id": queried_id, "client_order_id": str(linked_client_order_id),
                                "order_status": queried_status or "UNKNOWN",
                                "entry_order_cancel_unconfirmed": True,
                                "entry_cancel_response": cancel_response,
                                "entry_position_snapshot": live_position,
                                "response": response, "order_lookup": latest_query,
                            }
                    return {
                        "status": "unknown",
                        "error": "market entry order remains non-terminal after cancel attempts; refusing to treat it as filled or retry blindly",
                        "symbol": bx, "qty": qty, "leverage": leverage,
                        "sizing_price": sizing_price, "signal_price": float(price),
                        "order_reference_price": sizing_price,
                        "client_order_id": str(linked_client_order_id), "order_id": queried_id,
                        "order_status": queried_status or "UNKNOWN",
                        "escalation_required": True,
                        "entry_cancel_response": cancel_response,
                        "response": response, "order_lookup": latest_query,
                        "entry_ack_missing_order_id": True,
                    }


            if (
                queried.get("status") == "ok"
                and queried_id
                and queried_status in {"CANCELED", "EXPIRED", "REJECTED"}
                and queried_executed_qty <= 0
            ):
                if open_attempt == 0:
                    retry_client_order_id = _new_open_client_order_id(bx, trade_id, retry_index=1)
                    log.warning(
                        "[BINGX] Entry order %s is %s with verified zero fills; retrying once with fresh clientOrderId=%s",
                        queried_id, queried_status, retry_client_order_id,
                    )
                    client_order_id = retry_client_order_id
                    params["clientOrderId"] = client_order_id
                    continue
                return {
                    "status": "unknown",
                    "error": f"market entry retry order {queried_id} is {queried_status} with zero fills; retry budget exhausted",
                    "symbol": bx, "order_id": queried_id,
                    "client_order_id": str(linked_client_order_id),
                    "order_status": queried_status, "executed_qty": queried_executed_qty,
                    "response": response, "order_lookup": queried,
                    "entry_ack_missing_order_id": True,
                }

            if queried.get("status") == "ok" and queried_id and (
                queried_executed_qty > 0 or queried_status == "FILLED"
            ):
                order_id = queried_id
                order_status = queried_status
            else:
                try:
                    live_position = get_position_directional(symbol, direction)
                except Exception as exc:
                    live_position = {"status": "error", "error": str(exc)}
                if isinstance(live_position, dict) and str(live_position.get("status", "")).lower() == "found":
                    try:
                        live_qty = abs(float(live_position.get("positionAmt", 0) or 0))
                        live_avg = float(live_position.get("avgPrice") or live_position.get("entryPrice") or 0)
                    except (TypeError, ValueError, OverflowError):
                        live_qty = live_avg = 0.0
                    if live_qty > 0 and math.isfinite(live_qty) and live_avg > 0 and math.isfinite(live_avg):
                        log.critical(
                            "[BINGX] MARKET entry ack had no orderId but a live %s %s position is confirmed; continuing to protection with clientOrderId=%s and unresolved direct order id",
                            direction, bx, linked_client_order_id,
                        )
                        return {
                            "status": "opened",
                            "symbol": bx,
                            "qty": live_qty,
                            "leverage": leverage,
                            "sizing_price": sizing_price,
                            "signal_price": float(price),
                            "order_reference_price": sizing_price,
                            "order_id": queried_id or None,
                            "client_order_id": str(linked_client_order_id),
                            "order_status": queried_status or "POSITION_CONFIRMED",
                            "entry_ack_missing_order_id": True,
                            "entry_position_snapshot": live_position,
                            "response": response,
                            "order_lookup": queried,
                        }
                return {
                    "status": "unknown",
                    "error": "market entry success response missing orderId; direct order lookup and live-position confirmation did not resolve it",
                    "symbol": bx, "qty": qty, "leverage": leverage,
                    "sizing_price": sizing_price, "signal_price": float(price),
                    "order_reference_price": sizing_price,
                    "client_order_id": str(linked_client_order_id),
                    "order_status": queried_status or "UNKNOWN",
                    "response": response, "order_lookup": queried,
                    "position_lookup": live_position,
                    "entry_ack_missing_order_id": True,
                }
        return {
            "status": "opened",
            "symbol": bx,
            "qty": qty,
            "leverage": leverage,
            "sizing_price": sizing_price,
            "signal_price": float(price),
            "order_reference_price": sizing_price,
            "order_id": str(order_id),
            "client_order_id": str(linked_client_order_id),
            "order_status": str(order_status or order.get("status") or "UNKNOWN").upper(),
            "response": response,
            "open_attempt": open_attempt + 1,
        }

    return {
        "status": "error",
        "error": "market entry retry exhausted",
        "symbol": bx,
        "clientOrderId": client_order_id,
        "retry_exhausted": True,
    }


def emergency_close_position(symbol: str, direction: str, qty: float | None = None, reason_token: str | None = None) -> dict:
    """Fail-safe close used when a freshly opened position cannot be protected.

    It never blindly retries the same MARKET request. After each POST we query
    the live directional position and, when necessary, close only the remaining
    quantity with a fresh clientOrderId. This keeps the emergency path idempotent
    even when the exchange acknowledgement is lost.
    """
    direction = str(direction).upper()
    if direction not in {"LONG", "SHORT"}:
        return {"status": "error", "error": f"invalid direction={direction}"}
    # This engine only supports Hedge Mode. BingX explicitly rejects reduceOnly
    # in Hedge Mode; never add it and never retry a failed close with that flag
    # removed. A mismatched configured mode is a hard error before any POST.
    if POSITION_MODE != "HEDGE":
        return {"status": "error", "error": f"unsupported_position_mode_for_directional_close:{POSITION_MODE}", "escalation_required": True}
    actual_mode = get_position_mode()
    if actual_mode.get("status") != "ok":
        return {"status": "error", "error": actual_mode.get("error", "position_mode_unknown"), "escalation_required": True}
    if actual_mode.get("mode") != "HEDGE":
        return {"status": "error", "error": f"exchange_position_mode_mismatch_for_directional_close:{actual_mode.get('mode')}", "escalation_required": True}
    bx_symbol = to_bx_symbol(symbol)
    contract = get_contract(symbol)
    if not bx_symbol or not contract:
        return {"status": "error", "error": "contract_not_found"}
    try:
        precision = int(contract.get("quantityPrecision") or 0)
        requested_qty = abs(float(qty)) if qty is not None else 0.0
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": f"invalid close parameters: {exc}"}

    client_order_id = "EVTCLOSE" + hashlib.sha256(
        f"{reason_token or 'EMERGENCY'}:{bx_symbol}:{direction}".upper().encode()
    ).hexdigest().upper()[:24]

    def _confirm_absence(first_state: dict | None = None) -> dict:
        """Require a second explicit not_found response before declaring flat."""
        state = first_state
        if not isinstance(state, dict):
            try:
                state = get_position_directional(symbol, direction)
            except Exception as exc:
                return {"status": "error", "error": f"position_read_exception:{exc}"}
        if str(state.get("status", "")).lower() != "not_found":
            return state
        time.sleep(0.25)
        try:
            confirmed = get_position_directional(symbol, direction)
        except Exception as exc:
            return {"status": "error", "error": f"position_absence_confirmation_exception:{exc}"}
        if str(confirmed.get("status", "")).lower() == "not_found":
            return {"status": "not_found", "symbol": bx_symbol, "positionSide": direction, "absence_reads": 2}
        return confirmed

    def _poll_after_filled_close() -> dict:
        """Re-read a lagging position view before declaring a filled close unresolved.

        This function is read-only. It never submits an order. A confirmed flat
        state requires the existing two-read absence check; a residual is only
        considered stable after two consecutive positive position reads agree
        within the contract's quantity precision.
        """
        observations: list[tuple[float, dict]] = []
        errors: list[str] = []
        # Position snapshots can lag behind a FILLED MARKET order. Keep these
        # GET-only polls bounded (~5.25s sleep total) before raising an operator
        # alert; do not assume stale quantity is flat and do not blind-retry it.
        poll_delays = (0.25, 0.50, 1.00, 1.50, 2.00)
        qty_tolerance = max(1e-9, 0.5 * (10.0 ** (-max(0, precision))))
        for delay in poll_delays:
            time.sleep(delay)
            try:
                state = get_position_directional(symbol, direction)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}:{exc}")
                continue
            status = str(state.get("status", "")).lower() if isinstance(state, dict) else "unknown"
            if status == "not_found":
                state = _confirm_absence(state)
                status = str(state.get("status", "")).lower() if isinstance(state, dict) else "unknown"
                if status == "not_found":
                    return {
                        "status": "closed", "verification": "two_not_found_reads_after_filled_order",
                        "poll_count": len(observations) + len(errors) + 1,
                    }
            if status == "found":
                try:
                    value = abs(float(state.get("positionAmt", 0) or 0))
                except (TypeError, ValueError, OverflowError):
                    value = 0.0
                if math.isfinite(value) and value > 0:
                    observations.append((value, state))
                else:
                    errors.append("position_read_found_with_invalid_quantity")
            else:
                errors.append(str(state.get("error") or status or "unknown_position_state") if isinstance(state, dict) else "invalid_position_response")

        if observations:
            residual_qty, last_state = observations[-1]
            stable = len(observations) >= 2 and abs(observations[-1][0] - observations[-2][0]) <= qty_tolerance
            return {
                "status": "found", "position": last_state, "remaining_qty": residual_qty,
                "stable": stable, "poll_count": len(observations) + len(errors),
                "observed_quantities": [round(value, max(0, precision) + 2) for value, _ in observations],
                "last_error": errors[-1] if errors else None,
            }
        return {
            "status": "unknown", "error": errors[-1] if errors else "no_position_state_returned_during_post_fill_polls",
            "poll_count": len(errors),
        }

    for attempt in range(2):
        try:
            live = get_position_directional(symbol, direction)
        except Exception as exc:
            return {"status": "error", "error": f"position verification failed before emergency close: {exc}"}
        live_status = str(live.get("status", "")).lower()
        if live_status == "not_found":
            live = _confirm_absence(live)
            live_status = str(live.get("status", "")).lower()
            if live_status == "not_found":
                return {"status": "closed", "symbol": bx_symbol, "direction": direction, "attempts": attempt, "verification": "two_not_found_reads"}
        if live_status != "found":
            return {
                "status": "unknown", "error": f"position state is not safe to close: {live.get('error', live_status or 'unknown')}",
                "symbol": bx_symbol, "direction": direction, "attempts": attempt, "escalation_required": True,
            }

        try:
            live_qty = abs(float(live.get("positionAmt", 0) or 0))
        except (TypeError, ValueError, OverflowError):
            live_qty = 0.0
        if not math.isfinite(live_qty) or live_qty <= 0:
            return {
                "status": "unknown", "error": "live position quantity is not confirmed; refusing emergency close POST",
                "symbol": bx_symbol, "direction": direction, "attempts": attempt, "escalation_required": True,
            }
        close_qty = _round_qty(live_qty, precision)
        if close_qty <= 0:
            return {"status": "unknown", "error": "emergency close quantity rounds to zero", "symbol": bx_symbol, "escalation_required": True}

        side = "SELL" if direction == "LONG" else "BUY"
        params = {
            "symbol": bx_symbol,
            "side": side,
            "positionSide": direction,
            "type": "MARKET",
            "quantity": _format_qty(close_qty, precision),
            "clientOrderId": client_order_id,
        }
        response = _request("POST", ORDER_PATH, params)
        code = response.get("code") if isinstance(response, dict) else None
        if code not in (0, "0", -1, "-1"):
            last_error = str(response.get("msg", "emergency close failed")) if isinstance(response, dict) else str(response)
            if attempt == 1:
                return {"status": "error", "error": last_error, "symbol": bx_symbol, "client_order_id": client_order_id}
            continue

        if code in (-1, "-1"):
            try:
                queried = get_order_by_client_order_id(symbol, client_order_id)
            except Exception as exc:
                queried = {"status": "error", "error": str(exc)}

            if queried.get("status") == "ok":
                order_status = str(queried.get("order_status", "")).upper()
                executed_qty = float(queried.get("executed_qty", 0) or 0)
                if order_status in {"FILLED", "PARTIALLY_FILLED"} or executed_qty > 0:
                    log.warning(
                        "[EMERGENCY_CLOSE] %s found after transport error with status=%s executed_qty=%s; verifying residual before any new POST.",
                        client_order_id, order_status, executed_qty,
                    )
                    time.sleep(0.25)
                    try:
                        after_recovery = get_position_directional(symbol, direction)
                    except Exception as exc:
                        return {
                            "status": "unknown",
                            "error": f"emergency close order outcome found but residual verification failed: {exc}",
                            "symbol": bx_symbol, "direction": direction,
                            "client_order_id": client_order_id, "escalation_required": True,
                        }
                    after_status = str(after_recovery.get("status", "")).lower()
                    if after_status == "not_found":
                        after_recovery = _confirm_absence(after_recovery)
                        after_status = str(after_recovery.get("status", "")).lower()
                        if after_status == "not_found":
                            return {
                                "status": "closed", "symbol": bx_symbol, "direction": direction,
                                "client_order_id": client_order_id,
                                "order_id": queried.get("order_id"),
                                "execution_price": queried.get("avg_price") or None,
                                "idempotency": "order_queried_after_transport_error",
                                "verification": "two_not_found_reads",
                            }
                    if after_status != "found":
                        return {
                            "status": "unknown",
                            "error": f"close order outcome found but live position state is unknown: {after_recovery.get('error', after_status or 'unknown')}",
                            "symbol": bx_symbol, "direction": direction,
                            "client_order_id": client_order_id, "escalation_required": True,
                        }
                    residual_qty = abs(float(after_recovery.get("positionAmt", 0) or 0))
                    if not math.isfinite(residual_qty) or residual_qty <= 0:
                        return {
                            "status": "unknown", "error": "position read reported found with invalid residual quantity",
                            "symbol": bx_symbol, "direction": direction, "client_order_id": client_order_id,
                            "escalation_required": True,
                        }
                    if order_status == "FILLED":
                        post_fill = _poll_after_filled_close()
                        if post_fill.get("status") == "closed":
                            return {
                                "status": "closed", "symbol": bx_symbol, "direction": direction,
                                "client_order_id": client_order_id, "order_id": queried.get("order_id"),
                                "execution_price": queried.get("avg_price") or None,
                                "idempotency": "order_queried_after_transport_error",
                                "verification": post_fill.get("verification"),
                                "position_poll_count": post_fill.get("poll_count"),
                            }
                        if post_fill.get("status") == "found":
                            residual_qty = float(post_fill.get("remaining_qty") or 0.0)
                            qty_tolerance = max(1e-9, 0.5 * (10.0 ** (-max(0, precision))))
                            # A stable residual smaller than the pre-order position
                            # is a newly confirmed remaining quantity. Retry only
                            # once, with a fresh clientOrderId and a fresh position
                            # read at the top of the loop. An unchanged/stale view
                            # is escalated without another MARKET POST.
                            if (post_fill.get("stable") is True and residual_qty < live_qty - qty_tolerance
                                    and attempt + 1 < 2):
                                digest = hashlib.sha256(
                                    f"{reason_token or 'EMERGENCY'}:{bx_symbol}:{direction}:FILLED_RESIDUAL:{attempt}".upper().encode()
                                ).hexdigest().upper()[:24]
                                client_order_id = f"EVTCLOSE{digest}"
                                continue
                            return {
                                "status": "unknown",
                                "error": (
                                    "MARKET close order is FILLED but position view still shows residual; refusing blind duplicate "
                                    f"(residual_qty={residual_qty}; polls={post_fill.get('poll_count')}; "
                                    f"stable={post_fill.get('stable')}; observed_qty={post_fill.get('observed_quantities', [])})"
                                ),
                                "symbol": bx_symbol, "direction": direction,
                                "client_order_id": client_order_id,
                                "remaining_qty": residual_qty, "position_poll_count": post_fill.get("poll_count"),
                                "position_poll_stable": post_fill.get("stable"),
                                "position_poll_quantities": post_fill.get("observed_quantities", []),
                                "escalation_required": True,
                            }
                        return {
                            "status": "unknown",
                            "error": "MARKET close order outcome confirmed FILLED but position verification stayed unknown after bounded polls",
                            "symbol": bx_symbol, "direction": direction,
                            "client_order_id": client_order_id,
                            "position_poll_count": post_fill.get("poll_count"),
                            "position_poll_error": post_fill.get("error"),
                            "escalation_required": True,
                        }
                    # A verified partial fill justifies one fresh order for the
                    # exact residual quantity. Never reuse the filled order id.
                    digest = hashlib.sha256(
                        f"{reason_token or 'EMERGENCY'}:{bx_symbol}:{direction}:RESIDUAL:{attempt}".upper().encode()
                    ).hexdigest().upper()[:24]
                    client_order_id = f"EVTCLOSE{digest}"
                    requested_qty = residual_qty
                    continue

                if order_status in {"CANCELED", "EXPIRED"}:
                    digest = hashlib.sha256(
                        f"{reason_token or 'EMERGENCY'}:{bx_symbol}:{direction}:RETRY:{attempt}".upper().encode()
                    ).hexdigest().upper()[:24]
                    client_order_id = f"EVTCLOSE{digest}"
                    continue

                return {
                    "status": "unknown",
                    "error": f"MARKET close order pending after transport error: status={order_status or 'unknown'}; refusing blind duplicate",
                    "symbol": bx_symbol, "direction": direction,
                    "client_order_id": client_order_id,
                    "escalation_required": True,
                }

            if queried.get("status") == "absent":
                # A proven-absent order may be safely retried with the SAME
                # clientOrderId. If the first POST was actually accepted but
                # the query is only lagging, BingX's unique clientOrderId
                # prevents creation of a second order.
                if attempt + 1 < 2:
                    time.sleep(0.25)
                    continue
                continue

            # If the order lookup itself is unavailable, do not manufacture a
            # new MARKET order. The original clientOrderId remains the canonical
            # recovery key for the next reconciliation cycle.
            return {
                "status": "unknown",
                "error": f"MARKET close transport outcome unknown; order lookup failed: {queried.get('error', 'unknown')}",
                "symbol": bx_symbol, "direction": direction,
                "client_order_id": client_order_id,
                "escalation_required": True,
            }

        time.sleep(0.25)
        execution_price = None
        actual_order_id = None
        if isinstance(response, dict):
            data = response.get("data") or {}
            order = data.get("order") if isinstance(data, dict) else {}
            if isinstance(order, dict):
                actual_order_id = order.get("orderId") or order.get("orderID") or data.get("orderId")
                for key in ("avgPrice", "avg_price", "executedAvgPrice"):
                    try:
                        candidate = float(order.get(key))
                    except (TypeError, ValueError):
                        continue
                    if candidate > 0:
                        execution_price = candidate
                        break
        try:
            after = get_position_directional(symbol, direction)
        except Exception as exc:
            return {"status": "unknown", "error": f"emergency close submitted but verification failed: {exc}", "symbol": bx_symbol, "client_order_id": client_order_id}
        after_status = str(after.get("status", "")).lower()
        if after_status == "not_found":
            after = _confirm_absence(after)
            after_status = str(after.get("status", "")).lower()
            if after_status == "not_found":
                return {
                    "status": "closed", "symbol": bx_symbol, "direction": direction,
                    "client_order_id": client_order_id, "order_id": str(actual_order_id) if actual_order_id not in (None, "") else None,
                    "execution_price": execution_price,
                    "verification": "two_not_found_reads",
                }
        if after_status != "found":
            return {
                "status": "unknown",
                "error": f"emergency close submitted but position state is unknown: {after.get('error', after_status or 'unknown')}",
                "symbol": bx_symbol, "direction": direction,
                "client_order_id": client_order_id, "escalation_required": True,
            }
        try:
            after_qty = abs(float(after.get("positionAmt", 0) or 0))
        except (TypeError, ValueError, OverflowError):
            after_qty = 0.0
        if not math.isfinite(after_qty) or after_qty <= 0:
            return {
                "status": "unknown", "error": "position state found but residual quantity is invalid after close",
                "symbol": bx_symbol, "direction": direction, "client_order_id": client_order_id,
                "escalation_required": True,
            }

        # An HTTP-successful MARKET close can still race the position endpoint: the
        # order may already be FILLED while the live position view has not caught up.
        # Never infer "partial" solely from a stale position snapshot. Verify the
        # acknowledged order by its clientOrderId before issuing any second MARKET.
        try:
            verified_order = get_order_by_client_order_id(symbol, client_order_id)
        except Exception as exc:
            verified_order = {"status": "error", "error": str(exc)}

        if verified_order.get("status") != "ok":
            # The original POST was acknowledged, but the order lookup endpoint
            # is unavailable. A single bounded retry with the SAME clientOrderId
            # is safe: it cannot create a second independent order identity.
            # Never mint a fresh ID while the original outcome is unverified.
            if attempt + 1 < 2:
                time.sleep(0.25)
                continue
            continue

        verified_status = str(verified_order.get("order_status", "")).upper()
        verified_executed_qty = float(verified_order.get("executed_qty", 0) or 0)
        residual_qty = after_qty

        if verified_status == "FILLED" and residual_qty > 0:
            post_fill = _poll_after_filled_close()
            if post_fill.get("status") == "closed":
                return {
                    "status": "closed", "symbol": bx_symbol, "direction": direction,
                    "client_order_id": client_order_id,
                    "order_id": str(actual_order_id) if actual_order_id not in (None, "") else str(verified_order.get("order_id") or "") or None,
                    "execution_price": execution_price or verified_order.get("avg_price") or None,
                    "verification": post_fill.get("verification"),
                    "position_poll_count": post_fill.get("poll_count"),
                }
            if post_fill.get("status") == "found":
                residual_qty = float(post_fill.get("remaining_qty") or 0.0)
                qty_tolerance = max(1e-9, 0.5 * (10.0 ** (-max(0, precision))))
                if (post_fill.get("stable") is True and residual_qty < live_qty - qty_tolerance
                        and attempt + 1 < 2):
                    digest = hashlib.sha256(
                        f"{reason_token or 'EMERGENCY'}:{bx_symbol}:{direction}:FILLED_RESIDUAL:{attempt}".upper().encode()
                    ).hexdigest().upper()[:24]
                    client_order_id = f"EVTCLOSE{digest}"
                    continue
                return {
                    "status": "unknown",
                    "error": (
                        "MARKET close order is FILLED but position view still shows residual; refusing blind duplicate "
                        f"(residual_qty={residual_qty}; polls={post_fill.get('poll_count')}; "
                        f"stable={post_fill.get('stable')}; observed_qty={post_fill.get('observed_quantities', [])})"
                    ),
                    "symbol": bx_symbol,
                    "direction": direction,
                    "client_order_id": client_order_id,
                    "remaining_qty": residual_qty,
                    "executed_qty": verified_executed_qty,
                    "position_poll_count": post_fill.get("poll_count"),
                    "position_poll_stable": post_fill.get("stable"),
                    "position_poll_quantities": post_fill.get("observed_quantities", []),
                    "escalation_required": True,
                }
            return {
                "status": "unknown",
                "error": "MARKET close order FILLED but position verification stayed unknown after bounded polls",
                "symbol": bx_symbol,
                "direction": direction,
                "client_order_id": client_order_id,
                "executed_qty": verified_executed_qty,
                "position_poll_count": post_fill.get("poll_count"),
                "position_poll_error": post_fill.get("error"),
                "escalation_required": True,
            }

        if verified_status in {"CANCELED", "EXPIRED"}:
            digest = hashlib.sha256(
                f"{reason_token or 'EMERGENCY'}:{bx_symbol}:{direction}:ACK_CANCEL_RETRY:{attempt}".upper().encode()
            ).hexdigest().upper()[:24]
            client_order_id = f"EVTCLOSE{digest}"
            continue

        if verified_status not in {"PARTIALLY_FILLED", "FILLED"} or verified_executed_qty <= 0:
            return {
                "status": "unknown",
                "error": f"MARKET close acknowledged but order status={verified_status or 'unknown'} with residual position; refusing blind duplicate",
                "symbol": bx_symbol,
                "direction": direction,
                "client_order_id": client_order_id,
                "remaining_qty": residual_qty,
                "escalation_required": True,
            }

        # Verified partial fill: a second close is justified for the exact live
        # residual, but it must use a fresh clientOrderId because the first ID
        # already belongs to the exchange order.
        digest = hashlib.sha256(
            f"{reason_token or 'EMERGENCY'}:{bx_symbol}:{direction}:SUCCESS_RESIDUAL:{attempt}".upper().encode()
        ).hexdigest().upper()[:24]
        client_order_id = f"EVTCLOSE{digest}"

    # Two verified close attempts are the automatic safety limit. Do one final
    # read-only position check and surface an explicit unresolved state rather
    # than silently collapsing it into a generic error. The caller can then
    # trigger its existing rollback/alert path without a blind third POST.
    try:
        final_state = get_position_directional(symbol, direction)
    except Exception as exc:
        log.critical(
            "[EMERGENCY_CLOSE] position state unknown after two attempts: %s %s: %s",
            bx_symbol, direction, exc,
        )
        return {
            "status": "unknown",
            "error": "emergency close final verification failed after two attempts",
            "symbol": bx_symbol, "direction": direction, "attempts": 2,
            "escalation_required": True,
        }

    final_status = str(final_state.get("status", "")).lower()
    if final_status == "not_found":
        final_state = _confirm_absence(final_state)
        final_status = str(final_state.get("status", "")).lower()
    if final_status == "found":
        try:
            remaining_qty = abs(float(final_state.get("positionAmt", 0) or 0))
        except (TypeError, ValueError):
            remaining_qty = None
        log.critical(
            "[EMERGENCY_CLOSE] UNFLATTENED after two attempts: %s %s remaining_qty=%s",
            bx_symbol, direction, remaining_qty,
        )
        return {
            "status": "UNFLATTENED",
            "error": "emergency close did not flatten position after two attempts",
            "symbol": bx_symbol, "direction": direction, "attempts": 2,
            "remaining_qty": remaining_qty, "escalation_required": True,
        }

    if final_status == "not_found":
        return {
            "status": "closed", "symbol": bx_symbol, "direction": direction, "attempts": 2,
            "verification": "final_two_not_found_reads",
        }
    return {
        "status": "unknown",
        "error": f"emergency close final position state is unknown: {final_state.get('error', final_status or 'unknown')}",
        "symbol": bx_symbol, "direction": direction, "attempts": 2,
        "escalation_required": True,
    }


def get_position_mode() -> dict[str, Any]:
    """Return the exchange-reported position mode; unknown is not treated as Hedge."""
    try:
        response = _request("GET", POSITION_MODE_PATH, {}, signed=True, retryable=False)
    except Exception as exc:
        return {"status": "error", "error": f"position_mode_request_exception:{exc}"}
    if not isinstance(response, dict) or response.get("code") not in (0, "0"):
        error = response.get("msg", response) if isinstance(response, dict) else response
        return {"status": "error", "error": f"position_mode_query_failed:{error}"}
    data = response.get("data")
    if not isinstance(data, dict) or "dualSidePosition" not in data:
        return {"status": "error", "error": "position_mode_response_missing_dualSidePosition"}
    value = data.get("dualSidePosition")
    if isinstance(value, bool):
        hedge = value
    elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        hedge = value.strip().lower() == "true"
    else:
        return {"status": "error", "error": "position_mode_response_invalid_dualSidePosition"}
    return {"status": "ok", "mode": "HEDGE" if hedge else "ONE_WAY", "dualSidePosition": hedge}


def get_position_directional(symbol: str, direction: str) -> dict:
    """Read one directional position without confusing unknown data with flat state.

    Only a valid, recognized response with no matching nonzero position can
    produce ``not_found``. API errors, schema drift, malformed quantities, and
    an active position without a usable side remain ``error`` / ``found``.
    """
    bx_symbol = to_bx_symbol(symbol)
    direction = str(direction).upper()
    if not bx_symbol or direction not in {"LONG", "SHORT"}:
        return {"status": "error", "error": "invalid_position_query", "symbol": bx_symbol}

    try:
        resp = _request("GET", POSITION_PATH, {"symbol": bx_symbol})
    except Exception as exc:
        return {"status": "error", "error": f"position_request_exception:{exc}", "symbol": bx_symbol}
    if not isinstance(resp, dict) or resp.get("code") not in (0, "0"):
        error = resp.get("msg", resp) if isinstance(resp, dict) else resp
        return {"status": "error", "error": f"get_position failed: {error}", "symbol": bx_symbol}

    data = resp.get("data")
    if isinstance(data, list):
        positions = data
    elif isinstance(data, dict):
        positions = None
        for key in ("positions", "position", "list", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                positions = value
                break
            if isinstance(value, dict):
                positions = [value]
                break
        if positions is None and any(k in data for k in ("positionAmt", "positionSide", "avgPrice", "entryPrice")):
            positions = [data]
        if positions is None:
            return {"status": "error", "error": "position_response_unrecognized_data_shape", "symbol": bx_symbol}
    else:
        return {"status": "error", "error": "position_response_missing_or_invalid_data", "symbol": bx_symbol}

    if not all(isinstance(row, dict) for row in positions):
        return {"status": "error", "error": "position_response_contains_non_object_row", "symbol": bx_symbol}

    for position in positions:
        row_symbol = str(position.get("symbol") or bx_symbol).upper().replace("_", "-").replace("/", "-")
        if row_symbol != bx_symbol:
            continue

        raw_side = str(position.get("positionSide", "")).upper().strip()
        raw_qty = position.get("positionAmt")
        try:
            qty_signed = float(raw_qty)
        except (TypeError, ValueError, OverflowError):
            # If this row has directional information but its quantity cannot
            # be read, absence cannot be proven.
            if raw_side in {direction, "BOTH"}:
                return {"status": "error", "error": "position_response_invalid_positionAmt", "symbol": bx_symbol}
            if raw_side in {"LONG", "SHORT"} and raw_side != direction:
                continue
            return {"status": "error", "error": "position_response_missing_position_side_or_quantity", "symbol": bx_symbol}
        if not math.isfinite(qty_signed):
            return {"status": "error", "error": "position_response_non_finite_positionAmt", "symbol": bx_symbol}
        if abs(qty_signed) <= 0:
            continue

        if raw_side not in {"LONG", "SHORT", "BOTH"}:
            return {"status": "error", "error": "position_response_active_row_missing_valid_positionSide", "symbol": bx_symbol}
        if raw_side == "BOTH":
            if direction == "LONG" and qty_signed < 0:
                continue
            if direction == "SHORT" and qty_signed > 0:
                continue
        elif raw_side != direction:
            continue

        def _positive_price(*keys: str) -> float:
            for key in keys:
                try:
                    value = float(position.get(key) or 0)
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(value) and value > 0:
                    return value
            return 0.0

        avg_price = _positive_price("avgPrice", "entryPrice", "avgEntryPrice", "averagePrice")
        # Quantity and direction prove the position exists even if a newly
        # observed API response omits the average price. Keep it found so no
        # lifecycle path can mistake it for a closed position.
        return {
            "status": "found",
            "symbol": row_symbol,
            "positionSide": direction,
            "avgPrice": avg_price,
            "entryPrice": _positive_price("entryPrice") or avg_price,
            "positionAmt": abs(qty_signed),
            "price_status": "confirmed" if avg_price > 0 else "unknown",
        }

    return {"status": "not_found", "symbol": bx_symbol, "positionSide": direction}


def wait_for_position_fill_directional(symbol: str, direction: str, timeout_sec: int = 30, poll_interval: float = 0.5) -> dict:
    started = time.time()
    while time.time() - started < timeout_sec:
        pos = get_position_directional(symbol, direction)
        if pos.get("status") in {"found", "error"}:
            return pos
        time.sleep(poll_interval)

    return {"status": "timeout", "symbol": to_bx_symbol(symbol), "positionSide": str(direction).upper()}


def get_open_protection_directional(
    symbol: str,
    direction: str,
    *,
    timeout_sec: float | None = None,
    retryable: bool = True,
) -> dict:
    bx_symbol = to_bx_symbol(symbol)
    direction = str(direction).upper()
    if not bx_symbol:
        return {"status": "error", "error": "contract_not_found", "tp_orders": [], "sl_orders": []}

    resp = _request(
        "GET",
        OPEN_ORDERS_PATH,
        {"symbol": bx_symbol},
        timeout_sec=timeout_sec,
        retryable=retryable,
    )
    if resp.get("code") not in (0, "0"):
        try:
            code_int = int(resp.get("code"))
        except (TypeError, ValueError, OverflowError):
            code_int = None
        return {
            "status": "error", "error": f"openOrders failed: code={resp.get('code')}:{resp.get('msg')}",
            "tp_orders": [], "sl_orders": [], "code": code_int if code_int is not None else resp.get("code"),
            "retry_after_ms": resp.get("retry_after_ms"),
            "rate_limited": bool(resp.get("rate_limited") or resp.get("local_cooldown") or code_int == 100410),
            "local_cooldown": bool(resp.get("local_cooldown")),
        }

    try:
        orders = _normalize_orders_list(resp)
    except ValueError as exc:
        return {
            "status": "error", "error": f"openOrders response is ambiguous: {exc}",
            "symbol": bx_symbol, "positionSide": direction, "tp_orders": [], "sl_orders": [],
        }

    tp_orders = []
    sl_orders = []

    for order in orders:
        order_type = str(order.get("type", "")).strip().upper()
        if not order_type:
            return {
                "status": "error", "error": "openOrders row is missing order type; protection state is unknown",
                "symbol": bx_symbol, "positionSide": direction, "tp_orders": [], "sl_orders": [],
            }
        row_symbol = str(order.get("symbol") or "").strip()
        has_order_identity = any(str(order.get(key) or "").strip() for key in ("orderId", "orderID", "clientOrderId"))
        protective_type = order_type in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET", "STOP", "STOP_MARKET"}
        if protective_type and not row_symbol:
            # An order ID proves that an order exists, not which contract it
            # protects. Do not attribute a symbol-less conditional to this symbol.
            return {
                "status": "error", "error": "protective openOrders row is missing symbol; protection ownership is unknown",
                "symbol": bx_symbol, "positionSide": direction, "tp_orders": [], "sl_orders": [],
            }
        if not row_symbol and not has_order_identity:
            return {
                "status": "error", "error": "openOrders row is not identifiable by symbol or order id",
                "symbol": bx_symbol, "positionSide": direction, "tp_orders": [], "sl_orders": [],
            }
        if row_symbol:
            normalized_symbol = row_symbol.upper().replace("_", "-").replace("/", "-")
            # Contract aliases (e.g. LIT -> LIGHTER on BingX derivatives) must
            # be resolved to the exchange's canonical symbol before comparison.
            try:
                normalized_symbol = str(to_bx_symbol(normalized_symbol) or normalized_symbol).upper()
            except Exception:
                pass
            if normalized_symbol != bx_symbol:
                continue
        position_side = str(order.get("positionSide", "")).upper()
        if order_type in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET", "STOP", "STOP_MARKET"} and position_side not in {"LONG", "SHORT", "BOTH"}:
            return {
                "status": "error", "error": "protective openOrders row has missing/invalid positionSide",
                "symbol": bx_symbol, "positionSide": direction, "tp_orders": [], "sl_orders": [],
            }
        if position_side not in (direction, "BOTH"):
            continue

        if order_type in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}:
            tp_orders.append(order)
        elif order_type in {"STOP", "STOP_MARKET"}:
            sl_orders.append(order)

    return {"status": "ok", "symbol": bx_symbol, "positionSide": direction, "tp_orders": tp_orders, "sl_orders": sl_orders}


def _round_qty(qty: float, precision: int) -> float:
    if precision < 0:
        return float(qty)
    return float(Decimal(str(qty)).quantize(Decimal(1).scaleb(-precision), rounding=ROUND_DOWN))


def _format_qty(qty: float, precision: int) -> str:
    return f"{qty:.{precision}f}"


def _format_price(price: float, precision: int) -> str:
    return f"{price:.{precision}f}"


def build_tp_client_order_id(leg: str, trade_id: str | None = None) -> str:
    leg_u = str(leg).upper().replace("_", "")
    if leg_u not in {"TP1", "TP2", "TP3"}:
        leg_u = "TP" + "".join(ch for ch in leg_u if ch.isdigit())[:1] or "1"
    if trade_id:
        return f"EVTTP{leg_u[-1]}{_trade_digest(trade_id)}"
    return f"EVT{leg_u}"


def build_sl_client_order_id(trade_id: str | None = None) -> str:
    if trade_id:
        return f"EVTSL{_trade_digest(trade_id)}"
    return "EVTSL"


def _select_supported_tp_levels(tp_levels_norm: list[dict], position_qty: float, precision: int, min_qty: float) -> list[dict]:
    """Select at most two TP legs using actual filled quantity constraints.

    New strategy profiles already contain TP1 and terminal TP2. If a legacy or
    custom caller supplies three levels, keep its first and final target and drop
    the obsolete middle milestone; the final target is relabelled TP2. If the
    position cannot support both legs, use the nearest target (TP1) for the
    entire position. This avoids leaving micro-positions with an unreachable
    terminal target. Feasibility is tested with the same allocator used for
    real TP orders.
    """
    if not tp_levels_norm or position_qty <= 0:
        return []

    try:
        precision_i = int(precision)
        min_qty_f = max(float(min_qty or 0.0), 0.0)
        position = float(position_qty)
        if not math.isfinite(position) or position <= 0:
            return []
        if not math.isfinite(min_qty_f):
            min_qty_f = 0.0
    except (TypeError, ValueError, ArithmeticError):
        return []

    available_levels = [dict(x) for x in tp_levels_norm if isinstance(x, dict)]
    if not available_levels:
        return []
    if len(available_levels) > 2:
        # Keep the early partial and final target, remove the former middle TP.
        final_level = dict(available_levels[-1])
        final_level["leg"] = "tp2"
        available_levels = [available_levels[0], final_level]

    # Try the largest executable profile first. The allocator is the single
    # source of truth for whether the selected fractions can be represented by
    # the actual filled quantity, precision, and minimum executable quantity.
    for candidate_count in range(len(available_levels), 0, -1):
        if candidate_count == 1:
            # A single executable order must realize the nearest profit target,
            # not the farthest target intended for the remaining partial size.
            candidate = [dict(available_levels[0])]
            candidate[0]["leg"] = "tp1"
            candidate[0]["close_fraction"] = 1.0
        else:
            candidate = [dict(x) for x in available_levels[:candidate_count]]
        try:
            _allocate_tp_quantities(
                position_qty=position,
                precision=precision_i,
                min_qty=min_qty_f,
                fractions=[x["close_fraction"] for x in candidate],
            )
        except (TypeError, ValueError, ArithmeticError):
            continue
        return candidate

    return []


def _allocate_tp_quantities(position_qty: float, precision: int, min_qty: float, fractions: list[float]) -> list[float]:
    if position_qty <= 0:
        raise ValueError("position_qty must be > 0")
    if not fractions or any(f <= 0 for f in fractions):
        raise ValueError("fractions must be positive")

    step = Decimal(1).scaleb(-precision) if precision >= 0 else Decimal("1")
    pos = Decimal(str(position_qty))
    min_q = Decimal(str(max(min_qty, 0.0)))
    min_leg = max(step, min_q)
    k = len(fractions)

    if pos < min_leg * k:
        raise ValueError(f"position_qty={position_qty} cannot support {k} TP legs with min_leg={min_leg}")

    total_fraction = sum(Decimal(str(f)) for f in fractions)
    normalized = [Decimal(str(f)) / total_fraction for f in fractions]
    raw_targets = [pos * f for f in normalized]

    quantities = []
    for raw in raw_targets:
        q_step = (raw / step).to_integral_value(rounding=ROUND_FLOOR) * step
        quantities.append(max(q_step, min_leg))

    while sum(quantities) > pos:
        best_reduce = max(range(k), key=lambda i: (quantities[i] - min_leg, quantities[i] - raw_targets[i]))
        if quantities[best_reduce] <= min_leg:
            raise ValueError("Cannot reduce leg below min_leg")
        quantities[best_reduce] -= step

    while sum(quantities) < pos:
        best_add = max(range(k), key=lambda i: raw_targets[i] - quantities[i])
        quantities[best_add] += step

    remainder = pos - sum(quantities)
    if remainder != 0:
        quantities[-1] += remainder
        quantities[-1] = quantities[-1].quantize(step)

    return [float(q) for q in quantities]


def _normalize_tp_levels(tp_levels: list) -> list[dict]:
    normalized = []
    for tp in (tp_levels or []):
        leg = str(tp.get("leg", f"tp{len(normalized) + 1}"))
        try:
            pnl_pct = float(tp.get("pnl_pct", 0))
            fraction = float(tp.get("close_fraction", 0))
        except (TypeError, ValueError):
            continue

        if not math.isfinite(pnl_pct) or not math.isfinite(fraction) or pnl_pct <= 0 or fraction <= 0:
            continue
        normalized.append({"leg": leg, "pnl_pct": pnl_pct, "close_fraction": fraction})

    if not normalized:
        normalized = [{"leg": "tp1", "pnl_pct": 2.0, "close_fraction": 1.0}]

    total = sum(x["close_fraction"] for x in normalized)
    for x in normalized:
        x["close_fraction"] /= total
    return normalized


def _tp_leg_from_order(order: dict, expected_leg: str, expected_price: float, price_precision: int, trade_id: str | None = None) -> bool:
    order_type = str(order.get("type", "")).upper()
    if order_type not in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}:
        return False

    try:
        actual_price = float(order.get("stopPrice", 0) or order.get("price", 0) or 0)
    except (TypeError, ValueError):
        return False

    if actual_price <= 0:
        return False

    expected_leg = str(expected_leg).upper()
    client_id = str(order.get("clientOrderId", "")).upper()

    unit = 10 ** (-max(0, int(price_precision)))
    price_tolerance = max(unit * 0.51, 1e-12)
    def _price_matches() -> bool:
        return abs(actual_price - expected_price) <= price_tolerance

    expected_formatted = _format_price(expected_price, price_precision)
    actual_formatted = _format_price(actual_price, price_precision)

    # When a client id explicitly identifies a TP leg, that identity is
    # authoritative. Do not fall back to trigger-price matching for a different
    # leg, otherwise one order can be assigned to multiple TP legs when prices
    # happen to coincide. Current conditional BingX orders may have no
    # clientOrderId at all, so price matching remains the fallback for that case.
    declared_leg_num = None
    if client_id.startswith("EVTTP") and len(client_id) > len("EVTTP") and client_id[len("EVTTP")].isdigit():
        declared_leg_num = client_id[len("EVTTP")]
    else:
        marker = "_TP"
        marker_pos = client_id.rfind(marker)
        if marker_pos >= 0 and marker_pos + len(marker) < len(client_id) and client_id[marker_pos + len(marker)].isdigit():
            declared_leg_num = client_id[marker_pos + len(marker)]

    expected_leg_num = "".join(ch for ch in expected_leg if ch.isdigit())[:1]
    if declared_leg_num is not None:
        if declared_leg_num != expected_leg_num:
            return False

    # Prefer the legacy/deterministic client id when it is present. New
    # conditional BingX protection orders do not support clientOrderId, so for
    # those orders the trigger price becomes the stable leg identity.
    if trade_id and client_id:
        digest = _trade_digest(trade_id)
        if client_id == f"EVTTP{expected_leg_num}{digest}":
            return _price_matches()
        # Backward compatibility for protection orders created before the
        # alphanumeric clientOrderId hardening.
        if client_id == f"EVT_{digest}_{expected_leg}":
            return _price_matches()

    if client_id:
        if f"_{expected_leg}_" in f"_{client_id}_":
            return _price_matches()
        if client_id.startswith(f"EVTTP{expected_leg_num}"):
            return _price_matches()

    # Current BingX conditional protection orders: identify the leg by the
    # expected trigger price when no clientOrderId is available.
    return _price_matches()


def _parse_retry_after_ms(message: object) -> int | None:
    import re

    text = str(message or "")
    match = re.search(r"retry\s+after\s+time\s*:\s*(\d{13})", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _live_price_in_cooldown(bx_symbol: str, now: float) -> bool:
    global _LIVE_PRICE_INVALID_UNTIL, _LIVE_PRICE_COOLDOWN_UNTIL

    if _LIVE_PRICE_COOLDOWN_UNTIL > now:
        return True
    until = float(_LIVE_PRICE_INVALID_UNTIL.get(bx_symbol, 0.0))
    if until > now:
        return True
    if until:
        _LIVE_PRICE_INVALID_UNTIL.pop(bx_symbol, None)
    return False


def get_live_price(symbol: str) -> float | None:
    """Return the current BingX live ticker price, failing closed on errors.

    A 109425 response triggers one fresh contract-catalog lookup and one ticker
    retry when BingX still exposes a valid contract. Confirmed-invalid symbols are
    quarantined locally. There is no global invalid-symbol breaker, so an
    unrelated bad contract cannot disable all valid live-price requests.
    """
    bx_symbol = to_bx_symbol(symbol)
    if not bx_symbol:
        log.warning("[BINGX] Live price unavailable: unknown symbol=%s", symbol)
        return None

    now = time.time()
    if _live_price_in_cooldown(bx_symbol, now):
        return None

    global _LIVE_PRICE_COOLDOWN_UNTIL
    for attempt in range(2):
        try:
            resp = _request("GET", LIVE_PRICE_PATH, {"symbol": bx_symbol}, signed=False, retryable=False)
        except Exception as exc:
            log.warning("[BINGX] Failed to read live price for %s: %s", bx_symbol, exc)
            return None

        try:
            code = int(resp.get("code"))
        except (TypeError, ValueError, AttributeError):
            log.warning("[BINGX] Malformed live-price response for %s: %r", bx_symbol, resp)
            return None

        if code == 109425:
            # A contract can disappear between the cycle-level catalog refresh and
            # this ticker request. Re-resolve once from a fresh catalog; if BingX
            # maps the logical symbol to a different active contract, retry it once.
            if attempt == 0:
                try:
                    fresh_contract = get_contract(symbol, require_fresh=True)
                except Exception:
                    fresh_contract = None
                fresh_symbol = str((fresh_contract or {}).get("symbol", "")).strip().upper()
                if fresh_symbol and fresh_symbol != bx_symbol:
                    bx_symbol = fresh_symbol
                elif fresh_symbol == bx_symbol:
                    pass
                else:
                    _LIVE_PRICE_INVALID_UNTIL[bx_symbol] = now + LIVE_PRICE_INVALID_SYMBOL_COOLDOWN_SEC
                    log.warning("[BINGX] Live-price symbol unsupported after fresh catalog check: %s (109425); symbol quarantined %.0fs", bx_symbol, LIVE_PRICE_INVALID_SYMBOL_COOLDOWN_SEC)
                    return None
                continue
            _LIVE_PRICE_INVALID_UNTIL[bx_symbol] = now + LIVE_PRICE_INVALID_SYMBOL_COOLDOWN_SEC
            log.warning("[BINGX] Live-price symbol unsupported: %s (109425); symbol quarantined %.0fs", bx_symbol, LIVE_PRICE_INVALID_SYMBOL_COOLDOWN_SEC)
            return None

        if code == 109429:
            retry_after_ms = _parse_retry_after_ms(resp.get("msg"))
            retry_until = (retry_after_ms / 1000.0) if retry_after_ms else (now + LIVE_PRICE_RATE_LIMIT_FALLBACK_SEC)
            _LIVE_PRICE_COOLDOWN_UNTIL = max(_LIVE_PRICE_COOLDOWN_UNTIL, retry_until)
            log.error("[BINGX] Live-price API temporarily restricted (109429) for %s; live-price HTTP requests paused until %.0f.", bx_symbol, _LIVE_PRICE_COOLDOWN_UNTIL)
            return None

        if code != 0:
            log.warning("[BINGX] Live-price API error for %s: code=%s msg=%s", bx_symbol, code, resp.get("msg"))
            return None

        data = resp.get("data")
        if isinstance(data, list):
            # A symbol-filtered endpoint may still return a list. Never fall back
            # to its first row: that row may belong to another contract.
            matching_rows = [
                item for item in data
                if isinstance(item, dict)
                and str(item.get("symbol", "")).strip().upper().replace("_", "-").replace("/", "-") == bx_symbol
            ]
            row = matching_rows[0] if len(matching_rows) == 1 else None
            if len(matching_rows) > 1:
                log.warning("[BINGX] Live-price response contains duplicate rows for %s", bx_symbol)
                return None
        elif isinstance(data, dict):
            row = data
        else:
            row = None
        if not isinstance(row, dict):
            log.warning("[BINGX] Live-price response has no unambiguous row for %s", bx_symbol)
            return None
        row_symbol = str(row.get("symbol") or "").strip().upper().replace("_", "-").replace("/", "-")
        if row_symbol != bx_symbol:
            log.error(
                "[BINGX] Refusing live price for %s: response symbol is %r",
                bx_symbol, row.get("symbol"),
            )
            return None
        try:
            price = float(row.get("price"))
        except (TypeError, ValueError):
            log.warning("[BINGX] Invalid live price for %s: %r", bx_symbol, row.get("price"))
            return None
        if not math.isfinite(price) or price <= 0:
            log.warning("[BINGX] Non-positive live price for %s: %r", bx_symbol, row.get("price"))
            return None
        return price
    return None

def _current_close_price(symbol: str) -> float | None:
    """Backward-compatible name for the live execution/reference price."""
    return get_live_price(symbol)



def _qty_matches_position(order_qty: float, position_qty: float) -> bool:
    if order_qty <= 0 or position_qty <= 0:
        return False
    return abs(order_qty - position_qty) <= max(position_qty * 1e-6, 1e-12)


def _effective_weighted_rr(levels: list[dict], stop_loss_pct: float) -> float | None:
    if stop_loss_pct <= 0 or not levels:
        return None
    total = 0.0
    weighted = 0.0
    for level in levels:
        try:
            pnl = float(level.get("pnl_pct", 0))
            weight = float(level.get("qty", 0) or level.get("close_fraction", 0))
        except (TypeError, ValueError):
            continue
        if pnl > 0 and weight > 0:
            total += weight
            weighted += weight * (pnl / stop_loss_pct)
    return weighted / total if total > 0 else None


def _protection_order_matches_params(order: dict, params: dict) -> bool:
    """Match an uncertain conditional-order POST against an open order by shape.

    Conditional BingX orders do not support clientOrderId according to the current
    swap-trade API contract, so transport recovery must rely on immutable order
    attributes instead of a client id.
    """
    if not isinstance(order, dict) or not isinstance(params, dict):
        return False
    if str(order.get("type", "")).upper() != str(params.get("type", "")).upper():
        return False
    if str(order.get("positionSide", "")).upper() != str(params.get("positionSide", "")).upper():
        return False
    if str(order.get("side", "")).upper() != str(params.get("side", "")).upper():
        return False

    def _num(value):
        try:
            x = float(value)
            return x if math.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    expected_qty = _num(params.get("quantity"))
    actual_qty = _num(order.get("origQty", order.get("quantity")))
    if expected_qty is not None and actual_qty is not None:
        if not _qty_matches_position(actual_qty, expected_qty):
            return False
    elif expected_qty is not None:
        return False

    order_type = str(params.get("type", "")).upper()
    if order_type in {"STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET"}:
        expected_stop = _num(params.get("stopPrice"))
        actual_stop = _num(order.get("stopPrice", order.get("price")))
        if expected_stop is None or actual_stop is None:
            return False
        if order_type in {"STOP", "STOP_MARKET"}:
            price_matches = _sl_price_matches(actual_stop, expected_stop)
        else:
            price_matches = abs(actual_stop - expected_stop) / max(expected_stop, 1e-12) <= 0.002
        if not price_matches:
            return False
    return True


def _find_open_order_by_client_id(symbol: str, direction: str, client_order_id: str) -> tuple[str, dict | None]:
    """Legacy-compatible client-id lookup; MARKET/LIMIT orders may use this."""
    if not client_order_id:
        return "absent", None
    try:
        prot = get_open_protection_directional(symbol, direction, retryable=False)
    except Exception as exc:
        log.warning("[BINGX] Protection verification failed for %s %s: %s", symbol, direction, exc)
        return "unknown", None
    if prot.get("status") != "ok":
        return "unknown", None
    wanted = str(client_order_id).upper()
    for order in list(prot.get("sl_orders", [])) + list(prot.get("tp_orders", [])):
        if str(order.get("clientOrderId", "")).upper() == wanted:
            return "found", order
    return "absent", None


def _find_open_order_for_post(symbol: str, direction: str, params: dict, client_order_id: str | None = None) -> tuple[str, dict | None]:
    """Return (found/absent/unknown, order) for an uncertain POST outcome."""
    try:
        prot = get_open_protection_directional(symbol, direction, retryable=False)
    except Exception as exc:
        log.warning("[BINGX] Protection verification failed for %s %s: %s", symbol, direction, exc)
        return "unknown", None
    if prot.get("status") != "ok":
        return "unknown", None

    orders = list(prot.get("sl_orders", [])) + list(prot.get("tp_orders", []))
    if client_order_id:
        wanted = str(client_order_id).upper()
        for order in orders:
            if str(order.get("clientOrderId", "")).upper() == wanted:
                return "found", order

    for order in orders:
        if _protection_order_matches_params(order, params):
            return "found", order
    return "absent", None


def _post_protection_order_verified(
    symbol: str, direction: str, params: dict, client_order_id: str | None = None,
    *, max_attempts: int = 2, retry_delay: float = 0.25,
) -> dict:
    """POST protection order with safe recovery when the POST outcome is unknown.

    For conditional protection orders, current BingX docs do not support
    clientOrderId, so recovery matches the open order by order shape. When state
    cannot be proven, never issue a blind duplicate; reconciliation repairs it later.
    """
    last_resp: dict = {"code": -1, "msg": "protection request not attempted"}
    order_type = str(params.get("type", "")).upper()
    conditional = order_type in {"STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET"}

    request_params = dict(params)
    if not conditional and client_order_id and not request_params.get("clientOrderId"):
        request_params["clientOrderId"] = str(client_order_id)

    for attempt in range(max_attempts):
        try:
            resp = _request("POST", ORDER_PATH, request_params)
        except Exception as exc:
            resp = {"code": -1, "msg": str(exc)}
        if isinstance(resp, dict) and resp.get("code") in (0, "0"):
            data = resp.get("data") or {}
            order = data.get("order") if isinstance(data, dict) else None
            if not isinstance(order, dict):
                order = data if isinstance(data, dict) else {}
            order_id = str(order.get("orderId", "") or "").strip()
            if not order_id:
                return {
                    "code": -1,
                    "msg": "successful protection response is missing orderId",
                    "protection_state_unknown": True,
                    "malformed_success_response": True,
                }
            return resp

        last_resp = resp if isinstance(resp, dict) else {"code": -1, "msg": str(resp)}
        is_transport = last_resp.get("code") == -1 and "missing bingx credentials" not in str(last_resp.get("msg", "")).lower()
        if not is_transport:
            return last_resp

        state, found = _find_open_order_for_post(symbol, direction, params, client_order_id)
        if state == "found" and found is not None:
            return {
                "code": 0,
                "data": {"order": found},
                "recovered": True,
                "recovery": "open_order_verified_after_transport_error",
            }
        if state != "absent":
            return {
                "code": -1,
                "msg": f"{last_resp.get('msg', 'transport error')}; protection state unknown after POST",
                "protection_state_unknown": True,
            }

        # For a conditional order we cannot prove that the POST was lost if the
        # order is not open anymore: it may have immediately triggered/filled.
        # Never send a duplicate in that state; the next reconciliation cycle
        # will inspect the live position and repair what is actually missing.
        if conditional:
            # Distinguish a genuinely disappeared position from a still-open
            # position, but never infer that the conditional POST was lost. It
            # may already have triggered/filled between the POST and verification.
            try:
                live_pos = get_position_directional(symbol, direction)
                live_status = str(live_pos.get("status", "")).lower()
                live_qty = abs(float(live_pos.get("positionAmt", 0) or 0)) if live_status == "found" else 0.0
            except Exception as exc:
                return {
                    "code": -1,
                    "msg": f"{last_resp.get('msg', 'transport error')}; conditional protection verification failed: {exc}",
                    "protection_state_unknown": True,
                }
            if live_status == "not_found":
                return {
                    "code": -1,
                    "msg": f"{last_resp.get('msg', 'transport error')}; position absence confirmed by position endpoint",
                    "protection_state_unknown": True,
                    "position_gone": True,
                }
            if live_status != "found" or live_qty <= 0:
                return {
                    "code": -1,
                    "msg": f"{last_resp.get('msg', 'transport error')}; position state remains unknown after conditional-order timeout: {live_pos.get('error', live_status)}",
                    "protection_state_unknown": True,
                    "position_gone": False,
                }
            return {
                "code": -1,
                "msg": f"{last_resp.get('msg', 'transport error')}; conditional protection outcome unknown; no blind retry",
                "protection_state_unknown": True,
            }

        try:
            live_pos = get_position_directional(symbol, direction)
            live_status = str(live_pos.get("status", "")).lower()
            live_qty = abs(float(live_pos.get("positionAmt", 0) or 0)) if live_status == "found" else 0.0
        except Exception as exc:
            return {
                "code": -1,
                "msg": f"{last_resp.get('msg', 'transport error')}; live position verification failed: {exc}",
                "protection_state_unknown": True,
            }
        if live_status == "not_found":
            return {
                "code": -1,
                "msg": f"{last_resp.get('msg', 'transport error')}; position absence confirmed by position endpoint",
                "protection_state_unknown": True,
                "position_gone": True,
            }
        if live_status != "found" or live_qty <= 0:
            return {
                "code": -1,
                "msg": f"{last_resp.get('msg', 'transport error')}; position state remains unknown after timeout: {live_pos.get('error', live_status)}",
                "protection_state_unknown": True,
                "position_gone": False,
            }

        # MARKET TP/close can fill between POST and verification. Never recreate.
        if order_type == "MARKET":
            return {
                "code": -1,
                "msg": f"{last_resp.get('msg', 'transport error')}; MARKET outcome unknown after timeout; no blind retry",
                "protection_state_unknown": True,
            }

        # Kept for any future non-conditional, non-MARKET order type.
        params = dict(params)
        try:
            precision = int((get_contract(symbol) or {}).get("quantityPrecision") or 0)
            params["quantity"] = _format_qty(_round_qty(live_qty, precision), precision)
        except Exception:
            pass

        if attempt + 1 < max_attempts:
            time.sleep(retry_delay)
            continue
        return last_resp

    return last_resp

def _sl_price_matches(actual_price: float, expected_price: float, tolerance: float = 0.002) -> bool:
    return actual_price > 0 and expected_price > 0 and abs(actual_price - expected_price) / max(expected_price, 1e-12) <= tolerance




def _cancel_protection_order_verified(symbol: str, direction: str, order: dict, *, max_attempts: int = 3) -> tuple[bool, str]:
    order_id = str(order.get("orderId", ""))
    if not order_id:
        return False, "missing orderId"
    last = ""
    for attempt in range(max_attempts):
        try:
            resp = cancel_order(symbol, order_id)
        except Exception as exc:
            resp = {"code": -1, "msg": str(exc)}
        if isinstance(resp, dict) and resp.get("code") in (0, "0"):
            try:
                latest = get_open_protection_directional(symbol, direction)
                if latest.get("status") == "ok":
                    open_ids = {str(o.get("orderId", "")) for o in (latest.get("sl_orders", []) + latest.get("tp_orders", []))}
                    if order_id not in open_ids:
                        return True, "cancelled_and_verified"
                    last = "cancel acknowledged but order is still visible"
                else:
                    last = "cancel acknowledged but openOrders verification failed"
            except Exception as exc:
                last = f"cancel acknowledged but verification failed: {exc}"
        else:
            last = str(resp.get("msg", resp)) if isinstance(resp, dict) else str(resp)
        try:
            latest = get_open_protection_directional(symbol, direction)
            if latest.get("status") == "ok":
                open_ids = {str(o.get("orderId", "")) for o in (latest.get("sl_orders", []) + latest.get("tp_orders", []))}
                if order_id not in open_ids:
                    return True, "already_gone"
        except Exception:
            pass
        if attempt + 1 < max_attempts:
            time.sleep(0.25 * (attempt + 1))
    return False, last or "cancel failed"


def cleanup_directional_protection_after_close(
    symbol: str, direction: str, *, confirm_reads: int = 2, retry_delay: float = 0.15
) -> dict:
    """Remove and verify every conditional order only after closure is confirmed.

    Rollback entries are not normal tracked trades, so the tracker cannot clean
    their SL/TP later unless this explicit cleanup succeeds or is queued by the
    caller. Two fresh NOT_FOUND reads prevent cancelling protection for a position
    that is still open or whose state is merely unavailable.
    """
    direction = str(direction or "").upper()
    if direction not in {"LONG", "SHORT"} or not to_bx_symbol(symbol):
        return {"status": "error", "error": "invalid_symbol_or_direction"}

    confirmed_absent = 0
    last_position: dict = {}
    for read_index in range(max(2, int(confirm_reads))):
        try:
            last_position = get_position_directional(symbol, direction)
        except Exception as exc:
            return {"status": "pending", "error": f"position_confirmation_exception:{exc}", "cancelled_order_ids": []}
        pos_status = str((last_position or {}).get("status", "")).lower()
        if pos_status == "found" and abs(float((last_position or {}).get("positionAmt", 0) or 0)) > 1e-12:
            return {"status": "position_still_open", "error": "refusing to cancel protection while position is open", "position": last_position, "cancelled_order_ids": []}
        if pos_status != "not_found":
            return {"status": "pending", "error": f"position_state_not_confirmed:{pos_status or 'unknown'}", "cancelled_order_ids": []}
        confirmed_absent += 1
        if read_index + 1 < max(2, int(confirm_reads)):
            time.sleep(max(0.0, retry_delay))
    if confirmed_absent < 2:
        return {"status": "pending", "error": "position_absence_not_confirmed_twice", "cancelled_order_ids": []}

    try:
        snapshot = get_open_protection_directional(symbol, direction, retryable=False)
    except Exception as exc:
        return {"status": "pending", "error": f"open_protection_read_exception:{exc}", "cancelled_order_ids": []}
    if not isinstance(snapshot, dict) or snapshot.get("status") != "ok":
        return {"status": "pending", "error": str((snapshot or {}).get("error", "open_protection_state_unknown")) if isinstance(snapshot, dict) else "malformed_open_protection_snapshot", "cancelled_order_ids": []}

    orders = list(snapshot.get("sl_orders", [])) + list(snapshot.get("tp_orders", []))
    cancelled: list[str] = []
    failures: list[str] = []
    seen: set[str] = set()
    for order in orders:
        if not isinstance(order, dict):
            failures.append("non_object_protection_order")
            continue
        order_id = str(order.get("orderId") or order.get("orderID") or "").strip()
        if not order_id:
            failures.append("protection_order_missing_order_id")
            continue
        if order_id in seen:
            continue
        seen.add(order_id)
        try:
            ok, note = _cancel_protection_order_verified(symbol, direction, order, max_attempts=3)
        except Exception as exc:
            ok, note = False, str(exc)
        if ok:
            cancelled.append(order_id)
        else:
            failures.append(f"{order_id}:{note}")

    try:
        final_snapshot = get_open_protection_directional(symbol, direction, retryable=False)
    except Exception as exc:
        return {"status": "pending", "error": f"final_protection_verification_exception:{exc}", "cancelled_order_ids": cancelled}
    if not isinstance(final_snapshot, dict) or final_snapshot.get("status") != "ok":
        return {"status": "pending", "error": str((final_snapshot or {}).get("error", "final_protection_state_unknown")) if isinstance(final_snapshot, dict) else "malformed_final_protection_snapshot", "cancelled_order_ids": cancelled}
    remaining = list(final_snapshot.get("sl_orders", [])) + list(final_snapshot.get("tp_orders", []))
    if remaining:
        failures.append("protection_orders_remain_after_cleanup")
    return {
        "status": "cleaned" if not failures else "pending",
        "error": None if not failures else "; ".join(failures),
        "cancelled_order_ids": cancelled,
        "remaining_order_ids": [str(o.get("orderId") or o.get("orderID") or "") for o in remaining if isinstance(o, dict)],
    }


def ensure_directional_protection(
    symbol: str, direction: str, avg_price: float, qty: float,
    stop_loss_pct: float, tp_levels: list, trade_id: str | None = None,
    stop_loss_price: float | None = None,
) -> dict:
    direction = str(direction).upper()
    if direction not in {"LONG", "SHORT"}:
        return {"status": "error", "error": f"invalid direction={direction}"}

    try:
        avg_price = float(avg_price)
        qty = abs(float(qty))
        stop_loss_pct = float(stop_loss_pct)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc)}

    if not math.isfinite(avg_price) or not math.isfinite(qty) or not math.isfinite(stop_loss_pct) or avg_price <= 0 or qty <= 0 or not (0 < stop_loss_pct <= 25):
        return {"status": "error", "error": "invalid protection parameters"}

    bx_symbol = to_bx_symbol(symbol)
    contract = get_contract(symbol)
    if not bx_symbol or not contract:
        return {"status": "error", "error": f"contract not found: {bx_symbol}"}

    try:
        precision = int(contract.get("quantityPrecision") or 0)
        price_precision = int(contract.get("pricePrecision") or 4)
        min_qty = float(contract.get("tradeMinQuantity") or contract.get("minQty") or 0)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": f"invalid contract parameters: {exc}"}

    position_qty = _round_qty(qty, precision)
    if position_qty <= 0 or (min_qty > 0 and position_qty < min_qty):
        return {"status": "error", "error": f"qty={position_qty} < minQty={min_qty}"}

    existing = get_open_protection_directional(symbol, direction)
    if existing.get("status") != "ok":
        return {
            "status": "PROTECTION_FAILED",
            "error": existing.get("error", "openOrders unavailable"),
            "protection_audit": {
                "open_orders_status": existing.get("status"),
                "open_orders_error": existing.get("error"),
                "audit_reason": "OPEN_ORDERS_UNVERIFIABLE",
                "sl_count": None, "tp_count": None,
                "expected_position_qty": position_qty,
            },
        }

    existing_tp = list(existing.get("tp_orders", []))
    existing_sl = list(existing.get("sl_orders", []))
    tp_levels_norm = _normalize_tp_levels(tp_levels)

    if stop_loss_price is not None:
        try:
            explicit_sl_price = float(stop_loss_price)
        except (TypeError, ValueError):
            return {"status": "error", "error": "invalid explicit stop_loss_price"}
        if not math.isfinite(explicit_sl_price) or explicit_sl_price <= 0:
            return {"status": "error", "error": "invalid explicit stop_loss_price"}
        if direction == "LONG" and explicit_sl_price > avg_price:
            return {"status": "error", "error": "LONG stop_loss_price is above avg_price"}
        if direction == "SHORT" and explicit_sl_price < avg_price:
            return {"status": "error", "error": "SHORT stop_loss_price is below avg_price"}
        desired_sl_price = explicit_sl_price
    else:
        desired_sl_price = avg_price * (1.0 - stop_loss_pct / 100.0) if direction == "LONG" else avg_price * (1.0 + stop_loss_pct / 100.0)
    valid_existing_sl = None
    for sl in existing_sl:
        order_type = str(sl.get("type", "")).upper()
        if order_type not in {"STOP", "STOP_MARKET"}:
            continue
        try:
            sl_price = float(sl.get("stopPrice", 0) or sl.get("price", 0) or 0)
            sl_qty = float(sl.get("origQty", 0) or sl.get("quantity", 0) or 0)
        except (TypeError, ValueError):
            continue

        if sl_price <= 0 or sl_qty <= 0:
            continue

        is_break_even = _sl_price_matches(desired_sl_price, avg_price) and _sl_price_matches(sl_price, avg_price)
        protective_side = (sl_price < avg_price) if direction == "LONG" else (sl_price > avg_price)
        if is_break_even:
            protective_side = True
        if protective_side and _sl_price_matches(sl_price, desired_sl_price) and _qty_matches_position(sl_qty, position_qty):
            valid_existing_sl = sl
            break

    if valid_existing_sl is not None:
        # Keep exactly one current SL; remove duplicate/stale SL orders first.
        stale_sls = [o for o in existing_sl if str(o.get("orderId", "")) != str(valid_existing_sl.get("orderId", ""))]
        for stale in stale_sls:
            ok, note = _cancel_protection_order_verified(symbol, direction, stale)
            if not ok:
                return {"status": "PROTECTION_FAILED", "error": f"stale SL cancel failed: {note}"}
        sl = valid_existing_sl
        sl_result = {
            "status": "already_exists",
            "order_id": str(sl.get("orderId", "")),
            "client_order_id": str(sl.get("clientOrderId", "")),
            "stop_price": float(sl.get("stopPrice", 0) or sl.get("price", 0) or 0),
            "qty": float(sl.get("origQty", 0) or sl.get("quantity", 0) or position_qty),
        }
    else:
        # Never create a new SL while an old/wrong SL is still live.
        for stale in existing_sl:
            ok, note = _cancel_protection_order_verified(symbol, direction, stale)
            if not ok:
                return {"status": "PROTECTION_FAILED", "error": f"old SL cancel failed: {note}; refusing to create a second SL"}
        sl_price = desired_sl_price
        client_order_id = build_sl_client_order_id(trade_id)
        params = {
            "symbol": bx_symbol,
            "side": "SELL" if direction == "LONG" else "BUY",
            "positionSide": direction,
            "type": "STOP_MARKET",
            "stopPrice": _format_price(sl_price, price_precision),
            "quantity": _format_qty(position_qty, precision),
        }

        resp = _post_protection_order_verified(symbol, direction, params, client_order_id)
        if resp.get("code") not in (0, "0"):
            log.error("[BINGX] SL failed: code=%s msg=%s", resp.get("code"), resp.get("msg"))
            rollback = emergency_close_position(symbol, direction, position_qty, reason_token=f"SLFAIL:{trade_id or bx_symbol}")
            return {
                "status": "PROTECTION_FAILED",
                "error": f"SL failed: {resp.get('msg')}; emergency_close={rollback.get('status')}",
                "sl_result": {"status": "error", "error": resp.get("msg")},
                "tp_orders": [],
                "rolled_back": rollback.get("status") == "closed",
                "emergency_close": rollback,
            }

        order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
        sl_result = {
            "status": "created",
            "order_id": str(order.get("orderId", "")),
            "client_order_id": order.get("clientOrderId") or None,
            "stop_price": sl_price,
            "qty": position_qty,
        }

    verified = get_open_protection_directional(symbol, direction)
    verified_sl_valid, protection_audit = _audit_exact_single_sl(
        verified, direction=direction, avg_price=avg_price,
        expected_price=desired_sl_price, expected_qty=position_qty,
    )

    if not verified_sl_valid:
        # Preserve the existing fail-safe post-entry policy: if the exchange
        # confirms missing/duplicate/mismatched protection, try to flatten the
        # just-opened residual. A query error is explicitly identified as UNKNOWN
        # in the journal payload rather than described as proof that SL is absent.
        rollback = emergency_close_position(symbol, direction, position_qty, reason_token=f"SLVERIFY:{trade_id or bx_symbol}")
        return {
            "status": "SL_UNVERIFIED",
            "symbol": symbol,
            "bx_symbol": bx_symbol,
            "direction": direction,
            "avg_price": avg_price,
            "qty": position_qty,
            "sl_result": sl_result,
            "tp_orders": [],
            "protection_audit": protection_audit,
            "error": f"SL invariant failed after placement: {protection_audit.get('audit_reason')}",
            "rolled_back": rollback.get("status") == "closed",
            "emergency_close": rollback,
        }

    # Adapt the two-TP profile to the confirmed exchange position size. New
    # positions never receive three TP orders; if two legs are not executable,
    # retain one terminal TP2 leg rather than place invalid partial orders.
    tp_levels_norm = _select_supported_tp_levels(
        tp_levels_norm, position_qty=position_qty, precision=precision, min_qty=min_qty
    )
    if not tp_levels_norm:
        return {
            "status": "PROTECTION_FAILED",
            "symbol": symbol,
            "bx_symbol": bx_symbol,
            "direction": direction,
            "avg_price": avg_price,
            "qty": position_qty,
            "sl_result": sl_result,
            "tp_orders": [],
            "error": "no TP profile can be supported by the filled position quantity",
            "protection_audit": protection_audit,
            "rolled_back": False,
        }
    tp_mode = "single_tp" if len(tp_levels_norm) == 1 else "two_tp"
    log.info(
        "[BINGX] %s %s TP profile selected: mode=%s legs=%s qty=%s min_qty=%s",
        symbol, direction, tp_mode, ",".join(str(x.get("leg", "")).upper() for x in tp_levels_norm),
        _format_qty(position_qty, precision), _format_qty(min_qty, precision) if min_qty > 0 else "0",
    )

    try:
        desired_qtys = _allocate_tp_quantities(
            position_qty=position_qty,
            precision=precision,
            min_qty=min_qty,
            fractions=[x["close_fraction"] for x in tp_levels_norm],
        )
    except ValueError as exc:
        return {
            "status": "PROTECTION_FAILED",
            "symbol": symbol,
            "bx_symbol": bx_symbol,
            "direction": direction,
            "avg_price": avg_price,
            "qty": position_qty,
            "sl_result": sl_result,
            "tp_orders": [],
            "error": str(exc),
            "protection_audit": protection_audit,
        }

    tp_results = []
    stale_tp_cancellations = []
    # Covers failures both when replacing a wrong-quantity TP for a desired leg
    # and when removing an unclaimed stale TP after creating desired protection.
    # A failed cancellation means an unexpected live TP may remain on the exchange.
    stale_tp_cleanup_failed = False
    handled_existing_tp_ids: set[str] = set()
    current_price = None
    current_price_checked = False

    for level, tp_qty in zip(tp_levels_norm, desired_qtys):
        leg = str(level["leg"])
        pnl_pct = float(level["pnl_pct"])
        tp_price = avg_price * (1.0 + pnl_pct / 100.0) if direction == "LONG" else avg_price * (1.0 - pnl_pct / 100.0)

        existing_leg = None
        for order in existing_tp:
            existing_order_id = str(order.get("orderId", ""))
            if existing_order_id and existing_order_id in handled_existing_tp_ids:
                continue
            if _tp_leg_from_order(order, leg, tp_price, price_precision, trade_id):
                existing_leg = order
                break

        if existing_leg:
            existing_order_id = str(existing_leg.get("orderId", ""))
            if existing_order_id:
                handled_existing_tp_ids.add(existing_order_id)
            existing_qty = float(existing_leg.get("origQty", 0) or existing_leg.get("quantity", 0) or 0)
            # An existing TP is reusable only when both price and quantity match
            # the current desired leg. Reusing a smaller/older order can leave
            # part of the position unprotected; reusing a larger one can over-close.
            qty_matches = abs(existing_qty - tp_qty) <= max(tp_qty * 1e-6, 1e-12)
            if qty_matches:
                tp_results.append(
                    {
                        "leg": leg,
                        "status": "already_exists",
                        "order_id": str(existing_leg.get("orderId", "")),
                        "client_order_id": str(existing_leg.get("clientOrderId", "")),
                        "price": float(existing_leg.get("stopPrice", 0) or existing_leg.get("price", 0) or 0),
                        "qty": existing_qty,
                        "pnl_pct": pnl_pct,
                    }
                )
                continue

            old_order_id = str(existing_leg.get("orderId", ""))
            if old_order_id:
                cancel_ok, cancel_note = _cancel_protection_order_verified(symbol, direction, existing_leg)
                if not cancel_ok:
                    stale_tp_cleanup_failed = True
                    stale_tp_cancellations.append({
                        "order_id": old_order_id,
                        "leg": leg,
                        "status": "error",
                        "error": cancel_note,
                    })
                    tp_results.append({
                        "leg": leg,
                        "status": "error",
                        "error": f"stale TP cancel failed: {cancel_note}",
                        "qty": tp_qty,
                        "pnl_pct": pnl_pct,
                    })
                    continue

        if not current_price_checked:
            current_price = _current_close_price(symbol)
            current_price_checked = True

        if current_price is None:
            tp_results.append({"leg": leg, "status": "deferred", "reason": "current_price_unavailable", "price": tp_price, "qty": tp_qty, "pnl_pct": pnl_pct})
            continue

        trigger_invalid = (direction == "LONG" and tp_price <= current_price) or (direction == "SHORT" and tp_price >= current_price)

        if trigger_invalid:
            log.warning("[BINGX] TP market execution for %s %s: price=%s current=%s (trigger crossed)", symbol, leg, _format_price(tp_price, price_precision), _format_price(current_price, price_precision))
            client_order_id = build_tp_client_order_id(leg, trade_id)
            market_params = {
                "symbol": bx_symbol,
                "side": "SELL" if direction == "LONG" else "BUY",
                "positionSide": direction,
                "type": "MARKET",
                "quantity": _format_qty(tp_qty, precision),
            }

            resp = _post_protection_order_verified(symbol, direction, market_params, client_order_id, max_attempts=2, retry_delay=0.25)
            if resp.get("code") not in (0, "0"):
                log.error("[BINGX] TP market close failed: %s msg=%s", leg, resp.get("msg"))
                tp_results.append({"leg": leg, "status": "error", "error": f"code={resp.get('code')} msg={resp.get('msg')}", "qty": tp_qty, "pnl_pct": pnl_pct})
            else:
                order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
                tp_results.append({
                    "leg": leg,
                    "status": "created",
                    "order_id": str(order.get("orderId", "")),
                    "client_order_id": order.get("clientOrderId") or None,
                    "price": current_price,
                    "qty": tp_qty,
                    "pnl_pct": pnl_pct,
                })
            continue

        client_order_id = build_tp_client_order_id(leg, trade_id)
        params = {
            "symbol": bx_symbol,
            "side": "SELL" if direction == "LONG" else "BUY",
            "positionSide": direction,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": _format_price(tp_price, price_precision),
            "quantity": _format_qty(tp_qty, precision),
        }

        resp = _post_protection_order_verified(symbol, direction, params, client_order_id)
        if resp.get("code") not in (0, "0"):
            log.error("[BINGX] TP order failed: %s code=%s msg=%s", leg, resp.get("code"), resp.get("msg"))
            tp_results.append({"leg": leg, "status": "error", "error": f"code={resp.get('code')} msg={resp.get('msg')}", "qty": tp_qty, "pnl_pct": pnl_pct})
            continue

        order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
        tp_results.append({
            "leg": leg,
            "status": "created",
            "order_id": str(order.get("orderId", "")),
            "client_order_id": order.get("clientOrderId") or None,
            "price": tp_price,
            "qty": tp_qty,
            "pnl_pct": pnl_pct,
        })

    # Reconcile the complete desired TP set, not only its members. When a
    # position changes profile (e.g. two TPs -> single TP), any live conditional
    # order that was not consumed by the current desired set is stale and must
    # be cancelled and verified. This deliberately runs after desired orders
    # are healthy so there is no intentional protection gap.
    stale_tps = [
        order for order in existing_tp
        if str(order.get("orderId", "")) and str(order.get("orderId", "")) not in handled_existing_tp_ids
    ]
    for stale in stale_tps:
        order_id = str(stale.get("orderId", ""))
        ok, note = _cancel_protection_order_verified(symbol, direction, stale)
        stale_tp_cancellations.append({
            "order_id": order_id,
            "status": "cancelled" if ok else "error",
            "error": None if ok else note,
        })
        if not ok:
            stale_tp_cleanup_failed = True

    successful_tps = [t for t in tp_results if t.get("status") in {"created", "already_exists"}]
    if not verified_sl_valid or stale_tp_cleanup_failed:
        final_status = "PROTECTION_FAILED"
    elif len(successful_tps) == len(tp_levels_norm):
        final_status = "PROTECTED"
    else:
        final_status = "SL_ONLY"

    effective_levels = [
        {
            "leg": str(level["leg"]),
            "pnl_pct": float(level["pnl_pct"]),
            "close_fraction": float(qty / position_qty) if position_qty > 0 else 0.0,
            "qty": float(qty),
        }
        for level, qty in zip(tp_levels_norm, desired_qtys)
    ]
    effective_weighted_rr = _effective_weighted_rr(effective_levels, stop_loss_pct)

    return {
        "status": final_status,
        "symbol": symbol,
        "bx_symbol": bx_symbol,
        "direction": direction,
        "avg_price": avg_price,
        "qty": position_qty,
        "tp_mode": tp_mode,
        "effective_tp_levels": effective_levels,
        "effective_weighted_rr": effective_weighted_rr,
        "tp_orders": tp_results,
        "stale_tp_cancellations": stale_tp_cancellations,
        "sl_result": sl_result,
        "protection_audit": protection_audit,
    }


def _validate_sl_order_for_position(
    order: dict, direction: str, avg_price: float,
    *, expected_price: float | None = None, expected_qty: float | None = None,
) -> bool:
    order_type = str(order.get("type", "")).upper()
    if order_type not in {"STOP", "STOP_MARKET"}:
        return False

    try:
        sl_price = float(order.get("stopPrice", 0) or order.get("price", 0) or 0)
        qty = float(order.get("origQty", 0) or order.get("quantity", 0) or 0)
    except (TypeError, ValueError):
        return False

    if sl_price <= 0 or qty <= 0 or avg_price <= 0:
        return False

    direction = str(direction).upper()
    if direction == "LONG":
        protective = sl_price < avg_price
    elif direction == "SHORT":
        protective = sl_price > avg_price
    else:
        return False

    # A break-even stop at the exact entry price is a valid protection state,
    # but only when the caller explicitly expects that entry price. Do not
    # globally relax the directional check for arbitrary equal-price stops.
    if not protective and expected_price is not None:
        protective = _sl_price_matches(float(expected_price), avg_price) and _sl_price_matches(sl_price, avg_price)

    if not protective:
        return False
    if expected_price is not None and not _sl_price_matches(sl_price, float(expected_price)):
        return False
    if expected_qty is not None and not _qty_matches_position(qty, float(expected_qty)):
        return False
    return True


def _audit_exact_single_sl(
    protection_response: dict,
    *,
    direction: str,
    avg_price: float,
    expected_price: float | None,
    expected_qty: float,
) -> tuple[bool, dict[str, Any]]:
    """Require exactly one exchange-visible SL whose price and qty protect the residual."""
    response = protection_response if isinstance(protection_response, dict) else {}
    orders = response.get("sl_orders", [])
    orders = list(orders) if response.get("status") == "ok" and isinstance(orders, list) else []

    def _num(order: dict, *keys: str) -> float:
        for key in keys:
            try:
                value = float(order.get(key, 0) or 0)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                return value
        return 0.0

    valid = [
        order for order in orders
        if isinstance(order, dict) and _validate_sl_order_for_position(
            order, direction, avg_price,
            expected_price=expected_price, expected_qty=expected_qty,
        )
    ]
    passed = response.get("status") == "ok" and len(orders) == 1 and len(valid) == 1
    if response.get("status") != "ok":
        reason = "OPEN_ORDERS_UNVERIFIABLE"
    elif not orders:
        reason = "SL_MISSING"
    elif len(orders) > 1:
        reason = "SL_MULTIPLE_ORDERS"
    elif not valid:
        observed = orders[0] if isinstance(orders[0], dict) else {}
        observed_qty = _num(observed, "origQty", "quantity")
        observed_price = _num(observed, "stopPrice", "price")
        if expected_qty > 0 and not _qty_matches_position(observed_qty, expected_qty):
            reason = "SL_QTY_MISMATCH"
        elif expected_price is not None and expected_price > 0 and not _sl_price_matches(observed_price, expected_price):
            reason = "SL_PRICE_MISMATCH"
        elif str(observed.get("type", "")).upper() not in {"STOP", "STOP_MARKET"}:
            reason = "SL_ORDER_TYPE_INVALID"
        else:
            reason = "SL_NOT_PROTECTIVE_OR_INVALID"
    else:
        reason = "OK"
    audit = {
        "open_orders_status": response.get("status"),
        "open_orders_error": response.get("error"),
        "sl_count": len(orders),
        "valid_sl_count": len(valid),
        "sl_order_ids": [str(order.get("orderId") or order.get("order_id") or "") for order in orders if isinstance(order, dict)],
        "sl_order_qty": [_num(order, "origQty", "quantity") for order in orders if isinstance(order, dict)],
        "sl_order_prices": [_num(order, "stopPrice", "price") for order in orders if isinstance(order, dict)],
        "expected_position_qty": float(expected_qty),
        "expected_sl_price": float(expected_price) if expected_price is not None else None,
        "audit_reason": reason,
    }
    return bool(passed), audit
