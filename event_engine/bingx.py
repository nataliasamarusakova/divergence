from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import time
import uuid
from email.utils import parsedate_to_datetime
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR
from typing import Any
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger("event_engine.bingx")

API_KEY = os.environ.get("BINGX_API_KEY", "").strip()
SECRET_KEY = os.environ.get("BINGX_SECRET_KEY", "").strip()
BASE_URL = os.environ.get(
    "BINGX_BASE_URL",
    "https://open-api-vst.bingx.com",
).rstrip("/")

MARGIN_USDT = float(os.environ.get("BINGX_MARGIN_USDT", "1"))
LEVERAGE = int(os.environ.get("BINGX_LEVERAGE", "10"))
MAX_LEVERAGE = int(os.environ.get("BINGX_MAX_LEVERAGE", "50"))

SYMBOL_MAP = {}
try:
    import json
    SYMBOL_MAP = json.loads(os.environ.get("BINGX_SYMBOL_MAP", "{}"))
except Exception:
    SYMBOL_MAP = {}

CONTRACTS_PATH = "/openApi/swap/v2/quote/contracts"
KLINE_PATH = "/openApi/swap/v3/quote/klines"
ORDER_PATH = "/openApi/swap/v2/trade/order"
POSITION_PATH = os.environ.get("BINGX_POSITIONS_PATH", "/openApi/swap/v2/user/positions")
LEVERAGE_PATH = "/openApi/swap/v2/trade/leverage"
OPEN_ORDERS_PATH = "/openApi/swap/v2/trade/openOrders"

CACHE = {
    "ts": 0.0,
    "data": {},
    "by_display_name": {},
}

TTL = 3600

# Local clock correction derived from BingX HTTP Date header after timestamp errors.
SERVER_TIME_OFFSET_MS = 0

# Оптимизированная сессия с Keep-Alive
SESSION = requests.Session()
_adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=Retry(total=2, backoff_factor=0.3))
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)


def _sign(params: dict[str, Any]) -> str:
    qs = urlencode(params)
    return hmac.new(
        SECRET_KEY.encode(),
        qs.encode(),
        hashlib.sha256,
    ).hexdigest()


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
    log.warning("[BINGX_TIME_SYNC] server_offset_ms=%d", SERVER_TIME_OFFSET_MS)
    return True


def _request(
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    signed: bool = True,
):
    base_params = dict(params or {})
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
            response = SESSION.request(
                method=method,
                url=BASE_URL + path,
                params=request_params,
                headers=headers,
                timeout=10,
            )
            payload = response.json()
        except Exception as exc:
            return {"code": -1, "msg": str(exc)}

        try:
            code = int(payload.get("code"))
        except (TypeError, ValueError, AttributeError):
            code = None

        # 109400 is a signed-request timestamp error. Re-sync from the HTTP Date
        # header and retry exactly once. This is safe because the original request
        # was rejected with the timestamp error, not accepted for execution.
        if (
            signed
            and code == 109400
            and attempt < max_timestamp_retries
            and _update_server_time_offset(response)
        ):
            continue

        return payload

    return {"code": -1, "msg": "request retry exhausted"}


def refresh_contracts() -> dict[str, Any]:
    resp = _request("GET", CONTRACTS_PATH, signed=False)
    if resp.get("code") != 0:
        raise RuntimeError(f"BingX contracts error: {resp.get('msg')}")

    data = {}
    by_name = {}

    for c in resp.get("data", []) or []:
        sym = str(c.get("symbol", "")).strip().upper()
        name = str(c.get("displayName", "")).strip().upper()

        if sym:
            data[sym] = c
        if name:
            by_name[name] = c

    CACHE.update(
        ts=time.time(),
        data=data,
        by_display_name=by_name,
    )
    log.info("BingX contracts=%d", len(data))
    return data


def contracts() -> dict[str, dict]:
    if CACHE["data"] and time.time() - CACHE["ts"] < TTL:
        return CACHE["data"]
    try:
        return refresh_contracts()
    except Exception:
        return CACHE["data"]


def get_contract(symbol: str) -> dict | None:
    s = (symbol or "").strip().upper()
    if not s:
        return None

    mapped = SYMBOL_MAP.get(s)
    if mapped:
        c = contracts().get(str(mapped).strip().upper())
        if c:
            return c

    direct = s if s.endswith("-USDT") else f"{s.replace('-', '')}-USDT"
    c = contracts().get(direct)
    if c:
        return c

    base = s.replace("-USDT", "").replace("-", "")

    for c in CACHE["data"].values():
        cs = str(c.get("symbol", "")).upper()
        if cs == f"{base}-USDT" or cs == base:
            return c

    norm_base = base.replace("-", "").replace("/", "").replace(" ", "")
    for c in CACHE["data"].values():
        name = str(c.get("displayName", "")).upper().replace("-", "").replace("/", "").replace(" ", "")
        if name == f"{norm_base}USDT" or name == norm_base:
            return c

    return CACHE["by_display_name"].get(f"{base}-USDT")


def to_bx_symbol(symbol: str) -> str | None:
    c = get_contract(symbol)
    if not c:
        return None
    return str(c.get("symbol", "")).upper()


def contract_exists(symbol: str) -> bool:
    c = get_contract(symbol)
    return bool(
        c
        and c.get("status") == 1
        and str(c.get("apiStateOpen", "")).lower() == "true"
    )


def fetch_klines(
    symbol: str,
    interval: str,
    limit: int = 250,
) -> list[dict]:
    bx = to_bx_symbol(symbol)
    if not bx:
        raise ValueError(f"No BingX contract for {symbol}")

    resp = _request(
        "GET",
        KLINE_PATH,
        {"symbol": bx, "interval": interval, "limit": limit},
        signed=False,
    )

    code = resp.get("code")
    if code not in (0, "0"):
        raise RuntimeError(f"BingX klines error {bx}/{interval}: code={code} msg={resp.get('msg')}")

    rows = resp.get("data") or []
    out: list[dict] = []
    now_ms = int(time.time() * 1000)

    duration_ms = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "4h": 14_400_000,
        "6h": 21_600_000,
        "12h": 43_200_000,
        "1d": 86_400_000,
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

                close_time = (
                    int(row[6])
                    if len(row) >= 7 and row[6] is not None
                    else (open_time + duration_ms if duration_ms else open_time)
                )

                quote_volume = float(row[7]) if len(row) >= 8 and row[7] is not None else None
                taker_buy_base = float(row[9]) if len(row) >= 10 and row[9] is not None else None
                taker_buy_quote = float(row[10]) if len(row) >= 11 and row[10] is not None else None
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
                close_time = (
                    int(raw_close_time)
                    if raw_close_time is not None
                    else (open_time + duration_ms if duration_ms else open_time)
                )

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
            quote_volume is not None
            and taker_buy_base is not None
            and taker_buy_quote is not None
            and quote_volume >= 0
            and taker_buy_base >= 0
            and taker_buy_quote >= 0
            and taker_buy_base <= volume * 1.001 + 1e-8
            and taker_buy_quote <= quote_volume * 1.001 + 1e-8
        )

        bar_delta_usdt = (2.0 * taker_buy_quote - quote_volume) if taker_flow_valid else None

        out.append({
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
        })

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


def _set_leverage(bx_symbol: str, leverage: int) -> bool:
    for side in ("LONG", "BOTH"):
        resp = _request(
            "POST",
            LEVERAGE_PATH,
            {"symbol": bx_symbol, "side": side, "leverage": str(leverage)},
        )
        if resp.get("code") == 0:
            return True
    return False


def _normalize_orders_list(resp: dict) -> list[dict]:
    data = resp.get("data")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("orders", "positions", "order", "position"):
            value = data.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return [value]
    return []


def get_positions() -> list[dict]:
    resp = _request("GET", POSITION_PATH, {}, signed=True)
    if resp.get("code") != 0:
        return []
    return _normalize_orders_list(resp)


def get_order(symbol: str, order_id: str | int) -> dict:
    bx = to_bx_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "contract_not_found"}

    resp = _request(
        "GET",
        ORDER_PATH,
        {"symbol": bx, "orderId": str(order_id)},
        signed=True,
    )
    if resp.get("code") != 0:
        return {"status": "error", "error": resp.get("msg"), "code": resp.get("code")}

    data = resp.get("data") or {}
    order = data.get("order") or data
    return {
        "status": "ok",
        "order_id": str(order.get("orderId", order_id)),
        "order_status": str(order.get("status", "")).upper(),
        "avg_price": float(order.get("avgPrice", 0) or order.get("price", 0) or order.get("stopPrice", 0) or 0),
        "executed_qty": float(order.get("executedQty", 0) or order.get("cumQty", 0) or 0),
        "orig_qty": float(order.get("origQty", 0) or order.get("quantity", 0) or 0),
    }


def cancel_order(symbol: str, order_id: str | int) -> dict:
    bx = to_bx_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "contract_not_found"}

    return _request(
        "DELETE",
        ORDER_PATH,
        {"symbol": bx, "orderId": str(order_id)},
        signed=True,
    )


def has_open_position(symbol: str, direction: str) -> bool:
    bx = to_bx_symbol(symbol)
    if not bx:
        return False

    want = "LONG" if direction.upper() == "LONG" else "SHORT"

    for p in get_positions():
        if str(p.get("symbol", "")).upper() != bx:
            continue

        side = str(p.get("positionSide", p.get("positionAmt", ""))).upper()
        try:
            amt = float(p.get("positionAmt", 0) or 0)
        except Exception:
            amt = 0

        if amt != 0 and (want in side or (want == "LONG" and amt > 0) or (want == "SHORT" and amt < 0)):
            return True

    return False


def _new_open_client_order_id(bx_symbol: str, trade_id: str) -> str:
    digest = hashlib.sha256(f"{bx_symbol}:{trade_id}".encode()).hexdigest().upper()[:24]
    return f"EVT_OPEN_{digest}"


def open_market(
    symbol: str,
    direction: str,
    price: float,
    trade_id: str,
) -> dict:
    direction = str(direction).upper()
    if direction not in ("LONG", "SHORT"):
        return {"status": "error", "error": f"invalid direction={direction}"}

    bx = to_bx_symbol(symbol)
    if not bx:
        return {"status": "error", "error": "contract_not_found"}

    c = get_contract(symbol) or {}
    if not contract_exists(symbol):
        return {"status": "error", "error": "contract_unavailable", "symbol": bx}

    if has_open_position(symbol, direction):
        return {"status": "existing_position", "symbol": bx, "direction": direction}

    try:
        prec = int(c.get("quantityPrecision") or 0)
        min_qty = float(c.get("tradeMinQuantity") or c.get("minQty") or 0)
        mult = float(c.get("multiplier") or 1)
        max_lev = int(c.get("maxLongLeverage") or c.get("maxLeverage") or MAX_LEVERAGE)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": f"invalid contract parameters: {exc}", "symbol": bx}

    leverage = min(LEVERAGE, max_lev)
    qty = (MARGIN_USDT * leverage) / max(price * mult, 1e-12)
    q = Decimal(str(qty)).quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN) if prec >= 0 else Decimal(str(qty))
    qty = float(q)

    if qty < min_qty:
        need = math.ceil((min_qty * price * mult) / max(MARGIN_USDT, 1e-9))
        leverage = min(max(need, leverage), max_lev)
        qty = (MARGIN_USDT * leverage) / max(price * mult, 1e-12)
        q = Decimal(str(qty)).quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        qty = float(q)

    if qty <= 0 or qty < min_qty:
        return {"status": "error", "error": f"qty={qty} < min_qty={min_qty}", "symbol": bx, "qty": qty, "min_qty": min_qty}

    if not _set_leverage(bx, leverage):
        return {
            "status": "error",
            "error": f"failed to set leverage={leverage}",
            "symbol": bx,
            "leverage": leverage,
        }

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

    response = _request("POST", ORDER_PATH, params)
    if response.get("code") != 0:
        error = str(response.get("msg", ""))
        return {
            "status": "error",
            "error": error,
            "symbol": bx,
            "clientOrderId": client_order_id,
            "response": response,
        }

    data = response.get("data") or {}
    order = data.get("order") or {}
    order_id = order.get("orderId") or data.get("orderId")

    return {
        "status": "opened",
        "symbol": bx,
        "qty": qty,
        "leverage": leverage,
        "order_id": order_id,
        "client_order_id": order.get("clientOrderId") or client_order_id,
        "response": response,
    }


def get_position_directional(symbol: str, direction: str) -> dict:
    bx_symbol = to_bx_symbol(symbol)
    direction = str(direction).upper()

    resp = _request("GET", POSITION_PATH, {"symbol": bx_symbol})
    if resp.get("code") != 0:
        return {"status": "error", "error": f"get_position failed: code={resp.get('code')} msg={resp.get('msg')}", "symbol": bx_symbol}

    for p in _normalize_orders_list(resp):
        position_side = str(p.get("positionSide", "")).upper()
        if position_side not in (direction, "BOTH"):
            continue

        try:
            qty = abs(float(p.get("positionAmt", 0) or 0))
            avg_price = float(p.get("avgPrice", 0) or p.get("entryPrice", 0) or 0)
        except (TypeError, ValueError):
            continue

        if qty <= 0 or avg_price <= 0:
            continue

        return {
            "status": "found",
            "symbol": p.get("symbol", bx_symbol),
            "positionSide": direction,
            "avgPrice": avg_price,
            "positionAmt": qty,
            "entryPrice": float(p.get("entryPrice", 0) or avg_price),
        }

    return {"status": "not_found", "symbol": bx_symbol, "positionSide": direction}


def wait_for_position_fill_directional(
    symbol: str,
    direction: str,
    timeout_sec: int = 30,
    poll_interval: float = 1.0,
) -> dict:
    started = time.time()
    while time.time() - started < timeout_sec:
        pos = get_position_directional(symbol, direction)
        if pos.get("status") == "found":
            return pos
        time.sleep(poll_interval)

    return {"status": "timeout", "symbol": to_bx_symbol(symbol), "positionSide": direction}


def get_open_protection_directional(symbol: str, direction: str) -> dict:
    bx_symbol = to_bx_symbol(symbol)
    direction = str(direction).upper()

    resp = _request("GET", OPEN_ORDERS_PATH, {"symbol": bx_symbol})
    if resp.get("code") != 0:
        return {
            "status": "error",
            "error": f"openOrders failed: code={resp.get('code')} msg={resp.get('msg')}",
            "tp_orders": [],
            "sl_orders": [],
        }

    tp_orders = []
    sl_orders = []

    for order in _normalize_orders_list(resp):
        position_side = str(order.get("positionSide", "")).upper()
        if position_side not in (direction, "BOTH"):
            continue

        order_type = str(order.get("type", "")).upper()
        if order_type in ("TAKE_PROFIT", "TAKE_PROFIT_MARKET"):
            tp_orders.append(order)
        elif order_type in ("STOP", "STOP_MARKET"):
            sl_orders.append(order)

    return {
        "status": "ok",
        "symbol": bx_symbol,
        "positionSide": direction,
        "tp_orders": tp_orders,
        "sl_orders": sl_orders,
    }


def _round_qty(qty: float, precision: int) -> float:
    if precision < 0:
        return float(qty)
    return float(
        Decimal(str(qty)).quantize(
            Decimal(1).scaleb(-precision),
            rounding=ROUND_DOWN,
        )
    )


def _format_qty(qty: float, precision: int) -> str:
    return f"{qty:.{precision}f}"


def _format_price(price: float, precision: int) -> str:
    return f"{price:.{precision}f}"


def build_tp_client_order_id(leg: str, trade_id: str | None = None) -> str:
    token = uuid.uuid4().hex.upper()[:16]
    return f"EVT_{str(leg).upper()}_{token}"


def build_sl_client_order_id(trade_id: str | None = None) -> str:
    token = uuid.uuid4().hex.upper()[:16]
    return f"EVT_SL_{token}"


def _ceil_to_step(value: Decimal, step: Decimal) -> Decimal:
    """Round quantity upward to the exchange quantity step."""
    if step <= 0:
        raise ValueError("quantity step must be > 0")
    units = (value / step).to_integral_value(rounding=ROUND_CEILING)
    return units * step


def _allocate_tp_quantities(
    position_qty: float,
    precision: int,
    min_qty: float,
    fractions: list[float],
) -> list[float]:
    """Allocate TP quantities so every leg is valid for exchange minQty/precision.

    The function preserves the total quantity exactly in Decimal arithmetic after
    step rounding. It raises ValueError when the requested number of legs cannot
    be represented without violating minQty or exhausting the whole position.
    """
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
        raise ValueError(
            f"position_qty={position_qty} cannot support {k} TP legs "
            f"with min_leg={min_leg}"
        )

    total_fraction = sum(Decimal(str(f)) for f in fractions)
    if total_fraction <= 0:
        raise ValueError("fraction sum must be > 0")
    normalized = [Decimal(str(f)) / total_fraction for f in fractions]

    # Start every leg at the minimum valid quantity.
    quantities = [min_leg] * k
    remaining = pos - min_leg * k

    # Allocate remaining quantity by largest-remainder style using the requested
    # fractions, while keeping every result on the exchange step.
    raw_extra = [remaining * f for f in normalized]
    extra_steps = []
    for raw in raw_extra:
        n = (raw / step).to_integral_value(rounding=ROUND_FLOOR)
        extra_steps.append(n)

    allocated_extra = sum(extra_steps) * step
    quantities = [q + n * step for q, n in zip(quantities, extra_steps)]
    remainder = pos - sum(quantities)
    while remainder >= step:
        # Give the next step to the leg with the largest unmet fractional target.
        best = max(
            range(k),
            key=lambda i: raw_extra[i] - (quantities[i] - min_leg),
        )
        quantities[best] += step
        remainder -= step

    # Decimal representation can differ by tiny residue; enforce exact total in
    # the last leg while keeping it on-step.
    quantities[-1] += remainder
    quantities[-1] = quantities[-1].quantize(step)

    if any(q < min_leg for q in quantities):
        raise ValueError("TP allocation produced a leg below minimum quantity")
    if sum(quantities) != pos:
        raise ValueError(f"TP allocation mismatch: sum={sum(quantities)} position={pos}")

    return [float(q) for q in quantities]


def _normalize_tp_levels(tp_levels: list) -> list[dict]:
    normalized = []
    for tp in tp_levels or []:
        leg = str(tp.get("leg", f"tp{len(normalized) + 1}"))
        pnl_pct = float(tp.get("pnl_pct", 0))
        fraction = float(tp.get("close_fraction", 0))
        if not math.isfinite(pnl_pct) or not math.isfinite(fraction):
            continue
        if pnl_pct <= 0 or fraction <= 0:
            continue
        normalized.append({"leg": leg, "pnl_pct": pnl_pct, "close_fraction": fraction})
    if not normalized:
        normalized = [{"leg": "tp1", "pnl_pct": 2.0, "close_fraction": 1.0}]
    total = sum(x["close_fraction"] for x in normalized)
    for x in normalized:
        x["close_fraction"] /= total
    return normalized


def _tp_leg_from_order(order: dict, expected_leg: str, expected_price: float, price_precision: int) -> bool:
    expected_leg = str(expected_leg).upper()
    client_id = str(order.get("clientOrderId", "")).upper()
    if expected_leg and expected_leg in client_id:
        return True

    order_type = str(order.get("type", "")).upper()
    if order_type not in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}:
        return False

    actual_price = float(order.get("stopPrice", 0) or order.get("price", 0) or 0)
    if actual_price <= 0:
        return False
    return _format_price(actual_price, price_precision) == _format_price(expected_price, price_precision)


def _current_close_price(symbol: str) -> float | None:
    try:
        rows = fetch_klines(symbol, "1m", limit=2)
    except Exception as exc:
        log.warning("[BINGX_TP_PRICE_CHECK] failed to read current price for %s: %s", symbol, exc)
        return None
    if not rows:
        return None
    try:
        price = float(rows[-1].get("close", 0) or 0)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price > 0 else None


def ensure_directional_protection(
    symbol: str,
    direction: str,
    avg_price: float,
    qty: float,
    stop_loss_pct: float,
    tp_levels: list,
    trade_id: str | None = None,
) -> dict:
    """Install exchange-side SL first, verify it, then install valid TP legs."""
    direction = str(direction).upper()
    if direction not in ("LONG", "SHORT"):
        return {"status": "error", "error": f"invalid direction={direction}"}

    try:
        avg_price = float(avg_price)
        qty = abs(float(qty))
        stop_loss_pct = float(stop_loss_pct)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc)}

    if (
        not math.isfinite(avg_price)
        or not math.isfinite(qty)
        or not math.isfinite(stop_loss_pct)
        or avg_price <= 0
        or qty <= 0
        or not (0 < stop_loss_pct <= 25)
    ):
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
        return {"status": "PROTECTION_FAILED", "error": existing.get("error", "openOrders unavailable")}

    existing_tp = list(existing.get("tp_orders", []))
    existing_sl = list(existing.get("sl_orders", []))

    tp_levels_norm = _normalize_tp_levels(tp_levels)

    # SAFETY INVARIANT: an exchange-side SL must exist before any TP is created.
    if existing_sl:
        sl = existing_sl[0]
        sl_result = {
            "status": "already_exists",
            "order_id": str(sl.get("orderId", "")),
            "stop_price": float(sl.get("stopPrice", 0) or sl.get("price", 0) or 0),
            "qty": float(sl.get("origQty", 0) or sl.get("quantity", 0) or position_qty),
        }
    else:
        sl_price = (
            avg_price * (1.0 - stop_loss_pct / 100.0)
            if direction == "LONG"
            else avg_price * (1.0 + stop_loss_pct / 100.0)
        )
        client_order_id = build_sl_client_order_id(trade_id)
        params = {
            "symbol": bx_symbol,
            "side": "SELL" if direction == "LONG" else "BUY",
            "positionSide": direction,
            "type": "STOP_MARKET",
            "stopPrice": _format_price(sl_price, price_precision),
            "quantity": _format_qty(position_qty, precision),
            "clientOrderId": client_order_id,
        }
        resp = _request("POST", ORDER_PATH, params)
        if resp.get("code") != 0:
            log.error("[BINGX_SL_FAILED] code=%s msg=%s", resp.get("code"), resp.get("msg"))
            return {
                "status": "PROTECTION_FAILED",
                "error": f"SL failed: code={resp.get('code')} msg={resp.get('msg')}",
                "sl_result": {"status": "error", "error": f"SL failed: code={resp.get('code')} msg={resp.get('msg')}"},
                "tp_orders": [],
            }
        order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
        sl_result = {
            "status": "created",
            "order_id": str(order.get("orderId", "")),
            "client_order_id": order.get("clientOrderId") or client_order_id,
            "stop_price": sl_price,
            "qty": position_qty,
        }

    # Re-read exchange open orders after SL creation. Never continue to TP if the
    # exchange does not show an SL.
    verified = get_open_protection_directional(symbol, direction)
    verified_sl = list(verified.get("sl_orders", [])) if verified.get("status") == "ok" else []
    if not verified_sl:
        return {
            "status": "SL_UNVERIFIED",
            "symbol": symbol,
            "bx_symbol": bx_symbol,
            "direction": direction,
            "avg_price": avg_price,
            "qty": position_qty,
            "sl_result": sl_result,
            "tp_orders": [],
            "error": "SL was created/requested but is not visible on exchange",
        }

    # If current position is too small for the configured number of legs, use one
    # valid full-size TP rather than sending invalid minQty orders.
    if min_qty > 0 and position_qty < min_qty * len(tp_levels_norm):
        tp_levels_norm = [{
            "leg": tp_levels_norm[-1]["leg"],
            "pnl_pct": tp_levels_norm[-1]["pnl_pct"],
            "close_fraction": 1.0,
        }]

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
        }

    tp_results = []
    current_price = None
    current_price_checked = False
    for level, tp_qty in zip(tp_levels_norm, desired_qtys):
        leg = level["leg"]
        pnl_pct = level["pnl_pct"]

        tp_price = (
            avg_price * (1.0 + pnl_pct / 100.0)
            if direction == "LONG"
            else avg_price * (1.0 - pnl_pct / 100.0)
        )

        existing_leg = None
        for order in existing_tp:
            if _tp_leg_from_order(order, leg, tp_price, price_precision):
                existing_leg = order
                break

        if existing_leg:
            existing_qty = float(existing_leg.get("origQty", 0) or existing_leg.get("quantity", 0) or 0)
            if existing_qty > 0 and min_qty > 0 and existing_qty < min_qty:
                return {
                    "status": "PROTECTION_FAILED",
                    "error": f"existing {leg} qty={existing_qty} < minQty={min_qty}",
                    "sl_result": sl_result,
                    "tp_orders": tp_results,
                }
            tp_results.append({
                "leg": leg,
                "status": "already_exists",
                "order_id": str(existing_leg.get("orderId", "")),
                "price": float(existing_leg.get("stopPrice", 0) or existing_leg.get("price", 0) or 0),
                "qty": existing_qty,
            })
            continue

        # BingX rejects a TAKE_PROFIT trigger that is already on the wrong side
        # of the current market. Do not repeatedly POST the same impossible order.
        if not current_price_checked:
            current_price = _current_close_price(symbol)
            current_price_checked = True
        if current_price is None:
            tp_results.append({
                "leg": leg,
                "status": "deferred",
                "reason": "current_price_unavailable",
                "price": tp_price,
                "qty": tp_qty,
                "pnl_pct": pnl_pct,
            })
            continue

        trigger_invalid = (
            direction == "LONG" and tp_price <= current_price
        ) or (
            direction == "SHORT" and tp_price >= current_price
        )
        if trigger_invalid:
            log.warning(
                "[BINGX_TP_DEFERRED] %s %s price=%s current=%s: trigger already crossed",
                symbol,
                leg,
                _format_price(tp_price, price_precision),
                _format_price(current_price, price_precision),
            )
            tp_results.append({
                "leg": leg,
                "status": "deferred",
                "reason": "trigger_already_crossed",
                "price": tp_price,
                "current_price": current_price,
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
            "clientOrderId": client_order_id,
        }

        resp = _request("POST", ORDER_PATH, params)
        if resp.get("code") != 0:
            log.error("[BINGX_TP_FAILED] %s code=%s msg=%s", leg, resp.get("code"), resp.get("msg"))
            # Do not remove the verified SL. Return a partial protection status so
            # the reconciliation loop can safely retry missing TP legs later.
            tp_results.append({
                "leg": leg,
                "status": "error",
                "error": f"code={resp.get('code')} msg={resp.get('msg')}",
                "qty": tp_qty,
            })
            continue

        order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
        tp_results.append({
            "leg": leg,
            "status": "created",
            "order_id": str(order.get("orderId", "")),
            "client_order_id": order.get("clientOrderId") or client_order_id,
            "price": tp_price,
            "qty": tp_qty,
            "pnl_pct": pnl_pct,
        })

    sl_ok = bool(verified_sl) or sl_result.get("status") in {"created", "already_exists"}
    successful_tps = [t for t in tp_results if t.get("status") in {"created", "already_exists"}]
    if not sl_ok:
        final_status = "PROTECTION_FAILED"
    elif len(successful_tps) == len(tp_levels_norm):
        final_status = "PROTECTED"
    elif successful_tps:
        final_status = "SL_ONLY"
    else:
        final_status = "SL_ONLY"

    return {
        "status": final_status,
        "symbol": symbol,
        "bx_symbol": bx_symbol,
        "direction": direction,
        "avg_price": avg_price,
        "qty": position_qty,
        "tp_orders": tp_results,
        "sl_result": sl_result,
    }
