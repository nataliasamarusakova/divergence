from __future__ import annotations

import hashlib
import hmac
import logging
import math
import os
import time
from decimal import Decimal, ROUND_DOWN
from typing import Any
from urllib.parse import urlencode

import requests

log = logging.getLogger("event_engine.bingx")
API_KEY = os.environ.get("BINGX_API_KEY", "").strip()
SECRET_KEY = os.environ.get("BINGX_SECRET_KEY", "").strip()
BASE_URL = os.environ.get("BINGX_BASE_URL", "https://open-api-vst.bingx.com").rstrip("/")
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

CACHE = {"ts": 0.0, "data": {}, "by_display_name": {}}
TTL = 3600


def _sign(params: dict[str, Any]) -> str:
    qs = urlencode(params)
    return hmac.new(SECRET_KEY.encode(), qs.encode(), hashlib.sha256).hexdigest()


def _request(method: str, path: str, params: dict[str, Any] | None = None, signed: bool = True):
    params = dict(params or {})
    headers = {}
    if signed:
        if not API_KEY or not SECRET_KEY:
            return {"code": -1, "msg": "missing BingX credentials"}
        params["timestamp"] = str(int(time.time() * 1000))
        params["signature"] = _sign(params)
        headers["X-BX-APIKEY"] = API_KEY
    try:
        r = requests.request("GET" if method == "GET" else method, BASE_URL + path, params=params, headers=headers, timeout=15)
        return r.json()
    except Exception as exc:
        return {"code": -1, "msg": str(exc)}


def refresh_contracts() -> dict[str, Any]:
    resp = _request("GET", CONTRACTS_PATH, signed=False)
    if resp.get("code") != 0:
        raise RuntimeError(f"BingX contracts error: {resp.get('msg')}")
    data, by_name = {}, {}
    for c in resp.get("data", []) or []:
        sym = str(c.get("symbol", "")).strip().upper()
        name = str(c.get("displayName", "")).strip().upper()
        if sym: data[sym] = c
        if name: by_name[name] = c
    CACHE.update(ts=time.time(), data=data, by_display_name=by_name)
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
        if c: return c
    direct = s if s.endswith("-USDT") else f"{s.replace('-', '')}-USDT"
    c = contracts().get(direct)
    if c: return c
    base = s.replace("-USDT", "").replace("-", "")
    for c in CACHE["data"].values():
        name = str(c.get("displayName", "")).upper().replace("-", "")
        cs = str(c.get("symbol", "")).upper()
        if name == f"{base}-USDT" or name == base or cs.endswith(f"{base}-USDT"):
            return c
    return CACHE["by_display_name"].get(f"{base}-USDT")


def to_bx_symbol(symbol: str) -> str | None:
    c = get_contract(symbol)
    if not c:
        return None
    return str(c.get("symbol", "")).upper()


def contract_exists(symbol: str) -> bool:
    c = get_contract(symbol)
    return bool(c and c.get("status") == 1 and str(c.get("apiStateOpen", "")).lower() == "true")


def classify_contract(contract: dict | None) -> str:
    if not contract:
        return "unknown"
    s = str(contract.get("symbol", "")).upper()
    if s.startswith(("NCSK", "NCSI")): return "equity"
    if s.startswith("NCCO"): return "commodity"
    if s.startswith("NCFX"): return "forex"
    return "crypto"


def fetch_klines(symbol: str, interval: str, limit: int = 250) -> list[dict]:
    bx = to_bx_symbol(symbol)

    if not bx:
        raise ValueError(f"No BingX contract for {symbol}")

    resp = _request(
        "GET",
        KLINE_PATH,
        {
            "symbol": bx,
            "interval": interval,
            "limit": limit,
        },
        signed=False,
    )

    code = resp.get("code")

    if code not in (0, "0"):
        raise RuntimeError(
            f"BingX klines error "
            f"{bx}/{interval}: "
            f"code={code} "
            f"msg={resp.get('msg')}"
        )

    rows = resp.get("data") or []
    if rows and isinstance(rows[0], dict):
        print(
            "[BINGX_KLINE_FIELDS]",
            bx,
            sorted(rows[0].keys())
        )
    if not isinstance(rows, list):
        raise RuntimeError(
            f"Unexpected BingX klines payload "
            f"{bx}/{interval}: "
            f"data_type={type(rows).__name__}"
        )

    first_shape = "empty"

    if rows:
        if isinstance(rows[0], (list, tuple)):
            first_shape = f"array_len={len(rows[0])}"
        elif isinstance(rows[0], dict):
            first_shape = "dict"
        else:
            first_shape = type(rows[0]).__name__

    print(
        f"[BINGX_KLINES_RAW] "
        f"symbol={bx} "
        f"interval={interval} "
        f"rows={len(rows)} "
        f"first_shape={first_shape}"
    )

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

        # ------------------------------------------------------------
        # FORMAT 1: ARRAY
        #
        # [open_time, open, high, low, close, volume, close_time, ...]
        # ------------------------------------------------------------
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

                if len(row) >= 7 and row[6] is not None:
                    close_time = int(row[6])
                else:
                    close_time = (
                        open_time + duration_ms
                        if duration_ms
                        else open_time
                    )

                quote_volume = None
                taker_buy_base = None
                taker_buy_quote = None

                if len(row) >= 8 and row[7] is not None:
                    try:
                        quote_volume = float(row[7])
                    except (TypeError, ValueError):
                        quote_volume = None

                if len(row) >= 10 and row[9] is not None:
                    try:
                        taker_buy_base = float(row[9])
                    except (TypeError, ValueError):
                        taker_buy_base = None

                if len(row) >= 11 and row[10] is not None:
                    try:
                        taker_buy_quote = float(row[10])
                    except (TypeError, ValueError):
                        taker_buy_quote = None

            except (TypeError, ValueError, IndexError):
                continue

        # ------------------------------------------------------------
        # FORMAT 2: DICT / OBJECT
        # ------------------------------------------------------------
        elif isinstance(row, dict):

            def pick(*names):
                for name in names:
                    if name in row and row[name] is not None:
                        return row[name]
                return None

            try:
                open_time = int(
                    pick(
                        "openTime",
                        "open_time",
                        "time",
                    )
                )

                open_price = float(
                    pick("open")
                )

                high = float(
                    pick("high")
                )

                low = float(
                    pick("low")
                )

                close = float(
                    pick("close")
                )

                volume = float(
                    pick("volume")
                )

                raw_close_time = pick(
                    "closeTime",
                    "close_time",
                )

                close_time = (
                    int(raw_close_time)
                    if raw_close_time is not None
                    else (
                        open_time + duration_ms
                        if duration_ms
                        else open_time
                    )
                )

                quote_volume_raw = pick(
                    "quoteAssetVolume",
                    "quoteVolume",
                    "quote_volume",
                )

                taker_base_raw = pick(
                    "takerBuyBaseVolume",
                    "taker_buy_base",
                    "takerBuyBase",
                    "takerBuyBaseVolume",
                    "buyVolume",
                )
                
                taker_quote_raw = pick(
                    "takerBuyQuoteVolume",
                    "taker_buy_quote",
                    "takerBuyQuote",
                    "takerBuyQuoteVolume",
                    "buyQuoteVolume",
                )

                quote_volume = (
                    float(quote_volume_raw)
                    if quote_volume_raw is not None
                    else None
                )

                taker_buy_base = (
                    float(taker_base_raw)
                    if taker_base_raw is not None
                    else None
                )

                taker_buy_quote = (
                    float(taker_quote_raw)
                    if taker_quote_raw is not None
                    else None
                )

            except (TypeError, ValueError, KeyError):
                continue

        else:
            continue

        # ------------------------------------------------------------
        # CLOSED CANDLE GUARD
        # ------------------------------------------------------------
        if close_time > now_ms:
            continue

        # ------------------------------------------------------------
        # OHLCV VALIDATION
        # ------------------------------------------------------------
        if (
            open_price <= 0
            or high <= 0
            or low <= 0
            or close <= 0
            or volume < 0
        ):
            continue

        if (
            high < low
            or high < open_price
            or high < close
            or low > open_price
            or low > close
        ):
            continue

        # ------------------------------------------------------------
        # TAKER FLOW VALIDATION
        #
        # Missing taker data != zero.
        # ------------------------------------------------------------
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

        if taker_flow_valid:
            bar_delta_usdt = (
                2.0 * taker_buy_quote
                - quote_volume
            )
        else:
            bar_delta_usdt = None

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

    # ------------------------------------------------------------
    # SORT + DEDUP
    # ------------------------------------------------------------
    out.sort(
        key=lambda x: x["close_time"]
    )

    deduped = []

    seen_close_times = set()

    for bar in out:
        ct = bar["close_time"]

        if ct in seen_close_times:
            continue

        seen_close_times.add(ct)
        deduped.append(bar)

    out = deduped

    print(
        f"[BINGX_KLINES_PARSED] "
        f"symbol={bx} "
        f"interval={interval} "
        f"bars={len(out)} "
        f"taker_valid="
        f"{sum(1 for x in out if x['taker_flow_valid'])}"
    )

    return out


def _set_leverage(bx_symbol: str, leverage: int) -> bool:
    for side in ("LONG", "BOTH"):
        resp = _request("POST", LEVERAGE_PATH, {"symbol": bx_symbol, "side": side, "leverage": str(leverage)})
        if resp.get("code") == 0: return True
    return False


def get_positions() -> list[dict]:
    resp = _request("GET", POSITION_PATH, {}, signed=True)
    if resp.get("code") != 0: return []
    data = resp.get("data") or []
    return data if isinstance(data, list) else []


def has_open_position(symbol: str, direction: str) -> bool:
    bx = to_bx_symbol(symbol)
    if not bx: return False
    want = "LONG" if direction.upper() == "LONG" else "SHORT"
    for p in get_positions():
        if str(p.get("symbol", "")).upper() != bx: continue
        side = str(p.get("positionSide", p.get("positionAmt", ""))).upper()
        try: amt = float(p.get("positionAmt", p.get("positionAmt", 0)) or 0)
        except Exception: amt = 0
        if amt != 0 and (want in side or (want == "LONG" and amt > 0) or (want == "SHORT" and amt < 0)):
            return True
    return False


def open_market(symbol: str, direction: str, price: float, trade_id: str) -> dict:
    bx = to_bx_symbol(symbol)
    if not bx: return {"status": "error", "error": "contract_not_found"}
    c = get_contract(symbol) or {}
    if not contract_exists(symbol): return {"status": "error", "error": "contract_unavailable", "symbol": bx}
    if has_open_position(symbol, direction): return {"status": "existing_position", "symbol": bx}
    prec = int(c.get("quantityPrecision") or 0)
    min_qty = float(c.get("tradeMinQuantity") or c.get("minQty") or 0)
    mult = float(c.get("multiplier") or 1)
    max_lev = int(c.get("maxLongLeverage") or c.get("maxLeverage") or MAX_LEVERAGE)
    leverage = min(LEVERAGE, max_lev)
    qty = (MARGIN_USDT * leverage) / max(price * mult, 1e-12)
    q = (Decimal(str(qty)).quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN) if prec >= 0 else Decimal(str(qty)))
    qty = float(q)
    if qty < min_qty:
        need = math.ceil((min_qty * price * mult) / max(MARGIN_USDT, 1e-9))
        leverage = min(max(need, leverage), max_lev)
        qty = (MARGIN_USDT * leverage) / max(price * mult, 1e-12)
        q = Decimal(str(qty)).quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        qty = float(q)
    if qty <= 0 or qty < min_qty: return {"status": "error", "error": f"qty={qty} < min_qty={min_qty}"}
    _set_leverage(bx, leverage)
    side = "BUY" if direction.upper() == "LONG" else "SELL"
    params = {"symbol": bx, "side": side, "positionSide": direction.upper(), "type": "MARKET", "quantity": f"{qty:.{prec}f}", "clientOrderId": f"EVT_OPEN_{trade_id[:24]}"}
    resp = _request("POST", ORDER_PATH, params)
    if resp.get("code") != 0: return {"status": "error", "error": str(resp.get("msg")), "symbol": bx}
    return {"status": "opened", "symbol": bx, "qty": qty, "leverage": leverage, "response": resp}


# ============================================================
# EVENT-DRIVEN DIRECTIONAL PROTECTION
# ============================================================

def get_position_directional(symbol: str, direction: str) -> dict:
    """
    Получает реальную позицию BingX именно нужного направления.

    LONG  -> positionSide LONG
    SHORT -> positionSide SHORT

    НИЧЕГО не открывает и не изменяет.
    """
    bx_symbol = to_bx_symbol(symbol)
    direction = str(direction).upper()

    if direction not in ("LONG", "SHORT"):
        return {
            "status": "error",
            "error": f"invalid direction={direction}",
            "symbol": bx_symbol,
        }

    resp = _request(
        "GET",
        POSITION_PATH,
        {"symbol": bx_symbol},
    )

    if resp.get("code") != 0:
        return {
            "status": "error",
            "error": (
                f"get_position failed: "
                f"code={resp.get('code')} "
                f"msg={resp.get('msg')}"
            ),
            "symbol": bx_symbol,
        }

    for p in _normalize_orders_list(resp):
        position_side = str(
            p.get("positionSide", "")
        ).upper()

        if position_side != direction:
            continue

        try:
            qty = abs(
                float(
                    p.get("positionAmt", 0) or 0
                )
            )

            avg_price = float(
                p.get("avgPrice", 0)
                or p.get("entryPrice", 0)
                or 0
            )
        except (TypeError, ValueError):
            continue

        if qty <= 0 or avg_price <= 0:
            continue

        return {
            "status": "found",
            "symbol": p.get(
                "symbol",
                bx_symbol,
            ),
            "positionSide": direction,
            "avgPrice": avg_price,
            "positionAmt": qty,
            "entryPrice": float(
                p.get("entryPrice", 0)
                or avg_price
            ),
        }

    return {
        "status": "not_found",
        "symbol": bx_symbol,
        "positionSide": direction,
    }


def wait_for_position_fill_directional(
    symbol: str,
    direction: str,
    timeout_sec: int = 30,
    poll_interval: float = 1.0,
) -> dict:
    """
    После market entry ждём именно реальную позицию
    нужного направления.
    """
    import time

    started = time.time()

    while (
        time.time() - started
        < timeout_sec
    ):
        pos = get_position_directional(
            symbol,
            direction,
        )

        if pos.get("status") == "found":
            log.info(
                f"[{symbol}] directional position confirmed: "
                f"side={direction} "
                f"avgPrice={pos.get('avgPrice')} "
                f"qty={pos.get('positionAmt')}"
            )
            return pos

        time.sleep(
            poll_interval
        )

    return {
        "status": "timeout",
        "symbol": to_bx_symbol(symbol),
        "positionSide": direction,
    }


def get_open_protection_directional(
    symbol: str,
    direction: str,
) -> dict:
    """
    Читает существующие TP/SL для нужного positionSide.
    """
    bx_symbol = to_bx_symbol(symbol)
    direction = str(direction).upper()

    resp = _request(
        "GET",
        "/openApi/swap/v2/trade/openOrders",
        {"symbol": bx_symbol},
    )

    if resp.get("code") != 0:
        return {
            "status": "error",
            "error": (
                f"openOrders failed: "
                f"code={resp.get('code')} "
                f"msg={resp.get('msg')}"
            ),
            "tp_orders": [],
            "sl_orders": [],
        }

    tp_orders = []
    sl_orders = []

    for order in _normalize_orders_list(resp):

        position_side = str(
            order.get("positionSide", "")
        ).upper()

        if position_side not in (
            direction,
            "BOTH",
        ):
            continue

        order_type = str(
            order.get("type", "")
        ).upper()

        if order_type in (
            "TAKE_PROFIT",
            "TAKE_PROFIT_MARKET",
        ):
            tp_orders.append(order)

        elif order_type in (
            "STOP",
            "STOP_MARKET",
        ):
            sl_orders.append(order)

    return {
        "status": "ok",
        "symbol": bx_symbol,
        "positionSide": direction,
        "tp_orders": tp_orders,
        "sl_orders": sl_orders,
    }


def _directional_qty(
    qty: float,
    precision: int,
) -> float:
    return _round_qty(
        float(qty),
        int(precision),
    )


def _directional_protection_prices(
    avg_price: float,
    direction: str,
    stop_loss_pct: float,
    tp1_pct: float,
    tp2_pct: float,
    tp3_pct: float,
) -> dict:

    direction = str(direction).upper()

    if direction == "LONG":
        return {
            "sl": avg_price * (
                1.0 - stop_loss_pct / 100.0
            ),
            "tp1": avg_price * (
                1.0 + tp1_pct / 100.0
            ),
            "tp2": avg_price * (
                1.0 + tp2_pct / 100.0
            ),
            "tp3": avg_price * (
                1.0 + tp3_pct / 100.0
            ),
        }

    if direction == "SHORT":
        return {
            "sl": avg_price * (
                1.0 + stop_loss_pct / 100.0
            ),
            "tp1": avg_price * (
                1.0 - tp1_pct / 100.0
            ),
            "tp2": avg_price * (
                1.0 - tp2_pct / 100.0
            ),
            "tp3": avg_price * (
                1.0 - tp3_pct / 100.0
            ),
        }

    raise ValueError(
        f"invalid direction={direction}"
    )


def ensure_directional_protection(
    symbol: str,
    direction: str,
    avg_price: float,
    qty: float,
    stop_loss_pct: float,
    tp_levels: list,
    trade_id: str | None = None,
) -> dict:
    """
    Гарантирует наличие Exchange TP/SL для event-driven позиции.

    ВАЖНО:
      - не открывает позицию;
      - не закрывает позицию;
      - LONG и SHORT обрабатываются симметрично;
      - существующие protection orders НЕ дублируются.

    tp_levels:
      [
        {"leg":"tp1","pnl_pct":5.0,"close_fraction":0.30},
        {"leg":"tp2","pnl_pct":8.0,"close_fraction":0.30},
        {"leg":"tp3","pnl_pct":11.0,"close_fraction":0.40},
      ]
    """

    direction = str(direction).upper()

    if direction not in (
        "LONG",
        "SHORT",
    ):
        return {
            "status": "error",
            "error": f"invalid direction={direction}",
        }

    try:
        avg_price = float(avg_price)
        qty = abs(float(qty))
        stop_loss_pct = float(
            stop_loss_pct
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        return {
            "status": "error",
            "error": str(exc),
        }

    if avg_price <= 0:
        return {
            "status": "error",
            "error": "avg_price <= 0",
        }

    if qty <= 0:
        return {
            "status": "error",
            "error": "qty <= 0",
        }

    if not (
        0 < stop_loss_pct <= 25
    ):
        return {
            "status": "error",
            "error": (
                f"invalid stop_loss_pct="
                f"{stop_loss_pct}"
            ),
        }

    bx_symbol = to_bx_symbol(symbol)

    contract = _contracts().get(
        bx_symbol
    )

    if not contract:
        return {
            "status": "error",
            "error": (
                f"contract not found: "
                f"{bx_symbol}"
            ),
        }

    precision = int(
        contract.get(
            "quantityPrecision"
        ) or 0
    )

    price_precision = int(
        contract.get(
            "pricePrecision"
        ) or 4
    )

    min_qty = float(
        contract.get(
            "tradeMinQuantity"
        )
        or contract.get(
            "minQty"
        )
        or 0
    )

    position_qty = _directional_qty(
        qty,
        precision,
    )

    if (
        position_qty <= 0
        or (
            min_qty > 0
            and position_qty < min_qty
        )
    ):
        return {
            "status": "error",
            "error": (
                f"qty={position_qty} "
                f"< minQty={min_qty}"
            ),
        }

    existing = (
        get_open_protection_directional(
            symbol,
            direction,
        )
    )

    if existing.get("status") != "ok":
        return existing

    existing_tp = list(
        existing.get(
            "tp_orders",
            [],
        )
    )

    existing_sl = list(
        existing.get(
            "sl_orders",
            [],
        )
    )

    tp_side = (
        "SELL"
        if direction == "LONG"
        else "BUY"
    )

    sl_side = (
        "SELL"
        if direction == "LONG"
        else "BUY"
    )

    # ========================================================
    # TP QUANTIZATION
    # ========================================================

    normalized_levels = []

    for tp in tp_levels:
        leg = str(
            tp.get(
                "leg",
                f"tp{len(normalized_levels)+1}",
            )
        )

        pnl_pct = float(
            tp.get(
                "pnl_pct",
                0,
            )
        )

        fraction = float(
            tp.get(
                "close_fraction",
                0,
            )
        )

        if pnl_pct <= 0:
            continue

        if fraction <= 0:
            continue

        normalized_levels.append(
            {
                "leg": leg,
                "pnl_pct": pnl_pct,
                "close_fraction": fraction,
            }
        )

    if not normalized_levels:
        return {
            "status": "error",
            "error": "no valid tp_levels",
        }

    # Если доли не дают ровно единицу —
    # нормализуем их.
    fraction_sum = sum(
        x["close_fraction"]
        for x in normalized_levels
    )

    if fraction_sum <= 0:
        return {
            "status": "error",
            "error": "invalid TP fractions",
        }

    for x in normalized_levels:
        x["close_fraction"] /= fraction_sum

    # ========================================================
    # TP CREATION
    # ========================================================

    tp_results = []

    for level in normalized_levels:

        leg = level["leg"]
        pnl_pct = level["pnl_pct"]

        # Не создаём TP дважды.
        existing_leg = None

        for order in existing_tp:

            cid = str(
                order.get(
                    "clientOrderId",
                    "",
                )
            )

            if leg in cid:
                existing_leg = order
                break

        if existing_leg:

            tp_results.append({
                "leg": leg,
                "status": "already_exists",
                "order_id": str(
                    existing_leg.get(
                        "orderId",
                        "",
                    )
                ),
                "price": float(
                    existing_leg.get(
                        "stopPrice",
                        0,
                    )
                    or existing_leg.get(
                        "price",
                        0,
                    )
                    or 0
                ),
            })
            continue

        if direction == "LONG":
            tp_price = avg_price * (
                1.0 + pnl_pct / 100.0
            )
        else:
            tp_price = avg_price * (
                1.0 - pnl_pct / 100.0
            )

        tp_qty = _directional_qty(
            position_qty
            * level[
                "close_fraction"
            ],
            precision,
        )

        if (
            tp_qty <= 0
            or (
                min_qty > 0
                and tp_qty < min_qty
            )
        ):
            return {
                "status": "error",
                "error": (
                    f"{leg}: qty={tp_qty} "
                    f"< minQty={min_qty}"
                ),
                "tp_orders": tp_results,
            }

        client_order_id = (
            build_tp_client_order_id(
                leg,
                trade_id,
            )
        )

        params = {
            "symbol": bx_symbol,
            "side": tp_side,
            "positionSide": direction,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": _format_price(
                tp_price,
                price_precision,
            ),
            "quantity": _format_qty(
                tp_qty,
                precision,
            ),
            "clientOrderId": client_order_id,
        }

        resp = _request(
            "POST",
            ORDER_PATH,
            params,
        )

        if resp.get("code") != 0:

            return {
                "status": "error",
                "error": (
                    f"{leg} TP failed: "
                    f"code={resp.get('code')} "
                    f"msg={resp.get('msg')}"
                ),
                "tp_orders": tp_results,
            }

        order = (
            (resp.get("data") or {})
            .get("order")
            or {}
        )

        result = {
            "leg": leg,
            "status": "created",
            "order_id": str(
                order.get(
                    "orderId",
                    "",
                )
            ),
            "client_order_id": (
                order.get(
                    "clientOrderId"
                )
                or client_order_id
            ),
            "price": tp_price,
            "qty": tp_qty,
            "pnl_pct": pnl_pct,
        }

        tp_results.append(
            result
        )

        log.info(
            f"[{symbol}] {direction} "
            f"{leg} TP created: "
            f"price={tp_price} "
            f"qty={tp_qty}"
        )

    # ========================================================
    # SL
    # ========================================================

    if existing_sl:

        sl = existing_sl[0]

        sl_result = {
            "status": "already_exists",
            "order_id": str(
                sl.get(
                    "orderId",
                    "",
                )
            ),
            "stop_price": float(
                sl.get(
                    "stopPrice",
                    0,
                )
                or sl.get(
                    "price",
                    0,
                )
                or 0
            ),
        }

    else:

        if direction == "LONG":

            sl_price = (
                avg_price
                * (
                    1.0
                    - stop_loss_pct
                    / 100.0
                )
            )

        else:

            sl_price = (
                avg_price
                * (
                    1.0
                    + stop_loss_pct
                    / 100.0
                )
            )

        client_order_id = (
            build_sl_client_order_id(
                trade_id
            )
        )

        params = {
            "symbol": bx_symbol,
            "side": sl_side,
            "positionSide": direction,
            "type": "STOP_MARKET",
            "stopPrice": _format_price(
                sl_price,
                price_precision,
            ),
            "quantity": _format_qty(
                position_qty,
                precision,
            ),
            "clientOrderId": client_order_id,
        }

        resp = _request(
            "POST",
            ORDER_PATH,
            params,
        )

        if resp.get("code") != 0:

            return {
                "status": "TP_PLACED_SL_FAILED",
                "error": (
                    f"SL failed: "
                    f"code={resp.get('code')} "
                    f"msg={resp.get('msg')}"
                ),
                "tp_orders": tp_results,
                "sl_result": {
                    "status": "error",
                    "error": (
                        f"code={resp.get('code')} "
                        f"msg={resp.get('msg')}"
                    ),
                },
            }

        order = (
            (resp.get("data") or {})
            .get("order")
            or {}
        )

        sl_result = {
            "status": "created",
            "order_id": str(
                order.get(
                    "orderId",
                    "",
                )
            ),
            "client_order_id": (
                order.get(
                    "clientOrderId"
                )
                or client_order_id
            ),
            "stop_price": sl_price,
            "qty": position_qty,
        }

        log.info(
            f"[{symbol}] {direction} "
            f"SL created: "
            f"stop={sl_price} "
            f"qty={position_qty}"
        )

    tp_ok = (
        len(tp_results)
        == len(normalized_levels)
    )

    sl_ok = (
        sl_result.get("status")
        in (
            "created",
            "already_exists",
        )
    )

    if tp_ok and sl_ok:

        final_status = (
            "PROTECTED"
        )

    elif tp_ok:

        final_status = (
            "TP_PLACED_SL_FAILED"
        )

    else:

        final_status = (
            "PROTECTION_FAILED"
        )

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
