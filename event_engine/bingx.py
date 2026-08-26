from __future__ import annotations

import hashlib
import hmac
import json
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

BASE_URL = os.environ.get(
    "BINGX_BASE_URL",
    "https://open-api-vst.bingx.com",
).rstrip("/")

MARGIN_USDT = float(
    os.environ.get(
        "BINGX_MARGIN_USDT",
        "1",
    )
)

LEVERAGE = int(
    os.environ.get(
        "BINGX_LEVERAGE",
        "10",
    )
)

MAX_LEVERAGE = int(
    os.environ.get(
        "BINGX_MAX_LEVERAGE",
        "50",
    )
)

try:
    SYMBOL_MAP = json.loads(
        os.environ.get(
            "BINGX_SYMBOL_MAP",
            "{}",
        )
    )
except Exception:
    SYMBOL_MAP = {}


CONTRACTS_PATH = "/openApi/swap/v2/quote/contracts"
KLINE_PATH = "/openApi/swap/v3/quote/klines"
ORDER_PATH = "/openApi/swap/v2/trade/order"
OPEN_ORDERS_PATH = "/openApi/swap/v2/trade/openOrders"
POSITION_PATH = os.environ.get(
    "BINGX_POSITIONS_PATH",
    "/openApi/swap/v2/user/positions",
)
LEVERAGE_PATH = "/openApi/swap/v2/trade/leverage"


CACHE = {
    "ts": 0.0,
    "data": {},
    "by_display_name": {},
}

TTL = 3600


# ============================================================================
# GENERIC HELPERS
# ============================================================================

def _sign(
    params: dict[str, Any],
) -> str:
    qs = urlencode(params)
    return hmac.new(
        SECRET_KEY.encode(),
        qs.encode(),
        hashlib.sha256,
    ).hexdigest()


def _request(
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    signed: bool = True,
) -> dict:
    params = dict(params or {})
    headers = {}

    if signed:
        if not API_KEY or not SECRET_KEY:
            return {
                "code": -1,
                "msg": "missing BingX credentials",
            }

        params["timestamp"] = str(
            int(time.time() * 1000)
        )

        params["signature"] = _sign(
            params
        )

        headers["X-BX-APIKEY"] = API_KEY

    try:
        response = requests.request(
            "GET" if method == "GET" else method,
            BASE_URL + path,
            params=params,
            headers=headers,
            timeout=15,
        )

        try:
            return response.json()
        except Exception:
            return {
                "code": response.status_code,
                "msg": response.text,
            }

    except Exception as exc:
        return {
            "code": -1,
            "msg": str(exc),
        }


def _normalize_orders_list(
    response: dict,
) -> list[dict]:
    """
    BingX responses can expose lists under several wrappers.
    Normalize them to list[dict].
    """

    if not isinstance(
        response,
        dict,
    ):
        return []

    data = response.get("data")

    if isinstance(
        data,
        list,
    ):
        return [
            x
            for x in data
            if isinstance(
                x,
                dict,
            )
        ]

    if isinstance(
        data,
        dict,
    ):
        for key in (
            "positions",
            "position",
            "orders",
            "order",
            "list",
        ):
            value = data.get(key)

            if isinstance(
                value,
                list,
            ):
                return [
                    x
                    for x in value
                    if isinstance(
                        x,
                        dict,
                    )
                ]

            if isinstance(
                value,
                dict,
            ):
                return [value]

        return [data]

    return []


def _contracts() -> dict[str, dict]:
    """
    Compatibility alias used by protection code.
    """
    return contracts()


def _round_qty(
    qty: float,
    precision: int,
) -> float:
    if qty <= 0:
        return 0.0

    exponent = Decimal("1").scaleb(
        -int(precision)
    )

    rounded = Decimal(
        str(qty)
    ).quantize(
        exponent,
        rounding=ROUND_DOWN,
    )

    return float(
        rounded
    )


def _format_qty(
    qty: float,
    precision: int,
) -> str:
    precision = max(
        0,
        int(precision),
    )

    return f"{float(qty):.{precision}f}"


def _format_price(
    price: float,
    precision: int,
) -> str:
    precision = max(
        0,
        int(precision),
    )

    return f"{float(price):.{precision}f}"


def build_tp_client_order_id(
    leg: str,
    trade_id: str | None,
) -> str:
    suffix = str(
        trade_id or "NOID"
    )[:16]

    return (
        f"EVT_{suffix}_"
        f"{str(leg).upper()}"
    )[:32]


def build_sl_client_order_id(
    trade_id: str | None,
) -> str:
    suffix = str(
        trade_id or "NOID"
    )[:20]

    return (
        f"EVT_{suffix}_SL"
    )[:32]


# ============================================================================
# CONTRACTS
# ============================================================================

def refresh_contracts() -> dict[str, Any]:
    response = _request(
        "GET",
        CONTRACTS_PATH,
        signed=False,
    )

    if response.get("code") != 0:
        raise RuntimeError(
            f"BingX contracts error: "
            f"{response.get('msg')}"
        )

    data = {}
    by_name = {}

    for contract in (
        response.get("data") or []
    ):
        symbol = str(
            contract.get(
                "symbol",
                "",
            )
        ).strip().upper()

        display_name = str(
            contract.get(
                "displayName",
                "",
            )
        ).strip().upper()

        if symbol:
            data[symbol] = contract

        if display_name:
            by_name[
                display_name
            ] = contract

    CACHE.update(
        {
            "ts": time.time(),
            "data": data,
            "by_display_name": by_name,
        }
    )

    log.info(
        "BingX contracts=%d",
        len(data),
    )

    return data


def contracts() -> dict[str, dict]:
    if (
        CACHE["data"]
        and time.time()
        - CACHE["ts"]
        < TTL
    ):
        return CACHE["data"]

    try:
        return refresh_contracts()
    except Exception:
        return CACHE["data"]


def get_contract(
    symbol: str,
) -> dict | None:

    s = (
        symbol or ""
    ).strip().upper()

    if not s:
        return None

    mapped = SYMBOL_MAP.get(
        s
    )

    if mapped:
        contract = contracts().get(
            str(
                mapped
            ).strip().upper()
        )

        if contract:
            return contract

    direct = (
        s
        if s.endswith("-USDT")
        else f"{s.replace('-', '')}-USDT"
    )

    contract = contracts().get(
        direct
    )

    if contract:
        return contract

    base = (
        s.replace(
            "-USDT",
            "",
        )
        .replace(
            "-",
            "",
        )
    )

    for contract in CACHE[
        "data"
    ].values():

        display_name = str(
            contract.get(
                "displayName",
                "",
            )
        ).upper().replace(
            "-",
            "",
        )

        contract_symbol = str(
            contract.get(
                "symbol",
                "",
            )
        ).upper()

        if (
            display_name
            == f"{base}USDT"
            or display_name
            == base
            or contract_symbol.endswith(
                f"{base}-USDT"
            )
        ):
            return contract

    return CACHE[
        "by_display_name"
    ].get(
        f"{base}-USDT"
    )


def to_bx_symbol(
    symbol: str,
) -> str | None:

    contract = get_contract(
        symbol
    )

    if not contract:
        return None

    value = str(
        contract.get(
            "symbol",
            "",
        )
    ).strip().upper()

    return value or None


def contract_exists(
    symbol: str,
) -> bool:

    contract = get_contract(
        symbol
    )

    if not contract:
        return False

    status_ok = (
        str(
            contract.get(
                "status",
                "",
            )
        ) == "1"
        or contract.get(
            "status"
        ) == 1
    )

    api_open = str(
        contract.get(
            "apiStateOpen",
            "",
        )
    ).lower()

    return (
        status_ok
        and api_open == "true"
    )


def classify_contract(
    contract: dict | None,
) -> str:

    if not contract:
        return "unknown"

    symbol = str(
        contract.get(
            "symbol",
            "",
        )
    ).upper()

    if symbol.startswith(
        (
            "NCSK",
            "NCSI",
        )
    ):
        return "equity"

    if symbol.startswith(
        "NCCO"
    ):
        return "commodity"

    if symbol.startswith(
        "NCFX"
    ):
        return "forex"

    return "crypto"


# ============================================================================
# KLINES
# ============================================================================

def fetch_klines(
    symbol: str,
    interval: str,
    limit: int = 250,
) -> list[dict]:

    bx = to_bx_symbol(
        symbol
    )

    if not bx:
        raise ValueError(
            f"No BingX contract for {symbol}"
        )

    response = _request(
        "GET",
        KLINE_PATH,
        {
            "symbol": bx,
            "interval": interval,
            "limit": limit,
        },
        signed=False,
    )

    code = response.get(
        "code"
    )

    if code not in (
        0,
        "0",
    ):
        raise RuntimeError(
            f"BingX klines error "
            f"{bx}/{interval}: "
            f"code={code} "
            f"msg={response.get('msg')}"
        )

    rows = (
        response.get(
            "data"
        )
        or []
    )

    if not isinstance(
        rows,
        list,
    ):
        raise RuntimeError(
            f"Unexpected BingX klines payload "
            f"{bx}/{interval}: "
            f"data_type="
            f"{type(rows).__name__}"
        )

    if rows and isinstance(
        rows[0],
        dict,
    ):
        print(
            "[BINGX_KLINE_FIELDS]",
            bx,
            sorted(
                rows[0].keys()
            ),
        )

    first_shape = "empty"

    if rows:
        if isinstance(
            rows[0],
            (list, tuple),
        ):
            first_shape = (
                f"array_len="
                f"{len(rows[0])}"
            )
        elif isinstance(
            rows[0],
            dict,
        ):
            first_shape = "dict"
        else:
            first_shape = (
                type(
                    rows[0]
                ).__name__
            )

    print(
        f"[BINGX_KLINES_RAW] "
        f"symbol={bx} "
        f"interval={interval} "
        f"rows={len(rows)} "
        f"first_shape={first_shape}"
    )

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
    }.get(
        interval
    )

    now_ms = int(
        time.time() * 1000
    )

    output = []

    for row in rows:

        # ==============================================================
        # ARRAY
        # ==============================================================

        if isinstance(
            row,
            (list, tuple),
        ):

            if len(row) < 6:
                continue

            try:
                open_time = int(
                    row[0]
                )

                open_price = float(
                    row[1]
                )

                high = float(
                    row[2]
                )

                low = float(
                    row[3]
                )

                close = float(
                    row[4]
                )

                volume = float(
                    row[5]
                )

                if (
                    len(row) >= 7
                    and row[6] is not None
                ):
                    close_time = int(
                        row[6]
                    )
                else:
                    close_time = (
                        open_time
                        + duration_ms
                        if duration_ms
                        else open_time
                    )

                quote_volume = None
                taker_buy_base = None
                taker_buy_quote = None

                if (
                    len(row) >= 8
                    and row[7] is not None
                ):
                    try:
                        quote_volume = float(
                            row[7]
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        quote_volume = None

                if (
                    len(row) >= 10
                    and row[9] is not None
                ):
                    try:
                        taker_buy_base = float(
                            row[9]
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        taker_buy_base = None

                if (
                    len(row) >= 11
                    and row[10] is not None
                ):
                    try:
                        taker_buy_quote = float(
                            row[10]
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        taker_buy_quote = None

            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                continue

        # ==============================================================
        # DICT
        # ==============================================================

        elif isinstance(
            row,
            dict,
        ):

            def pick(
                *names,
            ):
                for name in names:
                    if (
                        name in row
                        and row[name] is not None
                    ):
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
                    pick(
                        "open"
                    )
                )

                high = float(
                    pick(
                        "high"
                    )
                )

                low = float(
                    pick(
                        "low"
                    )
                )

                close = float(
                    pick(
                        "close"
                    )
                )

                volume = float(
                    pick(
                        "volume"
                    )
                )

                raw_close_time = pick(
                    "closeTime",
                    "close_time",
                )

                close_time = (
                    int(
                        raw_close_time
                    )
                    if raw_close_time is not None
                    else (
                        open_time
                        + duration_ms
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
                    "buyVolume",
                )

                taker_quote_raw = pick(
                    "takerBuyQuoteVolume",
                    "taker_buy_quote",
                    "takerBuyQuote",
                    "buyQuoteVolume",
                )

                quote_volume = (
                    float(
                        quote_volume_raw
                    )
                    if quote_volume_raw is not None
                    else None
                )

                taker_buy_base = (
                    float(
                        taker_base_raw
                    )
                    if taker_base_raw is not None
                    else None
                )

                taker_buy_quote = (
                    float(
                        taker_quote_raw
                    )
                    if taker_quote_raw is not None
                    else None
                )

            except (
                TypeError,
                ValueError,
                KeyError,
            ):
                continue

        else:
            continue

        # ==============================================================
        # CLOSED CANDLE
        # ==============================================================

        if close_time > now_ms:
            continue

        # ==============================================================
        # OHLC VALIDATION
        # ==============================================================

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

        # ==============================================================
        # TAKER FLOW
        # ==============================================================

        taker_flow_valid = (
            quote_volume is not None
            and taker_buy_base is not None
            and taker_buy_quote is not None
            and quote_volume >= 0
            and taker_buy_base >= 0
            and taker_buy_quote >= 0
            and taker_buy_base
            <= volume * 1.001
            + 1e-8
            and taker_buy_quote
            <= quote_volume * 1.001
            + 1e-8
        )

        if taker_flow_valid:
            bar_delta_usdt = (
                2.0 * taker_buy_quote
                - quote_volume
            )
        else:
            bar_delta_usdt = None

        output.append(
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

    output.sort(
        key=lambda x: x["close_time"]
    )

    deduped = []
    seen_close_times = set()

    for bar in output:
        close_time = bar[
            "close_time"
        ]

        if close_time in seen_close_times:
            continue

        seen_close_times.add(
            close_time
        )
        deduped.append(
            bar
        )

    output = deduped

    print(
        f"[BINGX_KLINES_PARSED] "
        f"symbol={bx} "
        f"interval={interval} "
        f"bars={len(output)} "
        f"taker_valid="
        f"{sum(1 for x in output if x['taker_flow_valid'])}"
    )

    return output


# ============================================================================
# LEVERAGE / POSITIONS
# ============================================================================

def _set_leverage(
    bx_symbol: str,
    leverage: int,
) -> bool:

    success = False

    for side in (
        "LONG",
        "SHORT",
    ):
        response = _request(
            "POST",
            LEVERAGE_PATH,
            {
                "symbol": bx_symbol,
                "side": side,
                "leverage": str(
                    leverage
                ),
            },
        )

        if response.get(
            "code"
        ) == 0:
            success = True

    return success


def get_positions() -> list[dict]:

    response = _request(
        "GET",
        POSITION_PATH,
        {},
        signed=True,
    )

    if response.get(
        "code"
    ) != 0:
        return []

    return _normalize_orders_list(
        response
    )


def get_position_directional(
    symbol: str,
    direction: str,
) -> dict:

    bx_symbol = to_bx_symbol(
        symbol
    )

    direction = str(
        direction
    ).upper()

    if not bx_symbol:
        return {
            "status": "error",
            "error": "contract_not_found",
        }

    if direction not in (
        "LONG",
        "SHORT",
    ):
        return {
            "status": "error",
            "error": (
                f"invalid direction="
                f"{direction}"
            ),
            "symbol": bx_symbol,
        }

    response = _request(
        "GET",
        POSITION_PATH,
        {
            "symbol": bx_symbol,
        },
        signed=True,
    )

    if response.get(
        "code"
    ) != 0:
        return {
            "status": "error",
            "error": (
                f"get_position failed: "
                f"code="
                f"{response.get('code')} "
                f"msg="
                f"{response.get('msg')}"
            ),
            "symbol": bx_symbol,
        }

    for position in _normalize_orders_list(
        response
    ):

        position_symbol = str(
            position.get(
                "symbol",
                bx_symbol,
            )
        ).upper()

        if position_symbol != bx_symbol:
            continue

        position_side = str(
            position.get(
                "positionSide",
                "",
            )
        ).upper()

        if position_side != direction:
            continue

        try:
            qty = abs(
                float(
                    position.get(
                        "positionAmt",
                        0,
                    )
                    or 0
                )
            )

            avg_price = float(
                position.get(
                    "avgPrice",
                    0,
                )
                or position.get(
                    "entryPrice",
                    0,
                )
                or 0
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            qty <= 0
            or avg_price <= 0
        ):
            continue

        return {
            "status": "found",
            "symbol": bx_symbol,
            "positionSide": direction,
            "avgPrice": avg_price,
            "positionAmt": qty,
            "entryPrice": float(
                position.get(
                    "entryPrice",
                    0,
                )
                or avg_price
            ),
        }

    return {
        "status": "not_found",
        "symbol": bx_symbol,
        "positionSide": direction,
    }


def has_open_position(
    symbol: str,
    direction: str,
) -> bool:

    position = (
        get_position_directional(
            symbol,
            direction,
        )
    )

    return (
        position.get(
            "status"
        )
        == "found"
    )


def wait_for_position_fill_directional(
    symbol: str,
    direction: str,
    timeout_sec: int = 30,
    poll_interval: float = 1.0,
) -> dict:

    started = time.time()

    while (
        time.time()
        - started
        < timeout_sec
    ):

        position = (
            get_position_directional(
                symbol,
                direction,
            )
        )

        if (
            position.get(
                "status"
            )
            == "found"
        ):

            log.info(
                f"[{symbol}] "
                f"directional position confirmed: "
                f"side={direction} "
                f"avgPrice="
                f"{position.get('avgPrice')} "
                f"qty="
                f"{position.get('positionAmt')}"
            )

            return position

        time.sleep(
            poll_interval
        )

    return {
        "status": "timeout",
        "symbol": to_bx_symbol(
            symbol
        ),
        "positionSide": direction,
    }


# ============================================================================
# MARKET ENTRY
# ============================================================================

def open_market(
    symbol: str,
    direction: str,
    price: float,
    trade_id: str,
) -> dict:

    bx_symbol = to_bx_symbol(
        symbol
    )

    if not bx_symbol:
        return {
            "status": "error",
            "error": "contract_not_found",
        }

    contract = (
        get_contract(symbol)
        or {}
    )

    if not contract_exists(
        symbol
    ):
        return {
            "status": "error",
            "error": (
                "contract_unavailable"
            ),
            "symbol": bx_symbol,
        }

    if has_open_position(
        symbol,
        direction,
    ):
        return {
            "status": "existing_position",
            "symbol": bx_symbol,
        }

    try:
        quantity_precision = int(
            contract.get(
                "quantityPrecision"
            )
            or 0
        )
    except Exception:
        quantity_precision = 0

    min_qty = float(
        contract.get(
            "tradeMinQuantity"
        )
        or contract.get(
            "minQty"
        )
        or 0
    )

    multiplier = float(
        contract.get(
            "multiplier"
        )
        or 1
    )

    max_lev = int(
        contract.get(
            "maxLongLeverage"
        )
        or contract.get(
            "maxLeverage"
        )
        or MAX_LEVERAGE
    )

    leverage = min(
        LEVERAGE,
        max_lev,
    )

    quantity = (
        MARGIN_USDT
        * leverage
    ) / max(
        price
        * multiplier,
        1e-12,
    )

    quantity = _round_qty(
        quantity,
        quantity_precision,
    )

    if (
        quantity < min_qty
    ):

        required_leverage = math.ceil(
            (
                min_qty
                * price
                * multiplier
            )
            / max(
                MARGIN_USDT,
                1e-9,
            )
        )

        leverage = min(
            max(
                required_leverage,
                leverage,
            ),
            max_lev,
        )

        quantity = (
            MARGIN_USDT
            * leverage
        ) / max(
            price
            * multiplier,
            1e-12,
        )

        quantity = _round_qty(
            quantity,
            quantity_precision,
        )

    if (
        quantity <= 0
        or quantity < min_qty
    ):
        return {
            "status": "error",
            "error": (
                f"qty={quantity} "
                f"< min_qty={min_qty}"
            ),
        }

    _set_leverage(
        bx_symbol,
        leverage,
    )

    direction = str(
        direction
    ).upper()

    if direction == "LONG":
        side = "BUY"
    elif direction == "SHORT":
        side = "SELL"
    else:
        return {
            "status": "error",
            "error": (
                f"invalid direction="
                f"{direction}"
            ),
        }

    client_order_id = (
        f"EVT_OPEN_"
        f"{trade_id[:24]}"
    )[:32]

    params = {
        "symbol": bx_symbol,
        "side": side,
        "positionSide": direction,
        "type": "MARKET",
        "quantity": _format_qty(
            quantity,
            quantity_precision,
        ),
        "clientOrderId": client_order_id,
    }

    response = _request(
        "POST",
        ORDER_PATH,
        params,
    )

    if response.get(
        "code"
    ) != 0:
        return {
            "status": "error",
            "error": str(
                response.get(
                    "msg"
                )
            ),
            "symbol": bx_symbol,
            "response": response,
        }

    order = (
        (
            response.get(
                "data"
            )
            or {}
        ).get(
            "order"
        )
        or {}
    )

    return {
        "status": "opened",
        "symbol": bx_symbol,
        "qty": quantity,
        "leverage": leverage,
        "order_id": (
            order.get(
                "orderId"
            )
            or None
        ),
        "client_order_id": (
            order.get(
                "clientOrderId"
            )
            or client_order_id
        ),
        "response": response,
    }


# ============================================================================
# EXISTING OPEN PROTECTION
# ============================================================================

def get_open_protection_directional(
    symbol: str,
    direction: str,
) -> dict:

    bx_symbol = to_bx_symbol(
        symbol
    )

    direction = str(
        direction
    ).upper()

    response = _request(
        "GET",
        OPEN_ORDERS_PATH,
        {
            "symbol": bx_symbol,
        },
        signed=True,
    )

    if response.get(
        "code"
    ) != 0:

        return {
            "status": "error",
            "error": (
                f"openOrders failed: "
                f"code="
                f"{response.get('code')} "
                f"msg="
                f"{response.get('msg')}"
            ),
            "tp_orders": [],
            "sl_orders": [],
        }

    tp_orders = []
    sl_orders = []

    for order in _normalize_orders_list(
        response
    ):

        position_side = str(
            order.get(
                "positionSide",
                "",
            )
        ).upper()

        if position_side not in (
            direction,
            "BOTH",
        ):
            continue

        order_type = str(
            order.get(
                "type",
                "",
            )
        ).upper()

        if order_type in (
            "TAKE_PROFIT",
            "TAKE_PROFIT_MARKET",
        ):
            tp_orders.append(
                order
            )

        elif order_type in (
            "STOP",
            "STOP_MARKET",
        ):
            sl_orders.append(
                order
            )

    return {
        "status": "ok",
        "symbol": bx_symbol,
        "positionSide": direction,
        "tp_orders": tp_orders,
        "sl_orders": sl_orders,
    }


# ============================================================================
# PROTECTION
# ============================================================================

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

    direction = str(
        direction
    ).upper()

    if direction == "LONG":
        return {
            "sl": avg_price * (
                1.0
                - stop_loss_pct / 100.0
            ),
            "tp1": avg_price * (
                1.0
                + tp1_pct / 100.0
            ),
            "tp2": avg_price * (
                1.0
                + tp2_pct / 100.0
            ),
            "tp3": avg_price * (
                1.0
                + tp3_pct / 100.0
            ),
        }

    if direction == "SHORT":
        return {
            "sl": avg_price * (
                1.0
                + stop_loss_pct / 100.0
            ),
            "tp1": avg_price * (
                1.0
                - tp1_pct / 100.0
            ),
            "tp2": avg_price * (
                1.0
                - tp2_pct / 100.0
            ),
            "tp3": avg_price * (
                1.0
                - tp3_pct / 100.0
            ),
        }

    raise ValueError(
        f"invalid direction="
        f"{direction}"
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
    Устанавливает Exchange TP/SL для уже открытой позиции.

    LONG:
      entry BUY
      TP/SL close SELL positionSide=LONG

    SHORT:
      entry SELL
      TP/SL close BUY positionSide=SHORT
    """

    direction = str(
        direction
    ).upper()

    if direction not in (
        "LONG",
        "SHORT",
    ):
        return {
            "status": "error",
            "error": (
                f"invalid direction="
                f"{direction}"
            ),
        }

    try:
        avg_price = float(
            avg_price
        )

        qty = abs(
            float(qty)
        )

        stop_loss_pct = float(
            stop_loss_pct
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        return {
            "status": "error",
            "error": str(
                exc
            ),
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
        0
        < stop_loss_pct
        <= 25
    ):
        return {
            "status": "error",
            "error": (
                "invalid stop_loss_pct="
                f"{stop_loss_pct}"
            ),
        }

    bx_symbol = to_bx_symbol(
        symbol
    )

    if not bx_symbol:
        return {
            "status": "error",
            "error": "contract_not_found",
        }

    contract = contracts().get(
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
        )
        or 0
    )

    price_precision = int(
        contract.get(
            "pricePrecision"
        )
        or 8
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

    if existing.get(
        "status"
    ) != "ok":
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

    close_side = (
        "SELL"
        if direction == "LONG"
        else "BUY"
    )

    # ================================================================
    # NORMALIZE TP LEVELS
    # ================================================================

    normalized_levels = []

    for index, tp in enumerate(
        tp_levels
    ):

        if not isinstance(
            tp,
            dict,
        ):
            continue

        leg = str(
            tp.get(
                "leg",
                f"tp{index + 1}",
            )
        )

        pnl_pct = float(
            tp.get(
                "pnl_pct",
                0,
            )
            or 0
        )

        fraction = float(
            tp.get(
                "close_fraction",
                0,
            )
            or 0
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

    fraction_sum = sum(
        x[
            "close_fraction"
        ]
        for x in normalized_levels
    )

    if fraction_sum <= 0:
        return {
            "status": "error",
            "error": "invalid TP fractions",
        }

    for level in (
        normalized_levels
    ):
        level[
            "close_fraction"
        ] /= fraction_sum

    # ================================================================
    # CREATE TP
    # ================================================================

    tp_results = []

    for level in (
        normalized_levels
    ):

        leg = level[
            "leg"
        ]

        pnl_pct = level[
            "pnl_pct"
        ]

        existing_leg = None

        for order in (
            existing_tp
        ):

            client_id = str(
                order.get(
                    "clientOrderId",
                    "",
                )
            )

            if (
                leg.lower()
                in client_id.lower()
            ):
                existing_leg = order
                break

        if existing_leg:

            tp_results.append(
                {
                    "leg": leg,
                    "status": (
                        "already_exists"
                    ),
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
                }
            )

            continue

        if direction == "LONG":

            tp_price = avg_price * (
                1.0
                + pnl_pct / 100.0
            )

        else:

            tp_price = avg_price * (
                1.0
                - pnl_pct / 100.0
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
                    f"{leg}: qty="
                    f"{tp_qty} "
                    f"< minQty="
                    f"{min_qty}"
                ),
                "tp_orders": tp_results,
                "sl_result": {
                    "status": "not_created",
                },
            }

        client_order_id = (
            build_tp_client_order_id(
                leg,
                trade_id,
            )
        )

        params = {
            "symbol": bx_symbol,
            "side": close_side,
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

        response = _request(
            "POST",
            ORDER_PATH,
            params,
            signed=True,
        )

        if response.get(
            "code"
        ) != 0:

            return {
                "status": "error",
                "error": (
                    f"{leg} TP failed: "
                    f"code="
                    f"{response.get('code')} "
                    f"msg="
                    f"{response.get('msg')}"
                ),
                "tp_orders": tp_results,
                "sl_result": {
                    "status": "not_created",
                },
                "response": response,
            }

        order = (
            (
                response.get(
                    "data"
                )
                or {}
            ).get(
                "order"
            )
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
            f"[{symbol}] "
            f"{direction} "
            f"{leg} TP created: "
            f"price={tp_price} "
            f"qty={tp_qty}"
        )

    # ================================================================
    # CREATE SL
    # ================================================================

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
            sl_price = avg_price * (
                1.0
                - stop_loss_pct / 100.0
            )
        else:
            sl_price = avg_price * (
                1.0
                + stop_loss_pct / 100.0
            )

        client_order_id = (
            build_sl_client_order_id(
                trade_id
            )
        )

        params = {
            "symbol": bx_symbol,
            "side": close_side,
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

        response = _request(
            "POST",
            ORDER_PATH,
            params,
            signed=True,
        )

        if response.get(
            "code"
        ) != 0:

            return {
                "status": "TP_PLACED_SL_FAILED",
                "error": (
                    f"SL failed: "
                    f"code="
                    f"{response.get('code')} "
                    f"msg="
                    f"{response.get('msg')}"
                ),
                "tp_orders": tp_results,
                "sl_result": {
                    "status": "error",
                    "error": (
                        f"code="
                        f"{response.get('code')} "
                        f"msg="
                        f"{response.get('msg')}"
                    ),
                },
                "response": response,
            }

        order = (
            (
                response.get(
                    "data"
                )
                or {}
            ).get(
                "order"
            )
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
            f"[{symbol}] "
            f"{direction} SL created: "
            f"stop={sl_price} "
            f"qty={position_qty}"
        )

    # ================================================================
    # FINAL RESULT
    # ================================================================

    tp_ok = (
        len(tp_results)
        == len(
            normalized_levels
        )
        and all(
            str(
                x.get(
                    "status",
                    ""
                )
            ).lower()
            in {
                "created",
                "already_exists",
            }
            for x in tp_results
        )
    )

    sl_ok = (
        str(
            sl_result.get(
                "status",
                "",
            )
        ).lower()
        in {
            "created",
            "already_exists",
        }
    )

    if tp_ok and sl_ok:
        final_status = "PROTECTED"

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
