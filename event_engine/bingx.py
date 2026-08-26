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
    resp = _request("GET", KLINE_PATH, {"symbol": bx, "interval": interval, "limit": limit}, signed=False)
    if resp.get("code") != 0:
        raise RuntimeError(f"BingX klines error {bx}/{interval}: {resp.get('msg')}")
    rows = resp.get("data") or []
    out = []
    now = int(time.time() * 1000)
    duration_ms = {"15m": 15*60*1000, "1h": 60*60*1000}.get(interval)
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 11:
            continue
        ot = int(row[0]); ct = int(row[6]) if row[6] is not None else ot + duration_ms
        if ct > now: continue
        open_, high, low, close, volume = map(float, row[1:6])
        quote = float(row[7]); taker_b = float(row[9]); taker_q = float(row[10])
        if min(open_, high, low, close) <= 0 or volume < 0 or quote < 0: continue
        if taker_b < 0 or taker_q < 0 or taker_b > volume*1.001 or taker_q > quote*1.001: continue
        delta_usdt = 2*taker_q - quote
        out.append({"open_time": ot, "close_time": ct, "open": open_, "high": high, "low": low, "close": close, "volume": volume, "quote_volume": quote, "taker_buy_quote": taker_q, "bar_delta_usdt": delta_usdt})
    out.sort(key=lambda x: x["close_time"])
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
