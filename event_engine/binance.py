from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any

import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

log = logging.getLogger("binance_market")

BASE_URL = "https://fapi.binance.com"
KLINE_PATH = "/fapi/v1/klines"
EXCHANGE_INFO_PATH = "/fapi/v1/exchangeInfo"
TICKER_PRICE_PATH = "/fapi/v1/ticker/price"


class BinanceRateLimitError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None, retry_after_ms: int | None = None):
        super().__init__(message)
        self.code = code
        self.retry_after_ms = retry_after_ms


class BinanceSymbolUnavailableError(RuntimeError):
    def __init__(self, message: str, *, symbol: str):
        super().__init__(message)
        self.symbol = symbol


class BinanceHTTPError(RuntimeError):
    pass


class BinanceMarketClient:
    def __init__(self) -> None:
        self.base_url = str(os.environ.get("BINANCE_BASE_URL", BASE_URL)).rstrip("/")
        self.timeout_sec = float(os.environ.get("BINANCE_HTTP_TIMEOUT_SEC", "5"))
        self.min_interval_sec = max(0.0, float(os.environ.get("BINANCE_REQUEST_MIN_INTERVAL_SEC", "0.10")))
        self.retry_attempts = max(1, min(3, int(os.environ.get("BINANCE_MARKET_RETRY_ATTEMPTS", "2"))))
        self.retry_backoff_sec = max(0.0, float(os.environ.get("BINANCE_MARKET_RETRY_BACKOFF_SEC", "0.5")))
        self.exchange_info_ttl_sec = max(30.0, float(os.environ.get("BINANCE_EXCHANGE_INFO_TTL_SEC", "900")))
        self.binance_vpn_enabled = str(os.environ.get("BINANCE_VPN_ENABLED", "false")).strip().lower() in {
            "1", "true", "yes", "on",
        }
        self.binance_http_proxy = str(os.environ.get("BINANCE_HTTP_PROXY", "")).strip()
        if self.binance_vpn_enabled and not self.binance_http_proxy:
            raise BinanceHTTPError(
                "[BINANCE] BINANCE_VPN_ENABLED=true requires BINANCE_HTTP_PROXY; refusing direct fallback"
            )
        if self.binance_http_proxy and not self.binance_http_proxy.startswith(("http://", "https://")):
            raise BinanceHTTPError(
                "[BINANCE] BINANCE_HTTP_PROXY must use http:// or https://"
            )

        self._last_request_monotonic = 0.0
        self._request_lock = threading.Lock()
        self._exchange_loaded_at = 0.0
        self._symbols: dict[str, dict[str, Any]] = {}
        self._session = requests.Session()
        # Binance transport is deliberately isolated from runner-wide proxy env vars.
        # The engine workflow supplies an explicit per-session proxy when VPN mode is on;
        # all other values resolve directly and cannot accidentally inherit HTTP(S)_PROXY.
        self._session.trust_env = False
        if self.binance_vpn_enabled:
            self._session.proxies.update({
                "http": self.binance_http_proxy,
                "https": self.binance_http_proxy,
            })
        retry = Retry(total=0, connect=0, read=0, redirect=0, status=0)
        self._session.mount("https://", HTTPAdapter(max_retries=retry))
        self._session.headers.update({"Accept": "application/json"})
        log.info("[BINANCE] transport vpn_proxy_enabled=%s", self.binance_vpn_enabled)

    def _acquire_slot(self) -> None:
        with self._request_lock:
            now = time.monotonic()
            wait = self.min_interval_sec - (now - self._last_request_monotonic)
            if wait > 0:
                time.sleep(wait)
            self._last_request_monotonic = time.monotonic()

    @staticmethod
    def _parse_retry_after_ms(response: requests.Response) -> int | None:
        header = response.headers.get("Retry-After")
        if header:
            try:
                value = float(header)
                return int(time.time() * 1000 + max(0.0, value) * 1000.0)
            except (TypeError, ValueError):
                pass
        return None

    def _request_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.retry_attempts):
            self._acquire_slot()
            try:
                resp = self._session.get(
                    f"{self.base_url}{path}",
                    params=params or {},
                    timeout=self.timeout_sec,
                )
            except (requests.RequestException, TimeoutError) as exc:
                last_error = exc
                if attempt + 1 >= self.retry_attempts:
                    break
                delay = self.retry_backoff_sec * (2 ** attempt)
                if delay:
                    time.sleep(delay)
                continue

            if resp.status_code in (418, 429):
                retry_after_ms = self._parse_retry_after_ms(resp)
                raise BinanceRateLimitError(
                    f"[BINANCE] rate limited {path}: HTTP {resp.status_code} {resp.text[:300]}",
                    code=resp.status_code,
                    retry_after_ms=retry_after_ms,
                )

            if resp.status_code >= 500:
                last_error = BinanceHTTPError(f"[BINANCE] HTTP {resp.status_code}: {resp.text[:300]}")
                if attempt + 1 >= self.retry_attempts:
                    break
                delay = self.retry_backoff_sec * (2 ** attempt)
                if delay:
                    time.sleep(delay)
                continue

            if resp.status_code >= 400:
                raise BinanceHTTPError(f"[BINANCE] HTTP {resp.status_code}: {resp.text[:500]}")

            try:
                return resp.json()
            except ValueError as exc:
                raise BinanceHTTPError(f"[BINANCE] invalid JSON from {path}: {resp.text[:500]}") from exc

        raise BinanceHTTPError(f"[BINANCE] request failed after {self.retry_attempts} attempts for {path}: {last_error}") from last_error

    def refresh_exchange_info(self, *, force: bool = False) -> dict[str, dict[str, Any]]:
        if not force and self._symbols and (time.monotonic() - self._exchange_loaded_at) < self.exchange_info_ttl_sec:
            return self._symbols
        data = self._request_json(EXCHANGE_INFO_PATH)
        if not isinstance(data, dict):
            raise BinanceHTTPError("[BINANCE] exchangeInfo response is not an object")
        symbols: dict[str, dict[str, Any]] = {}
        for raw in data.get("symbols", []) or []:
            if not isinstance(raw, dict):
                continue
            symbol = str(raw.get("symbol", "")).upper().strip()
            if not symbol:
                continue
            if str(raw.get("quoteAsset", "")).upper() != "USDT":
                continue
            if str(raw.get("contractType", "")).upper() != "PERPETUAL":
                continue
            if str(raw.get("status", "")).upper() != "TRADING":
                continue
            symbols[symbol] = raw
        self._symbols = symbols
        self._exchange_loaded_at = time.monotonic()
        log.info("[BINANCE] Active USDT perpetual contracts=%d", len(symbols))
        return symbols

    @staticmethod
    def _normalise_symbol(symbol: str) -> str:
        value = str(symbol or "").strip().upper()
        value = value.replace("/", "").replace("-", "").replace(" ", "")
        value = re.sub(r"_(PERP|USDT)$", "", value)
        return value

    def resolve_symbol(self, symbol: str) -> str | None:
        target = self._normalise_symbol(symbol)
        symbols = self.refresh_exchange_info()
        if target in symbols:
            return target
        if target.endswith("USDT") and target[:-4] + "USDT" in symbols:
            return target[:-4] + "USDT"
        base = target[:-4] if target.endswith("USDT") else target
        candidate = f"{base}USDT"
        return candidate if candidate in symbols else None

    def contract_exists(self, symbol: str) -> bool:
        return self.resolve_symbol(symbol) is not None

    def fetch_klines(self, symbol: str, interval: str, limit: int = 250) -> list[dict[str, Any]]:
        resolved = self.resolve_symbol(symbol)
        if not resolved:
            raise BinanceSymbolUnavailableError(
                f"[BINANCE] no active USDT perpetual contract for {symbol}",
                symbol=str(symbol),
            )
        requested_limit = max(1, int(limit))
        # Binance includes the currently forming candle in kline responses.
        # Signal generation must remain causal, so request one extra row and
        # discard any still-open candle while preserving the requested number
        # of closed bars whenever the exchange returns enough history.
        api_limit = min(1500, requested_limit + 1)
        rows = self._request_json(
            KLINE_PATH,
            {"symbol": resolved, "interval": interval, "limit": api_limit},
        )
        if not isinstance(rows, list):
            raise BinanceHTTPError(f"[BINANCE] Klines response for {resolved}/{interval} is not a list")

        duration_ms = {
            "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
            "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000,
            "6h": 21_600_000, "12h": 43_200_000, "1d": 86_400_000,
        }.get(interval)
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 11:
                continue
            try:
                open_time = int(row[0])
                open_price = float(row[1])
                high = float(row[2])
                low = float(row[3])
                close = float(row[4])
                volume = float(row[5])
                close_time = int(row[6])
                quote_volume = float(row[7])
                try:
                    trade_count = int(row[8])
                except (TypeError, ValueError):
                    trade_count = None
                taker_buy_base = float(row[9])
                taker_buy_quote = float(row[10])
            except (TypeError, ValueError, IndexError):
                continue
            if not all(map(lambda x: x == x and abs(x) != float("inf"), [open_price, high, low, close, volume, quote_volume, taker_buy_base, taker_buy_quote])):
                continue
            if close_time > int(time.time() * 1000):
                continue
            out.append({
                "open_time": open_time,
                "close_time": close_time if close_time else (open_time + duration_ms if duration_ms else open_time),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "quote_volume": quote_volume,
                "trade_count": trade_count,
                "taker_buy_base": taker_buy_base,
                "taker_buy_quote": taker_buy_quote,
                # Keep the exchange row verbatim for forensic/audit purposes.
                # This preserves index 8 (trade count) and index 11 (reserved/ignore)
                # without changing the established OHLCV field semantics.
                "raw_row": list(row),
                "taker_flow_valid": True,
                "bar_delta_usdt": 2.0 * taker_buy_quote - quote_volume,
            })
        return out[-requested_limit:]

    def fetch_price(self, symbol: str) -> float:
        resolved = self.resolve_symbol(symbol)
        if not resolved:
            raise BinanceSymbolUnavailableError(
                f"[BINANCE] no active USDT perpetual contract for {symbol}",
                symbol=str(symbol),
            )
        payload = self._request_json(TICKER_PRICE_PATH, {"symbol": resolved})
        if not isinstance(payload, dict):
            raise BinanceHTTPError(f"[BINANCE] ticker response for {resolved} is not an object")
        try:
            price = float(payload.get("price"))
        except (TypeError, ValueError):
            price = 0.0
        if price <= 0:
            raise BinanceHTTPError(f"[BINANCE] invalid ticker price for {resolved}: {payload}")
        return price


CLIENT = BinanceMarketClient()


def refresh_exchange_info(*, force: bool = False) -> dict[str, dict[str, Any]]:
    return CLIENT.refresh_exchange_info(force=force)


def resolve_symbol(symbol: str) -> str | None:
    return CLIENT.resolve_symbol(symbol)


def contract_exists(symbol: str) -> bool:
    return CLIENT.contract_exists(symbol)


def fetch_klines(symbol: str, interval: str, limit: int = 250) -> list[dict[str, Any]]:
    return CLIENT.fetch_klines(symbol, interval, limit)


def fetch_price(symbol: str) -> float:
    return CLIENT.fetch_price(symbol)
