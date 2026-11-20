"""Regression tests for shared exchange/notification HTTP boundaries."""
from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from event_engine import bingx, telegram


class _JsonResponse:
    def __init__(self, status_code: int, payload: object):
        self.status_code = status_code
        self.headers = {"Content-Type": "application/json"}
        self._payload = payload

    def json(self):
        return self._payload


def test_bingx_rejects_non_2xx_even_when_body_claims_success(monkeypatch):
    class _Session:
        def request(self, **kwargs):
            self.kwargs = kwargs
            return _JsonResponse(502, {"code": 0, "msg": "ok", "data": {"orderId": "SIMULATED"}})

    session = _Session()
    monkeypatch.setattr(bingx, "FAST_SESSION", session)
    metadata = {}

    result = bingx._request(
        "GET", bingx.LIVE_PRICE_PATH, {}, signed=False,
        retryable=False, response_metadata=metadata,
    )

    assert result["code"] == -1
    assert result["http_status"] == 502
    assert metadata["http_status"] == 502
    assert metadata["http_error"] is True


def test_bingx_does_not_follow_cross_host_redirect_with_api_key(monkeypatch):
    target_requests: list[str | None] = []

    class _TargetHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib handler API
            target_requests.append(self.headers.get("X-BX-APIKEY"))
            body = json.dumps({"code": 0, "msg": "ok", "data": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    target = ThreadingHTTPServer(("127.0.0.1", 0), _TargetHandler)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()
    target_url = f"http://127.0.0.1:{target.server_port}/redirect-target"

    class _RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib handler API
            self.send_response(302)
            self.send_header("Location", target_url)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            return

    source = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    source_thread = threading.Thread(target=source.serve_forever, daemon=True)
    source_thread.start()
    try:
        monkeypatch.setattr(bingx, "BASE_URL", f"http://127.0.0.1:{source.server_port}")
        monkeypatch.setattr(bingx, "API_KEY", "DUMMY_BX_API_KEY_NOT_REAL")
        monkeypatch.setattr(bingx, "SECRET_KEY", "DUMMY_BX_SECRET_NOT_REAL")
        monkeypatch.setattr(bingx, "FAST_SESSION", requests.Session())

        result = bingx._request(
            "GET", "/authenticated-test", {}, signed=True, retryable=False,
        )

        assert result["code"] == -1
        assert result["http_status"] == 302
        assert target_requests == []
    finally:
        source.shutdown()
        target.shutdown()
        source.server_close()
        target.server_close()
        source_thread.join(timeout=2)
        target_thread.join(timeout=2)


def test_telegram_redacts_bot_token_from_request_exceptions(monkeypatch, capsys):
    token = "123456789:THIS_IS_A_DUMMY_TOKEN_NOT_REAL"
    monkeypatch.setenv("TG_BOT_TOKEN", token)
    monkeypatch.setenv("TG_CHAT_IDS", "123456")

    def _raise_with_request_url(url, **kwargs):
        raise requests.ConnectionError(f"connection failed for {url}/sendMessage")

    monkeypatch.setattr(telegram.requests, "post", _raise_with_request_url)
    result = telegram.send_detailed("test notification")
    output = capsys.readouterr().out

    assert result["123456"]["sent"] is False
    assert token not in result["123456"]["error"]
    assert token not in output


def test_telegram_disables_redirects_and_rejects_redirect_response(monkeypatch):
    token = "123456789:THIS_IS_ANOTHER_DUMMY_TOKEN_NOT_REAL"
    monkeypatch.setenv("TG_BOT_TOKEN", token)
    monkeypatch.setenv("TG_CHAT_IDS", "123456")
    calls = []

    class _Response:
        status_code = 302
        text = '{"ok": true, "result": {"message_id": 7}}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True, "result": {"message_id": 7}}

    def _fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return _Response()

    monkeypatch.setattr(telegram.requests, "post", _fake_post)
    result = telegram.send_detailed("test notification")

    assert calls[0][1]["allow_redirects"] is False
    assert result["123456"]["sent"] is False


def test_bingx_does_not_return_signed_request_url_in_transport_error(monkeypatch):
    secret_signature = "SIGNED_QUERY_SIGNATURE_MUST_NOT_LEAK"

    class _Session:
        def request(self, **kwargs):
            raise requests.ConnectionError(
                "connection failed for https://open-api-vst.bingx.com/openApi/swap/v2/trade/order?"
                f"timestamp=123&signature={secret_signature}"
            )

    monkeypatch.setattr(bingx, "FAST_SESSION", _Session())
    monkeypatch.setattr(bingx, "API_KEY", "DUMMY_BX_API_KEY_NOT_REAL")
    monkeypatch.setattr(bingx, "SECRET_KEY", "DUMMY_BX_SECRET_NOT_REAL")

    result = bingx._request(
        "GET", bingx.POSITION_PATH, {"symbol": "BTC-USDT"}, signed=True,
        retryable=False,
    )

    assert result["code"] == -1
    assert secret_signature not in result["msg"]
    assert "ConnectionError" in result["msg"]


def test_refresh_contracts_rejects_null_collection_without_erasing_valid_cache(monkeypatch):
    cached_contract = {"symbol": "BTC-USDT", "status": "1", "apiStateOpen": "true"}
    previous_cache = {
        "ts": 123.0,
        "data": {"BTC-USDT": cached_contract},
        "by_display_name": {"BTCUSDT": cached_contract},
    }
    monkeypatch.setattr(bingx, "CACHE", dict(previous_cache))
    monkeypatch.setattr(
        bingx, "_request",
        lambda *_args, **_kwargs: {"code": 0, "msg": "ok", "data": None},
    )

    with pytest.raises(RuntimeError, match="unexpected data shape"):
        bingx.refresh_contracts()

    assert bingx.CACHE["data"] == previous_cache["data"]
    assert bingx.CACHE["by_display_name"] == previous_cache["by_display_name"]
    assert bingx.CACHE["ts"] == previous_cache["ts"]


def test_fetch_klines_rejects_null_collection_instead_of_returning_empty(monkeypatch):
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "BTC-USDT")
    monkeypatch.setattr(
        bingx, "_request",
        lambda *_args, **_kwargs: {"code": 0, "msg": "ok", "data": None},
    )

    with pytest.raises(RuntimeError, match="unexpected data shape"):
        bingx.fetch_klines("BTC-USDT", "1h", limit=10)


def test_bingx_fetch_klines_discards_non_finite_ohlcv_and_sanitizes_optional_flow(monkeypatch):
    import time

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "TEST-USDT")
    open_time = int(time.time() * 1000) - 6 * 3_600_000
    close_time = open_time + 3_600_000 - 1
    valid_with_bad_optional = [
        open_time, "100", "101", "99", "100.5", "10", close_time,
        "1000", "0", "1", "inf",
    ]
    invalid_nan_ohlc = [
        open_time + 3_600_000, "nan", "101", "99", "100", "10",
        close_time + 3_600_000,
    ]
    invalid_inf_volume = [
        open_time + 2 * 3_600_000, "100", "101", "99", "100", "inf",
        close_time + 2 * 3_600_000,
    ]
    valid_without_flow = [
        open_time + 3 * 3_600_000, "100", "101", "99", "100.5", "10",
        close_time + 3 * 3_600_000,
    ]
    monkeypatch.setattr(
        bingx,
        "_request",
        lambda *_args, **_kwargs: {
            "code": 0,
            "data": [valid_with_bad_optional, invalid_nan_ohlc, invalid_inf_volume, valid_without_flow],
        },
    )

    rows = bingx.fetch_klines("TEST", "1h", limit=10)

    assert len(rows) == 2
    assert all(
        math.isfinite(row[key])
        for row in rows
        for key in ("open", "high", "low", "close", "volume")
    )
    assert rows[0]["quote_volume"] == 1000.0
    assert rows[0]["taker_buy_quote"] is None
    assert rows[0]["taker_flow_valid"] is False
    assert rows[0]["bar_delta_usdt"] is None
    assert rows[1]["taker_flow_valid"] is False


def test_bingx_user_symbol_map_precedes_builtin_contract_alias(monkeypatch):
    from event_engine import bingx as bx

    custom = {"symbol": "CUSTOM-USDT", "displayName": "CUSTOM-USDT"}
    built_in = {"symbol": "LIGHTER-USDT", "displayName": "LIGHTER-USDT"}
    catalog = {"CUSTOM-USDT": custom, "LIGHTER-USDT": built_in}
    monkeypatch.setattr(bx, "contracts", lambda **_kwargs: catalog)
    monkeypatch.setattr(bx, "SYMBOL_MAP", {"LIT-USDT": "CUSTOM-USDT"})
    monkeypatch.setattr(bx, "CACHE", {
        "ts": 0.0,
        "data": catalog,
        "by_display_name": {},
    })

    assert bx.get_contract("LIT") is custom
    assert bx.canonical_contract_symbol("LIT") == "CUSTOM-USDT"

    # When both the dashed and base aliases are configured, both resolution
    # paths must honor canonical_contract_symbol's dashed-key precedence.
    dashed_custom = {"symbol": "DASHED-USDT", "displayName": "DASHED-USDT"}
    base_custom = {"symbol": "BASE-USDT", "displayName": "BASE-USDT"}
    catalog.update({"DASHED-USDT": dashed_custom, "BASE-USDT": base_custom})
    monkeypatch.setattr(bx, "contracts", lambda **_kwargs: catalog)
    monkeypatch.setattr(bx, "SYMBOL_MAP", {
        "LIT-USDT": "DASHED-USDT",
        "LIT": "BASE-USDT",
    })
    monkeypatch.setattr(bx, "CACHE", {
        "ts": 0.0,
        "data": catalog,
        "by_display_name": {},
    })
    assert bx.get_contract("LIT") is dashed_custom
    assert bx.canonical_contract_symbol("LIT") == "DASHED-USDT"


def test_bingx_explicit_trump_sol_contract_is_not_collapsed_into_legacy_trumpsol_alias(monkeypatch):
    from event_engine import bingx as bx

    trump_sol = {"symbol": "TRUMP-SOL-USDT", "displayName": "TRUMP-SOL-USDT"}
    trump = {"symbol": "TRUMP-USDT", "displayName": "TRUMP-USDT"}
    catalog = {"TRUMP-SOL-USDT": trump_sol, "TRUMP-USDT": trump}
    monkeypatch.setattr(bx, "contracts", lambda **_kwargs: catalog)
    monkeypatch.setattr(bx, "SYMBOL_MAP", {})
    monkeypatch.setattr(bx, "CACHE", {
        "ts": 0.0,
        "data": catalog,
        "by_display_name": {},
    })

    assert bx.get_contract("TRUMP-SOL-USDT") is trump_sol
    assert bx.canonical_contract_symbol("TRUMP-SOL-USDT") == "TRUMP-SOL-USDT"
    assert bx.get_contract("TRUMPSOL") is trump
