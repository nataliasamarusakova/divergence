import time

import pytest

from event_engine import binance as bx


def test_binance_kline_normalizes_to_signal_schema(monkeypatch):
    client = bx.BinanceMarketClient()
    monkeypatch.setattr(client, "resolve_symbol", lambda symbol: "BTCUSDT")
    monkeypatch.setattr(
        client,
        "_request_json",
        lambda path, params=None: [[
            1_700_000_000_000,
            "100.0", "110.0", "90.0", "105.0", "1000.0",
            1_700_000_059_999, "100000.0", "42", "550.0", "55000.0", "0",
        ]],
    )
    rows = client.fetch_klines("BTC-USDT", "1m", 1)
    assert rows == [{
        "open_time": 1_700_000_000_000,
        "close_time": 1_700_000_059_999,
        "open": 100.0,
        "high": 110.0,
        "low": 90.0,
        "close": 105.0,
        "volume": 1000.0,
        "quote_volume": 100000.0,
        "trade_count": 42,
        "taker_buy_base": 550.0,
        "taker_buy_quote": 55000.0,
        "raw_row": [
            1_700_000_000_000,
            "100.0", "110.0", "90.0", "105.0", "1000.0",
            1_700_000_059_999, "100000.0", "42", "550.0", "55000.0", "0",
        ],
        "taker_flow_valid": True,
        "bar_delta_usdt": 10000.0,
    }]


def test_binance_kline_preserves_all_12_exchange_fields(monkeypatch):
    client = bx.BinanceMarketClient()
    monkeypatch.setattr(client, "resolve_symbol", lambda symbol: "BTCUSDT")
    raw_row = [
        1_700_000_000_000,
        "100.0", "110.0", "90.0", "105.0", "1000.0",
        1_700_000_059_999, "100000.0", "42", "550.0", "55000.0", "reserved",
    ]
    monkeypatch.setattr(client, "_request_json", lambda path, params=None: [raw_row])

    rows = client.fetch_klines("BTC-USDT", "1m", 1)

    assert rows[0]["trade_count"] == 42
    assert rows[0]["raw_row"] == raw_row
    assert len(rows[0]["raw_row"]) == 12
    assert rows[0]["raw_row"][8] == "42"
    assert rows[0]["raw_row"][11] == "reserved"


def test_binance_kline_excludes_unclosed_bar_and_preserves_closed_limit(monkeypatch):
    client = bx.BinanceMarketClient()
    monkeypatch.setattr(client, "resolve_symbol", lambda symbol: "BTCUSDT")
    requested = {}
    now_ms = 1_700_000_060_000
    monkeypatch.setattr(bx.time, "time", lambda: now_ms / 1000.0)

    closed = [
        1_700_000_000_000,
        "100.0", "110.0", "90.0", "105.0", "1000.0",
        1_700_000_059_999, "100000.0", "42", "550.0", "55000.0", "0",
    ]
    open_bar = [
        1_700_000_060_000,
        "105.0", "111.0", "104.0", "109.0", "900.0",
        1_700_000_119_999, "98000.0", "40", "500.0", "54500.0", "0",
    ]

    def fake_request(path, params=None):
        requested.update(params or {})
        return [closed, open_bar]

    monkeypatch.setattr(client, "_request_json", fake_request)
    rows = client.fetch_klines("BTC-USDT", "1m", 1)

    assert requested["limit"] == 2
    assert len(rows) == 1
    assert rows[0]["close_time"] == 1_700_000_059_999
    assert rows[0]["close"] == 105.0


def test_binance_paused_or_nontrading_symbol_is_not_resolved(monkeypatch):
    client = bx.BinanceMarketClient()
    monkeypatch.setattr(
        client,
        "_request_json",
        lambda path, params=None: {
            "symbols": [
                {"symbol": "BTCUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"},
                {"symbol": "PAUSEDUSDT", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "SETTLING"},
            ]
        },
    )
    client.refresh_exchange_info(force=True)
    assert client.contract_exists("BTC-USDT") is True
    assert client.contract_exists("PAUSED-USDT") is False


def test_binance_http_429_fails_fast_without_hidden_retry(monkeypatch):
    client = bx.BinanceMarketClient()
    client.retry_attempts = 3
    calls = []

    class FakeResponse:
        status_code = 429
        headers = {"Retry-After": "60"}
        text = '{"code":-1003,"msg":"Too many requests"}'

    def fake_get(*args, **kwargs):
        calls.append(1)
        return FakeResponse()

    monkeypatch.setattr(client._session, "get", fake_get)
    with pytest.raises(bx.BinanceRateLimitError) as exc:
        client._request_json("/fapi/v1/klines", {"symbol": "BTCUSDT", "interval": "1h", "limit": 10})
    assert len(calls) == 1
    assert exc.value.retry_after_ms > int(time.time() * 1000)


def test_binance_price_rejects_unknown_symbol(monkeypatch):
    client = bx.BinanceMarketClient()
    monkeypatch.setattr(client, "resolve_symbol", lambda symbol: None)
    with pytest.raises(bx.BinanceSymbolUnavailableError):
        client.fetch_price("NOTREAL")
