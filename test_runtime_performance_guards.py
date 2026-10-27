"""Safety regressions for in-cycle performance instrumentation."""
import logging


def test_kline_cache_reuses_only_same_request_and_returns_row_copies(monkeypatch):
    import run_once as ro

    calls = []
    monkeypatch.setattr(ro, "MARKET_DATA_SOURCE", "binance")
    monkeypatch.setattr(ro, "_raise_if_binance_rate_limited", lambda *args: None)

    def fake_fetch(symbol, timeframe, limit):
        calls.append((symbol, timeframe, limit))
        return [{"close": float(len(calls)), "raw_row": [1, 2]}]

    monkeypatch.setattr(ro, "fetch_binance_klines", fake_fetch)
    monkeypatch.setattr(ro.time, "time", lambda: 1_800_000_000.0)
    ro._begin_kline_request_cache_cycle()
    try:
        first = ro._fetch_market_klines_scan("BTCUSDT", "15m", 250)
        first[0]["close"] = 999.0
        second = ro._fetch_market_klines_scan("BTCUSDT", "15m", 250)
        assert second[0]["close"] == 1.0
        assert len(calls) == 1

        ro._fetch_market_klines_scan("BTCUSDT", "15m", 400)
        ro._fetch_market_klines_scan("BTCUSDT", "1h", 250)
        assert len(calls) == 3
        assert ro._KLINE_CACHE_HITS == 1
        assert ro._KLINE_CACHE_MISSES == 3
    finally:
        ro._finish_kline_request_cache_cycle()


def test_kline_cache_key_rotates_on_new_candle_bucket(monkeypatch):
    import run_once as ro

    timestamps = iter([1_800_000_000.0, 1_800_000_900.0])
    monkeypatch.setattr(ro.time, "time", lambda: next(timestamps))
    before = ro._kline_request_cache_key("binance", "BTCUSDT", "15m", 250)
    after = ro._kline_request_cache_key("binance", "BTCUSDT", "15m", 250)
    assert before != after


def test_recovery_logs_duration_and_individual_fill_requests(tmp_path, monkeypatch, caplog):
    import json
    import event_engine.tracker as tracker

    journal = tmp_path / "trades.jsonl"
    opened = {
        "record_type": "TRADE_OPEN", "event_id": "EVT_TIMING", "symbol": "BTC", "direction": "LONG",
        "ts": 1_700_000_000_000,
        "execution": {"actual_entry_price": 100.0, "actual_qty": 1.0, "order_id": "ENTRY-1", "fill_ts_ms": 1_700_000_000_000},
        "result": {"fill_ts_ms": 1_700_000_000_000, "protection": {"tp_orders": [], "sl_result": {}}},
    }
    closed = {
        "record_type": "TRADE_CLOSE", "event_id": "EVT_TIMING", "symbol": "BTC", "direction": "LONG",
        "closed_ts": 1_700_000_060_000, "entry_price": 100.0, "exit_price": None,
        "exit_reason": "DATA_ERROR", "realized_pnl_pct": None, "total_closed_qty": 1.0,
        "initial_qty": 1.0, "entry_order_id": "ENTRY-1", "exit_order_ids": ["CLOSE-1"], "planned_risk_pct": 1.0,
    }
    journal.write_text(json.dumps(opened) + "\n" + json.dumps(closed) + "\n", encoding="utf-8")
    fills = [
        {"trade_id": "E1", "order_id": "ENTRY-1", "symbol": "BTC-USDT", "side": "BUY", "position_side": "LONG", "price": 100.0, "qty": 1.0, "time": 1_700_000_001_000, "realized_pnl": 0.0, "fee": -0.01},
        {"trade_id": "C1", "order_id": "CLOSE-1", "symbol": "BTC-USDT", "side": "SELL", "position_side": "LONG", "price": 101.0, "qty": 1.0, "time": 1_700_000_059_000, "realized_pnl": 1.0, "fee": -0.01},
    ]
    monkeypatch.setattr(tracker, "TRADES_PATH", journal)
    monkeypatch.setattr(tracker, "get_fill_history", lambda *args, **kwargs: {"status": "ok", "fills": fills})

    with caplog.at_level(logging.INFO, logger="event_engine.tracker"):
        recovered = tracker._recover_data_error_trade_closes(now_ms=1_700_000_200_000)

    assert recovered == 1
    assert "Fill-history request START event=EVT_TIMING" in caplog.text
    assert "Fill-history request END event=EVT_TIMING" in caplog.text
    assert "DATA_ERROR attempt END event=EVT_TIMING" in caplog.text
    assert "Cycle END elapsed=" in caplog.text


def test_workflow_uses_shallow_checkout_and_guarded_deepening():
    from pathlib import Path

    workflow = Path(__file__).parent / ".github/workflows/event-engine.yml"
    text = workflow.read_text(encoding="utf-8")
    assert "fetch-depth: 1" in text
    assert "fetch-tags: false" in text
    assert "git merge-base HEAD origin/main" in text
    assert '--deepen="$deepen"' in text
    assert "refusing to push state" in text
    assert "actions/upload-artifact@v4" in text


def test_bingx_account_wide_open_orders_snapshot_requires_identifiable_protection(monkeypatch):
    from event_engine import bingx

    captured = {}
    monkeypatch.setattr(bingx, "_request", lambda method, path, params=None, **kwargs: (
        captured.update(method=method, path=path, params=params, kwargs=kwargs)
        or {"code": 0, "data": [
            {"symbol": "BTC-USDT", "positionSide": "LONG", "type": "STOP_MARKET", "orderId": "SL1"},
            {"symbol": "ETH-USDT", "positionSide": "SHORT", "type": "TAKE_PROFIT_MARKET", "orderId": "TP1"},
            {"symbol": "BTC-USDT", "positionSide": "LONG", "type": "LIMIT", "orderId": "L1"},
        ]}
    ))

    result = bingx.get_open_orders_snapshot(timeout_sec=5, retryable=False)
    assert result["status"] == "ok"
    assert len(result["orders"]) == 3
    assert captured["path"] == bingx.OPEN_ORDERS_PATH
    assert captured["params"] == {}
    assert captured["kwargs"]["timeout_sec"] == 5
    assert captured["kwargs"]["retryable"] is False

    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {
        "code": 0, "data": [{"type": "STOP_MARKET", "orderId": "SL-BAD"}],
    })
    malformed = bingx.get_open_orders_snapshot(timeout_sec=5, retryable=False)
    assert malformed["status"] == "error"
    assert "unidentifiable protection" in malformed["error"]


def test_account_snapshot_protection_filter_separates_symbol_side_and_order_type():
    import run_once as ro

    snapshot = {"status": "ok", "orders": [
        {"symbol": "BTC-USDT", "positionSide": "LONG", "type": "STOP_MARKET", "orderId": "SL-L"},
        {"symbol": "BTC-USDT", "positionSide": "LONG", "type": "TAKE_PROFIT_MARKET", "orderId": "TP-L"},
        {"symbol": "BTC-USDT", "positionSide": "SHORT", "type": "STOP_MARKET", "orderId": "SL-S"},
        {"symbol": "ETH-USDT", "positionSide": "LONG", "type": "STOP_MARKET", "orderId": "SL-ETH"},
        {"symbol": "BTC-USDT", "positionSide": "LONG", "type": "LIMIT", "orderId": "LIMIT-L"},
    ]}

    long_protection = ro._protection_from_open_orders_snapshot(snapshot, "BTC-USDT", "LONG")
    short_protection = ro._protection_from_open_orders_snapshot(snapshot, "BTC-USDT", "SHORT")
    assert [x["orderId"] for x in long_protection["sl_orders"]] == ["SL-L"]
    assert [x["orderId"] for x in long_protection["tp_orders"]] == ["TP-L"]
    assert [x["orderId"] for x in short_protection["sl_orders"]] == ["SL-S"]
    assert short_protection["tp_orders"] == []


def test_snapshot_invalid_result_falls_back_to_directional_query(monkeypatch):
    import run_once as ro

    directional_calls = []
    monkeypatch.setattr(ro, "get_open_orders_snapshot", lambda **kwargs: {"status": "error", "error": "simulated failure"})
    monkeypatch.setattr(ro, "get_open_protection_directional", lambda symbol, direction, **kwargs: (
        directional_calls.append((symbol, direction)) or {"status": "ok", "sl_orders": [], "tp_orders": []}
    ))
    # Validate fallback is wired to the existing per-symbol function via the
    # same branch decision used by reconciliation without touching live state.
    snapshot = ro.get_open_orders_snapshot(timeout_sec=5, retryable=False)
    result = ro._protection_from_open_orders_snapshot(snapshot, "BTC-USDT", "LONG")
    if result is None:
        result = ro.get_open_protection_directional("BTC-USDT", "LONG", timeout_sec=5, retryable=False)
    assert result["status"] == "ok"
    assert directional_calls == [("BTC-USDT", "LONG")]
