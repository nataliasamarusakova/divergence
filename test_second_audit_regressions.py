"""Regression tests for independently reviewed V14.8 lifecycle findings in the V14.9 patch.

All exchange and Telegram interactions are mocked; no external side effects.
"""
import json

import pytest


def test_pending_close_journal_retries_immutable_record_without_reclassification(tmp_path, monkeypatch):
    from event_engine import tracker

    trade = {
        "symbol": "BTC-USDT", "direction": "LONG", "remaining_qty": 0.0,
        "realized_pnl_qty": 0.5, "exit_reason": "STOP_LOSS",
        "close_journal_pending": True,
        "close_journal_snapshot": {
            "record_type": "TRADE_CLOSE", "event_id": "EVT-1",
            "exit_reason": "STOP_LOSS", "exit_price_source": "SL_FILL",
        },
    }
    appended = []
    monkeypatch.setattr(tracker, "_close_record_exists", lambda event_id: False)
    monkeypatch.setattr(tracker, "_append_trade_record", lambda record: appended.append(dict(record)))

    assert tracker._retry_pending_close_journal(trade, "EVT-1") is True
    assert appended[0]["exit_reason"] == "STOP_LOSS"
    assert trade["close_journal_pending"] is False
    assert trade["close_notification_pending"] is True
    assert "close_journal_snapshot" not in trade


def test_pending_close_journal_without_snapshot_fails_closed(tmp_path, monkeypatch):
    from event_engine import tracker

    trade = {"close_journal_pending": True, "remaining_qty": 0.0, "realized_pnl_qty": 0.5}
    monkeypatch.setattr(tracker, "_close_record_exists", lambda event_id: False)
    monkeypatch.setattr(tracker, "_append_trade_record", lambda record: pytest.fail("must not invent close record"))
    assert tracker._retry_pending_close_journal(trade, "EVT-legacy") is False
    assert trade["close_journal_pending"] is True


def test_sr_data_retry_history_is_loaded_when_mandatory_directional_gate_is_enabled(monkeypatch):
    import run_once

    monkeypatch.setattr(run_once, "SR_3H_DIRECTIONAL_GATE_ENABLED", True)
    monkeypatch.setattr(run_once, "AJAY_SR_ROOM_ENABLED", False)
    monkeypatch.setattr(run_once, "AJAY_SR_ROOM_MODE", "off")
    assert run_once._should_load_sr_data_failure_counts() is True


def test_trade_close_and_tp_leg_dynamic_html_are_escaped():
    from event_engine.tracker import format_trade_closed_message, format_tp_hit_message

    closed = format_trade_closed_message(
        name="BTC", symbol="BTC-USDT", direction="LONG", entry_price=100,
        exit_price=99, pnl_pct=-1.0, realized_rr=-0.5, planned_rr=1.2,
        duration_min=2, peak_pnl=0, max_drawdown=1,
        exit_reason="STOP <unknown> & failed", event_type="TYPE <A>", timeframe="1H&4H",
    )
    assert "STOP &lt;unknown&gt; &amp; failed" in closed
    assert "TYPE &lt;A&gt;" in closed
    assert "1h&amp;4h" in closed
    tp = format_tp_hit_message("BTC", "BTC-USDT", "tp<one>&", 1, 101, 1, 0, 0)
    assert "tp&lt;one&gt;&amp;" in tp


def test_query_order_maps_documented_order_absent_to_absent(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 109421, "msg": "Order does not exist"})
    result = bingx._query_order("BTC", client_order_id="CID-1")
    assert result["status"] == "absent"


def test_query_order_null_data_is_not_success(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": None})
    result = bingx._query_order("BTC", client_order_id="CID-1")
    assert result["status"] != "ok"


def test_cashflow_fill_evidence_supports_more_than_seven_days_and_paginates(monkeypatch):
    from event_engine import tracker

    start = 1_700_000_000_000
    end = start + 30 * 24 * 60 * 60 * 1000
    calls = []
    def fake_get_fill_history(symbol, query_start, query_end, *, order_id, page_index, page_size, **kwargs):
        calls.append((order_id, page_index, query_start, query_end))
        base = {
            "status": "ok", "has_more": page_index == 1,
            "fills": [],
        }
        if order_id == "ENTRY" and page_index == 1:
            base["fills"] = [{"order_id": "ENTRY", "symbol": "BTC-USDT", "time": start + 1000,
                              "qty": 1, "price": 100, "fee": -0.01, "fee_asset": "USDT",
                              "trade_id": "E1", "position_side": "LONG", "realized_pnl": 0}]
        elif order_id == "EXIT" and page_index == 1:
            base["fills"] = [{"order_id": "EXIT", "symbol": "BTC-USDT", "time": end - 1000,
                              "qty": 1, "price": 99, "fee": -0.01, "fee_asset": "USDT",
                              "trade_id": "X1", "position_side": "LONG", "realized_pnl": -1}]
        elif order_id == "ENTRY" and page_index == 2:
            base["has_more"] = False
        elif order_id == "EXIT" and page_index == 2:
            base["has_more"] = False
        return base
    monkeypatch.setattr(tracker, "get_fill_history", fake_get_fill_history)

    fills, error = tracker._load_exchange_fill_evidence(
        "BTC-USDT", "LONG", start, end, "ENTRY", {"ENTRY", "EXIT"},
    )
    assert error is None
    assert {row["trade_id"] for row in fills} == {"E1", "X1"}
    assert {(order, page) for order, page, *_ in calls} == {("ENTRY", 1), ("ENTRY", 2), ("EXIT", 1), ("EXIT", 2)}


def test_cashflow_fill_evidence_rejects_unfinished_pagination(monkeypatch):
    from event_engine import tracker
    start = 1_700_000_000_000
    monkeypatch.setattr(tracker, "get_fill_history", lambda *a, **k: {
        "status": "ok", "has_more": True, "fills": [],
    })
    fills, error = tracker._load_exchange_fill_evidence(
        "BTC-USDT", "LONG", start, start + 30_000, "ENTRY", {"ENTRY", "EXIT"},
    )
    assert fills is None
    assert error and "pagination" in error


def test_restored_old_sl_order_id_is_adopted_into_trade_and_history():
    from event_engine import tracker

    trade = {"sl_order": {"order_id": "OLD-CANCELED", "stop_price": 105}, "sl_order_history": []}
    result = {"status": "error", "restored_sl_order": {
        "status": "created", "order_id": "RESTORED-NEW", "stop_price": 105, "qty": 0.4,
    }}
    changed = tracker._apply_restored_sl_to_trade(trade, "OLD-CANCELED", 105.0, result, 123456)
    assert changed is True
    assert trade["sl_order"]["order_id"] == "RESTORED-NEW"
    assert trade["sl_order_history"] == [{
        "order_id": "OLD-CANCELED", "stop_price": 105.0, "replaced_ts": 123456,
    }]


def test_signed_get_http_retry_recomputes_timestamp_and_signature(monkeypatch):
    from types import SimpleNamespace
    from event_engine import bingx

    calls = []
    class FakeSession:
        def request(self, **kwargs):
            calls.append(dict(kwargs["params"]))
            status = 502 if len(calls) == 1 else 200
            payload = {"code": 0, "data": {"ok": True}}
            return SimpleNamespace(status_code=status, headers={}, json=lambda: payload)

    monkeypatch.setattr(bingx, "API_KEY", "test-api-key")
    monkeypatch.setattr(bingx, "SECRET_KEY", "test-secret")
    monkeypatch.setattr(bingx, "FAST_SESSION", FakeSession())
    monkeypatch.setattr(bingx.time, "sleep", lambda delay: None)
    tick = {"n": 0}
    def fake_apply(params):
        tick["n"] += 1
        params["timestamp"] = 1000 + tick["n"]
        params["signature"] = f"sig-{tick['n']}"
    monkeypatch.setattr(bingx, "_apply_request_timestamp", fake_apply)

    result = bingx._request("GET", "/test/signed", {"symbol": "BTC-USDT"}, signed=True, retryable=True)
    assert result == {"code": 0, "data": {"ok": True}}
    assert len(calls) == 2
    assert calls[0]["timestamp"] != calls[1]["timestamp"]
    assert calls[0]["signature"] != calls[1]["signature"]


def test_postfill_rollback_with_partial_residual_is_registered_as_protection_failed(monkeypatch):
    import run_once

    monkeypatch.setattr(run_once, "MAX_ENTRY_DRIFT_PCT", 2.0)
    monkeypatch.setattr(run_once, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", False)
    monkeypatch.setattr(run_once, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: {
        "status": "opened", "order_id": "ENTRY-1", "leverage": 10,
        "order_reference_price": 100.0,
    })
    monkeypatch.setattr(run_once, "wait_for_position_fill_directional", lambda **k: {
        "status": "found", "positionAmt": "1.0", "avgPrice": "104.0", "entryPrice": "104.0",
    })
    monkeypatch.setattr(run_once, "emergency_close_position", lambda *a, **k: {
        "status": "unknown", "error": "simulated close response lost",
    })
    monkeypatch.setattr(run_once, "_finalize_rollback_protection_cleanup", lambda *a, **k: {
        "status": "pending", "pending_cleanup_id": "ROLLBACK-CLEANUP",
    })
    monkeypatch.setattr(run_once, "get_position_directional", lambda *a, **k: {
        "status": "found", "positionAmt": "0.4", "avgPrice": "104.0", "entryPrice": "104.0",
    })

    result = run_once.execute_new_position(
        "TEST", "LONG", 100.0,
        {"risk_pct": 1.0, "signal_price": 100.0, "event_type": "REGULAR_BULLISH_RSI"},
        "EVT_ROLLBACK_PARTIAL",
    )
    assert result["entry_failure_status"] == "ENTRY_DRIFT_EXCEEDED"
    assert result["status"] == "opened_protection_failed"
    assert result["position"]["positionAmt"] == pytest.approx(0.4)
    assert result["protection"]["sl_result"] == {}
    assert result["protection"]["tp_orders"] == []
    assert result["setup_used_for_protection"]["event_type"] == "REGULAR_BULLISH_RSI"


def test_postfill_rollback_single_not_found_read_keeps_last_confirmed_position(monkeypatch):
    import run_once

    monkeypatch.setattr(run_once, "MAX_ENTRY_DRIFT_PCT", 2.0)
    monkeypatch.setattr(run_once, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", False)
    monkeypatch.setattr(run_once, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(run_once, "open_market", lambda *a, **k: {
        "status": "opened", "order_id": "ENTRY-2", "leverage": 10,
        "order_reference_price": 100.0,
    })
    monkeypatch.setattr(run_once, "wait_for_position_fill_directional", lambda **k: {
        "status": "found", "positionAmt": "1.0", "avgPrice": "104.0", "entryPrice": "104.0",
    })
    monkeypatch.setattr(run_once, "emergency_close_position", lambda *a, **k: {"status": "unknown"})
    monkeypatch.setattr(run_once, "_finalize_rollback_protection_cleanup", lambda *a, **k: {"status": "pending"})
    monkeypatch.setattr(run_once, "get_position_directional", lambda *a, **k: {"status": "not_found"})

    result = run_once.execute_new_position(
        "TEST", "LONG", 100.0,
        {"risk_pct": 1.0, "signal_price": 100.0, "event_type": "REGULAR_BULLISH_RSI"},
        "EVT_ROLLBACK_UNKNOWN",
    )
    assert result["status"] == "opened_protection_check_required"
    assert result["position"]["positionAmt"] == pytest.approx(1.0)
    assert result["rollback_state_unknown"] is True


def test_signed_get_http_retry_handles_non_json_502_and_resigns(monkeypatch):
    from types import SimpleNamespace
    from event_engine import bingx

    calls = []
    class FakeResponse:
        def __init__(self, status, payload=None):
            self.status_code = status
            self.headers = {}
            self._payload = payload
        def json(self):
            if self._payload is None:
                raise ValueError("HTML gateway response is not JSON")
            return self._payload

    class FakeSession:
        def request(self, **kwargs):
            calls.append(dict(kwargs["params"]))
            if len(calls) == 1:
                return FakeResponse(502)
            return FakeResponse(200, {"code": 0, "data": {"ok": True}})

    monkeypatch.setattr(bingx, "API_KEY", "test-api-key")
    monkeypatch.setattr(bingx, "SECRET_KEY", "test-secret")
    monkeypatch.setattr(bingx, "FAST_SESSION", FakeSession())
    monkeypatch.setattr(bingx.time, "sleep", lambda delay: None)
    tick = {"n": 0}
    def fake_apply(params):
        tick["n"] += 1
        params["timestamp"] = str(2000 + tick["n"])
        params["signature"] = f"sig-{tick['n']}"
    monkeypatch.setattr(bingx, "_apply_request_timestamp", fake_apply)

    result = bingx._request("GET", "/test/signed", {"symbol": "BTC-USDT"}, signed=True, retryable=True)
    assert result == {"code": 0, "data": {"ok": True}}
    assert len(calls) == 2
    assert calls[0]["timestamp"] != calls[1]["timestamp"]
    assert calls[0]["signature"] != calls[1]["signature"]


def test_tracker_html_escapes_symbol_and_direction_in_operator_alert_headings(monkeypatch, tmp_path):
    from event_engine import tracker

    queued = []
    monkeypatch.setattr(tracker, "_queue_notification", lambda *a, **kw: queued.append(kw))
    tracker._notify_position_state_unknown("BTC<USDT>&", "LONG<bad>", "qty < min & state unknown")
    tracker._notify_be_failure("ETH<USDT>&", "SHORT<bad>", "SL < min_qty & restore failed")

    assert len(queued) == 2
    assert "BTC&lt;USDT&gt;&amp; LONG&lt;bad&gt;" in queued[0]["text"]
    assert "ETH&lt;USDT&gt;&amp; SHORT&lt;bad&gt;" in queued[1]["text"]
    assert "qty &lt; min &amp; state unknown" in queued[0]["text"]


def test_reconcile_daily_journal_symbol_uses_canonical_alias():
    from event_engine import reconcile_daily

    assert reconcile_daily._daily_journal_symbol({"symbol": "LIT"}) == "LIGHTER-USDT"
    assert reconcile_daily._daily_journal_symbol({"symbol": "LIGHTER-USDT"}) == "LIGHTER-USDT"


def test_update_active_trades_handles_close_journal_pending_before_exchange_reads(tmp_path, monkeypatch):
    import copy
    from event_engine import tracker

    state = {
        "EVT-CLOSE-PENDING": {
            "event_id": "EVT-CLOSE-PENDING", "symbol": "BTC-USDT", "direction": "LONG",
            "entry_price": 100.0, "initial_qty": 1.0, "remaining_qty": 0.0,
            "realized_pnl_qty": 1.0, "exit_reason": "STOP_LOSS", "closed": False,
            "close_journal_pending": True, "close_notification_pending": False,
            "close_journal_snapshot": {
                "record_type": "TRADE_CLOSE", "event_id": "EVT-CLOSE-PENDING",
                "exit_reason": "STOP_LOSS", "exit_price_source": "SL_FILL",
            },
        },
    }
    path = tmp_path / "active_trades.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    journal = []
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", path)
    monkeypatch.setattr(tracker, "_recover_data_error_trade_closes", lambda: None)
    monkeypatch.setattr(tracker, "_recover_trade_cashflows", lambda **kwargs: None)
    monkeypatch.setattr(tracker, "_retry_pending_notifications", lambda: None)
    monkeypatch.setattr(tracker, "_close_record_exists", lambda event_id: False)
    monkeypatch.setattr(tracker, "_append_trade_record", lambda row: journal.append(copy.deepcopy(row)))
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: pytest.fail("must not re-read exchange position while close journal is pending"))

    tracker.update_active_trades()

    saved = json.loads(path.read_text(encoding="utf-8"))
    trade = saved["EVT-CLOSE-PENDING"]
    assert len(journal) == 1
    assert journal[0]["exit_reason"] == "STOP_LOSS"
    assert trade["close_journal_pending"] is False
    assert trade["close_notification_pending"] is True
    assert trade["closed"] is False
