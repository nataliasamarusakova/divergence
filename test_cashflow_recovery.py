"""Regression tests for exchange fill and account-income reconciliation."""
import json
from pathlib import Path

import pytest


def test_fill_history_accepts_trade_id_fee_and_profit_aliases(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "ETH-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {
        "code": 0,
        "data": {"result": {"data": {"rows": [{
            "tradeID": "F-1", "orderID": "O-2", "symbol": "ETH-USDT",
            "side": "SELL", "position_side": "LONG", "dealPrice": "2500.5",
            "dealVolume": "0.4", "timestamp": 1_700_000_001,
            "realized_profit": "3.2", "tradeFee": "-0.03", "feeCurrency": "USDT",
        }]}}},
    })

    result = bingx.get_fill_history("ETH", 1_700_000_000_000, 1_700_000_002_000)
    assert result["status"] == "ok"
    fill = result["fills"][0]
    assert fill["trade_id"] == "F-1"
    assert fill["order_id"] == "O-2"
    assert fill["price"] == pytest.approx(2500.5)
    assert fill["qty"] == pytest.approx(0.4)
    assert fill["time"] == 1_700_000_001_000
    assert fill["realized_pnl"] == pytest.approx(3.2)
    assert fill["fee"] == pytest.approx(-0.03)
    assert fill["fee_asset"] == "USDT"


def test_fill_history_keeps_unknown_arbitrary_list_key_fail_closed(monkeypatch, caplog):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": {"mysteryRows": []}})
    result = bingx.get_fill_history("BTC", 1_700_000_000_000, 1_700_000_002_000)
    assert result["status"] == "error"
    assert result["fills"] == []
    assert result["shape_fingerprint"]
    assert "fingerprint=" in caplog.text


def test_income_history_normalizes_actual_signed_cashflows(monkeypatch):
    from event_engine import bingx

    captured = {}
    def fake_request(method, path, params=None, **kwargs):
        captured.update(method=method, path=path, params=dict(params or {}), kwargs=kwargs)
        return {"code": 0, "data": [
            {"tranId": "R1", "tradeId": "T1", "symbol": "BTC-USDT", "incomeType": "REALIZED_PNL", "income": "5.0", "asset": "USDT", "time": 1_700_000_001_000},
            {"tranId": "F1", "tradeId": "T2", "symbol": "BTC-USDT", "incomeType": "TRADING_FEE", "income": "-0.2", "asset": "USDT", "time": 1_700_000_002_000},
            {"tranId": "U1", "tradeId": "T3", "symbol": "BTC-USDT", "incomeType": "FUNDING_FEE", "income": "0.05", "asset": "USDT", "time": 1_700_000_003_000},
        ]}
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", fake_request)

    result = bingx.get_income_history("BTC", 1_700_000_000_000, 1_700_000_004_000)
    assert result["status"] == "ok" and result["complete"] is True
    assert [r["income_type"] for r in result["records"]] == ["REALIZED_PNL", "TRADING_FEE", "FUNDING_FEE"]
    assert [r["amount"] for r in result["records"]] == pytest.approx([5.0, -0.2, 0.05])
    assert result["records"][0]["order_id"] == ""
    assert captured["path"] == bingx.INCOME_HISTORY_PATH
    assert captured["params"]["startTime"] == 1_700_000_000_000
    assert captured["kwargs"]["retryable"] is False


def _journal_fixture(tmp_path: Path, *, overlapping: bool = False):
    journal = tmp_path / "trades.jsonl"
    open_ts = 1_700_000_000_000
    close_ts = open_ts + 60 * 60_000
    rows = [
        {"record_type": "TRADE_OPEN", "event_id": "EVT_A", "symbol": "BTC", "direction": "LONG", "ts": open_ts,
         "execution": {"fill_ts_ms": open_ts, "order_id": "ENTRY-1"},
         "result": {"protection": {"sl_result": {"order_id": "SL-1"}, "tp_orders": [{"order_id": "TP-1"}]}},
         "protection": {"sl_result": {"order_id": "SL-1"}, "tp_orders": [{"order_id": "TP-1"}]}},
        {"record_type": "TRADE_CLOSE", "event_id": "EVT_A", "symbol": "BTC", "direction": "LONG", "closed_ts": close_ts,
         "exit_reason": "STOP_LOSS", "realized_pnl_usdt": -1.0,
         "realized_pnl_usdt_source": "TRACKER_DERIVED_FROM_WEIGHTED_PNL_EXCLUDING_COSTS",
         "fees_usdt": None, "funding_usdt": None, "net_pnl_usdt": None,
         "filled_sl_order_id": "SL-1", "exit_order_ids": ["SL-1"],
         "pnl_costs_status": "FEES_AND_FUNDING_NOT_ATTRIBUTED"},
    ]
    if overlapping:
        rows.extend([
            {"record_type": "TRADE_OPEN", "event_id": "EVT_B", "symbol": "BTC-USDT", "direction": "SHORT", "ts": open_ts + 10 * 60_000,
             "execution": {"fill_ts_ms": open_ts + 10 * 60_000, "order_id": "ENTRY-2"}},
            {"record_type": "TRADE_CLOSE", "event_id": "EVT_B", "symbol": "BTC-USDT", "direction": "SHORT", "closed_ts": close_ts + 10 * 60_000},
        ])
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return journal, open_ts, close_ts


def test_cashflow_recovery_uses_actual_income_and_sets_confirmed_net(tmp_path, monkeypatch):
    from event_engine import tracker

    journal, open_ts, close_ts = _journal_fixture(tmp_path)
    monkeypatch.setattr(tracker, "TRADES_PATH", journal)
    fills = {
        "ENTRY-1": [{"trade_id": "E-FILL", "order_id": "ENTRY-1", "symbol": "BTC-USDT", "position_side": "LONG",
                     "time": open_ts + 1000, "qty": 1.0, "price": 100.0, "realized_pnl": 0.0, "fee": -0.04, "fee_asset": "USDT"}],
        "SL-1": [{"trade_id": "X-FILL", "order_id": "SL-1", "symbol": "BTC-USDT", "position_side": "LONG",
                  "time": close_ts - 1000, "qty": 1.0, "price": 98.8, "realized_pnl": -1.2, "fee": -0.06, "fee_asset": "USDT"}],
        "TP-1": [],
    }
    monkeypatch.setattr(tracker, "get_fill_history", lambda symbol, *a, order_id=None, **k: {
        "status": "ok", "has_more": False, "fills": fills.get(str(order_id), []),
    })
    monkeypatch.setattr(tracker, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "source": "BINGX_USER_INCOME", "records": [
            {"record_id": "R1", "trade_id": "X-FILL", "symbol": "BTC-USDT", "income_type": "REALIZED_PNL", "amount": -1.2, "asset": "USDT", "time": close_ts - 1000},
            {"record_id": "F1", "trade_id": "E-FILL", "symbol": "BTC-USDT", "income_type": "TRADING_FEE", "amount": -0.04, "asset": "USDT", "time": open_ts + 1000},
            {"record_id": "F2", "trade_id": "X-FILL", "symbol": "BTC-USDT", "income_type": "TRADING_FEE", "amount": -0.06, "asset": "USDT", "time": close_ts - 500},
            {"record_id": "U1", "trade_id": "E-FILL", "symbol": "BTC-USDT", "income_type": "FUNDING_FEE", "amount": 0.01, "asset": "USDT", "time": open_ts + 20 * 60_000},
        ]})

    assert tracker._recover_trade_cashflows(now_ms=close_ts + 10_000) == 1
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    close = next(row for row in rows if row.get("record_type") == "TRADE_CLOSE" and row.get("event_id") == "EVT_A")
    assert close["realized_pnl_usdt"] == pytest.approx(-1.2)
    assert close["realized_pnl_usdt_source"] == "BINGX_ORDER_LINKED_FILL_HISTORY"
    assert close["fees_usdt"] == pytest.approx(-0.1)
    assert close["funding_usdt"] == pytest.approx(0.01)
    assert close["net_pnl_usdt"] == pytest.approx(-1.29)
    assert close["pnl_costs_status"] == "BINGX_ORDER_LINKED_FILLS_AND_INCOME_CONFIRMED"
    assert close["net_pnl_attribution_status"] == "EXACT_EXCHANGE_ORDER_FILL_IDS_MATCHED_TO_ACCOUNT_INCOME_TRADE_IDS"
    assert close["cashflow_recovery"]["status"] == "RECOVERED"


def test_cashflow_recovery_refuses_symbol_window_only_without_order_ids(tmp_path, monkeypatch):
    from event_engine import tracker

    journal, open_ts, close_ts = _journal_fixture(tmp_path)
    rows = [json.loads(x) for x in journal.read_text(encoding="utf-8").splitlines()]
    rows[0]["execution"].pop("order_id", None)
    rows[0].pop("result", None)
    rows[0].pop("protection", None)
    rows[1].pop("filled_sl_order_id", None)
    rows[1].pop("exit_order_ids", None)
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(tracker, "TRADES_PATH", journal)
    monkeypatch.setattr(tracker, "get_income_history", lambda *a, **k: pytest.fail("income totals must not be assigned without order IDs"))
    monkeypatch.setattr(tracker, "get_fill_history", lambda *a, **k: pytest.fail("fill history must not be queried without order IDs"))

    assert tracker._recover_trade_cashflows(now_ms=close_ts + 10_000) == 0
    output = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    close = output[1]
    assert close["net_pnl_usdt"] is None
    assert close["fees_usdt"] is None and close["funding_usdt"] is None
    assert close["pnl_costs_status"] == "UNVERIFIED_POSITION_ATTRIBUTION"
    assert close["realized_pnl_usdt"] is None
    assert close["estimated_gross_pnl_usdt"] == pytest.approx(-1.0)
    assert close["cashflow_recovery"]["status"] == "ABANDONED_NO_DIRECT_ORDER_IDS"



def test_legacy_symbol_window_pnl_is_demoted_even_when_recovery_cannot_start(tmp_path, monkeypatch):
    from event_engine import tracker

    journal, open_ts, close_ts = _journal_fixture(tmp_path)
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    close = rows[1]
    close.update({
        "realized_pnl_usdt": -4.5,
        "realized_pnl_usdt_source": "BINGX_ACCOUNT_INCOME_SYMBOL_WINDOW",
        "fees_usdt": -0.08,
        "funding_usdt": 0.02,
        "net_pnl_usdt": -4.56,
        "other_cashflows_usdt": 0.0,
        "pnl_costs_status": "BINGX_SYMBOL_WINDOW_CASHFLOWS_ATTRIBUTED",
        "cashflow_recovery": {"status": "RECOVERED"},
    })
    rows[0]["execution"].pop("order_id", None)
    rows[0].pop("result", None)
    rows[0].pop("protection", None)
    close.pop("filled_sl_order_id", None)
    close.pop("exit_order_ids", None)
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(tracker, "TRADES_PATH", journal)
    monkeypatch.setattr(tracker, "get_income_history", lambda *a, **k: pytest.fail("must not query aggregate income without direct order IDs"))
    monkeypatch.setattr(tracker, "get_fill_history", lambda *a, **k: pytest.fail("must not query fills without direct order IDs"))

    assert tracker._recover_trade_cashflows(now_ms=close_ts + 10_000) == 0
    output = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    saved = output[1]
    assert saved["realized_pnl_usdt"] is None
    assert saved["realized_pnl_usdt_source"] == "UNATTRIBUTED_SYMBOL_WINDOW_AGGREGATE"
    assert saved["unattributed_symbol_window_realized_pnl_usdt"] == pytest.approx(-4.5)
    assert saved["fees_usdt"] is None and saved["funding_usdt"] is None and saved["net_pnl_usdt"] is None
    assert saved["pnl_costs_status"] == "UNVERIFIED_POSITION_ATTRIBUTION"
    assert saved["cashflow_recovery"]["status"] == "ABANDONED_NO_DIRECT_ORDER_IDS"

def test_cashflow_recovery_refuses_overlapping_same_symbol_positions(tmp_path, monkeypatch):
    from event_engine import tracker

    journal, open_ts, close_ts = _journal_fixture(tmp_path, overlapping=True)
    monkeypatch.setattr(tracker, "TRADES_PATH", journal)
    calls = []
    monkeypatch.setattr(tracker, "get_income_history", lambda *a, **k: calls.append((a, k)) or {"status": "ok", "complete": True, "records": []})

    assert tracker._recover_trade_cashflows(now_ms=close_ts + 10_000) == 0
    assert calls == []
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    first_close = next(row for row in rows if row.get("record_type") == "TRADE_CLOSE" and row.get("event_id") == "EVT_A")
    assert first_close["cashflow_recovery"]["status"] == "ABANDONED_AMBIGUOUS_SYMBOL_OVERLAP"


def test_stop_trigger_price_is_never_used_as_fill_price(monkeypatch):
    from event_engine import tracker

    trade = {"sl_order": {"order_id": "SL-1", "stop_price": 95.0}, "sl_order_history": []}
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "TRIGGERED", "avg_price": 0.0, "trigger_price": 95.0,
    })
    monkeypatch.setattr(tracker, "get_fill_history", lambda *a, **k: {"status": "ok", "has_more": False, "fills": []})
    assert tracker._get_filled_sl_from_trade("BTC", trade) == (None, None)

    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "FILLED", "avg_price": 94.72, "executed_qty": 1.0, "trigger_price": 95.0,
    })
    assert tracker._get_filled_sl_from_trade("BTC", trade) == (94.72, "SL-1")

    # FILLED + average price without executed quantity is not complete execution proof.
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "FILLED", "avg_price": 94.72, "trigger_price": 95.0,
    })
    monkeypatch.setattr(tracker, "get_fill_history", lambda *a, **k: {"status": "ok", "has_more": False, "fills": []})
    assert tracker._get_filled_sl_from_trade("BTC", trade) == (None, None)


def test_filled_stop_uses_actual_fill_history_vwap_and_complete_qty_not_trigger_price(monkeypatch):
    from event_engine import tracker

    trade = {"direction": "LONG", "entry_ts": 1_700_000_000_000, "sl_order": {"order_id": "SL-2", "stop_price": 95.0}}
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "FILLED", "avg_price": 0.0, "executed_qty": 4.0, "trigger_price": 95.0,
    })
    import time
    now_ms = int(time.time() * 1000)
    monkeypatch.setattr(tracker, "get_fill_history", lambda *a, **k: {
        "status": "ok", "has_more": False, "fills": [
            {"order_id": "SL-2", "trade_id": "FILL-1", "position_side": "LONG", "time": now_ms - 1000, "price": 94.0, "qty": 1.0},
            {"order_id": "SL-2", "trade_id": "FILL-2", "position_side": "LONG", "time": now_ms - 999, "price": 94.5, "qty": 3.0},
        ],
    })
    assert tracker._get_filled_sl_from_trade("BTC", trade) == pytest.approx((94.375, "SL-2"))


def test_filled_stop_missing_average_recovers_only_from_identified_actual_fills(monkeypatch):
    from event_engine import tracker
    import time

    now_ms = int(time.time() * 1000)
    trade = {"direction": "SHORT", "entry_ts": now_ms - 1000, "sl_order": {"order_id": "SL-3", "stop_price": 105.0}}
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "FILLED", "avg_price": 0.0, "executed_qty": 2.0, "trigger_price": 105.0,
    })
    monkeypatch.setattr(tracker, "get_fill_history", lambda *a, **k: {
        "status": "ok", "has_more": False, "fills": [
            {"order_id": "SL-3", "tradeId": "FILL-3", "positionSide": "SHORT", "time": now_ms - 500, "price": 103.0, "qty": 2.0},
        ],
    })
    assert tracker._get_filled_sl_from_trade("BTC", trade) == (103.0, "SL-3")


def test_stop_fill_rows_without_trade_identity_are_rejected(monkeypatch):
    from event_engine import tracker
    import time

    now_ms = int(time.time() * 1000)
    trade = {"direction": "LONG", "entry_ts": now_ms - 1000, "sl_order": {"order_id": "SL-4", "stop_price": 95.0}}
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "TRIGGERED", "avg_price": 0.0, "trigger_price": 95.0,
    })
    monkeypatch.setattr(tracker, "get_fill_history", lambda *a, **k: {
        "status": "ok", "has_more": False, "fills": [
            {"order_id": "SL-4", "positionSide": "LONG", "time": now_ms - 500, "price": 94.0, "qty": 1.0},
        ],
    })
    assert tracker._get_filled_sl_from_trade("BTC", trade) == (None, None)


def test_partial_stop_with_actual_fills_is_not_accepted_as_full_exit(monkeypatch):
    from event_engine import tracker

    trade = {"direction": "LONG", "entry_ts": 1_700_000_000_000, "sl_order": {"order_id": "SL-PARTIAL", "stop_price": 95.0}}
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "PARTIALLY_FILLED", "avg_price": 94.0,
        "executed_qty": 1.0, "orig_qty": 4.0, "trigger_price": 95.0,
    })
    monkeypatch.setattr(tracker, "get_fill_history", lambda *a, **k: pytest.fail("partial stop must not be used as a complete exit"))
    assert tracker._get_filled_sl_from_trade("BTC", trade) == (None, None)


def test_filled_stop_rejects_fill_history_whose_quantity_does_not_match_order(monkeypatch):
    from event_engine import tracker
    import time

    now_ms = int(time.time() * 1000)
    trade = {"direction": "LONG", "entry_ts": now_ms - 1000, "sl_order": {"order_id": "SL-SHORT-FILL", "stop_price": 95.0}}
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "FILLED", "avg_price": 0.0, "executed_qty": 2.0, "trigger_price": 95.0,
    })
    monkeypatch.setattr(tracker, "get_fill_history", lambda *a, **k: {
        "status": "ok", "has_more": False, "fills": [
            {"order_id": "SL-SHORT-FILL", "trade_id": "FILL-ONLY-1", "positionSide": "LONG", "time": now_ms - 500, "price": 94.0, "qty": 1.0},
        ],
    })
    assert tracker._get_filled_sl_from_trade("BTC", trade) == (None, None)

def test_hedge_close_mode_guard_fails_before_post_if_config_conflicts(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "POSITION_MODE", "ONE_WAY")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: pytest.fail("must not post under mismatched mode"))
    result = bingx.emergency_close_position("BTC", "LONG", 0.01, reason_token="UNIT")
    assert result["status"] == "error"
    assert "unsupported_position_mode_for_directional_close" in result["error"]


def test_active_trade_not_closed_after_only_one_missing_position_snapshot(tmp_path, monkeypatch):
    from event_engine import tracker

    active_path = tmp_path / "active_trades.json"
    trades_path = tmp_path / "trades.jsonl"
    notifications_path = tmp_path / "notifications.json"
    active_path.write_text(json.dumps({"EVT_A": {
        "event_id": "EVT_A", "symbol": "BTC", "direction": "LONG",
        "entry_price": 100.0, "initial_qty": 1.0, "remaining_qty": 1.0,
        "entry_ts": 1_700_000_000_000, "hit_legs": [], "tp_filled_qty": {},
        "realized_pnl_qty": 0.0, "realized_pnl_weighted_sum": 0.0,
        "tp_orders": [], "sl_order": {}, "sl_order_history": [], "closed": False,
    }}), encoding="utf-8")
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tracker, "TRADES_PATH", trades_path)
    monkeypatch.setattr(tracker, "NOTIFICATIONS_PATH", notifications_path)
    states = iter([
        {"status": "not_found", "positionAmt": 0},
        {"status": "found", "positionAmt": 1.0, "avgPrice": 100.0},
    ])
    calls = []
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: calls.append(1) or next(states))
    monkeypatch.setattr(tracker.time, "sleep", lambda *_: None)

    tracker.update_active_trades()

    assert len(calls) == 2
    saved = json.loads(active_path.read_text(encoding="utf-8"))
    assert saved["EVT_A"]["closed"] is False
    assert saved["EVT_A"]["remaining_qty"] == pytest.approx(1.0)
    assert saved["EVT_A"]["position_gone_confirmation"]["status"] == "PENDING"
    assert not trades_path.exists()


def test_income_history_bounds_recursive_queries_when_range_is_saturated(monkeypatch):
    from event_engine import bingx

    calls = []
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: calls.append(1) or {
        "code": 0, "data": [{
            "tranId": "same", "symbol": "BTC-USDT", "incomeType": "REALIZED_PNL",
            "income": "1", "asset": "USDT", "time": 1_700_000_000_000,
        }],
    })

    result = bingx.get_income_history("BTC", 1_700_000_000_000, 1_800_000_000_000, limit=1)
    assert result["status"] == "error"
    assert result["complete"] is False
    assert result["error"] in {
        "income_history_query_budget_exhausted",
        "income_history_range_still_truncated_after_time_split",
    }
    assert 1 <= len(calls) <= 64


def test_emergency_close_fails_closed_when_exchange_reports_one_way(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "POSITION_MODE", "HEDGE")
    requests = []
    def fake_request(method, path, params=None, **kwargs):
        requests.append((method, path))
        if path == bingx.POSITION_MODE_PATH:
            return {"code": 0, "data": {"dualSidePosition": False}}
        pytest.fail("close POST must not be sent if exchange mode is One-way")
    monkeypatch.setattr(bingx, "_request", fake_request)
    result = bingx.emergency_close_position("BTC", "LONG", 0.01, reason_token="UNIT")
    assert result["status"] == "error"
    assert "exchange_position_mode_mismatch_for_directional_close" in result["error"]
    assert requests == [("GET", bingx.POSITION_MODE_PATH)]


def test_position_mode_query_normalizes_exchange_dual_side_flag(monkeypatch):
    from event_engine import bingx

    captured = []
    monkeypatch.setattr(bingx, "_request", lambda method, path, params=None, **kwargs: captured.append((method, path, params, kwargs)) or {
        "code": 0, "data": {"dualSidePosition": True},
    })
    result = bingx.get_position_mode()
    assert result["status"] == "ok"
    assert result["mode"] == "HEDGE"
    assert result["dualSidePosition"] is True
    assert captured[0][0:2] == ("GET", bingx.POSITION_MODE_PATH)


def test_position_mode_query_fails_closed_on_unknown_flag(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": {"dualSidePosition": "unknown"}})
    result = bingx.get_position_mode()
    assert result["status"] == "error"
    assert "invalid_dualSidePosition" in result["error"]


def test_income_history_background_request_budget_fails_closed(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": [{"id": "one"}]})
    result = bingx.get_income_history(
        "BTC", 1_700_000_000_000, 1_700_000_100_000,
        limit=1, timeout_sec=0.1, max_range_requests=1,
    )
    assert result["status"] == "error"
    assert result["complete"] is False
    assert result["error"] == "income_history_query_budget_exhausted"


def test_income_history_accepts_known_nested_income_wrappers(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {
        "code": 0,
        "data": {"incomeHistory": {"result": {"rows": [
            {"tranId": "NESTED-1", "tradeId": "T-NESTED", "symbol": "BTC-USDT",
             "incomeType": "TRADING_FEE", "income": "-0.03", "asset": "USDT", "time": 1_700_000_002_000}
        ], "total": 1}}},
    })
    result = bingx.get_income_history("BTC", 1_700_000_000_000, 1_700_000_003_000)
    assert result["status"] == "ok" and result["complete"] is True
    assert len(result["records"]) == 1
    assert result["records"][0]["amount"] == pytest.approx(-0.03)


def test_income_history_unknown_shape_returns_safe_diagnostics_not_zero(monkeypatch, caplog):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {
        "code": 0, "data": {"responseMeta": {"status": "success", "payloadKind": "opaque"}}
    })
    result = bingx.get_income_history("BTC", 1_700_000_000_000, 1_700_000_003_000)
    assert result["status"] == "error"
    assert result["complete"] is False
    assert result["records"] == []
    assert result["error"] == "income_history_unexpected_data_shape"
    assert result.get("shape_fingerprint")
    assert "responseMeta" in result.get("shape_summary", "")
    assert "fingerprint=" in caplog.text
