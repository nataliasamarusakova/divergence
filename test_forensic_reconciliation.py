"""Focused regression tests for forensic trade recovery, candidate logging and protection audit."""
import json

import pytest


def test_fill_history_supports_bingx_tradeprice_volume_aliases(monkeypatch):
    import event_engine.bingx as bingx

    captured = {}

    def fake_request(method, path, params=None, **kwargs):
        captured.update(method=method, path=path, params=dict(params or {}), kwargs=kwargs)
        return {
            "code": 0,
            "data": {
                "fill_orders": [{
                    "tradeId": "fill-1", "orderId": "order-7", "symbol": "BTC-USDT",
                    "side": "SELL", "positionSide": "LONG", "tradePrice": "105.5",
                    "volume": "2", "tradeTime": 1_700_000_000,
                    "realizedPnl": "11", "fee": "-0.04",
                }],
                "total": 1,
            },
        }

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", fake_request)
    result = bingx.get_fill_history(
        "BTC", 1_699_999_999_000, 1_700_000_001_000,
        order_id="order-7", page_index=2, page_size=25,
    )

    assert result["status"] == "ok"
    assert result["returned"] == 1
    fill = result["fills"][0]
    assert fill["trade_id"] == "fill-1"
    assert fill["order_id"] == "order-7"
    assert fill["price"] == pytest.approx(105.5)
    assert fill["qty"] == pytest.approx(2.0)
    assert fill["time"] == 1_700_000_000_000  # seconds normalized to ms
    assert fill["realized_pnl"] == pytest.approx(11.0)
    assert fill["fee"] == pytest.approx(-0.04)
    assert captured["path"] == bingx.FILL_HISTORY_PATH
    assert captured["params"]["orderId"] == "order-7"
    assert captured["params"]["pageIndex"] == 2
    assert captured["kwargs"]["retryable"] is True


def test_official_fill_history_shape_uses_explicit_currency_filter_for_fee_asset(monkeypatch):
    import event_engine.bingx as bingx

    captured = {}
    def fake_request(method, path, params=None, **kwargs):
        captured.update(params=dict(params or {}))
        return {"code": 0, "data": {"fill_orders": [{
            "tradeId": "official-fill-1", "orderId": "official-order-1",
            "symbol": "BTC-USDT", "side": "SELL", "price": "99.5",
            "qty": "1", "time": 1_700_000_000_000,
            "realizedPnl": "0.5", "fee": "-0.01",
        }]}}

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", fake_request)
    result = bingx.get_fill_history(
        "BTC", 1_699_999_999_000, 1_700_000_001_000, currency="USDT",
    )
    assert result["status"] == "ok"
    assert captured["params"]["currency"] == "USDT"
    assert result["fills"][0]["fee_asset"] == "USDT"

    # Without an explicit currency filter, missing fee-asset metadata stays unknown.
    result_unfiltered = bingx.get_fill_history("BTC", 1_699_999_999_000, 1_700_000_001_000)
    assert result_unfiltered["fills"][0]["fee_asset"] is None


def test_fill_history_fails_closed_for_unknown_api_shape(monkeypatch):
    import event_engine.bingx as bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *args, **kwargs: {"code": 0, "data": {"unknownRows": []}})
    result = bingx.get_fill_history("BTC", 1_699_999_999_000, 1_700_000_001_000)

    assert result["status"] == "error"
    assert result["fills"] == []
    assert "unexpected_data_shape" in result["error"]


def _close_fixture(*, incomplete=False, malformed_row=False):
    event_id = "EVT_FORENSIC_TEST"
    open_ts = 1_700_000_000_000
    close_ts = open_ts + 60_000
    opened = {
        "record_type": "TRADE_OPEN", "event_id": event_id,
        "symbol": "BTC", "direction": "LONG", "ts": open_ts,
        "execution": {
            "actual_entry_price": 100.0, "actual_qty": 3.0,
            "order_id": "ENTRY-1", "fill_ts_ms": open_ts,
        },
        "result": {
            "fill_ts_ms": open_ts,
            "protection": {"tp_orders": [], "sl_result": {}},
        },
    }
    closed = {
        "record_type": "TRADE_CLOSE", "event_id": event_id,
        "symbol": "BTC", "direction": "LONG", "closed_ts": close_ts,
        "entry_price": 100.0, "exit_price": None, "exit_reason": "DATA_ERROR",
        "realized_pnl_pct": None, "total_closed_qty": 3.0, "initial_qty": 3.0,
        "entry_order_id": "ENTRY-1", "exit_order_ids": ["EARLY-CLOSE", "FINAL-MARKET"], "planned_risk_pct": 0.5,
    }
    fills = [
        {"trade_id": "entry-fill", "order_id": "ENTRY-1", "symbol": "BTC-USDT", "side": "BUY", "position_side": "LONG", "price": 100.0, "qty": 3.0, "time": open_ts + 1000, "realized_pnl": 0.0, "fee": -0.03, "fee_asset": "USDT"},
        {"trade_id": "close-fill-1", "order_id": "EARLY-CLOSE", "symbol": "BTC-USDT", "side": "SELL", "position_side": "LONG", "price": 102.0, "qty": 1.0, "time": close_ts - 10_000, "realized_pnl": 2.0, "fee": -0.01, "fee_asset": "USDT"},
        {"trade_id": "close-fill-2", "order_id": "FINAL-MARKET", "symbol": "BTC-USDT", "side": "SELL", "position_side": "LONG", "price": 103.0, "qty": 1.0, "time": close_ts - 2_000, "realized_pnl": 3.0, "fee": -0.01, "fee_asset": "USDT"},
    ]
    if not incomplete:
        # Same final market order, two fills with different timestamps. The final
        # order average must use both fills, not only the latest timestamp row.
        fills.append({"trade_id": "close-fill-3", "order_id": "FINAL-MARKET", "symbol": "BTC-USDT", "side": "SELL", "position_side": "LONG", "price": 104.0, "qty": 1.0, "time": close_ts - 1000, "realized_pnl": 4.0, "fee": -0.01, "fee_asset": "USDT"})
    return opened, closed, fills


def test_data_error_reconstruction_uses_full_final_order_average_and_preserves_cost_uncertainty():
    from event_engine.tracker import _rebuild_data_error_close

    opened, closed, fills = _close_fixture()
    rebuilt, error = _rebuild_data_error_close(closed, opened, fills)

    assert error is None
    assert rebuilt is not None
    assert rebuilt["total_close_fill_qty"] == pytest.approx(3.0)
    assert rebuilt["blended_exit_price"] == pytest.approx(103.0)
    assert rebuilt["exit_price"] == pytest.approx(103.5)  # average for both FINAL-MARKET fills
    assert rebuilt["exit_reason"] == "POSITION_CLOSED_BY_FILL_HISTORY"
    assert rebuilt["realized_pnl_pct"] == pytest.approx(3.0)
    assert rebuilt["realized_pnl_usdt"] == pytest.approx(9.0)
    assert rebuilt["fees_usdt"] == pytest.approx(-0.06)


def test_data_error_reconstruction_rejects_unlinked_symbol_time_quantity_coincidence():
    from event_engine.tracker import _rebuild_data_error_close

    opened, closed, fills = _close_fixture()
    closed.pop("entry_order_id", None)
    opened["execution"].pop("order_id", None)
    # No journaled exit IDs and no known protective order IDs. These rows match
    # symbol, side, time window, and total quantity but may belong to another trade.
    closed.pop("exit_order_ids", None)
    fills = [
        {"trade_id": "foreign-fill", "order_id": "FOREIGN-ORDER", "symbol": "BTC-USDT",
         "side": "SELL", "position_side": "LONG", "price": 90.0, "qty": 3.0,
         "time": closed["closed_ts"] - 1000, "realized_pnl": -30.0, "fee": -0.03, "fee_asset": "USDT"}
    ]
    rebuilt, error = _rebuild_data_error_close(closed, opened, fills)
    assert rebuilt is None
    assert error == "missing_entry_order_id_for_reconstruction"


def test_data_error_reconstruction_rejects_foreign_exit_fills_even_when_entry_id_exists():
    from event_engine.tracker import _rebuild_data_error_close

    opened, closed, fills = _close_fixture()
    closed.pop("exit_order_ids", None)
    fills = [fills[0], {
        "trade_id": "foreign-fill", "order_id": "FOREIGN-ORDER", "symbol": "BTC-USDT",
        "side": "SELL", "position_side": "LONG", "price": 90.0, "qty": 3.0,
        "time": closed["closed_ts"] - 1000, "realized_pnl": -30.0, "fee": -0.03, "fee_asset": "USDT",
    }]
    rebuilt, error = _rebuild_data_error_close(closed, opened, fills)
    assert rebuilt is None
    assert error == "missing_direct_exit_order_ids_for_reconstruction"


def test_data_error_reconstruction_does_not_fallback_to_foreign_fills_when_protection_id_exists():
    from event_engine.tracker import _rebuild_data_error_close

    opened, closed, fills = _close_fixture()
    opened["result"]["protection"]["sl_result"] = {"order_id": "SL-OWN-TRADE"}
    closed.pop("exit_order_ids", None)
    fills = [fills[0], {
        "trade_id": "foreign-fill", "order_id": "FOREIGN-ORDER", "symbol": "BTC-USDT",
        "side": "SELL", "position_side": "LONG", "price": 90.0, "qty": 3.0,
        "time": closed["closed_ts"] - 1000, "realized_pnl": -30.0, "fee": -0.03, "fee_asset": "USDT",
    }]
    rebuilt, error = _rebuild_data_error_close(closed, opened, fills)
    assert rebuilt is None
    assert error == "no_fills_for_journaled_exit_order_ids"


def test_data_error_reconstruction_uses_entry_order_fill_vwap_not_journal_estimate():
    from event_engine.tracker import _rebuild_data_error_close

    opened, closed, fills = _close_fixture()
    # Split the confirmed entry order into two partial fills at different prices.
    fills[0]["qty"] = 1.5
    fills[0]["price"] = 99.0
    second = dict(fills[0], trade_id="entry-fill-2", qty=1.5, price=101.0, time=fills[0]["time"] + 1)
    fills.insert(1, second)
    rebuilt, error = _rebuild_data_error_close(closed, opened, fills)
    assert error is None
    assert rebuilt is not None
    assert rebuilt["entry_price"] == pytest.approx(100.0)


def test_successful_recovery_persists_pnl_provenance_and_cost_uncertainty(tmp_path, monkeypatch):
    import event_engine.tracker as tracker

    journal = tmp_path / "trades.jsonl"
    opened, closed, fills = _close_fixture(incomplete=False)
    journal.write_text(json.dumps(opened) + "\n" + json.dumps(closed) + "\n", encoding="utf-8")
    monkeypatch.setattr(tracker, "TRADES_PATH", journal)
    query_kwargs = []
    def fake_fill_history(*args, **kwargs):
        query_kwargs.append(dict(kwargs))
        order_id = str(kwargs.get("order_id") or "")
        rows = [row for row in fills if row["order_id"] == order_id]
        return {"status": "ok", "fills": rows, "has_more": False}
    monkeypatch.setattr(tracker, "get_fill_history", fake_fill_history)
    monkeypatch.setattr(tracker, "get_order", lambda *_a, **_k: {"status": "error", "error": "not_available_in_test"})

    recovered = tracker._recover_data_error_trade_closes(now_ms=1_700_000_200_000)
    assert recovered == 1
    assert query_kwargs and query_kwargs[0].get("currency") == "USDT"
    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    close = next(row for row in rows if row.get("record_type") == "TRADE_CLOSE")
    assert close["realized_pnl_pct_source"] == "BINGX_FILL_HISTORY_PRICE_WEIGHTED"
    assert close["realized_pnl_usdt_source"] == "BINGX_FILL_REALIZED_PNL"
    assert close["fees_usdt"] == pytest.approx(-0.06)
    assert close["funding_usdt"] is None
    assert close["net_pnl_usdt"] is None
    assert close["pnl_costs_status"] == "FEES_RECONCILED_FUNDING_UNKNOWN"


def test_incomplete_data_error_fills_remain_unresolved_with_retry_metadata(tmp_path, monkeypatch):
    import event_engine.tracker as tracker

    journal = tmp_path / "trades.jsonl"
    opened, closed, fills = _close_fixture(incomplete=True)
    journal.write_text(json.dumps(opened) + "\n" + json.dumps(closed) + "\n", encoding="utf-8")
    monkeypatch.setattr(tracker, "TRADES_PATH", journal)
    def fake_fill_history(*args, **kwargs):
        order_id = str(kwargs.get("order_id") or "")
        return {"status": "ok", "fills": [row for row in fills if row["order_id"] == order_id], "has_more": False}
    monkeypatch.setattr(tracker, "get_fill_history", fake_fill_history)
    monkeypatch.setattr(tracker, "get_order", lambda *_a, **_k: {"status": "error", "error": "not_available_in_test"})

    recovered = tracker._recover_data_error_trade_closes(now_ms=1_700_000_200_000)
    assert recovered == 0
    result = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    close = next(row for row in result if row.get("record_type") == "TRADE_CLOSE")
    assert close["exit_reason"] == "DATA_ERROR"
    assert close["realized_pnl_pct"] is None
    assert close["fill_history_recovery"]["status"] == "PENDING"
    assert close["fill_history_recovery"]["last_error"].startswith("incomplete_close_fills")
    assert close["fill_history_recovery"]["next_retry_ts"] > 1_700_000_200_000


def test_recovery_preserves_unparseable_forensic_rows(tmp_path, monkeypatch):
    import event_engine.tracker as tracker

    journal = tmp_path / "trades.jsonl"
    opened, closed, fills = _close_fixture(incomplete=True)
    damaged = '{"malformed": not-json}\n'
    journal.write_text(json.dumps(opened) + "\n" + json.dumps(closed) + "\n" + damaged, encoding="utf-8")
    monkeypatch.setattr(tracker, "TRADES_PATH", journal)
    def fake_fill_history(*args, **kwargs):
        order_id = str(kwargs.get("order_id") or "")
        return {"status": "ok", "fills": [row for row in fills if row["order_id"] == order_id], "has_more": False}
    monkeypatch.setattr(tracker, "get_fill_history", fake_fill_history)
    monkeypatch.setattr(tracker, "get_order", lambda *_a, **_k: {"status": "error", "error": "not_available_in_test"})

    tracker._recover_data_error_trade_closes(now_ms=1_700_000_200_000)
    assert journal.read_text(encoding="utf-8").splitlines()[-1] == damaged.rstrip("\n")


def test_exact_sl_audit_rejects_duplicates_wrong_qty_and_unknown_query():
    from event_engine.bingx import _audit_exact_single_sl

    good = {"orderId": "SL-1", "type": "STOP_MARKET", "stopPrice": 95.0, "origQty": 2.0}
    passed, audit = _audit_exact_single_sl(
        {"status": "ok", "sl_orders": [good]}, direction="LONG", avg_price=100.0,
        expected_price=95.0, expected_qty=2.0,
    )
    assert passed is True
    assert audit["audit_reason"] == "OK"

    passed, audit = _audit_exact_single_sl(
        {"status": "ok", "sl_orders": [good, {**good, "orderId": "SL-2"}]},
        direction="LONG", avg_price=100.0, expected_price=95.0, expected_qty=2.0,
    )
    assert passed is False
    assert audit["audit_reason"] == "SL_MULTIPLE_ORDERS"

    passed, audit = _audit_exact_single_sl(
        {"status": "ok", "sl_orders": [{**good, "origQty": 1.5}]},
        direction="LONG", avg_price=100.0, expected_price=95.0, expected_qty=2.0,
    )
    assert passed is False
    assert audit["audit_reason"] == "SL_QTY_MISMATCH"

    passed, audit = _audit_exact_single_sl(
        {"status": "error", "error": "timeout", "sl_orders": []},
        direction="LONG", avg_price=100.0, expected_price=95.0, expected_qty=2.0,
    )
    assert passed is False
    assert audit["audit_reason"] == "OPEN_ORDERS_UNVERIFIABLE"


def test_candidate_decision_journal_is_json_safe_and_cycle_specific(tmp_path, monkeypatch):
    import run_once as ro

    journal = tmp_path / "actions.jsonl"
    monkeypatch.setattr(ro, "ACTIONS", journal)
    cid = ro._candidate_identifier("EVT-1", "BTC", "REGULAR_BULLISH_RSI", "LONG", 1234, 5678)
    assert cid != ro._candidate_identifier("EVT-1", "BTC", "REGULAR_BULLISH_RSI", "LONG", 1234, 5679)
    ro._record_candidate_decision(
        candidate_id=cid, event_id="EVT-1", symbol="BTC", direction="LONG",
        event_type="REGULAR_BULLISH_RSI", timeframe="1h", stage="FUNDING_FILTER",
        decision="REJECTED", reason="FUNDING_ABOVE_LIMIT", evaluation_ts=5678,
        features={"funding": 0.005, "non_finite": float("nan")},
        raw_market_features={"volume24": 123.0},
    )
    row = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
    assert row["record_type"] == "CANDIDATE_FILTER_DECISION"
    assert row["candidate_id"] == cid
    assert row["stage"] == "FUNDING_FILTER"
    assert row["decision"] == "REJECTED"
    assert row["reason"] == "FUNDING_ABOVE_LIMIT"
    assert row["features"]["non_finite"] is None
    assert row["raw_market_features"]["volume24"] == pytest.approx(123.0)


def test_reconciliation_logs_unknown_protection_query_and_does_not_guess(tmp_path, monkeypatch):
    import run_once as ro

    actions = []
    monkeypatch.setattr(ro, "get_positions", lambda **kwargs: [{
        "symbol": "BTC-USDT", "positionSide": "LONG", "positionAmt": "2", "avgPrice": "100",
    }])
    monkeypatch.setattr(ro, "_load_active_trades", lambda: {})
    monkeypatch.setattr(ro, "get_open_protection_directional", lambda *args, **kwargs: {
        "status": "error", "error": "simulated timeout", "sl_orders": [], "tp_orders": [],
    })
    monkeypatch.setattr(ro, "record_action", lambda row: actions.append(row))

    ro.reconcile_all_open_positions()

    assert len(actions) == 1
    assert actions[0]["record_type"] == "PROTECTION_STATE_UNKNOWN"
    assert actions[0]["reason"] == "OPEN_ORDERS_QUERY_FAILED"
    assert actions[0]["position_qty"] == pytest.approx(2.0)
    assert actions[0]["repair_attempted"] is False


def test_reconciliation_defers_repair_when_second_open_orders_query_is_unknown(monkeypatch):
    import run_once as ro

    actions = []
    responses = iter([
        {"status": "ok", "sl_orders": [], "tp_orders": []},
        {"status": "error", "error": "simulated recheck timeout", "sl_orders": [], "tp_orders": []},
    ])
    monkeypatch.setattr(ro, "get_positions", lambda **kwargs: [{
        "symbol": "BTC-USDT", "positionSide": "LONG", "positionAmt": "2", "avgPrice": "100",
    }])
    monkeypatch.setattr(ro, "_load_active_trades", lambda: {})
    monkeypatch.setattr(ro, "get_open_protection_directional", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(ro, "record_action", lambda row: actions.append(row))
    monkeypatch.setattr(ro, "ensure_directional_protection", lambda **kwargs: pytest.fail("must not repair on unknown recheck"))

    ro.reconcile_all_open_positions()

    assert any(row.get("record_type") == "PROTECTION_RECONCILIATION_MISMATCH" for row in actions)
    recheck = next(row for row in actions if row.get("record_type") == "PROTECTION_RECHECK_UNKNOWN")
    assert recheck["reason"] == "PRE_REPAIR_OPEN_ORDERS_QUERY_FAILED"
    assert recheck["repair_attempted"] is False


def test_stop_slippage_uses_filled_order_trigger_after_break_even_move():
    from event_engine.tracker import _calc_stop_slippage_pct, _stop_price_for_order

    trade = {
        "planned_invalidation_price": 107.0,
        "sl_order": {"order_id": "BE-STOP", "stop_price": 100.2},
        "sl_order_history": [{"order_id": "INITIAL-STOP", "stop_price": 107.0}],
    }

    # The final BE order, not the original 7%-away stop, is the right reference.
    stop_price = _stop_price_for_order(trade, "BE-STOP")
    assert stop_price == pytest.approx(100.2)
    assert _calc_stop_slippage_pct("SHORT", 100.5, stop_price) == pytest.approx((100.5 - 100.2) / 100.2 * 100.0)

    # When an older stop is the order actually found filled, use its historical price.
    old_stop_price = _stop_price_for_order(trade, "INITIAL-STOP")
    assert old_stop_price == pytest.approx(107.0)
    assert _calc_stop_slippage_pct("SHORT", 107.5, old_stop_price) == pytest.approx((107.5 - 107.0) / 107.0 * 100.0)


def test_stop_slippage_handles_direction_and_invalid_prices():
    from event_engine.tracker import _calc_stop_slippage_pct

    assert _calc_stop_slippage_pct("LONG", 94.0, 95.0) == pytest.approx((95.0 - 94.0) / 95.0 * 100.0)
    assert _calc_stop_slippage_pct("SHORT", 106.0, 105.0) == pytest.approx((106.0 - 105.0) / 105.0 * 100.0)
    # Better-than-trigger fills are not adverse slippage.
    assert _calc_stop_slippage_pct("LONG", 96.0, 95.0) == 0.0
    assert _calc_stop_slippage_pct("SHORT", 104.0, 105.0) == 0.0
    assert _calc_stop_slippage_pct("LONG", None, 95.0) is None
    assert _calc_stop_slippage_pct("SIDEWAYS", 95.0, 95.0) is None


def test_fill_history_uses_total_to_distinguish_full_final_page(monkeypatch):
    import event_engine.bingx as bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "BTC-USDT")
    rows = [
        {"tradeId": f"F-{i}", "orderId": "O-1", "symbol": "BTC-USDT", "side": "SELL",
         "positionSide": "LONG", "price": "100", "qty": "0.01", "time": 1_700_000_000_000,
         "realizedPnl": "0", "fee": "0", "feeAsset": "USDT"}
        for i in range(1000)
    ]
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": {"fill_orders": rows, "total": 1000}})
    result = bingx.get_fill_history("BTC", 1_699_999_999_000, 1_700_000_001_000, page_index=1, page_size=1000)
    assert result["status"] == "ok"
    assert result["returned"] == 1000
    assert result["total"] == 1000
    assert result["has_more"] is False

    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": {"fill_orders": rows, "total": 1001}})
    result = bingx.get_fill_history("BTC", 1_699_999_999_000, 1_700_000_001_000, page_index=1, page_size=1000)
    assert result["has_more"] is True


def test_data_error_reconstruction_recognizes_replaced_break_even_sl_from_journaled_history():
    from event_engine.tracker import _rebuild_data_error_close

    opened, closed, _ = _close_fixture()
    opened["result"]["protection"] = {
        "tp_orders": [{"leg": "tp1", "order_id": "TP1-OWN"}],
        "sl_result": {"order_id": "SL-ORIGINAL"},
    }
    closed.update({
        "tp_order_ids": ["TP1-OWN"],
        "sl_order_history": [{"order_id": "SL-BREAK-EVEN", "stop_price": 100.0}],
        "exit_order_ids": ["TP1-OWN", "SL-ORIGINAL", "SL-BREAK-EVEN"],
        "be_activated": True,
    })
    fills = [
        {"trade_id": "entry-1", "order_id": "ENTRY-1", "symbol": "BTC-USDT", "side": "BUY", "position_side": "LONG", "price": 100.0, "qty": 3.0, "time": opened["ts"] + 1000, "realized_pnl": 0.0, "fee": -0.03, "fee_asset": "USDT"},
        {"trade_id": "tp1-1", "order_id": "TP1-OWN", "symbol": "BTC-USDT", "side": "SELL", "position_side": "LONG", "price": 104.0, "qty": 1.0, "time": closed["closed_ts"] - 10_000, "realized_pnl": 4.0, "fee": -0.01, "fee_asset": "USDT"},
        {"trade_id": "sl-be-1", "order_id": "SL-BREAK-EVEN", "symbol": "BTC-USDT", "side": "SELL", "position_side": "LONG", "price": 100.0, "qty": 2.0, "time": closed["closed_ts"] - 1_000, "realized_pnl": 0.0, "fee": -0.02, "fee_asset": "USDT"},
    ]

    rebuilt, error = _rebuild_data_error_close(closed, opened, fills)

    assert error is None
    assert rebuilt is not None
    assert rebuilt["exit_reason"] == "BREAK_EVEN"
    assert rebuilt["exit_price"] == pytest.approx(100.0)
    assert rebuilt["total_close_fill_qty"] == pytest.approx(3.0)


def test_fill_history_accepts_observed_fill_history_orders_gateway_key(monkeypatch):
    """Regression for the exact top-level key present in the v13 recovery logs."""
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "PUMP-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {
        "code": 0,
        "msg": "",
        "data": {
            "fill_history_orders": [{
                "tradeId": "FILL-1", "orderId": "ORDER-1", "symbol": "PUMP-USDT",
                "side": "BUY", "positionSide": "SHORT", "tradePrice": "0.004091",
                "volume": "1261", "tradeTime": 1_793_890_936,
                "realizedPnl": "0.123", "fee": "-0.002", "feeAsset": "USDT",
            }],
            "total": 1,
        },
    })

    result = bingx.get_fill_history(
        "PUMP", 1_793_890_000_000, 1_793_891_000_000,
        order_id="ORDER-1", currency="USDT",
    )

    assert result["status"] == "ok"
    assert result["returned"] == 1
    assert result["has_more"] is False
    assert result["fills"][0]["order_id"] == "ORDER-1"
    assert result["fills"][0]["price"] == pytest.approx(0.004091)
    assert result["fills"][0]["qty"] == pytest.approx(1261.0)
    assert result["fills"][0]["fee"] == pytest.approx(-0.002)
    assert result["fills"][0]["fee_asset"] == "USDT"


def test_data_error_reconstruction_falls_back_to_exact_terminal_order_details(monkeypatch):
    from event_engine import tracker

    opened, closed, _fills = _close_fixture(incomplete=True)
    queried = []

    orders = {
        "ENTRY-1": {
            "status": "ok", "order_id": "ENTRY-1", "order_status": "FILLED",
            "avg_price": 100.0, "executed_qty": 3.0, "orig_qty": 3.0,
            "raw_order": {
                "orderId": "ENTRY-1", "symbol": "BTC-USDT", "positionSide": "LONG",
                "side": "BUY", "status": "FILLED", "time": 1_700_000_000_000,
            },
        },
        "EARLY-CLOSE": {
            "status": "ok", "order_id": "EARLY-CLOSE", "order_status": "CANCELED",
            "avg_price": 102.0, "executed_qty": 1.0, "orig_qty": 1.0,
            "raw_order": {
                "orderId": "EARLY-CLOSE", "symbol": "BTC-USDT", "positionSide": "LONG",
                "side": "SELL", "status": "CANCELED", "time": 1_700_000_050_000,
            },
        },
        "FINAL-MARKET": {
            "status": "ok", "order_id": "FINAL-MARKET", "order_status": "FILLED",
            "avg_price": 103.5, "executed_qty": 2.0, "orig_qty": 2.0,
            "raw_order": {
                "orderId": "FINAL-MARKET", "symbol": "BTC-USDT", "positionSide": "LONG",
                "side": "SELL", "status": "FILLED", "time": 1_700_000_059_000,
            },
        },
    }

    def fake_get_order(symbol, order_id):
        queried.append((symbol, str(order_id)))
        return orders[str(order_id)]

    monkeypatch.setattr(tracker, "get_order", fake_get_order)
    monkeypatch.setattr(tracker.time, "sleep", lambda *_: None)

    rebuilt, error = tracker._rebuild_data_error_close_from_order_details(closed, opened)

    assert error is None
    assert rebuilt is not None
    assert set(order_id for _, order_id in queried) == {"ENTRY-1", "EARLY-CLOSE", "FINAL-MARKET"}
    assert rebuilt["recovery_source"] == "BINGX_ORDER_DETAILS"
    assert rebuilt["total_close_fill_qty"] == pytest.approx(3.0)
    assert rebuilt["blended_exit_price"] == pytest.approx(103.0)
    assert rebuilt["realized_pnl_pct"] == pytest.approx(3.0)
    assert rebuilt["realized_pnl_usdt"] == pytest.approx(9.0)
    assert rebuilt["fees_usdt"] is None
    assert rebuilt["exit_reason"] == "POSITION_CLOSED_BY_ORDER_DETAILS"


def test_data_error_order_detail_fallback_rejects_nonterminal_or_incomplete_orders(monkeypatch):
    from event_engine import tracker

    opened, closed, _fills = _close_fixture(incomplete=True)
    calls = []

    def fake_get_order(symbol, order_id):
        calls.append(str(order_id))
        if str(order_id) == "ENTRY-1":
            return {
                "status": "ok", "order_id": "ENTRY-1", "order_status": "FILLED",
                "avg_price": 100.0, "executed_qty": 3.0, "orig_qty": 3.0,
                "raw_order": {"orderId": "ENTRY-1", "symbol": "BTC-USDT", "positionSide": "LONG", "side": "BUY"},
            }
        if str(order_id) == "EARLY-CLOSE":
            return {
                "status": "ok", "order_id": "EARLY-CLOSE", "order_status": "PARTIALLY_FILLED",
                "avg_price": 102.0, "executed_qty": 1.0, "orig_qty": 1.0,
                "raw_order": {"orderId": "EARLY-CLOSE", "symbol": "BTC-USDT", "positionSide": "LONG", "side": "SELL", "time": 1_700_000_050_000},
            }
        return {
            "status": "ok", "order_id": "FINAL-MARKET", "order_status": "FILLED",
            "avg_price": 103.5, "executed_qty": 2.0, "orig_qty": 2.0,
            "raw_order": {"orderId": "FINAL-MARKET", "symbol": "BTC-USDT", "positionSide": "LONG", "side": "SELL", "time": 1_700_000_059_000},
        }

    monkeypatch.setattr(tracker, "get_order", fake_get_order)
    monkeypatch.setattr(tracker.time, "sleep", lambda *_: None)

    rebuilt, error = tracker._rebuild_data_error_close_from_order_details(closed, opened)

    assert rebuilt is None
    assert error.startswith("exit_order_not_terminal:EARLY-CLOSE")
    assert calls == ["ENTRY-1", "EARLY-CLOSE"]


def test_data_error_order_detail_fallback_requires_exact_entry_order_id(monkeypatch):
    from event_engine import tracker

    opened, closed, _fills = _close_fixture(incomplete=True)
    opened["execution"].pop("order_id")
    closed.pop("entry_order_id", None)
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: pytest.fail("must not query unlinked order ids"))

    rebuilt, error = tracker._rebuild_data_error_close_from_order_details(closed, opened)

    assert rebuilt is None
    assert error == "missing_entry_order_id_for_reconstruction"


def test_close_notification_distinguishes_last_exit_execution_from_aggregate_pnl():
    from event_engine.tracker import format_trade_closed_message

    message = format_trade_closed_message(
        name="USELESS", symbol="USELESS-USDT", direction="SHORT",
        entry_price=0.08244, exit_price=0.08249, pnl_pct=1.213,
        realized_rr=0.173, planned_rr=1.2, duration_min=138.4,
        peak_pnl=4.38, max_drawdown=7.48, exit_reason="BREAK_EVEN",
        event_type="CRT_BEARISH", timeframe="1h",
    )

    assert "Последнее исполнение выхода" in message
    assert "Совокупный PnL сделки" in message
    assert "Вход <code>0.08244</code> → Выход" not in message


def test_data_error_recovery_uses_exact_order_details_when_fill_history_is_empty(tmp_path, monkeypatch):
    import event_engine.tracker as tracker

    journal = tmp_path / "trades.jsonl"
    opened, closed, _fills = _close_fixture(incomplete=True)
    journal.write_text(json.dumps(opened) + "\n" + json.dumps(closed) + "\n", encoding="utf-8")
    monkeypatch.setattr(tracker, "TRADES_PATH", journal)
    monkeypatch.setattr(tracker, "get_fill_history", lambda *_a, **_k: {
        "status": "ok", "fills": [], "has_more": False,
    })
    snapshots = {
        "ENTRY-1": {
            "status": "ok", "order_id": "ENTRY-1", "order_status": "FILLED",
            "avg_price": 100.0, "executed_qty": 3.0, "orig_qty": 3.0,
            "raw_order": {"orderId": "ENTRY-1", "symbol": "BTC-USDT", "positionSide": "LONG", "side": "BUY"},
        },
        "EARLY-CLOSE": {
            "status": "ok", "order_id": "EARLY-CLOSE", "order_status": "CANCELED",
            "avg_price": 102.0, "executed_qty": 1.0, "orig_qty": 1.0,
            "raw_order": {"orderId": "EARLY-CLOSE", "symbol": "BTC-USDT", "positionSide": "LONG", "side": "SELL", "time": 1_700_000_050_000},
        },
        "FINAL-MARKET": {
            "status": "ok", "order_id": "FINAL-MARKET", "order_status": "FILLED",
            "avg_price": 103.5, "executed_qty": 2.0, "orig_qty": 2.0,
            "raw_order": {"orderId": "FINAL-MARKET", "symbol": "BTC-USDT", "positionSide": "LONG", "side": "SELL", "time": 1_700_000_059_000},
        },
    }
    monkeypatch.setattr(tracker, "get_order", lambda _symbol, order_id: snapshots[str(order_id)])
    monkeypatch.setattr(tracker.time, "sleep", lambda *_a: None)

    recovered = tracker._recover_data_error_trade_closes(now_ms=1_700_000_200_000, max_records=1)

    rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    close = next(row for row in rows if row.get("record_type") == "TRADE_CLOSE")
    assert recovered == 1
    assert close["fill_history_recovery"]["status"] == "RECOVERED"
    assert close["fill_history_recovery"]["source"] == "BINGX_ORDER_DETAILS"
    assert close["exit_price_source"] == "BINGX_ORDER_DETAILS_FINAL_ORDER_AVG_PRICE"
    assert close["realized_pnl_pct_source"] == "BINGX_ORDER_DETAILS_PRICE_WEIGHTED_GROSS"
    assert close["fees_usdt"] is None and close["funding_usdt"] is None
    assert close["net_pnl_usdt"] is None
    assert close["pnl_costs_status"] == "GROSS_ORDER_DETAILS_CONFIRMED_FEES_UNKNOWN_FUNDING_UNKNOWN"
