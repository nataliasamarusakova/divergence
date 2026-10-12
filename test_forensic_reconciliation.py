import json
from pathlib import Path

import pytest


def test_bingx_fill_history_normalizes_execution_rows_and_paginates(monkeypatch):
    import event_engine.bingx as bingx

    seen = {}

    def fake_request(method, path, params=None, *, signed=True, timeout_sec=None, retryable=True):
        seen.update(method=method, path=path, params=dict(params or {}), signed=signed, retryable=retryable)
        return {
            "code": 0,
            "data": {
                "fill_orders": [{
                    "tradeId": "fill-1", "orderId": "order-7", "symbol": "BTC-USDT",
                    "side": "SELL", "positionSide": "LONG", "tradePrice": "105.5",
                    "volume": "2", "tradeTime": 1700000000123, "realizedPnl": "11", "commission": "-0.04",
                }],
                "total": 1,
            },
        }

    monkeypatch.setattr(bingx, "_request", fake_request)
    result = bingx.get_fill_history("BTC", 1699999999000, 1700000001000, order_id="order-7", page_index=2, page_size=25)

    assert result["status"] == "ok"
    assert result["returned"] == 1
    assert result["fills"][0]["trade_id"] == "fill-1"
    assert result["fills"][0]["order_id"] == "order-7"
    assert result["fills"][0]["qty"] == 2.0
    assert result["fills"][0]["price"] == 105.5
    assert result["fills"][0]["realized_pnl"] == 11.0
    assert result["fills"][0]["fee"] == -0.04
    assert seen["path"] == bingx.FILL_HISTORY_PATH
    assert seen["params"]["orderId"] == "order-7"
    assert seen["params"]["pageIndex"] == 2
    assert seen["retryable"] is False


def test_bingx_fill_history_fails_closed_on_unknown_response_shape(monkeypatch):
    import event_engine.bingx as bingx

    monkeypatch.setattr(bingx, "_request", lambda *args, **kwargs: {"code": 0, "data": {"mystery": []}})
    result = bingx.get_fill_history("BTC", 1699999999000, 1700000001000)

    assert result["status"] == "error"
    assert "unexpected data shape" in result["error"]
    assert result["fills"] == []


def _trade_journal_fixture(path: Path, *, complete=True, include_malformed=False):
    event_id = "EVT_FILL_RECOVERY"
    opened_ts = 1_699_999_900_000
    closed_ts = 1_700_000_000_000
    opened = {
        "record_type": "TRADE_OPEN", "event_id": event_id, "symbol": "BTC",
        "direction": "LONG", "ts": opened_ts,
        "execution": {"actual_entry_price": 100.0, "actual_qty": 2.0, "order_id": "ENTRY-1", "fill_ts_ms": opened_ts},
        "protection": {"tp_orders": [{"order_id": "TP-1"}], "sl_result": {"order_id": "SL-1"}},
    }
    closed = {
        "record_type": "TRADE_CLOSE", "event_id": event_id, "symbol": "BTC",
        "direction": "LONG", "closed_ts": closed_ts, "entry_price": 100.0,
        "exit_price": None, "exit_reason": "DATA_ERROR", "realized_pnl_pct": None,
        "total_closed_qty": 2.0, "initial_qty": 2.0, "entry_order_id": "ENTRY-1",
        "tp_order_ids": ["TP-1"], "sl_order_id": "SL-1", "planned_risk_pct": 0.5,
    }
    fills = [
        {"trade_id": "entry-fill", "order_id": "ENTRY-1", "symbol": "BTC-USDT", "side": "BUY", "position_side": "LONG", "price": 100.0, "qty": 2.0, "time": opened_ts + 1000, "realized_pnl": 0.0, "fee": -0.01},
        {"trade_id": "tp-fill", "order_id": "TP-1", "symbol": "BTC-USDT", "side": "SELL", "position_side": "LONG", "price": 102.0, "qty": 1.0, "time": closed_ts - 30_000, "realized_pnl": 2.0, "fee": -0.01},
    ]
    if complete:
        fills.append({"trade_id": "sl-fill", "order_id": "SL-1", "symbol": "BTC-USDT", "side": "SELL", "position_side": "LONG", "price": 104.0, "qty": 1.0, "time": closed_ts - 1000, "realized_pnl": 4.0, "fee": -0.01})
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(opened) + "\n")
        stream.write(json.dumps(closed) + "\n")
        if include_malformed:
            stream.write('{"malformed": not-json}\n')
    return event_id, fills


def test_data_error_trade_recovery_uses_weighted_exchange_fills_and_preserves_bad_rows(tmp_path, monkeypatch):
    import event_engine.tracker as tracker

    journal = tmp_path / "trades.jsonl"
    event_id, fills = _trade_journal_fixture(journal, complete=True, include_malformed=True)
    original_malformed = '{"malformed": not-json}\n'
    monkeypatch.setattr(tracker, "TRADES_PATH", journal)
    monkeypatch.setattr(tracker, "get_fill_history", lambda *args, **kwargs: {"status": "ok", "fills": fills})

    recovered = tracker._recover_data_error_trade_closes(now_ms=1_700_000_010_000)

    assert recovered == 1
    rows = journal.read_text(encoding="utf-8").splitlines()
    close = next(json.loads(line) for line in rows if '"record_type": "TRADE_CLOSE"' in line)
    assert close["event_id"] == event_id
    assert close["exit_reason"] == "STOP_LOSS"
    assert close["exit_price_source"] == "BINGX_FILL_HISTORY_WEIGHTED_AVERAGE"
    assert close["exit_price"] == pytest.approx(103.0)
    assert close["exit_execution_qty"] == pytest.approx(2.0)
    assert close["realized_pnl_pct"] == pytest.approx(3.0)
    assert close["realized_pnl_usdt"] == pytest.approx(6.0)
    assert close["fees_usdt"] == pytest.approx(-0.03)
    assert close["net_pnl_usdt"] is None
    assert close["fill_history_recovery"]["status"] == "RECOVERED"
    assert rows[-1] == original_malformed.rstrip("\n")


def test_data_error_trade_with_incomplete_fills_stays_unresolved_and_uses_backoff(tmp_path, monkeypatch):
    import event_engine.tracker as tracker

    journal = tmp_path / "trades.jsonl"
    _event_id, fills = _trade_journal_fixture(journal, complete=False)
    monkeypatch.setattr(tracker, "TRADES_PATH", journal)
    monkeypatch.setattr(tracker, "get_fill_history", lambda *args, **kwargs: {"status": "ok", "fills": fills})

    assert tracker._recover_data_error_trade_closes(now_ms=1_700_000_010_000) == 0
    close = next(json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if '"record_type": "TRADE_CLOSE"' in line)
    assert close["exit_reason"] == "DATA_ERROR"
    assert close["realized_pnl_pct"] is None
    assert close["fill_history_recovery"]["status"] == "PENDING"
    assert close["fill_history_recovery"]["last_error"].startswith("incomplete_close_fills")
    assert close["fill_history_recovery"]["next_retry_ts"] > 1_700_000_010_000


def test_exact_stop_loss_audit_rejects_duplicate_or_mismatched_quantity_orders():
    import event_engine.bingx as bingx

    good = {"orderId": "sl-1", "type": "STOP_MARKET", "stopPrice": 95.0, "origQty": 2.0}
    verified, audit = bingx._audit_exact_single_sl(
        {"status": "ok", "sl_orders": [good]}, direction="LONG", avg_price=100,
        expected_price=95, expected_qty=2,
    )
    assert verified is True
    assert audit["audit_reason"] == "OK"

    verified, audit = bingx._audit_exact_single_sl(
        {"status": "ok", "sl_orders": [good, {**good, "orderId": "sl-duplicate"}]},
        direction="LONG", avg_price=100, expected_price=95, expected_qty=2,
    )
    assert verified is False
    assert audit["audit_reason"] == "SL_MULTIPLE_ORDERS"

    verified, audit = bingx._audit_exact_single_sl(
        {"status": "ok", "sl_orders": [{**good, "origQty": 1.5}]}, direction="LONG",
        avg_price=100, expected_price=95, expected_qty=2,
    )
    assert verified is False
    assert audit["audit_reason"] == "SL_QTY_OR_PRICE_MISMATCH"

    verified, audit = bingx._audit_exact_single_sl(
        {"status": "error", "error": "timeout", "sl_orders": []}, direction="LONG",
        avg_price=100, expected_price=95, expected_qty=2,
    )
    assert verified is False
    assert audit["audit_reason"] == "OPEN_ORDERS_UNVERIFIABLE"


def test_candidate_ledger_writes_features_rejection_reasons_and_cycle_ids(tmp_path, monkeypatch):
    import run_once as ro

    path = tmp_path / "actions.jsonl"
    monkeypatch.setattr(ro, "ACTIONS", path)
    candidate_id = ro._candidate_identifier("EVT-123", "BTC", "REGULAR_BULLISH_RSI", "LONG", 1000, 120_000)
    assert candidate_id == ro._candidate_identifier("EVT-123", "BTC", "REGULAR_BULLISH_RSI", "LONG", 1000, 120_000)
    assert candidate_id != ro._candidate_identifier("EVT-123", "BTC", "REGULAR_BULLISH_RSI", "LONG", 1000, 120_001)

    ro._record_candidate_decision(
        candidate_id=candidate_id, event_id="EVT-123", symbol="BTC", direction="LONG",
        event_type="REGULAR_BULLISH_RSI", timeframe="1h", stage="FUNDING_FILTER",
        decision="REJECTED", reason="FUNDING_ABOVE_LIMIT", evaluation_ts=123456,
        features={"funding": 0.005, "not_finite": float("nan")}, raw_market_features={"volume24": 123.0},
    )
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["record_type"] == "CANDIDATE_FILTER_DECISION"
    assert row["candidate_id"] == candidate_id
    assert row["stage"] == "FUNDING_FILTER"
    assert row["decision"] == "REJECTED"
    assert row["reason"] == "FUNDING_ABOVE_LIMIT"
    assert row["features"]["not_finite"] is None
    assert row["raw_market_features"]["volume24"] == 123.0
