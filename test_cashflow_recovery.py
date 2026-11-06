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


def test_cashflow_symbol_uses_canonical_bingx_display_name(monkeypatch):
    from event_engine import bingx, tracker

    monkeypatch.setattr(bingx, "CACHE", {
        "ts": 1_800_000_000.0,
        "data": {"NIULAI-USDT": {"symbol": "NIULAI-USDT", "displayName": "牛来"}},
        "by_display_name": {"牛来": {"symbol": "NIULAI-USDT", "displayName": "牛来"}},
    })
    assert tracker._cashflow_symbol("牛来") == "NIULAI-USDT"
    assert tracker._cashflow_symbol("NIULAI-USDT") == "NIULAI-USDT"


def test_canonical_bingx_symbol_honors_compact_operator_symbol_map(monkeypatch):
    from event_engine import bingx, tracker

    monkeypatch.setattr(bingx, "SYMBOL_MAP", {"LOCALCOIN": "REMOTECOINUSDT"})
    monkeypatch.setattr(bingx, "CACHE", {"ts": 0, "data": {}, "by_display_name": {}})

    assert tracker._cashflow_symbol("LOCALCOIN") == "REMOTECOIN-USDT"
    assert tracker._cashflow_symbol("LOCALCOIN-USDT") == "REMOTECOIN-USDT"


def test_cashflow_recovery_matches_chinese_display_name_to_bingx_fill_and_income_symbol(tmp_path, monkeypatch):
    from event_engine import bingx, tracker

    journal = tmp_path / "trades.jsonl"
    open_ts = 1_700_000_000_000
    close_ts = open_ts + 60 * 60_000
    rows = [
        {
            "record_type": "TRADE_OPEN", "event_id": "EVT_NIULAI", "symbol": "牛来",
            "direction": "SHORT", "ts": open_ts,
            "execution": {"fill_ts_ms": open_ts + 1000, "order_id": "ENTRY-N"},
            "protection": {"tp_orders": [{"order_id": "TP-N"}]},
        },
        {
            "record_type": "TRADE_CLOSE", "event_id": "EVT_NIULAI", "symbol": "牛来",
            "direction": "SHORT", "closed_ts": close_ts, "exit_reason": "TAKE_PROFIT_FULL",
            "exit_order_ids": ["TP-N"], "realized_pnl_usdt": None,
            "fees_usdt": None, "funding_usdt": None, "net_pnl_usdt": None,
            "pnl_costs_status": "UNVERIFIED_POSITION_ATTRIBUTION",
            "cashflow_recovery": {"status": "PENDING", "attempts": 0, "next_retry_ts": 0},
        },
    ]
    journal.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(tracker, "TRADES_PATH", journal)

    def load_catalog():
        bingx.CACHE.update(
            ts=1_800_000_000.0,
            data={"NIULAI-USDT": {"symbol": "NIULAI-USDT", "displayName": "牛来"}},
            by_display_name={"牛来": {"symbol": "NIULAI-USDT", "displayName": "牛来"}},
        )
        return bingx.CACHE["data"]

    monkeypatch.setattr(tracker, "bingx_contracts", load_catalog)

    def fake_fill_history(symbol, start, end, *, order_id=None, **kwargs):
        assert symbol == "牛来"
        assert tracker._cashflow_symbol(symbol) == "NIULAI-USDT"
        rows_by_order = {
            "ENTRY-N": [{
                "trade_id": "FILL-ENTRY", "order_id": "ENTRY-N", "symbol": "NIULAI-USDT",
                "position_side": "SHORT", "time": open_ts + 1000, "qty": 1.0,
                "price": 0.1, "realized_pnl": 0.0, "fee": -0.01, "fee_asset": "USDT",
            }],
            "TP-N": [{
                "trade_id": "FILL-EXIT", "order_id": "TP-N", "symbol": "NIULAI-USDT",
                "position_side": "SHORT", "time": close_ts - 1000, "qty": 1.0,
                "price": 0.09, "realized_pnl": 0.1, "fee": -0.01, "fee_asset": "USDT",
            }],
        }
        return {"status": "ok", "has_more": False, "fills": rows_by_order[str(order_id)]}

    monkeypatch.setattr(tracker, "get_fill_history", fake_fill_history)
    monkeypatch.setattr(tracker, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [
            {"record_id": "INC-PNL", "trade_id": "FILL-EXIT", "symbol": "NIULAI-USDT", "time": close_ts - 1000, "asset": "USDT", "income_type": "REALIZED_PNL", "amount": 0.1},
            {"record_id": "INC-FEE-1", "trade_id": "FILL-ENTRY", "symbol": "NIULAI-USDT", "time": open_ts + 1000, "asset": "USDT", "income_type": "TRADING_FEE", "amount": -0.01},
            {"record_id": "INC-FEE-2", "trade_id": "FILL-EXIT", "symbol": "NIULAI-USDT", "time": close_ts - 1000, "asset": "USDT", "income_type": "TRADING_FEE", "amount": -0.01},
        ],
    })

    recovered = tracker._recover_trade_cashflows(now_ms=close_ts + 10_000, max_records=2)
    assert recovered == 1
    close_record = json.loads(journal.read_text(encoding="utf-8").splitlines()[1])
    assert close_record["pnl_costs_status"] == "BINGX_ORDER_LINKED_FILLS_AND_INCOME_CONFIRMED"
    assert close_record["realized_pnl_usdt"] == pytest.approx(0.1)
    assert close_record["fees_usdt"] == pytest.approx(-0.02)
    assert close_record["net_pnl_usdt"] == pytest.approx(0.08)


def test_fill_history_none_data_is_not_misreported_as_empty_and_has_envelope_diagnostics(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "ZEC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {
        "code": 0, "msg": "", "data": None,
    })

    result = bingx.get_fill_history("ZEC", 1_700_000_000_000, 1_700_000_001_000, order_id="ENTRY-1")

    assert result["status"] == "error"
    assert result["error"] == "fill_history_unexpected_data_shape"
    assert result["response_code"] == 0
    assert result["response_data_type"] == "NoneType"
    assert result["response_keys"] == ["code", "data", "msg"]


def test_income_history_none_data_remains_incomplete_with_envelope_diagnostics(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "ZEC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {
        "code": 0, "msg": "", "data": None,
    })

    result = bingx.get_income_history("ZEC", 1_700_000_000_000, 1_700_000_001_000)

    assert result["status"] == "error"
    assert result["complete"] is False
    assert result["error"] == "income_history_unexpected_data_shape"
    assert result["response_code"] == 0
    assert result["response_data_type"] == "NoneType"
    assert result["response_keys"] == ["code", "data", "msg"]


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


def test_fill_recovery_error_preserves_bounded_exchange_envelope_diagnostics(monkeypatch):
    from event_engine import tracker

    queried_order_ids = []
    def failed_fill_query(*a, **kwargs):
        queried_order_ids.append(kwargs.get("order_id"))
        return {
            "status": "error", "error": "fill_history_unexpected_data_shape",
            "response_type": "dict", "response_code": 0, "response_msg": "unexpected response",
            "response_keys": ["code", "data", "msg"], "response_data_type": "NoneType",
            "shape_summary": "NoneType", "shape_fingerprint": "ABC123",
        }
    monkeypatch.setattr(tracker, "get_fill_history", failed_fill_query)
    close_ts = 1_700_000_100_000
    opened = {
        "symbol": "BTC", "direction": "LONG", "ts": close_ts - 60_000,
        "execution": {"fill_ts_ms": close_ts - 60_000, "order_id": "ENTRY-1"},
        "protection": {"tp_orders": [{"order_id": "TP-1"}]},
    }
    closed = {
        "symbol": "BTC", "direction": "LONG", "closed_ts": close_ts,
        "exit_order_ids": ["TP-1"],
    }

    fills, error = tracker._fetch_trade_fill_history(closed, opened, close_ts + 1000)

    assert fills == []
    # The first exact order query fails closed; no broad symbol/time scan follows.
    assert queried_order_ids == ["ENTRY-1"]
    assert "fill_history_unexpected_data_shape" in error
    assert "response_code=0" in error
    assert "response_data_type=NoneType" in error
    assert "response_keys=code,data,msg" in error
    assert "shape_fingerprint=ABC123" in error


def test_income_history_supports_account_wide_query_without_symbol(monkeypatch):
    from event_engine import bingx

    captured = {}
    def fake_request(method, path, params=None, **kwargs):
        captured.update(method=method, path=path, params=dict(params or {}), kwargs=kwargs)
        return {"code": 0, "data": [
            {"tranId": "BTC-FEE", "symbol": "BTC-USDT", "incomeType": "TRADING_FEE", "income": "-0.12", "asset": "USDT", "time": 1_700_000_001_000},
            {"tranId": "ETH-FEE", "symbol": "ETH-USDT", "incomeType": "TRADING_FEE", "income": "-0.50", "asset": "USDT", "time": 1_700_000_002_000},
            {"tranId": "TRANSFER-1", "incomeType": "TRANSFER", "income": "10", "asset": "USDT", "time": 1_700_000_003_000},
        ]}
    monkeypatch.setattr(bingx, "_request", fake_request)

    result = bingx.get_income_history(None, 1_700_000_000_000, 1_700_000_004_000)

    assert result["status"] == "ok" and result["complete"] is True
    assert result["query_scope"] == "ACCOUNT_WIDE"
    assert "symbol" not in captured["params"]
    filtered = bingx.filter_income_history_by_symbol(result, "BTC")
    assert filtered["status"] == "ok" and filtered["complete"] is True
    assert [row["record_id"] for row in filtered["records"]] == ["BTC-FEE"]


def test_account_wide_income_filter_rejects_unidentifiable_non_transfer_rows(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": [
        {"tranId": "UNKNOWN", "incomeType": "TRADING_FEE", "income": "-0.12", "asset": "USDT", "time": 1_700_000_001_000},
    ]})
    result = bingx.get_income_history(None, 1_700_000_000_000, 1_700_000_004_000)
    filtered = bingx.filter_income_history_by_symbol(result, "BTC")
    assert filtered["status"] == "error"
    assert filtered["complete"] is False
    assert filtered["error"] == "account_wide_income_row_missing_symbol"


def test_tracker_income_recovery_uses_account_wide_fallback_after_symbol_data_null(monkeypatch):
    from event_engine import tracker

    calls = []
    def income(symbol, *args, **kwargs):
        calls.append(symbol)
        if symbol is not None:
            return {
                "status": "error", "complete": False, "records": [],
                "error": "income_history_unexpected_data_shape",
                "response_data_type": "NoneType", "response_code": 0,
            }
        return {
            "status": "ok", "complete": True, "records": [
                {"symbol": "BTC-USDT", "record_id": "BTC-FEE", "income_type": "TRADING_FEE", "amount": -0.12, "asset": "USDT", "time": 1_700_000_001_000},
                {"symbol": "ETH-USDT", "record_id": "ETH-FEE", "income_type": "TRADING_FEE", "amount": -9.99, "asset": "USDT", "time": 1_700_000_001_000},
                {"symbol": "", "record_id": "TRANSFER", "income_type": "TRANSFER", "amount": 10.0, "asset": "USDT", "time": 1_700_000_001_000},
            ],
        }

    monkeypatch.setattr(tracker, "get_income_history", income)
    response = tracker._get_income_history_for_symbol(
        "BTC", 1_700_000_000_000, 1_700_000_002_000, timeout_sec=2.0,
    )

    assert calls == ["BTC", None]
    assert response["status"] == "ok" and response["complete"] is True
    assert response["query_scope"] == "ACCOUNT_WIDE_FILTERED_BY_ROW_SYMBOL"
    assert [row["record_id"] for row in response["records"]] == ["BTC-FEE"]


def test_tracker_income_fallback_does_not_run_on_ordinary_timeout(monkeypatch):
    from event_engine import tracker

    calls = []
    def income(symbol, *args, **kwargs):
        calls.append(symbol)
        return {"status": "error", "complete": False, "records": [], "error": "request_exception:timeout"}

    monkeypatch.setattr(tracker, "get_income_history", income)
    response = tracker._get_income_history_for_symbol(
        "BTC", 1_700_000_000_000, 1_700_000_002_000,
    )

    assert calls == ["BTC"]
    assert response["error"] == "request_exception:timeout"

def test_tracker_income_fallback_does_not_run_on_nonzero_code_even_when_data_is_null(monkeypatch):
    from event_engine import tracker

    calls = []
    def income(symbol, *args, **kwargs):
        calls.append(symbol)
        return {"status": "error", "complete": False, "records": [],
                "error": "100410", "response_code": 100410, "response_data_type": "NoneType"}
    monkeypatch.setattr(tracker, "get_income_history", income)
    response = tracker._get_income_history_for_symbol("BTC", 1_700_000_000_000, 1_700_000_002_000)
    assert calls == ["BTC"]
    assert response["response_code"] == 100410


def test_tracker_income_fallback_requires_code_zero_and_expected_shape_error(monkeypatch):
    from event_engine import tracker

    calls = []
    def income(symbol, *args, **kwargs):
        calls.append(symbol)
        return {"status": "error", "complete": False, "records": [],
                "error": "income_history_unexpected_data_shape",
                "response_code": 100410, "response_data_type": "NoneType"}
    monkeypatch.setattr(tracker, "get_income_history", income)
    tracker._get_income_history_for_symbol("BTC", 1_700_000_000_000, 1_700_000_002_000)
    assert calls == ["BTC"]

