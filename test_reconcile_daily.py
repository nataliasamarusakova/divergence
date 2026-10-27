"""Read-only daily reconciliation regression tests; all exchange I/O is stubbed."""
import json
from datetime import datetime, timezone

from scripts import reconcile_daily as recon


def _stub_protection(symbol, direction, **kwargs):
    qty = 2.0 if symbol.startswith("BTC") else 0.5
    return {
        "status": "ok",
        "sl_orders": [{"orderId": "SL-TEST", "type": "STOP_MARKET", "stopPrice": "95.0", "positionSide": direction, "origQty": str(qty)}],
        "tp_orders": [{"orderId": "TP-TEST", "type": "TAKE_PROFIT_MARKET", "positionSide": direction, "origQty": str(qty)}],
    }


def _now_ms():
    return int(datetime(2026, 10, 27, 10, 0, tzinfo=timezone.utc).timestamp() * 1000)


def test_daily_reconciliation_detects_position_mismatch_and_persists_health(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    now_ms = _now_ms()
    (data / "active_trades.json").write_text(json.dumps({
        "EVT-A": {"symbol": "BTC", "direction": "LONG", "remaining_qty": 1.0, "closed": False},
    }), encoding="utf-8")
    (data / "trades.jsonl").write_text(json.dumps({
        "record_type": "TRADE_OPEN", "event_id": "EVT-A", "symbol": "BTC", "direction": "LONG", "ts": now_ms,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [{
        "symbol": "BTC-USDT", "positionSide": "LONG", "positionAmt": "2.0",
    }])
    monkeypatch.setattr(recon.bingx, "get_open_protection_directional", _stub_protection)
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [
            {"symbol": "BTC-USDT", "asset": "USDT", "income_type": "REALIZED_PNL", "amount": -1.0},
            {"symbol": "BTC-USDT", "asset": "USDT", "income_type": "TRADING_FEE", "amount": -0.02},
        ],
    })
    alerts = []
    monkeypatch.setattr(recon, "send_detailed", lambda text: alerts.append(text) or {"test": {"sent": True}})

    report = recon.run_reconciliation(now_ms=now_ms, data_dir=data, notify=True)

    assert report["status"] == "CRITICAL"
    assert any(item["code"] == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH" for item in report["discrepancies"])
    assert report["checks"]["daily_exchange_income_aggregates"]["symbols_checked"]["BTC-USDT"]["aggregate_net"] == -1.02
    assert "never used to attribute per-position" in report["checks"]["daily_exchange_income_aggregates"]["scope"]
    assert alerts and "CRITICAL" in alerts[0]
    health_rows = (data / "health.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(health_rows) == 1
    assert json.loads(health_rows[0])["status"] == "CRITICAL"


def test_missing_active_state_is_unknown_not_interpreted_as_empty(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [{
        "symbol": "ETH-USDT", "positionSide": "SHORT", "positionAmt": "0.5",
    }])
    monkeypatch.setattr(recon.bingx, "get_open_protection_directional", _stub_protection)
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })
    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    assert report["checks"]["local_active_state"]["status"] == "UNKNOWN"
    assert not any(item["code"] == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH" for item in report["discrepancies"])
    assert any(item["code"] == "LOCAL_ACTIVE_STATE_UNAVAILABLE" for item in report["discrepancies"])


def test_daily_reconciliation_rejects_stop_order_without_positive_trigger_price(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text("{}", encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [{
        "symbol": "BTC-USDT", "positionSide": "LONG", "positionAmt": "2.0",
    }])
    monkeypatch.setattr(recon.bingx, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok", "sl_orders": [{
            "orderId": "SL-NO-TRIGGER", "type": "STOP_MARKET", "positionSide": "LONG", "origQty": "2.0",
        }], "tp_orders": [],
    })
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })
    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)
    matches = [item for item in report["discrepancies"] if item["code"] == "EXCHANGE_STOP_LOSS_NOT_VERIFIED"]
    assert matches
    assert matches[0]["level"] == "CRITICAL"
    assert matches[0]["sl_orders"][0]["trigger_price_valid"] is False
    assert report["status"] == "CRITICAL"


def test_daily_reconciliation_does_not_hide_zero_stop_price_with_conflicting_alias(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text("{}", encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [{
        "symbol": "BTC-USDT", "positionSide": "LONG", "positionAmt": "2.0",
    }])
    monkeypatch.setattr(recon.bingx, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok", "sl_orders": [{
            "orderId": "SL-CONFLICT", "type": "STOP_MARKET", "stopPrice": "0",
            "triggerPrice": "95", "positionSide": "LONG", "origQty": "2.0",
        }], "tp_orders": [],
    })
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })
    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)
    matches = [item for item in report["discrepancies"] if item["code"] == "EXCHANGE_STOP_LOSS_NOT_VERIFIED"]
    assert matches
    assert matches[0]["sl_orders"][0]["trigger_price_valid"] is False
    assert report["status"] == "CRITICAL"


def test_reconciliation_exchange_api_error_is_recorded_and_does_not_raise(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text("{}", encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("stub outage")))
    monkeypatch.setattr(recon, "send_detailed", lambda *_: {})

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=True)

    assert report["status"] == "CRITICAL"
    assert report["checks"]["exchange_positions"]["status"] == "ERROR"
    assert any(item["code"] == "EXCHANGE_POSITION_QUERY_FAILED" for item in report["discrepancies"])


def test_unknown_net_attribution_is_not_misclassified_as_failed_close(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    now_ms = _now_ms()
    (data / "active_trades.json").write_text("{}", encoding="utf-8")
    (data / "trades.jsonl").write_text(json.dumps({
        "record_type": "TRADE_CLOSE", "event_id": "EVT-U", "symbol": "BTC-USDT",
        "direction": "LONG", "closed_ts": now_ms, "exit_reason": "TAKE_PROFIT_FULL",
        "pnl_costs_status": "UNVERIFIED_POSITION_ATTRIBUTION", "net_pnl_usdt": None,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })
    report = recon.run_reconciliation(now_ms=now_ms, data_dir=data, notify=False)
    assert not any(item["code"] == "DATA_ERROR_CLOSES_IN_UTC_DAY" for item in report["discrepancies"])
    assert report["status"] != "CRITICAL"
