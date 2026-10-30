"""Read-only daily reconciliation regression tests; all exchange I/O is stubbed."""
import json
from datetime import datetime, timezone

from event_engine import reconcile_daily as recon


def _stub_snapshot(**kwargs):
    return {"status": "ok", "orders": [
        {"symbol": "BTC-USDT", "orderId": "SL-BTC", "type": "STOP_MARKET", "stopPrice": "95.0", "positionSide": "LONG", "origQty": "2.0"},
        {"symbol": "BTC-USDT", "orderId": "TP-BTC", "type": "TAKE_PROFIT_MARKET", "positionSide": "LONG", "origQty": "2.0"},
        {"symbol": "ETH-USDT", "orderId": "SL-ETH", "type": "STOP_MARKET", "stopPrice": "105.0", "positionSide": "SHORT", "origQty": "0.5"},
        {"symbol": "ETH-USDT", "orderId": "TP-ETH", "type": "TAKE_PROFIT_MARKET", "positionSide": "SHORT", "origQty": "0.5"},
    ]}


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
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", _stub_snapshot)
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
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", _stub_snapshot)
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
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **k: {"status": "ok", "orders": [{
        "symbol": "BTC-USDT", "orderId": "SL-NO-TRIGGER", "type": "STOP_MARKET", "positionSide": "LONG", "origQty": "2.0",
    }]})
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
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **k: {"status": "ok", "orders": [{
        "symbol": "BTC-USDT", "orderId": "SL-CONFLICT", "type": "STOP_MARKET", "stopPrice": "0",
        "triggerPrice": "95", "positionSide": "LONG", "origQty": "2.0",
    }]})
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


def test_main_failure_alert_contains_real_newline(tmp_path, monkeypatch):
    from event_engine import reconcile_daily as recon

    alerts = []
    monkeypatch.setattr(recon, "HEALTH_PATH", tmp_path / "health.jsonl")
    monkeypatch.setattr(recon, "run_reconciliation", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(recon, "_append_health", lambda path, report: None)
    monkeypatch.setattr(recon, "send_detailed", lambda message: alerts.append(message))

    assert recon.main() == 0
    assert len(alerts) == 1
    assert "</b>\\n<code>" not in alerts[0]
    assert "</b>\n<code>" in alerts[0]


def test_daily_reconciliation_uses_one_account_wide_open_orders_snapshot(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text("{}", encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    positions = [
        {"symbol": "BTC-USDT", "positionSide": "LONG", "positionAmt": "2.0"},
        {"symbol": "ETH-USDT", "positionSide": "SHORT", "positionAmt": "-0.5"},
        {"symbol": "SOL-USDT", "positionSide": "LONG", "positionAmt": "3.0"},
    ]
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: positions)
    calls = []
    def snapshot(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "orders": _stub_snapshot()["orders"] + [
            {"symbol": "SOL-USDT", "orderId": "SL-SOL", "type": "STOP_MARKET", "stopPrice": "95", "positionSide": "LONG", "origQty": "3"},
            {"symbol": "SOL-USDT", "orderId": "TP-SOL", "type": "TAKE_PROFIT_MARKET", "positionSide": "LONG", "origQty": "3"},
        ]}
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", snapshot)
    monkeypatch.setattr(recon.bingx, "get_open_protection_directional", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not issue per-symbol openOrders calls")))
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {"status": "ok", "complete": True, "records": []})

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)
    assert len(calls) == 1
    assert report["checks"]["exchange_protection"]["positions_checked"] == 3
    assert report["checks"]["exchange_protection"]["source"] == "account_open_orders_snapshot"
    assert not any(item["code"] == "EXCHANGE_PROTECTION_STATE_UNKNOWN" for item in report["discrepancies"])


def test_open_orders_100410_is_aggregated_and_repeat_alert_is_stable(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text("{}", encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    positions = [
        {"symbol": "BTC-USDT", "positionSide": "LONG", "positionAmt": "2.0"},
        {"symbol": "ETH-USDT", "positionSide": "SHORT", "positionAmt": "-0.5"},
        {"symbol": "SOL-USDT", "positionSide": "LONG", "positionAmt": "3.0"},
    ]
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: positions)
    timeouts = iter((1793193767293, 1793193767294))
    snapshot_calls = []
    def limited_snapshot(**kwargs):
        snapshot_calls.append(kwargs)
        unblock_ms = next(timeouts)
        return {
            "status": "error", "code": 100410, "rate_limited": True,
            "retry_after_ms": unblock_ms,
            "error": f"openOrders snapshot failed: code=100410:The endpoint trigger frequency limit rule is currently in the disabled period and will be unblocked after {unblock_ms}",
        }
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", limited_snapshot)
    monkeypatch.setattr(recon.bingx, "get_open_protection_directional", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not retry the rate-limited endpoint per symbol")))
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {"status": "ok", "complete": True, "records": []})
    alerts = []
    monkeypatch.setattr(recon, "send_detailed", lambda text: alerts.append(text) or {"chat": {"sent": True}})

    first = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=True)
    second = recon.run_reconciliation(now_ms=_now_ms() + 30_000, data_dir=data, notify=True)
    errors = [item for item in first["discrepancies"] if item["code"] == "EXCHANGE_PROTECTION_STATE_UNKNOWN"]
    assert first["status"] == "CRITICAL"
    assert len(errors) == 1
    assert errors[0]["position_count"] == 3
    assert first["checks"]["exchange_protection"]["status"] == "UNKNOWN"
    assert first["checks"]["exchange_protection"]["retry_after_ms"] == 1793193767293
    assert second["notification"]["status"] == "SUPPRESSED_REPEAT"
    assert len(snapshot_calls) == 2  # one account-level call per reconciliation, never one per position
    assert len(alerts) == 1
    assert "No per-symbol retries were attempted" in alerts[0]


def test_daily_income_reconciliation_retries_once_after_incomplete_timeout(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    now_ms = _now_ms()
    (data / "active_trades.json").write_text("{}", encoding="utf-8")
    (data / "trades.jsonl").write_text(json.dumps({
        "record_type": "TRADE_CLOSE", "event_id": "EVT-RETRY", "symbol": "BTC-USDT",
        "direction": "LONG", "closed_ts": now_ms, "exit_reason": "STOP_LOSS",
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(recon, "MAX_INCOME_SYMBOLS", 1)
    monkeypatch.setattr(recon, "INCOME_TIMEOUT_SEC", 3.0)
    monkeypatch.setattr(recon, "INCOME_REQUEST_BUDGET", 4)
    monkeypatch.setattr(recon, "INCOME_RETRIES", 1)
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", _stub_snapshot)
    calls = []
    def income(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"status": "error", "error": "read timeout", "complete": False, "records": []}
        return {"status": "ok", "complete": True, "records": [
            {"symbol": "BTC-USDT", "asset": "USDT", "income_type": "TRADING_FEE", "amount": -0.01},
        ]}
    monkeypatch.setattr(recon.bingx, "get_income_history", income)
    monkeypatch.setattr(recon.time, "sleep", lambda *_: None)

    report = recon.run_reconciliation(now_ms=now_ms, data_dir=data, notify=False)

    assert len(calls) == 2
    assert calls[0]["timeout_sec"] == 3.0
    summary = report["checks"]["daily_exchange_income_aggregates"]["symbols_checked"]["BTC-USDT"]
    assert summary["status"] == "OK"
    assert summary["attempts"] == 2
    assert summary["trading_fees"] == -0.01
    assert not any(item["code"] == "DAILY_INCOME_QUERY_INCOMPLETE" for item in report["discrepancies"])


def test_income_unexpected_shape_details_are_exposed_in_daily_warning(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    now_ms = _now_ms()
    (data / "active_trades.json").write_text("{}", encoding="utf-8")
    (data / "trades.jsonl").write_text(json.dumps({
        "record_type": "TRADE_OPEN", "event_id": "EVT-INCOME-SHAPE", "symbol": "BTC",
        "direction": "LONG", "ts": now_ms,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **kwargs: {"status": "ok", "orders": []})
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "error", "error": "income_history_unexpected_data_shape", "complete": False,
        "records": [], "shape_fingerprint": "ABC123", "shape_summary": "dict:fields=incomeHistory:dict",
    })
    alerts = []
    monkeypatch.setattr(recon, "send_detailed", lambda text: alerts.append(text) or {"test": {"sent": True}})
    monkeypatch.setenv("DIVERGENCE_BUILD_VERSION", "V10.4.25")

    report = recon.run_reconciliation(now_ms=now_ms, data_dir=data, notify=True)
    warning = next(item for item in report["discrepancies"] if item["code"] == "DAILY_INCOME_QUERY_INCOMPLETE")
    assert "shape_fingerprint=ABC123" in warning["detail"]
    assert "incomeHistory" in warning["detail"]
    assert report["build_version"] == "V10.4.25"
    assert "Build: <code>V10.4.25</code>" in alerts[0]


def test_trump_legacy_exchange_symbol_is_canonicalized_without_merging_sol():
    assert recon._canonical_position_symbol("TRUMP") == "TRUMP-USDT"
    assert recon._canonical_position_symbol("TRUMPUSDT") == "TRUMP-USDT"
    assert recon._canonical_position_symbol("TRUMP-USDT") == "TRUMP-USDT"
    assert recon._canonical_position_symbol("TRUMPSOL-USDT") == "TRUMP-USDT"
    assert recon._canonical_position_symbol("TRUMPSOLUSDT") == "TRUMP-USDT"
    assert recon._canonical_position_symbol("SOL") == "SOL-USDT"
    assert recon._canonical_position_symbol("SOL-USDT") == "SOL-USDT"
    assert recon._canonical_position_symbol("TRUMP-SOL-USDT") == "TRUMP-SOL-USDT"
