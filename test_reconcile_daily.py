"""Read-only daily reconciliation regression tests; all exchange I/O is stubbed."""
import json
from datetime import datetime, timezone

from event_engine import reconcile_daily as recon
import pytest


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


def test_clean_first_run_bootstraps_empty_local_state_only_after_confirmed_flat_exchange(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    assert report["status"] == "OK"
    assert report["checks"]["runtime_state_initialization"]["status"] == "BOOTSTRAPPED_EMPTY_ACCOUNT"
    assert report["checks"]["local_active_state"]["status"] == "OK"
    assert report["checks"]["trade_journal"]["status"] == "OK"
    assert json.loads((data / "active_trades.json").read_text(encoding="utf-8")) == {}
    assert (data / "trades.jsonl").read_text(encoding="utf-8") == ""
    marker = json.loads((data / "runtime_state_initialized.json").read_text(encoding="utf-8"))
    assert marker["initialized"] is True
    assert marker["reason"] == "confirmed_empty_exchange_snapshot"
    assert not any(item["code"] == "LOCAL_ACTIVE_STATE_UNAVAILABLE" for item in report["discrepancies"])


def test_runtime_state_loss_after_initialization_is_not_silently_bootstrapped(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "runtime_state_initialized.json").write_text(json.dumps({
        "version": 1, "initialized": True, "initialized_at_ts_ms": _now_ms(),
        "reason": "confirmed_empty_exchange_snapshot",
    }), encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    assert report["checks"]["local_active_state"]["status"] == "UNKNOWN"
    assert any(item["code"] == "LOCAL_ACTIVE_STATE_UNAVAILABLE" for item in report["discrepancies"])
    assert not (data / "active_trades.json").exists()


def test_clean_state_is_not_bootstrapped_when_exchange_snapshot_has_live_exposure(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [{
        "symbol": "ETH-USDT", "positionSide": "SHORT", "positionAmt": "0.5",
    }])
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **kwargs: _stub_snapshot())

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    assert report["checks"]["runtime_state_initialization"]["status"] == "EXPOSURE_WITHOUT_LOCAL_STATE"
    assert report["checks"]["local_active_state"]["status"] == "UNKNOWN"
    assert any(item["code"] == "LOCAL_ACTIVE_STATE_UNAVAILABLE" for item in report["discrepancies"])
    assert not (data / "active_trades.json").exists()
    assert not (data / "trades.jsonl").exists()


def test_runtime_state_is_not_bootstrapped_when_exchange_snapshot_fails(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("exchange unavailable")))

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    assert report["checks"]["runtime_state_initialization"]["status"] == "UNKNOWN"
    assert report["checks"]["local_active_state"]["status"] == "UNKNOWN"
    assert not (data / "active_trades.json").exists()
    assert not (data / "trades.jsonl").exists()


def test_missing_state_with_prior_successful_reconciliation_evidence_is_state_loss(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    previous = {
        "record_type": "DAILY_RECONCILIATION",
        "checks": {
            "local_active_state": {"status": "OK", "active_event_records": 0},
            "trade_journal": {"status": "OK", "records": 0},
        },
    }
    (data / "health.jsonl").write_text(json.dumps(previous) + "\n", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    assert report["checks"]["runtime_state_initialization"]["status"] == "MARKED_EXISTING_STATE"
    assert report["checks"]["local_active_state"]["status"] == "UNKNOWN"
    assert any(item["code"] == "LOCAL_ACTIVE_STATE_UNAVAILABLE" for item in report["discrepancies"])
    assert not (data / "active_trades.json").exists()
    assert not (data / "trades.jsonl").exists()


def test_corrupt_health_history_prevents_empty_state_bootstrap(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "health.jsonl").write_text('{truncated-json\n', encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    assert report["checks"]["runtime_state_initialization"]["status"] == "MARKED_EXISTING_STATE"
    assert report["checks"]["local_active_state"]["status"] == "UNKNOWN"
    assert any(item["code"] == "LOCAL_ACTIVE_STATE_UNAVAILABLE" for item in report["discrepancies"])
    assert not (data / "active_trades.json").exists()
    assert not (data / "trades.jsonl").exists()


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
    assert not any(item["code"] == "DATA_ERROR_CLOSES_UNRESOLVED" for item in report["discrepancies"])
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
    income_calls = []
    def income_unavailable(symbol, *args, **kwargs):
        income_calls.append(symbol)
        return {
            "status": "error", "error": "income_history_unexpected_data_shape", "complete": False,
            "records": [], "shape_fingerprint": "ABC123", "shape_summary": "dict:fields=incomeHistory:dict",
            "response_type": "dict", "response_code": 0, "response_msg": "unexpected response",
            "response_keys": ["code", "data", "msg"], "response_data_type": "NoneType",
        }
    monkeypatch.setattr(recon.bingx, "get_income_history", income_unavailable)
    monkeypatch.setattr(recon, "INCOME_NULL_GRACE_SEC", 0.0)
    alerts = []
    monkeypatch.setattr(recon, "send_detailed", lambda text: alerts.append(text) or {"test": {"sent": True}})
    # A stale environment override must never split the displayed build version
    # from the canonical VERSION file.
    monkeypatch.setenv("DIVERGENCE_BUILD_VERSION", "obsolete-build")

    report = recon.run_reconciliation(now_ms=now_ms, data_dir=data, notify=True)
    warning = next(item for item in report["discrepancies"] if item["code"] == "DAILY_INCOME_QUERY_INCOMPLETE")
    assert "shape_fingerprint=ABC123" in warning["detail"]
    assert "incomeHistory" in warning["detail"]
    assert "response_data_type\":\"NoneType" in warning["detail"]
    assert warning["response_diagnostics"]["response_code"] == 0
    assert warning["response_diagnostics"]["response_keys"] == ["code", "data", "msg"]
    assert income_calls == ["BTC-USDT", "BTC-USDT", None]
    assert "BTC-USDT" in recon._discrepancy_message_detail(warning)
    assert "account_wide_income_unexpected_data_shape" in warning["detail"] or "income_history_unexpected_data_shape" in warning["detail"]
    assert report["build_version"] == recon.APP_VERSION
    assert f"Build: <code>{recon.APP_VERSION}</code>" in alerts[0]


def test_income_null_data_is_pending_during_grace_and_warns_only_after_expiry(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    now_ms = _now_ms()
    (data / "active_trades.json").write_text("{}", encoding="utf-8")
    (data / "trades.jsonl").write_text(json.dumps({
        "record_type": "TRADE_OPEN", "event_id": "EVT-INCOME-PENDING", "symbol": "BTC",
        "direction": "LONG", "ts": now_ms,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(recon, "INCOME_RETRIES", 0)
    monkeypatch.setattr(recon, "INCOME_NULL_GRACE_SEC", 1800.0)
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])
    null_data = {
        "status": "error", "error": "income_history_unexpected_data_shape", "complete": False,
        "records": [], "shape_fingerprint": "NULL123", "shape_summary": "NoneType",
        "response_type": "dict", "response_code": 0, "response_msg": "",
        "response_keys": ["code", "data", "msg"], "response_data_type": "NoneType",
    }
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: dict(null_data))

    first = recon.run_reconciliation(now_ms=now_ms, data_dir=data, notify=False)
    first_summary = first["checks"]["daily_exchange_income_aggregates"]["symbols_checked"]["BTC-USDT"]
    assert first_summary["status"] == "PENDING"
    assert first_summary["accounting_value"] == "UNKNOWN_NOT_ZERO"
    assert first["checks"]["daily_exchange_income_aggregates"]["status"] == "PENDING"
    assert not any(item["code"] == "DAILY_INCOME_QUERY_INCOMPLETE" for item in first["discrepancies"])
    state_path = data / "reconciliation_income_pending.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["pending"]["2026-10-27|BTC-USDT"]["since_ts_ms"] == now_ms

    second = recon.run_reconciliation(now_ms=now_ms + 10 * 60_000, data_dir=data, notify=False)
    second_summary = second["checks"]["daily_exchange_income_aggregates"]["symbols_checked"]["BTC-USDT"]
    assert second_summary["status"] == "PENDING"
    assert second_summary["pending_since_ts_ms"] == now_ms
    assert not any(item["code"] == "DAILY_INCOME_QUERY_INCOMPLETE" for item in second["discrepancies"])

    expired = recon.run_reconciliation(now_ms=now_ms + 1800 * 1000, data_dir=data, notify=False)
    expired_summary = expired["checks"]["daily_exchange_income_aggregates"]["symbols_checked"]["BTC-USDT"]
    assert expired_summary["status"] == "ERROR"
    assert expired_summary["accounting_value"] == "UNKNOWN_NOT_ZERO"
    warning = next(item for item in expired["discrepancies"] if item["code"] == "DAILY_INCOME_QUERY_INCOMPLETE")
    assert "null_data_pending_for_sec=1800" in warning["detail"]
    assert expired["checks"]["daily_income_pending_state"]["status"] == "WARNING"
    assert expired["checks"]["daily_income_pending_state"]["expired_symbols"] == ["BTC-USDT"]

    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [], "query_scope": "SYMBOL",
    })
    recovered = recon.run_reconciliation(now_ms=now_ms + 1900 * 1000, data_dir=data, notify=False)
    recovered_summary = recovered["checks"]["daily_exchange_income_aggregates"]["symbols_checked"]["BTC-USDT"]
    assert recovered_summary["status"] == "OK"
    assert not any(item["code"] == "DAILY_INCOME_QUERY_INCOMPLETE" for item in recovered["discrepancies"])
    assert json.loads(state_path.read_text(encoding="utf-8"))["pending"] == {}


def test_income_null_pending_state_write_failure_does_not_silence_warning(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    now_ms = _now_ms()
    (data / "active_trades.json").write_text("{}", encoding="utf-8")
    (data / "trades.jsonl").write_text(json.dumps({
        "record_type": "TRADE_OPEN", "event_id": "EVT-INCOME-WRITE-FAIL", "symbol": "BTC",
        "direction": "LONG", "ts": now_ms,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(recon, "INCOME_RETRIES", 0)
    monkeypatch.setattr(recon, "INCOME_NULL_GRACE_SEC", 1800.0)
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "error", "error": "income_history_unexpected_data_shape", "complete": False,
        "records": [], "response_code": 0, "response_data_type": "NoneType",
        "response_keys": ["code", "data", "msg"],
    })
    monkeypatch.setattr(recon, "_save_income_pending_state", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    report = recon.run_reconciliation(now_ms=now_ms, data_dir=data, notify=False)

    codes = [item["code"] for item in report["discrepancies"]]
    assert "DAILY_INCOME_PENDING_STATE_UNAVAILABLE" in codes
    assert "DAILY_INCOME_QUERY_INCOMPLETE" in codes
    summary = report["checks"]["daily_exchange_income_aggregates"]["symbols_checked"]["BTC-USDT"]
    assert summary["status"] == "ERROR"
    assert summary["accounting_value"] == "UNKNOWN_NOT_ZERO"
    assert not any(item["status"] == "PENDING" for item in report["checks"]["daily_exchange_income_aggregates"]["symbols_checked"].values())


def test_corrupt_income_pending_state_fails_closed_without_overwrite(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    now_ms = _now_ms()
    (data / "active_trades.json").write_text("{}", encoding="utf-8")
    (data / "trades.jsonl").write_text(json.dumps({
        "record_type": "TRADE_OPEN", "event_id": "EVT-INCOME-CORRUPT", "symbol": "BTC",
        "direction": "LONG", "ts": now_ms,
    }) + "\n", encoding="utf-8")
    pending_path = data / "reconciliation_income_pending.json"
    original = b'{"version":99,"pending":{}}'
    pending_path.write_bytes(original)
    monkeypatch.setattr(recon, "INCOME_RETRIES", 0)
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "error", "error": "income_history_unexpected_data_shape", "complete": False,
        "records": [], "response_code": 0, "response_data_type": "NoneType",
        "response_keys": ["code", "data", "msg"],
    })

    report = recon.run_reconciliation(now_ms=now_ms, data_dir=data, notify=False)

    codes = {item["code"] for item in report["discrepancies"]}
    assert "DAILY_INCOME_PENDING_STATE_UNAVAILABLE" in codes
    assert "DAILY_INCOME_QUERY_INCOMPLETE" in codes
    assert pending_path.read_bytes() == original


def test_trump_legacy_exchange_symbol_is_canonicalized_without_merging_sol():
    assert recon._canonical_position_symbol("TRUMP") == "TRUMP-USDT"
    assert recon._canonical_position_symbol("TRUMPUSDT") == "TRUMP-USDT"
    assert recon._canonical_position_symbol("TRUMP-USDT") == "TRUMP-USDT"
    assert recon._canonical_position_symbol("TRUMPSOL-USDT") == "TRUMP-USDT"
    assert recon._canonical_position_symbol("TRUMPSOLUSDT") == "TRUMP-USDT"
    assert recon._canonical_position_symbol("SOL") == "SOL-USDT"
    assert recon._canonical_position_symbol("SOL-USDT") == "SOL-USDT"
    assert recon._canonical_position_symbol("TRUMP-SOL-USDT") == "TRUMP-SOL-USDT"


def test_bingx_display_name_is_canonicalized_from_loaded_contract_catalog(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "CACHE", {
        "ts": 1_800_000_000.0,
        "data": {"NIULAI-USDT": {"symbol": "NIULAI-USDT", "displayName": "牛来"}},
        "by_display_name": {"牛来": {"symbol": "NIULAI-USDT", "displayName": "牛来"}},
    })

    assert recon._canonical_position_symbol("牛来") == "NIULAI-USDT"
    assert recon._canonical_position_symbol("牛来-USDT") == "NIULAI-USDT"
    assert recon._canonical_position_symbol("NIULAI-USDT") == "NIULAI-USDT"


def test_reconciliation_loads_catalog_once_to_match_display_name_and_api_symbol(tmp_path, monkeypatch):
    from event_engine import bingx

    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text(json.dumps({
        "EVT-NIULAI": {
            "symbol": "牛来", "direction": "SHORT", "remaining_qty": 144.4,
            "closed": False,
            "sl_order": {"order_id": "SL-N", "stop_price": 0.074, "qty": 144.4},
            "tp_orders": [{"leg": "tp1", "order_id": "TP-N", "price": 0.066, "qty": 144.4}],
        },
    }, ensure_ascii=False), encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")

    catalog_calls = []
    def load_catalog():
        catalog_calls.append(True)
        bingx.CACHE.update(
            ts=1_800_000_000.0,
            data={"NIULAI-USDT": {"symbol": "NIULAI-USDT", "displayName": "牛来"}},
            by_display_name={"牛来": {"symbol": "NIULAI-USDT", "displayName": "牛来"}},
        )
        return bingx.CACHE["data"]

    monkeypatch.setattr(recon.bingx, "contracts", load_catalog)
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [{
        "symbol": "NIULAI-USDT", "positionSide": "SHORT", "positionAmt": "144.4",
    }])
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **kwargs: {"status": "ok", "orders": [
        {"symbol": "NIULAI-USDT", "orderId": "SL-N", "type": "STOP_MARKET", "stopPrice": "0.074", "positionSide": "SHORT", "side": "BUY", "origQty": "144.4"},
        {"symbol": "NIULAI-USDT", "orderId": "TP-N", "type": "TAKE_PROFIT_MARKET", "stopPrice": "0.066", "positionSide": "SHORT", "side": "BUY", "origQty": "144.4"},
    ]})
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    assert catalog_calls == [True]
    assert report["checks"]["bingx_contract_catalog"]["status"] == "OK"
    assert not any(item["code"] == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH" for item in report["discrepancies"])
    assert report["checks"]["exchange_positions"]["active_position_keys"] == 1




def test_reconciliation_keeps_non_latin_identity_unknown_if_catalog_cannot_resolve(tmp_path, monkeypatch):
    from event_engine import bingx

    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text(json.dumps({
        "EVT-NIULAI": {"symbol": "牛来", "direction": "SHORT", "remaining_qty": 144.4, "closed": False},
    }, ensure_ascii=False), encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(bingx, "CACHE", {"ts": 0, "data": {}, "by_display_name": {}})
    catalog_calls = []
    monkeypatch.setattr(recon.bingx, "contracts", lambda: catalog_calls.append(True) or {})
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [{
        "symbol": "NIULAI-USDT", "positionSide": "SHORT", "positionAmt": -144.4,
    }])
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **kwargs: {"status": "ok", "orders": []})
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    assert catalog_calls == [True]
    assert any(item["code"] == "POSITION_SYMBOL_IDENTITY_UNRESOLVED" for item in report["discrepancies"])
    assert not any(item["code"] == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH" for item in report["discrepancies"])
    assert report["checks"]["exchange_protection"]["by_position"]["NIULAI-USDT:SHORT"]["status"] == "UNKNOWN"



def test_catalog_unavailable_does_not_hide_unmatched_ascii_position(tmp_path, monkeypatch):
    from event_engine import bingx

    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text(json.dumps({
        "EVT-LOCAL": {"symbol": "LOCALCOIN", "direction": "SHORT", "remaining_qty": 2.0, "closed": False},
    }), encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(bingx, "CACHE", {"ts": 0, "data": {}, "by_display_name": {}})
    monkeypatch.setattr(recon.bingx, "contracts", lambda: {})
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [{
        "symbol": "OTHERCOIN-USDT", "positionSide": "SHORT", "positionAmt": -2.0,
    }])
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **kwargs: {"status": "ok", "orders": []})
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    assert any(item["code"] == "POSITION_SYMBOL_CATALOG_UNAVAILABLE" for item in report["discrepancies"])
    mismatch = [item for item in report["discrepancies"] if item["code"] == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH"]
    assert len(mismatch) == 2
    assert {item["symbol"] for item in mismatch} == {"LOCALCOIN-USDT", "OTHERCOIN-USDT"}


def test_active_local_positions_excludes_confirmed_closed_lifecycle_records():
    active = {
        "LIVE": {"symbol": "BTC", "direction": "SHORT", "remaining_qty": 2.0, "closed": False},
        "CLEANUP": {"symbol": "ETH", "direction": "SHORT", "remaining_qty": 0.5,
                    "closed": False, "close_cleanup_pending": True},
        "NOTIFY": {"symbol": "ADA", "direction": "LONG", "remaining_qty": 13.0,
                   "closed": False, "close_notification_pending": True},
        "JOURNAL": {"symbol": "DOT", "direction": "SHORT", "remaining_qty": 7.8,
                    "closed": False, "close_journal_pending": True},
        "GONE": {"symbol": "LTC", "direction": "SHORT", "remaining_qty": 0.1,
                 "closed": False, "position_gone_confirmation": {"status": "CONFIRMED"}},
        "QUARANTINE": {"symbol": "XRP", "direction": "SHORT", "remaining_qty": 0.0,
                       "closed": False, "protection_cleanup_only": True},
    }

    positions, errors = recon._active_local_positions(active)

    assert positions == {("BTC-USDT", "SHORT"): 2.0}
    assert errors == []
    assert len(recon._post_close_lifecycle_pending(active)) == 5


def test_daily_reconciliation_classifies_exact_unapplied_tp_fill_as_warning(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text(json.dumps({
        "EVT-HBAR": {
            "symbol": "HBAR", "direction": "SHORT", "remaining_qty": 139.0,
            "initial_qty": 139.0, "closed": False, "tp_filled_qty": {},
            "tp_orders": [
                {"leg": "tp1", "order_id": "TP1-HBAR"},
                {"leg": "tp2", "order_id": "TP2-HBAR"},
            ],
            "sl_order": {"order_id": "SL-HBAR"},
        },
    }), encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "SYMBOL_MAP", {})
    position_reads = []
    def read_positions(**kwargs):
        position_reads.append(kwargs)
        return [{"symbol": "HBAR-USDT", "positionSide": "SHORT", "positionAmt": -104.0}]
    monkeypatch.setattr(recon.bingx, "get_positions", read_positions)
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **kwargs: {"status": "ok", "orders": [
        {"symbol": "HBAR-USDT", "orderId": "SL-HBAR", "type": "STOP_MARKET", "stopPrice": "0.08",
         "positionSide": "SHORT", "origQty": "104"},
        {"symbol": "HBAR-USDT", "orderId": "TP2-HBAR", "type": "TAKE_PROFIT_MARKET",
         "positionSide": "SHORT", "origQty": "104"},
    ]})
    def read_order(symbol, order_id, **kwargs):
        assert kwargs.get("timeout_sec", 0) <= recon.TRACKER_FILL_ORDER_TIMEOUT_SEC
        return {
            "status": "ok", "order_id": order_id,
            "order_status": "FILLED" if order_id == "TP1-HBAR" else "NEW",
            "executed_qty": 35.0 if order_id == "TP1-HBAR" else 0.0,
            "avg_price": 0.06839 if order_id == "TP1-HBAR" else 0.0,
        }
    monkeypatch.setattr(recon.bingx, "get_order", read_order)
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    assert not any(item["code"] == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH" for item in report["discrepancies"])
    pending = [item for item in report["discrepancies"] if item["code"] == "TRACKER_EXIT_FILL_PENDING"]
    assert len(pending) == 1
    assert pending[0]["confirmed_unapplied_fill_qty"] == 35.0
    assert pending[0]["local_qty"] == 139.0
    assert pending[0]["exchange_qty"] == 104.0
    assert pending[0]["fill_evidence"][0]["order_id"] == "TP1-HBAR"
    assert len(position_reads) == 2  # the post-fill evidence must be checked against a fresh position snapshot
    assert report["checks"]["tracker_exit_fill_reconciliation"]["position_refresh_status"] == "OK"
    assert report["status"] == "WARNING"


def test_daily_reconciliation_keeps_critical_when_confirmed_tp_fill_does_not_explain_delta(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text(json.dumps({
        "EVT-HBAR": {
            "symbol": "HBAR", "direction": "SHORT", "remaining_qty": 139.0,
            "closed": False, "tp_filled_qty": {},
            "tp_orders": [{"leg": "tp1", "order_id": "TP1-HBAR"}],
        },
    }), encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "SYMBOL_MAP", {})
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [{
        "symbol": "HBAR-USDT", "positionSide": "SHORT", "positionAmt": -104.0,
    }])
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **kwargs: {"status": "ok", "orders": [
        {"symbol": "HBAR-USDT", "orderId": "SL-HBAR", "type": "STOP_MARKET", "stopPrice": "0.08",
         "positionSide": "SHORT", "origQty": "104"},
        {"symbol": "HBAR-USDT", "orderId": "TP-HBAR", "type": "TAKE_PROFIT_MARKET",
         "positionSide": "SHORT", "origQty": "104"},
    ]})
    monkeypatch.setattr(recon.bingx, "get_order", lambda *_a, **_k: {
        "status": "ok", "order_status": "FILLED", "executed_qty": 25.0,
    })
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    mismatch = [item for item in report["discrepancies"] if item["code"] == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH"]
    assert len(mismatch) == 1
    assert mismatch[0]["level"] == "CRITICAL"
    assert mismatch[0]["confirmed_unapplied_fill_qty"] == 25.0
    assert report["status"] == "CRITICAL"


def test_pending_exit_fill_evidence_does_not_start_request_below_transport_minimum(tmp_path, monkeypatch):
    active = {
        "EVT-HBAR": {
            "symbol": "HBAR", "direction": "SHORT", "remaining_qty": 139.0,
            "closed": False, "tp_filled_qty": {},
            "tp_orders": [{"leg": "tp1", "order_id": "TP1-HBAR"}],
        },
    }
    queried = []
    monkeypatch.setattr(recon.bingx, "get_order", lambda *a, **k: queried.append((a, k)))
    monkeypatch.setattr(recon.time, "monotonic", lambda: 100.0)
    state = {
        "queries_used": 0,
        "query_budget": 5,
        "started_monotonic": 100.0,
        "time_budget_sec": 0.5,
    }

    qty, evidence, exhausted = recon._pending_exit_fill_evidence(
        active, ("HBAR-USDT", "SHORT"), 35.0, state,
    )

    assert qty == 0.0
    assert evidence == []
    assert exhausted is True
    assert state["queries_used"] == 0
    assert queried == []


def test_daily_reconciliation_keeps_critical_when_fresh_position_read_disagrees_with_fill(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text(json.dumps({
        "EVT-HBAR": {
            "symbol": "HBAR", "direction": "SHORT", "remaining_qty": 139.0,
            "closed": False, "tp_filled_qty": {},
            "tp_orders": [{"leg": "tp1", "order_id": "TP1-HBAR"}],
        },
    }), encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "SYMBOL_MAP", {})
    position_reads = iter((
        [{"symbol": "HBAR-USDT", "positionSide": "SHORT", "positionAmt": -104.0}],
        [{"symbol": "HBAR-USDT", "positionSide": "SHORT", "positionAmt": -100.0}],
    ))
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: next(position_reads))
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **kwargs: {"status": "ok", "orders": []})
    monkeypatch.setattr(recon.bingx, "get_order", lambda *_a, **_k: {
        "status": "ok", "order_status": "FILLED", "executed_qty": 35.0,
    })
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    mismatch = [item for item in report["discrepancies"] if item["code"] == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH"]
    assert len(mismatch) == 1
    assert mismatch[0]["level"] == "CRITICAL"
    assert mismatch[0]["confirmed_unapplied_fill_qty"] == 35.0
    assert mismatch[0]["exchange_qty"] == 100.0
    assert "latest exchange position" in mismatch[0]["detail"]
    assert report["checks"]["tracker_exit_fill_reconciliation"]["position_refresh_status"] == "OK"


def test_daily_reconciliation_keeps_critical_when_post_fill_position_refresh_fails(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text(json.dumps({
        "EVT-HBAR": {
            "symbol": "HBAR", "direction": "SHORT", "remaining_qty": 139.0,
            "closed": False, "tp_filled_qty": {},
            "tp_orders": [{"leg": "tp1", "order_id": "TP1-HBAR"}],
        },
    }), encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "SYMBOL_MAP", {})
    position_reads = iter((
        [{"symbol": "HBAR-USDT", "positionSide": "SHORT", "positionAmt": -104.0}],
        RuntimeError("position endpoint unavailable during refresh"),
    ))

    def read_positions(**kwargs):
        value = next(position_reads)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(recon.bingx, "get_positions", read_positions)
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **kwargs: {"status": "ok", "orders": []})
    monkeypatch.setattr(recon.bingx, "get_order", lambda *_a, **_k: {
        "status": "ok", "order_status": "FILLED", "executed_qty": 35.0,
    })
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    mismatch = [item for item in report["discrepancies"] if item["code"] == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH"]
    assert len(mismatch) == 1
    assert mismatch[0]["level"] == "CRITICAL"
    assert mismatch[0]["position_refresh_error"].startswith("RuntimeError:")
    assert report["checks"]["tracker_exit_fill_reconciliation"]["position_refresh_status"] == "ERROR"
    assert report["status"] == "CRITICAL"


def test_daily_reconciliation_classifies_filled_break_even_stop_as_pending_exit(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text(json.dumps({
        "EVT-STRK": {
            "symbol": "STRK", "direction": "SHORT", "remaining_qty": 174.094,
            "closed": False, "tp_filled_qty": {"tp1": 58.032},
            "tp_orders": [{"leg": "tp1", "order_id": "TP1-STRK"}],
            "sl_order": {"order_id": "BE-SL-STRK"},
        },
    }), encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "SYMBOL_MAP", {})
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **kwargs: {"status": "ok", "orders": []})
    queried = []
    def order_lookup(symbol, order_id, **kwargs):
        queried.append(order_id)
        if order_id == "TP1-STRK":
            return {"status": "ok", "order_status": "FILLED", "executed_qty": 58.032}
        if order_id == "BE-SL-STRK":
            return {"status": "ok", "order_status": "FILLED", "executed_qty": 174.094}
        return {"status": "error"}
    monkeypatch.setattr(recon.bingx, "get_order", order_lookup)
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    pending = [item for item in report["discrepancies"] if item["code"] == "TRACKER_EXIT_FILL_PENDING"]
    assert len(pending) == 1
    assert pending[0]["confirmed_unapplied_fill_qty"] == 174.094
    assert [item["source"] for item in pending[0]["fill_evidence"]] == ["SL"]
    assert queried == ["TP1-STRK", "BE-SL-STRK"]
    assert not any(item["code"] == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH" for item in report["discrepancies"])


def test_entry_resolution_quarantine_is_not_mistaken_for_confirmed_close():
    active = {
        "ENTRY_PENDING": {
            "symbol": "XRP", "direction": "SHORT", "remaining_qty": 1.0,
            "closed": False, "entry_order_resolution_pending": True,
            "close_cleanup_pending": True,
        },
    }
    positions, errors = recon._active_local_positions(active)
    assert positions == {("XRP-USDT", "SHORT"): 1.0}
    assert errors == []


def test_daily_reconciliation_keeps_close_lifecycle_visible_without_counting_as_position(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text(json.dumps({
        "EVT-CLOSED": {
            "symbol": "LTC", "direction": "SHORT", "remaining_qty": 0.1,
            "closed": False, "close_cleanup_pending": True,
        },
    }), encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "SYMBOL_MAP", {})
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **kwargs: {"status": "ok", "orders": []})
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    assert report["checks"]["local_active_state"]["active_position_keys"] == 0
    assert report["checks"]["local_active_state"]["post_close_lifecycle_pending_records"] == 1
    assert not any(item["code"] == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH" for item in report["discrepancies"])
    assert any(item["code"] == "LOCAL_CLOSE_LIFECYCLE_PENDING" for item in report["discrepancies"])
    assert report["status"] == "WARNING"


def test_daily_reconciliation_respects_bounded_tracker_order_query_budget(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "active_trades.json").write_text(json.dumps({
        "EVT-HBAR": {
            "symbol": "HBAR", "direction": "SHORT", "remaining_qty": 139.0,
            "closed": False, "tp_filled_qty": {},
            "tp_orders": [
                {"leg": "tp2", "order_id": "TP2-HBAR"},
                {"leg": "tp1", "order_id": "TP1-HBAR"},
            ],
        },
    }), encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(recon, "MAX_TRACKER_FILL_ORDER_QUERIES", 1)
    monkeypatch.setattr(recon.bingx, "SYMBOL_MAP", {})
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [{
        "symbol": "HBAR-USDT", "positionSide": "SHORT", "positionAmt": -104.0,
    }])
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **kwargs: {"status": "ok", "orders": [
        {"symbol": "HBAR-USDT", "orderId": "SL-HBAR", "type": "STOP_MARKET", "stopPrice": "0.08",
         "positionSide": "SHORT", "origQty": "104"},
        {"symbol": "HBAR-USDT", "orderId": "TP2-HBAR", "type": "TAKE_PROFIT_MARKET",
         "positionSide": "SHORT", "origQty": "104"},
    ]})
    queried = []
    def order_lookup(symbol, order_id, **kwargs):
        queried.append(order_id)
        return {"status": "ok", "order_status": "NEW", "executed_qty": 0.0}
    monkeypatch.setattr(recon.bingx, "get_order", order_lookup)
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })

    report = recon.run_reconciliation(now_ms=_now_ms(), data_dir=data, notify=False)

    assert queried == ["TP1-HBAR"]
    assert report["checks"]["tracker_exit_fill_reconciliation"]["order_queries_used"] == 1
    assert any(item["code"] == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH" for item in report["discrepancies"])
    mismatch = next(item for item in report["discrepancies"] if item["code"] == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH")
    assert "budget was exhausted" in mismatch["detail"]


def test_daily_income_does_not_fallback_for_nonzero_code_even_when_data_is_null(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    now_ms = _now_ms()
    (data / "active_trades.json").write_text("{}", encoding="utf-8")
    (data / "trades.jsonl").write_text(json.dumps({
        "record_type": "TRADE_OPEN", "event_id": "EVT-BTC", "symbol": "BTC",
        "direction": "LONG", "ts": now_ms,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(recon, "MAX_INCOME_SYMBOLS", 1)
    monkeypatch.setattr(recon, "INCOME_RETRIES", 0)
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", _stub_snapshot)
    calls = []
    def income(symbol, *args, **kwargs):
        calls.append(symbol)
        return {"status": "error", "complete": False, "records": [],
                "error": "100410", "response_code": 100410, "response_data_type": "NoneType"}
    monkeypatch.setattr(recon.bingx, "get_income_history", income)
    report = recon.run_reconciliation(now_ms=now_ms, data_dir=data, notify=False)
    assert calls == ["BTC-USDT"]
    summary = report["checks"]["daily_exchange_income_aggregates"]["symbols_checked"]["BTC-USDT"]
    assert summary["status"] != "OK"
    assert any(item["code"] == "DAILY_INCOME_QUERY_INCOMPLETE" for item in report["discrepancies"])


def test_daily_income_uses_one_account_wide_fallback_and_filters_by_real_symbol(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    now_ms = _now_ms()
    (data / "active_trades.json").write_text("{}", encoding="utf-8")
    (data / "trades.jsonl").write_text(json.dumps({
        "record_type": "TRADE_OPEN", "event_id": "EVT-BTC", "symbol": "BTC",
        "direction": "LONG", "ts": now_ms,
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(recon, "MAX_INCOME_SYMBOLS", 1)
    monkeypatch.setattr(recon, "INCOME_RETRIES", 0)
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", _stub_snapshot)
    calls = []

    def income(symbol, *args, **kwargs):
        calls.append(symbol)
        if symbol is not None:
            return {
                "status": "error", "complete": False, "records": [],
                "error": "income_history_unexpected_data_shape",
                "response_data_type": "NoneType", "response_code": 0,
                "response_keys": ["code", "data", "msg"],
            }
        return {
            "status": "ok", "complete": True, "symbol": None,
            "query_scope": "ACCOUNT_WIDE", "records": [
                {"symbol": "BTC-USDT", "asset": "USDT", "income_type": "TRADING_FEE", "amount": -0.12, "time": now_ms},
                {"symbol": "ETH-USDT", "asset": "USDT", "income_type": "TRADING_FEE", "amount": -5.0, "time": now_ms},
                {"symbol": "", "asset": "USDT", "income_type": "TRANSFER", "amount": 100.0, "time": now_ms},
            ],
        }

    monkeypatch.setattr(recon.bingx, "get_income_history", income)
    report = recon.run_reconciliation(now_ms=now_ms, data_dir=data, notify=False)

    assert calls == ["BTC-USDT", None]
    summary = report["checks"]["daily_exchange_income_aggregates"]["symbols_checked"]["BTC-USDT"]
    assert summary["status"] == "OK"
    assert summary["trading_fees"] == pytest.approx(-0.12)
    assert summary["source_scope"].startswith("ACCOUNT_WIDE_FILTERED_BY_SYMBOL")
    assert not any(item["code"] == "DAILY_INCOME_QUERY_INCOMPLETE" for item in report["discrepancies"])


def test_unresolved_data_error_from_previous_utc_day_remains_critical_with_recovery_detail(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    now_ms = int(datetime(2026, 11, 6, 11, 0, tzinfo=timezone.utc).timestamp() * 1000)
    previous_day_close = int(datetime(2026, 11, 5, 19, 0, tzinfo=timezone.utc).timestamp() * 1000)
    (data / "active_trades.json").write_text("{}", encoding="utf-8")
    (data / "trades.jsonl").write_text(json.dumps({
        "record_type": "TRADE_CLOSE", "event_id": "EVT-VIRTUAL", "symbol": "VIRTUAL",
        "direction": "SHORT", "closed_ts": previous_day_close, "exit_reason": "DATA_ERROR",
        "data_error": "close_fill_not_confirmed", "exit_order_ids": ["SL-REPLACED"],
        "fill_history_recovery": {
            "status": "PENDING", "attempts": 4,
            "last_error": "incomplete_close_fills:4.235/16.94", "next_retry_ts": now_ms + 1000,
        },
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [])
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", _stub_snapshot)
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *a, **k: {
        "status": "ok", "complete": True, "records": [],
    })

    report = recon.run_reconciliation(now_ms=now_ms, data_dir=data, notify=False)

    item = next(x for x in report["discrepancies"] if x["code"] == "DATA_ERROR_CLOSES_UNRESOLVED")
    assert item["count"] == 1
    assert item["trades"][0]["symbol"] == "VIRTUAL"
    assert item["trades"][0]["fill_history_recovery"]["last_error"].startswith("incomplete_close_fills")
    detail = recon._discrepancy_message_detail(item)
    assert "VIRTUAL SHORT" in detail
    assert "EVT-VIRTUAL" in detail
    assert "incomplete_close_fills" in detail
    assert report["status"] == "CRITICAL"
