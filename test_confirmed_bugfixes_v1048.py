"""Regression tests for independently verified V10.4.8 findings."""
from __future__ import annotations

import json

import pytest


def _seed_contract(bx, symbol="TEST-USDT"):
    contract = {"symbol": symbol, "displayName": symbol, "status": 1, "apiStateOpen": "true", "quantityPrecision": 3, "pricePrecision": 2}
    bx.CACHE["data"] = {symbol: contract}
    bx.CACHE["by_display_name"] = {symbol: contract}
    bx.CACHE["ts"] = 9_999_999_999


def test_live_price_refuses_other_symbol_in_list_or_dict(monkeypatch):
    from event_engine import bingx as bx

    _seed_contract(bx, "BTC-USDT")
    bx._LIVE_PRICE_INVALID_UNTIL.clear()
    bx._LIVE_PRICE_COOLDOWN_UNTIL = 0.0
    responses = [
        {"code": 0, "data": [{"symbol": "ETH-USDT", "price": "2000"}]},
        {"code": 0, "data": {"symbol": "ETH-USDT", "price": "2000"}},
    ]
    monkeypatch.setattr(bx, "_request", lambda *a, **k: responses.pop(0))

    assert bx.get_live_price("BTC-USDT") is None
    assert bx.get_live_price("BTC-USDT") is None


def test_protection_reader_rejects_protective_order_without_symbol_even_with_order_id(monkeypatch):
    from event_engine import bingx as bx

    _seed_contract(bx, "BTC-USDT")
    monkeypatch.setattr(bx, "_request", lambda *a, **k: {"code": 0, "data": [{
        "orderId": "STOP-OTHER-SYMBOL", "type": "STOP_MARKET", "positionSide": "LONG",
        "stopPrice": "95", "origQty": "1",
    }]})
    out = bx.get_open_protection_directional("BTC-USDT", "LONG")
    assert out["status"] == "error"
    assert "missing symbol" in out["error"]
    assert out["sl_orders"] == []


def test_posted_be_stop_verification_failure_does_not_restore_old_stop(monkeypatch):
    from event_engine import tracker as tr

    snapshots = iter([
        {"status": "ok", "sl_orders": [{"orderId": "OLD1", "stopPrice": "97.00", "origQty": "1.000"}], "tp_orders": []},
        {"status": "ok", "sl_orders": [], "tp_orders": []},  # old stop cancellation verified
        {"status": "error", "error": "openOrders timeout after POST"},
    ])
    posted = []
    alerts = []
    monkeypatch.setattr(tr, "to_bx_symbol", lambda _s: "TEST-USDT")
    monkeypatch.setattr(tr, "get_contract", lambda _s: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tr, "get_position_directional", lambda *a, **k: {"status": "found", "positionAmt": "1"})
    monkeypatch.setattr(tr, "get_open_protection_directional", lambda *a, **k: next(snapshots))
    monkeypatch.setattr(tr, "cancel_order", lambda *a, **k: {"code": 0})
    monkeypatch.setattr(tr, "_post_protection_order_verified", lambda *a, **k: posted.append(1) or {"code": 0, "data": {"order": {"orderId": "NEW-BE"}}})
    monkeypatch.setattr(tr, "_notify_be_failure", lambda *a, **k: alerts.append((a, k)))
    monkeypatch.setattr(tr.time, "sleep", lambda _s: None)

    out = tr._move_sl_to_break_even("TEST-USDT", "LONG", 100.0, 1.0, "OLD1", "T", old_sl_price=97.0)

    assert out["status"] == "error"
    assert out["new_order_may_exist"] is True
    assert out["order_id"] == "NEW-BE"
    assert posted == [1], "ambiguous verification must not POST a second restore SL"
    assert alerts


def test_rollback_protection_cleanup_requires_confirmed_absence_and_cancels_orphans(monkeypatch):
    from event_engine import bingx as bx

    _seed_contract(bx, "TEST-USDT")
    monkeypatch.setattr(bx, "to_bx_symbol", lambda _s: "TEST-USDT")
    states = iter([{"status": "not_found"}, {"status": "not_found"}])
    monkeypatch.setattr(bx, "get_position_directional", lambda *a, **k: next(states))
    snapshots = iter([
        {"status": "ok", "sl_orders": [{"orderId": "ORPHAN-SL"}], "tp_orders": [{"orderId": "ORPHAN-TP"}]},
        {"status": "ok", "sl_orders": [], "tp_orders": []},
    ])
    monkeypatch.setattr(bx, "get_open_protection_directional", lambda *a, **k: next(snapshots))
    cancelled = []
    monkeypatch.setattr(bx, "_cancel_protection_order_verified", lambda symbol, direction, order, **kwargs: cancelled.append(order["orderId"]) or (True, "cancelled"))
    monkeypatch.setattr(bx.time, "sleep", lambda _s: None)

    out = bx.cleanup_directional_protection_after_close("TEST-USDT", "LONG")

    assert out["status"] == "cleaned"
    assert set(cancelled) == {"ORPHAN-SL", "ORPHAN-TP"}
    assert set(out["cancelled_order_ids"]) == set(cancelled)


def test_cleanup_retry_does_not_cancel_protection_after_only_one_not_found(monkeypatch):
    from event_engine import tracker as tr

    states = iter([{"status": "not_found"}, {"status": "error", "error": "position query timeout"}])
    monkeypatch.setattr(tr, "get_position_directional", lambda *a, **k: next(states))
    monkeypatch.setattr(tr, "get_open_protection_directional", lambda *a, **k: pytest.fail("must not query/cancel protection without two NOT_FOUND reads"))
    monkeypatch.setattr(tr.time, "sleep", lambda _s: None)

    ok, reason = tr._cleanup_closed_trade_protection({"symbol": "TEST-USDT", "direction": "LONG", "protection_cleanup_only": True})
    assert ok is False
    assert "not proven closed" in reason


def test_pending_rollback_cleanup_is_durable_and_blocks_same_side_entry(tmp_path, monkeypatch):
    from event_engine import tracker as tr
    import run_once as ro

    state_path = tmp_path / "active_trades.json"
    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", state_path)
    monkeypatch.setattr(tr, "to_bx_symbol", lambda symbol: "LIGHTER-USDT" if str(symbol).upper().startswith("LIT") else "TEST-USDT")
    pending_id = tr.queue_protection_cleanup_pending("TEST-USDT", "SHORT", "cancel timeout", source_event_id="EVT1")
    stored = json.loads(state_path.read_text(encoding="utf-8"))
    assert stored[pending_id]["close_cleanup_pending"] is True
    assert stored[pending_id]["protection_cleanup_only"] is True
    assert stored[pending_id]["remaining_qty"] == 0.0

    monkeypatch.setattr(ro, "_load_active_trades", lambda: stored)
    monkeypatch.setattr(ro, "_symbol_identity_for_state", lambda symbol: "TEST-USDT")
    monkeypatch.setattr(ro, "open_market", lambda *a, **k: pytest.fail("entry must be blocked until protection cleanup succeeds"))
    out = ro.execute_new_position("TEST-USDT", "SHORT", 1.0, {"risk_pct": 1.0}, "EVT2")
    assert out["status"] == "PROTECTION_CLEANUP_PENDING"
    assert out["pending_cleanup_id"] == pending_id


def test_tracker_removes_cleanup_pending_trade_after_successful_cleanup(tmp_path, monkeypatch):
    from event_engine import tracker as tr

    path = tmp_path / "active_trades.json"
    path.write_text(json.dumps({"CLEANUP1": {
        "symbol": "TEST-USDT", "direction": "LONG", "closed": False,
        "remaining_qty": 0.0, "close_cleanup_pending": True,
        "protection_cleanup_only": True,
    }}), encoding="utf-8")
    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", path)
    monkeypatch.setattr(tr, "_recover_data_error_trade_closes", lambda: None)
    monkeypatch.setattr(tr, "_recover_trade_cashflows", lambda **kwargs: None)
    monkeypatch.setattr(tr, "_retry_pending_notifications", lambda: None)
    monkeypatch.setattr(tr, "_cleanup_closed_trade_protection", lambda *a, **k: (True, "cleanup_verified"))

    tr.update_active_trades()

    assert json.loads(path.read_text(encoding="utf-8")) == {}


def test_reconciliation_symbol_alias_maps_lit_to_lighter_without_network(monkeypatch):
    from event_engine import reconcile_daily as recon

    monkeypatch.setattr(recon.bingx, "SYMBOL_MAP", {})
    monkeypatch.setattr(recon.bingx, "to_bx_symbol", lambda *_a, **_k: pytest.fail("reconciliation symbol normalization must not perform network contract lookup"))
    assert recon._canonical_position_symbol("LIT-USDT") == "LIGHTER-USDT"
    assert recon._canonical_position_symbol("LIGHTER-USDT") == "LIGHTER-USDT"


def _reconciliation_fixture(tmp_path, monkeypatch, *, local_symbol, local_qty, exchange_symbol, exchange_qty):
    from event_engine import reconcile_daily as recon
    from datetime import datetime, timezone

    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    now_ms = int(datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    (data / "active_trades.json").write_text(json.dumps({
        "EVT-LOCAL": {
            "symbol": local_symbol, "direction": "SHORT", "remaining_qty": local_qty,
            "closed": False,
        },
    }), encoding="utf-8")
    (data / "trades.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(recon.bingx, "SYMBOL_MAP", {})
    monkeypatch.setattr(recon.bingx, "get_positions", lambda **kwargs: [{
        "symbol": exchange_symbol, "positionSide": "SHORT", "positionAmt": -abs(exchange_qty),
    }])
    monkeypatch.setattr(recon.bingx, "get_open_orders_snapshot", lambda **kwargs: {
        "status": "ok", "orders": [
            {"symbol": exchange_symbol, "orderId": "SL-1", "type": "STOP_MARKET", "stopPrice": "101",
             "positionSide": "SHORT", "origQty": str(abs(exchange_qty))},
            {"symbol": exchange_symbol, "orderId": "TP-1", "type": "TAKE_PROFIT_MARKET",
             "positionSide": "SHORT", "origQty": str(abs(exchange_qty))},
        ],
    })
    monkeypatch.setattr(recon.bingx, "get_income_history", lambda *args, **kwargs: {
        "status": "ok", "complete": True, "records": [],
    })
    return recon, data, now_ms


def test_daily_reconciliation_does_not_false_critical_on_lit_lighter_alias(tmp_path, monkeypatch):
    """Reproduce the actual 3.311 LIT/LIGHTER mismatch from the user log."""
    recon, data, now_ms = _reconciliation_fixture(
        tmp_path, monkeypatch, local_symbol="LIT-USDT", local_qty=3.311,
        exchange_symbol="LIGHTER-USDT", exchange_qty=3.311,
    )
    report = recon.run_reconciliation(now_ms=now_ms, data_dir=data, notify=False)
    mismatches = [item for item in report["discrepancies"] if item["code"] == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH"]
    assert mismatches == []
    assert report["checks"]["local_active_state"]["active_position_keys"] == 1
    assert report["checks"]["exchange_positions"]["active_position_keys"] == 1
    assert report["status"] == "OK"


def test_reconciliation_normalizes_compact_usdt_symbols_without_network(monkeypatch):
    from event_engine import reconcile_daily as recon

    monkeypatch.setattr(recon.bingx, "SYMBOL_MAP", {})
    monkeypatch.setattr(recon.bingx, "to_bx_symbol", lambda *_a, **_k: pytest.fail("must not perform network catalog lookup"))
    assert recon._symbol("BTCUSDT") == "BTC-USDT"
    assert recon._canonical_position_symbol("LITUSDT") == "LIGHTER-USDT"


def test_reconciliation_suppresses_only_unchanged_successfully_delivered_alert(tmp_path, monkeypatch):
    recon, data, now_ms = _reconciliation_fixture(
        tmp_path, monkeypatch, local_symbol="BTC-USDT", local_qty=1.0,
        exchange_symbol="BTC-USDT", exchange_qty=2.0,
    )
    alerts = []
    monkeypatch.setattr(recon, "send_detailed", lambda text: alerts.append(text) or {"chat": {"sent": True}})

    first = recon.run_reconciliation(now_ms=now_ms, data_dir=data, notify=True)
    second = recon.run_reconciliation(now_ms=now_ms + 30_000, data_dir=data, notify=True)

    assert first["notification"]["status"] == "SENT"
    assert second["notification"]["status"] == "SUPPRESSED_REPEAT"
    assert len(alerts) == 1
    assert "local_qty=1.0 exchange_qty=2.0 delta=1.0" in alerts[0]


def test_reconciliation_retries_alert_after_unconfirmed_telegram_delivery(tmp_path, monkeypatch):
    recon, data, now_ms = _reconciliation_fixture(
        tmp_path, monkeypatch, local_symbol="BTC-USDT", local_qty=1.0,
        exchange_symbol="BTC-USDT", exchange_qty=2.0,
    )
    alerts = []
    monkeypatch.setattr(recon, "send_detailed", lambda text: alerts.append(text) or {"chat": {"sent": False, "error": "test failure"}})

    first = recon.run_reconciliation(now_ms=now_ms, data_dir=data, notify=True)
    second = recon.run_reconciliation(now_ms=now_ms + 30_000, data_dir=data, notify=True)

    assert first["notification"]["status"] == "FAILED"
    assert second["notification"]["status"] == "FAILED"
    assert len(alerts) == 2
    assert not (data / "reconciliation_alert_state.json").exists()


def test_ambiguous_be_order_is_not_reposted_when_open_orders_snapshot_does_not_show_it(monkeypatch):
    from event_engine import bingx as bx
    from event_engine import tracker as tr

    _seed_contract(bx)
    monkeypatch.setattr(tr, "to_bx_symbol", lambda _s: "TEST-USDT")
    monkeypatch.setattr(tr, "get_contract", lambda _s: {"quantityPrecision": 3, "pricePrecision": 2})
    monkeypatch.setattr(tr, "get_position_directional", lambda *a, **k: {"status": "found", "positionAmt": "1"})
    monkeypatch.setattr(tr, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok", "sl_orders": [{"orderId": "OLD-SL", "stopPrice": "97", "origQty": "1"}], "tp_orders": [],
    })
    monkeypatch.setattr(tr, "cancel_order", lambda *a, **k: pytest.fail("must not cancel any order while prior BE POST is ambiguous"))
    monkeypatch.setattr(tr, "_post_protection_order_verified", lambda *a, **k: pytest.fail("must not repeat an ambiguous BE POST"))
    monkeypatch.setattr(tr, "_notify_be_failure", lambda *a, **k: None)

    out = tr._move_sl_to_break_even(
        "TEST-USDT", "LONG", 100.0, 1.0, "OLD-SL", "TRADE-1", old_sl_price=97.0,
        prior_order_may_exist=True, prior_order_id="NEW-BE-1",
    )

    assert out["status"] == "error"
    assert out["new_order_may_exist"] is True
    assert out["sl_mutation_attempted"] is False
    assert out["order_id"] == "NEW-BE-1"
