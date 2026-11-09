"""Regression tests for protection races observed in the v13 exchange journal."""
import pytest


def test_missing_tp_that_is_filled_is_deferred_not_recreated(monkeypatch):
    import run_once as ro

    matched_trade = {
        "tp_orders": [
            {"leg": "tp1", "order_id": "TP1-ORIGINAL"},
            {"leg": "tp2", "order_id": "TP2"},
        ],
        "tp_filled_qty": {},
    }
    monkeypatch.setattr(ro, "get_order", lambda symbol, order_id, **kwargs: {
        "status": "ok", "order_id": order_id,
        "order_status": "FILLED", "executed_qty": 2.5,
    })

    result = ro._missing_tp_order_state("PUMP-USDT", matched_trade, {"tp1"})

    assert result["status"] == "fill_pending_tracker"
    assert result["orders"][0]["order_status"] == "FILLED"


def test_missing_tp_is_replaceable_only_after_terminal_zero_fill(monkeypatch):
    import run_once as ro

    matched_trade = {
        "tp_orders": [{"leg": "tp1", "order_id": "TP1-ORIGINAL"}],
        "tp_filled_qty": {},
    }
    monkeypatch.setattr(ro, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "CANCELED", "executed_qty": 0.0,
    })

    result = ro._missing_tp_order_state("PUMP-USDT", matched_trade, {"tp1"})

    assert result["status"] == "safe_to_replace"


def test_missing_tp_with_unknown_order_state_is_not_replaceable(monkeypatch):
    import run_once as ro

    matched_trade = {
        "tp_orders": [{"leg": "tp1", "order_id": "TP1-ORIGINAL"}],
        "tp_filled_qty": {},
    }
    monkeypatch.setattr(ro, "get_order", lambda *a, **k: {
        "status": "error", "error": "timeout",
    })

    result = ro._missing_tp_order_state("PUMP-USDT", matched_trade, {"tp1"})

    assert result["status"] == "unknown"


def test_fresh_protection_audit_accepts_exchange_resized_sl(monkeypatch):
    import run_once as ro

    position = {
        "status": "found", "positionAmt": "1261",
        "avgPrice": "0.00429", "entryPrice": "0.00429",
    }
    sl = {
        "orderId": "SL-1", "type": "STOP_MARKET", "stopPrice": "0.00459",
        "origQty": "1261", "symbol": "PUMP-USDT", "positionSide": "SHORT",
    }
    monkeypatch.setattr(ro, "get_position_directional", lambda *a, **k: dict(position))
    monkeypatch.setattr(ro, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok", "sl_orders": [sl], "tp_orders": [],
    })

    result = ro._stable_live_protection_audit(
        "PUMP-USDT", "SHORT", stop_loss_pct=7.0,
        expected_sl_price=0.0045903,
    )

    assert result["status"] == "ok"
    assert result["sl_valid"] is True
    assert result["live_qty"] == pytest.approx(1261.0)
    assert result["sl_audit"]["audit_reason"] == "OK"


def test_fresh_protection_audit_refuses_to_infer_missing_sl_when_orders_unknown(monkeypatch):
    import run_once as ro

    position = {
        "status": "found", "positionAmt": "9.169",
        "avgPrice": "0.5903", "entryPrice": "0.5903",
    }
    monkeypatch.setattr(ro, "get_position_directional", lambda *a, **k: dict(position))
    monkeypatch.setattr(ro, "get_open_protection_directional", lambda *a, **k: {
        "status": "error", "error": "rate limited", "sl_orders": [], "tp_orders": [],
    })

    result = ro._stable_live_protection_audit(
        "VIRTUAL-USDT", "SHORT", stop_loss_pct=7.0,
    )

    assert result["status"] == "unknown"
    assert result["reason"] == "open_orders_state_unavailable"
    assert "sl_valid" not in result


def test_reconciliation_does_not_recreate_tp1_when_original_order_filled(monkeypatch):
    import run_once as ro

    position = {
        "symbol": "PUMP-USDT", "positionSide": "SHORT",
        "positionAmt": "1261", "avgPrice": "0.00429", "entryPrice": "0.00429",
    }
    sl = {
        "symbol": "PUMP-USDT", "positionSide": "SHORT", "type": "STOP_MARKET",
        "orderId": "SL-1", "stopPrice": "0.00459", "origQty": "1261",
    }
    tp2 = {
        "symbol": "PUMP-USDT", "positionSide": "SHORT", "type": "TAKE_PROFIT_MARKET",
        "orderId": "TP2", "stopPrice": "0.00384", "origQty": "1261",
    }
    active = {
        "EVT_PUMP": {
            "event_id": "EVT_PUMP", "symbol": "PUMP", "direction": "SHORT",
            "closed": False, "entry_price": 0.00429, "initial_qty": 1748.0,
            "remaining_qty": 1748.0, "planned_risk_pct": 7.0,
            "tp_orders": [
                {"leg": "tp1", "order_id": "TP1-ORIGINAL", "qty": 487.0},
                {"leg": "tp2", "order_id": "TP2", "qty": 1261.0},
            ],
            "sl_order": {"order_id": "SL-1", "stop_price": 0.00459},
            "hit_legs": [], "tp_filled_qty": {}, "be_activated": False,
            "effective_tp_levels": [
                {"leg": "tp1", "pnl_pct": 4.55, "close_fraction": 0.25},
                {"leg": "tp2", "pnl_pct": 10.5, "close_fraction": 0.75},
            ],
        }
    }
    actions = []
    monkeypatch.setattr(ro, "get_positions", lambda **kwargs: [position])
    monkeypatch.setattr(ro, "get_open_orders_snapshot", lambda **kwargs: {
        "status": "ok", "orders": [sl, tp2],
    })
    monkeypatch.setattr(ro, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok", "sl_orders": [sl], "tp_orders": [tp2],
    })
    monkeypatch.setattr(ro, "_load_active_trades", lambda: active)
    monkeypatch.setattr(ro, "_find_active_trade_for_position", lambda *a, **k: active["EVT_PUMP"])
    monkeypatch.setattr(ro, "get_order", lambda symbol, order_id, **kwargs: {
        "status": "ok", "order_id": order_id,
        "order_status": "FILLED", "executed_qty": 487.0,
    })
    monkeypatch.setattr(ro, "ensure_directional_protection", lambda **kwargs: pytest.fail("filled TP1 must not be recreated"))
    monkeypatch.setattr(ro, "emergency_close_position", lambda *a, **k: pytest.fail("valid resized SL must not be emergency-closed"))
    monkeypatch.setattr(ro, "record_action", lambda row: actions.append(row))

    ro.reconcile_all_open_positions()

    deferred = [row for row in actions if row.get("record_type") == "PROTECTION_REPAIR_DEFERRED_TP_STATE"]
    assert len(deferred) == 1
    assert deferred[0]["tp_order_state"]["status"] == "fill_pending_tracker"
    assert deferred[0]["missing_tp_legs"] == ["tp1"]


def test_emergency_close_is_skipped_when_tp_fills_during_repair_and_sl_is_resized(monkeypatch):
    import run_once as ro

    state = {
        "qty": 100.0,
        "position": {
            "symbol": "RACE-USDT", "positionSide": "LONG",
            "positionAmt": "100", "avgPrice": "100", "entryPrice": "100",
        },
    }
    tp2 = {
        "symbol": "RACE-USDT", "positionSide": "LONG", "type": "TAKE_PROFIT_MARKET",
        "orderId": "TP2", "stopPrice": "102", "origQty": "75",
    }

    def live_sl():
        return {
            "symbol": "RACE-USDT", "positionSide": "LONG", "type": "STOP_MARKET",
            "orderId": "SL-1", "stopPrice": "93", "origQty": str(state["qty"]),
        }

    trade = {
        "event_id": "EVT_RACE", "symbol": "RACE", "direction": "LONG",
        "closed": False, "entry_price": 100.0, "initial_qty": 100.0,
        "remaining_qty": 100.0, "planned_risk_pct": 7.0,
        "tp_orders": [
            {"leg": "tp1", "order_id": "TP1-ORIGINAL", "qty": 25.0},
            {"leg": "tp2", "order_id": "TP2", "qty": 75.0},
        ],
        "sl_order": {"order_id": "SL-1", "stop_price": 93.0},
        "hit_legs": [], "tp_filled_qty": {}, "be_activated": False,
        "effective_tp_levels": [
            {"leg": "tp1", "pnl_pct": 4.55, "close_fraction": 0.25},
            {"leg": "tp2", "pnl_pct": 10.5, "close_fraction": 0.75},
        ],
    }
    actions = []
    emergency_calls = []

    monkeypatch.setattr(ro, "get_positions", lambda **kwargs: [dict(state["position"])])
    monkeypatch.setattr(ro, "get_position_directional", lambda *a, **k: {
        "status": "found", "positionAmt": str(state["qty"]),
        "avgPrice": "100", "entryPrice": "100",
    })
    monkeypatch.setattr(ro, "get_open_orders_snapshot", lambda **kwargs: {
        "status": "ok", "orders": [live_sl(), tp2],
    })
    monkeypatch.setattr(ro, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok", "sl_orders": [live_sl()], "tp_orders": [tp2],
    })
    monkeypatch.setattr(ro, "_load_active_trades", lambda: {"EVT_RACE": trade})
    monkeypatch.setattr(ro, "_find_active_trade_for_position", lambda *a, **k: trade)
    monkeypatch.setattr(ro, "_tp_orders_to_tracker", lambda *a, **k: [{"leg": "tp2"}])
    monkeypatch.setattr(ro, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "CANCELED", "executed_qty": 0.0,
    })

    def fake_ensure(**kwargs):
        # Simulate TP1 executing immediately while the replacement orders are
        # being installed; BingX resizes the stop to the new live residual.
        state["qty"] = 75.0
        state["position"]["positionAmt"] = "75"
        return {
            "status": "PROTECTED",
            "sl_result": {"status": "already_exists", "order_id": "SL-1", "stop_price": 93.0, "qty": 100.0},
            "tp_orders": [
                {"status": "created", "leg": "tp1", "order_id": "TP1-REPLACEMENT"},
                {"status": "already_exists", "leg": "tp2", "order_id": "TP2"},
            ],
        }

    monkeypatch.setattr(ro, "ensure_directional_protection", fake_ensure)
    monkeypatch.setattr(ro, "emergency_close_position", lambda *a, **k: emergency_calls.append((a, k)) or {"status": "closed"})
    monkeypatch.setattr(ro, "record_action", lambda row: actions.append(row))
    monkeypatch.setattr(ro, "update_active_trade_protection", lambda **kwargs: True)
    monkeypatch.setattr(ro, "register_active_trade", lambda **kwargs: None)
    monkeypatch.setattr(ro, "record_trade", lambda *args, **kwargs: None)

    ro.reconcile_all_open_positions()

    assert emergency_calls == []
    skipped = [row for row in actions if row.get("record_type") == "PROTECTION_EMERGENCY_CLOSE_SKIPPED"]
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "FRESH_LIVE_SL_VALID_AFTER_POSITION_CHANGE"
    assert skipped[0]["position_qty"] == pytest.approx(75.0)


def _patch_emergency_close_prerequisites(monkeypatch, bingx, positions, protection, post_response=None):
    monkeypatch.setattr(bingx, "POSITION_MODE", "HEDGE")
    monkeypatch.setattr(bingx, "get_position_mode", lambda: {"status": "ok", "mode": "HEDGE"})
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda symbol: "RACE-USDT")
    monkeypatch.setattr(bingx, "get_contract", lambda symbol: {"quantityPrecision": 2, "pricePrecision": 2})
    position_iter = iter(positions)
    monkeypatch.setattr(bingx, "get_position_directional", lambda *a, **k: next(position_iter))
    monkeypatch.setattr(bingx, "get_open_protection_directional", lambda *a, **k: protection)
    if post_response is not None:
        monkeypatch.setattr(bingx, "_request", lambda *a, **k: post_response)
    monkeypatch.setattr(bingx.time, "sleep", lambda *_a, **_k: None)


def test_emergency_close_final_guard_skips_market_when_fresh_sl_is_valid(monkeypatch):
    from event_engine import bingx

    position = {"status": "found", "positionAmt": "1.0", "avgPrice": "100", "entryPrice": "100"}
    protection = {
        "status": "ok",
        "sl_orders": [{
            "orderId": "SL-1", "type": "STOP_MARKET", "stopPrice": "93",
            "origQty": "1.0", "positionSide": "LONG",
        }],
        "tp_orders": [],
    }
    _patch_emergency_close_prerequisites(
        monkeypatch, bingx, [position, position], protection,
    )
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: pytest.fail("must not submit MARKET when a valid SL exists"))

    result = bingx.emergency_close_position(
        "RACE", "LONG", 1.0,
        require_live_position=True,
        require_unprotected_position=True,
        expected_sl_price=93.0,
    )

    assert result["status"] == "protected"
    assert result["close_order_submitted"] is False


def test_emergency_close_final_guard_fails_closed_when_open_orders_unknown(monkeypatch):
    from event_engine import bingx

    position = {"status": "found", "positionAmt": "1.0", "avgPrice": "100", "entryPrice": "100"}
    _patch_emergency_close_prerequisites(
        monkeypatch, bingx, [position], {"status": "error", "error": "timeout"},
    )
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: pytest.fail("must not submit MARKET with unknown protection state"))

    result = bingx.emergency_close_position(
        "RACE", "LONG", 1.0,
        require_live_position=True,
        require_unprotected_position=True,
        expected_sl_price=93.0,
    )

    assert result["status"] == "unknown"
    assert result["close_order_submitted"] is False


def test_emergency_close_final_guard_submits_market_only_after_stable_missing_sl(monkeypatch):
    from event_engine import bingx

    found = {"status": "found", "positionAmt": "1.0", "avgPrice": "100", "entryPrice": "100"}
    not_found = {"status": "not_found"}
    protection = {"status": "ok", "sl_orders": [], "tp_orders": []}
    _patch_emergency_close_prerequisites(
        monkeypatch, bingx, [found, found, not_found, not_found], protection,
        {"code": 0, "data": {"order": {"orderId": "CLOSE-1", "avgPrice": "99"}}},
    )
    monkeypatch.setattr(bingx, "get_order_by_client_order_id", lambda *a, **k: {
        "status": "ok", "order_id": "CLOSE-1", "order_status": "FILLED",
        "executed_qty": 1.0, "avg_price": 99.0,
    })

    result = bingx.emergency_close_position(
        "RACE", "LONG", 1.0,
        require_live_position=True,
        require_unprotected_position=True,
        expected_sl_price=93.0,
    )

    assert result["status"] == "closed"
    assert result["verification"] == "two_not_found_reads_after_filled_order"


def test_emergency_close_final_guard_refuses_market_when_entry_average_changes(monkeypatch):
    from event_engine import bingx

    before = {"status": "found", "positionAmt": "1.0", "avgPrice": "100", "entryPrice": "100"}
    after = {"status": "found", "positionAmt": "1.0", "avgPrice": "101", "entryPrice": "101"}
    protection = {"status": "ok", "sl_orders": [], "tp_orders": []}
    _patch_emergency_close_prerequisites(
        monkeypatch, bingx, [before, after, before, after], protection,
    )
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: pytest.fail("must not submit MARKET while position state changes"))

    result = bingx.emergency_close_position(
        "RACE", "LONG", 1.0,
        require_live_position=True,
        require_unprotected_position=True,
        expected_sl_price=93.0,
    )

    assert result["status"] == "unknown"
    assert result["close_order_submitted"] is False
    assert result["observed_avg_before"] == pytest.approx(100.0)
    assert result["observed_avg_after"] == pytest.approx(101.0)
