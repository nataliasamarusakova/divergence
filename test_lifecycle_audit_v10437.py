from __future__ import annotations

import copy
import json


def test_notification_without_recipients_is_delivered_after_chat_ids_are_configured(monkeypatch, tmp_path):
    from event_engine import tracker

    path = tmp_path / "notifications.json"
    monkeypatch.setattr(tracker, "NOTIFICATIONS_PATH", path)
    monkeypatch.delenv("TG_CHAT_IDS", raising=False)
    monkeypatch.delenv("TG_CHAT_ID", raising=False)

    calls = []
    monkeypatch.setattr(
        tracker,
        "send_detailed",
        lambda text, only_chat_ids=None: calls.append((text, list(only_chat_ids or [])))
        or {cid: {"sent": True, "message_id": f"msg-{cid}"} for cid in (only_chat_ids or [])},
    )

    nid = tracker._notification_id("EVT_NO_RECIPIENTS", "TRADE_CLOSE")
    queued = tracker._queue_notification(
        nid,
        event_id="EVT_NO_RECIPIENTS",
        kind="TRADE_CLOSE",
        symbol="BTC-USDT",
        direction="LONG",
        text="close notification",
    )
    assert queued is False
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved[nid]["resolved"] is False
    assert saved[nid]["chats"] == {}
    assert calls == []

    monkeypatch.setenv("TG_CHAT_IDS", "chat-a;chat-b,chat-a")
    tracker._retry_pending_notifications()

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert calls == [("close notification", ["chat-a", "chat-b"])]
    assert saved[nid]["resolved"] is True
    assert set(saved[nid]["chats"]) == {"chat-a", "chat-b"}
    assert all(row["sent"] for row in saved[nid]["chats"].values())


def test_tracker_books_terminal_canceled_partial_tp_once(monkeypatch, tmp_path):
    from event_engine import tracker

    state = {
        "EVT_CANCELED_TP": {
            "event_id": "EVT_CANCELED_TP",
            "symbol": "TEST-USDT",
            "direction": "LONG",
            "entry_price": 100.0,
            "initial_qty": 1.0,
            "remaining_qty": 1.0,
            "entry_ts": 1_700_000_000_000,
            "tp_orders": [{"leg": "tp1", "order_id": "TP-CANCELED", "qty": 0.5}],
            "hit_legs": [],
            "tp_filled_qty": {},
            "realized_pnl_qty": 0.0,
            "realized_pnl_weighted_sum": 0.0,
            "be_activated": False,
            "sl_order": {"order_id": "SL-1", "stop_price": 93.0},
            "closed": False,
        }
    }
    saved = {}
    monkeypatch.setattr(tracker, "_recover_data_error_trade_closes", lambda: None)
    monkeypatch.setattr(tracker, "_recover_trade_cashflows", lambda **kwargs: None)
    monkeypatch.setattr(tracker, "_retry_pending_notifications", lambda: None)
    monkeypatch.setattr(tracker, "_load_active_trades", lambda: copy.deepcopy(state))
    monkeypatch.setattr(tracker, "_save_active_trades", lambda rows: (saved.clear(), saved.update(copy.deepcopy(rows))))
    monkeypatch.setattr(tracker, "get_position_directional", lambda *a, **k: {
        "status": "found", "positionAmt": "0.75", "avgPrice": "100",
    })
    monkeypatch.setattr(tracker, "get_live_price", lambda *_a, **_k: 100.0)
    monkeypatch.setattr(tracker, "fetch_klines", lambda *_a, **_k: [])
    monkeypatch.setattr(tracker, "_update_mfe_mae", lambda *a, **k: None)
    monkeypatch.setattr(tracker, "_update_live_pnl_extrema", lambda *a, **k: None)
    monkeypatch.setattr(tracker, "_get_filled_sl_from_trade", lambda *a, **k: (None, None))
    monkeypatch.setattr(tracker.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(tracker, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "CANCELED", "executed_qty": 0.25,
        "orig_qty": 0.5, "avg_price": 102.0,
    })

    tracker.update_active_trades()
    first = copy.deepcopy(saved["EVT_CANCELED_TP"])
    assert first["tp_filled_qty"]["tp1"] == 0.25
    assert first["realized_pnl_qty"] == 0.25
    assert first["remaining_qty"] == 0.75
    assert first["hit_legs"] == []

    # The same cumulative exchange fill is observed again. Its delta is zero,
    # so realized quantity and remaining position must not be booked twice.
    state.clear()
    state.update(copy.deepcopy(saved))
    tracker.update_active_trades()
    second = saved["EVT_CANCELED_TP"]
    assert second["tp_filled_qty"]["tp1"] == 0.25
    assert second["realized_pnl_qty"] == 0.25
    assert second["remaining_qty"] == 0.75


def test_missing_tp_partial_terminal_fill_allows_only_exact_residual_replacement(monkeypatch):
    import run_once as ro

    trade = {
        "tp_orders": [{"leg": "tp1", "order_id": "TP1", "qty": 25.0}],
        "tp_filled_qty": {"tp1": 5.0},
    }
    monkeypatch.setattr(ro, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "CANCELED", "executed_qty": 5.0,
        "orig_qty": 25.0,
    })

    result = ro._missing_tp_order_state("TEST-USDT", trade, {"tp1"})

    assert result["status"] == "safe_to_replace"
    assert result["orders"][0]["replacement_qty"] == 20.0
    assert result["orders"][0]["original_qty"] == 25.0


def test_daily_reconciliation_recognizes_unapplied_canceled_tp_fill(monkeypatch):
    from event_engine import reconcile_daily as recon

    active = {
        "EVT_CANCELED_DAILY": {
            "symbol": "HBAR", "direction": "SHORT", "remaining_qty": 139.0,
            "closed": False, "tp_filled_qty": {},
            "tp_orders": [{"leg": "tp1", "order_id": "TP1-HBAR"}],
        },
    }
    monkeypatch.setattr(recon.bingx, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "CANCELED", "executed_qty": 35.0,
    })
    state = {
        "queries_used": 0,
        "query_budget": 5,
        "started_monotonic": recon.time.monotonic(),
        "time_budget_sec": 10.0,
    }

    qty, evidence, exhausted = recon._pending_exit_fill_evidence(
        active, ("HBAR-USDT", "SHORT"), 35.0, state,
    )

    assert qty == 35.0
    assert exhausted is False
    assert len(evidence) == 1
    assert evidence[0]["order_status"] == "CANCELED"
    assert evidence[0]["unapplied_fill_qty"] == 35.0


def test_single_remaining_tp2_keeps_its_identity_and_exact_position_qty():
    from event_engine import bingx

    levels = bingx._select_supported_tp_levels(
        [{"leg": "tp2", "pnl_pct": 10.5, "close_fraction": 0.75, "qty": 0.75}],
        position_qty=0.75, precision=3, min_qty=0.001,
    )

    assert levels == [{
        "leg": "tp2", "pnl_pct": 10.5, "close_fraction": 0.75, "qty": 0.75,
    }]
    assert bingx._allocate_tp_quantities(
        0.75, 3, 0.001, [0.75], fixed_quantities=[0.75],
    ) == [0.75]


def test_fixed_tp_quantities_must_sum_to_live_position():
    from event_engine import bingx

    levels = bingx._select_supported_tp_levels(
        [
            {"leg": "tp1", "pnl_pct": 4.0, "close_fraction": 0.25, "qty": 0.25},
            {"leg": "tp2", "pnl_pct": 10.0, "close_fraction": 0.75, "qty": 0.50},
        ],
        position_qty=0.80, precision=3, min_qty=0.001,
    )

    assert levels == []


def test_emergency_close_cancels_new_order_before_retrying_terminal_zero_fill(monkeypatch):
    from event_engine import bingx

    positions = iter([
        {"status": "found", "positionAmt": "1.000", "avgPrice": "100"},
        {"status": "found", "positionAmt": "1.000", "avgPrice": "100"},
        {"status": "not_found"},
        {"status": "not_found"},
    ])
    posts, lookups, cancels = [], [], []
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "TEST-USDT")
    monkeypatch.setattr(bingx, "get_contract", lambda _symbol: {"quantityPrecision": 3})
    monkeypatch.setattr(bingx, "get_position_mode", lambda: {
        "status": "ok", "mode": "HEDGE", "dualSidePosition": True,
    })
    monkeypatch.setattr(bingx, "get_position_directional", lambda *_a, **_k: next(positions))
    monkeypatch.setattr(bingx.time, "sleep", lambda *_a, **_k: None)

    def request(_method, _path, params):
        posts.append(dict(params))
        return {"code": 0, "data": {"order": {"orderId": f"CLOSE-{len(posts)}"}}}

    def lookup(_symbol, client_id):
        lookups.append(client_id)
        if len(lookups) == 1:
            return {
                "status": "ok", "order_id": "CLOSE-1", "client_order_id": client_id,
                "order_status": "NEW", "executed_qty": 0.0,
            }
        if len(lookups) == 2:
            return {
                "status": "ok", "order_id": "CLOSE-1", "client_order_id": client_id,
                "order_status": "CANCELED", "executed_qty": 0.0,
            }
        return {
            "status": "ok", "order_id": "CLOSE-2", "client_order_id": client_id,
            "order_status": "FILLED", "executed_qty": 1.0, "avg_price": 99.5,
        }

    monkeypatch.setattr(bingx, "_request", request)
    monkeypatch.setattr(bingx, "get_order_by_client_order_id", lookup)
    monkeypatch.setattr(bingx, "cancel_order", lambda _symbol, order_id: cancels.append(str(order_id)) or {"code": 0})

    result = bingx.emergency_close_position("TEST", "LONG", 1.0, reason_token="NEW_TO_CANCELED")

    assert result["status"] == "closed"
    assert len(posts) == 2
    assert posts[0]["clientOrderId"] != posts[1]["clientOrderId"]
    assert lookups == [posts[0]["clientOrderId"], posts[0]["clientOrderId"], posts[1]["clientOrderId"]]
    assert cancels == ["CLOSE-1"]


def test_emergency_close_does_not_retry_while_new_order_cannot_be_terminalized(monkeypatch):
    from event_engine import bingx

    position = {"status": "found", "positionAmt": "1.000", "avgPrice": "100"}
    posts, lookups, cancels = [], [], []
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "TEST-USDT")
    monkeypatch.setattr(bingx, "get_contract", lambda _symbol: {"quantityPrecision": 3})
    monkeypatch.setattr(bingx, "get_position_mode", lambda: {
        "status": "ok", "mode": "HEDGE", "dualSidePosition": True,
    })
    monkeypatch.setattr(bingx, "get_position_directional", lambda *_a, **_k: position)
    monkeypatch.setattr(bingx.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(bingx, "_request", lambda _m, _p, params: posts.append(dict(params)) or {
        "code": 0, "data": {"order": {"orderId": "CLOSE-1"}},
    })
    monkeypatch.setattr(bingx, "get_order_by_client_order_id", lambda _s, client_id: (
        lookups.append(client_id) or {
            "status": "ok", "order_id": "CLOSE-1", "client_order_id": client_id,
            "order_status": "NEW", "executed_qty": 0.0,
        }
    ))
    monkeypatch.setattr(bingx, "cancel_order", lambda _s, order_id: cancels.append(str(order_id)) or {"code": 0})

    result = bingx.emergency_close_position("TEST", "LONG", 1.0, reason_token="NEW_STAYS_ACTIVE")

    assert result["status"] == "unknown"
    assert result["escalation_required"] is True
    assert len(posts) == 1
    assert len(cancels) == 3
    assert lookups == [posts[0]["clientOrderId"]] * 4


def test_filled_emergency_close_with_zero_executed_qty_fails_closed(monkeypatch):
    from event_engine import bingx

    position = {"status": "found", "positionAmt": "1.000", "avgPrice": "100"}
    posts = []
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "TEST-USDT")
    monkeypatch.setattr(bingx, "get_contract", lambda _symbol: {"quantityPrecision": 3})
    monkeypatch.setattr(bingx, "get_position_mode", lambda: {
        "status": "ok", "mode": "HEDGE", "dualSidePosition": True,
    })
    monkeypatch.setattr(bingx, "get_position_directional", lambda *_a, **_k: position)
    monkeypatch.setattr(bingx.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(bingx, "_request", lambda _m, _p, params: posts.append(dict(params)) or {
        "code": 0, "data": {"order": {"orderId": "CLOSE-1"}},
    })
    monkeypatch.setattr(bingx, "get_order_by_client_order_id", lambda *_a, **_k: {
        "status": "ok", "order_id": "CLOSE-1", "order_status": "FILLED",
        "executed_qty": 0.0,
    })

    result = bingx.emergency_close_position("TEST", "LONG", 1.0, reason_token="FILLED_ZERO")

    assert result["status"] == "unknown"
    assert result["error"] == "filled_close_order_has_zero_executed_qty"
    assert len(posts) == 1


def test_tracker_does_not_mark_zero_quantity_filled_tp_as_hit(monkeypatch, tmp_path):
    from event_engine import tracker

    state = {
        "EVT_FILLED_ZERO_TP": {
            "event_id": "EVT_FILLED_ZERO_TP", "symbol": "TEST-USDT", "direction": "LONG",
            "entry_price": 100.0, "initial_qty": 1.0, "remaining_qty": 1.0,
            "entry_ts": 1_700_000_000_000,
            "tp_orders": [{"leg": "tp1", "order_id": "TP-FILLED-ZERO", "qty": 0.5}],
            "hit_legs": [], "tp_filled_qty": {}, "realized_pnl_qty": 0.0,
            "realized_pnl_weighted_sum": 0.0, "be_activated": False,
            "sl_order": {"order_id": "SL-1", "stop_price": 93.0}, "closed": False,
        }
    }
    saved = {}
    monkeypatch.setattr(tracker, "_recover_data_error_trade_closes", lambda: None)
    monkeypatch.setattr(tracker, "_recover_trade_cashflows", lambda **kwargs: None)
    monkeypatch.setattr(tracker, "_retry_pending_notifications", lambda: None)
    monkeypatch.setattr(tracker, "_load_active_trades", lambda: copy.deepcopy(state))
    monkeypatch.setattr(tracker, "_save_active_trades", lambda rows: (saved.clear(), saved.update(copy.deepcopy(rows))))
    monkeypatch.setattr(tracker, "get_position_directional", lambda *_a, **_k: {
        "status": "found", "positionAmt": "1.0", "avgPrice": "100",
    })
    monkeypatch.setattr(tracker, "get_live_price", lambda *_a, **_k: 100.0)
    monkeypatch.setattr(tracker, "fetch_klines", lambda *_a, **_k: [])
    monkeypatch.setattr(tracker, "_update_mfe_mae", lambda *_a, **_k: None)
    monkeypatch.setattr(tracker, "_update_live_pnl_extrema", lambda *_a, **_k: None)
    monkeypatch.setattr(tracker, "_get_filled_sl_from_trade", lambda *_a, **_k: (None, None))
    monkeypatch.setattr(tracker.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(tracker, "get_order", lambda *_a, **_k: {
        "status": "ok", "order_status": "FILLED", "executed_qty": 0.0,
        "orig_qty": 0.5, "avg_price": 102.0,
    })

    tracker.update_active_trades()
    result = saved["EVT_FILLED_ZERO_TP"]
    assert result["remaining_qty"] == 1.0
    assert result["hit_legs"] == []
    assert result["realized_pnl_qty"] == 0.0
    assert result["tp_filled_qty"] == {}
