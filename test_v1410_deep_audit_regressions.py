from __future__ import annotations

import inspect
import time

import pytest


def test_missing_tp_with_explicit_deferred_or_rejected_state_can_be_repaired(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "get_order", lambda *a, **k: pytest.fail("no order ID means no order query"))
    deferred = {
        "tp_orders": [{"leg": "tp1", "status": "deferred", "reason": "current_price_unavailable"}],
        "tp_repair_state": "attempted",
    }
    rejected = {
        "tp_orders": [{"leg": "tp1", "status": "error", "protection_state_unknown": False, "error": "API rejected"}],
        "tp_repair_state": "attempted",
    }
    ambiguous = {
        "tp_orders": [{"leg": "tp1", "status": "error", "protection_state_unknown": True}],
        "tp_repair_state": "attempted",
    }
    never_attempted = {"tp_orders": [], "tp_repair_state": "not_attempted"}

    assert ro._missing_tp_order_state("TEST-USDT", deferred, {"tp1"})["status"] == "safe_to_replace"
    assert ro._missing_tp_order_state("TEST-USDT", rejected, {"tp1"})["status"] == "safe_to_replace"
    assert ro._missing_tp_order_state("TEST-USDT", ambiguous, {"tp1"})["status"] == "unknown"
    assert ro._missing_tp_order_state("TEST-USDT", never_attempted, {"tp1"})["status"] == "safe_to_replace"
    # Missing metadata must remain fail-closed.
    assert ro._missing_tp_order_state("TEST-USDT", {"tp_orders": []}, {"tp1"})["status"] == "unknown"


def test_orphan_be_and_profit_lock_stop_are_preserved_and_validated():
    import run_once as ro
    from event_engine import bingx

    assert ro._is_be_or_profit_side_stop("LONG", 100.0, 100.0)
    assert ro._is_be_or_profit_side_stop("LONG", 100.0, 100.25)
    assert ro._is_be_or_profit_side_stop("SHORT", 100.0, 99.75)
    assert not ro._is_be_or_profit_side_stop("LONG", 100.0, 100.31)
    assert not ro._is_be_or_profit_side_stop("SHORT", 100.0, 99.69)

    order = {"type": "STOP_MARKET", "stopPrice": "100.25", "origQty": "2"}
    assert bingx._validate_sl_order_for_position(
        order, "LONG", 100.0, expected_price=100.25, expected_qty=2.0,
    )
    assert not bingx._validate_sl_order_for_position(order, "LONG", 100.0, expected_qty=2.0)


def test_register_active_trade_persists_exchange_confirmed_be_and_tp_repair_state(monkeypatch):
    from event_engine import tracker

    saved = {}
    monkeypatch.setattr(tracker, "_load_active_trades", lambda: {})
    monkeypatch.setattr(tracker, "_save_active_trades", lambda rows: saved.update(rows))
    tracker.register_active_trade(
        event_id="EVT_ORPHAN_BE", symbol="TEST", name="Test", direction="LONG",
        entry_price=100.0, qty=2.0, tp_orders=[],
        sl_result={"order_id": "SL-BE", "stop_price": 100.25},
        event_type="RECONCILED_POSITION", setup={"risk_pct": 7.0},
        be_activated=True, be_activation_ts=123456, tp_repair_state="not_attempted",
    )
    row = saved["EVT_ORPHAN_BE"]
    assert row["be_activated"] is True
    assert row["be_activation_ts"] == 123456
    assert row["tp_repair_state"] == "not_attempted"
    assert row["tp_fill_ledger_schema_version"] == 1


def test_replacement_tp_order_has_an_independent_cumulative_fill_baseline():
    from event_engine.tracker import _tp_order_fill_delta

    filled_by_leg = {"tp1": 0.10}
    filled_by_order = {"TP1-OLD": 0.10, "TP1-NEW": 0.0}
    previous, delta = _tp_order_fill_delta(
        "tp1", "TP1-NEW", 0.15, filled_by_leg, filled_by_order,
        legacy_order_ledger=False,
    )
    assert previous == pytest.approx(0.0)
    assert delta == pytest.approx(0.15)

    previous_old, delta_old = _tp_order_fill_delta(
        "tp1", "TP1-OLD", 0.10, filled_by_leg, filled_by_order,
        legacy_order_ledger=False,
    )
    assert previous_old == pytest.approx(0.10)
    assert delta_old == pytest.approx(0.0)




def test_legacy_v149_fill_ledger_distinguishes_original_and_residual_tp_orders():
    from event_engine.tracker import _legacy_tp_order_fill_baseline

    trade = {
        "initial_qty": 1.0,
        "tp_filled_qty": {"tp1": 0.10},
        "setup": {"effective_tp_levels": [{"leg": "tp1", "qty": 0.25, "close_fraction": 0.25}]},
    }
    # Original order: the recorded leg aggregate belongs to this order.
    baseline, reason = _legacy_tp_order_fill_baseline(
        trade, "tp1", {"orig_qty": 0.25, "executed_qty": 0.15},
    )
    assert baseline == pytest.approx(0.10)
    assert reason == "legacy_original_order"

    # Residual replacement: prior 0.10 remains in the leg aggregate, while the
    # replacement order's own cumulative executedQty starts independently at 0.
    baseline, reason = _legacy_tp_order_fill_baseline(
        trade, "tp1", {"orig_qty": 0.15, "executed_qty": 0.15},
    )
    assert baseline == pytest.approx(0.0)
    assert reason == "legacy_replacement_order"


def test_legacy_order_ledger_migration_does_not_relabel_prior_fills_as_new_order(tmp_path, monkeypatch):
    from event_engine import tracker

    saved = {}
    original_setup = {"effective_tp_levels": [{"leg": "tp1", "qty": 0.25, "close_fraction": 0.25}]}
    state = {
        "EVT_LEGACY_REPLACEMENT": {
            "symbol": "TEST", "direction": "LONG", "closed": False,
            "tp_orders": [{"leg": "tp1", "order_id": "TP1-OLD", "qty": 0.15}],
            "tp_filled_qty": {"tp1": 0.10}, "tp_filled_qty_by_order": {},
            "tp_fill_ledger_schema_version": 0,
            "effective_tp_levels": [{"leg": "tp1", "qty": 0.15, "close_fraction": 0.25}],
            "setup": original_setup,
        }
    }
    monkeypatch.setattr(tracker, "_load_active_trades", lambda: state)
    monkeypatch.setattr(tracker, "_save_active_trades", lambda rows: saved.update(rows))
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda value: "TEST-USDT")

    assert tracker.update_active_trade_protection(
        symbol="TEST", direction="LONG",
        tp_orders=[{"leg": "tp1", "order_id": "TP1-NEW", "qty": 0.15}],
        sl_result={"order_id": "SL1", "stop_price": 95.0},
        effective_tp_levels=[{"leg": "tp1", "qty": 0.15, "close_fraction": 0.25}],
    )
    row = saved["EVT_LEGACY_REPLACEMENT"]
    assert row["effective_tp_levels"][0]["qty"] == pytest.approx(0.25)
    assert row["tp_filled_qty"]["tp1"] == pytest.approx(0.10)
    assert row["tp_filled_qty_by_order"] == {}
    assert row["tp_fill_ledger_migration_pending"] is True


def test_quantity_epsilon_tracks_float_ulp_without_erasing_real_small_residuals():
    from event_engine.tracker import _quantity_epsilon

    epsilon = _quantity_epsilon(1_000_000.3)
    residual = 1000000.3 - 250000.1 - 750000.2
    assert residual > 1e-12
    assert residual <= epsilon
    assert epsilon < 1e-8


def test_daily_reconciliation_uses_per_order_tp_fill_baseline(monkeypatch):
    from event_engine import reconcile_daily as recon

    active = {
        "EVT_ORDER_LEDGER": {
            "symbol": "HBAR", "direction": "SHORT", "remaining_qty": 0.75,
            "closed": False, "tp_filled_qty": {"tp1": 0.10},
            "tp_filled_qty_by_order": {"TP1-OLD": 0.10, "TP1-NEW": 0.0},
            "tp_orders": [{"leg": "tp1", "order_id": "TP1-NEW"}],
        },
    }
    monkeypatch.setattr(recon.bingx, "get_order", lambda *a, **k: {
        "status": "ok", "order_status": "FILLED", "executed_qty": 0.15,
    })
    state = {
        "queries_used": 0, "query_budget": 5,
        "started_monotonic": recon.time.monotonic(), "time_budget_sec": 10.0,
    }
    qty, evidence, exhausted = recon._pending_exit_fill_evidence(
        active, ("HBAR-USDT", "SHORT"), 0.15, state,
    )
    assert qty == pytest.approx(0.15)
    assert exhausted is False
    assert evidence[0]["unapplied_fill_qty"] == pytest.approx(0.15)


def test_high_quantity_float_dust_is_inside_scale_aware_tolerance():
    from event_engine.tracker import _quantity_epsilon

    remaining = 1_000_000.3 - 250_000.1 - 750_000.2
    assert remaining > 1e-12
    assert remaining <= _quantity_epsilon(1_000_000.3)


def test_all_eight_strategy_detectors_freeze_the_detection_close_price():
    import event_engine.signals as sig

    names = [
        "detect_liquidity_sweep_reclaim", "detect_ema_pullback_continuation",
        "detect_order_block", "detect_breaker_block", "detect_mitigation_block",
        "detect_sfp", "detect_liquidation_cascade_fvg", "detect_crt",
    ]
    for name in names:
        source = inspect.getsource(getattr(sig, name))
        assert "detection_close_price" in source, f"{name} does not persist its signal candle close"


def test_open_orders_100410_is_not_retried_after_cooldown_is_persisted(tmp_path, monkeypatch):
    from event_engine import bingx as bx

    class Response:
        status_code = 429
        headers = {"Retry-After": "0"}

        def json(self):
            return {"code": 100410, "msg": "endpoint disabled for cooldown", "data": None}

    class Session:
        def __init__(self):
            self.calls = 0

        def request(self, *args, **kwargs):
            self.calls += 1
            return Response()

    session = Session()
    monkeypatch.setattr(bx, "FAST_SESSION", session)
    monkeypatch.setattr(bx, "API_KEY", "test-key")
    monkeypatch.setattr(bx, "SECRET_KEY", "test-secret")
    monkeypatch.setattr(bx, "OPEN_ORDERS_COOLDOWN_STATE_PATH", tmp_path / "cooldown.json")
    monkeypatch.setattr(bx, "_OPEN_ORDERS_COOLDOWN_UNTIL_MS", 0)
    monkeypatch.setattr(bx, "_OPEN_ORDERS_COOLDOWN_REASON", "")
    monkeypatch.setattr(bx.time, "sleep", lambda *_a, **_k: None)

    result = bx._request("GET", bx.OPEN_ORDERS_PATH, {}, signed=True, retryable=True)
    assert result["code"] == 100410
    assert session.calls == 1


def test_protection_setup_invalid_verified_rollback_terminalizes_once():
    import run_once as ro

    assert ro._postfill_setup_terminal_reason(
        "PROTECTION_SETUP_INVALID", {"rolled_back": True}, 0.0,
    ) == "PROTECTION_SETUP_INVALID_ROLLED_BACK"
    assert ro._postfill_setup_terminal_reason(
        "PROTECTION_SETUP_INVALID", {"rolled_back": False}, 0.0,
    ) is None
    assert ro._postfill_setup_terminal_reason(
        "PROTECTION_SETUP_INVALID", {"rolled_back": True}, 0.1,
    ) is None


def test_action_journal_index_reads_archives_once_and_only_active_tail_incrementally(tmp_path, monkeypatch):
    import gzip
    import json
    import run_once as ro

    actions = tmp_path / "actions.jsonl"
    index = tmp_path / "action_journal_index.json"
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    archive = archive_dir / "actions-20261115T120000.000000Z.jsonl.gz"
    archive_rows = [
        {"event_id": "EVT_SENT", "telegram_sent": True},
        {"record_type": "CANDIDATE_FILTER_DECISION", "stage": "EVENT_CACHE_RECONCILIATION", "candidate_id": "CAND_1"},
    ]
    with gzip.open(archive, "wt", encoding="utf-8") as stream:
        for row in archive_rows:
            stream.write(json.dumps(row) + "\n")
    actions.write_text(json.dumps({"event_id": "EVT_ACTIVE", "telegram_sent": True}) + "\n", encoding="utf-8")

    monkeypatch.setattr(ro, "ACTIONS", actions)
    monkeypatch.setattr(ro, "ACTION_INDEX", index)
    telegram_ids, candidate_ids = ro._load_action_journal_index()
    assert telegram_ids == {"EVT_SENT", "EVT_ACTIVE"}
    assert candidate_ids == {"CAND_1"}
    assert index.exists()

    # The next cycle appends one row. The old gzip archive must not be opened again.
    with actions.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event_id": "EVT_NEW_TAIL", "telegram_sent": True}) + "\n")
    original_gzip_open = ro.gzip.open
    archive_reads = {"count": 0}

    def counted_gzip_open(path, *args, **kwargs):
        if Path(path) == archive:
            archive_reads["count"] += 1
        return original_gzip_open(path, *args, **kwargs)

    monkeypatch.setattr(ro.gzip, "open", counted_gzip_open)
    telegram_ids, candidate_ids = ro._load_action_journal_index()
    assert archive_reads["count"] == 0
    assert telegram_ids == {"EVT_SENT", "EVT_ACTIVE", "EVT_NEW_TAIL"}
    assert candidate_ids == {"CAND_1"}

    monkeypatch.setattr(ro, "EVENT_CACHE", tmp_path / "nonexistent-cache.json")
    monkeypatch.setattr(ro, "EVENTS", tmp_path / "nonexistent-events.jsonl")
    monkeypatch.setattr(ro, "_load_json", lambda *_a, **_k: {"events": []})
    monkeypatch.setattr(ro, "_iter_action_records", lambda: pytest.fail("incremental index must avoid archive rescans"))
    assert ro._load_cached_events(
        now_ms=int(time.time() * 1000), terminal_ids=set(),
        journaled_candidate_ids=candidate_ids,
    ) == []


def test_action_journal_index_rebuilds_after_active_prefix_mutation(tmp_path, monkeypatch):
    import json
    import run_once as ro

    actions = tmp_path / "actions.jsonl"
    index = tmp_path / "action_journal_index.json"
    first = json.dumps({"event_id": "EVT_OLD", "telegram_sent": True}) + "\n"
    actions.write_text(first, encoding="utf-8")
    monkeypatch.setattr(ro, "ACTIONS", actions)
    monkeypatch.setattr(ro, "ACTION_INDEX", index)
    assert ro._load_action_journal_index()[0] == {"EVT_OLD"}

    # Simulate truncation/replacement by a new journal with a different prefix.
    actions.write_text(json.dumps({"event_id": "EVT_REPLACED", "telegram_sent": True}) + "\n", encoding="utf-8")
    telegram_ids, _ = ro._load_action_journal_index()
    assert telegram_ids == {"EVT_REPLACED"}


def test_run_once_protection_snapshot_normalizes_underscore_symbols():
    import run_once as ro

    snapshot = {
        "status": "ok",
        "orders": [{
            "symbol": "TEST_USDT", "type": "STOP_MARKET",
            "positionSide": "LONG", "orderId": "SL1",
            "stopPrice": "95", "origQty": "2",
        }],
    }
    result = ro._protection_from_open_orders_snapshot(snapshot, "TEST-USDT", "LONG")
    assert result is not None
    assert [row["orderId"] for row in result["sl_orders"]] == ["SL1"]


def test_run_once_does_not_match_post_close_trade_to_reopened_position(monkeypatch):
    import run_once as ro

    stale_rows = {
        "close-notify": {"symbol": "TEST", "direction": "LONG", "closed": False, "close_notification_pending": True},
        "close-journal": {"symbol": "TEST", "direction": "LONG", "closed": False, "close_journal_pending": True},
        "close-cleanup": {"symbol": "TEST", "direction": "LONG", "closed": False, "close_cleanup_pending": True},
        "gone-confirmed": {"symbol": "TEST", "direction": "LONG", "closed": False, "position_gone_confirmation": {"status": "CONFIRMED"}},
        "live": {"symbol": "TEST", "direction": "LONG", "closed": False, "remaining_qty": 2.0},
    }
    monkeypatch.setattr(ro, "to_bx_symbol", lambda value: "TEST-USDT")
    assert ro._find_active_trade_for_position("TEST-USDT", "LONG", stale_rows) is stale_rows["live"]


def test_update_protection_does_not_mutate_post_close_trade(monkeypatch):
    from event_engine import tracker

    state = {"old": {
        "symbol": "TEST", "direction": "LONG", "closed": False,
        "close_notification_pending": True, "tp_orders": [],
    }}
    monkeypatch.setattr(tracker, "_load_active_trades", lambda: state)
    monkeypatch.setattr(tracker, "_save_active_trades", lambda rows: None)
    monkeypatch.setattr(tracker, "to_bx_symbol", lambda value: "TEST-USDT")
    assert tracker.update_active_trade_protection(
        symbol="TEST", direction="LONG", tp_orders=[{"leg": "tp1", "order_id": "NEW"}],
        sl_result={"order_id": "SL-NEW"},
    ) is False
    assert state["old"]["tp_orders"] == []


def test_bingx_sessions_ignore_global_proxy_environment(monkeypatch):
    from event_engine import bingx

    monkeypatch.setenv("HTTP_PROXY", "http://203.0.113.10:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://203.0.113.10:8080")
    monkeypatch.setenv("ALL_PROXY", "http://203.0.113.10:8080")
    assert bingx.SESSION.trust_env is False
    assert bingx.FAST_SESSION.trust_env is False
