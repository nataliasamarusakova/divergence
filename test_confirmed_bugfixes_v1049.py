from __future__ import annotations

import pytest


def test_history_symbol_normalizer_handles_compact_usdt_symbol(monkeypatch):
    import event_engine.bingx as bx

    monkeypatch.setattr(bx, "to_bx_symbol", lambda s: "BTC-USDT" if s == "BTC-USDT" else None)
    assert bx._normalize_history_symbol("BTCUSDT") == "BTC-USDT"
    assert bx._normalize_history_symbol("BTC-USDT") == "BTC-USDT"
    assert bx._normalize_history_symbol("BTC") == "BTC-USDT"


def test_fill_history_compact_symbol_uses_valid_contract(monkeypatch):
    import event_engine.bingx as bx

    symbols = []
    monkeypatch.setattr(bx, "to_bx_symbol", lambda s: "BTC-USDT" if s == "BTC-USDT" else None)
    monkeypatch.setattr(bx, "_request", lambda method, path, params, **kwargs: symbols.append(params["symbol"]) or {"code": 0, "data": {"fill_orders": [], "total": 0}})
    out = bx.get_fill_history("BTCUSDT", 1_700_000_000_000, 1_700_000_001_000)
    assert out["status"] == "ok"
    assert symbols == ["BTC-USDT"]


def test_income_history_compact_symbol_uses_valid_contract(monkeypatch):
    import event_engine.bingx as bx

    symbols = []
    monkeypatch.setattr(bx, "to_bx_symbol", lambda s: "BTC-USDT" if s == "BTC-USDT" else None)
    monkeypatch.setattr(bx, "_request", lambda method, path, params, **kwargs: symbols.append(params["symbol"]) or {"code": 0, "data": {"list": []}})
    out = bx.get_income_history("BTCUSDT", 1_700_000_000_000, 1_700_000_001_000, limit=10)
    assert out["status"] == "ok"
    assert out["complete"] is True
    assert symbols == ["BTC-USDT"]


def _prepare_market_order_without_id(monkeypatch, bx):
    contract = {"symbol": "TEST-USDT", "status": 1, "apiStateOpen": "true", "quantityPrecision": 3,
                "tradeMinQuantity": 0.001, "tradeMinUSDT": 1, "maxLongLeverage": 10, "maxShortLeverage": 10}
    bx.CACHE["data"] = {"TEST-USDT": contract}
    bx.CACHE["by_display_name"] = {"TEST-USDT": contract}
    bx.CACHE["ts"] = 9_999_999_999
    monkeypatch.setattr(bx, "to_bx_symbol", lambda _s: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda *_a, **_k: contract)
    monkeypatch.setattr(bx, "contract_exists", lambda *_a: True)
    monkeypatch.setattr(bx, "has_open_position", lambda *_a, **_k: False)
    monkeypatch.setattr(bx, "_current_close_price", lambda _s: 100.0)
    monkeypatch.setattr(bx, "_set_leverage", lambda *_a, **_k: True)
    monkeypatch.setattr(bx, "_request", lambda *_a, **_k: {"code": 0, "data": {"order": {"status": "NEW"}}})
    monkeypatch.setattr(bx, "get_order_by_client_order_id", lambda *_a, **_k: {"status": "error", "error": "not visible"})


def test_successful_market_ack_without_order_id_with_unknown_position_is_not_claimed_opened(monkeypatch):
    import event_engine.bingx as bx
    _prepare_market_order_without_id(monkeypatch, bx)
    monkeypatch.setattr(bx, "get_position_directional", lambda *_a, **_k: {"status": "error", "error": "not visible"})

    result = bx.open_market("TEST", "LONG", 100.0, "TRADE-X")
    assert result["status"] == "unknown"
    assert result["entry_ack_missing_order_id"] is True
    assert result.get("order_id") is None


def test_successful_market_ack_without_order_id_but_live_position_continues_to_protection(monkeypatch):
    import event_engine.bingx as bx
    _prepare_market_order_without_id(monkeypatch, bx)
    monkeypatch.setattr(bx, "get_position_directional", lambda *_a, **_k: {
        "status": "found", "symbol": "TEST-USDT", "positionSide": "LONG",
        "positionAmt": "2.5", "avgPrice": "100.1",
    })

    result = bx.open_market("TEST", "LONG", 100.0, "TRADE-X")
    assert result["status"] == "opened"
    assert result["order_id"] is None
    assert result["entry_ack_missing_order_id"] is True
    assert result["client_order_id"]
    assert result["entry_position_snapshot"]["positionAmt"] == "2.5"


def test_coinalyze_click_pagination_failure_marks_universe_incomplete(monkeypatch):
    import playwright.sync_api as sync_api
    import event_engine.coinalyze as coinalyze

    class Pagination:
        def query_selector_all(self, _selector):
            return [self]
        def inner_text(self):
            return "2"
        def click(self):
            pass
        def query_selector(self, _selector):
            return None

    class Row:
        def get_attribute(self, _name):
            return "AAA"

    class Page:
        def __init__(self):
            self.pagination = Pagination()
        def query_selector(self, selector):
            if selector == ".pagination":
                return self.pagination
            if selector == "tbody tr":
                return Row()
            return None
        def wait_for_timeout(self, _ms):
            pass
        def content(self):
            return "page-html"

    class Browser:
        def close(self):
            pass

    class Context:
        def __enter__(self):
            return object()
        def __exit__(self, *_args):
            return False

    page = Page()
    monkeypatch.setattr(sync_api, "sync_playwright", lambda: Context())
    monkeypatch.setattr(coinalyze, "_setup_browser_context", lambda _p: (Browser(), page))
    monkeypatch.setattr(coinalyze, "_load_page", lambda *_a: "page-html")
    monkeypatch.setattr(coinalyze, "get_page_urls", lambda _html: [coinalyze.COINALYZE_URL])
    monkeypatch.setattr(coinalyze, "parse_table", lambda _html: [type("R", (), {"symbol": "AAA"})()])

    with pytest.raises(coinalyze.CoinalyzeIncompleteDataError, match="pagination incomplete"):
        coinalyze.fetch_data()


@pytest.mark.parametrize("live_sl_price,live_sl_qty", [(100.0, 1.0), (95.0, 1.0), (95.0, 0.8)])
def test_orphan_live_sl_is_passed_to_repair_instead_of_default_7pct(monkeypatch, live_sl_price, live_sl_qty):
    import run_once as ro

    position = {"symbol": "TEST-USDT", "positionSide": "LONG", "positionAmt": "1.0", "avgPrice": "100.0", "entryPrice": "100.0"}
    sl = {"orderId": "SL-ORPHAN", "type": "STOP_MARKET", "stopPrice": str(live_sl_price), "origQty": str(live_sl_qty), "symbol": "TEST-USDT", "positionSide": "LONG"}
    repaired_sl = {**sl, "origQty": "1.0"}
    repaired_tps = [
        {"orderId": "TP1", "type": "TAKE_PROFIT_MARKET", "stopPrice": "101", "origQty": "0.3", "symbol": "TEST-USDT", "positionSide": "LONG"},
        {"orderId": "TP2", "type": "TAKE_PROFIT_MARKET", "stopPrice": "102", "origQty": "0.7", "symbol": "TEST-USDT", "positionSide": "LONG"},
    ]
    open_order_reads = iter([
        {"status": "ok", "sl_orders": [sl], "tp_orders": []},
        {"status": "ok", "sl_orders": [repaired_sl], "tp_orders": repaired_tps},
    ])
    calls = []
    registered = []
    monkeypatch.setattr(ro, "get_positions", lambda **_kwargs: [position])
    monkeypatch.setattr(ro, "get_open_orders_snapshot", lambda **_kwargs: {"status": "ok", "orders": [sl]})
    monkeypatch.setattr(ro, "get_open_protection_directional", lambda *_a, **_k: next(open_order_reads))
    monkeypatch.setattr(ro, "_load_active_trades", lambda: {})
    monkeypatch.setattr(ro, "FIXED_STOP_LOSS_PCT", 7.0)
    def fake_ensure(**kwargs):
        calls.append(kwargs)
        return {
            "status": "PROTECTED", "sl_result": {"status": "already_exists", "order_id": "SL-ORPHAN", "stop_price": live_sl_price, "qty": 1.0},
            "tp_orders": [
                {"status": "created", "leg": "tp1", "order_id": "TP1", "qty": 0.3, "pnl_pct": 1.0},
                {"status": "created", "leg": "tp2", "order_id": "TP2", "qty": 0.7, "pnl_pct": 2.0},
            ],
            "effective_tp_levels": [{"leg": "tp1", "pnl_pct": 1.0, "close_fraction": 0.3}, {"leg": "tp2", "pnl_pct": 2.0, "close_fraction": 0.7}],
            "tp_mode": "two_tp",
        }
    monkeypatch.setattr(ro, "ensure_directional_protection", fake_ensure)
    monkeypatch.setattr(ro, "update_active_trade_protection", lambda **_kwargs: False)
    monkeypatch.setattr(ro, "register_active_trade", lambda **kwargs: registered.append(kwargs))
    monkeypatch.setattr(ro, "record_action", lambda *_a, **_k: None)
    monkeypatch.setattr(ro, "record_trade", lambda *_a, **_k: None)
    monkeypatch.setattr(ro, "_record_reconciled_trade_open", lambda *_a, **_k: None)

    ro.reconcile_all_open_positions()

    assert len(calls) == 1
    assert calls[0]["stop_loss_price"] == live_sl_price
    assert calls[0]["stop_loss_pct"] == 7.0  # metadata fallback is not allowed to overwrite the live trigger
    assert registered, "reconciled orphan should be registered after protection is verified"


def test_rollback_followup_does_not_cleanup_protection_when_residual_is_found(monkeypatch):
    import run_once as ro

    position = {"symbol": "TEST-USDT", "positionSide": "LONG", "positionAmt": "1", "avgPrice": "100"}
    monkeypatch.setattr(ro, "get_position_directional", lambda *_a, **_k: {
        "status": "found", "symbol": "TEST-USDT", "positionSide": "LONG", "positionAmt": "0.4", "avgPrice": "100.2"
    })
    rescue_calls = []
    monkeypatch.setattr(ro, "install_protection", lambda **kwargs: rescue_calls.append(kwargs) or {
        "status": "PROTECTED", "sl_result": {"status": "already_exists", "order_id": "SL1"},
        "tp_orders": [{"status": "already_exists", "order_id": "TP1"}],
    })
    monkeypatch.setattr(ro, "_finalize_rollback_protection_cleanup", lambda *_a, **_k: pytest.fail("must not cleanup while a residual is live"))
    monkeypatch.setattr(ro, "record_action", lambda *_a, **_k: None)

    status, recovered_position, protection = ro._handle_rollback_followup(
        symbol="TEST-USDT", direction="LONG", event_id="EV1", trade_id="T1",
        last_known_position=position, protection={"rolled_back": True, "emergency_close": {"status": "closed"}},
        setup={"risk_pct": 1}, sl_pct=1.0, tp_levels=[{"leg": "tp1", "pnl_pct": 1.0, "close_fraction": 1.0}],
        last_known_qty=1.0, last_known_avg_price=100.0,
    )
    assert status == "opened_protected"
    assert recovered_position["positionAmt"] == 0.4
    assert rescue_calls[0]["position"]["positionAmt"] == 0.4
    assert protection["rolled_back"] is False


def test_rollback_followup_defers_cleanup_when_position_read_is_unknown(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "get_position_directional", lambda *_a, **_k: {"status": "error", "error": "timeout"})
    monkeypatch.setattr(ro, "_finalize_rollback_protection_cleanup", lambda *_a, **_k: pytest.fail("unknown state cannot authorize cleanup"))
    monkeypatch.setattr(ro, "record_action", lambda *_a, **_k: None)
    status, recovered_position, protection = ro._handle_rollback_followup(
        symbol="TEST-USDT", direction="SHORT", event_id="EV2", trade_id="T2",
        last_known_position={"symbol": "TEST-USDT", "positionAmt": 2.0, "avgPrice": 50.0},
        protection={"rolled_back": True, "emergency_close": {"status": "closed"}},
        setup={}, sl_pct=2.0, tp_levels=[], last_known_qty=2.0, last_known_avg_price=50.0,
    )
    assert status == "opened_protection_check_required"
    assert recovered_position["positionAmt"] == 2.0
    assert protection["rollback_state_unknown"] is True


def test_get_contract_normalizes_compact_usdt_symbol(monkeypatch):
    import event_engine.bingx as bx
    contract = {"symbol": "BTC-USDT", "displayName": "BTC-USDT", "status": 1, "apiStateOpen": "true"}
    monkeypatch.setattr(bx, "SYMBOL_MAP", {})
    monkeypatch.setattr(bx, "contracts", lambda **_kwargs: {"BTC-USDT": contract})
    assert bx.get_contract("BTCUSDT") is contract


def test_cashflow_recovery_resolves_missing_entry_order_id_from_client_id(tmp_path, monkeypatch):
    import json
    from event_engine import tracker as tr

    now_ms = 1_800_000_000_000
    open_ts = now_ms - 60_000
    close_ts = now_ms - 1_000
    path = tmp_path / "trades.jsonl"
    records = [
        {
            "record_type": "TRADE_OPEN", "event_id": "EVT_ACK_MISSING", "symbol": "BTC",
            "direction": "LONG", "ts": open_ts,
            "execution": {"order_id": None},
            "result": {"status": "opened_protected", "open_result": {"order_id": None, "client_order_id": "CID-ENTRY-1", "entry_ack_missing_order_id": True}},
        },
        {
            "record_type": "TRADE_CLOSE", "event_id": "EVT_ACK_MISSING", "symbol": "BTC",
            "direction": "LONG", "closed_ts": close_ts, "ts": close_ts,
            "realized_pnl_usdt": None, "fees_usdt": None, "funding_usdt": None, "net_pnl_usdt": None,
            "pnl_costs_status": "UNVERIFIED_POSITION_ATTRIBUTION",
            "cashflow_recovery": {"status": "PENDING", "attempts": 0, "next_retry_ts": 0},
        },
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    monkeypatch.setattr(tr, "TRADES_PATH", path)
    lookups = []
    monkeypatch.setattr(tr, "get_order_by_client_order_id", lambda symbol, cid: lookups.append((symbol, cid)) or {"status": "ok", "order_id": "EXCHANGE-ENTRY-1"})
    def fail_fill(symbol, direction, start, end, entry_id, order_ids):
        assert entry_id == "EXCHANGE-ENTRY-1"
        assert "EXCHANGE-ENTRY-1" in order_ids
        return None, "fill_history_temporarily_unavailable"
    monkeypatch.setattr(tr, "_load_exchange_fill_evidence", fail_fill)

    assert tr._recover_trade_cashflows(now_ms=now_ms, max_records=1) == 0
    updated_close = json.loads(path.read_text(encoding="utf-8").splitlines()[1])
    assert lookups == [("BTC", "CID-ENTRY-1")]
    assert updated_close["cashflow_recovery"]["resolved_entry_order_id"] == "EXCHANGE-ENTRY-1"
    assert updated_close["cashflow_recovery"]["status"] == "PENDING"
    assert updated_close["net_pnl_usdt"] is None


def test_coinalyze_page_cap_marks_truncation_as_incomplete(monkeypatch):
    import playwright.sync_api as sync_api
    import event_engine.coinalyze as coinalyze

    class Page:
        def query_selector(self, selector):
            return object() if selector == ".pagination" else None
        def content(self):
            return "page-html"
    class Browser:
        def close(self):
            pass
    class Context:
        def __enter__(self):
            return object()
        def __exit__(self, *_args):
            return False

    page = Page()
    monkeypatch.setattr(coinalyze, "MAX_PAGES", 1)
    monkeypatch.setattr(sync_api, "sync_playwright", lambda: Context())
    monkeypatch.setattr(coinalyze, "_setup_browser_context", lambda _p: (Browser(), page))
    monkeypatch.setattr(coinalyze, "_load_page", lambda *_a: "page-html")
    monkeypatch.setattr(coinalyze, "get_page_urls", lambda _html: [coinalyze.COINALYZE_URL])
    monkeypatch.setattr(coinalyze, "parse_table", lambda _html: [type("R", (), {"symbol": "AAA"})()])
    monkeypatch.setattr(coinalyze, "click_next_page", lambda *_a: True)

    with pytest.raises(coinalyze.CoinalyzeIncompleteDataError, match="pagination incomplete"):
        coinalyze.fetch_data()


def test_successful_market_ack_without_order_id_resolves_direct_order_id_without_name_error(monkeypatch):
    """The direct-ID recovery branch must not call an undefined float helper."""
    import event_engine.bingx as bx

    _prepare_market_order_without_id(monkeypatch, bx)
    monkeypatch.setattr(bx, "get_order_by_client_order_id", lambda *_a, **_k: {
        "status": "ok", "order_id": "OID-RECOVERED", "order_status": "FILLED", "executed_qty": "2.5"
    })

    result = bx.open_market("TEST", "LONG", 100.0, "TRADE-X")

    assert result["status"] == "opened"
    assert result["order_id"] == "OID-RECOVERED"
    assert result["order_status"] == "FILLED"


def test_close_notification_pending_skips_exchange_lifecycle_calls_and_recovers_queue(monkeypatch, tmp_path):
    """A closed trade waiting for queue persistence must not be processed as live."""
    import json
    import event_engine.tracker as tr

    active_path = tmp_path / "active_trades.json"
    notifications_path = tmp_path / "notifications.json"
    trades_path = tmp_path / "trades.jsonl"
    trade = {
        "event_id": "EVT_CLOSE_NOTIFY_PENDING",
        "trade_id": "TR_CLOSE_NOTIFY_PENDING",
        "symbol": "TEST-USDT",
        "direction": "LONG",
        "entry_price": 100.0,
        "initial_qty": 1.0,
        "remaining_qty": 0.0,
        "entry_ts": 1_700_000_000_000,
        "exit_price": 99.0,
        "exit_reason": "STOP_LOSS",
        "duration_min": 12.0,
        "planned_weighted_rr": 1.5,
        "closed": False,
        "close_notification_pending": True,
        "close_notification_text": "persisted close message",
        "close_cleanup_pending": False,
        "close_cleanup_verified": True,
    }
    active_path.write_text(json.dumps({trade["event_id"]: trade}), encoding="utf-8")
    trades_path.write_text(json.dumps({
        "record_type": "TRADE_CLOSE", "event_id": trade["event_id"], "symbol": "TEST-USDT",
        "direction": "LONG", "entry_price": 100.0, "exit_price": 99.0,
        "realized_pnl_pct": -1.0, "realized_rr": -1.0, "effective_weighted_rr": 1.5,
        "duration_min": 12.0, "exit_reason": "STOP_LOSS", "event_type": "TEST", "max_drawdown_pct": 1.0,
    }) + "\n", encoding="utf-8")

    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tr, "NOTIFICATIONS_PATH", notifications_path)
    monkeypatch.setattr(tr, "TRADES_PATH", trades_path)
    monkeypatch.setattr(tr, "_retry_pending_notifications", lambda: None)
    monkeypatch.setattr(tr, "get_position_directional", lambda *_a, **_k: pytest.fail("must not poll position for pending close notification"))
    monkeypatch.setattr(tr, "get_order", lambda *_a, **_k: pytest.fail("must not query orders for pending close notification"))
    monkeypatch.setattr(tr, "fetch_klines", lambda *_a, **_k: pytest.fail("must not fetch candles for pending close notification"))
    monkeypatch.setattr(tr, "_cleanup_closed_trade_protection", lambda *_a, **_k: pytest.fail("cleanup was already verified"))

    def fail_queue(*_args, **_kwargs):
        raise OSError("temporary notification storage failure")
    monkeypatch.setattr(tr, "_queue_notification", fail_queue)
    tr.update_active_trades()
    saved = json.loads(active_path.read_text(encoding="utf-8"))
    assert saved[trade["event_id"]]["close_notification_pending"] is True
    assert saved[trade["event_id"]]["closed"] is False

    def persist_queue(notification_id, **kwargs):
        notifications_path.write_text(json.dumps({notification_id: {
            "notification_id": notification_id, "event_id": kwargs["event_id"],
            "kind": kwargs["kind"], "text": kwargs["text"], "resolved": False, "chats": {"A": {"sent": False}}
        }}), encoding="utf-8")
        return False  # Durable queue entry, delivery may still be retried later.
    monkeypatch.setattr(tr, "_queue_notification", persist_queue)
    tr.update_active_trades()
    assert json.loads(active_path.read_text(encoding="utf-8")) == {}
    persisted = json.loads(notifications_path.read_text(encoding="utf-8"))
    assert next(iter(persisted.values()))["event_id"] == trade["event_id"]
