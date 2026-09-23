import json
from pathlib import Path


def test_repeated_pre_order_drift_failures_are_counted(tmp_path, monkeypatch):
    import run_once

    path = tmp_path / "trades.jsonl"
    rows = [
        {"record_type": "EXECUTION_ATTEMPT", "event_id": "EVT_A", "result": {"status": "PRE_ORDER_DRIFT_EXCEEDED"}},
        {"record_type": "EXECUTION_ATTEMPT", "event_id": "EVT_A", "result": {"status": "PRE_ORDER_DRIFT_EXCEEDED"}},
        {"record_type": "EXECUTION_ATTEMPT", "event_id": "EVT_B", "result": {"status": "OPEN_FAILED"}},
    ]
    path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    assert run_once.load_pre_order_drift_failure_counts(path) == {"EVT_A": 2}


def test_terminal_event_ids_are_loaded(tmp_path):
    import run_once

    path = tmp_path / "trades.jsonl"
    rows = [
        {"record_type": "EVENT_TERMINAL", "event_id": "EVT_A", "reason": "ENTRY_DRIFT_EXHAUSTED"},
        {"record_type": "TRADE_CLOSE", "event_id": "EVT_B"},
    ]
    path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    assert run_once.load_terminal_event_ids(path) == {"EVT_A"}


REAL_DIVERGENCE_EVENT_TYPES = [
    "REGULAR_BULLISH_RSI", "REGULAR_BEARISH_STOCH", "REGULAR_BULLISH_MACD",
    "HIDDEN_BULLISH_OBV", "HIDDEN_BEARISH_RSI", "REGULAR_BEARISH_OI",
]


def _event(event_type, engine="DIVERGENCE"):
    return {"event_id": "EVT_X", "event_type": event_type, "direction": "LONG",
            "timeframe": "1h", "event_fact": {"engine": engine}}


def test_divergence_is_recognised_by_engine_not_by_event_type():
    """Behavioural guard for the shadow/enable predicate.

    Divergence detectors never emit event_type == "DIVERGENCE"; they emit
    REGULAR_/HIDDEN_<INDICATOR>. A literal comparison matched nothing and
    silently voided DIVERGENCE_SHADOW_ONLY.
    """
    import run_once

    for event_type in REAL_DIVERGENCE_EVENT_TYPES:
        assert run_once._is_divergence_event(_event(event_type)), event_type


def test_divergence_predicate_falls_back_to_event_type_prefix():
    import run_once

    # Older journal rows may carry no engine field at all.
    assert run_once._is_divergence_event(_event("REGULAR_BULLISH_RSI", engine=""))
    assert run_once._is_divergence_event(_event("HIDDEN_BEARISH_MACD", engine=""))


def test_non_divergence_engines_are_not_treated_as_divergence():
    import run_once

    for event_type, engine in [
        ("MA_COMPRESSION_BREAKOUT", "MA_COMPRESSION"),
        ("BREAKER_BLOCK_BULLISH", "BREAKER_BLOCK"),
        ("MITIGATION_BLOCK_BEARISH", "MITIGATION_BLOCK"),
        ("CRT_BULLISH", "CRT"),
        ("VOLATILITY_SQUEEZE_RELEASE", "VOLATILITY_SQUEEZE"),
        ("SHORT_SQUEEZE", ""),
    ]:
        assert not run_once._is_divergence_event(_event(event_type, engine)), event_type


def test_every_requires_retest_engine_has_a_configured_retest_window():
    """A retest engine without its own window silently falls back to 30 minutes.

    LIQUIDITY_SWEEP shipped without an entry, giving it two 15m candidate bars
    against the 4-8 bars every other retest engine receives.
    """
    import run_once

    retest_engines = [
        "MA_COMPRESSION", "BREAKOUT_MOMENTUM", "DONCHIAN_RETEST", "EMA_PULLBACK",
        "ORDER_BLOCK", "BREAKER_BLOCK", "MITIGATION_BLOCK", "SFP",
        "LIQUIDATION_CASCADE_FVG", "CRT", "LIQUIDITY_SWEEP",
    ]
    for engine in retest_engines:
        ev = {"event_type": "X", "event_fact": {"engine": engine, "requires_retest": True}}
        window = run_once._event_trigger_max_delay_min(ev)
        assert window > run_once.MAX_TRIGGER_DELAY, (
            f"{engine} falls back to the non-retest default of {run_once.MAX_TRIGGER_DELAY} min"
        )


def test_divergence_shadow_state_records_and_closes(tmp_path):
    from pathlib import Path
    from event_engine.shadow import record_divergence_shadow_open, update_divergence_shadow_state

    path = Path(tmp_path) / 'shadow.json'
    setup = {
        'invalidation_price': 99.0,
        'target_price': 102.5,
        'risk_pct': 2.5,
    }
    opened = record_divergence_shadow_open(
        path, event_id='EVT1', symbol='TEST', direction='LONG',
        event_type='REGULAR_BULLISH_RSI', timeframe='1h',
        entry_price=100.0, setup=setup, score=70.0, opened_ts=1000,
    )
    assert opened['status'] == 'opened'
    result = update_divergence_shadow_state(path, {'TEST': 106.25}, 2000)
    assert result['closed'] == 1
    data = __import__('json').loads(path.read_text())
    assert data['EVT1']['status'] == 'CLOSED'
    assert data['EVT1']['exit_reason'] == 'SHADOW_TP3'
    assert round(data['EVT1']['realized_pnl_pct'], 6) == 6.25


def test_real_journal_divergence_events_route_to_shadow(tmp_path):
    """End-to-end guard against the shipped P0.

    Replays real events.jsonl rows through the predicate and the shadow
    lifecycle, so a regression cannot hide behind a synthetic event_type.
    """
    import json
    import run_once
    from event_engine.shadow import record_divergence_shadow_open, update_divergence_shadow_state

    fixture_path = Path("tests/fixtures/real_events.jsonl")
    assert fixture_path.exists(), "missing committed real-event fixture"

    divergence, other = [], []
    with fixture_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            ev = json.loads(line)
            engine = str((ev.get("event_fact") or {}).get("engine") or "")
            (divergence if engine == "DIVERGENCE" else other).append(ev)

    assert divergence, "real-event fixture has no divergence events"
    assert other, "real-event fixture has no non-divergence events"
    assert all(run_once._is_divergence_event(ev) for ev in divergence)
    assert not any(run_once._is_divergence_event(ev) for ev in other)
    # The literal comparison that shipped would have matched none of them.
    assert not any(str(ev.get("event_type", "")).upper() == "DIVERGENCE" for ev in divergence)

    state = tmp_path / "shadow.json"
    sample = divergence[0]
    opened = record_divergence_shadow_open(
        state,
        event_id=str(sample["event_id"]),
        symbol=str(sample["symbol"]),
        direction=str(sample["direction"]),
        event_type=str(sample["event_type"]),
        timeframe=str(sample.get("timeframe", "1h")),
        entry_price=100.0,
        setup={"risk_pct": 2.0},
        score=80.0,
        opened_ts=1_700_000_000_000,
    )
    assert opened["status"] == "opened"
    assert opened["event_type"].startswith(("REGULAR_", "HIDDEN_"))
    moved = update_divergence_shadow_state(
        state, {str(sample["symbol"]): 105.1}, 1_700_000_600_000
    )
    assert moved["closed"] == 1 if str(sample["direction"]).upper() == "LONG" else moved["active"] >= 0

def test_retest_window_failure_is_classified_as_no_window_not_data_error():
    import run_once

    stats = {"rejected_trigger": 0, "trigger_no_window": 0, "trigger_data_failed": 0,
             "trigger_breakout_failed": 0, "trigger_volume_failed": 0}
    tf_stats = dict(stats)

    run_once._record_trigger_failure(stats, tf_stats, "no_retest_window")

    assert stats["rejected_trigger"] == 1
    assert stats["trigger_no_window"] == 1
    assert stats["trigger_data_failed"] == 0
    assert tf_stats["trigger_no_window"] == 1
    assert tf_stats["trigger_data_failed"] == 0


def test_trigger_failure_categories_remain_distinct():
    import run_once

    for reason, expected in [
        ("no_trigger_window", "trigger_no_window"),
        ("breakout_failed", "trigger_breakout_failed"),
        ("volume_failed", "trigger_volume_failed"),
        ("invalid_15m_data", "trigger_data_failed"),
    ]:
        stats = {"rejected_trigger": 0, "trigger_no_window": 0, "trigger_data_failed": 0,
                 "trigger_breakout_failed": 0, "trigger_volume_failed": 0}
        tf_stats = dict(stats)
        run_once._record_trigger_failure(stats, tf_stats, reason)
        assert stats["rejected_trigger"] == 1
        assert stats[expected] == 1
        for key in ("trigger_no_window", "trigger_data_failed", "trigger_breakout_failed", "trigger_volume_failed"):
            if key != expected:
                assert stats[key] == 0, (reason, key)

def test_scan_errors_are_preserved_by_stage():
    import run_once

    stats = {"scan_errors": 0, "scan_errors_by_stage": {}}
    tf_stats = {"scan_errors": 0}

    run_once._record_scan_error(stats, "risk_1h_fetch")
    run_once._record_scan_error(stats, "risk_1h_fetch")
    run_once._record_scan_error(stats, "timeframe_1h_fetch_detection", tf_stats)

    assert stats["scan_errors"] == 3
    assert stats["scan_errors_by_stage"] == {
        "risk_1h_fetch": 2,
        "timeframe_1h_fetch_detection": 1,
    }
    assert tf_stats["scan_errors"] == 1



def test_bingx_contract_exists_returns_false_for_unknown_symbol(monkeypatch):
    import event_engine.bingx as bingx

    monkeypatch.setattr(bingx, "get_contract", lambda symbol: None)
    assert bingx.contract_exists("XAU") is False


def _coinalyze_test_row(symbol="BTC"):
    from event_engine.coinalyze import CoinalyzeRow

    return CoinalyzeRow(
        symbol, symbol, 100.0, 0.0, 50_000_000.0, 15_000_000.0,
        0.0, 0.0, 0.0, 0.0, 0.05, None, 100_000.0, 100_000.0,
        1.0, 0.0, 0.0, 0.0, {},
    )


def test_coinalyze_page_failure_raises_with_partial_rows(monkeypatch):
    import playwright.sync_api as sync_api
    import event_engine.coinalyze as coinalyze

    row1 = _coinalyze_test_row("AAA")
    browser = type("Browser", (), {"close": lambda self: None})()
    page = type("Page", (), {"query_selector": lambda self, selector: None})()
    state = {"failed": True}
    htmls = {"p1": "HTML1", "p2": "HTML2"}

    class _PlaywrightContext:
        def __enter__(self):
            return object()
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(sync_api, "sync_playwright", lambda: _PlaywrightContext())
    monkeypatch.setattr(coinalyze, "_setup_browser_context", lambda _p: (browser, page))
    monkeypatch.setattr(coinalyze, "get_page_urls", lambda _html: ["p1", "p2"])
    monkeypatch.setattr(coinalyze, "parse_table", lambda html: [row1] if html == "HTML1" else [_coinalyze_test_row("BBB")])

    def fake_load(_page, url):
        if url == coinalyze.COINALYZE_URL:
            return htmls["p1"]
        if url == "p2" and state["failed"]:
            raise RuntimeError("page 2 transient failure")
        return htmls[url]

    monkeypatch.setattr(coinalyze, "_load_page", fake_load)

    try:
        coinalyze.fetch_data()
        assert False, "expected incomplete pagination error"
    except coinalyze.CoinalyzeIncompleteDataError as exc:
        assert [row.symbol for row in exc.rows] == ["AAA"]
        assert "1 page" in str(exc)


def test_coinalyze_page_retry_then_all_pages_succeed(monkeypatch):
    import playwright.sync_api as sync_api
    import event_engine.coinalyze as coinalyze

    browser = type("Browser", (), {"close": lambda self: None})()
    page = type("Page", (), {"query_selector": lambda self, selector: None})()

    class _PlaywrightContext:
        def __enter__(self):
            return object()
        def __exit__(self, *args):
            return False

    monkeypatch.setattr(sync_api, "sync_playwright", lambda: _PlaywrightContext())
    monkeypatch.setattr(coinalyze, "_setup_browser_context", lambda _p: (browser, page))
    monkeypatch.setattr(coinalyze, "get_page_urls", lambda _html: ["p1", "p2"])
    monkeypatch.setattr(
        coinalyze,
        "_load_page",
        lambda _page, url: "p1" if url == coinalyze.COINALYZE_URL else url,
    )
    monkeypatch.setattr(
        coinalyze,
        "parse_table",
        lambda html: [_coinalyze_test_row("AAA")] if html == "p1" else [_coinalyze_test_row("BBB")],
    )

    rows = coinalyze.fetch_data()
    assert [row.symbol for row in rows] == ["AAA", "BBB"]


def test_incomplete_coinalyze_rows_cannot_enter_new_entry_universe():
    import run_once

    row = _coinalyze_test_row("AAA")
    assert run_once._coinalyze_rows_for_new_entries([row], complete=True) == [row]
    assert run_once._coinalyze_rows_for_new_entries([row], complete=False) == []


def test_reconciliation_restart_partial_qty_reuses_existing_trade_without_reregister(monkeypatch):
    import run_once as ro

    position = {
        "symbol": "TEST-USDT",
        "positionSide": "LONG",
        "positionAmt": "0.600",
        "avgPrice": "100.0",
    }
    active = {
        "EVT_ORIGINAL": {
            "symbol": "TEST",
            "direction": "LONG",
            "closed": False,
            "hit_legs": ["tp1"],
            "be_activated": True,
            "effective_tp_levels": [
                {"leg": "tp1", "pnl_pct": 1.0, "close_fraction": 0.25, "qty": 0.25},
                {"leg": "tp2", "pnl_pct": 2.0, "close_fraction": 0.40, "qty": 0.40},
                {"leg": "tp3", "pnl_pct": 3.0, "close_fraction": 0.35, "qty": 0.35},
            ],
            "tp_mode": "multi_tp",
            "effective_weighted_rr": 2.0,
            "planned_risk_pct": 1.0,
        }
    }
    sl = {"orderId": "SL1", "type": "STOP_MARKET", "stopPrice": "100.0", "origQty": "0.6"}
    tps = [
        {"orderId": "TP2", "type": "TAKE_PROFIT_MARKET", "stopPrice": "102.0", "origQty": "0.24"},
        {"orderId": "TP3", "type": "TAKE_PROFIT_MARKET", "stopPrice": "103.0", "origQty": "0.21"},
    ]
    calls = {"update": 0, "register": 0, "repair": 0}

    monkeypatch.setattr(ro, "to_bx_symbol", lambda symbol: "TEST-USDT")
    monkeypatch.setattr(ro, "get_positions", lambda **kwargs: [position])
    monkeypatch.setattr(ro, "_load_active_trades", lambda: active)
    monkeypatch.setattr(ro, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok", "sl_orders": [sl], "tp_orders": tps
    })
    monkeypatch.setattr(ro, "update_active_trade_protection", lambda **kwargs: calls.__setitem__("update", calls["update"] + 1) or True)
    monkeypatch.setattr(ro, "register_active_trade", lambda **kwargs: calls.__setitem__("register", calls["register"] + 1))
    monkeypatch.setattr(ro, "ensure_directional_protection", lambda **kwargs: calls.__setitem__("repair", calls["repair"] + 1) or {"status": "PROTECTED"})

    ro.reconcile_all_open_positions()

    assert calls["update"] == 1
    assert calls["register"] == 0
    assert calls["repair"] == 0


def test_incomplete_coinalyze_still_runs_position_reconciliation(monkeypatch):
    import run_once
    from event_engine.coinalyze import CoinalyzeIncompleteDataError

    trace = []
    row = _coinalyze_test_row("AAA")

    monkeypatch.setattr(run_once, "EXECUTION_ENABLED", True)
    monkeypatch.setattr(run_once, "_validate_execution_config", lambda: (True, ""))
    monkeypatch.setattr(run_once, "update_active_trades", lambda: trace.append("tracker"))
    monkeypatch.setattr(run_once, "reconcile_all_open_positions", lambda: trace.append("reconcile"))
    monkeypatch.setattr(run_once, "_fetch_market_klines_scan", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_once, "fetch_data", lambda: (_ for _ in ()).throw(CoinalyzeIncompleteDataError("page 2 failed", [row])))
    monkeypatch.setattr(run_once, "_record_oi_snapshots", lambda *args, **kwargs: 0)
    monkeypatch.setattr(run_once, "_record_funding_snapshots", lambda *args, **kwargs: 0)
    monkeypatch.setattr(run_once, "refresh_contracts", lambda: [])
    monkeypatch.setattr(run_once, "get_positions", lambda: [])
    monkeypatch.setattr(run_once, "_load_recent_successful_entries", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_once, "_load_symbol_quarantines", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_once, "load_successful_telegram_ids", lambda *args, **kwargs: set())
    monkeypatch.setattr(run_once, "send_pending_open_trade_notifications", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_once, "_refresh_timeframe_events", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_once, "_save_json_atomic", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_once, "_save_timeframe_scan_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_once, "_load_timeframe_scan_state", lambda: {})
    monkeypatch.setattr(run_once, "_load_cached_events", lambda: [])
    monkeypatch.setattr(run_once, "load_ids", lambda *args, **kwargs: set())
    monkeypatch.setattr(run_once, "load_successful_trade_ids", lambda *args, **kwargs: set())
    monkeypatch.setattr(run_once, "load_terminal_event_ids", lambda *args, **kwargs: set())
    monkeypatch.setattr(run_once, "load_pre_order_drift_failure_counts", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_once, "append_shadow_health", lambda *args, **kwargs: None)

    run_once.main()

    assert trace == ["tracker", "reconcile"]


def _set_execution_config(monkeypatch, *, mode, base, allow_live="false"):
    import run_once

    monkeypatch.setattr(run_once, "EXECUTION_ENABLED", True)
    monkeypatch.setattr(run_once, "API_KEY", "test-key")
    monkeypatch.setattr(run_once, "SECRET_KEY", "test-secret")
    monkeypatch.setattr(run_once, "POSITION_MODE", "HEDGE")
    monkeypatch.setattr(run_once, "EXECUTION_MODE", mode)
    monkeypatch.setattr(run_once, "BASE_URL", base)
    monkeypatch.setenv("ALLOW_LIVE_TRADING", allow_live)
    return run_once


def test_execution_config_accepts_supported_vst_modes_and_exact_vst_base(monkeypatch):
    for mode in ("vst", "test", "demo", "simulated"):
        run_once = _set_execution_config(
            monkeypatch, mode=mode, base="https://open-api-vst.bingx.com/"
        )
        ok, reason = run_once._validate_execution_config()
        assert ok, (mode, reason)
        assert reason == "OK"


def test_execution_config_rejects_unknown_mode(monkeypatch):
    run_once = _set_execution_config(
        monkeypatch, mode="foobar", base="https://open-api-vst.bingx.com"
    )
    ok, reason = run_once._validate_execution_config()
    assert not ok
    assert "Unsupported EXECUTION_MODE='foobar'" == reason


def test_execution_config_rejects_vst_mode_on_live_base(monkeypatch):
    run_once = _set_execution_config(
        monkeypatch, mode="vst", base="https://open-api.bingx.com"
    )
    ok, reason = run_once._validate_execution_config()
    assert not ok
    assert "requires BingX VST base URL" in reason


def test_execution_config_accepts_live_mode_only_on_live_base_with_explicit_flag(monkeypatch):
    run_once = _set_execution_config(
        monkeypatch,
        mode="live",
        base="https://open-api.bingx.com/",
        allow_live="true",
    )
    ok, reason = run_once._validate_execution_config()
    assert ok, reason
    assert reason == "OK"


def test_execution_config_rejects_live_mode_on_vst_base_even_when_flag_is_enabled(monkeypatch):
    run_once = _set_execution_config(
        monkeypatch,
        mode="live",
        base="https://open-api-vst.bingx.com",
        allow_live="true",
    )
    ok, reason = run_once._validate_execution_config()
    assert not ok
    assert "requires BingX live base URL" in reason


def test_execution_config_rejects_live_mode_without_explicit_flag(monkeypatch):
    run_once = _set_execution_config(
        monkeypatch, mode="prod-live", base="https://open-api.bingx.com", allow_live="false"
    )
    ok, reason = run_once._validate_execution_config()
    assert not ok
    assert reason == "Live execution requires explicit ALLOW_LIVE_TRADING=true"


def test_tp_leg_rejects_explicit_mismatched_client_order_id():
    from event_engine.bingx import _tp_leg_from_order

    order = {
        "type": "TAKE_PROFIT_MARKET",
        "stopPrice": "105.0",
        "clientOrderId": "EVTTP3ABC123",
    }
    assert _tp_leg_from_order(order, "tp2", 105.0, 2, "ABC123") is False
    assert _tp_leg_from_order(order, "tp3", 105.0, 2, "ABC123") is True

    legacy = dict(order, clientOrderId="EVT_ABC123_TP3")
    assert _tp_leg_from_order(legacy, "tp2", 105.0, 2, "ABC123") is False
    assert _tp_leg_from_order(legacy, "tp3", 105.0, 2, "ABC123") is True


def test_protection_does_not_reuse_one_existing_tp_order_for_two_legs(monkeypatch):
    from event_engine import bingx as bx

    monkeypatch.setattr(bx, "to_bx_symbol", lambda s: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda s: {
        "quantityPrecision": 3, "pricePrecision": 2, "tradeMinQuantity": 0,
    })

    existing_sl = {
        "orderId": "SL1", "type": "STOP_MARKET", "stopPrice": "95.00", "origQty": "1.0",
    }
    existing_tp = {
        "orderId": "TP1", "type": "TAKE_PROFIT_MARKET", "stopPrice": "101.00", "origQty": "0.5",
    }
    post_calls = []

    monkeypatch.setattr(
        bx,
        "get_open_protection_directional",
        lambda *a, **k: {"status": "ok", "sl_orders": [existing_sl], "tp_orders": [existing_tp]},
    )
    monkeypatch.setattr(bx, "_current_close_price", lambda s: 100.0)

    def fake_post(symbol, direction, params, client_order_id, **kwargs):
        post_calls.append((params, client_order_id))
        return {"code": 0, "data": {"order": {"orderId": "TP2"}}}

    monkeypatch.setattr(bx, "_post_protection_order_verified", fake_post)

    result = bx.ensure_directional_protection(
        "TEST", "LONG", 100.0, 1.0, 5.0,
        [
            {"leg": "tp1", "pnl_pct": 1.0, "close_fraction": 0.5},
            {"leg": "tp2", "pnl_pct": 1.0, "close_fraction": 0.5},
        ],
        trade_id="TRD1",
    )

    assert result["status"] == "PROTECTED"
    assert len(post_calls) == 1
    assert post_calls[0][0]["stopPrice"] == "101.00"
    assert [x["leg"] for x in result["tp_orders"]] == ["tp1", "tp2"]
    assert [x["order_id"] for x in result["tp_orders"]] == ["TP1", "TP2"]


def test_btc_regime_snapshot_rejects_non_finite_values():
    import pandas as pd
    import run_once

    base = [100.0, 101.0, 102.0, 103.0, 104.0]
    for bad_index in (-1, -2, -5):
        values = list(base)
        values[bad_index] = float("nan")
        out = run_once._btc_regime_snapshot(pd.DataFrame({"close": values}))
        assert out == {
            "btc_chg_1h_pct": None,
            "btc_chg_4h_pct": None,
            "btc_close": None,
            "btc_available": False,
        }

    for bad_index in (-1, -2, -5):
        values = list(base)
        values[bad_index] = float("inf")
        out = run_once._btc_regime_snapshot(pd.DataFrame({"close": values}))
        assert out == {
            "btc_chg_1h_pct": None,
            "btc_chg_4h_pct": None,
            "btc_close": None,
            "btc_available": False,
        }


def test_btc_filter_rejects_non_finite_values_as_insufficient_data():
    import pandas as pd
    from event_engine.signals import check_btc_regime

    base = [100.0, 101.0, 102.0, 103.0, 104.0]
    for bad_value in (float("nan"), float("inf"), float("-inf")):
        for bad_index in (-1, -2, -5):
            values = list(base)
            values[bad_index] = bad_value
            ok, reason = check_btc_regime(pd.DataFrame({"close": values}), "LONG")
            assert ok is True
            assert reason == "INSUFFICIENT_DATA"


def test_open_market_transport_error_queries_client_order_id_before_accepting_fill(monkeypatch):
    from event_engine import bingx as bx

    contract = {
        "symbol": "TEST-USDT", "displayName": "TEST-USDT", "status": 1,
        "apiStateOpen": "true", "quantityPrecision": 3,
        "tradeMinQuantity": 0.001, "tradeMinUSDT": 2,
        "maxLongLeverage": 10, "maxShortLeverage": 10,
    }
    monkeypatch.setattr(bx, "to_bx_symbol", lambda symbol: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda symbol: contract)
    monkeypatch.setattr(bx, "contract_exists", lambda symbol: True)
    monkeypatch.setattr(bx, "has_open_position", lambda symbol, direction: False)
    monkeypatch.setattr(bx, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(bx, "_set_leverage", lambda *args, **kwargs: True)
    monkeypatch.setattr(bx, "_request", lambda *args, **kwargs: {"code": -1, "msg": "read timed out"})
    monkeypatch.setattr(
        bx,
        "get_order_by_client_order_id",
        lambda symbol, client_id: {
            "status": "ok", "order_id": "OID1", "order_status": "FILLED",
            "executed_qty": 0.25, "avg_price": 100.2, "client_order_id": client_id,
        },
    )

    out = bx.open_market("TEST", "LONG", 100.0, "TRD_TRANSPORT")

    assert out["status"] == "opened"
    assert out["idempotency"] == "order_queried_after_transport_error"
    assert out["order_id"] == "OID1"
    assert out["executed_qty"] == 0.25


def test_open_market_transport_canceled_retries_once_with_fresh_client_order_id(monkeypatch):
    from event_engine import bingx as bx

    contract = {
        "symbol": "TEST-USDT", "status": 1, "apiStateOpen": "true",
        "quantityPrecision": 3, "tradeMinQuantity": 0.001,
        "tradeMinUSDT": 2, "maxLongLeverage": 10, "maxShortLeverage": 10,
    }
    post_calls = []
    lookup_calls = []
    monkeypatch.setattr(bx, "to_bx_symbol", lambda symbol: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda symbol: contract)
    monkeypatch.setattr(bx, "contract_exists", lambda symbol: True)
    monkeypatch.setattr(bx, "has_open_position", lambda symbol, direction: False)
    monkeypatch.setattr(bx, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(bx, "_set_leverage", lambda *args, **kwargs: True)

    def fake_request(method, path, params):
        post_calls.append(dict(params))
        if len(post_calls) == 1:
            return {"code": -1, "msg": "read timed out"}
        return {"code": 0, "data": {"order": {"orderId": "OID-RETRY", "clientOrderId": params["clientOrderId"]}}}

    monkeypatch.setattr(bx, "_request", fake_request)
    monkeypatch.setattr(
        bx,
        "get_order_by_client_order_id",
        lambda symbol, client_id: lookup_calls.append(client_id) or {
            "status": "ok", "order_id": "OID-CANCELED", "order_status": "CANCELED",
            "executed_qty": 0.0, "client_order_id": client_id,
        },
    )

    out = bx.open_market("TEST", "LONG", 100.0, "TRD_CANCELED_RETRY")

    assert out["status"] == "opened"
    assert out["open_attempt"] == 2
    assert len(post_calls) == 2
    assert len(lookup_calls) == 1
    assert post_calls[0]["clientOrderId"] == lookup_calls[0]
    assert post_calls[0]["clientOrderId"] != post_calls[1]["clientOrderId"]


def test_unknown_market_entry_is_explicitly_terminalized_for_next_cycle():
    import run_once as ro

    assert ro._market_entry_outcome_unknown({
        "status": "OPEN_FAILED",
        "open_result": {"status": "unknown", "client_order_id": "EVTOPENABC123"},
    }) is True
    assert ro._market_entry_outcome_unknown({
        "status": "OPEN_FAILED",
        "open_result": {"status": "error"},
    }) is False
    assert ro._market_entry_outcome_unknown({"status": "opened"}) is False

def test_open_market_transport_error_does_not_claim_pending_order_as_filled(monkeypatch):
    from event_engine import bingx as bx

    contract = {
        "symbol": "TEST-USDT", "status": 1, "apiStateOpen": "true",
        "quantityPrecision": 3, "tradeMinQuantity": 0.001,
        "tradeMinUSDT": 2, "maxLongLeverage": 10, "maxShortLeverage": 10,
    }
    monkeypatch.setattr(bx, "to_bx_symbol", lambda symbol: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda symbol: contract)
    monkeypatch.setattr(bx, "contract_exists", lambda symbol: True)
    monkeypatch.setattr(bx, "has_open_position", lambda symbol, direction: False)
    monkeypatch.setattr(bx, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(bx, "_set_leverage", lambda *args, **kwargs: True)
    monkeypatch.setattr(bx, "_request", lambda *args, **kwargs: {"code": -1, "msg": "read timed out"})
    monkeypatch.setattr(
        bx,
        "get_order_by_client_order_id",
        lambda symbol, client_id: {
            "status": "ok", "order_id": "OID2", "order_status": "NEW",
            "executed_qty": 0.0, "client_order_id": client_id,
        },
    )

    out = bx.open_market("TEST", "LONG", 100.0, "TRD_PENDING")

    assert out["status"] == "unknown"
    assert out["order_status"] == "NEW"
    assert out["clientOrderId"].startswith("EVTOPEN")


def test_protection_success_without_order_id_is_not_accepted(monkeypatch):
    import event_engine.bingx as bx

    calls = []
    monkeypatch.setattr(
        bx,
        "_request",
        lambda method, path, params: calls.append((method, path, dict(params))) or {"code": 0, "data": {"order": {}}},
    )
    out = bx._post_protection_order_verified(
        "TEST",
        "LONG",
        {
            "symbol": "TEST-USDT",
            "side": "SELL",
            "positionSide": "LONG",
            "type": "STOP_MARKET",
            "stopPrice": "95",
            "quantity": "1",
        },
        "EVTSLTEST",
        max_attempts=2,
    )
    assert out["code"] == -1
    assert out["malformed_success_response"] is True
    assert out["protection_state_unknown"] is True
    assert len(calls) == 1


def test_emergency_close_transport_error_uses_order_lookup_without_duplicate_post(monkeypatch):
    from event_engine import bingx as bx

    qty_states = iter([
        {"status": "found", "positionAmt": "1.250", "avgPrice": "100"},
        {"status": "not_found", "positionAmt": "0"},
    ])
    calls = []
    monkeypatch.setattr(bx, "to_bx_symbol", lambda symbol: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda symbol: {"quantityPrecision": 3})
    monkeypatch.setattr(bx, "get_position_directional", lambda symbol, direction: next(qty_states))
    monkeypatch.setattr(bx.time, "sleep", lambda *_: None)

    def fake_request(method, path, params):
        calls.append(dict(params))
        return {"code": -1, "msg": "read timed out"}

    monkeypatch.setattr(bx, "_request", fake_request)
    monkeypatch.setattr(
        bx,
        "get_order_by_client_order_id",
        lambda symbol, client_id: {
            "status": "ok", "order_id": "CLOSE1", "order_status": "FILLED",
            "executed_qty": 1.250, "avg_price": 99.5, "client_order_id": client_id,
        },
    )

    out = bx.emergency_close_position("TEST", "LONG", 1.25, reason_token="TRANSPORT")

    assert out["status"] == "closed"
    assert out["idempotency"] == "order_queried_after_transport_error"
    assert out["execution_price"] == 99.5
    assert len(calls) == 1


def test_emergency_close_transport_error_does_not_retry_when_order_lookup_unknown(monkeypatch):
    from event_engine import bingx as bx

    calls = []
    monkeypatch.setattr(bx, "to_bx_symbol", lambda symbol: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda symbol: {"quantityPrecision": 3})
    monkeypatch.setattr(
        bx, "get_position_directional",
        lambda symbol, direction: {"status": "found", "positionAmt": "1.250", "avgPrice": "100"},
    )
    monkeypatch.setattr(bx.time, "sleep", lambda *_: None)

    def fake_request(method, path, params):
        calls.append(dict(params))
        return {"code": -1, "msg": "read timed out"}

    monkeypatch.setattr(bx, "_request", fake_request)
    monkeypatch.setattr(
        bx,
        "get_order_by_client_order_id",
        lambda symbol, client_id: {"status": "error", "error": "order lookup timed out"},
    )

    out = bx.emergency_close_position("TEST", "LONG", 1.25, reason_token="UNKNOWN")

    assert out["status"] == "unknown"
    assert out["escalation_required"] is True
    assert len(calls) == 1


def test_emergency_close_transport_absent_lookup_retries_same_client_order_id(monkeypatch):
    from event_engine import bingx as bx

    states = iter([
        {"status": "found", "positionAmt": "1.250", "avgPrice": "100"},
        {"status": "found", "positionAmt": "1.250", "avgPrice": "100"},
        {"status": "not_found", "positionAmt": "0"},
    ])
    calls = []
    lookups = []
    monkeypatch.setattr(bx, "to_bx_symbol", lambda symbol: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda symbol: {"quantityPrecision": 3})
    monkeypatch.setattr(bx, "get_position_directional", lambda symbol, direction: next(states))
    monkeypatch.setattr(bx.time, "sleep", lambda *_: None)

    def fake_request(method, path, params):
        calls.append(dict(params))
        return {"code": -1, "msg": "read timed out"}

    monkeypatch.setattr(bx, "_request", fake_request)
    monkeypatch.setattr(
        bx,
        "get_order_by_client_order_id",
        lambda symbol, client_id: lookups.append(client_id) or (
            {"status": "ok", "order_id": "CLOSE4", "order_status": "FILLED",
             "executed_qty": 1.250, "avg_price": 99.4, "client_order_id": client_id}
            if len(lookups) == 2
            else {"status": "absent"}
        ),
    )

    out = bx.emergency_close_position("TEST", "LONG", 1.25, reason_token="ABSENT")

    assert out["status"] == "closed"
    assert len(calls) == 2
    assert calls[0]["clientOrderId"] == calls[1]["clientOrderId"]
    assert lookups == [calls[0]["clientOrderId"], calls[0]["clientOrderId"]]


def test_emergency_close_retry_after_verified_cancel_uses_new_client_order_id(monkeypatch):
    from event_engine import bingx as bx

    states = iter([
        {"status": "found", "positionAmt": "1.250", "avgPrice": "100"},
        {"status": "found", "positionAmt": "1.250", "avgPrice": "100"},
        {"status": "not_found", "positionAmt": "0"},
    ])
    calls = []
    lookup_calls = []
    monkeypatch.setattr(bx, "to_bx_symbol", lambda symbol: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda symbol: {"quantityPrecision": 3})
    monkeypatch.setattr(bx, "get_position_directional", lambda symbol, direction: next(states))
    monkeypatch.setattr(bx.time, "sleep", lambda *_: None)

    def fake_request(method, path, params):
        calls.append(dict(params))
        if len(calls) == 1:
            return {"code": -1, "msg": "read timed out"}
        return {"code": 0, "msg": "OK", "data": {"order": {"orderId": "CLOSE2", "avgPrice": "99.5"}}}

    monkeypatch.setattr(bx, "_request", fake_request)
    monkeypatch.setattr(
        bx,
        "get_order_by_client_order_id",
        lambda symbol, client_id: lookup_calls.append(client_id) or {
            "status": "ok", "order_id": "CANCEL1", "order_status": "CANCELED",
            "executed_qty": 0.0, "client_order_id": client_id,
        },
    )

    out = bx.emergency_close_position("TEST", "LONG", 1.25, reason_token="CANCELED")

    assert out["status"] == "closed"
    assert len(calls) == 2
    assert calls[0]["clientOrderId"] != calls[1]["clientOrderId"]
    assert lookup_calls == [calls[0]["clientOrderId"]]


def test_emergency_close_acknowledged_fill_with_stale_position_refuses_duplicate(monkeypatch):
    import event_engine.bingx as bx

    calls = []
    position_states = iter([
        {"status": "found", "positionAmt": "1.250", "avgPrice": "100"},
        {"status": "found", "positionAmt": "1.250", "avgPrice": "100"},
    ])
    monkeypatch.setattr(bx, "to_bx_symbol", lambda symbol: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda symbol: {"quantityPrecision": 3})
    monkeypatch.setattr(bx, "get_position_directional", lambda *a, **k: next(position_states))

    def fake_request(method, path, params):
        calls.append(dict(params))
        return {"code": 0, "data": {"order": {"orderId": "CLOSE1", "clientOrderId": params["clientOrderId"], "avgPrice": "99.5"}}}

    monkeypatch.setattr(bx, "_request", fake_request)
    monkeypatch.setattr(
        bx,
        "get_order_by_client_order_id",
        lambda symbol, client_id: {
            "status": "ok", "order_id": "CLOSE1", "client_order_id": client_id,
            "order_status": "FILLED", "executed_qty": 1.25, "avg_price": 99.5,
        },
    )
    monkeypatch.setattr(bx.time, "sleep", lambda *_: None)

    out = bx.emergency_close_position("TEST", "LONG", 1.25, reason_token="STALE_POSITION")
    assert out["status"] == "unknown"
    assert out["escalation_required"] is True
    assert len(calls) == 1


def test_emergency_close_partial_fill_uses_new_order_for_verified_residual(monkeypatch):
    from event_engine import bingx as bx

    states = iter([
        {"status": "found", "positionAmt": "1.250", "avgPrice": "100"},
        {"status": "found", "positionAmt": "0.750", "avgPrice": "100"},
        {"status": "found", "positionAmt": "0.750", "avgPrice": "100"},
        {"status": "not_found", "positionAmt": "0"},
    ])
    calls = []
    monkeypatch.setattr(bx, "to_bx_symbol", lambda symbol: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda symbol: {"quantityPrecision": 3})
    monkeypatch.setattr(bx, "get_position_directional", lambda symbol, direction: next(states))
    monkeypatch.setattr(bx.time, "sleep", lambda *_: None)

    def fake_request(method, path, params):
        calls.append(dict(params))
        if len(calls) == 1:
            return {"code": -1, "msg": "read timed out"}
        return {"code": 0, "msg": "OK", "data": {"order": {"orderId": "CLOSE3", "avgPrice": "99.0"}}}

    monkeypatch.setattr(bx, "_request", fake_request)
    monkeypatch.setattr(
        bx,
        "get_order_by_client_order_id",
        lambda symbol, client_id: {
            "status": "ok", "order_id": "PART1", "order_status": "PARTIALLY_FILLED",
            "executed_qty": 0.500, "avg_price": 99.0, "client_order_id": client_id,
        },
    )

    out = bx.emergency_close_position("TEST", "LONG", 1.25, reason_token="PARTIAL")

    assert out["status"] == "closed"
    assert len(calls) == 2
    assert calls[0]["clientOrderId"] != calls[1]["clientOrderId"]
    assert calls[1]["quantity"] == "0.750"


def test_get_order_by_client_order_id_queries_exchange_with_client_id(monkeypatch):
    from event_engine import bingx as bx

    captured = []
    monkeypatch.setattr(bx, "to_bx_symbol", lambda symbol: "TEST-USDT")

    def fake_request(method, path, params, signed=True, **kwargs):
        captured.append((method, path, dict(params), signed, kwargs))
        return {
            "code": 0,
            "data": {
                "order": {
                    "orderId": "OID9", "status": "FILLED", "avgPrice": "99.5",
                    "executedQty": "1.25", "origQty": "1.25",
                    "clientOrderId": "evtcid9",
                }
            },
        }

    monkeypatch.setattr(bx, "_request", fake_request)
    out = bx.get_order_by_client_order_id("TEST", "EVTCID9")

    assert out["status"] == "ok"
    assert out["order_id"] == "OID9"
    assert out["order_status"] == "FILLED"
    assert out["executed_qty"] == 1.25
    assert captured == [
        (
            "GET",
            bx.ORDER_PATH,
            {"symbol": "TEST-USDT", "clientOrderId": "EVTCID9"},
            True,
            {"retryable": False},
        )
    ]
