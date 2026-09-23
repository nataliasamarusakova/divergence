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

    events_path = Path("data/events.jsonl")
    if not events_path.exists():
        import pytest
        pytest.skip("no journal in this checkout")

    divergence, other = [], []
    with events_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            ev = json.loads(line)
            engine = str((ev.get("event_fact") or {}).get("engine") or "")
            (divergence if engine == "DIVERGENCE" else other).append(ev)
            if len(divergence) >= 300 and len(other) >= 300:
                break

    assert divergence, "journal has no divergence events to replay"
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
