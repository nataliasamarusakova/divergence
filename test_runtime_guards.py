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
