import json


def json_clone(value):
    return json.loads(json.dumps(value))

import run_once


def test_same_five_minute_window_does_not_advance_completed_buckets():
    # With the production 2-minute close grace, 13:02 and 13:12 UTC belong
    # to the same last safely-completed 1H and 4H buckets used by scan state.
    a = 13 * 3_600_000 + 2 * 60_000
    b = 13 * 3_600_000 + 12 * 60_000
    assert run_once._completed_bucket(3_600_000, a, 2.0) == 12
    assert run_once._completed_bucket(3_600_000, b, 2.0) == 12
    assert run_once._completed_bucket(14_400_000, a, 2.0) == 2
    assert run_once._completed_bucket(14_400_000, b, 2.0) == 2


def test_symbol_scan_due_false_when_watermark_matches_bucket():
    state = {"version": 2, "symbols": {"FIL": {"1h": 12, "4h": 2}}}
    assert run_once._symbol_scan_due(state, "FIL", "1h", 12) is False
    assert run_once._symbol_scan_due(state, "FIL", "4h", 2) is False
    assert run_once._symbol_scan_due(state, "FIL", "1h", 13) is True
    assert run_once._symbol_scan_due(state, "FIL", "4h", 3) is True


def test_timeframe_scan_state_loads_persisted_v2_symbols(tmp_path, monkeypatch):
    state_path = tmp_path / "timeframe_scan_state.json"
    state_path.write_text(
        '{"version": 2, "symbols": {"FIL": {"1h": 12, "4h": 2}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(run_once, "TIMEFRAME_STATE", state_path)

    state = run_once._load_timeframe_scan_state()

    assert state["version"] == 2
    assert state["symbols"]["FIL"]["1h"] == 12
    assert state["symbols"]["FIL"]["4h"] == 2


def test_timeframe_scan_state_missing_file_is_explicitly_empty(tmp_path, monkeypatch):
    state_path = tmp_path / "missing.json"
    monkeypatch.setattr(run_once, "TIMEFRAME_STATE", state_path)

    state = run_once._load_timeframe_scan_state()

    assert state == {"version": 2, "symbols": {}}


def test_timeframe_scan_state_malformed_file_is_explicitly_empty(tmp_path, monkeypatch):
    state_path = tmp_path / "broken.json"
    state_path.write_text('{broken', encoding="utf-8")
    monkeypatch.setattr(run_once, "TIMEFRAME_STATE", state_path)

    state = run_once._load_timeframe_scan_state()

    assert state == {"version": 2, "symbols": {}}


def test_timeframe_scan_state_unsupported_version_is_explicitly_empty(tmp_path, monkeypatch):
    state_path = tmp_path / "future.json"
    state_path.write_text(
        '{"version": 99, "symbols": {"FIL": {"1h": 12, "4h": 2}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(run_once, "TIMEFRAME_STATE", state_path)

    state = run_once._load_timeframe_scan_state()

    assert state == {"version": 2, "symbols": {}}


def test_scan_buckets_replays_contiguous_missing_closed_buckets():
    state = {"version": 2, "symbols": {"BTC": {"1h": 98}}}
    available = {96, 97, 98, 99, 100}
    assert run_once._scan_buckets_to_process(state, "BTC", "1h", 100, available) == [99, 100]


def test_scan_buckets_never_jumps_over_missing_bucket():
    state = {"version": 2, "symbols": {"BTC": {"1h": 98}}}
    available = {98, 100}
    assert run_once._scan_buckets_to_process(state, "BTC", "1h", 100, available) == []


def test_new_symbol_replays_only_bounded_recent_contiguous_buckets(monkeypatch):
    monkeypatch.setattr(run_once, "NEW_SYMBOL_BACKFILL_MIN", 120.0)
    state = {"version": 2, "symbols": {}}
    available = {97, 98, 99, 100}
    assert run_once._scan_buckets_to_process(state, "BTC", "1h", 100, available) == [99, 100]


def test_new_symbol_backfill_stops_at_recent_history_gap(monkeypatch):
    monkeypatch.setattr(run_once, "NEW_SYMBOL_BACKFILL_MIN", 180.0)
    state = {"version": 2, "symbols": {}}
    available = {96, 98, 99, 100}
    assert run_once._scan_buckets_to_process(state, "BTC", "1h", 100, available) == [98, 99, 100]


def test_cached_events_are_recovered_from_durable_event_journal(tmp_path, monkeypatch):
    now_ms = 1_800_000_000_000
    events_path = tmp_path / "events.jsonl"
    cache_path = tmp_path / "recent_event_cache.json"
    event = {
        "event_id": "EVT_RECOVER",
        "event_type": "SFP_BULLISH",
        "symbol": "BTC",
        "direction": "LONG",
        "timeframe": "1h",
        "timestamps": {"detected_at_ts": now_ms - 60_000},
        "event_fact": {"requires_retest": True},
    }
    events_path.write_text(__import__("json").dumps(event) + "\n", encoding="utf-8")
    cache_path.write_text('{"updated_ts": 0, "events": []}', encoding="utf-8")
    monkeypatch.setattr(run_once, "EVENTS", events_path)
    monkeypatch.setattr(run_once, "EVENT_CACHE", cache_path)

    recovered = run_once._load_cached_events(now_ms, set())

    assert [ev["event_id"] for ev in recovered] == ["EVT_RECOVER"]


def test_cached_event_recovery_excludes_terminal_events(tmp_path, monkeypatch):
    now_ms = 1_800_000_000_000
    events_path = tmp_path / "events.jsonl"
    cache_path = tmp_path / "recent_event_cache.json"
    event = {
        "event_id": "EVT_TERMINAL",
        "event_type": "SFP_BULLISH",
        "symbol": "BTC",
        "direction": "LONG",
        "timeframe": "1h",
        "timestamps": {"detected_at_ts": now_ms - 60_000},
        "event_fact": {"requires_retest": True},
    }
    events_path.write_text(__import__("json").dumps(event) + "\n", encoding="utf-8")
    cache_path.write_text('{"updated_ts": 0, "events": []}', encoding="utf-8")
    monkeypatch.setattr(run_once, "EVENTS", events_path)
    monkeypatch.setattr(run_once, "EVENT_CACHE", cache_path)

    assert run_once._load_cached_events(now_ms, {"EVT_TERMINAL"}) == []


def test_emit_event_returns_false_when_durable_journal_write_fails(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(run_once, "append_jsonl", fail)
    assert run_once.emit_event({"event_id": "EVT_FAIL"}) is False


def test_refresh_timeframe_events_replays_each_missing_bucket_without_detector_changes(tmp_path, monkeypatch):
    interval = 3_600_000
    now_ms = 101 * interval + 120_000
    rows = [
        {
            "open_time": b * interval,
            "close_time": b * interval + interval - 1,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0 + b * 0.01,
            "volume": 1.0,
        }
        for b in range(20, 101)
    ]
    events_path = tmp_path / "events.jsonl"
    state_path = tmp_path / "timeframe_scan_state.json"
    monkeypatch.setattr(run_once, "EVENTS", events_path)
    monkeypatch.setattr(run_once, "TIMEFRAME_STATE", state_path)
    monkeypatch.setattr(run_once, "_fetch_market_klines_scan", lambda *args, **kwargs: rows)
    monkeypatch.setattr(run_once, "_load_oi_history", lambda: {})
    monkeypatch.setattr(run_once, "_load_funding_history", lambda: {})
    monkeypatch.setattr(run_once, "add_cvd", lambda df: df)
    monkeypatch.setattr(run_once, "attach_oi_series", lambda df, _: df)
    monkeypatch.setattr(run_once, "attach_funding_series", lambda df, _: df)

    calls = []

    def fake_divergence(df, symbol, timeframe):
        bucket = int(df["close_time"].iloc[-1]) // interval
        calls.append(bucket)
        return [{
            "event_id": f"EVT_{bucket}",
            "event_type": "REGULAR_BULLISH_RSI",
            "symbol": symbol,
            "direction": "LONG",
            "timeframe": timeframe,
            "timestamps": {"detected_at_ts": int(df["close_time"].iloc[-1])},
            "event_fact": {"engine": "DIVERGENCE"},
        }]

    monkeypatch.setattr(run_once, "detect_divergences", fake_divergence)
    for name in (
        "detect_volume_profile_divergence", "detect_harmonic_patterns",
        "detect_squeeze_release", "detect_liquidation_squeeze", "detect_macd_4h",
        "detect_ma_compression_breakout", "detect_breakout_momentum",
        "detect_donchian_retest", "detect_liquidity_sweep_reclaim",
        "detect_ema_pullback_continuation", "detect_order_block", "detect_breaker_block",
        "detect_mitigation_block", "detect_sfp", "detect_liquidation_cascade_fvg", "detect_crt",
    ):
        monkeypatch.setattr(run_once, name, lambda *args, **kwargs: [])

    scan_state = {"version": 2, "symbols": {"BTC-USDT": {"1h": 98}}}
    stats = {"events_total": 0, "divergence_events": 0, "squeeze_events": 0, "scan_errors": 0}
    candidates = [type("Candidate", (), {"symbol": "BTC-USDT"})()]

    fresh = run_once._refresh_timeframe_events(
        candidates, "1h", 400, now_ms, set(), stats, scan_state, 100, {}
    )

    assert calls == [99, 100]
    assert scan_state["symbols"]["BTC-USDT"]["1h"] == 100
    assert len(fresh) == 1
    journal = [__import__("json").loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert [row["event_id"] for row in journal] == ["EVT_99", "EVT_100"]


def test_watermark_stops_before_bucket_with_failed_event_persistence(tmp_path, monkeypatch):
    interval = 3_600_000
    now_ms = 101 * interval + 120_000
    rows = [
        {
            "open_time": b * interval,
            "close_time": b * interval + interval - 1,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0 + b * 0.01,
            "volume": 1.0,
        }
        for b in range(20, 101)
    ]
    state_path = tmp_path / "timeframe_scan_state.json"
    monkeypatch.setattr(run_once, "TIMEFRAME_STATE", state_path)
    monkeypatch.setattr(run_once, "_fetch_market_klines_scan", lambda *args, **kwargs: rows)
    monkeypatch.setattr(run_once, "_load_oi_history", lambda: {})
    monkeypatch.setattr(run_once, "_load_funding_history", lambda: {})
    monkeypatch.setattr(run_once, "add_cvd", lambda df: df)
    monkeypatch.setattr(run_once, "attach_oi_series", lambda df, _: df)
    monkeypatch.setattr(run_once, "attach_funding_series", lambda df, _: df)

    def fake_divergence(df, symbol, timeframe):
        bucket = int(df["close_time"].iloc[-1]) // interval
        return [{
            "event_id": f"EVT_{bucket}",
            "event_type": "REGULAR_BULLISH_RSI",
            "symbol": symbol,
            "direction": "LONG",
            "timeframe": timeframe,
            "timestamps": {"detected_at_ts": int(df["close_time"].iloc[-1])},
            "event_fact": {"engine": "DIVERGENCE"},
        }]

    monkeypatch.setattr(run_once, "detect_divergences", fake_divergence)
    for name in (
        "detect_volume_profile_divergence", "detect_harmonic_patterns",
        "detect_squeeze_release", "detect_liquidation_squeeze", "detect_macd_4h",
        "detect_ma_compression_breakout", "detect_breakout_momentum",
        "detect_donchian_retest", "detect_liquidity_sweep_reclaim",
        "detect_ema_pullback_continuation", "detect_order_block", "detect_breaker_block",
        "detect_mitigation_block", "detect_sfp", "detect_liquidation_cascade_fvg", "detect_crt",
    ):
        monkeypatch.setattr(run_once, name, lambda *args, **kwargs: [])

    persisted = []
    monkeypatch.setattr(run_once, "_save_timeframe_scan_state", lambda state: persisted.append(json_clone(state)))

    def emit_with_one_failure(ev):
        return ev.get("event_id") != "EVT_100"

    monkeypatch.setattr(run_once, "emit_event", emit_with_one_failure)

    scan_state = {"version": 2, "symbols": {"BTC-USDT": {"1h": 98}}}
    stats = {"events_total": 0, "divergence_events": 0, "squeeze_events": 0, "scan_errors": 0}
    candidates = [type("Candidate", (), {"symbol": "BTC-USDT"})()]

    run_once._refresh_timeframe_events(
        candidates, "1h", 400, now_ms, set(), stats, scan_state, 100, {}
    )

    assert scan_state["symbols"]["BTC-USDT"]["1h"] == 99
    assert persisted and persisted[-1]["symbols"]["BTC-USDT"]["1h"] == 99


def test_trigger_age_uses_closed_trigger_bar_not_observation_time(monkeypatch):
    import run_once
    monkeypatch.setattr(run_once.time, "time", lambda: 1_200.0)
    meta = {"trigger_observed_at_ts": 1_180_000, "trigger_bar_close_ts": 1_000_000}
    assert run_once._trigger_age_min(meta) == 3.3333333333333335


def test_trigger_age_returns_none_without_closed_bar_timestamp():
    import run_once
    assert run_once._trigger_age_min({"trigger_observed_at_ts": 1_000_000}, 2_000_000) is None


def test_same_direction_candidate_prefers_newer_event_even_with_lower_score():
    import run_once
    older = {"event": {"timestamps": {"detected_at_ts": 1_000}}, "score": 95.0}
    newer = {"event": {"timestamps": {"detected_at_ts": 2_000}}, "score": 60.0}
    assert run_once._candidate_is_newer(newer, older) is True
    assert run_once._candidate_is_newer(older, newer) is False


def test_htf_context_frame_excludes_future_candles():
    import run_once
    import pandas as pd
    df = pd.DataFrame({"close_time": [100, 200, 300], "close": [1.0, 2.0, 3.0]})
    out = run_once._frame_through_event_ts(df, 200)
    assert list(out["close_time"]) == [100, 200]
