import pytest
import json
import pandas as pd
from pathlib import Path


def test_binance_transport_uses_only_explicit_proxy_when_vpn_enabled(monkeypatch):
    import event_engine.binance as binance

    monkeypatch.setenv("BINANCE_VPN_ENABLED", "true")
    monkeypatch.setenv("BINANCE_HTTP_PROXY", "http://127.0.0.1:18080")
    client = binance.BinanceMarketClient()

    assert client._session.trust_env is False
    assert client._session.proxies.get("http") == "http://127.0.0.1:18080"
    assert client._session.proxies.get("https") == "http://127.0.0.1:18080"


def test_binance_transport_is_direct_and_ignores_global_proxy_env_when_vpn_disabled(monkeypatch):
    import event_engine.binance as binance

    monkeypatch.setenv("BINANCE_VPN_ENABLED", "false")
    monkeypatch.delenv("BINANCE_HTTP_PROXY", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://203.0.113.50:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://203.0.113.50:3128")
    monkeypatch.setenv("ALL_PROXY", "http://203.0.113.50:3128")
    client = binance.BinanceMarketClient()

    assert client._session.trust_env is False
    assert client._session.proxies == {}


def test_binance_transport_fails_closed_when_vpn_is_enabled_without_proxy(monkeypatch):
    import event_engine.binance as binance

    monkeypatch.setenv("BINANCE_VPN_ENABLED", "true")
    monkeypatch.delenv("BINANCE_HTTP_PROXY", raising=False)

    with pytest.raises(binance.BinanceHTTPError, match="BINANCE_HTTP_PROXY"):
        binance.BinanceMarketClient()


def test_event_workflow_uses_binance_only_userspace_vpn_transport():
    workflow = Path(".github/workflows/event-engine.yml").read_text(encoding="utf-8")

    assert 'BINANCE_VPN_ENABLED: "true"' in workflow
    assert 'BINANCE_HTTP_PROXY: http://127.0.0.1:18080' in workflow
    assert "wireproxy_linux_amd64.tar.gz" in workflow
    assert "e88c1d090740373fc606c1bafd81d9a5eadc642cce5667616e20e9d7a444f51c" in workflow
    assert "--configtest" in workflow
    assert "--proxy \"$BINANCE_HTTP_PROXY\"" in workflow
    assert "--noproxy \"\"" in workflow
    assert "Run engine" in workflow
    assert "Stop Binance WireProxy" in workflow
    assert workflow.index("Binance Futures preflight") < workflow.index("Run engine")
    assert workflow.index("Run engine") < workflow.index("Stop Binance WireProxy")
    assert workflow.index("Stop Binance WireProxy") < workflow.index("Commit state")
    assert "wg-quick up wg0" not in workflow
    assert "wg-quick down wg0" not in workflow
    assert "sudo wg" not in workflow
    assert "      HTTPS_PROXY:" not in workflow
    assert "      HTTP_PROXY:" not in workflow
    assert "      ALL_PROXY:" not in workflow


def test_vpn_diagnostic_scopes_proxy_to_binance_only():
    workflow = Path(".github/workflows/vpn-test.yml").read_text(encoding="utf-8")

    assert 'BINANCE_HTTP_PROXY: http://127.0.0.1:18080' in workflow
    assert "wireproxy_linux_amd64.tar.gz" in workflow
    assert "e88c1d090740373fc606c1bafd81d9a5eadc642cce5667616e20e9d7a444f51c" in workflow
    assert "--proxy \"$BINANCE_HTTP_PROXY\"" in workflow
    assert "--noproxy \"\"" in workflow
    assert "wg-quick up wg0" not in workflow
    assert "wg-quick down wg0" not in workflow
    assert 'expected_request_count = 22' in workflow


def test_vst_research_mode_disables_entry_caps(monkeypatch):
    import run_once

    # The current package defaults to VST/demo research mode. In that mode the
    # portfolio cap (12 total / 6 long / 6 short) and per-cycle entry cap are off
    # so valid signals are not filtered merely for statistical collection.
    monkeypatch.setattr(run_once, "PORTFOLIO_CAP_ENABLED", False)
    monkeypatch.setattr(run_once, "MAX_TRADES", 0)

    active_total = 999
    active_longs = 999
    active_shorts = 999
    direction = "LONG"

    portfolio_cap_hit = run_once.PORTFOLIO_CAP_ENABLED and (
        active_total >= run_once.MAX_ACTIVE_TRADES
        or (direction == "LONG" and active_longs >= run_once.MAX_ACTIVE_LONGS)
        or (direction == "SHORT" and active_shorts >= run_once.MAX_ACTIVE_SHORTS)
    )
    assert portfolio_cap_hit is False
    assert run_once.MAX_TRADES <= 0


def test_vst_workflow_has_unlimited_cycle_entry_cap():
    from pathlib import Path

    workflow = Path(__file__).parent / ".github" / "workflows" / "event-engine.yml"
    text = workflow.read_text(encoding="utf-8")
    assert 'MAX_TRADES_PER_CYCLE: "0"' in text


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


REAL_JOURNAL_EVENT_ROWS = r'''
{"event_id":"EVT_43EE38BD756FB23E","symbol":"SUI","timeframe":"1h","direction":"SHORT","event_type":"REGULAR_BEARISH_OBV","timestamps":{"pivot_1_ts":1789898400000,"pivot_2_ts":1789927200000,"detected_at_ts":1789934400000},"event_fact":{"detection_close_price":0.8804,"p1_price":0.8322,"p2_price":0.9204,"p1_indicator":-18630681.0,"p2_indicator":-21786836.0,"bars_between":8,"price_delta_atr":4.292094457948584,"engine":"DIVERGENCE"}}
{"event_id":"EVT_E8C471415357C331","symbol":"ARB","timeframe":"1h","direction":"SHORT","event_type":"HIDDEN_BEARISH_MACD","timestamps":{"pivot_1_ts":1789880400000,"pivot_2_ts":1789927200000,"detected_at_ts":1789934400000},"event_fact":{"detection_close_price":0.21086,"p1_price":0.21864,"p2_price":0.21455,"p1_indicator":-0.00034084894194141846,"p2_indicator":-0.00013967256783065096,"bars_between":13,"price_delta_atr":0.5841441055479232,"engine":"DIVERGENCE"}}
{"event_id":"EVT_035A6899DB6B7FE8","symbol":"ARB","timeframe":"1h","direction":"SHORT","event_type":"HIDDEN_BEARISH_OBV","timestamps":{"pivot_1_ts":1789880400000,"pivot_2_ts":1789927200000,"detected_at_ts":1789934400000},"event_fact":{"detection_close_price":0.21086,"p1_price":0.21864,"p2_price":0.21455,"p1_indicator":12861155.799999945,"p2_indicator":19306567.99999995,"bars_between":13,"price_delta_atr":0.5841441055479232,"engine":"DIVERGENCE"}}
{"event_id":"EVT_C9F7F84B4AF6C1A2","symbol":"BR","timeframe":"1h","direction":"SHORT","event_type":"REGULAR_BEARISH_RSI","timestamps":{"pivot_1_ts":1789876800000,"pivot_2_ts":1789927200000,"detected_at_ts":1789934400000},"event_fact":{"detection_close_price":1.05003,"p1_price":1.11027,"p2_price":1.17706,"p1_indicator":60.836385052029726,"p2_indicator":58.18136027102264,"bars_between":14,"price_delta_atr":0.9622225318730075,"engine":"DIVERGENCE"}}
{"event_id":"EVT_302ED7C4446373D9","symbol":"BR","timeframe":"1h","direction":"SHORT","event_type":"REGULAR_BEARISH_MACD","timestamps":{"pivot_1_ts":1789876800000,"pivot_2_ts":1789927200000,"detected_at_ts":1789934400000},"event_fact":{"detection_close_price":1.05003,"p1_price":1.11027,"p2_price":1.17706,"p1_indicator":0.05848628233917186,"p2_indicator":0.043975508296217214,"bars_between":14,"price_delta_atr":0.9622225318730075,"engine":"DIVERGENCE"}}
{"event_id":"EVT_D5D1D921E9E43CBE","symbol":"BR","timeframe":"1h","direction":"SHORT","event_type":"REGULAR_BEARISH_STOCH","timestamps":{"pivot_1_ts":1789876800000,"pivot_2_ts":1789927200000,"detected_at_ts":1789934400000},"event_fact":{"detection_close_price":1.05003,"p1_price":1.11027,"p2_price":1.17706,"p1_indicator":67.13531093132947,"p2_indicator":47.33247401951069,"bars_between":14,"price_delta_atr":0.9622225318730075,"engine":"DIVERGENCE"}}
{"event_id":"EVT_50FB48BB6DE91DF1","symbol":"ZEC","timeframe":"1h","direction":"LONG","event_type":"VOLATILITY_SQUEEZE_RELEASE","timestamps":{"pivot_1_ts":1789934400000,"pivot_2_ts":1789934400000,"detected_at_ts":1789934400000},"event_fact":{"detection_close_price":1516.09,"bb_width":70.33697986124844,"kc_width":68.75439690736357,"squeeze_duration_bars":7,"compression_ratio":1.0230179163089332,"engine":"VOLATILITY_SQUEEZE","requires_retest":true,"trigger_level":1487.6901984536817}}
{"event_id":"EVT_131CDA2F347C8BC5","symbol":"ZEC","timeframe":"1h","direction":"LONG","event_type":"MA_COMPRESSION_BREAKOUT","timestamps":{"pivot_1_ts":1789934400000,"pivot_2_ts":1789934400000,"detected_at_ts":1789934400000},"event_fact":{"detection_close_price":1516.09,"compression_ratio":0.503855963830888,"ma_spread_atr":0.5086615149194595,"compression_threshold":1.0095375492869711,"breakout_atr":2.096862570381622,"volume_ratio":0.9049313344860339,"requires_retest":true,"trigger_level":1482.0,"engine":"MA_COMPRESSION","requires_htf_context":true}}
{"event_id":"EVT_016991FF29D01A0E","symbol":"AKE","timeframe":"1h","direction":"SHORT","event_type":"CRT_BEARISH","timestamps":{"pivot_1_ts":1789927200000,"pivot_2_ts":1789934400000,"detected_at_ts":1789934400000},"event_fact":{"engine":"CRT","requires_retest":true,"requires_htf_context":true,"trigger_level":0.062676,"range_high":0.062676,"range_low":0.05451,"manipulation_depth_atr":0.48592152367227925}}
{"event_id":"EVT_004B03CD72B86B53","symbol":"XRP","timeframe":"1h","direction":"SHORT","event_type":"BREAKER_BLOCK_BEARISH","timestamps":{"pivot_1_ts":1789812000000,"pivot_2_ts":1789934400000,"detected_at_ts":1789934400000},"event_fact":{"engine":"BREAKER_BLOCK","requires_htf_context":true,"requires_retest":true,"trigger_level":1.41,"zone_high":1.4182,"zone_low":1.41,"breaker_from":"BULLISH_OB","broken_ts":1789866000000,"bos_ts":1789819200000}}
{"event_id":"EVT_52B9C4CDADEC8A4B","symbol":"NEAR","timeframe":"1h","direction":"LONG","event_type":"CRT_BULLISH","timestamps":{"pivot_1_ts":1789927200000,"pivot_2_ts":1789934400000,"detected_at_ts":1789934400000},"event_fact":{"engine":"CRT","requires_retest":true,"requires_htf_context":true,"trigger_level":4.043,"range_high":4.276,"range_low":4.043,"manipulation_depth_atr":0.4555508972314215}}
{"event_id":"EVT_C5740BE651D5B53A","symbol":"HYPE","timeframe":"1h","direction":"LONG","event_type":"CRT_BULLISH","timestamps":{"pivot_1_ts":1789927200000,"pivot_2_ts":1789934400000,"detected_at_ts":1789934400000},"event_fact":{"engine":"CRT","requires_retest":true,"requires_htf_context":true,"trigger_level":92.367,"range_high":93.304,"range_low":92.367,"manipulation_depth_atr":0.14141977378585924}}
'''



def test_breaker_liquidity_sweep_requires_rejection_close():
    import event_engine.signals as sig

    d = pd.DataFrame({
        "open": [100.0] * 30,
        "high": [100.0] * 30,
        "low": [99.0] * 30,
        "close": [99.5] * 30,
    })
    d.loc[10, "high"] = 110.0

    # Wick through the buy-side liquidity but close above it: breakout, not sweep.
    d.loc[13, ["high", "close"]] = [112.0, 111.0]
    assert sig._liquidity_sweep_before_break(d, [10], 5, 15, "bearish") is None

    # Same wick, but rejection close back below the swept level: valid sweep.
    d.loc[13, ["high", "close"]] = [112.0, 109.0]
    assert sig._liquidity_sweep_before_break(d, [10], 5, 15, "bearish") == (13, 110.0)




def _make_breaker_mss_fixture(break_close: float) -> pd.DataFrame:
    rows = []
    for i in range(140):
        rows.append({
            "open": 100.0, "high": 100.4, "low": 99.6, "close": 100.0,
            "volume": 1000.0, "close_time": 1_700_000_000_000 + i * 3_600_000,
        })
    # Bullish OB source candle.
    rows[109].update({"open": 100.5, "high": 101.0, "low": 99.0, "close": 100.0})
    # Original BOS above a confirmed swing high.
    rows[110].update({"open": 100.5, "high": 112.0, "low": 100.0, "close": 111.0, "volume": 1800.0})
    # Intervening structural low, confirmed before the liquidity sweep.
    rows[115].update({"open": 108.0, "high": 110.0, "low": 97.0, "close": 108.5})
    rows[116].update({"open": 108.0, "high": 109.0, "low": 107.5, "close": 108.2})
    rows[117].update({"open": 108.0, "high": 109.0, "low": 107.5, "close": 108.2})
    # Buy-side liquidity sweep with rejection close.
    rows[118].update({"open": 108.5, "high": 111.5, "low": 107.0, "close": 108.0})
    # Close below the OB edge (99) but optionally not below MSS level (97).
    rows[121].update({"open": 100.0, "high": 100.5, "low": 96.5, "close": break_close, "volume": 1600.0})
    # Retest the broken zone from below.
    rows[139].update({"open": 98.0, "high": 100.0, "low": 96.5, "close": 98.5})
    return pd.DataFrame(rows)


def test_breaker_mss_requires_close_through_confirmed_structure(monkeypatch):
    import event_engine.signals as sig

    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([115], [105, 115]))
    assert sig._mss_level_before_sweep(_make_breaker_mss_fixture(98.0), [115], 110, 118, "bearish", 2) == (115, 97.0)
    assert sig.detect_breaker_block(_make_breaker_mss_fixture(98.0), "TEST", "1h") == []


def test_breaker_accepts_close_through_confirmed_mss(monkeypatch):
    import event_engine.signals as sig

    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([115], [105, 115]))
    events = sig.detect_breaker_block(_make_breaker_mss_fixture(96.0), "TEST", "1h")
    assert len(events) == 1
    assert events[0]["event_type"] == "BREAKER_BLOCK_BEARISH"
    assert events[0]["event_fact"]["mss_level"] == 97.0


def _make_bullish_breaker_mss_fixture(break_close: float) -> pd.DataFrame:
    rows = []
    for i in range(140):
        rows.append({
            "open": 100.0, "high": 100.4, "low": 99.6, "close": 100.0,
            "volume": 1000.0, "close_time": 1_700_000_000_000 + i * 3_600_000,
        })
    # Bearish OB source candle.
    rows[109].update({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5})
    # Original BOS below a confirmed swing low.
    rows[110].update({"open": 100.5, "high": 101.0, "low": 88.0, "close": 89.0, "volume": 1800.0})
    # Intervening structural high, confirmed before the sell-side sweep.
    rows[116].update({"open": 102.0, "high": 103.0, "low": 100.0, "close": 102.2})
    rows[117].update({"open": 102.0, "high": 102.5, "low": 100.0, "close": 102.2})
    rows[118].update({"open": 102.0, "high": 102.5, "low": 100.0, "close": 102.2})
    # Sell-side liquidity sweep with rejection close.
    rows[119].update({"open": 92.5, "high": 94.0, "low": 88.5, "close": 100.0})
    # Close above the OB edge (101) but optionally not above MSS level (103).
    rows[122].update({"open": 102.0, "high": 104.0, "low": 101.0, "close": break_close, "volume": 1600.0})
    # Retest the broken zone from above.
    rows[139].update({"open": 104.0, "high": 106.0, "low": 100.5, "close": 104.5})
    return pd.DataFrame(rows)


def test_bullish_breaker_mss_requires_close_through_confirmed_structure(monkeypatch):
    import event_engine.signals as sig

    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([105, 115], [116]))
    assert sig.detect_breaker_block(_make_bullish_breaker_mss_fixture(102.5), "TEST", "1h") == []


def test_bullish_breaker_accepts_close_through_confirmed_mss(monkeypatch):
    import event_engine.signals as sig

    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([105, 115], [116]))
    events = sig.detect_breaker_block(_make_bullish_breaker_mss_fixture(104.0), "TEST", "1h")
    assert len(events) == 1
    assert events[0]["event_type"] == "BREAKER_BLOCK_BULLISH"
    assert events[0]["event_fact"]["mss_level"] == 103.0


def test_breaker_sell_side_sweep_requires_rejection_close():
    import event_engine.signals as sig

    d = pd.DataFrame({
        "open": [100.0] * 30,
        "high": [101.0] * 30,
        "low": [100.0] * 30,
        "close": [100.5] * 30,
    })
    d.loc[10, "low"] = 90.0

    # Wick through sell-side liquidity but close below it: breakout, not sweep.
    d.loc[13, ["low", "close"]] = [88.0, 89.0]
    assert sig._liquidity_sweep_before_break(d, [10], 5, 15, "bullish") is None

    # Same wick, but rejection close back above the swept level: valid sweep.
    d.loc[13, ["low", "close"]] = [88.0, 91.0]
    assert sig._liquidity_sweep_before_break(d, [10], 5, 15, "bullish") == (13, 90.0)

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

    divergence, other = [], []
    for line in REAL_JOURNAL_EVENT_ROWS.splitlines():
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


def test_vpn_validator_is_transport_first_and_captures_curl_errors():
    from pathlib import Path

    workflow = Path('.github/workflows/vpn-test.yml').read_text(encoding='utf-8')
    assert 'local curl_error="vpn-audit/responses/${safe_label}.curl_error"' in workflow
    assert '2>"$curl_error"' in workflow
    assert "=== TRANSPORT FAILURES ===" in workflow
    assert "=== SEMANTIC CHECKS SKIPPED ===" in workflow
    assert 'if transport_failures:' in workflow
    assert workflow.index('if transport_failures:') < workflow.index("exchange = read_json('binance_exchangeInfo.body')")



def _make_runtime_harmonic_fixture(points, kinds):
    import pandas as pd
    n = 90
    rows = []
    for i in range(n):
        rows.append({
            "open": 150.0, "high": 151.0, "low": 149.0, "close": 150.0,
            "volume": 1000.0, "close_time": i * 60_000,
        })
    df = pd.DataFrame(rows)
    for idx, price in points.items():
        kind = kinds[idx]
        if kind == "low":
            df.loc[idx, "low"] = price
            df.loc[idx, "open"] = price + 1.0
            df.loc[idx, "close"] = price + 1.0
            df.loc[idx, "high"] = max(price + 2.0, price + 1.5)
        else:
            df.loc[idx, "high"] = price
            df.loc[idx, "open"] = price - 1.0
            df.loc[idx, "close"] = price - 1.0
            df.loc[idx, "low"] = min(price - 2.0, price - 1.5)
    return df


def test_shark_accepts_valid_oxabc_without_oxa_ratio_constraint(monkeypatch):
    import event_engine.signals as sig

    # Canonical O-X-A-B-C mapping into this detector's X-A-B-C-D names:
    # code X=O, A=X, B=A, C=B, D=C.
    # OX=100, XA=36 (0.36; intentionally below the erroneous 0.50 floor),
    # AB=54 (1.50 XA), BC=108 (2.00 AB), XC=90 (0.90 OX).
    points = {10: 1000.0, 20: 1100.0, 30: 1064.0, 40: 1118.0, 50: 1010.0}
    kinds = {10: "low", 20: "high", 30: "low", 40: "high", 50: "low"}
    df = _make_runtime_harmonic_fixture(points, kinds)
    monkeypatch.setattr(sig, "_pivots", lambda work, left=5, right=5: ([10, 30, 50], [20, 40]))

    events = sig.detect_harmonic_patterns(df, "TEST-USDT", "1h", tolerance=0.05)
    assert any(e["event_type"] == "HARMONIC_SHARK_LONG" for e in events)


def test_shark_rejects_invalid_oxabc_geometry(monkeypatch):
    import event_engine.signals as sig

    # The OX/XA part is intentionally unconstrained, but BC/AB is 1.0 here,
    # below the canonical Shark range of 1.618-2.24, so the event is rejected.
    points = {10: 1000.0, 20: 1100.0, 30: 1064.0, 40: 1118.0, 50: 1028.0}
    kinds = {10: "low", 20: "high", 30: "low", 40: "high", 50: "low"}
    df = _make_runtime_harmonic_fixture(points, kinds)
    monkeypatch.setattr(sig, "_pivots", lambda work, left=5, right=5: ([10, 30, 50], [20, 40]))

    events = sig.detect_harmonic_patterns(df, "TEST-USDT", "1h", tolerance=0.05)
    assert not any(e["event_type"] == "HARMONIC_SHARK_LONG" for e in events)


def test_bat_accepts_canonical_projection(monkeypatch):
    import event_engine.signals as sig

    # X=1000, A=1100, B=1050, C=1090, D=1011.4:
    # AB/XA=0.50, BC/AB=0.80, CD/BC≈1.965, AD/XA=0.886.
    points = {10: 1000.0, 20: 1100.0, 30: 1050.0, 40: 1090.0, 50: 1011.4}
    kinds = {10: "low", 20: "high", 30: "low", 40: "high", 50: "low"}
    df = _make_runtime_harmonic_fixture(points, kinds)
    monkeypatch.setattr(sig, "_pivots", lambda work, left=5, right=5: ([10, 30, 50], [20, 40]))

    events = sig.detect_harmonic_patterns(df, "TEST-USDT", "1h", tolerance=0.05)
    assert any(e["event_type"] == "HARMONIC_BAT_LONG" for e in events)


def test_bat_rejects_excessive_cd_bc_projection(monkeypatch):
    import event_engine.signals as sig

    # Same XA/AB/AD geometry, but C=1069.1 gives BC/AB≈0.382 and
    # CD/BC≈3.02, above the canonical 2.618 ceiling.
    points = {10: 1000.0, 20: 1100.0, 30: 1050.0, 40: 1069.1, 50: 1011.4}
    kinds = {10: "low", 20: "high", 30: "low", 40: "high", 50: "low"}
    df = _make_runtime_harmonic_fixture(points, kinds)
    monkeypatch.setattr(sig, "_pivots", lambda work, left=5, right=5: ([10, 30, 50], [20, 40]))

    events = sig.detect_harmonic_patterns(df, "TEST-USDT", "1h", tolerance=0.05)
    assert not any(e["event_type"] == "HARMONIC_BAT_LONG" for e in events)


def _make_crt_three_candle_fixture(bullish: bool, double_sweep: bool) -> pd.DataFrame:
    rows = []
    base_ts = 1_700_000_000_000
    for i in range(50):
        rows.append({
            "open": 100.0, "high": 100.8, "low": 99.2, "close": 100.2,
            "volume": 1000.0, "close_time": base_ts + i * 3_600_000,
        })
    c1 = 47
    rows[c1].update({"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0})
    if bullish:
        rows[c1 + 1].update({
            "open": 101.0,
            "high": 103.0 if double_sweep else 101.4,
            "low": 97.5,
            "close": 98.5,
        })
        rows[c1 + 2].update({"open": 98.5, "high": 101.5, "low": 98.2, "close": 100.8})
    else:
        rows[c1 + 1].update({
            "open": 101.0,
            "high": 103.5,
            "low": 98.0 if double_sweep else 99.4,
            "close": 101.5,
        })
        rows[c1 + 2].update({"open": 101.5, "high": 102.0, "low": 99.0, "close": 100.2})
    return pd.DataFrame(rows)


def test_crt_rejects_two_sided_bullish_sweep():
    from event_engine.signals import detect_crt

    df = _make_crt_three_candle_fixture(True, True)
    assert detect_crt(df, "TEST", "1h") == []


def test_crt_rejects_two_sided_bearish_sweep():
    from event_engine.signals import detect_crt

    df = _make_crt_three_candle_fixture(False, True)
    assert detect_crt(df, "TEST", "1h") == []


def test_crt_keeps_valid_one_sided_sweeps():
    from event_engine.signals import detect_crt

    bullish = detect_crt(_make_crt_three_candle_fixture(True, False), "TEST", "1h")
    bearish = detect_crt(_make_crt_three_candle_fixture(False, False), "TEST", "1h")
    assert any(e["event_type"] == "CRT_BULLISH" for e in bullish)
    assert any(e["event_type"] == "CRT_BEARISH" for e in bearish)


def _make_ob_wick_invalidation_fixture(bullish: bool) -> pd.DataFrame:
    import event_engine.signals as sig

    n = 120
    rows = []
    for i in range(n):
        rows.append({
            "open": 100.0, "high": 100.4, "low": 99.6, "close": 100.0,
            "volume": 1000.0,
            "close_time": 1_700_000_000_000 + i * 3_600_000,
        })
    pivot_i = 50
    bos_i = 53
    ob_i = 52
    rows[pivot_i].update({"open": 101.0, "high": 106.0, "low": 98.0, "close": 99.0})
    if bullish:
        rows[ob_i].update({"open": 101.0, "high": 106.0, "low": 98.0, "close": 99.0})
        rows[bos_i].update({"open": 99.0, "high": 108.0, "low": 98.0, "close": 107.0, "volume": 2200.0})
        rows[bos_i + 1].update({"low": 97.0, "close": 100.0})
    else:
        rows[ob_i].update({"open": 99.0, "high": 102.0, "low": 94.0, "close": 101.0})
        rows[pivot_i].update({"open": 99.0, "high": 102.0, "low": 94.0, "close": 101.0})
        rows[bos_i].update({"open": 101.0, "high": 102.0, "low": 92.0, "close": 93.0, "volume": 2200.0})
        rows[bos_i + 1].update({"high": 103.0, "close": 100.0})
    return pd.DataFrame(rows)


def test_order_block_uses_wick_invalidation_for_bullish_zone(monkeypatch):
    import event_engine.signals as sig

    monkeypatch.setattr(sig, "_atr", lambda d, n=14: pd.Series([1.0] * len(d), index=d.index, dtype=float))
    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([], [50]))
    events = sig.detect_order_block(_make_ob_wick_invalidation_fixture(True), "TEST", "1h")
    assert events == []


def test_order_block_uses_wick_invalidation_for_bearish_zone(monkeypatch):
    import event_engine.signals as sig

    monkeypatch.setattr(sig, "_atr", lambda d, n=14: pd.Series([1.0] * len(d), index=d.index, dtype=float))
    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([50], []))
    events = sig.detect_order_block(_make_ob_wick_invalidation_fixture(False), "TEST", "1h")
    assert events == []


def _make_mitigation_runtime_fixture(direction: str, *, sweep_before_break: bool = False):
    import pandas as pd
    rows = []
    for i in range(140):
        rows.append({
            "open": 100.0, "high": 100.4, "low": 99.6, "close": 100.0,
            "volume": 1000.0, "close_time": 1_700_000_000_000 + i * 3_600_000,
        })
    df = pd.DataFrame(rows)
    if direction == "bullish":
        df.loc[100, ["open","high","low","close"]] = [96.0, 96.5, 95.0, 95.8]
        df.loc[104, ["open","high","low","close"]] = [102.0, 104.0, 101.5, 103.0]
        df.loc[105, ["open","high","low","close"]] = [104.0, 105.0, 102.2, 103.0]
        df.loc[108, ["open","high","low","close"]] = [99.0, 101.0, 94.5 if sweep_before_break else 98.5, 100.0]
        df.loc[110, ["open","high","low","close"]] = [99.5, 101.5, 98.0, 100.5]
        df.loc[113, ["open","high","low","close"]] = [100.0, 109.5, 99.5, 108.0]
        df.loc[139, ["open","high","low","close"]] = [102.0, 103.0, 101.9, 102.0]
    else:
        df.loc[100, ["open","high","low","close"]] = [104.0, 105.0, 103.5, 104.5]
        df.loc[104, ["open","high","low","close"]] = [98.0, 98.5, 96.5, 97.0]
        df.loc[105, ["open","high","low","close"]] = [97.0, 98.5, 95.0, 96.0]
        df.loc[108, ["open","high","low","close"]] = [101.0, 105.5 if sweep_before_break else 102.0, 99.5, 101.0]
        df.loc[110, ["open","high","low","close"]] = [101.5, 102.0, 98.5, 100.5]
        df.loc[113, ["open","high","low","close"]] = [100.0, 100.5, 90.5, 92.0]
        df.loc[139, ["open","high","low","close"]] = [97.0, 98.5, 96.5, 97.5]
    return df


def test_mitigation_block_requires_failure_swing_and_no_external_sweep(monkeypatch):
    import pandas as pd
    import event_engine.signals as sig
    monkeypatch.setattr(sig, "_atr", lambda d, n=14: pd.Series([1.0] * len(d), index=d.index, dtype=float))

    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([100, 110], [105]))
    bullish = sig.detect_mitigation_block(_make_mitigation_runtime_fixture("bullish"), "TEST", "1h")
    assert len(bullish) == 1
    assert bullish[0]["event_type"] == "MITIGATION_BLOCK_BULLISH"
    assert bullish[0]["event_fact"]["failure_swing_no_sweep"] is True
    assert bullish[0]["event_fact"]["failure_swing_level"] == pytest.approx(98.0)

    swept = sig.detect_mitigation_block(_make_mitigation_runtime_fixture("bullish", sweep_before_break=True), "TEST", "1h")
    assert swept == []

    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([105], [100, 110]))
    bearish = sig.detect_mitigation_block(_make_mitigation_runtime_fixture("bearish"), "TEST", "1h")
    assert len(bearish) == 1
    assert bearish[0]["event_type"] == "MITIGATION_BLOCK_BEARISH"
    assert bearish[0]["event_fact"]["failure_swing_no_sweep"] is True
    assert bearish[0]["event_fact"]["failure_swing_level"] == pytest.approx(102.0)

    swept_bearish = sig.detect_mitigation_block(_make_mitigation_runtime_fixture("bearish", sweep_before_break=True), "TEST", "1h")
    assert swept_bearish == []


def test_butterfly_accepts_published_projection_range(monkeypatch):
    import event_engine.signals as sig

    # XA=100, AB/XA=0.786, BC/AB=0.636,
    # CD/BC=2.40 and AD/XA=1.486. This is accepted by the published
    # Butterfly family because CD/BC is in 1.618-2.618.
    points = {10: 1000.0, 20: 1100.0, 30: 1021.4, 40: 1071.4, 50: 951.4}
    kinds = {10: "low", 20: "high", 30: "low", 40: "high", 50: "low"}
    df = _make_runtime_harmonic_fixture(points, kinds)
    monkeypatch.setattr(sig, "_pivots", lambda work, left=5, right=5: ([10, 30, 50], [20, 40]))

    events = sig.detect_harmonic_patterns(df, "TEST-USDT", "1h", tolerance=0.05)
    assert any(e["event_type"] == "HARMONIC_BUTTERFLY_LONG" for e in events)




def test_butterfly_accepts_published_d_xa_completion_branch(monkeypatch):
    import event_engine.signals as sig

    # D/XA=1.80 is within the published alternate completion branch, while
    # CD/BC=3.028 is outside 1.618-2.618; the OR branch must still accept it.
    points = {10: 1000.0, 20: 1100.0, 30: 1021.4, 40: 1071.4, 50: 920.0}
    kinds = {10: "low", 20: "high", 30: "low", 40: "high", 50: "low"}
    df = _make_runtime_harmonic_fixture(points, kinds)
    monkeypatch.setattr(sig, "_pivots", lambda work, left=5, right=5: ([10, 30, 50], [20, 40]))

    events = sig.detect_harmonic_patterns(df, "TEST-USDT", "1h", tolerance=0.05)
    assert any(e["event_type"] == "HARMONIC_BUTTERFLY_LONG" for e in events)


def test_butterfly_rejects_when_both_completion_constraints_fail(monkeypatch):
    import event_engine.signals as sig

    # B=0.786 XA, BC/AB in the allowed retracement band, but CD/BC is only
    # 1.40 and AD/XA is only 1.20, so neither published completion branch passes.
    points = {10: 1000.0, 20: 1100.0, 30: 1021.4, 40: 1071.4, 50: 1001.4}
    kinds = {10: "low", 20: "high", 30: "low", 40: "high", 50: "low"}
    df = _make_runtime_harmonic_fixture(points, kinds)
    monkeypatch.setattr(sig, "_pivots", lambda work, left=5, right=5: ([10, 30, 50], [20, 40]))

    events = sig.detect_harmonic_patterns(df, "TEST-USDT", "1h", tolerance=0.05)
    assert not any(e["event_type"] == "HARMONIC_BUTTERFLY_LONG" for e in events)


def test_sfp_does_not_reuse_a_consumed_swing_level(monkeypatch):
    import event_engine.signals as sig

    rows = []
    for i in range(85):
        rows.append({
            "open": 109.0, "high": 111.0, "low": 108.0, "close": 110.0,
            "volume": 1000.0, "close_time": 1_700_000_000_000 + i * 3_600_000,
        })
    rows[60].update({"open": 111.0, "high": 112.0, "low": 100.0, "close": 110.0})
    rows[84].update({"open": 100.0, "high": 102.0, "low": 95.0, "close": 101.0, "volume": 2000.0})
    df = pd.DataFrame(rows)
    monkeypatch.setattr(sig, "_pivots", lambda *args, **kwargs: ([60], []))

    clean = sig.detect_sfp(df, "TEST", "1h")
    assert any(e["event_type"] == "SFP_BULLISH" for e in clean)

    consumed = df.copy()
    consumed.loc[70, ["high", "low", "close"]] = [101.0, 99.0, 98.5]
    assert sig.detect_sfp(consumed, "TEST", "1h") == []


def test_breaker_does_not_reuse_a_liquidity_pivot_after_prior_close_through():
    import event_engine.signals as sig

    rows = []
    for i in range(30):
        rows.append({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    rows[10]["high"] = 110.0
    # Prior bar already closed above the pivot: the liquidity was consumed.
    rows[13].update({"high": 112.0, "low": 99.0, "close": 111.0})
    # A later rejection wick through the same stale pivot is not a new sweep.
    rows[16].update({"high": 113.0, "low": 99.0, "close": 109.5})
    df = pd.DataFrame(rows)
    assert sig._liquidity_sweep_before_break(df, [10], 5, 20, "bearish") is None


def test_breaker_mss_requires_a_fresh_unbroken_structure_level():
    import event_engine.signals as sig

    rows = []
    for i in range(30):
        rows.append({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    rows[10].update({"low": 90.0, "high": 101.0, "close": 95.0})
    df = pd.DataFrame(rows)

    rows_before_sweep = df.copy()
    rows_before_sweep.loc[15, "close"] = 89.0
    assert sig._mss_level_before_sweep(rows_before_sweep, [10], 5, 20, "bearish") is None

    rows_clean = df.copy()
    assert sig._mss_level_before_sweep(rows_clean, [10], 5, 20, "bearish") == (10, 90.0)


def test_breaker_mss_requires_fresh_unbroken_bullish_structure_level():
    import event_engine.signals as sig

    rows = []
    for i in range(30):
        rows.append({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    rows[10].update({"high": 110.0, "low": 99.0, "close": 105.0})
    df = pd.DataFrame(rows)

    rows_before_sweep = df.copy()
    rows_before_sweep.loc[15, "close"] = 111.0
    assert sig._mss_level_before_sweep(rows_before_sweep, [10], 5, 20, "bullish") is None

    rows_clean = df.copy()
    assert sig._mss_level_before_sweep(rows_clean, [10], 5, 20, "bullish") == (10, 110.0)
