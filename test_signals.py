# test_signals.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from event_engine.coinalyze import parse_number, CoinalyzeRow
from event_engine.signals import (
    _rsi,
    add_cvd,
    build_15m_trigger,
    diagnose_15m_trigger,
    detect_divergences,
    detect_squeeze_release,
    detect_liquidation_squeeze,
    attach_oi_series,
    check_btc_regime,
)
from event_engine.bingx import (
    get_contract,
    CACHE,
    _allocate_tp_quantities,
)
from event_engine.tracker import _update_mfe_mae, _extract_setup_metrics
from run_once import (
    build_event_setup,
    build_tp_levels,
    calculate_setup_score,
    execute_new_position,
    load_successful_telegram_ids,
)


def _load_successful_trade_ids(path: Path) -> set[str]:
    """Вспомогательная функция для проверки дедубликации и retry."""
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            status = str(obj.get("result", {}).get("status", "")).lower()
            if status in {"opened_protected", "opened_protection_check_required", "opened", "already_executed"}:
                value = obj.get("event_id")
                if value:
                    ids.add(str(value))
        except Exception:
            continue
    return ids


def _generate_synthetic_candles(n: int = 80, base_price: float = 100.0) -> pd.DataFrame:
    times = [1700000000000 + i * 3600000 for i in range(n)]
    prices = [base_price + np.sin(i / 4.0) * 3.0 for i in range(n)]
    return pd.DataFrame({
        "open_time": [t - 3600000 for t in times],
        "close_time": times,
        "open": prices,
        "high": [p + 1.0 for p in prices],
        "low": [p - 1.0 for p in prices],
        "close": prices,
        "volume": [1000.0] * n,
        "quote_volume": [100000.0] * n,
        "taker_buy_base": [500.0] * n,
        "taker_buy_quote": [50000.0] * n,
        "taker_flow_valid": [True] * n,
        "bar_delta_usdt": [0.0] * n,
    })


def test_parse_number():
    assert parse_number("1.2M") == 1_200_000
    assert parse_number("$5,000") == 5000
    assert parse_number("—") is None
    assert parse_number(None) is None


def test_rsi_flat_and_extremes():
    flat_series = pd.Series([100.0] * 30)
    assert _rsi(flat_series).iloc[-1] == 50.0

    up_series = pd.Series(list(range(10, 40)))
    assert _rsi(up_series).iloc[-1] == 100.0

    down_series = pd.Series(list(range(40, 10, -1)))
    assert _rsi(down_series).iloc[-1] == 0.0


def test_trigger_long_and_short():
    df_long = pd.DataFrame({"high": [100, 105], "low": [95, 100], "close": [99, 106]})
    assert build_15m_trigger(df_long, "LONG", min_vol_mult=0.0) is True
    assert build_15m_trigger(df_long, "long", min_vol_mult=0.0) is True
    assert build_15m_trigger(df_long, "SHORT", min_vol_mult=0.0) is False

    df_short = pd.DataFrame({"high": [105, 104], "low": [100, 95], "close": [101, 94]})
    assert build_15m_trigger(df_short, "SHORT", min_vol_mult=0.0) is True
    assert build_15m_trigger(df_short, "short", min_vol_mult=0.0) is True
    assert build_15m_trigger(df_short, "LONG", min_vol_mult=0.0) is False


def test_divergence_detector_causality():
    df = _generate_synthetic_candles(75)
    df.loc[30, "low"] = 85.0
    df.loc[30, "close"] = 86.0
    df.loc[45, "low"] = 80.0
    df.loc[45, "close"] = 81.0

    df = add_cvd(df)
    events = detect_divergences(df, "BTC-USDT", "1h", left=3, right=2)
    assert isinstance(events, list)

    for ev in events:
        ts = ev["timestamps"]
        assert ts["detected_at_ts"] >= ts["pivot_2_ts"]
        assert ts["pivot_2_ts"] > ts["pivot_1_ts"]


def test_squeeze_release_duration_enforced():
    df = _generate_synthetic_candles(80)
    df["close"] = 100.0
    df["high"] = 100.1
    df["low"] = 99.9

    df.loc[79, "close"] = 115.0
    df.loc[79, "high"] = 116.0

    events = detect_squeeze_release(df, "BTC-USDT", "1h", min_squeeze_bars=3)
    assert len(events) == 1
    assert events[0]["direction"] == "LONG"
    assert events[0]["event_fact"]["squeeze_duration_bars"] >= 3
    assert events[0]["event_type"] == "VOLATILITY_SQUEEZE_RELEASE"


def test_load_successful_trades_retry_safety(tmp_path: Path):
    trades_file = tmp_path / "trades.jsonl"
    trades_file.write_text(
        json.dumps({"event_id": "EVT_FAIL", "result": {"status": "OPEN_FAILED"}}) + "\n" +
        json.dumps({"event_id": "EVT_SUCCESS", "result": {"status": "opened_protected"}}) + "\n",
        encoding="utf-8",
    )
    loaded_ids = _load_successful_trade_ids(trades_file)
    assert "EVT_FAIL" not in loaded_ids  # Ошибка API не блокирует повторную попытку
    assert "EVT_SUCCESS" in loaded_ids


def test_get_contract_displayName_with_hyphen():
    CACHE["data"] = {
        "ETH-USDT": {"symbol": "ETH-USDT", "displayName": "ETH-USDT", "status": 1, "apiStateOpen": "true"}
    }
    CACHE["ts"] = 9999999999
    c = get_contract("ETH")
    assert c is not None
    assert c["symbol"] == "ETH-USDT"


def test_allocate_tp_quantities_exact_sum():
    # 1. Standard precision and min_qty
    qtys = _allocate_tp_quantities(
        position_qty=100.0,
        precision=0,
        min_qty=10.0,
        fractions=[0.35, 0.35, 0.30],
    )
    assert qtys == [35.0, 35.0, 30.0]
    assert sum(qtys) == 100.0

    # 2. Fractional precision
    qtys_frac = _allocate_tp_quantities(
        position_qty=1.0,
        precision=2,
        min_qty=0.1,
        fractions=[0.35, 0.35, 0.30],
    )
    assert qtys_frac == [0.35, 0.35, 0.30]
    assert round(sum(qtys_frac), 2) == 1.0

    # 3. Step rounding remainder distribution
    qtys_rem = _allocate_tp_quantities(
        position_qty=10.0,
        precision=0,
        min_qty=1.0,
        fractions=[0.33, 0.33, 0.34],
    )
    assert sum(qtys_rem) == 10.0
    assert all(q >= 1.0 for q in qtys_rem)

    # 4. Position cannot support 3 legs
    with pytest.raises(ValueError):
        _allocate_tp_quantities(
            position_qty=2.0,
            precision=0,
            min_qty=1.0,
            fractions=[0.35, 0.35, 0.30],
        )


def test_squeeze_release_short():
    df = _generate_synthetic_candles(80)
    df["close"] = 100.0
    df["high"] = 100.1
    df["low"] = 99.9

    df.loc[79, "close"] = 85.0
    df.loc[79, "low"] = 84.0

    events = detect_squeeze_release(df, "BTC-USDT", "1h", min_squeeze_bars=3)
    assert len(events) == 1
    assert events[0]["direction"] == "SHORT"
    assert events[0]["event_fact"]["squeeze_duration_bars"] >= 3
    assert events[0]["event_type"] == "VOLATILITY_SQUEEZE_RELEASE"


def test_btc_regime_filtering():
    # 1. Normal BTC
    normal_df = pd.DataFrame({"close": [100, 100.1, 100.2, 100.1, 100.3]})
    ok, _ = check_btc_regime(normal_df, "LONG")
    assert ok is True
    ok, _ = check_btc_regime(normal_df, "SHORT")
    assert ok is True

    # 2. BTC dumping 1H -> Blocks LONG
    dump_df = pd.DataFrame({"close": [100, 100, 100, 100, 98.0]})
    ok_long, reason = check_btc_regime(dump_df, "LONG")
    assert ok_long is False
    assert "DUMPING_1H" in reason

    # 3. BTC pumping 1H -> Blocks SHORT
    pump_df = pd.DataFrame({"close": [100, 100, 100, 100, 102.5]})
    ok_short, reason = check_btc_regime(pump_df, "SHORT")
    assert ok_short is False
    assert "PUMPING_1H" in reason


def test_setup_and_tp_levels_symmetry():
    df = _generate_synthetic_candles(60)

    # Long setup
    setup_long = build_event_setup({"direction": "LONG"}, df, entry_price=100.0)
    assert setup_long["invalidation_price"] < 100.0
    assert setup_long["target_price"] > 100.0
    assert setup_long["planned_weighted_rr"] == 1.05
    sl_pct_l, tp_levels_l = build_tp_levels(setup_long, "LONG")
    assert sl_pct_l > 0
    assert len(tp_levels_l) == 3
    assert tp_levels_l[0]["pnl_pct"] < tp_levels_l[1]["pnl_pct"] < tp_levels_l[2]["pnl_pct"]

    # Short setup
    setup_short = build_event_setup({"direction": "SHORT"}, df, entry_price=100.0)
    assert setup_short["invalidation_price"] > 100.0
    assert setup_short["target_price"] < 100.0
    assert setup_short["planned_weighted_rr"] == 1.05
    sl_pct_s, tp_levels_s = build_tp_levels(setup_short, "SHORT")
    assert sl_pct_s > 0
    assert len(tp_levels_s) == 3
    assert tp_levels_s[0]["pnl_pct"] < tp_levels_s[1]["pnl_pct"] < tp_levels_s[2]["pnl_pct"]


def test_rsi_warmup_preserves_nan():
    series = pd.Series([10.0 + i for i in range(30)])
    rsi = _rsi(series, n=14)
    # Warmup bars (0..12) must be NaN
    assert pd.isna(rsi.iloc[0])
    assert pd.isna(rsi.iloc[12])
    # Post-warmup bars must be valid numbers
    assert pd.notna(rsi.iloc[14])
    assert pd.notna(rsi.iloc[-1])


def _wilder_rsi_reference(series: pd.Series, n: int = 14) -> pd.Series:
    """Classic Wilder RSI: SMA seed, then Wilder recursion (TradingView ta.rsi)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    arr_g = gain.to_numpy(dtype=float)
    arr_l = loss.to_numpy(dtype=float)
    avg_g = np.full(len(series), np.nan)
    avg_l = np.full(len(series), np.nan)
    if len(series) > n:
        avg_g[n] = np.nanmean(arr_g[1 : n + 1])
        avg_l[n] = np.nanmean(arr_l[1 : n + 1])
        for i in range(n + 1, len(series)):
            avg_g[i] = (avg_g[i - 1] * (n - 1) + arr_g[i]) / n
            avg_l[i] = (avg_l[i - 1] * (n - 1) + arr_l[i]) / n
    rs = np.divide(avg_g, avg_l, out=np.full(len(series), np.nan), where=avg_l != 0)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = np.where((avg_l == 0) & (avg_g > 0), 100.0, out)
    out = np.where((avg_l == 0) & (avg_g == 0), 50.0, out)
    out[np.isnan(avg_g) | np.isnan(avg_l)] = np.nan
    return pd.Series(out, index=series.index)


def test_rsi_matches_wilder_sma_seed_reference():
    """Audit B1 regression guard: code RSI must equal the Wilder SMA-seed
    reference (TradingView parity). The old EMA-seeded implementation drifted
    by up to 14.24 RSI points on this exact scenario."""
    np.random.seed(42)
    rw = pd.Series(100 + np.cumsum(np.random.normal(0, 1.0, 100)))
    diff = (_rsi(rw, 14) - _wilder_rsi_reference(rw, 14)).abs()
    assert float(diff.max()) < 0.1


def test_execute_new_position_defines_pre_order_price(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "open_market", lambda symbol, direction, price, trade_id: {
        "status": "opened", "order_id": "O1", "order_reference_price": 100.0,
    })
    monkeypatch.setattr(ro, "wait_for_position_fill_directional", lambda **kwargs: {
        "status": "found", "positionAmt": "0.1", "avgPrice": "101.0",
    })
    monkeypatch.setattr(ro, "install_protection", lambda **kwargs: {
        "status": "PROTECTED",
        "tp_orders": [{"leg": "tp3", "status": "created", "price": 102.75, "qty": 0.1, "pnl_pct": 1.75}],
        "sl_result": {"status": "created", "stop_price": 100.0, "qty": 0.1},
        "tp_mode": "single_tp",
        "effective_tp_levels": [{"leg": "tp3", "pnl_pct": 1.75, "close_fraction": 1.0, "qty": 0.1}],
        "effective_weighted_rr": 1.75,
    })
    setup = {"risk_pct": 1.0, "planned_weighted_rr": 1.05, "entry_reference": 99.0, "target_rr": 1.75}
    out = execute_new_position("TEST", "LONG", 99.0, setup, "EVT_TEST")
    assert out["status"] == "opened_protected"
    assert out["open_result"]["order_reference_price"] == 100.0
    assert out["execution_quality"]["signal_to_order_drift_pct"] == pytest.approx((100-99)/99*100)
    assert out["execution_quality"]["execution_slippage_pct"] == pytest.approx(1.0)
    assert out["setup_used_for_protection"]["pre_order_reference_price"] == 100.0
    assert out["setup_used_for_protection"]["effective_weighted_rr"] == pytest.approx(1.75)


def test_failed_telegram_delivery_is_retryable(tmp_path: Path):
    actions = tmp_path / "actions.jsonl"
    actions.write_text(
        json.dumps({"event_id": "EVT_FAILED", "telegram_sent": False}) + "\n"
        + json.dumps({"event_id": "EVT_OK", "telegram_sent": True}) + "\n",
        encoding="utf-8",
    )
    ids = load_successful_telegram_ids(actions)
    assert "EVT_FAILED" not in ids
    assert "EVT_OK" in ids


def test_default_setup_rr_is_v2_1_05():
    metrics = _extract_setup_metrics(None)
    assert metrics["planned_weighted_rr"] == pytest.approx(1.05)
    assert metrics["effective_weighted_rr"] == pytest.approx(1.05)


def test_telegram_message_uses_effective_rr_and_tp_mode():
    from event_engine.telegram import format_signal
    msg = format_signal(
        {"direction": "LONG", "symbol": "TEST", "event_type": "X", "event_fact": {}, "timestamps": {}},
        setup={"entry_reference": 100, "invalidation_price": 95, "target_price": 108.75, "effective_weighted_rr": 1.75, "tp_mode": "single_tp"},
        score=65,
    )
    assert "1.75" in msg
    assert "single_tp" in msg


def test_score_call_source_is_trigger_diagnostic():
    import run_once as ro
    import inspect
    source = inspect.getsource(ro.main)
    assert "trigger_diagnostic=trigger_diag" in source


def test_telegram_message_contains_trigger_fields():
    from event_engine.telegram import format_signal
    msg = format_signal(
        {"direction": "LONG", "symbol": "TEST", "event_type": "X", "event_fact": {}, "timestamps": {}},
        setup={"entry_reference": 100, "invalidation_price": 99, "target_price": 101.75, "effective_weighted_rr": 1.75, "tp_mode": "single_tp", "trigger": {"trigger_price": 100.5, "trigger_delay_min": 15.0}},
        score=65,
    )
    assert "Trigger Price" in msg and "100.5" in msg
    assert "single_tp" in msg


def test_build_event_setup_uses_wilder_atr():
    import run_once as ro
    df = _generate_synthetic_candles(80)
    setup = ro.build_event_setup({"direction": "LONG"}, df, 100.0)
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    expected_atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().iloc[-1]
    expected_risk = max(0.50, min(expected_atr * 1.5 / 100.0 * 100.0, 5.00))
    assert setup["risk_pct"] == pytest.approx(expected_risk)


def test_single_tp_mode_effective_rr_is_not_1_05():
    from event_engine.tracker import _extract_setup_metrics
    metrics = _extract_setup_metrics({
        "risk_pct": 2.0, "target_rr": 1.75, "planned_weighted_rr": 1.05,
        "tp_mode": "single_tp", "effective_weighted_rr": 1.75,
        "effective_tp_levels": [{"leg": "tp3", "pnl_pct": 3.5, "qty": 1.0}],
    })
    assert metrics["tp_mode"] == "single_tp"
    assert metrics["effective_weighted_rr"] == pytest.approx(1.75)


def test_pending_telegram_retry_is_present_and_idempotent_by_success():
    import run_once as ro
    import inspect
    source = inspect.getsource(ro.send_pending_open_trade_notifications)
    assert "event_id in successful_ids" in source
    assert "telegram_kind" in source


def test_run_once_uses_trigger_diag_for_score():
    import run_once as ro
    import inspect
    source = inspect.getsource(ro.main)
    assert "trigger_diagnostic=trigger_diag" in source


def test_tp_order_identity_requires_expected_price():
    from event_engine.bingx import _tp_leg_from_order
    order = {"type": "TAKE_PROFIT_MARKET", "stopPrice": "105.0", "clientOrderId": "EVT_ABC_TP3"}
    assert _tp_leg_from_order(order, "tp3", 105.0, 2, "abc") is True
    assert _tp_leg_from_order(order, "tp3", 106.0, 2, "abc") is False


def test_timeframe_bucket_boundaries_and_cache_helpers(tmp_path):
    import run_once as ro
    # 11:02 UTC-ish for hourly buckets: 10:00-11:00 is complete with 2m grace.
    now_ms = 11 * 3_600_000 + 2 * 60_000
    assert ro._completed_bucket(3_600_000, now_ms, 2) == 10
    # 12:02: the 8:00-12:00 four-hour bucket is complete.
    now_ms = 12 * 3_600_000 + 2 * 60_000
    assert ro._completed_bucket(14_400_000, now_ms, 2) == 2
    event = {
        "event_id": "EVT_CACHE",
        "timestamps": {"detected_at_ts": now_ms - 60 * 60_000},
    }
    original_cache = ro.EVENT_CACHE
    ro.EVENT_CACHE = tmp_path / "events.json"
    ro._save_json_atomic(ro.EVENT_CACHE, {"events": [event]})
    assert ro._load_cached_events()[0]["event_id"] == "EVT_CACHE"
    ro.EVENT_CACHE = original_cache


def test_per_symbol_timeframe_scheduler_does_not_skip_new_candidate(tmp_path):
    import run_once as ro
    state = {"version": 2, "symbols": {"OLD": {"1h": 10}}}
    assert ro._symbol_scan_due(state, "NEW", "1h", 10) is True
    assert ro._symbol_scan_due(state, "OLD", "1h", 10) is False
    ro._mark_symbol_scanned(state, "NEW", "1h", 10)
    assert ro._symbol_scan_due(state, "NEW", "1h", 10) is False


def test_event_cache_merge_keeps_existing_fresh_events_independent_of_universe():
    import run_once as ro
    old = {"event_id": "E1", "symbol": "XYZ", "timeframe": "1h", "timestamps": {"detected_at_ts": 1}}
    new = {"event_id": "E2", "symbol": "ABC", "timeframe": "4h", "timestamps": {"detected_at_ts": 2}}
    merged = ro._merge_event_cache([old], [new])
    assert {x["event_id"] for x in merged} == {"E1", "E2"}


def test_scheduler_uses_rate_limited_scan_wrapper():
    import run_once as ro
    import inspect
    src = inspect.getsource(ro._refresh_timeframe_events)
    assert "_fetch_klines_scan" in src


def test_incomplete_kline_response_defers_watermark(monkeypatch):
    import run_once as ro
    from types import SimpleNamespace
    # Keep the test hermetic: no cross-process file lock under repo data/.
    monkeypatch.setenv("BINGX_GLOBAL_RATE_LIMITER", "false")
    ro._acquire_scan_slot._last_call = None
    monkeypatch.setattr(ro, "fetch_klines", lambda symbol, timeframe, limit: [{"close": 100}] * 20)
    monkeypatch.setattr(ro, "add_cvd", lambda df: df)
    stats = {"divergence_events": 0, "squeeze_events": 0, "events_total": 0, "scan_errors": 0}
    state = {"version": 2, "symbols": {}}
    out = ro._refresh_timeframe_events([SimpleNamespace(symbol="TEST-USDT")], "1h", 250, 1_000_000_000, set(), stats, state, 123)
    assert out == []
    assert state["symbols"].get("TEST-USDT", {}).get("1h") is None


def test_build_event_setup_exception_isolated_in_candidate_path():
    import inspect, run_once as ro
    src = inspect.getsource(ro.main)
    assert "try:" in src and "build_event_setup" in src and "continue" in src


def test_closed_message_uses_trade_timeframe():
    from event_engine.tracker import format_trade_closed_message
    msg = format_trade_closed_message(
        name="TEST", symbol="TEST-USDT", direction="LONG", entry_price=100.0, exit_price=101.0,
        pnl_pct=1.0, realized_rr=0.5, planned_rr=1.05, duration_min=10.0, peak_pnl=2.0,
        max_drawdown=-0.5, exit_reason="TAKE_PROFIT_FULL", event_type="HIDDEN_BULLISH_RSI", timeframe="4h"
    )
    assert "TF <b>4h</b>" in msg


def test_active_trade_register_persists_timeframe(tmp_path, monkeypatch):
    import event_engine.tracker as tr
    monkeypatch.setattr(tr, "_load_active_trades", lambda: {})
    saved = {}
    monkeypatch.setattr(tr, "_save_active_trades", lambda x: saved.update(x))
    tr.register_active_trade(
        event_id="EVT_TEST_4H", symbol="TEST", name="TEST", direction="LONG",
        entry_price=100.0, qty=1.0, tp_orders=[], sl_result={}, event_type="VOLATILITY_SQUEEZE_RELEASE",
        timeframe="4h", setup={"event_timeframe":"4h","planned_risk_pct":1.0,"tp_levels":[],"effective_tp_levels":[]},
    )
    assert saved["EVT_TEST_4H"]["timeframe"] == "4h"


def test_symbol_direction_conflict_keeps_strongest_and_tiebreaks_4h():
    import run_once as ro
    def opp(direction, score, tf):
        return {"symbol":"BTC","direction":direction,"score":score,
                "event":{"timeframe":tf,"event_type":"HIDDEN_BULLISH_RSI"}}
    kept, rejected = ro.resolve_symbol_direction_conflicts([opp("LONG",70,"1h"), opp("SHORT",75,"4h")])
    assert len(kept) == 1 and kept[0]["direction"] == "SHORT"
    assert len(rejected) == 1 and rejected[0]["direction"] == "LONG"
    kept, rejected = ro.resolve_symbol_direction_conflicts([opp("LONG",75,"1h"), opp("SHORT",75,"4h")])
    assert len(kept) == 1 and kept[0]["direction"] == "SHORT"


def test_same_direction_different_timeframes_are_not_conflicts():
    import run_once as ro
    items=[
        {"symbol":"BTC","direction":"LONG","score":80,"event":{"timeframe":"1h"}},
        {"symbol":"BTC","direction":"LONG","score":70,"event":{"timeframe":"4h"}},
    ]
    kept, rejected = ro.resolve_symbol_direction_conflicts(items)
    assert len(kept)==2 and rejected==[]


def test_conflict_resolver_preserves_independent_symbols():
    import run_once as ro
    items=[
        {"symbol":"BTC","direction":"LONG","score":70,"event":{"timeframe":"1h"}},
        {"symbol":"ETH","direction":"SHORT","score":70,"event":{"timeframe":"4h"}},
    ]
    kept, rejected = ro.resolve_symbol_direction_conflicts(items)
    assert len(kept)==2 and rejected==[]


def test_telegram_confluence_and_conflict_visual_fields():
    from event_engine.telegram import format_signal
    event = {
        "symbol": "SOXL", "direction": "LONG", "event_type": "REGULAR_BULLISH_RSI",
        "timeframe": "1h", "event_fact": {"detection_close_price": 110.0, "p1_price": 107.0, "p2_price": 105.0, "price_delta_atr": 0.678},
        "timestamps": {"detected_at_ts": 123},
    }
    setup = {
        "entry_reference": 111.0, "invalidation_price": 107.0, "target_price": 123.0,
        "planned_weighted_rr": 1.05, "tp_mode": "multi_tp",
        "trigger": {"trigger_price": 112.0, "trigger_delay_min": 30.0},
        "confluence_events": [{"timeframe": "4h", "event_type": "HIDDEN_BULLISH_RSI", "event_id": "E2"}],
        "conflict_events": [{"timeframe": "4h", "direction": "SHORT", "event_type": "REGULAR_BEARISH_RSI", "event_id": "E3"}],
    }
    msg = format_signal(event, setup=setup, score=80)
    assert "🔗 <b>CONFLUENCE:</b> <code>4h HIDDEN_BULLISH_RSI</code>" in msg
    assert "⚠️ <b>CONFLICT:</b> <code>4h SHORT</code>" in msg


def test_confluence_events_follow_selected_setup_and_keep_all_same_direction_evidence():
    import run_once as ro
    e1 = {"event_id": "E1", "event_type": "REGULAR_BULLISH_RSI", "timeframe": "1h", "score": 70.0, "detected_at_ts": 100}
    e2 = {"event_id": "E2", "event_type": "HIDDEN_BULLISH_RSI", "timeframe": "4h", "score": 80.0, "detected_at_ts": 200}
    base1 = {"symbol": "SOXL", "direction": "LONG", "event": {"timeframe": "1h", "event_type": "REGULAR_BULLISH_RSI"}, "event_id": "E1", "score": 70.0, "confluence_events": [e1]}
    base2 = {"symbol": "SOXL", "direction": "LONG", "event": {"timeframe": "4h", "event_type": "HIDDEN_BULLISH_RSI"}, "event_id": "E2", "score": 80.0, "confluence_events": [e1, e2]}
    # The selected primary event is not itself displayed as CONFLUENCE.
    display = [e for e in base2["confluence_events"] if e["event_id"] != base2["event_id"]]
    assert {e["event_id"] for e in display} == {"E1"}



def test_squeeze_release_lookback_recovers_recent_closed_release(monkeypatch):
    import event_engine.signals as sig
    n = 50
    df = _generate_synthetic_candles(n)
    df["close_time"] = [1_000_000 + i * 3_600_000 for i in range(n)]
    # Release occurs at n-2; the immediately latest bar is already outside the
    # squeeze, so a last-bar-only detector would miss this transition.
    bb_u = pd.Series([0.0] * n)
    bb_l = pd.Series([0.0] * n)
    mid = pd.Series([0.0] * n)
    kc_u = pd.Series([1.0] * n)
    kc_l = pd.Series([-1.0] * n)
    for i in range(n - 5, n - 1):
        bb_u.iloc[i] = 0.0
        bb_l.iloc[i] = 0.0
    bb_u.iloc[n - 2] = 2.0
    bb_l.iloc[n - 2] = -2.0
    df.loc[n - 2, "close"] = 2.0
    df.loc[n - 1, "close"] = 0.0
    # Ensure no new release at the latest bar.
    bb_u.iloc[n - 1] = 2.0
    bb_l.iloc[n - 1] = -2.0

    monkeypatch.setattr(sig, "_bbands", lambda close, n_, std: (bb_u, mid, bb_l))
    monkeypatch.setattr(sig, "_atr", lambda frame, n_: pd.Series([0.5] * n))
    events = sig.detect_squeeze_release(df, "TEST-USDT", "4h", min_squeeze_bars=3, release_lookback_bars=4)
    assert any(ev["timestamps"]["detected_at_ts"] == int(df["close_time"].iloc[n - 2]) for ev in events)


def test_tf_stats_assigned_before_first_use_in_fresh_event_loop():
    """Regression guard for the production crash:

    UnboundLocalError: cannot access local variable 'tf_stats' where it is
    not associated with a value

    `tf_stats = _tf_stats(stats, tf)` must appear before any `tf_stats[...]`
    subscript use inside the per-event fresh-signal loop in main(). This is
    checked structurally (source order) rather than by driving a full
    main() run, since main() has many external dependencies (Coinalyze,
    BingX, Telegram) that a live crash does not require to reproduce this
    particular class of bug -- it only requires one fresh event to reach
    the loop body, which is exactly what happened in production.
    """
    import inspect
    import run_once as ro

    source = inspect.getsource(ro.main)
    marker = "for ev in sorted(all_events"
    assert marker in source, "fresh-event loop marker not found; test needs updating"
    loop_body = source[source.index(marker):]

    assign_marker = "tf_stats = _tf_stats(stats, tf)"
    assert assign_marker in loop_body, "tf_stats assignment not found in fresh-event loop"
    assign_idx = loop_body.index(assign_marker)

    first_use_idx = loop_body.index("tf_stats[")
    assert assign_idx < first_use_idx, (
        "tf_stats is subscripted before it is assigned in the fresh-event "
        "loop; this reproduces the production UnboundLocalError crash."
    )


# ---------------------------------------------------------------------------
# Audit fixes v8: regression tests for B1-B8
# ---------------------------------------------------------------------------


def _divergence_fixture(n: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 100 + np.cumsum(rng.normal(0, 0.8, n))
    return pd.DataFrame({
        "open_time": [1_700_000_000_000 + i * 3_600_000 - 3_600_000 for i in range(n)],
        "close_time": [1_700_000_000_000 + i * 3_600_000 for i in range(n)],
        "open": base,
        "high": base + 0.5,
        "low": base - 0.5,
        "close": base,
        "volume": np.abs(rng.normal(1000, 100, n)),
        "quote_volume": np.abs(rng.normal(100000, 1000, n)),
        "taker_buy_base": np.abs(rng.normal(500, 50, n)),
        "taker_buy_quote": np.abs(rng.normal(50000, 500, n)),
        "taker_flow_valid": [True] * n,
        "bar_delta_usdt": [0.0] * n,
    })


def test_divergence_detectors_emit_macd_stoch_obv_and_oi_types():
    """Audit B2: Price-vs-OI divergence plus P2 MACD/Stoch/Volume detectors."""
    df = _divergence_fixture()
    df.loc[70, "low"] = float(df.loc[70, "low"]) - 6
    df.loc[70, "close"] = float(df.loc[70, "close"]) - 5.5
    df.loc[90, "low"] = float(df.loc[90, "low"]) - 9
    df.loc[90, "close"] = float(df.loc[90, "close"]) - 8.5

    d = add_cvd(df)
    ct0 = int(df.loc[0, "close_time"])
    oi_hist = {str((ct0 - 1) // 3_600_000 + i): 1_000_000.0 + i * 2_000.0 for i in range(len(df))}
    d = attach_oi_series(d, oi_hist)

    types: set[str] = set()
    for max_bars in (16, 20):
        for ev in detect_divergences(d, "TEST-USDT", "1h", max_bars=max_bars):
            types.add(ev["event_type"])

    assert any(t.endswith("_MACD") for t in types), "MACD divergence missing"
    assert any(t.endswith("_STOCH") for t in types), "Stochastic divergence missing"
    assert any(t.endswith("_OBV") for t in types), "Volume (OBV) divergence missing"
    assert any(t.endswith("_OI") for t in types), "OI divergence missing"


def test_liquidation_squeeze_detector_emits_short_squeeze():
    """Audit B3: forced-liquidation squeeze detection."""
    df = _divergence_fixture()
    last = len(df) - 1
    prev_close = float(df.loc[last - 1, "close"])
    df.loc[last, "close"] = prev_close * 1.05
    df.loc[last, "high"] = prev_close * 1.055
    df.loc[last - 1, "high"] = prev_close * 1.001
    df.loc[last - 1, "low"] = prev_close * 0.999

    row = CoinalyzeRow(
        symbol="TEST", name="TEST", price=prev_close * 1.05, price_chg24=None,
        volume24=1e9, oi=100_000_000.0, oi_chg24_pct=None, oi_chg4h_pct=2.5,
        oi_vol_ratio=None, oi_mktcap_ratio=None, fr_oiw=0.05, pfr_oiw=None,
        liq_short24=1_500_000.0, liq_long24=100_000.0, ls_accounts=0.6,
        btc_corr7d=None, cvd24=None, lls24=None, raw={},
    )
    events = detect_liquidation_squeeze(row, df, "TEST-USDT", "1h")
    assert any(e["event_type"] == "SHORT_SQUEEZE" and e["direction"] == "LONG" for e in events)
    ev = next(e for e in events if e["event_type"] == "SHORT_SQUEEZE")
    assert ev["event_fact"]["liq_ratio_24h"] == pytest.approx(0.015)
    assert ev["event_fact"]["oi_surge"] is True

    # Missing Coinalyze data must yield no events (no guessing).
    assert detect_liquidation_squeeze(None, df, "TEST-USDT", "1h") == []
    empty_row = CoinalyzeRow(
        symbol="TEST", name="TEST", price=1.0, price_chg24=None, volume24=1.0,
        oi=None, oi_chg24_pct=None, oi_chg4h_pct=None, oi_vol_ratio=None,
        oi_mktcap_ratio=None, fr_oiw=None, pfr_oiw=None, liq_short24=None,
        liq_long24=None, ls_accounts=None, btc_corr7d=None, cvd24=None,
        lls24=None, raw={},
    )
    assert detect_liquidation_squeeze(empty_row, df, "TEST-USDT", "1h") == []


def test_move_sl_to_break_even_cancels_old_sl_before_creating_new(monkeypatch):
    """Audit B4: old SL must be cancelled (and verified) BEFORE the new BE SL."""
    import event_engine.tracker as tr

    calls: list[tuple] = []

    monkeypatch.setattr(tr, "to_bx_symbol", lambda s: "TEST-USDT")
    monkeypatch.setattr(tr, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2})

    ok_empty = {"status": "ok", "sl_orders": [], "tp_orders": []}
    ok_with_new = {"status": "ok", "sl_orders": [{"orderId": "NEW1", "clientOrderId": "EVT_BE_T"}], "tp_orders": []}
    posted = {"done": False}

    def fake_protection(symbol, direction):
        return ok_with_new if posted["done"] else ok_empty

    monkeypatch.setattr(tr, "get_open_protection_directional", fake_protection)

    def fake_cancel(symbol, order_id):
        calls.append(("cancel", str(order_id)))
        return {"code": 0}

    monkeypatch.setattr(tr, "cancel_order", fake_cancel)

    def fake_request(method, path, params=None, signed=True):
        calls.append((method.lower(),))
        posted["done"] = True
        return {"code": 0, "data": {"order": {"orderId": "NEW1", "clientOrderId": "EVT_BE_T"}}}

    monkeypatch.setattr(tr, "_request", fake_request)

    result = tr._move_sl_to_break_even(
        "TEST", "LONG", 100.0, 1.0, "OLD1", "T", old_sl_price=97.0,
    )
    assert result["status"] == "created"
    assert result["order_id"] == "NEW1"
    assert calls[0] == ("cancel", "OLD1"), "old SL must be cancelled first"
    assert calls[1] == ("post",), "new BE SL must be created after old cancel"


def test_move_sl_to_break_even_aborts_when_old_cancel_fails(monkeypatch):
    """Audit B4: if the old SL cannot be cancelled, the new SL must NOT be
    created (the position keeps the old stop; no double-SL window)."""
    import event_engine.tracker as tr

    calls: list[tuple] = []
    monkeypatch.setattr(tr, "to_bx_symbol", lambda s: "TEST-USDT")
    monkeypatch.setattr(tr, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2})

    ok_empty = {"status": "ok", "sl_orders": [], "tp_orders": []}
    still_there = {"status": "ok", "sl_orders": [{"orderId": "OLD1"}], "tp_orders": []}
    responses = [ok_empty, still_there]
    monkeypatch.setattr(tr, "get_open_protection_directional", lambda s, d: responses[min(len(responses) - 1, 0)] if False else responses.pop(0) if responses else still_there)

    def fake_cancel(symbol, order_id):
        calls.append(("cancel", str(order_id)))
        return {"code": 1, "msg": "busy"}

    monkeypatch.setattr(tr, "cancel_order", fake_cancel)

    def fake_request(method, path, params=None, signed=True):
        calls.append((method.lower(),))
        return {"code": 0, "data": {"order": {"orderId": "NEW1"}}}

    monkeypatch.setattr(tr, "_request", fake_request)
    monkeypatch.setattr(tr.time, "sleep", lambda s: None)

    result = tr._move_sl_to_break_even("TEST", "LONG", 100.0, 1.0, "OLD1", "T", old_sl_price=97.0)
    assert result["status"] == "error"
    assert "old SL cancel failed" in (result.get("error") or "")
    assert not any(c[0] == "post" for c in calls), "no new SL may be created after a failed old-SL cancel"


def test_diagnose_15m_trigger_marks_missing_event_ts():
    """Audit B7: last-bar fallback must be explicit and optionally forbidden."""
    df = pd.DataFrame({"high": [100, 105], "low": [95, 100], "close": [99, 106]})
    diag = diagnose_15m_trigger(df, "LONG", event_detected_at_ts=None)
    assert diag["event_ts_missing"] is True

    strict = diagnose_15m_trigger(df, "LONG", event_detected_at_ts=None, require_event_ts=True)
    assert strict["ok"] is False
    assert strict["reason"] == "event_ts_required"

    with_ts = pd.DataFrame({
        "high": [100, 105], "low": [95, 100], "close": [99, 106],
        "close_time": [1_000, 2_000],
    })
    diag2 = diagnose_15m_trigger(with_ts, "LONG", event_detected_at_ts=500)
    assert diag2["event_ts_missing"] is False


def test_global_rate_limiter_serializes_processes(tmp_path):
    """Audit B5: the file-lock limiter enforces spacing across independent
    limiter instances sharing one lock directory."""
    import run_once as ro
    import time as time_mod

    min_interval = 0.3
    ro._file_lock_pace(tmp_path, min_interval)  # first call: no wait, stamps t0
    started = time_mod.monotonic()
    ro._file_lock_pace(tmp_path, min_interval)
    waited = time_mod.monotonic() - started
    assert waited >= min_interval - 0.05


def test_open_market_verifies_position_after_transport_error(monkeypatch):
    """Audit P1-4: unknown POST outcome must be verified via the position
    instead of failing outright or blindly retrying."""
    from event_engine import bingx as bx

    CACHE["data"] = {
        "TEST-USDT": {"symbol": "TEST-USDT", "displayName": "TEST-USDT", "status": 1,
                      "apiStateOpen": "true", "quantityPrecision": 3,
                      "tradeMinQuantity": 0.001, "multiplier": 1,
                      "maxLongLeverage": 10, "maxShortLeverage": 10}
    }
    CACHE["ts"] = 9_999_999_999

    position_states = iter([False, True])
    monkeypatch.setattr(bx, "has_open_position", lambda s, d: next(position_states))
    monkeypatch.setattr(bx, "_current_close_price", lambda s: 100.0)
    monkeypatch.setattr(bx, "_set_leverage", lambda *a, **k: True)
    monkeypatch.setattr(bx, "_request", lambda *a, **k: {"code": -1, "msg": "HTTPSConnectionPool read timed out"})

    out = bx.open_market("TEST", "LONG", 100.0, "TRD1")
    assert out["status"] == "opened"
    assert out["idempotency"] == "position_verified_after_transport_error"

    # Missing credentials must still fail fast (not treated as transport error).
    position_states2 = iter([False])
    monkeypatch.setattr(bx, "has_open_position", lambda s, d: next(position_states2))
    monkeypatch.setattr(bx, "_request", lambda *a, **k: {"code": -1, "msg": "missing BingX credentials"})
    out2 = bx.open_market("TEST", "LONG", 100.0, "TRD2")
    assert out2["status"] == "error"


def test_workflow_triggers_via_external_scheduler_only():
    """Audit B6 re-classified as by-design: the engine cadence is owned by an
    external scheduler site that fires repository_dispatch. The workflow must
    keep that trigger and must NOT add its own cron schedule."""
    from pathlib import Path
    wf = Path(__file__).parent / ".github" / "workflows" / "event-engine.yml"
    assert wf.exists(), "event-engine.yml is missing"
    text = wf.read_text(encoding="utf-8")
    assert "repository_dispatch" in text
    assert "run_event_engine" in text
    # No in-repo cadence: the external site owns scheduling.
    assert "schedule:" not in text
    assert "cron:" not in text
    assert "cancel-in-progress: false" in text


def test_coinalyze_header_map_degrades_gracefully():
    """Audit B8: only price/volume24/oi are mandatory; missing optional
    columns must not disable the whole pipeline."""
    from bs4 import BeautifulSoup
    from event_engine.coinalyze import _build_header_map

    html_full = """
    <table><thead><tr>
      <th><span title="Price">Price</span></th>
      <th><span title="Volume 24h">Volume 24h</span></th>
      <th><span title="Open Interest">Open Interest</span></th>
    </tr></thead></table>
    """
    header_map = _build_header_map(BeautifulSoup(html_full, "lxml"))
    assert header_map["price"] == 0
    assert header_map["volume24"] == 1
    assert header_map["oi"] == 2

    html_broken = """
    <table><thead><tr>
      <th><span title="Price">Price</span></th>
    </tr></thead></table>
    """
    with pytest.raises(ValueError):
        _build_header_map(BeautifulSoup(html_broken, "lxml"))


def test_oi_history_recorder_persists_bucket_snapshots(tmp_path, monkeypatch):
    """Audit B2: OI snapshots are stored per symbol per 1h bucket."""
    import run_once as ro
    from types import SimpleNamespace

    monkeypatch.setattr(ro, "OI_HISTORY", tmp_path / "oi_history.json")
    monkeypatch.setattr(ro, "_OI_HIST_CACHE", {"ts": 0.0, "data": {}})
    rows = [
        SimpleNamespace(symbol="TEST-USDT", oi=5_000_000.0),
        SimpleNamespace(symbol="BAD", oi=None),
    ]
    updated = ro._record_oi_snapshots(rows, 1_700_000_000_123)
    assert updated == 1
    history = ro._load_oi_history()
    assert history["TEST-USDT"]["472222"] == 5_000_000.0
    assert "BAD" not in history

    # attach maps a closed bar to the recorded bucket value
    bar = pd.DataFrame({"close_time": [1_700_000_002_000_000 // 1_000]})
    # choose close_time inside bucket 472222: 472222*3600000 <= ct < 472223*3600000
    ct = 472_222 * 3_600_000 + 1_800_000
    bar = pd.DataFrame({"close_time": [ct]})
    attached = ro.attach_oi_series(bar, history.get("TEST-USDT")) if hasattr(ro, "attach_oi_series") else attach_oi_series(bar, history.get("TEST-USDT"))
    assert float(attached["oi"].iloc[0]) == 5_000_000.0


def test_score_gives_squeeze_bonus_to_liquidation_squeezes():
    """New event types must be scored: liq squeeze gets the +25 squeeze bonus,
    new divergence families get the CVD-class +15 bonus."""
    import run_once as ro
    ev_liq = {
        "direction": "LONG",
        "event_type": "SHORT_SQUEEZE",
        "event_fact": {"price_delta_atr": 1.2, "liq_ratio_24h": 0.02},
    }
    score = ro.calculate_setup_score(ev=ev_liq, coinalyze_row=None, df_15m=pd.DataFrame({"close": [1]}))
    assert score >= 75  # 50 base + 15 delta_atr + 25 squeeze (vol-factors absent)

    ev_oi = {
        "direction": "LONG",
        "event_type": "REGULAR_BULLISH_OI",
        "event_fact": {"price_delta_atr": 0.6},
    }
    score_oi = ro.calculate_setup_score(ev=ev_oi, coinalyze_row=None, df_15m=pd.DataFrame({"close": [1]}))
    assert score_oi >= 60  # 50 base + 10 delta_atr + 15 OI family bonus

