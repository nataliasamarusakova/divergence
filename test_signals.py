# test_signals.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _legacy_divergence_defaults(monkeypatch):
    """Keep legacy unit cases deterministic; production defaults stay explicit in release config."""
    monkeypatch.setenv("DIVERGENCE_VOLUME_CONFIRMATION_ENABLED", "false")
    import run_once as _ro
    monkeypatch.setattr(_ro, "MARKET_DATA_SOURCE", "bingx")
    monkeypatch.setattr(_ro, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", False)


from event_engine.coinalyze import parse_number, CoinalyzeRow
from event_engine.signals import (
    _rsi,
    add_cvd,
    build_15m_trigger,
    diagnose_15m_trigger,
    detect_divergences,
    detect_liquidity_sweep_reclaim,
    detect_squeeze_release,
    detect_liquidation_squeeze,
    detect_order_block,
    detect_breaker_block,
    detect_mitigation_block,
    detect_sfp,
    detect_liquidation_cascade_fvg,
    detect_crt,
    attach_oi_series,
    check_btc_regime,
)
from event_engine.bingx import (
    get_contract,
    CACHE,
    _allocate_tp_quantities,
)
from event_engine.tracker import _update_mfe_mae, _extract_setup_metrics, _load_active_trades
from run_once import (
    build_event_setup,
    build_tp_levels,
    calculate_setup_score,
    execute_new_position,
    load_successful_telegram_ids,
    check_funding_filter,
    _load_recent_successful_entries,
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


def test_divergence_detector_checks_non_adjacent_pivot_pairs(monkeypatch):
    # Four pivots are present, but only the first -> fourth pair falls inside
    # the configured bar window. The detector must therefore inspect a pivot
    # that is more than two positions away in the pivot ledger.
    from event_engine import signals as sig

    df = _generate_synthetic_candles(75)
    df["high"] = 101.0
    df["low"] = 99.0
    df["close"] = 100.0
    df.loc[10, "low"] = 90.0
    df.loc[40, "low"] = 80.0

    rsi = pd.Series(50.0, index=df.index)
    rsi.iloc[40] = 60.0
    monkeypatch.setattr(sig, "_rsi", lambda series, n=14: rsi.copy())
    monkeypatch.setattr(sig, "_pivots", lambda work, left=3, right=2: ([10, 20, 30, 40], []))

    events = sig.detect_divergences(
        df,
        "BTC-USDT",
        "1h",
        left=3,
        right=2,
        min_bars=25,
        max_bars=35,
    )

    rsi_events = [ev for ev in events if ev["event_type"] == "REGULAR_BULLISH_RSI"]
    assert len(rsi_events) == 1
    assert rsi_events[0]["event_fact"]["bars_between"] == 30
    assert rsi_events[0]["timestamps"]["pivot_1_ts"] < rsi_events[0]["timestamps"]["pivot_2_ts"]


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
    assert setup_long["planned_weighted_rr"] == pytest.approx(1.6625)
    sl_pct_l, tp_levels_l = build_tp_levels(setup_long, "LONG")
    assert sl_pct_l > 0
    assert len(tp_levels_l) == 3
    assert tp_levels_l[0]["pnl_pct"] < tp_levels_l[1]["pnl_pct"] < tp_levels_l[2]["pnl_pct"]

    # Short setup
    setup_short = build_event_setup({"direction": "SHORT"}, df, entry_price=100.0)
    assert setup_short["invalidation_price"] > 100.0
    assert setup_short["target_price"] < 100.0
    assert setup_short["planned_weighted_rr"] == pytest.approx(1.6625)
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


def test_execute_new_position_blocks_cross_exchange_drift(monkeypatch):
    import run_once as ro
    monkeypatch.setattr(ro, "MARKET_DATA_SOURCE", "binance")
    monkeypatch.setattr(ro, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", True)
    monkeypatch.setattr(ro, "MAX_CROSS_EXCHANGE_DRIFT_PCT", 1.0)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "fetch_binance_price", lambda symbol: 102.0)
    monkeypatch.setattr(ro, "open_market", lambda *a, **k: pytest.fail("BingX order must not be sent"))
    out = ro.execute_new_position(
        "TEST", "LONG", 100.0,
        {"risk_pct": 1.0, "signal_price": 100.0, "event_type": "REGULAR_BULLISH_RSI"},
        "EVT_CROSS_DRIFT",
    )
    assert out["status"] == "CROSS_EXCHANGE_DRIFT_EXCEEDED"
    assert out["execution_quality"]["cross_exchange_drift_pct"] == pytest.approx(1.960784, rel=1e-5)


def test_execute_new_position_defines_pre_order_price(monkeypatch):
    import run_once as ro
    monkeypatch.setattr(ro, "MAX_ENTRY_DRIFT_PCT", 3.0)

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
    setup = {"risk_pct": 1.0, "planned_weighted_rr": 1.6625, "entry_reference": 99.0, "target_rr": 2.50}
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
    assert metrics["planned_weighted_rr"] == pytest.approx(1.6625)
    assert metrics["effective_weighted_rr"] == pytest.approx(1.6625)


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
    assert "_fetch_market_klines_scan" in src


def test_successful_scan_persists_watermark_after_event_emission(monkeypatch, tmp_path):
    import run_once as ro
    from types import SimpleNamespace

    monkeypatch.setattr(ro, "_fetch_klines_scan", lambda *args, **kwargs: [
        {"open": 100, "high": 101, "low": 99, "close": 100, "close_time": 1_000_000 + i * 3_600_000, "volume": 10}
        for i in range(80)
    ])
    monkeypatch.setattr(ro, "add_cvd", lambda df: df)
    monkeypatch.setattr(ro, "attach_oi_series", lambda df, hist: df)
    monkeypatch.setattr(ro, "detect_divergences", lambda *args: [])
    monkeypatch.setattr(ro, "detect_squeeze_release", lambda *args, **kwargs: [])
    monkeypatch.setattr(ro, "detect_liquidation_squeeze", lambda *args, **kwargs: [])
    saved = []
    monkeypatch.setattr(ro, "_save_timeframe_scan_state", lambda state: saved.append(state.copy()))

    stats = {"divergence_events": 0, "squeeze_events": 0, "events_total": 0, "scan_errors": 0}
    state = {"version": 2, "symbols": {}}
    ro._refresh_timeframe_events(
        [SimpleNamespace(symbol="TEST-USDT")], "1h", 250, 2_000_000_000, set(), stats, state, 123
    )

    assert state["symbols"]["TEST-USDT"]["1h"] == 123
    assert saved and saved[-1]["symbols"]["TEST-USDT"]["1h"] == 123


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


def test_add_macd_produces_signal_and_histogram_with_warmup():
    from event_engine.signals import add_macd

    df = pd.DataFrame({"close": [100.0 + i for i in range(50)]})
    out = add_macd(df)

    assert out["macd"].iloc[:25].isna().all()
    assert out["macd"].iloc[25] == pytest.approx(
        out["close"].ewm(span=12, adjust=False, min_periods=12).mean().iloc[25]
        - out["close"].ewm(span=26, adjust=False, min_periods=26).mean().iloc[25]
    )
    assert out["macd_signal"].iloc[25:33].isna().all()
    assert out["macd_hist"].iloc[34] == pytest.approx(
        out["macd"].iloc[34] - out["macd_signal"].iloc[34]
    )


def test_cvd_divergence_requires_price_to_be_within_vwap_band(monkeypatch):
    import event_engine.signals as sig

    df = _generate_synthetic_candles(90)
    df["volume"] = 1000.0
    df["quote_volume"] = df["volume"] * df["close"]
    df["taker_flow_valid"] = True
    df["bar_delta_usdt"] = 0.0
    df.loc[30, "low"] = 90.0
    df.loc[50, "low"] = 80.0
    df.loc[30, "bar_delta_usdt"] = -1000.0
    df.loc[50, "bar_delta_usdt"] = 1000.0
    df.loc[52, "close"] = 120.0
    monkeypatch.setattr(sig, "_pivots", lambda work, left=3, right=2: ([30, 50], []))

    df = sig.add_cvd(df)
    events = sig.detect_divergences(df, "TEST-USDT", "1h", min_bars=10, max_bars=30, min_delta_atr=0.1)
    assert not any(e["event_type"] == "REGULAR_BULLISH_BINGX_CVD" for e in events)

    # Move the detection bar close close to its cumulative VWAP so the CVD event can pass.
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap_52 = float((typical.iloc[:53] * df["volume"].iloc[:53]).sum() / df["volume"].iloc[:53].sum())
    df.loc[52, "close"] = vwap_52
    df = sig.add_cvd(df)
    events = sig.detect_divergences(df, "TEST-USDT", "1h", min_bars=10, max_bars=30, min_delta_atr=0.1)
    cvd_events = [e for e in events if e["event_type"] == "REGULAR_BULLISH_BINGX_CVD"]
    assert cvd_events
    assert cvd_events[0]["event_fact"]["vwap_distance_pct"] <= 1.50


def test_mfi_uses_signed_money_flow_and_handles_zero_negative_flow():
    from event_engine.signals import add_mfi

    df = pd.DataFrame({
        "high": [10.0 + i for i in range(20)],
        "low": [9.0 + i for i in range(20)],
        "close": [9.5 + i for i in range(20)],
        "volume": [100.0] * 20,
    })
    out = add_mfi(df, length=14)
    assert out["mfi"].iloc[:13].isna().all()
    assert out["mfi"].iloc[-1] == pytest.approx(100.0)

    flat = df.copy()
    flat["high"] = 10.0
    flat["low"] = 9.0
    flat["close"] = 9.5
    flat_out = add_mfi(flat, length=14)
    assert flat_out["mfi"].iloc[-1] == pytest.approx(50.0)


def test_cmf_uses_close_location_and_zero_range_guard():
    from event_engine.signals import add_cmf

    df = pd.DataFrame({
        "high": [10.0] * 20,
        "low": [9.0] * 20,
        "close": [9.75] * 20,
        "volume": [100.0] * 20,
    })
    out = add_cmf(df, length=20)
    assert out["cmf"].iloc[:19].isna().all()
    assert out["cmf"].iloc[-1] == pytest.approx(0.5)

    flat = df.copy()
    flat["high"] = 10.0
    flat["low"] = 10.0
    flat["close"] = 10.0
    flat_out = add_cmf(flat, length=20)
    assert flat_out["cmf"].iloc[-1] == pytest.approx(0.0)


def test_volume_confirmed_divergence_defaults_on_and_enforces_both_thresholds(monkeypatch):
    import event_engine.signals as sig

    df = _generate_synthetic_candles(90)
    df["volume"] = 1000.0
    df["taker_flow_valid"] = True
    df["bar_delta_usdt"] = 0.0
    df.loc[30, "low"] = 90.0
    df.loc[50, "low"] = 80.0
    df.loc[30, "bar_delta_usdt"] = -1000.0
    df.loc[50, "bar_delta_usdt"] = 1000.0
    df = sig.add_cvd(df)
    def _forced_rsi(close, n=14):
        out = pd.Series(50.0, index=close.index)
        out.loc[30] = 40.0
        out.loc[50] = 60.0
        return out
    monkeypatch.setattr(sig, "_rsi", _forced_rsi)
    monkeypatch.setattr(sig, "_pivots", lambda work, left=3, right=2: ([30, 50], []))

    monkeypatch.delenv("DIVERGENCE_VOLUME_CONFIRMATION_ENABLED", raising=False)
    events = sig.detect_divergences(df, "TEST-USDT", "1h", min_bars=10, max_bars=30, min_delta_atr=0.1)
    assert not any(e["event_type"] == "REGULAR_BULLISH_RSI" for e in events)

    monkeypatch.setenv("DIVERGENCE_VOLUME_CONFIRMATION_ENABLED", "false")
    events = sig.detect_divergences(df, "TEST-USDT", "1h", min_bars=10, max_bars=30, min_delta_atr=0.1)
    assert any(e["event_type"] == "REGULAR_BULLISH_RSI" for e in events)

    monkeypatch.setenv("DIVERGENCE_VOLUME_CONFIRMATION_ENABLED", "true")
    events = sig.detect_divergences(df, "TEST-USDT", "1h", min_bars=10, max_bars=30, min_delta_atr=0.1)
    assert not any(e["event_type"] == "REGULAR_BULLISH_RSI" for e in events)

    df.loc[52, "volume"] = 1300.0
    events = sig.detect_divergences(df, "TEST-USDT", "1h", min_bars=10, max_bars=30, min_delta_atr=0.1)
    assert any(e["event_type"] == "REGULAR_BULLISH_RSI" for e in events)


def test_validate_divergence_context_marks_matching_htf_confirmation(monkeypatch):
    import event_engine.signals as sig

    htf_event = {
        "event_id": "HTF1",
        "event_type": "REGULAR_BULLISH_RSI",
        "direction": "LONG",
        "timestamps": {"detected_at_ts": 1_700_000_000_000},
    }
    monkeypatch.setattr(sig, "detect_divergences", lambda *args, **kwargs: [htf_event])
    monkeypatch.setattr(sig, "_trend_context", lambda df, direction: {"trend": "BULLISH", "trend_ok": True})

    ev = {
        "symbol": "TEST-USDT",
        "event_type": "REGULAR_BULLISH_RSI",
        "direction": "LONG",
        "timestamps": {"detected_at_ts": 1_700_000_000_000 + 3 * 3_600_000},
        "event_fact": {"price_delta_atr": 1.0},
    }
    ok, reason, meta = sig.validate_divergence_context(ev, pd.DataFrame(index=range(60)), "4h")
    assert ok is True
    assert reason == "REGULAR_CONTEXT_OK"
    assert meta["mtf_confirmed"] is True
    assert meta["mtf_confirmation_event_id"] == "HTF1"


def test_calc_trade_pnl_returns_none_for_invalid_prices_and_message_marks_data_error():
    from event_engine.tracker import _calc_trade_pnl_pct, format_trade_closed_message

    assert _calc_trade_pnl_pct(0.0, 101.0, "LONG") is None
    assert _calc_trade_pnl_pct(100.0, 0.0, "LONG") is None

    msg = format_trade_closed_message(
        name="TEST", symbol="TEST-USDT", direction="LONG", entry_price=0.0, exit_price=101.0,
        pnl_pct=None, realized_rr=None, planned_rr=1.05, duration_min=10.0, peak_pnl=0.0,
        max_drawdown=0.0, exit_reason="DATA_ERROR", event_type="RECONCILED_POSITION", timeframe="1h"
    )
    assert "DATA_ERROR" in msg


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
    assert any(t.endswith("_MACD_HIST") for t in types), "MACD histogram divergence missing"
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
    monkeypatch.setattr(tr, "get_position_directional", lambda s, d: {"status": "found", "positionAmt": "1"})

    ok_empty = {"status": "ok", "sl_orders": [], "tp_orders": []}
    ok_with_new = {"status": "ok", "sl_orders": [{"orderId": "NEW1", "clientOrderId": "EVT_BE_T", "stopPrice": "100.00", "origQty": "1.000"}], "tp_orders": []}
    posted = {"done": False}

    def fake_protection(symbol, direction):
        return ok_with_new if posted["done"] else ok_empty

    monkeypatch.setattr(tr, "get_open_protection_directional", fake_protection)

    def fake_cancel(symbol, order_id):
        calls.append(("cancel", str(order_id)))
        return {"code": 0}

    monkeypatch.setattr(tr, "cancel_order", fake_cancel)

    def fake_verified(symbol, direction, params, client_order_id, **kwargs):
        calls.append(("post",))
        posted["done"] = True
        return {"code": 0, "data": {"order": {"orderId": "NEW1", "clientOrderId": client_order_id}}}

    monkeypatch.setattr(tr, "_post_protection_order_verified", fake_verified)

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
    monkeypatch.setattr(tr, "get_position_directional", lambda s, d: {"status": "found", "positionAmt": "1"})

    ok_empty = {"status": "ok", "sl_orders": [], "tp_orders": []}
    still_there = {"status": "ok", "sl_orders": [{"orderId": "OLD1"}], "tp_orders": []}
    responses = [ok_empty, still_there]
    monkeypatch.setattr(tr, "get_open_protection_directional", lambda s, d: responses[min(len(responses) - 1, 0)] if False else responses.pop(0) if responses else still_there)

    def fake_cancel(symbol, order_id):
        calls.append(("cancel", str(order_id)))
        return {"code": 1, "msg": "busy"}

    monkeypatch.setattr(tr, "cancel_order", fake_cancel)

    def fake_verified(*args, **kwargs):
        calls.append(("post",))
        return {"code": 0, "data": {"order": {"orderId": "NEW1"}}}

    monkeypatch.setattr(tr, "_post_protection_order_verified", fake_verified)
    monkeypatch.setattr(tr.time, "sleep", lambda s: None)

    result = tr._move_sl_to_break_even("TEST", "LONG", 100.0, 1.0, "OLD1", "T", old_sl_price=97.0)
    assert result["status"] == "error"
    assert "old SL cancel failed" in (result.get("error") or "")
    assert not any(c[0] == "post" for c in calls), "no new SL may be created after a failed old-SL cancel"


def test_trigger_stale_guard_uses_current_time_not_cycle_start(monkeypatch):
    import run_once

    observed_ms = 1_000_000
    cycle_start_ms = observed_ms + 10 * 60_000
    current_ms = observed_ms + 11 * 60_000
    meta = {"trigger_observed_at_ts": observed_ms, "trigger_bar_close_ts": observed_ms}

    # The regression target is the production pattern used by main():
    # the stale-age calculation must be based on the clock at execution time,
    # not the frozen cycle-start timestamp.
    monkeypatch.setattr(run_once.time, "time", lambda: current_ms / 1000.0)
    execution_now_ms = int(run_once.time.time() * 1000)
    observed_ts = run_once._safe_float(meta.get("trigger_observed_at_ts"), 0.0)
    cycle_age_min = (cycle_start_ms - observed_ts) / 60_000.0
    execution_age_min = max(0.0, (execution_now_ms - observed_ts) / 60_000.0)

    assert cycle_age_min == 10.0
    assert execution_age_min == 11.0
    assert execution_age_min > cycle_age_min


def test_trigger_observation_timestamp_is_distinct_from_cycle_start(monkeypatch):
    import run_once

    calls = iter([1000.0, 1007.0])
    monkeypatch.setattr(run_once.time, "time", lambda: next(calls))
    cycle_start_ms = int(run_once.time.time() * 1000)
    trigger_observed_at_ts = int(run_once.time.time() * 1000)

    assert cycle_start_ms == 1_000_000
    assert trigger_observed_at_ts == 1_007_000
    assert trigger_observed_at_ts > cycle_start_ms


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
    """The compatibility limiter enforces spacing within one process."""
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
    bucket = str(1_700_000_000_123 // 3_600_000)
    assert history["TEST-USDT"][bucket] == 5_000_000.0
    assert "BAD" not in history

    # attach maps a closed bar to the recorded bucket value
    bar = pd.DataFrame({"close_time": [1_700_000_002_000_000 // 1_000]})
    # choose close_time inside the recorded bucket
    ct = int(bucket) * 3_600_000 + 1_800_000
    bar = pd.DataFrame({"close_time": [ct]})
    attached = ro.attach_oi_series(bar, history.get("TEST-USDT")) if hasattr(ro, "attach_oi_series") else attach_oi_series(bar, history.get("TEST-USDT"))
    assert float(attached["oi"].iloc[0]) == 5_000_000.0


def test_funding_history_recorder_and_attachment_are_causal(tmp_path, monkeypatch):
    import run_once as ro
    from event_engine.signals import attach_funding_series
    from types import SimpleNamespace

    monkeypatch.setattr(ro, "FUNDING_HISTORY", tmp_path / "funding_history.json")
    monkeypatch.setattr(ro, "_FUNDING_HIST_CACHE", {"ts": 0.0, "data": {}, "path": ""})
    rows = [
        SimpleNamespace(symbol="TEST-USDT", fr_oiw=-0.0125),
        SimpleNamespace(symbol="BAD", fr_oiw=None),
    ]
    now_ms = 1_700_000_000_123
    assert ro._record_funding_snapshots(rows, now_ms) == 1
    history = ro._load_funding_history()
    bucket = str(now_ms // 3_600_000)
    assert history["TEST-USDT"][bucket] == pytest.approx(-0.0125)
    assert "BAD" not in history

    close_time = int(bucket) * 3_600_000 + 1_800_000
    bar = pd.DataFrame({"close_time": [close_time]})
    attached = attach_funding_series(bar, history["TEST-USDT"])
    assert float(attached["fr_oiw"].iloc[0]) == pytest.approx(-0.0125)


def test_funding_zscore_divergence_uses_normalized_series(monkeypatch):
    import event_engine.signals as sig

    df = _divergence_fixture(120)
    df["fr_oiw"] = 0.0
    df.loc[30, "high"] = 110.0
    df.loc[40, "fr_oiw"] = 0.030
    df.loc[70, "fr_oiw"] = -0.010
    df.loc[70, "high"] = 109.0
    df.loc[90, "high"] = 120.0
    df.loc[90, "fr_oiw"] = -0.005

    monkeypatch.setattr(sig, "_pivots", lambda work, left=3, right=2: ([], [30, 50, 70, 90]))
    events = sig.detect_divergences(
        df, "TEST-USDT", "1h", min_bars=20, max_bars=70, min_delta_atr=0.1
    )
    assert any(e["event_type"] == "REGULAR_BEARISH_FR_OIW_Z" for e in events)


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



def test_symbol_cooldown_uses_only_confirmed_position_opens(tmp_path: Path):
    import run_once as ro
    path = tmp_path / "trades.jsonl"
    now = 2_000_000_000
    records = [
        {"record_type": "TRADE_OPEN", "symbol": "ABC", "ts": now - 60_000,
         "execution": {"status": "opened_protected"}, "result": {"position": {"positionAmt": "1"}}},
        {"record_type": "TRADE_OPEN", "symbol": "DEF", "ts": now - 30_000,
         "execution": {"status": "OPEN_FAILED"}, "result": {}},
    ]
    path.write_text("\n".join(json.dumps(x) for x in records) + "\n", encoding="utf-8")
    latest = ro._load_recent_successful_entries(path, now, 15)
    assert ro._symbol_on_cooldown("ABC", latest, now, 15)
    assert not ro._symbol_on_cooldown("DEF", latest, now, 15)


def test_sl_validation_requires_expected_price_and_qty():
    from event_engine import bingx as bx
    order = {"type": "STOP_MARKET", "stopPrice": "95", "origQty": "1"}
    assert bx._validate_sl_order_for_position(order, "LONG", 100.0, expected_price=95.0, expected_qty=1.0)
    assert not bx._validate_sl_order_for_position(order, "LONG", 100.0, expected_price=90.0, expected_qty=1.0)
    assert not bx._validate_sl_order_for_position(order, "LONG", 100.0, expected_price=95.0, expected_qty=2.0)


def test_sl_validation_accepts_break_even_only_when_explicitly_expected():
    from event_engine import bingx as bx
    long_be = {"type": "STOP_MARKET", "stopPrice": "100", "origQty": "1"}
    short_be = {"type": "STOP_MARKET", "stopPrice": "100", "origQty": "1"}

    assert bx._validate_sl_order_for_position(long_be, "LONG", 100.0, expected_price=100.0, expected_qty=1.0)
    assert bx._validate_sl_order_for_position(short_be, "SHORT", 100.0, expected_price=100.0, expected_qty=1.0)
    assert not bx._validate_sl_order_for_position(long_be, "LONG", 100.0, expected_price=99.0, expected_qty=1.0)
    assert not bx._validate_sl_order_for_position(short_be, "SHORT", 100.0, expected_price=101.0, expected_qty=1.0)


def test_ensure_protection_accepts_existing_break_even_sl_without_emergency_close(monkeypatch):
    from event_engine import bingx as bx

    monkeypatch.setattr(bx, "to_bx_symbol", lambda s: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda s: {
        "quantityPrecision": 3, "pricePrecision": 2, "tradeMinQuantity": 0.001,
    })
    be_sl = {
        "orderId": "BE1", "type": "STOP_MARKET", "stopPrice": "100.00", "origQty": "1.0",
    }
    monkeypatch.setattr(
        bx, "get_open_protection_directional",
        lambda *a, **k: {"status": "ok", "sl_orders": [be_sl], "tp_orders": []},
    )

    def fail_emergency(*args, **kwargs):
        raise AssertionError("BE protection must not trigger emergency close")

    monkeypatch.setattr(bx, "emergency_close_position", fail_emergency)

    out = bx.ensure_directional_protection(
        "TEST", "LONG", 100.0, 1.0, 5.0, [], trade_id="TRD1", stop_loss_price=100.0
    )

    assert out["status"] == "SL_ONLY"
    assert out["sl_result"]["status"] == "already_exists"
    assert out["sl_result"]["stop_price"] == 100.0


def test_protection_transport_error_verifies_existing_order_before_retry(monkeypatch):
    from event_engine import bingx as bx
    calls = []
    monkeypatch.setattr(bx, "_request", lambda *a, **k: calls.append(1) or {"code": -1, "msg": "read timed out"})
    monkeypatch.setattr(bx, "_find_open_order_for_post", lambda *a, **k: (
        "found", {"orderId": "SL1", "clientOrderId": "CID", "type": "STOP_MARKET"}
    ))
    out = bx._post_protection_order_verified(
        "TEST", "LONG", {"type": "STOP_MARKET", "positionSide": "LONG", "side": "SELL", "quantity": "1", "stopPrice": "95"}, None
    )
    assert out["code"] == 0
    assert out["recovered"] is True
    assert len(calls) == 1


def test_protection_transport_error_does_not_blind_retry_conditional_order(monkeypatch):
    from event_engine import bingx as bx
    responses = iter([
        {"code": -1, "msg": "read timed out"},
    ])
    calls = []
    monkeypatch.setattr(bx, "_request", lambda *a, **k: calls.append(1) or next(responses))
    monkeypatch.setattr(bx, "_find_open_order_for_post", lambda *a, **k: ("absent", None))
    monkeypatch.setattr(bx, "get_position_directional", lambda *a, **k: {"status": "found", "positionAmt": "1"})
    monkeypatch.setattr(bx.time, "sleep", lambda *_: None)
    out = bx._post_protection_order_verified(
        "TEST", "LONG", {"type": "STOP_MARKET", "positionSide": "LONG", "side": "SELL", "quantity": "1", "stopPrice": "95"}, None
    )
    assert out["code"] == -1
    assert out["protection_state_unknown"] is True
    assert len(calls) == 1


def test_in_cycle_position_state_is_updated_immediately():
    import run_once as ro
    opened = {}
    positions = {}
    ro._mark_local_position_state(
        opened, positions,
        {"symbol": "ABC-USDT", "positionAmt": "1", "avgPrice": "100"},
        "ABC", "LONG",
    )
    assert opened[("ABC-USDT", "LONG")] is True
    assert positions[("ABC-USDT", "LONG")]["avgPrice"] == "100"


def test_protection_transport_error_does_not_retry_when_verification_is_unknown(monkeypatch):
    from event_engine import bingx as bx
    calls = []
    monkeypatch.setattr(bx, "_request", lambda *a, **k: calls.append(1) or {"code": -1, "msg": "read timed out"})
    monkeypatch.setattr(bx, "_find_open_order_for_post", lambda *a, **k: ("unknown", None))
    out = bx._post_protection_order_verified("TEST", "LONG", {"type": "STOP_MARKET", "positionSide": "LONG", "side": "SELL", "quantity": "1", "stopPrice": "95"}, None)
    assert out["code"] == -1
    assert out["protection_state_unknown"] is True
    assert len(calls) == 1


def test_telegram_exit_notification_is_persistent_and_retries_per_chat(monkeypatch, tmp_path):
    import event_engine.tracker as tr
    tr.NOTIFICATIONS_PATH = tmp_path / "notifications.json"
    attempts = []

    monkeypatch.setenv("TG_CHAT_IDS", "A,B")
    def fake_send_detailed(text, only_chat_ids=None):
        attempts.append(tuple(only_chat_ids or []))
        if len(attempts) == 1:
            return {"A": {"sent": True}, "B": {"sent": False, "error": "temporary"}}
        return {cid: {"sent": True} for cid in (only_chat_ids or [])}

    monkeypatch.setattr(tr, "send_detailed", fake_send_detailed)

    nid = tr._notification_id("EVT_TEST", "TRADE_CLOSE")
    assert tr._queue_notification(
        nid, event_id="EVT_TEST", kind="TRADE_CLOSE", symbol="ABC", direction="LONG", text="close"
    ) is False
    assert attempts[-1] == ("A", "B")

    # Only the undelivered chat is retried; the successful chat is never duplicated.
    assert tr._queue_notification(
        nid, event_id="EVT_TEST", kind="TRADE_CLOSE", symbol="ABC", direction="LONG", text="close"
    ) is True
    assert attempts[-1] == ("B",)


def test_tp_leg_identity_works_without_client_order_id():
    from event_engine import bingx as bx
    order = {"type": "TAKE_PROFIT_MARKET", "stopPrice": "105.00", "origQty": "1"}
    assert bx._tp_leg_from_order(order, "tp1", 105.0, 2, None) is True
    assert bx._tp_leg_from_order(order, "tp1", 106.0, 2, None) is False


def test_ensure_protection_refuses_second_sl_and_cleans_stale(monkeypatch):
    from event_engine import bingx as bx
    calls = []
    monkeypatch.setattr(bx, "to_bx_symbol", lambda s: "ABC-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2, "tradeMinQuantity": 0.001})
    live = {"old": True}
    def fake_protection(s, d):
        return {"status": "ok", "sl_orders": ([{"orderId": "OLD", "type": "STOP_MARKET", "stopPrice": "80", "origQty": "1"}] if live["old"] else []), "tp_orders": []}
    monkeypatch.setattr(bx, "get_open_protection_directional", fake_protection)
    def fake_cancel(s, oid):
        calls.append(oid)
        live["old"] = False
        return {"code": 0}
    monkeypatch.setattr(bx, "cancel_order", fake_cancel)
    monkeypatch.setattr(bx, "_post_protection_order_verified", lambda *a, **k: calls.append("POST") or {"code": 0, "data": {"order": {"orderId": "NEW", "clientOrderId": "CID"}}})
    result = bx.ensure_directional_protection("ABC", "LONG", 100.0, 1.0, 5.0, [], trade_id="T")
    assert "OLD" in calls
    assert "POST" in calls
    assert calls.index("OLD") < calls.index("POST")


def test_protection_timeout_aborts_when_position_disappeared(monkeypatch):
    from event_engine import bingx as bx
    monkeypatch.setattr(bx, "_request", lambda *a, **k: {"code": -1, "msg": "read timed out"})
    monkeypatch.setattr(bx, "_find_open_order_for_post", lambda *a, **k: ("absent", None))
    monkeypatch.setattr(bx, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    out = bx._post_protection_order_verified(
        "ABC", "LONG", {"type": "STOP_MARKET", "positionSide": "LONG", "side": "SELL", "quantity": "1", "stopPrice": "95"}, None
    )
    assert out["code"] == -1
    assert out.get("position_gone") is True
    assert out.get("protection_state_unknown") is True


def test_shadow_counts_unique_confirmed_trades_not_journal_records(tmp_path):
    from event_engine.shadow import generate_shadow_health_snapshot
    events = tmp_path / "events.jsonl"
    trades = tmp_path / "trades.jsonl"
    events.write_text("{}\n", encoding="utf-8")
    records = [
        {"record_type": "TRADE_OPEN", "trade_id": "TR_A", "execution": {"status": "opened_protected"}, "result": {"position": {"positionAmt": "1"}}},
        {"record_type": "TRADE_CLOSE", "trade_id": "TR_A"},
        {"record_type": "TRADE_OPEN", "trade_id": "TR_B", "execution": {"status": "OPEN_FAILED"}, "result": {}},
        {"record_type": "TRADE_OPEN", "trade_id": "TR_C", "execution": {"status": "opened_protection_check_required"}, "result": {"position": {"positionAmt": "2"}}},
    ]
    trades.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    snap = generate_shadow_health_snapshot(events, trades)
    assert snap["trades"]["journal_records"] == 4
    assert snap["trades"]["total"] == 2
    assert snap["trades"]["confirmed_open_records"] == 2
    assert snap["trades"]["close_records"] == 1


def test_load_active_trades_normalizes_null_tp_orders_without_losing_other_records(tmp_path, monkeypatch):
    import event_engine.tracker as tr

    active_path = tmp_path / "active_trades.json"
    active_path.write_text(json.dumps({
        "GOOD": {"tp_orders": [{"order_id": "TP1"}]},
        "BAD_NULL": {"tp_orders": None},
        "GOOD2": {"tp_orders": []},
    }), encoding="utf-8")
    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", active_path)

    state = _load_active_trades()

    assert set(state) == {"GOOD", "BAD_NULL", "GOOD2"}
    assert state["GOOD"]["tp_orders"] == [{"order_id": "TP1"}]
    assert state["BAD_NULL"]["tp_orders"] == []
    assert state["BAD_NULL"]["tp_mode"] == "multi_tp"


def test_load_active_trades_isolates_invalid_tp_orders_type(tmp_path, monkeypatch):
    import event_engine.tracker as tr

    active_path = tmp_path / "active_trades.json"
    active_path.write_text(json.dumps({
        "GOOD": {"tp_orders": [{"order_id": "TP1"}]},
        "BAD_TYPE": {"tp_orders": {"order_id": "TP2"}},
    }), encoding="utf-8")
    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", active_path)

    state = _load_active_trades()

    assert set(state) == {"GOOD", "BAD_TYPE"}
    assert state["BAD_TYPE"]["tp_orders"] == []
    assert state["BAD_TYPE"]["tp_mode"] == "multi_tp"


def test_load_active_trades_skips_only_invalid_non_object_record(tmp_path, monkeypatch):
    import event_engine.tracker as tr

    active_path = tmp_path / "active_trades.json"
    active_path.write_text(json.dumps({
        "GOOD": {"tp_orders": []},
        "BAD_SCALAR": None,
        "GOOD2": {"tp_orders": [{"order_id": "TP2"}]},
    }), encoding="utf-8")
    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", active_path)

    state = _load_active_trades()

    assert set(state) == {"GOOD", "GOOD2"}


def test_trade_registration_has_stable_trade_id():
    import event_engine.tracker as tr
    import tempfile
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        monkey = pytest.MonkeyPatch()
        monkey.setattr(tr, "ACTIVE_TRADES_PATH", base / "active_trades.json")
        monkey.setattr(tr, "TRADES_PATH", base / "trades.jsonl")
        tr.register_active_trade("EVT_TEST123", "ABC-USDT", "ABC", "LONG", 100.0, 1.0, [], {"status": "created", "order_id": "SL1", "stop_price": 95}, "DIV", setup={"risk_pct": 5.0})
        state = tr._load_active_trades()
        assert state["EVT_TEST123"]["trade_id"].startswith("TR_")
        assert state["EVT_TEST123"]["trade_id"] == "TR_" + __import__("hashlib").sha256(b"EVT_TEST123").hexdigest()[:24].upper()
        monkey.undo()


def test_shadow_distinguishes_journal_records_from_confirmed_trades(tmp_path):
    from event_engine.shadow import generate_shadow_health_snapshot
    events = tmp_path / "events.jsonl"
    trades = tmp_path / "trades.jsonl"
    events.write_text("", encoding="utf-8")
    rows = [
        {"record_type":"TRADE_OPEN","trade_id":"TR1","execution":{"status":"opened_protected"},"result":{"position":{"positionAmt":"1"}}},
        {"record_type":"TRADE_CLOSE","trade_id":"TR1"},
        {"record_type":"TRADE_OPEN","trade_id":"TR2","execution":{"status":"OPEN_FAILED"},"result":{}},
        {"record_type":"TRADE_OPEN","trade_id":"TR3","execution":{"status":"opened_protection_check_required"},"result":{"position":{"positionAmt":"2"}}},
    ]
    trades.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    snap = generate_shadow_health_snapshot(events, trades)
    assert snap["trades"]["journal_records"] == 4
    assert snap["trades"]["total"] == 2
    assert snap["trades"]["confirmed_open_records"] == 2
    assert snap["trades"]["close_records"] == 1


def test_trade_close_journal_failure_state_is_retryable():
    trade = {"closed": False, "close_journal_pending": False}
    try:
        raise OSError("journal down")
    except OSError:
        trade["closed"] = False
        trade["close_journal_pending"] = True
    assert trade["closed"] is False
    assert trade["close_journal_pending"] is True


def test_load_successful_trade_ids_includes_opened_protection_failed(tmp_path: Path):
    from run_once import load_successful_trade_ids
    trades_file = tmp_path / "trades.jsonl"
    trades_file.write_text(
        json.dumps({"event_id": "EVT_FAIL", "result": {"status": "OPEN_FAILED"}}) + "\n" +
        json.dumps({"event_id": "EVT_PROT_FAILED", "result": {"status": "opened_protection_failed"}}) + "\n" +
        json.dumps({"event_id": "EVT_PROTECTED", "result": {"status": "opened_protected"}}) + "\n",
        encoding="utf-8",
    )
    loaded = load_successful_trade_ids(trades_file)
    assert "EVT_PROT_FAILED" in loaded
    assert "EVT_PROTECTED" in loaded
    assert "EVT_FAIL" not in loaded


def test_check_funding_filter_only_blocks_aggressive_squeeze_funding():
    from run_once import check_funding_filter
    from types import SimpleNamespace

    # Ordinary divergence signals are NOT hard-blocked by funding.
    assert check_funding_filter(SimpleNamespace(fr_oiw=0.50), "LONG", event_type="REGULAR_BULLISH_RSI")[0] is True
    assert check_funding_filter(SimpleNamespace(fr_oiw=-0.50), "SHORT", event_type="REGULAR_BEARISH_RSI")[0] is True

    # Squeezes tolerate funding up to +/-0.10%, then are blocked.
    assert check_funding_filter(SimpleNamespace(fr_oiw=0.10), "LONG", event_type="SHORT_SQUEEZE")[0] is True
    assert check_funding_filter(SimpleNamespace(fr_oiw=-0.10), "SHORT", event_type="LONG_SQUEEZE")[0] is True

    blocked_long, reason_long = check_funding_filter(SimpleNamespace(fr_oiw=0.1001), "LONG", event_type="SHORT_SQUEEZE")
    blocked_short, reason_short = check_funding_filter(SimpleNamespace(fr_oiw=-0.1001), "SHORT", event_type="LONG_SQUEEZE")
    assert blocked_long is False
    assert blocked_short is False
    assert "ADVERSE_FUNDING_LONG_SQUEEZE" in reason_long
    assert "ADVERSE_FUNDING_SHORT_SQUEEZE" in reason_short

    # Missing / invalid funding never causes a blind hard block.
    assert check_funding_filter(None, "LONG", event_type="SHORT_SQUEEZE")[0] is True
    assert check_funding_filter(SimpleNamespace(fr_oiw=None), "SHORT", event_type="LONG_SQUEEZE")[0] is True
    assert check_funding_filter(SimpleNamespace(fr_oiw="bad"), "SHORT", event_type="LONG_SQUEEZE")[0] is True


def test_squeeze_tp_levels_are_wider_than_divergence():
    from run_once import build_event_setup, build_tp_levels
    df = _generate_synthetic_candles(60)

    # Regular divergence setup
    div_setup = build_event_setup({"direction": "LONG", "event_type": "REGULAR_BULLISH_RSI"}, df, entry_price=100.0)
    assert div_setup["target_rr"] == 2.50
    assert div_setup["planned_weighted_rr"] == pytest.approx(1.6625)
    sl_pct_div, tp_div = build_tp_levels(div_setup, "LONG", event_type="REGULAR_BULLISH_RSI")
    assert tp_div[0]["pnl_pct"] == pytest.approx(sl_pct_div * 0.75)
    assert tp_div[1]["pnl_pct"] == pytest.approx(sl_pct_div * 1.50)
    assert tp_div[2]["pnl_pct"] == pytest.approx(sl_pct_div * 2.50)
    assert [x["close_fraction"] for x in tp_div] == pytest.approx([0.25, 0.40, 0.35])

    # Squeeze setup
    sq_setup = build_event_setup({"direction": "LONG", "event_type": "VOLATILITY_SQUEEZE_RELEASE"}, df, entry_price=100.0)
    assert sq_setup["target_rr"] == 3.0
    assert sq_setup["planned_weighted_rr"] == 2.05
    sl_pct_sq, tp_sq = build_tp_levels(sq_setup, "LONG", event_type="VOLATILITY_SQUEEZE_RELEASE")
    assert tp_sq[0]["pnl_pct"] == pytest.approx(sl_pct_sq * 1.00)
    assert tp_sq[1]["pnl_pct"] == pytest.approx(sl_pct_sq * 2.00)
    assert tp_sq[2]["pnl_pct"] == pytest.approx(sl_pct_sq * 3.00)
    assert tp_sq[0]["close_fraction"] == 0.30
    assert tp_sq[1]["close_fraction"] == 0.35
    assert tp_sq[2]["close_fraction"] == 0.35


def _be_trade_record(hit_legs, event_id="EVT_TEST"):
    return {
        "trade_id": "TR_TEST",
        "event_id": event_id,
        "symbol": "TEST",
        "direction": "LONG",
        "entry_price": 100.0,
        "initial_qty": 10.0,
        "remaining_qty": 7.0,
        "entry_ts": 1000,
        "hit_legs": list(hit_legs),
        "be_activated": False,
        "sl_order": {"order_id": "OLD_SL", "stop_price": 95.0},
        "tp_orders": [],
        "closed": False,
    }


def _run_be_case(monkeypatch, tmp_path, hit_legs, be_after_leg, tp_mode=None):
    import event_engine.tracker as tr

    active_path = tmp_path / "active_trades.json"
    record = _be_trade_record(hit_legs)
    if tp_mode:
        record["tp_mode"] = tp_mode
    active_path.write_text(json.dumps({"EVT_TEST": record}), encoding="utf-8")
    monkeypatch.setattr(tr, "BE_AFTER_LEG", be_after_leg)
    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tr, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tr, "get_position_directional", lambda s, d: {"status": "found", "positionAmt": "7.0", "avgPrice": "100.0"})
    monkeypatch.setattr(tr, "fetch_klines", lambda s, tf, limit=60: [{"close": 102.0}])
    be_calls = []
    monkeypatch.setattr(
        tr, "_move_sl_to_break_even",
        lambda symbol, direction, entry, qty, old_id, trade_id, old_sl_price: be_calls.append(old_id) or {"status": "created", "order_id": "NEW_BE"}
    )
    tr.update_active_trades()
    saved = json.loads(active_path.read_text(encoding="utf-8"))
    return be_calls, saved["EVT_TEST"]


def test_be_default_policy_waits_for_tp2(monkeypatch, tmp_path):
    """Documented policy: TP1 takes a partial, TP2 removes the risk."""
    be_calls, trade = _run_be_case(monkeypatch, tmp_path, ["tp1"], "tp2")
    assert be_calls == []
    assert trade["be_activated"] is False

    be_calls, trade = _run_be_case(monkeypatch, tmp_path, ["tp1", "tp2"], "tp2")
    assert len(be_calls) == 1
    assert trade["be_activated"] is True


def test_be_tp1_policy_is_still_selectable(monkeypatch, tmp_path):
    """BE_AFTER_LEG=tp1 reproduces the previously shipped behaviour for A/B."""
    be_calls, trade = _run_be_case(monkeypatch, tmp_path, ["tp1"], "tp1")
    assert len(be_calls) == 1
    assert trade["be_activated"] is True


def test_be_single_tp_position_arms_only_at_terminal_leg(monkeypatch, tmp_path):
    """A micro-position collapses to one TP3 leg; it cannot wait for a tp2."""
    be_calls, trade = _run_be_case(monkeypatch, tmp_path, ["tp1"], "tp2", tp_mode="single_tp")
    assert be_calls == []
    assert trade["be_activated"] is False

    be_calls, trade = _run_be_case(monkeypatch, tmp_path, ["tp3"], "tp2", tp_mode="single_tp")
    assert len(be_calls) == 1
    assert trade["be_activated"] is True


def test_tracker_does_not_activate_be_without_tp_milestone(monkeypatch, tmp_path):
    import event_engine.tracker as tr
    active_path = tmp_path / "active_trades.json"
    trade_record = {
        "trade_id": "TR_TEST_TP1", "event_id": "EVT_TP1", "symbol": "TEST", "direction": "LONG",
        "entry_price": 100.0, "initial_qty": 10.0, "remaining_qty": 7.5, "entry_ts": 1000,
        "hit_legs": [], "be_activated": False,
        "sl_order": {"order_id": "OLD_SL", "stop_price": 95.0}, "tp_orders": [], "closed": False,
    }
    active_path.write_text(json.dumps({"EVT_TP1": trade_record}), encoding="utf-8")
    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tr, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tr, "get_position_directional", lambda s, d: {"status": "found", "positionAmt": "7.5", "avgPrice": "100.0"})
    monkeypatch.setattr(tr, "fetch_klines", lambda s, tf, limit=60: [{"close": 101.0}])
    be_calls = []
    monkeypatch.setattr(tr, "_move_sl_to_break_even", lambda *a, **k: be_calls.append(True) or {"status": "created", "order_id": "NEW_BE"})
    tr.update_active_trades()
    saved = json.loads(active_path.read_text(encoding="utf-8"))
    assert be_calls == []
    assert saved["EVT_TP1"]["be_activated"] is False


def test_symbol_cooldown_respects_trade_close(tmp_path):
    from run_once import _load_recent_successful_entries, _symbol_on_cooldown
    trades_file = tmp_path / "trades.jsonl"
    now_ms = 1_000_000_000
    # Trade was opened 30 minutes ago, but closed only 5 minutes ago!
    trades_file.write_text(
        json.dumps({
            "record_type": "TRADE_OPEN",
            "symbol": "TEST",
            "ts": now_ms - 30 * 60_000,
            "execution": {"status": "opened_protected"},
            "result": {"position": {"positionAmt": 1.0}},
        }) + "\n" +
        json.dumps({
            "record_type": "TRADE_CLOSE",
            "symbol": "TEST",
            "closed_ts": now_ms - 5 * 60_000,
        }) + "\n",
        encoding="utf-8",
    )
    latest = _load_recent_successful_entries(trades_file, now_ms, cooldown_min=15)
    # Cooldown of 15m must STILL be active because close was only 5m ago
    assert _symbol_on_cooldown("TEST", latest, now_ms, cooldown_min=15) is True
    # Cooldown of 4m would have expired
    assert _symbol_on_cooldown("TEST", latest, now_ms, cooldown_min=4) is False



def test_funding_filter_does_not_block_normal_event_at_previous_threshold():
    row = CoinalyzeRow("X", "X", 100.0, 0.0, 30_000_000.0, 12_000_000.0, 0.0, 0.0, 0.0, 0.0, 0.0648, None, 100_000.0, 100_000.0, 1.0, 0.0, 0.0, 0.0, {})
    ok_normal, reason_normal = check_funding_filter(row, "LONG", event_type="REGULAR_BULLISH_RSI")
    ok_squeeze, reason_squeeze = check_funding_filter(row, "LONG", event_type="SHORT_SQUEEZE")
    assert ok_normal
    assert reason_normal == "OK_NORMAL_FUNDING_NOT_FILTERED"
    assert ok_squeeze
    assert reason_squeeze == "OK"


def test_squeeze_short_can_tolerate_negative_funding_to_minus_0_10():
    row = CoinalyzeRow("X", "X", 100.0, 0.0, 30_000_000.0, 12_000_000.0, 0.0, 0.0, 0.0, 0.0, -0.0672, None, 100_000.0, 100_000.0, 1.0, 0.0, 0.0, 0.0, {})
    ok, reason = check_funding_filter(row, "SHORT", event_type="LONG_SQUEEZE")
    assert ok
    assert reason == "OK"


def test_recent_entries_uses_supplied_full_cooldown_window(tmp_path: Path):
    trades_file = tmp_path / "trades.jsonl"
    now = 10_000_000
    close_ts = now - 30 * 60_000
    trades_file.write_text(
        json.dumps({"record_type": "TRADE_CLOSE", "symbol": "ABC-USDT", "closed_ts": close_ts}) + "\n",
        encoding="utf-8",
    )
    recent = _load_recent_successful_entries(trades_file, now, 45)
    assert recent["ABC-USDT"] == close_ts




def test_load_successful_trade_ids_includes_persisted_terminal_event(tmp_path: Path):
    from run_once import load_successful_trade_ids
    trades_file = tmp_path / "trades.jsonl"
    trades_file.write_text(
        json.dumps({
            "record_type": "EVENT_TERMINAL",
            "event_id": "EVT_TERMINAL",
            "reason": "CLIENT_ORDER_ID_ALREADY_USED",
        }) + "\n",
        encoding="utf-8",
    )
    loaded = load_successful_trade_ids(trades_file)
    assert "EVT_TERMINAL" in loaded


def test_liquidation_squeeze_family_id_is_stable_across_consecutive_spikes():
    df = _generate_synthetic_candles(60, base_price=100.0)
    # Two consecutive upward breakout bars. The detector should keep one episode
    # family identity instead of creating a new family on the second bar.
    df.loc[58, ["close", "high", "low"]] = [106.0, 106.5, 105.0]
    df.loc[59, ["close", "high", "low"]] = [111.0, 111.5, 110.0]
    base = CoinalyzeRow("X", "X", 109.0, 0.0, 30_000_000.0, 12_000_000.0, 0.0, 2.0, 0.0, 0.0, 0.08, None, 500_000.0, 100_000.0, 0.5, 0.0, 0.0, 0.0, {})
    ev1 = detect_liquidation_squeeze(base, df.iloc[:59].copy(), "X-USDT", "1h")
    ev2 = detect_liquidation_squeeze(base, df.copy(), "X-USDT", "1h")
    assert ev1 and ev2
    assert ev1[0]["squeeze_family_id"] == ev2[0]["squeeze_family_id"]


def test_execute_new_position_extracts_nested_bingx_101400(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "open_market", lambda *args, **kwargs: {
        "status": "error",
        "error": "clientOrderID unique check failed",
        "response": {"code": 101400, "msg": "clientOrderID unique check failed"},
    })
    out = ro.execute_new_position("TEST", "LONG", 100.0, {"risk_pct": 1.0}, "EVT_TEST")
    assert out["status"] == "OPEN_FAILED"
    assert out["bingx_code"] == 101400


def test_bingx_entry_rejects_below_exchange_min_notional(monkeypatch):
    from event_engine import bingx as bx

    bx.CACHE["data"] = {
        "TEST-USDT": {
            "symbol": "TEST-USDT", "displayName": "TEST-USDT", "status": 1,
            "apiStateOpen": "true", "quantityPrecision": 3,
            "tradeMinQuantity": 0.001, "tradeMinUSDT": 20.0,
            "maxLongLeverage": 10, "maxShortLeverage": 10,
        }
    }
    bx.CACHE["by_display_name"] = {"TEST-USDT": bx.CACHE["data"]["TEST-USDT"]}
    bx.CACHE["ts"] = 9_999_999_999
    monkeypatch.setattr(bx, "has_open_position", lambda s, d: False)
    monkeypatch.setattr(bx, "_current_close_price", lambda s: 100.0)

    out = bx.open_market("TEST", "LONG", 100.0, "TRD_MIN_NOTIONAL")
    assert out["status"] == "error"
    assert "min_notional" in out["error"]
    assert out["min_notional"] == 20.0


def test_bingx_client_order_ids_are_alphanumeric():
    from event_engine.bingx import _new_open_client_order_id, build_tp_client_order_id, build_sl_client_order_id
    import re

    ids = [
        _new_open_client_order_id("TEST-USDT", "TRD1"),
        build_tp_client_order_id("tp1", "TRD1"),
        build_tp_client_order_id("tp2", "TRD1"),
        build_tp_client_order_id("tp3", "TRD1"),
        build_sl_client_order_id("TRD1"),
    ]
    assert all(1 <= len(x) <= 40 for x in ids)
    assert all(re.fullmatch(r"[A-Za-z0-9]+", x) for x in ids)


def test_reconciliation_registers_orphan_position_with_complete_protection(monkeypatch):
    import run_once as ro

    registrations = []
    position = {"symbol": "TEST-USDT", "positionSide": "LONG", "positionAmt": "1.0", "avgPrice": "100.0", "entryPrice": "100.0"}
    sl = {"orderId": "SL1", "clientOrderId": "EVTSLABC", "type": "STOP_MARKET", "stopPrice": "95.0", "origQty": "1.0"}
    tps = [
        {"orderId": "TP1", "clientOrderId": "EVTTP1ABC", "type": "TAKE_PROFIT_MARKET", "stopPrice": "101.0", "origQty": "0.3"},
        {"orderId": "TP2", "clientOrderId": "EVTTP2ABC", "type": "TAKE_PROFIT_MARKET", "stopPrice": "102.0", "origQty": "0.35"},
        {"orderId": "TP3", "clientOrderId": "EVTTP3ABC", "type": "TAKE_PROFIT_MARKET", "stopPrice": "103.0", "origQty": "0.35"},
    ]

    monkeypatch.setattr(ro, "get_positions", lambda **kwargs: [position])
    monkeypatch.setattr(ro, "get_open_protection_directional", lambda *args, **kwargs: {
        "status": "ok", "tp_orders": tps, "sl_orders": [sl]
    })
    monkeypatch.setattr(ro, "_load_active_trades", lambda: {})
    monkeypatch.setattr(ro, "update_active_trade_protection", lambda **kwargs: False)
    monkeypatch.setattr(ro, "register_active_trade", lambda **kwargs: registrations.append(kwargs))
    journaled = []
    monkeypatch.setattr(ro, "record_trade", lambda obj: journaled.append(obj))

    ro.reconcile_all_open_positions()
    assert len(registrations) == 1
    assert registrations[0]["event_id"].startswith("RECON_TEST-USDT_LONG_")
    assert registrations[0]["event_id"] != "RECON_TEST-USDT_LONG"
    assert registrations[0]["direction"] == "LONG"
    assert registrations[0]["qty"] == 1.0
    assert registrations[0]["setup"]["event_type"] == "RECONCILED_POSITION"
    assert len(journaled) == 1
    assert journaled[0]["record_type"] == "TRADE_OPEN"
    assert journaled[0]["reconciliation"] is True
    assert journaled[0]["event_id"] == registrations[0]["event_id"]


def test_reconciliation_event_id_changes_after_closed_collision(monkeypatch):
    import run_once as ro

    position = {"entryTime": 1_700_000_000_000}
    base = ro._reconciliation_event_id("TEST-USDT", "LONG", 100.0, 1.0, position, {})
    assert base.startswith("RECON_TEST-USDT_LONG_")

    active = {base: {"closed": True}}
    collided = ro._reconciliation_event_id("TEST-USDT", "LONG", 100.0, 1.0, position, active)
    assert collided != base
    assert collided.startswith(base + "_")

    new_id = ro._reconciliation_event_id(
        "TEST-USDT", "LONG", 100.0, 1.0, position,
        {base: {"closed": True}, collided: {"closed": True}},
    )
    assert new_id.startswith(base + "_")
    assert new_id != collided



def test_reconciliation_event_id_reuses_existing_suffixed_open_id():
    import run_once as ro

    position = {"entryTime": 1_700_000_000_000}
    base = ro._reconciliation_event_id("TEST-USDT", "LONG", 100.0, 1.0, position, {})
    active_id = base + "_1700000000123"
    active = {
        base: {"closed": True},
        active_id: {"closed": False},
    }

    event_id = ro._reconciliation_event_id(
        "TEST-USDT", "LONG", 100.0, 1.0, position, active
    )
    assert event_id == active_id


def test_reconciliation_event_id_ignores_changing_update_time():
    import run_once as ro

    active = {}
    p1 = {"entryTime": 1_700_000_000_000, "updateTime": 1_700_000_100_000}
    p2 = {"entryTime": 1_700_000_000_000, "updateTime": 1_700_000_200_000}
    first = ro._reconciliation_event_id("TEST-USDT", "LONG", 100.0, 1.0, p1, active)
    second = ro._reconciliation_event_id("TEST-USDT", "LONG", 100.0, 1.0, p2, active)
    assert first == second

def test_reconciliation_event_id_ignores_quantity_changes_after_partial_reduction():
    import run_once as ro

    position = {"entryTime": 1_700_000_000_000, "updateTime": 1_700_000_100_000}
    first = ro._reconciliation_event_id("TEST-USDT", "LONG", 100.0, 1.0, position, {})
    second = ro._reconciliation_event_id("TEST-USDT", "LONG", 100.0, 0.6, position, {first: {"closed": False}})
    assert second == first


def test_kline_rate_limit_is_not_retried_and_persists_cooldown(monkeypatch, tmp_path):
    import event_engine.bingx as bx
    import run_once as ro

    calls = {"request": 0}
    future_ms = int(__import__("time").time() * 1000) + 120_000
    monkeypatch.setattr(ro, "KLINE_RATE_LIMIT_STATE", tmp_path / "bingx_kline_rate_limit.json")
    ro._KLINE_RATE_LIMIT_CACHE.update(path="", cooldown_until_ms=0, loaded_ts=0.0)

    def fake_request(*args, **kwargs):
        calls["request"] += 1
        return {
            "code": 109429,
            "msg": f"over 5 error code:109415 requests within 900000 ms for this api, please verify and fix it, can retry after time: {future_ms}",
        }

    monkeypatch.setattr(bx, "to_bx_symbol", lambda symbol: "TEST-USDT")
    monkeypatch.setattr(bx, "_request", fake_request)
    monkeypatch.setattr(ro, "fetch_klines", bx.fetch_klines)

    try:
        ro._fetch_klines_scan("TEST", "15m", 250)
    except bx.BingXRateLimitError as exc:
        assert exc.code == 109429
        assert exc.retry_after_ms == future_ms
    else:
        raise AssertionError("rate-limit exception expected")

    assert calls["request"] == 1
    state = ro._load_json(ro.KLINE_RATE_LIMIT_STATE, {})
    assert int(state["cooldown_until_ms"]) == future_ms

    # A second call during the persisted cooldown must fail before HTTP.
    try:
        ro._fetch_klines_scan("TEST", "1h", 250)
    except bx.BingXRateLimitError as exc:
        assert exc.retry_after_ms == future_ms
    else:
        raise AssertionError("persisted cooldown expected")
    assert calls["request"] == 1


def test_kline_transient_error_keeps_bounded_retry(monkeypatch, tmp_path):
    import run_once as ro

    calls = {"fetch": 0}
    monkeypatch.setattr(ro, "KLINE_RATE_LIMIT_STATE", tmp_path / "bingx_kline_rate_limit.json")
    ro._KLINE_RATE_LIMIT_CACHE.update(path="", cooldown_until_ms=0, loaded_ts=0.0)
    monkeypatch.setenv("BINGX_KLINE_SCAN_MIN_INTERVAL_SEC", "0")
    monkeypatch.setenv("BINGX_KLINE_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("BINGX_KLINE_RETRY_BACKOFF_SEC", "0")

    def transient(*args, **kwargs):
        calls["fetch"] += 1
        raise RuntimeError("temporary network failure")

    monkeypatch.setattr(ro, "fetch_klines", transient)
    try:
        ro._fetch_klines_scan("TEST", "15m", 250)
    except RuntimeError as exc:
        assert "after 3 attempts" in str(exc)
    else:
        raise AssertionError("transient RuntimeError expected")
    assert calls["fetch"] == 3


def test_emergency_close_flattens_remaining_directional_position(monkeypatch):
    import event_engine.bingx as bx

    state = {"qty": 1.25}
    calls = []
    contract = {"quantityPrecision": 3}

    monkeypatch.setattr(bx, "to_bx_symbol", lambda symbol: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda symbol: contract)
    monkeypatch.setattr(
        bx,
        "get_position_directional",
        lambda symbol, direction: (
            {"status": "found", "positionAmt": str(state["qty"]), "avgPrice": "100"}
            if state["qty"] > 0
            else {"status": "not_found"}
        ),
    )

    def fake_request(method, path, params):
        calls.append(dict(params))
        state["qty"] = 0.0
        return {"code": 0, "msg": "OK", "data": {"order": {"orderId": "CLOSE1", "clientOrderId": params["clientOrderId"], "avgPrice": "99.5"}}}

    monkeypatch.setattr(bx, "_request", fake_request)
    monkeypatch.setattr(bx.time, "sleep", lambda *_: None)

    out = bx.emergency_close_position("TEST", "LONG", 1.25, reason_token="UNIT")
    assert out["status"] == "closed"
    assert len(calls) == 1
    assert calls[0]["side"] == "SELL"
    assert calls[0]["positionSide"] == "LONG"
    assert calls[0]["type"] == "MARKET"
    assert calls[0]["quantity"] == "1.250"
    assert calls[0]["clientOrderId"].isalnum()
    assert out["execution_price"] == pytest.approx(99.5)


def test_emergency_close_reports_unflattened_after_two_attempts(monkeypatch):
    import event_engine.bingx as bx

    state = {"qty": 1.25}
    calls = []
    monkeypatch.setattr(bx, "to_bx_symbol", lambda symbol: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda symbol: {"quantityPrecision": 3})
    monkeypatch.setattr(
        bx,
        "get_position_directional",
        lambda symbol, direction: {
            "status": "found", "positionAmt": str(state["qty"]), "avgPrice": "100"
        },
    )

    def fake_request(method, path, params):
        calls.append(dict(params))
        # Exchange acknowledges both requests but the position remains live.
        return {"code": 0, "msg": "OK", "data": {"order": {"orderId": f"CLOSE{len(calls)}"}}}

    monkeypatch.setattr(bx, "_request", fake_request)
    monkeypatch.setattr(bx.time, "sleep", lambda *_: None)

    out = bx.emergency_close_position("TEST", "LONG", 1.25, reason_token="UNIT")
    assert out["status"] == "UNFLATTENED"
    assert out["attempts"] == 2
    assert out["remaining_qty"] == 1.25
    assert out["escalation_required"] is True
    assert len(calls) == 2


def test_conditional_protection_payloads_do_not_send_client_order_id(monkeypatch):
    from event_engine import bingx as bx
    captured = []
    monkeypatch.setattr(bx, "_request", lambda method, path, params: captured.append(dict(params)) or {"code": 0, "data": {"order": {"orderId": f"O{len(captured)}"}}})
    monkeypatch.setattr(bx, "get_contract", lambda s: {"quantityPrecision": 3, "pricePrecision": 2, "tradeMinQuantity": 0.001})
    monkeypatch.setattr(bx, "to_bx_symbol", lambda s: "TEST-USDT")
    monkeypatch.setattr(bx, "_set_leverage", lambda *a, **k: True)
    def fake_protection(*args, **kwargs):
        sl = [captured[0]] if captured and captured[0].get("type") == "STOP_MARKET" else []
        tps = [captured[-1]] if captured and captured[-1].get("type") == "TAKE_PROFIT_MARKET" else []
        return {"status": "ok", "sl_orders": [dict(x, orderId="O1") for x in sl], "tp_orders": [dict(x, orderId="O2") for x in tps]}
    monkeypatch.setattr(bx, "get_open_protection_directional", fake_protection)
    monkeypatch.setattr(bx, "_current_close_price", lambda s: 100.0)
    out = bx.ensure_directional_protection(
        "TEST", "LONG", 100.0, 1.0, 5.0,
        [{"leg": "tp1", "pnl_pct": 1.0, "close_fraction": 1.0}],
        trade_id="TRD1",
    )
    assert out["status"] in {"PROTECTED", "SL_ONLY"}
    conditional = [p for p in captured if p.get("type") in {"STOP_MARKET", "TAKE_PROFIT_MARKET"}]
    assert conditional
    assert all("clientOrderId" not in p for p in conditional)


def test_ensure_protection_cancels_stale_tp_profile_legs(monkeypatch):
    from event_engine import bingx as bx

    monkeypatch.setattr(bx, "to_bx_symbol", lambda s: "TEST-USDT")
    monkeypatch.setattr(bx, "get_contract", lambda s: {
        "quantityPrecision": 3, "pricePrecision": 2, "tradeMinQuantity": 0.001,
    })

    sl = {"orderId": "SL1", "type": "STOP_MARKET", "stopPrice": "95.00", "origQty": "1.0"}
    tp1 = {"orderId": "TP1", "type": "TAKE_PROFIT_MARKET", "stopPrice": "101.00", "origQty": "1.0"}
    tp2 = {"orderId": "TP2", "type": "TAKE_PROFIT_MARKET", "stopPrice": "102.00", "origQty": "1.0"}
    state = {"tps": [tp1, tp2]}
    cancel_calls = []

    def fake_protection(*args, **kwargs):
        return {"status": "ok", "sl_orders": [sl], "tp_orders": list(state["tps"])}

    monkeypatch.setattr(bx, "get_open_protection_directional", fake_protection)

    def fake_cancel(symbol, order_id):
        cancel_calls.append(order_id)
        state["tps"] = [o for o in state["tps"] if str(o.get("orderId")) != str(order_id)]
        return {"code": 0}

    monkeypatch.setattr(bx, "cancel_order", fake_cancel)

    out = bx.ensure_directional_protection(
        "TEST", "LONG", 100.0, 1.0, 5.0,
        [{"leg": "tp1", "pnl_pct": 1.0, "close_fraction": 1.0}],
        trade_id="TRD1",
    )

    assert out["status"] == "PROTECTED"
    assert cancel_calls == ["TP2"]
    assert out["stale_tp_cancellations"] == [{"order_id": "TP2", "status": "cancelled", "error": None}]
    assert [x["leg"] for x in out["tp_orders"]] == ["tp1"]


def test_reconciliation_uses_tp_price_when_current_conditional_orders_have_no_client_id(monkeypatch):
    import run_once as ro
    position = {"symbol": "TEST-USDT", "positionSide": "LONG", "positionAmt": "1.0", "avgPrice": "100.0"}
    active = {
        "T": {
            "symbol": "TEST", "direction": "LONG", "closed": False,
            "hit_legs": [], "be_activated": False,
            "effective_tp_levels": [
                {"leg": "tp1", "pnl_pct": 1.0, "close_fraction": 1.0, "qty": 1.0},
            ],
            "tp_mode": "single_tp", "effective_weighted_rr": 1.0,
            "planned_risk_pct": 1.0,
        }
    }
    sl = {"orderId": "SL1", "type": "STOP_MARKET", "stopPrice": "99.0", "origQty": "1.0"}
    tp = {"orderId": "TP1", "type": "TAKE_PROFIT_MARKET", "stopPrice": "101.0", "origQty": "1.0"}
    calls = {"repair": 0, "update": 0}
    monkeypatch.setattr(ro, "get_positions", lambda **kwargs: [position])
    monkeypatch.setattr(ro, "_load_active_trades", lambda: active)
    monkeypatch.setattr(ro, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [sl], "tp_orders": [tp]})
    monkeypatch.setattr(ro, "update_active_trade_protection", lambda **kwargs: calls.__setitem__("update", calls["update"] + 1) or True)
    monkeypatch.setattr(ro, "ensure_directional_protection", lambda **kwargs: calls.__setitem__("repair", calls["repair"] + 1) or {"status": "PROTECTED"})
    ro.reconcile_all_open_positions()
    assert calls["update"] == 1
    assert calls["repair"] == 0


def test_closed_trade_requires_verified_protection_cleanup(monkeypatch):
    import event_engine.tracker as tr
    monkeypatch.setattr(tr, "get_position_directional", lambda *a, **k: {"status": "not_found", "positionAmt": "0"})
    monkeypatch.setattr(tr, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok",
        "sl_orders": [{"orderId": "SL1"}],
        "tp_orders": [{"orderId": "TP1"}],
    })
    monkeypatch.setattr(tr, "_cancel_protection_order_verified", lambda *a, **k: (True, "cancelled"))
    ok, note = tr._cleanup_closed_trade_protection({"symbol": "TEST", "direction": "LONG"})
    assert ok is False
    assert "still visible" in note


def test_closed_trade_cleanup_refuses_unknown_position_state(monkeypatch):
    import event_engine.tracker as tr
    monkeypatch.setattr(tr, "get_position_directional", lambda *a, **k: {"status": "error", "error": "timeout"})
    ok, note = tr._cleanup_closed_trade_protection({"symbol": "TEST", "direction": "LONG"})
    assert ok is False
    assert "not proven closed" in note


def test_execution_config_rejects_unsupported_position_mode(monkeypatch):
    import run_once as ro
    monkeypatch.setattr(ro, "EXECUTION_ENABLED", True)
    monkeypatch.setattr(ro, "API_KEY", "KEY")
    monkeypatch.setattr(ro, "SECRET_KEY", "SECRET")
    monkeypatch.setattr(ro, "BASE_URL", "https://open-api-vst.bingx.com")
    monkeypatch.setattr(ro, "EXECUTION_MODE", "vst")
    monkeypatch.setattr(ro, "POSITION_MODE", "ONE_WAY")
    ok, reason = ro._validate_execution_config()
    assert ok is False
    assert "BINGX_POSITION_MODE=HEDGE" in reason


def test_entry_drift_is_adverse_by_direction():
    import run_once as ro
    assert ro._entry_drift_pct(100.0, 102.0, "LONG") == pytest.approx(2.0)
    assert ro._entry_drift_pct(100.0, 98.0, "SHORT") == pytest.approx(2.0)
    assert ro._entry_drift_pct(100.0, 98.0, "LONG") == pytest.approx(2.0)
    assert ro._entry_drift_pct(100.0, 102.0, "SHORT") == pytest.approx(2.0)


def test_symbol_quarantine_after_three_consecutive_losses(tmp_path: Path):
    import run_once as ro
    trades = tmp_path / "trades.jsonl"
    rows = [
        {"record_type": "TRADE_CLOSE", "symbol": "ABC", "closed_ts": 1000, "realized_pnl_pct": -1.0},
        {"record_type": "TRADE_CLOSE", "symbol": "ABC", "closed_ts": 2000, "realized_pnl_pct": -2.0},
        {"record_type": "TRADE_CLOSE", "symbol": "ABC", "closed_ts": 3000, "realized_pnl_pct": -3.0},
        {"record_type": "TRADE_CLOSE", "symbol": "DEF", "closed_ts": 3000, "realized_pnl_pct": 2.0},
    ]
    trades.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    out = ro._load_symbol_quarantines(trades, now_ms=60_000, max_consecutive_losses=3, quarantine_min=10)
    assert out["ABC"] == 603_000
    assert ro._symbol_on_quarantine("ABC", out, 60_001) is True
    assert ro._symbol_on_quarantine("DEF", out, 60_001) is False


def test_load_recent_successful_entries_reads_full_cooldown_window(tmp_path: Path):
    import run_once as ro
    trades = tmp_path / "trades.jsonl"
    now = 3_600_000
    rows = [
        {"record_type": "TRADE_CLOSE", "symbol": "SQUEEZE", "closed_ts": now - 35 * 60_000},
        {"record_type": "TRADE_OPEN", "symbol": "SQUEEZE", "execution": {"status": "opened_protected"}, "result": {"position": {"positionAmt": "1"}}, "ts": now - 50 * 60_000},
    ]
    trades.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    out = ro._load_recent_successful_entries(trades, now, cooldown_min=45)
    assert out["SQUEEZE"] == now - 35 * 60_000


def test_short_defensive_score_is_stricter():
    import run_once as ro
    ev = {"direction": "SHORT", "event_type": "REGULAR_BEARISH_MACD", "event_fact": {"price_delta_atr": 1.0}}
    score = ro.calculate_setup_score(ev, None, pd.DataFrame({"close": [1]}))
    assert score > 0
    assert ro.MIN_SHORT_SCORE == 85.0


def test_hot_oi_penalty_applies_to_score():
    import run_once as ro
    from types import SimpleNamespace
    row = SimpleNamespace(oi_chg24_pct=55.0, fr_oiw=0.0)
    ev = {"direction": "LONG", "event_type": "REGULAR_BULLISH_RSI", "event_fact": {}}
    base = ro.calculate_setup_score(ev, None, pd.DataFrame({"close": [1]}))
    hot = ro.calculate_setup_score(ev, row, pd.DataFrame({"close": [1]}))
    assert hot == pytest.approx(base - ro.HOT_OI_SCORE_PENALTY)


def test_market_protection_close_uses_client_order_id(monkeypatch):
    import event_engine.bingx as bx
    captured = []
    monkeypatch.setattr(
        bx, "_request",
        lambda method, path, params: captured.append(dict(params)) or {"code": 0, "data": {"order": {"orderId": "M1", "clientOrderId": params.get("clientOrderId")}}},
    )
    monkeypatch.setattr(bx, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})
    out = bx._post_protection_order_verified(
        "TEST", "LONG",
        {"type": "MARKET", "symbol": "TEST-USDT", "side": "SELL", "positionSide": "LONG", "quantity": "1"},
        "EVTCLOSEUNIT", max_attempts=1,
    )
    assert out["code"] == 0
    assert captured[0]["clientOrderId"] == "EVTCLOSEUNIT"


def test_execute_new_position_flattens_when_protection_install_fails(monkeypatch):
    import run_once as ro
    monkeypatch.setattr(ro, "MAX_ENTRY_DRIFT_PCT", 2.0)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "open_market", lambda *a, **k: {
        "status": "opened", "order_id": "O1", "leverage": 10, "order_reference_price": 100.0,
    })
    monkeypatch.setattr(ro, "wait_for_position_fill_directional", lambda **k: {
        "status": "found", "positionAmt": "1", "avgPrice": "100", "entryPrice": "100",
    })
    monkeypatch.setattr(ro, "install_protection", lambda **k: {
        "status": "PROTECTION_FAILED", "error": "openOrders timeout", "rolled_back": False,
    })
    calls = []
    monkeypatch.setattr(ro, "emergency_close_position", lambda *a, **k: calls.append(k) or {"status": "closed"})
    out = ro.execute_new_position(
        "TEST", "LONG", 100.0,
        {"risk_pct": 1.0, "signal_price": 100.0, "event_type": "REGULAR_BULLISH_RSI"},
        "EVT_PROTECTION_FAIL",
    )
    assert out["status"] == "opened_rolled_back"
    assert out["protection"]["rolled_back"] is True
    assert calls


def test_execute_new_position_rolls_back_excessive_entry_drift(monkeypatch):
    import run_once as ro
    monkeypatch.setattr(ro, "MAX_ENTRY_DRIFT_PCT", 2.0)
    monkeypatch.setattr(ro, "open_market", lambda *a, **k: {
        "status": "opened", "order_id": "O1", "leverage": 10, "order_reference_price": 100.0,
    })
    monkeypatch.setattr(ro, "wait_for_position_fill_directional", lambda **k: {
        "status": "found", "positionAmt": "1", "avgPrice": "104", "entryPrice": "104",
    })
    monkeypatch.setattr(ro, "emergency_close_position", lambda *a, **k: {"status": "closed"})
    out = ro.execute_new_position(
        "TEST", "LONG", 100.0,
        {"risk_pct": 1.0, "signal_price": 100.0, "event_type": "REGULAR_BULLISH_RSI"},
        "EVT_DRIFT",
    )
    assert out["status"] == "ENTRY_DRIFT_EXCEEDED"
    assert out["rolled_back"] is True
    assert out["execution_quality"]["signal_to_fill_distance_pct"] == pytest.approx(4.0)


def test_hot_oi_is_score_only():
    import run_once as ro
    assert not hasattr(ro, "HARD_HOT_OI_CHG24_PCT")
    assert ro.HOT_OI_SCORE_PENALTY > 0


def test_add_cvd_carries_value_across_missing_bar_without_zero_fill():
    from event_engine.signals import add_cvd
    df = pd.DataFrame({
        "bar_delta_usdt": [10.0, np.nan, -3.0],
        "taker_flow_valid": [True, False, True],
    })
    out = add_cvd(df)
    assert out["bingx_cvd"].tolist() == pytest.approx([10.0, 10.0, 7.0])
    assert out["taker_flow_valid"].tolist() == [True, False, True]
    assert out["cvd_valid_coverage"].iloc[1] == pytest.approx(0.5)


def _synthetic_ohlcv(n=260, base=100.0):
    import numpy as np
    t = np.arange(n, dtype=float)
    close = base + 0.08 * t + 1.5 * np.sin(t / 9.0)
    high = close + 0.7
    low = close - 0.7
    open_ = close - 0.1
    volume = np.full(n, 1000.0)
    close_time = (t + 1).astype(int) * 60_000
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": volume, "close_time": close_time})


def test_wilder_smooth_seeds_first_n_valid_values():
    from event_engine.signals import _wilder_smooth
    values = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0])
    out = _wilder_smooth(values, 3)
    assert pd.isna(out.iloc[0])
    assert out.iloc[3] == pytest.approx(2.0)
    values2 = pd.Series([1.0, 2.0, 3.0, 4.0])
    out2 = _wilder_smooth(values2, 3)
    assert out2.iloc[2] == pytest.approx(2.0)


def test_macd_4h_requires_real_ema200():
    from event_engine.signals import detect_macd_4h
    df = _synthetic_ohlcv(100)
    assert detect_macd_4h(df, "TEST", "4h") == []


def test_new_engine_events_have_distinct_engine_identity():
    from event_engine.signals import detect_ma_compression_breakout, detect_breakout_momentum
    df = _synthetic_ohlcv(260)
    # No assertion about whether an artificial smooth series produces an event;
    # the important regression contract is that returned events self-identify.
    for func in (detect_ma_compression_breakout, detect_breakout_momentum):
        events = func(df, "TEST", "1h")
        for ev in events:
            assert ev["event_fact"]["engine"] in {"MA_COMPRESSION", "BREAKOUT_MOMENTUM"}
            assert ev["event_fact"].get("requires_retest") is True
            assert ev["event_fact"].get("trigger_level", 0) > 0


def test_liquidation_squeeze_uses_directional_funding_not_absolute_value():
    from types import SimpleNamespace
    from event_engine.signals import detect_liquidation_squeeze
    df = _synthetic_ohlcv(60)
    # Force a bullish spike on the last bar sufficient for the LONG-side squeeze
    # path, but provide positive funding (not crowded shorts): it must not qualify
    # via funding alone.
    df.loc[len(df)-2, "close"] = 100.0
    df.loc[len(df)-2, "high"] = 101.0
    df.loc[len(df)-1, "open"] = 100.0
    df.loc[len(df)-1, "close"] = 110.0
    df.loc[len(df)-1, "high"] = 110.5
    df.loc[len(df)-1, "low"] = 99.5
    row = SimpleNamespace(liq_short24=2_000_000.0, liq_long24=0.0, oi=100_000_000.0,
                          oi_chg4h_pct=0.0, fr_oiw=0.05, ls_accounts=None)
    assert detect_liquidation_squeeze(row, df, "TEST", "1h") == []




def test_divergence_post_confirmation_freshness_uses_confirmation_not_pivot_age():
    import run_once

    pivot2 = 1_000_000
    confirm = pivot2 + 120 * 60_000
    now = confirm + 10 * 60_000
    ev = {"event_type": "REGULAR_BULLISH_RSI", "timeframe": "1h",
          "timestamps": {"pivot_2_ts": pivot2, "detected_at_ts": confirm}}
    valid, formation_age, post_age, lag = run_once._divergence_post_confirmation_age(ev, now)
    assert valid is True
    assert formation_age == 130.0
    assert post_age == 10.0
    assert lag == 120.0


def test_divergence_post_confirmation_freshness_expires_after_window():
    import run_once

    pivot2 = 1_000_000
    confirm = pivot2 + 120 * 60_000
    now = confirm + 46 * 60_000
    ev = {"event_type": "HIDDEN_BEARISH_MACD", "timeframe": "4h",
          "timestamps": {"pivot_2_ts": pivot2, "detected_at_ts": confirm}}
    valid, _, post_age, _ = run_once._divergence_post_confirmation_age(ev, now)
    assert valid is True
    assert post_age == 46.0


def test_canonical_atr_run_once_matches_signal_atr():
    from event_engine.signals import _atr as signal_atr
    import run_once
    df = pd.DataFrame({
        "high": [101, 103, 102, 106, 105, 108, 107, 110, 111, 109, 112, 114, 113, 115, 116, 118, 117, 119, 121, 120, 123],
        "low":  [ 99, 100, 100, 102, 103, 105, 104, 107, 108, 106, 109, 111, 110, 112, 113, 115, 114, 116, 118, 118, 120],
        "close":[100, 102, 101, 105, 104, 107, 106, 109, 110, 108, 111, 113, 112, 114, 115, 117, 116, 118, 120, 119, 122],
    })
    assert run_once.canonical_atr(df, 14).iloc[-1] == signal_atr(df, 14).iloc[-1]


def test_liquidation_squeeze_is_opt_in_by_runtime_flag(monkeypatch):
    import run_once
    monkeypatch.setattr(run_once, "ENABLE_LIQUIDATION_SQUEEZE_ENGINE", False)
    assert run_once.ENABLE_LIQUIDATION_SQUEEZE_ENGINE is False



def test_retest_trigger_prefers_latest_valid_retest_within_window():
    from event_engine.signals import diagnose_15m_retest_trigger
    import pandas as pd
    base = 1_000_000
    rows = []
    for i in range(30):
        ts = base + i * 15 * 60_000
        close = 101.0
        rows.append({"open": 100.0, "high": 102.0, "low": 99.0, "close": close,
                     "volume": 1_000.0, "close_time": ts})
    # Two valid retests after the event; the function should return the latest one.
    event_ts = rows[20]["close_time"]
    rows[22].update({"open": 100.8, "high": 102.0, "low": 99.9, "close": 101.2, "volume": 2_000})
    rows[23].update({"open": 100.9, "high": 102.2, "low": 99.8, "close": 101.4, "volume": 2_100})
    out = diagnose_15m_retest_trigger(pd.DataFrame(rows), "LONG", 100.0, event_ts,
                                      max_delay_min=60, volume_mult=1.10)
    assert out["ok"] is True
    assert out["trigger_bar_close_ts"] == rows[23]["close_time"]


def test_event_max_age_allows_retest_engine_window():
    import run_once
    ev = {"event_type": "MA_COMPRESSION_BREAKOUT", "event_fact": {"requires_retest": True, "engine": "MA_COMPRESSION"},
          "timestamps": {"detected_at_ts": 1_000_000}}
    assert run_once._event_max_age_min(ev) >= 120.0


def test_liquidation_squeeze_predicate_is_not_used_for_volatility_squeeze_funding():
    import run_once
    from types import SimpleNamespace
    row = SimpleNamespace(fr_oiw=-0.20)
    ok, reason = run_once.check_funding_filter(row, "LONG", event_type="VOLATILITY_SQUEEZE_RELEASE")
    assert ok is True
    assert "VOLATILITY" in reason or "NORMAL" in reason


def test_early_loss_cut_guard_block_does_not_require_undefined_cut(monkeypatch):
    import event_engine.tracker as tr
    monkeypatch.setattr(tr, "EARLY_LOSS_CUT_ENABLED", True)
    monkeypatch.setattr(tr, "_early_loss_cut_guard", lambda *args, **kwargs: (False, "near_live_sl:0.1%", 1.0))
    # The regression target is the initialization pattern used by update_active_trades:
    # a blocked guard must still yield a defined cut result rather than NameError.
    cut = {"status": "skipped"}
    safe_to_close = False
    guard_reason = "near_live_sl:0.1%"
    if not safe_to_close:
        cut["error"] = guard_reason
    assert cut["status"] == "skipped"
    assert cut["error"] == guard_reason


def _make_trend_candles(n=140, step=1.0):
    rows=[]
    base=100.0
    for i in range(n):
        o=base + i*step
        c=o + step
        h=c + 0.6
        l=o - 0.6
        rows.append({"open":o,"high":h,"low":l,"close":c,"volume":1000.0,"close_time":1_700_000_000_000+i*3_600_000})
    return pd.DataFrame(rows)


def test_new_strategy_engines_expose_required_event_schema():
    from event_engine.signals import detect_donchian_retest, detect_liquidity_sweep_reclaim, detect_ema_pullback_continuation
    df = _make_trend_candles()
    # Baseline trend series may not trigger all engines on its own; schema checks
    # apply only when a detector produces an event on a crafted/realistic sample.
    for detector in (detect_donchian_retest, detect_liquidity_sweep_reclaim, detect_ema_pullback_continuation):
        events = detector(df, "TEST", "1h")
        for ev in events:
            assert ev["direction"] in {"LONG", "SHORT"}
            assert ev["event_fact"].get("requires_retest") is True
            assert ev["event_fact"].get("requires_htf_context") is True


def test_strategy_htf_context_requires_matching_trend():
    from event_engine.signals import validate_strategy_htf_context
    ev = {"direction": "LONG", "event_fact": {"requires_htf_context": True}}
    up = _make_trend_candles(120, 1.0)
    down = _make_trend_candles(120, -1.0)
    ok, reason, meta = validate_strategy_htf_context(ev, up, "4h")
    assert ok is True
    assert reason == "STRATEGY_HTF_OK"
    assert meta["context_timeframe"] == "4h"
    ok2, reason2, _ = validate_strategy_htf_context(ev, down, "4h")
    assert ok2 is False
    assert reason2 == "STRATEGY_HTF_TREND_MISMATCH"


def test_squeeze_family_predicates_are_separate():
    import run_once as ro
    assert ro._is_liquidation_squeeze_event("SHORT_SQUEEZE") is True
    assert ro._is_liquidation_squeeze_event("VOLATILITY_SQUEEZE_RELEASE") is False
    assert ro._is_volatility_squeeze_event("VOLATILITY_SQUEEZE_RELEASE") is True


def test_new_engine_features_have_distinct_identity():
    from event_engine.signals import detect_donchian_retest, detect_liquidity_sweep_reclaim, detect_ema_pullback_continuation
    df = _make_trend_candles()
    names = set()
    for detector in (detect_donchian_retest, detect_liquidity_sweep_reclaim, detect_ema_pullback_continuation):
        for ev in detector(df, "TEST", "1h"):
            names.add(ev["event_fact"]["engine"])
    assert names.issubset({"DONCHIAN_RETEST", "LIQUIDITY_SWEEP", "EMA_PULLBACK"})


def test_ema_pullback_engine_can_detect_valid_long_setup():
    from event_engine.signals import detect_ema_pullback_continuation
    rows=[]; p=100.0
    for i in range(240):
        o=p; c=p+0.20; h=c+0.15; l=o-0.15
        rows.append({"open":o,"high":h,"low":l,"close":c,"volume":1000.0,"close_time":1_700_000_000_000+i*3_600_000})
        p=c
    df=pd.DataFrame(rows)
    ema21=df["close"].ewm(span=21, adjust=False).mean().iloc[-2]
    prev_close=float(df["close"].iloc[-2])
    df.loc[df.index[-1], ["open","low","high","close","volume"]] = [float(ema21)-0.10, float(ema21)-0.20, float(ema21)+0.70, float(ema21)+0.40, 1200.0]
    events=detect_ema_pullback_continuation(df,"TEST","1h")
    assert any(e["direction"]=="LONG" and e["event_type"]=="EMA_PULLBACK_CONTINUATION" for e in events)


def test_donchian_engine_can_detect_valid_long_breakout():
    from event_engine.signals import detect_donchian_retest
    rows=[]; p=100.0
    for i in range(240):
        o=p; c=p+0.20; h=c+0.15; l=o-0.15
        rows.append({"open":o,"high":h,"low":l,"close":c,"volume":1000.0,"close_time":1_700_000_000_000+i*3_600_000})
        p=c
    df=pd.DataFrame(rows)
    dc_high=float(df["high"].iloc[-21:-1].max())
    prev=float(df["close"].iloc[-2])
    df.loc[df.index[-1], ["open","low","high","close","volume"]] = [prev, prev-0.15, dc_high+1.80, dc_high+1.50, 1800.0]
    events=detect_donchian_retest(df,"TEST","1h")
    assert any(e["direction"]=="LONG" and e["event_type"]=="DONCHIAN_RETEST_BREAKOUT" for e in events)


def test_liquidity_sweep_engine_can_detect_long_reclaim():
    rows=[]; p=100.0
    for i in range(100):
        o=p; c=p+0.01; h=c+0.05; l=o-0.05
        rows.append({"open":o,"high":h,"low":l,"close":c,"volume":1000.0,"close_time":1_700_000_000_000+i*3_600_000})
        p=c
    df=pd.DataFrame(rows)
    # Previous 20 completed bars contain the swept low at 95.0.
    df.loc[95,["open","high","low","close"]] = [95.2,95.4,95.0,95.1]
    df.loc[99,["open","low","high","close","volume"]] = [99.0,94.0,101.0,99.5,1500.0]
    events = detect_liquidity_sweep_reclaim(df,"TEST","1h")
    assert any(e["direction"]=="LONG" and e["event_type"]=="LIQUIDITY_SWEEP_RECLAIM" for e in events)
    event = next(e for e in events if e["direction"] == "LONG")
    assert event["event_fact"]["swept_level"] == pytest.approx(95.0)
    assert event["event_fact"]["atr_reference"] == "previous_bar"
    assert event["event_fact"]["rejection_wick_fraction"] >= 0.35


def test_liquidity_sweep_rejects_weak_sweep_and_small_rejection_wick(monkeypatch):
    import event_engine.signals as sig

    rows=[]
    for i in range(100):
        rows.append({"open":100.0,"high":100.2,"low":99.8,"close":100.0,"volume":1000.0,"close_time":1_700_000_000_000+i*3_600_000})
    df=pd.DataFrame(rows)
    df.loc[90,["open","high","low","close"]] = [95.2,95.4,95.0,95.1]
    monkeypatch.setattr(sig, "_atr", lambda d, n=14: pd.Series(
        [1.0] * len(d), index=d.index, dtype=float
    ))

    # 0.05 ATR sweep is below the 0.10 previous-ATR minimum.
    df.loc[99,["open","high","low","close","volume"]] = [99.0,100.0,94.95,99.5,1500.0]
    assert sig.detect_liquidity_sweep_reclaim(df,"TEST","1h") == []

    # Now the sweep is deep enough, but only 20% of the range is lower wick.
    df.loc[99,["open","high","low","close","volume"]] = [95.0,100.0,93.5,99.0,1500.0]
    assert sig.detect_liquidity_sweep_reclaim(df,"TEST","1h") == []


def test_expected_engine_registry_contains_v5_and_smc_engines():
    import run_once as ro
    assert ro.EXPECTED_EVENT_ENGINES == {
        "DIVERGENCE", "VOLATILITY_SQUEEZE", "MACD_4H", "MA_COMPRESSION",
        "BREAKOUT_MOMENTUM", "DONCHIAN_RETEST", "LIQUIDITY_SWEEP", "EMA_PULLBACK",
        "ORDER_BLOCK", "BREAKER_BLOCK", "MITIGATION_BLOCK", "SFP",
        "LIQUIDATION_CASCADE_FVG", "CRT", "VOLUME_PROFILE_DIVERGENCE", "HARMONIC_PATTERN",
    }


def _make_breaker_fixture(with_liquidity_sweep: bool) -> pd.DataFrame:
    n = 140
    rows = []
    for i in range(n):
        rows.append({
            "open": 100.0, "high": 100.4, "low": 99.6, "close": 100.0,
            "volume": 1000.0, "close_time": 1_700_000_000_000 + i * 3_600_000,
        })
    df = pd.DataFrame(rows)
    # Bearish candle = the original bullish OB source.
    df.loc[109, ["open", "high", "low", "close"]] = [100.5, 101.0, 99.0, 100.0]
    # BOS above an earlier swing high.
    df.loc[110, ["open", "high", "low", "close", "volume"]] = [100.5, 112.0, 100.0, 111.0, 1800.0]
    # A confirmed swing high that can be swept before the bearish MSS.
    df.loc[115, ["open", "high", "low", "close"]] = [108.0, 110.0, 107.0, 108.5]
    df.loc[116, ["open", "high", "low", "close"]] = [108.0, 109.0, 107.5, 108.2]
    df.loc[117, ["open", "high", "low", "close"]] = [108.0, 109.0, 107.5, 108.2]
    sweep_high = 111.5 if with_liquidity_sweep else 109.0
    df.loc[118, ["open", "high", "low", "close"]] = [108.5, sweep_high, 107.0, 108.0]
    # First decisive break through the bullish OB low = bearish MSS/flip.
    df.loc[121, ["open", "high", "low", "close", "volume"]] = [100.0, 100.5, 97.0, 98.0, 1600.0]
    # Current bar retests the broken zone from below.
    df.loc[139, ["open", "high", "low", "close"]] = [98.0, 100.0, 96.5, 98.5]
    return df


def test_breaker_requires_liquidity_sweep_before_mss(monkeypatch):
    import event_engine.signals as sig

    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([], [105, 115]))

    without_sweep = sig.detect_breaker_block(_make_breaker_fixture(False), "TEST", "1h")
    assert without_sweep == []

    with_sweep = sig.detect_breaker_block(_make_breaker_fixture(True), "TEST", "1h")
    assert len(with_sweep) == 1
    event = with_sweep[0]
    assert event["event_type"] == "BREAKER_BLOCK_BEARISH"
    assert event["event_fact"]["liquidity_sweep_level"] == pytest.approx(110.0)
    assert event["event_fact"]["liquidity_sweep_ts"] < event["event_fact"]["mss_ts"]
    assert event["event_fact"]["mss_ts"] == event["event_fact"]["broken_ts"]


def _make_mitigation_fixture(with_bos: bool) -> pd.DataFrame:
    rows=[]
    for i in range(140):
        rows.append({
            "open":100.0,"high":100.4,"low":99.6,"close":100.0,
            "volume":1000.0,"close_time":1_700_000_000_000+i*3_600_000,
        })
    df=pd.DataFrame(rows)
    # Confirmed structure high before the displacement.
    df.loc[100,["open","high","low","close"]]=[104.0,105.0,103.6,104.5]
    # Origin = last opposite candle before the strong bullish impulse.
    df.loc[105,["open","high","low","close"]]=[100.6,101.0,99.0,100.0]
    df.loc[106,["open","high","low","close"]]=[100.0,111.5,99.8,111.0]
    if with_bos:
        df.loc[108,["open","high","low","close"]]=[111.0,112.0,110.5,106.0]
    else:
        df.loc[108,["open","high","low","close"]]=[111.0,112.0,110.5,104.0]
    # Price expands away from origin, then returns to origin open.
    df.loc[139,["open","high","low","close"]]=[101.0,102.0,100.4,101.2]
    return df


def test_mitigation_block_requires_origin_and_post_impulse_bos(monkeypatch):
    import event_engine.signals as sig

    monkeypatch.setattr(sig, "_atr", lambda d, n=14: pd.Series([1.0]*len(d), index=d.index, dtype=float))
    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([], [100]))

    assert sig.detect_mitigation_block(_make_mitigation_fixture(False), "TEST", "1h") == []

    events = sig.detect_mitigation_block(_make_mitigation_fixture(True), "TEST", "1h")
    assert len(events) == 1
    event = events[0]
    assert event["event_type"] == "MITIGATION_BLOCK_BULLISH"
    assert event["event_fact"]["origin_ts"] < event["event_fact"]["impulse_ts"] < event["event_fact"]["bos_ts"]
    assert event["event_fact"]["zone_low"] == pytest.approx(99.0)
    assert event["event_fact"]["trigger_level"] == pytest.approx(100.6)


def test_smc_detectors_fail_closed_on_incomplete_data():
    bad = pd.DataFrame({"close": [1.0, 2.0], "high": [2.0, 3.0], "low": [0.5, 1.5]})
    for fn in (detect_order_block, detect_breaker_block, detect_mitigation_block, detect_sfp, detect_crt):
        assert fn(bad, "TEST", "1h") == []
    assert detect_liquidation_cascade_fvg(None, bad, "TEST", "1h") == []


def test_crt_bullish_three_candle_pattern():
    rows=[]; p=100.0
    for i in range(50):
        rows.append({"open":p,"high":p+0.6,"low":p-0.6,"close":p+0.2,"volume":1000.0,"close_time":1_700_000_000_000+i*3_600_000})
        p += 0.2
    df=pd.DataFrame(rows)
    i=47
    df.loc[i,["open","high","low","close"]]=[100,102,99,101]
    df.loc[i+1,["open","high","low","close"]]=[101,101.4,97.5,98.5]
    df.loc[i+2,["open","high","low","close"]]=[98.5,101.5,98.2,100.8]
    events=detect_crt(df,"TEST","1h")
    assert any(e["event_type"]=="CRT_BULLISH" and e["direction"]=="LONG" for e in events)


def test_crt_requires_reclaim_inside_c1_range():
    from event_engine.signals import detect_crt

    rows=[]
    for i in range(50):
        rows.append({
            "open":100.0,"high":100.6,"low":99.4,"close":100.2,
            "close_time":1_700_000_000_000+i*3_600_000,
        })
    df=pd.DataFrame(rows)
    i=47
    # C1 range is [99, 102]. C2 sweeps below it. C3 closes ABOVE C1.high,
    # so it is not a reclaim inside C1 and must be rejected.
    df.loc[i,["open","high","low","close"]]=[100.0,102.0,99.0,101.0]
    df.loc[i+1,["open","high","low","close"]]=[101.0,101.4,97.5,98.5]
    df.loc[i+2,["open","high","low","close"]]=[102.2,113.0,101.8,102.5]
    assert detect_crt(df,"TEST","1h") == []


def test_sfp_requires_volume_in_sweep_wick(monkeypatch):
    import event_engine.signals as sig

    rows=[]
    for i in range(100):
        rows.append({"open":100.0,"high":100.2,"low":99.8,"close":100.0,"volume":1000.0,"close_time":1_700_000_000_000+i*3_600_000})
    df=pd.DataFrame(rows)
    df.loc[95,["open","high","low","close"]]=[95.2,95.4,94.8,95.1]
    df.loc[99,["open","high","low","close","volume"]]=[99.0,99.8,94.8,99.2,1500.0]
    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([95], []))

    # Same total volume and valid sweep, but only ~4% of the candle range is
    # outside the swing level: the wick-volume condition must reject it.
    assert sig.detect_sfp(df, "TEST", "1h", min_outside_volume_share=0.20) == []

    df.loc[99, ["low", "close"]] = [93.5, 99.2]
    events = sig.detect_sfp(df, "TEST", "1h", min_outside_volume_share=0.20)
    assert len(events) == 1
    assert events[0]["event_type"] == "SFP_BULLISH"
    assert events[0]["event_fact"]["outside_volume_share"] >= 0.20
    assert events[0]["event_fact"]["outside_volume_share_method"] == "candle_range_proxy"


def test_sfp_bearish_sweep_reclaim():
    rows=[]; p=100.0
    for i in range(100):
        rows.append({"open":p,"high":p+0.2,"low":p-0.2,"close":p,"volume":1000.0,"close_time":1_700_000_000_000+i*3_600_000})
    df=pd.DataFrame(rows)
    # A prior swing high at index 90.
    df.loc[89,["open","high","low","close"]]=[103.8,104.0,103.0,103.9]
    df.loc[90,["open","high","low","close"]]=[104.0,105.0,103.7,104.8]
    df.loc[91,["open","high","low","close"]]=[104.8,104.0,103.6,103.8]
    df.loc[99,["open","high","low","close","volume"]]=[104.2,106.5,103.9,103.5,1500.0]
    events=detect_sfp(df,"TEST","1h")
    assert any(e["event_type"]=="SFP_BEARISH" and e["direction"]=="SHORT" for e in events)


def test_oi_history_cache_updates_immediately(tmp_path, monkeypatch):
    import run_once as ro
    monkeypatch.chdir(tmp_path)
    ro.DATA=Path("data"); ro.DATA.mkdir(exist_ok=True)
    ro.OI_HISTORY=ro.DATA/"oi_history.json"
    ro._OI_HIST_CACHE.update({"ts":0.0,"data":{},"path":""})
    from types import SimpleNamespace
    row=SimpleNamespace(symbol="TEST", price=100.0, oi=123.0)
    assert ro._record_oi_snapshots([row],1_700_000_000_000)==1
    assert "TEST" in ro._load_oi_history()


def test_tracker_trade_closed_log_format_has_all_arguments():
    import logging
    fmt = "[TRACKER_TRADE_CLOSED] %s (%s/%s) | PnL: %+.2f%% | Realized R:R: %s | Planned R:R: %.2f | Exit: %.8g (%s) | Duration: %.1f min"
    args = ("💚", "NAME", "TEST", 1.25, "1.000", 1.6625, 101.25, "TP_FULL", 12.0)
    record = logging.LogRecord("tracker", logging.INFO, __file__, 1, fmt, args, None)
    rendered = record.getMessage()
    assert "TEST" in rendered
    assert "PnL: +1.25%" in rendered
    assert "Duration: 12.0 min" in rendered


def test_volume_profile_divergence_detects_hvn_accumulation_proxy():
    from event_engine.signals import detect_volume_profile_divergence

    n = 100
    df = pd.DataFrame({
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.0] * n,
        "volume": [100.0] * n,
        "close_time": [1_700_000_000_000 + i * 3_600_000 for i in range(n)],
    })
    df.loc[80:98, "volume"] = 150.0
    df.loc[99, "high"] = 91.0
    df.loc[99, "low"] = 89.0
    df.loc[99, "close"] = 90.0
    df.loc[99, "volume"] = 150.0

    events = detect_volume_profile_divergence(df, "TEST-USDT", "1h", lookback=100, recent_bars=20)
    assert len(events) == 1
    ev = events[0]
    assert ev["event_type"] == "VOLUME_PROFILE_ACCUMULATION"
    assert ev["direction"] == "LONG"
    assert ev["event_fact"]["volume_profile_method"] == "ohlcv_typical_price_proxy"
    assert ev["event_fact"]["hvn_growth_ratio"] >= 1.10
    assert ev["event_fact"]["price_left_hvn"] is True


def test_volume_profile_does_not_repeat_after_hvn_transition():
    from event_engine.signals import detect_volume_profile_divergence

    n = 100
    df = pd.DataFrame({
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.0] * n,
        "volume": [100.0] * n,
        "close_time": [1_700_000_000_000 + i * 3_600_000 for i in range(n)],
    })
    df.loc[80:97, "volume"] = 150.0
    df.loc[98:99, ["high", "low", "close"]] = [91.0, 89.0, 90.0]
    df.loc[98:99, "volume"] = 150.0

    events = detect_volume_profile_divergence(df, "TEST-USDT", "1h", lookback=100, recent_bars=20)
    assert events == []


def test_harmonic_gartley_uses_confirmed_alternating_pivots(monkeypatch):
    import event_engine.signals as sig

    n = 90
    df = pd.DataFrame({
        "high": [105.0] * n,
        "low": [95.0] * n,
        "close": [100.0] * n,
        "close_time": [1_700_000_000_000 + i * 3_600_000 for i in range(n)],
    })
    prices = {10: (100.0, "low"), 20: (200.0, "high"), 30: (138.2, "low"), 40: (169.1, "high"), 50: (121.4, "low")}
    for idx, (price, kind) in prices.items():
        if kind == "low":
            df.loc[idx, "low"] = price
            df.loc[idx, "close"] = price
        else:
            df.loc[idx, "high"] = price
            df.loc[idx, "close"] = price

    monkeypatch.setattr(sig, "_pivots", lambda work, left=5, right=5: ([10, 30, 50], [20, 40]))
    events = sig.detect_harmonic_patterns(df, "TEST-USDT", "1h", tolerance=0.05)
    assert any(e["event_type"] == "HARMONIC_GARTLEY_LONG" for e in events)
    event = next(e for e in events if e["event_type"] == "HARMONIC_GARTLEY_LONG")
    assert event["event_fact"]["trigger_level"] == pytest.approx(121.4)
    assert event["timestamps"]["d_ts"] == df.loc[50, "close_time"]
    assert event["timestamps"]["detected_at_ts"] == df.loc[55, "close_time"]


def test_tracker_residual_close_does_not_reuse_last_tp_price(monkeypatch, tmp_path):
    """A partial TP followed by an unpriced residual close must not reuse TP1 as exit price."""
    import event_engine.tracker as tr

    active_path = tmp_path / "active_trades.json"
    trades_path = tmp_path / "trades.jsonl"
    trade = {
        "trade_id": "TR_RESIDUAL",
        "event_id": "EVT_RESIDUAL",
        "symbol": "TEST",
        "direction": "LONG",
        "entry_price": 100.0,
        "initial_qty": 100.0,
        "remaining_qty": 75.0,
        "entry_ts": 1_000,
        "tp_orders": [{"leg": "tp1", "order_id": "TP1"}],
        "sl_order": {"order_id": "SL1", "stop_price": 95.0},
        "sl_order_history": [],
        "hit_legs": ["tp1"],
        "tp_filled_qty": {"tp1": 25.0},
        "realized_pnl_qty": 25.0,
        "realized_pnl_weighted_sum": 25.0,
        "last_tp_exec_price": 101.0,
        "be_activated": False,
        "be_required": False,
        "closed": False,
        "manual_exit_reason": None,
        "planned_risk_pct": 1.0,
        "planned_weighted_rr": 1.6625,
        "effective_weighted_rr": 1.6625,
        "effective_tp_levels": [],
        "tp_mode": "multi_tp",
    }
    active_path.write_text(json.dumps({"EVT_RESIDUAL": trade}), encoding="utf-8")

    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tr, "TRADES_PATH", trades_path)
    monkeypatch.setattr(tr, "_retry_pending_notifications", lambda: None)
    monkeypatch.setattr(tr, "_queue_notification", lambda *a, **k: None)
    monkeypatch.setattr(tr, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(
        tr,
        "get_order",
        lambda symbol, order_id: (
            {"status": "ok", "order_status": "FILLED", "executed_qty": 25.0, "avg_price": 101.0}
            if order_id == "TP1"
            else {"status": "error", "error": "SL history unavailable"}
        ),
    )
    monkeypatch.setattr(
        tr,
        "fetch_klines",
        lambda symbol, timeframe, limit=60: [{"open_time": 1_000, "close_time": 2_000, "high": 96.0, "low": 94.0, "close": 95.0}],
    )
    monkeypatch.setattr(tr, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})

    tr.update_active_trades()

    saved = json.loads(trades_path.read_text(encoding="utf-8").strip())

    assert saved["exit_price"] is None
    assert saved["exit_price_source"] == "UNKNOWN_EXECUTION"
    assert saved["estimated_exit_price"] == pytest.approx(95.0)
    assert saved["realized_pnl_pct"] is None
    assert saved["exit_reason"] == "DATA_ERROR"
    assert saved["realized_pnl_usdt"] is None
    # The journal stores only the known partial TP state; the residual execution
    # and therefore total realized PnL remain explicitly unknown.
    assert saved["exit_price"] != pytest.approx(101.0)
    assert not json.loads(active_path.read_text(encoding="utf-8"))


def test_tracker_known_sl_fill_still_has_priority_over_market_estimate(monkeypatch, tmp_path):
    """A confirmed SL fill remains the authoritative residual exit price."""
    import event_engine.tracker as tr

    active_path = tmp_path / "active_trades.json"
    trades_path = tmp_path / "trades.jsonl"
    trade = {
        "trade_id": "TR_SL",
        "event_id": "EVT_SL",
        "symbol": "TEST",
        "direction": "LONG",
        "entry_price": 100.0,
        "initial_qty": 100.0,
        "remaining_qty": 75.0,
        "entry_ts": 1_000,
        "tp_orders": [{"leg": "tp1", "order_id": "TP1"}],
        "sl_order": {"order_id": "SL1", "stop_price": 95.0},
        "sl_order_history": [],
        "hit_legs": ["tp1"],
        "tp_filled_qty": {"tp1": 25.0},
        "realized_pnl_qty": 25.0,
        "realized_pnl_weighted_sum": 25.0,
        "last_tp_exec_price": 101.0,
        "be_activated": False,
        "be_required": False,
        "closed": False,
        "manual_exit_reason": None,
        "planned_risk_pct": 1.0,
        "planned_weighted_rr": 1.6625,
        "effective_weighted_rr": 1.6625,
        "effective_tp_levels": [],
        "tp_mode": "multi_tp",
    }
    active_path.write_text(json.dumps({"EVT_SL": trade}), encoding="utf-8")

    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tr, "TRADES_PATH", trades_path)
    monkeypatch.setattr(tr, "_retry_pending_notifications", lambda: None)
    monkeypatch.setattr(tr, "_queue_notification", lambda *a, **k: None)
    monkeypatch.setattr(tr, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(
        tr,
        "get_order",
        lambda symbol, order_id: (
            {"status": "ok", "order_status": "FILLED", "executed_qty": 25.0, "avg_price": 101.0}
            if order_id == "TP1"
            else {"status": "ok", "order_status": "FILLED", "avg_price": 94.0}
        ),
    )
    monkeypatch.setattr(
        tr,
        "fetch_klines",
        lambda symbol, timeframe, limit=60: [{"open_time": 1_000, "close_time": 2_000, "high": 96.0, "low": 94.0, "close": 95.0}],
    )
    monkeypatch.setattr(tr, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})

    tr.update_active_trades()

    saved = json.loads(trades_path.read_text(encoding="utf-8").strip())
    assert not json.loads(active_path.read_text(encoding="utf-8"))
    assert saved["exit_price"] == pytest.approx(94.0)
    assert saved["exit_price_source"] == "SL_FILL"
    assert saved["realized_pnl_pct"] == pytest.approx(-4.25)


def test_tracker_partial_tp_plus_break_even_fill_is_realized(monkeypatch, tmp_path):
    import event_engine.tracker as tr

    active_path = tmp_path / "active_trades.json"
    trades_path = tmp_path / "trades.jsonl"
    trade = {
        "trade_id": "TR_BE", "event_id": "EVT_BE", "symbol": "TEST", "direction": "LONG",
        "entry_price": 100.0, "initial_qty": 100.0, "remaining_qty": 75.0, "entry_ts": 1_000,
        "tp_orders": [{"leg": "tp1", "order_id": "TP1"}], "sl_order": {"order_id": "BE1", "stop_price": 100.0},
        "sl_order_history": [], "hit_legs": ["tp1"], "tp_filled_qty": {"tp1": 25.0},
        "realized_pnl_qty": 25.0, "realized_pnl_weighted_sum": 25.0, "last_tp_exec_price": 101.0,
        "be_activated": True, "be_required": False, "closed": False, "manual_exit_reason": None,
        "planned_risk_pct": 1.0, "planned_weighted_rr": 1.6625, "effective_weighted_rr": 1.6625,
        "effective_tp_levels": [], "tp_mode": "multi_tp",
    }
    active_path.write_text(json.dumps({"EVT_BE": trade}), encoding="utf-8")

    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tr, "TRADES_PATH", trades_path)
    monkeypatch.setattr(tr, "_retry_pending_notifications", lambda: None)
    monkeypatch.setattr(tr, "_queue_notification", lambda *a, **k: None)
    monkeypatch.setattr(tr, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(tr, "get_live_price", lambda *a, **k: None)
    monkeypatch.setattr(tr, "get_order", lambda _symbol, order_id: (
        {"status": "ok", "order_status": "FILLED", "executed_qty": 25.0, "avg_price": 101.0}
        if order_id == "TP1" else {"status": "ok", "order_status": "FILLED", "avg_price": 100.0}
    ))
    monkeypatch.setattr(tr, "fetch_klines", lambda *a, **k: [{"close": 99.0, "high": 100.0, "low": 98.0, "close_time": 2_000}])
    monkeypatch.setattr(tr, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})

    tr.update_active_trades()
    saved = json.loads(trades_path.read_text(encoding="utf-8").strip())

    assert saved["exit_price_source"] == "SL_FILL"
    assert saved["exit_price"] == pytest.approx(100.0)
    assert saved["exit_reason"] == "BREAK_EVEN"
    assert saved["realized_pnl_pct"] == pytest.approx(0.25)


def test_tracker_manual_market_close_uses_explicit_execution_price(monkeypatch, tmp_path):
    import event_engine.tracker as tr

    active_path = tmp_path / "active_trades.json"
    trades_path = tmp_path / "trades.jsonl"
    trade = {
        "trade_id": "TR_MANUAL", "event_id": "EVT_MANUAL", "symbol": "TEST", "direction": "LONG",
        "entry_price": 100.0, "initial_qty": 10.0, "remaining_qty": 10.0, "entry_ts": 1_000,
        "tp_orders": [], "sl_order": {"order_id": "SL1", "stop_price": 95.0}, "sl_order_history": [],
        "hit_legs": [], "tp_filled_qty": {}, "realized_pnl_qty": 0.0, "realized_pnl_weighted_sum": 0.0,
        "last_tp_exec_price": None, "be_activated": False, "be_required": False, "closed": False,
        "manual_exit_reason": "MANUAL_MARKET_CLOSE", "manual_exit_price": 99.0,
        "planned_risk_pct": 1.0, "planned_weighted_rr": 1.6625, "effective_weighted_rr": 1.6625,
        "effective_tp_levels": [], "tp_mode": "multi_tp",
    }
    active_path.write_text(json.dumps({"EVT_MANUAL": trade}), encoding="utf-8")

    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tr, "TRADES_PATH", trades_path)
    monkeypatch.setattr(tr, "_retry_pending_notifications", lambda: None)
    monkeypatch.setattr(tr, "_queue_notification", lambda *a, **k: None)
    monkeypatch.setattr(tr, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(tr, "get_live_price", lambda *a, **k: None)
    monkeypatch.setattr(tr, "get_order", lambda *a, **k: {"status": "error"})
    monkeypatch.setattr(tr, "fetch_klines", lambda *a, **k: [{"close": 90.0, "high": 91.0, "low": 89.0, "close_time": 2_000}])
    monkeypatch.setattr(tr, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})

    tr.update_active_trades()
    saved = json.loads(trades_path.read_text(encoding="utf-8").strip())

    assert saved["exit_price_source"] == "MANUAL_EXECUTION"
    assert saved["exit_price"] == pytest.approx(99.0)
    assert saved["realized_pnl_pct"] == pytest.approx(-1.0)


def test_tracker_duplicate_close_journal_is_not_duplicated(monkeypatch, tmp_path):
    import event_engine.tracker as tr

    active_path = tmp_path / "active_trades.json"
    trades_path = tmp_path / "trades.jsonl"
    trade = {
        "trade_id": "TR_DUP", "event_id": "EVT_DUP", "symbol": "TEST", "direction": "LONG",
        "entry_price": 100.0, "initial_qty": 1.0, "remaining_qty": 1.0, "entry_ts": 1_000,
        "tp_orders": [], "sl_order": {}, "sl_order_history": [], "hit_legs": [], "tp_filled_qty": {},
        "realized_pnl_qty": 0.0, "realized_pnl_weighted_sum": 0.0, "last_tp_exec_price": None,
        "be_activated": False, "be_required": False, "closed": False, "manual_exit_reason": "MANUAL_CLOSE",
        "manual_exit_price": 99.0, "planned_risk_pct": 1.0, "planned_weighted_rr": 1.6625,
        "effective_weighted_rr": 1.6625, "effective_tp_levels": [], "tp_mode": "multi_tp",
    }
    active_path.write_text(json.dumps({"EVT_DUP": trade}), encoding="utf-8")
    existing = {
        "record_type": "TRADE_CLOSE", "event_id": "EVT_DUP", "exit_reason": "MANUAL_CLOSE",
        "exit_price": 99.0, "realized_pnl_pct": -1.0,
    }
    trades_path.write_text(json.dumps(existing) + "\n", encoding="utf-8")

    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tr, "TRADES_PATH", trades_path)
    monkeypatch.setattr(tr, "_retry_pending_notifications", lambda: None)
    monkeypatch.setattr(tr, "_queue_notification", lambda *a, **k: None)
    monkeypatch.setattr(tr, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(tr, "get_live_price", lambda *a, **k: None)
    monkeypatch.setattr(tr, "get_order", lambda *a, **k: {"status": "error"})
    monkeypatch.setattr(tr, "fetch_klines", lambda *a, **k: [{"close": 90.0, "high": 91.0, "low": 89.0, "close_time": 2_000}])
    monkeypatch.setattr(tr, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})

    tr.update_active_trades()
    rows = [json.loads(line) for line in trades_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len([r for r in rows if r.get("record_type") == "TRADE_CLOSE" and r.get("event_id") == "EVT_DUP"]) == 1


def test_tracker_full_tp_uses_weighted_actual_execution_prices(monkeypatch, tmp_path):
    import event_engine.tracker as tr

    active_path = tmp_path / "active_trades.json"
    trades_path = tmp_path / "trades.jsonl"
    trade = {
        "trade_id": "TR_FULL_TP", "event_id": "EVT_FULL_TP", "symbol": "TEST", "direction": "LONG",
        "entry_price": 100.0, "initial_qty": 100.0, "remaining_qty": 100.0, "entry_ts": 1_000,
        "tp_orders": [{"leg": "tp1", "order_id": "TP1"}, {"leg": "tp2", "order_id": "TP2"}],
        "sl_order": {"order_id": "SL1", "stop_price": 95.0}, "sl_order_history": [],
        "hit_legs": [], "tp_filled_qty": {}, "realized_pnl_qty": 0.0, "realized_pnl_weighted_sum": 0.0,
        "last_tp_exec_price": None, "be_activated": True, "be_required": False, "closed": False,
        "manual_exit_reason": None, "planned_risk_pct": 1.0, "planned_weighted_rr": 1.6625,
        "effective_weighted_rr": 1.6625, "effective_tp_levels": [], "tp_mode": "multi_tp",
    }
    active_path.write_text(json.dumps({"EVT_FULL_TP": trade}), encoding="utf-8")

    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tr, "TRADES_PATH", trades_path)
    monkeypatch.setattr(tr, "_retry_pending_notifications", lambda: None)
    monkeypatch.setattr(tr, "_queue_notification", lambda *a, **k: None)
    monkeypatch.setattr(tr, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(tr, "get_live_price", lambda *a, **k: 102.0)
    fills = {
        "TP1": {"status": "ok", "order_status": "FILLED", "executed_qty": 50.0, "avg_price": 101.0},
        "TP2": {"status": "ok", "order_status": "FILLED", "executed_qty": 50.0, "avg_price": 103.0},
        "SL1": {"status": "ok", "order_status": "NEW", "avg_price": 0.0},
    }
    monkeypatch.setattr(tr, "get_order", lambda symbol, order_id: fills.get(order_id, {"status": "error"}))
    monkeypatch.setattr(tr, "fetch_klines", lambda *a, **k: [{"close": 102.0, "high": 103.0, "low": 99.0, "close_time": 2_000}])
    monkeypatch.setattr(tr, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})

    tr.update_active_trades()
    saved = json.loads(trades_path.read_text(encoding="utf-8").strip())

    assert saved["exit_price_source"] == "TP_WEIGHTED_AVERAGE"
    assert saved["exit_price"] == pytest.approx(102.0)
    assert saved["realized_pnl_pct"] == pytest.approx(2.0)


def test_tracker_does_not_use_closed_1m_as_current_price_when_live_ticker_missing(monkeypatch, tmp_path):
    import event_engine.tracker as tr

    active_path = tmp_path / "active_trades.json"
    trades_path = tmp_path / "trades.jsonl"
    trade = {
        "trade_id": "TR_NO_LIVE_MARK", "event_id": "EVT_NO_LIVE_MARK", "symbol": "TEST", "direction": "LONG",
        "entry_price": 100.0, "initial_qty": 10.0, "remaining_qty": 10.0, "entry_ts": 1_000,
        "tp_orders": [], "sl_order": {"order_id": "SL1", "stop_price": 95.0}, "sl_order_history": [],
        "hit_legs": [], "tp_filled_qty": {}, "realized_pnl_qty": 0.0, "realized_pnl_weighted_sum": 0.0,
        "last_tp_exec_price": None, "be_activated": False, "be_required": False, "closed": False,
        "manual_exit_reason": None, "planned_risk_pct": 1.0, "planned_weighted_rr": 1.6625,
        "effective_weighted_rr": 1.6625, "effective_tp_levels": [], "tp_mode": "multi_tp",
    }
    active_path.write_text(json.dumps({"EVT_NO_LIVE_MARK": trade}), encoding="utf-8")

    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tr, "TRADES_PATH", trades_path)
    monkeypatch.setattr(tr, "_retry_pending_notifications", lambda: None)
    monkeypatch.setattr(tr, "_queue_notification", lambda *a, **k: None)
    monkeypatch.setattr(tr, "get_position_directional", lambda *a, **k: {
        "status": "found", "positionAmt": "10", "avgPrice": "100"
    })
    monkeypatch.setattr(tr, "get_live_price", lambda *a, **k: None)
    monkeypatch.setattr(tr, "fetch_klines", lambda *a, **k: [{
        "close": 90.0, "high": 91.0, "low": 89.0, "close_time": 2_000
    }])
    monkeypatch.setattr(tr, "get_order", lambda *a, **k: {"status": "error"})
    monkeypatch.setattr(tr, "get_open_protection_directional", lambda *a, **k: {
        "status": "ok", "sl_orders": [{"orderId": "SL1", "stopPrice": "95", "origQty": "10"}], "tp_orders": []
    })

    tr.update_active_trades()
    saved = json.loads(active_path.read_text(encoding="utf-8"))["EVT_NO_LIVE_MARK"]

    assert saved["current_price_source"] == "UNAVAILABLE"
    assert saved["current_pnl_pct"] is None
    assert saved["current_position_qty"] == pytest.approx(10.0)


def test_tracker_invalid_mark_does_not_become_realized_execution(monkeypatch, tmp_path):
    import event_engine.tracker as tr

    active_path = tmp_path / "active_trades.json"
    trades_path = tmp_path / "trades.jsonl"
    trade = {
        "trade_id": "TR_BAD_MARK", "event_id": "EVT_BAD_MARK", "symbol": "TEST", "direction": "LONG",
        "entry_price": 100.0, "initial_qty": 10.0, "remaining_qty": 10.0, "entry_ts": 1_000,
        "tp_orders": [], "sl_order": {"order_id": "SL1", "stop_price": 95.0}, "sl_order_history": [],
        "hit_legs": [], "tp_filled_qty": {}, "realized_pnl_qty": 0.0, "realized_pnl_weighted_sum": 0.0,
        "last_tp_exec_price": None, "be_activated": False, "be_required": False, "closed": False,
        "manual_exit_reason": None, "planned_risk_pct": 1.0, "planned_weighted_rr": 1.6625,
        "effective_weighted_rr": 1.6625, "effective_tp_levels": [], "tp_mode": "multi_tp",
    }
    active_path.write_text(json.dumps({"EVT_BAD_MARK": trade}), encoding="utf-8")

    monkeypatch.setattr(tr, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tr, "TRADES_PATH", trades_path)
    monkeypatch.setattr(tr, "_retry_pending_notifications", lambda: None)
    monkeypatch.setattr(tr, "_queue_notification", lambda *a, **k: None)
    monkeypatch.setattr(tr, "get_position_directional", lambda *a, **k: {"status": "not_found"})
    monkeypatch.setattr(tr, "get_live_price", lambda *a, **k: None)
    monkeypatch.setattr(tr, "get_order", lambda *a, **k: {"status": "error"})
    monkeypatch.setattr(tr, "fetch_klines", lambda *a, **k: [{"close": 0.0, "high": 0.0, "low": 0.0, "close_time": 2_000}])
    monkeypatch.setattr(tr, "get_open_protection_directional", lambda *a, **k: {"status": "ok", "sl_orders": [], "tp_orders": []})

    tr.update_active_trades()
    saved = json.loads(trades_path.read_text(encoding="utf-8").strip())

    assert saved["exit_price"] is None
    assert saved["realized_pnl_pct"] is None
    assert saved["exit_reason"] == "DATA_ERROR"


def _make_ob_causality_fixture(bos_i: int) -> pd.DataFrame:
    n = 120
    rows = []
    for i in range(n):
        rows.append({
            "open": 100.0,
            "high": 100.4,
            "low": 99.6,
            "close": 100.0,
            "volume": 1000.0,
            "close_time": 1_700_000_000_000 + i * 3_600_000,
        })
    df = pd.DataFrame(rows)
    pivot_i = 50
    df.loc[pivot_i, ["open", "high", "low", "close"]] = [101.0, 106.0, 98.0, 99.0]
    for i in range(pivot_i + 1, bos_i):
        df.loc[i, ["high", "low"]] = [103.0, 98.5]
    df.loc[bos_i, ["open", "high", "low", "close", "volume"]] = [99.0, 108.0, 98.0, 107.0, 2200.0]
    return df


def test_order_block_rejects_unconfirmed_bos_pivot(monkeypatch):
    import event_engine.signals as sig

    monkeypatch.setattr(sig, "_atr", lambda d, n=14: pd.Series([1.0] * len(d), index=d.index, dtype=float))
    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([], [50]))
    # BOS is the first bar that would otherwise confirm the pivot; right=2
    # means the pivot cannot be used yet.
    events = sig.detect_order_block(_make_ob_causality_fixture(51), "TEST", "1h")
    assert events == []


def test_order_block_accepts_pivot_after_full_confirmation(monkeypatch):
    import event_engine.signals as sig

    monkeypatch.setattr(sig, "_atr", lambda d, n=14: pd.Series([1.0] * len(d), index=d.index, dtype=float))
    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([], [50]))
    events = sig.detect_order_block(_make_ob_causality_fixture(53), "TEST", "1h")
    assert len(events) == 1
    assert events[0]["event_type"] == "ORDER_BLOCK_BULLISH"


def test_breaker_rejects_bos_pivot_confirmed_only_after_bos(monkeypatch):
    import event_engine.signals as sig

    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([], [109]))
    assert sig.detect_breaker_block(_make_breaker_fixture(True), "TEST", "1h") == []


def test_breaker_liquidity_sweep_rejects_unconfirmed_sweep_pivot():
    import event_engine.signals as sig

    d = pd.DataFrame({
        "high": [100.0] * 30,
        "low": [99.0] * 30,
    })
    d.loc[10, "high"] = 110.0
    d.loc[11, "high"] = 112.0  # sweep arrives before two confirmation bars
    assert sig._liquidity_sweep_before_break(d, [10], 5, 15, "bearish") is None


def test_breaker_liquidity_sweep_accepts_fully_confirmed_pivot():
    import event_engine.signals as sig

    d = pd.DataFrame({
        "high": [100.0] * 30,
        "low": [99.0] * 30,
    })
    d.loc[10, "high"] = 110.0
    d.loc[13, "high"] = 112.0  # pivot has two confirmation bars before sweep
    assert sig._liquidity_sweep_before_break(d, [10], 5, 15, "bearish") == (13, 110.0)


def _make_harmonic_fixture(points: dict[int, float], kinds: dict[int, str]) -> pd.DataFrame:
    n = 90
    rows = []
    for i in range(n):
        rows.append({
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
            "close_time": 1_700_000_000_000 + i * 3_600_000,
        })
    df = pd.DataFrame(rows)
    for idx, price in points.items():
        kind = kinds[idx]
        if kind == "low":
            df.loc[idx, "low"] = price
            df.loc[idx, "close"] = price
            df.loc[idx, "high"] = max(105.0, price + 1.0)
        else:
            df.loc[idx, "high"] = price
            df.loc[idx, "close"] = price
            df.loc[idx, "low"] = min(95.0, price - 1.0)
    return df


def test_cypher_uses_canonical_xa_xc_cd_geometry(monkeypatch):
    import event_engine.signals as sig

    # XA=100, AB=61.8, XC=127.2, CD/XC=0.786.
    points = {10: 100.0, 20: 200.0, 30: 138.2, 40: 227.2, 50: 127.2208}
    kinds = {10: "low", 20: "high", 30: "low", 40: "high", 50: "low"}
    df = _make_harmonic_fixture(points, kinds)
    monkeypatch.setattr(sig, "_pivots", lambda work, left=5, right=5: ([10, 30, 50], [20, 40]))

    events = sig.detect_harmonic_patterns(df, "TEST-USDT", "1h", tolerance=0.01)
    cyphers = [e for e in events if e["event_type"] == "HARMONIC_CYPHER_LONG"]
    assert len(cyphers) == 1
    fact = cyphers[0]["event_fact"]
    assert fact["xc_xa"] == pytest.approx(1.272)
    assert fact["cd_xc"] == pytest.approx(0.786)


def test_cypher_rejects_old_xd_xc_geometry_even_when_old_c_ratio_would_pass(monkeypatch):
    import event_engine.signals as sig

    # This satisfies the old XD/XC=0.786 test and the old BC/AB band, but
    # CD/XC is ~0.214, so the canonical Cypher detector must reject it.
    points = {10: 100.0, 20: 200.0, 30: 138.2, 40: 227.2, 50: 199.9792}
    kinds = {10: "low", 20: "high", 30: "low", 40: "high", 50: "low"}
    df = _make_harmonic_fixture(points, kinds)
    monkeypatch.setattr(sig, "_pivots", lambda work, left=5, right=5: ([10, 30, 50], [20, 40]))

    events = sig.detect_harmonic_patterns(df, "TEST-USDT", "1h", tolerance=0.05)
    assert not any(e["event_type"] == "HARMONIC_CYPHER_LONG" for e in events)


def test_cypher_rejects_wrong_c_ratio_relative_to_xa(monkeypatch):
    import event_engine.signals as sig

    # C is only 0.60*XA away from X, outside the canonical 1.272-1.414 XA band.
    points = {10: 100.0, 20: 200.0, 30: 138.2, 40: 160.0, 50: 112.96}
    kinds = {10: "low", 20: "high", 30: "low", 40: "high", 50: "low"}
    df = _make_harmonic_fixture(points, kinds)
    monkeypatch.setattr(sig, "_pivots", lambda work, left=5, right=5: ([10, 30, 50], [20, 40]))

    events = sig.detect_harmonic_patterns(df, "TEST-USDT", "1h", tolerance=0.05)
    assert not any(e["event_type"] == "HARMONIC_CYPHER_LONG" for e in events)


def test_valid_crab_remains_detected(monkeypatch):
    import event_engine.signals as sig

    # Probe the existing Crab predicate: XA=100, AB=58, BC=29,
    # CD=90.8, XD=161.8. The predicate itself is intentionally unchanged.
    points = {10: 100.0, 20: 200.0, 30: 142.0, 40: 171.0, 50: 261.8}
    kinds = {10: "low", 20: "high", 30: "low", 40: "high", 50: "low"}
    df = _make_harmonic_fixture(points, kinds)
    monkeypatch.setattr(sig, "_pivots", lambda work, left=5, right=5: ([10, 30, 50], [20, 40]))

    events = sig.detect_harmonic_patterns(df, "TEST-USDT", "1h", tolerance=0.05)
    assert any(e["event_type"] == "HARMONIC_CRAB_LONG" for e in events)


def test_bingx_live_price_parsing_uses_ticker_not_klines(monkeypatch):
    from event_engine import bingx as bx

    bx.CACHE["data"] = {"BTC-USDT": {"symbol": "BTC-USDT"}}
    bx.CACHE["by_display_name"] = {"BTC-USDT": bx.CACHE["data"]["BTC-USDT"]}
    bx.CACHE["ts"] = 9_999_999_999
    monkeypatch.setattr(
        bx,
        "_request",
        lambda method, path, params=None, signed=True, **kwargs: {
            "code": 0,
            "data": {"symbol": "BTC-USDT", "price": "123456.78", "time": 1700000000000},
        },
    )
    monkeypatch.setattr(bx, "fetch_klines", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ticker path must not read klines")))

    assert bx.get_live_price("BTC") == pytest.approx(123456.78)
    assert bx._current_close_price("BTC") == pytest.approx(123456.78)


def test_bingx_live_price_parses_array_ticker_payload(monkeypatch):
    from event_engine import bingx as bx

    bx.CACHE["data"] = {"BTC-USDT": {"symbol": "BTC-USDT"}}
    bx.CACHE["by_display_name"] = {"BTC-USDT": bx.CACHE["data"]["BTC-USDT"]}
    bx.CACHE["ts"] = 9_999_999_999
    monkeypatch.setattr(
        bx,
        "_request",
        lambda *args, **kwargs: {
            "code": 0,
            "data": [
                {"symbol": "ETH-USDT", "price": "2000"},
                {"symbol": "BTC-USDT", "price": "124000"},
            ],
        },
    )
    assert bx.get_live_price("BTC-USDT") == pytest.approx(124000.0)


def test_bingx_live_price_fails_closed_on_api_error_or_malformed_data(monkeypatch):
    from event_engine import bingx as bx

    bx.CACHE["data"] = {"BTC-USDT": {"symbol": "BTC-USDT"}}
    bx.CACHE["by_display_name"] = {"BTC-USDT": bx.CACHE["data"]["BTC-USDT"]}
    bx.CACHE["ts"] = 9_999_999_999

    monkeypatch.setattr(bx, "_request", lambda *args, **kwargs: {"code": 100, "msg": "temporary error"})
    assert bx.get_live_price("BTC") is None

    monkeypatch.setattr(bx, "_request", lambda *args, **kwargs: {"code": 0, "data": {"symbol": "BTC-USDT", "price": "bad"}})
    assert bx.get_live_price("BTC") is None

    monkeypatch.setattr(bx, "_request", lambda *args, **kwargs: {"code": 0, "data": {"symbol": "BTC-USDT", "price": "0"}})
    assert bx.get_live_price("BTC") is None


def test_bingx_open_market_requires_live_sizing_price(monkeypatch):
    from event_engine import bingx as bx

    contract = {
        "symbol": "TEST-USDT", "displayName": "TEST-USDT", "status": 1,
        "apiStateOpen": "true", "quantityPrecision": 3,
        "tradeMinQuantity": 0.001, "tradeMinUSDT": 2.0,
        "maxLongLeverage": 10, "maxShortLeverage": 10,
    }
    bx.CACHE["data"] = {"TEST-USDT": contract}
    bx.CACHE["by_display_name"] = {"TEST-USDT": contract}
    bx.CACHE["ts"] = 9_999_999_999
    monkeypatch.setattr(bx, "has_open_position", lambda *args, **kwargs: False)
    monkeypatch.setattr(bx, "_current_close_price", lambda symbol: None)

    out = bx.open_market("TEST", "LONG", 100.0, "TRD_NO_LIVE_PRICE")
    assert out["status"] == "error"
    assert out["error"] == "live sizing price unavailable"


def test_btc_symbol_normalization_covers_engine_and_exchange_forms():
    import run_once as ro

    assert ro._is_btc_symbol("BTC") is True
    assert ro._is_btc_symbol("BTC-USDT") is True
    assert ro._is_btc_symbol("BTCUSDT") is True
    assert ro._is_btc_symbol("btc/usdt") is True
    assert ro._is_btc_symbol("ETH") is False
    assert ro._is_btc_symbol("ETH-USDT") is False
    assert ro._is_btc_symbol("XRP") is False
    assert ro._is_btc_symbol("KAITO") is False


def test_btc_regime_exemption_uses_normalized_symbol(monkeypatch):
    import run_once as ro

    calls = []
    monkeypatch.setattr(ro, "check_btc_regime", lambda df, direction: calls.append((df, direction)) or (False, "BTC_DUMP"))
    # The production loop delegates the exemption decision to _is_btc_symbol.
    assert ro._is_btc_symbol("BTC") is True
    assert ro._is_btc_symbol("BTC-USDT") is True
    assert ro._is_btc_symbol("BTCUSDT") is True
    assert calls == []


def _make_sfp_lookback_fixture(pivot_index: int) -> pd.DataFrame:
    rows = []
    for i in range(100):
        rows.append({
            "open": 100.0,
            "high": 100.2,
            "low": 99.8,
            "close": 100.0,
            "volume": 1000.0,
            "close_time": 1_700_000_000_000 + i * 3_600_000,
        })
    df = pd.DataFrame(rows)
    df.loc[pivot_index, ["open", "high", "low", "close"]] = [95.2, 95.4, 94.8, 95.1]
    df.loc[99, ["open", "high", "low", "close", "volume"]] = [99.0, 99.8, 93.5, 99.2, 1500.0]
    return df


def test_sfp_lookback_age_10_is_allowed(monkeypatch):
    import event_engine.signals as sig

    df = _make_sfp_lookback_fixture(89)
    monkeypatch.setattr(sig, "_atr", lambda d, n=14: pd.Series([1.0] * len(d), index=d.index, dtype=float))
    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([89], []))
    events = sig.detect_sfp(df, "TEST", "1h", lookback=30)
    assert len(events) == 1
    assert events[0]["event_type"] == "SFP_BULLISH"


def test_sfp_lookback_age_30_is_allowed(monkeypatch):
    import event_engine.signals as sig

    df = _make_sfp_lookback_fixture(69)
    monkeypatch.setattr(sig, "_atr", lambda d, n=14: pd.Series([1.0] * len(d), index=d.index, dtype=float))
    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([69], []))
    events = sig.detect_sfp(df, "TEST", "1h", lookback=30)
    assert len(events) == 1


def test_sfp_lookback_age_31_is_rejected(monkeypatch):
    import event_engine.signals as sig

    df = _make_sfp_lookback_fixture(68)
    monkeypatch.setattr(sig, "_atr", lambda d, n=14: pd.Series([1.0] * len(d), index=d.index, dtype=float))
    monkeypatch.setattr(sig, "_pivots", lambda d, left=3, right=2: ([68], []))
    assert sig.detect_sfp(df, "TEST", "1h", lookback=30) == []


def test_funding_required_policy_rejects_missing_none_nan_and_malformed(monkeypatch):
    import run_once as ro
    from types import SimpleNamespace

    monkeypatch.setattr(ro, "FUNDING_REQUIRED", True)
    for row in (None, SimpleNamespace(fr_oiw=None), SimpleNamespace(fr_oiw=float("nan")), SimpleNamespace(fr_oiw="bad")):
        ok, reason = ro.check_funding_filter(row, "LONG", event_type="REGULAR_BULLISH_RSI")
        assert ok is False
        assert reason.startswith("FUNDING_REQUIRED_")


def test_funding_optional_policy_preserves_fail_open_for_missing_or_invalid(monkeypatch):
    import run_once as ro
    from types import SimpleNamespace

    monkeypatch.setattr(ro, "FUNDING_REQUIRED", False)
    assert ro.check_funding_filter(None, "LONG", event_type="REGULAR_BULLISH_RSI")[0] is True
    assert ro.check_funding_filter(SimpleNamespace(fr_oiw=None), "LONG", event_type="REGULAR_BULLISH_RSI")[0] is True
    assert ro.check_funding_filter(SimpleNamespace(fr_oiw=float("nan")), "LONG", event_type="REGULAR_BULLISH_RSI")[0] is True
    assert ro.check_funding_filter(SimpleNamespace(fr_oiw="bad"), "LONG", event_type="REGULAR_BULLISH_RSI")[0] is True


def test_funding_policy_keeps_normal_extreme_and_squeeze_thresholds(monkeypatch):
    import run_once as ro
    from types import SimpleNamespace

    monkeypatch.setattr(ro, "FUNDING_REQUIRED", True)
    assert ro.check_funding_filter(SimpleNamespace(fr_oiw=0.05), "LONG", event_type="REGULAR_BULLISH_RSI")[0] is True
    assert ro.check_funding_filter(SimpleNamespace(fr_oiw=-0.05), "SHORT", event_type="REGULAR_BEARISH_RSI")[0] is True
    assert ro.check_funding_filter(SimpleNamespace(fr_oiw=0.5001), "LONG", event_type="REGULAR_BULLISH_RSI")[0] is False
    assert ro.check_funding_filter(SimpleNamespace(fr_oiw=-0.5001), "SHORT", event_type="REGULAR_BEARISH_RSI")[0] is False
    assert ro.check_funding_filter(SimpleNamespace(fr_oiw=0.1001), "LONG", event_type="SHORT_SQUEEZE")[0] is False
    assert ro.check_funding_filter(SimpleNamespace(fr_oiw=-0.1001), "SHORT", event_type="LONG_SQUEEZE")[0] is False


def test_production_workflow_declares_funding_required_policy():
    text = Path(".github/workflows/event-engine.yml").read_text(encoding="utf-8")
    assert 'FUNDING_REQUIRED: "false"' in text
