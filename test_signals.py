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
    detect_divergences,
    detect_squeeze_release,
    check_btc_regime,
)
from event_engine.bingx import (
    get_contract,
    CACHE,
    _allocate_tp_quantities,
)
from run_once import (
    build_event_setup,
    build_tp_levels,
    calculate_setup_score,
)
from event_engine.tracker import _update_mfe_mae


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


def test_trigger_uses_event_window_not_latest_bar():
    base = 1700000000000
    rows = []
    # Event happens after bar 0. Only the second post-event bar is a valid trigger.
    for i in range(8):
        close = 100.0
        high = 101.0
        low = 99.0
        if i == 3:
            high, close = 101.0, 100.5
        if i == 5:
            high, close = 101.0, 102.0
        rows.append({
            "open_time": base + i * 900000,
            "close_time": base + i * 900000 + 899000,
            "open": 100.0, "high": high, "low": low, "close": close,
            "volume": 1000.0,
        })
    df = pd.DataFrame(rows)
    # Event is after bar 3; a later trigger within 60m should be found.
    event_ts = rows[3]["close_time"]
    assert build_15m_trigger(df, "LONG", min_vol_mult=0.0, event_detected_at_ts=event_ts, max_trigger_delay_min=60.0) is True


def test_mfe_mae_excludes_pre_entry_bars():
    entry_ts = 1700000000000
    trade = {"entry_price": 100.0, "direction": "LONG", "entry_ts": entry_ts, "peak_pnl_pct": 0.0, "mae_pct": 0.0, "max_drawdown_pct": 0.0}
    candles = [
        {"open_time": entry_ts - 120000, "close_time": entry_ts - 60000, "high": 150.0, "low": 50.0},
        {"open_time": entry_ts, "close_time": entry_ts + 60000, "high": 103.0, "low": 98.0},
    ]
    _update_mfe_mae(trade, candles)
    assert trade["peak_pnl_pct"] == pytest.approx(3.0)
    assert trade["mae_pct"] == pytest.approx(-2.0)
    assert trade["max_drawdown_pct"] == pytest.approx(-5.0)


def test_one_tp_effective_rr_is_supported():
    from event_engine.bingx import _effective_weighted_rr
    levels = [{"leg": "tp3", "pnl_pct": 3.5, "qty": 10.0}]
    assert _effective_weighted_rr(levels, 2.0) == pytest.approx(1.75)
