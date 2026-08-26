from __future__ import annotations

import numpy as np
import pandas as pd

from event_engine.coinalyze import parse_number
from event_engine.signals import (
    add_cvd,
    build_15m_trigger,
    detect_divergences,
    detect_squeeze_release,
)


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


def test_trigger_long_and_short():
    df_long = pd.DataFrame({"high": [100, 105], "low": [95, 100], "close": [99, 106]})
    assert build_15m_trigger(df_long, "LONG") is True
    assert build_15m_trigger(df_long, "long") is True
    assert build_15m_trigger(df_long, "SHORT") is False

    df_short = pd.DataFrame({"high": [105, 104], "low": [100, 95], "close": [101, 94]})
    assert build_15m_trigger(df_short, "SHORT") is True
    assert build_15m_trigger(df_short, "short") is True
    assert build_15m_trigger(df_short, "LONG") is False


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
