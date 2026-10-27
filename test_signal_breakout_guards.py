from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import event_engine.signals as sig


def _bars(n: int = 260, *, close: float = 101.5, volume_ratio: float = 2.0) -> pd.DataFrame:
    close_arr = np.full(n, 100.0)
    high = np.full(n, 100.5)
    low = np.full(n, 99.5)
    volume = np.full(n, 1000.0)
    close_arr[-1] = close
    high[-1] = max(close, 100.5)
    low[-1] = min(close, 99.5)
    volume[-1] = 1000.0 * volume_ratio
    return pd.DataFrame({
        "open": close_arr.copy(),
        "high": high,
        "low": low,
        "close": close_arr,
        "volume": volume,
        "close_time": np.arange(n, dtype=np.int64) * 3_600_000 + 1_700_000_000_000,
    })


def _fake_indicators(monkeypatch, n: int, *, ma_compression: bool = False, width_mode: str = "immediate"):
    # A controlled indicator fixture isolates the decisions under test from the
    # numerical warm-up/seed behavior of EMA, ATR, ADX and Bollinger calculations.
    spread = np.full(n, 0.6, dtype=float)
    spread[-6:-1] = [0.1, 0.1, 0.1, 0.8, 0.8]

    def add_ema(df, periods=(5, 10, 30, 50, 200)):
        work = df.copy()
        for period in periods:
            work[f"ema{int(period)}"] = 100.0
        if ma_compression:
            work["ema5"] = 100.0 + spread[:len(work)]
            work["ema30"] = 100.0
        return work

    def atr(df, n=14):
        return pd.Series(1.0, index=df.index, dtype=float)

    def add_adx(df, n=14):
        work = df.copy()
        work["adx"] = 25.0
        return work

    def bbands(series, n=20, std=2.0):
        widths = np.full(len(series), 0.5, dtype=float)
        widths[-10] = 0.1  # old squeeze, deliberately not adjacent to breakout
        if width_mode == "immediate":
            widths[-2] = 0.4
        elif width_mode == "stale_only":
            widths[-2] = 0.8
        mid = pd.Series(100.0, index=series.index, dtype=float)
        half = pd.Series(widths, index=series.index, dtype=float) / 2.0
        return mid * (1.0 + half), mid, mid * (1.0 - half)

    monkeypatch.setattr(sig, "add_ema", add_ema)
    monkeypatch.setattr(sig, "_atr", atr)
    monkeypatch.setattr(sig, "add_adx", add_adx)
    monkeypatch.setattr(sig, "_bbands", bbands)


def test_ma_compression_uses_window_extreme_not_previous_candle(monkeypatch):
    df = _bars(close=101.0)
    df.loc[len(df) - 10, "high"] = 102.0  # structural level inside the 20-bar window
    df.loc[len(df) - 2, "high"] = 100.5  # latest candle high is much lower
    _fake_indicators(monkeypatch, len(df), ma_compression=True)
    assert sig.detect_ma_compression_breakout(df, "TEST", "1h") == []


def test_ma_compression_rejects_volume_below_1_5x(monkeypatch):
    df = _bars(close=101.0, volume_ratio=0.5)
    _fake_indicators(monkeypatch, len(df), ma_compression=True)
    assert sig.detect_ma_compression_breakout(df, "TEST", "1h") == []


def test_ma_compression_requires_three_of_five_compressed_bars(monkeypatch):
    df = _bars(close=101.0)
    _fake_indicators(monkeypatch, len(df), ma_compression=True)
    # Exactly two low-spread bars in the preceding five bars must not qualify.
    spread = np.full(len(df), 0.6, dtype=float)
    spread[-6:-1] = [0.1, 0.1, 0.8, 0.8, 0.8]
    original_add_ema = sig.add_ema
    def add_ema_two_of_five(work, periods=(5, 10, 30, 50, 200)):
        result = original_add_ema(work, periods)
        result["ema5"] = 100.0 + spread
        result["ema30"] = 100.0
        return result
    monkeypatch.setattr(sig, "add_ema", add_ema_two_of_five)
    assert sig.detect_ma_compression_breakout(df, "TEST", "1h") == []


def test_ma_compression_rejects_extension_over_two_atr(monkeypatch):
    df = _bars(close=103.0)
    _fake_indicators(monkeypatch, len(df), ma_compression=True)
    assert sig.detect_ma_compression_breakout(df, "TEST", "1h") == []


def test_ma_compression_accepts_valid_three_of_five_setup(monkeypatch):
    df = _bars(close=101.0, volume_ratio=2.0)
    _fake_indicators(monkeypatch, len(df), ma_compression=True)
    events = sig.detect_ma_compression_breakout(df, "TEST", "1h")
    assert len(events) == 1
    assert events[0]["direction"] == "LONG"
    assert events[0]["event_fact"]["volume_ratio"] == pytest.approx(2.0)
    assert events[0]["event_fact"]["breakout_atr"] == pytest.approx(0.5)


def test_breakout_momentum_requires_immediately_preceding_compression(monkeypatch):
    df = _bars(close=101.5, volume_ratio=2.0)
    _fake_indicators(monkeypatch, len(df), width_mode="stale_only")
    assert sig.detect_breakout_momentum(df, "TEST", "1h") == []


def test_breakout_momentum_caps_extension_at_two_atr(monkeypatch):
    df = _bars(close=103.0, volume_ratio=2.0)
    _fake_indicators(monkeypatch, len(df), width_mode="immediate")
    assert sig.detect_breakout_momentum(df, "TEST", "1h") == []


def test_breakout_momentum_allows_fresh_compressed_breakout_within_cap(monkeypatch):
    df = _bars(close=101.5, volume_ratio=2.0)
    _fake_indicators(monkeypatch, len(df), width_mode="immediate")
    events = sig.detect_breakout_momentum(df, "TEST", "1h")
    assert len(events) == 1
    assert events[0]["event_fact"]["breakout_atr"] == pytest.approx(1.0)


def test_donchian_retest_rejects_extension_over_two_atr(monkeypatch):
    df = _bars(close=103.0, volume_ratio=2.0)
    _fake_indicators(monkeypatch, len(df), width_mode="immediate")
    assert sig.detect_donchian_retest(df, "TEST", "1h") == []


def test_donchian_retest_allows_breakout_within_two_atr(monkeypatch):
    df = _bars(close=101.5, volume_ratio=2.0)
    _fake_indicators(monkeypatch, len(df), width_mode="immediate")
    events = sig.detect_donchian_retest(df, "TEST", "1h")
    assert len(events) == 1
    assert events[0]["event_fact"]["breakout_atr"] == pytest.approx(1.0)
