from __future__ import annotations

import hashlib
import math
import os
from typing import Any

import numpy as np
import pandas as pd


def _wilder_smooth(values: pd.Series, n: int) -> pd.Series:
    """Wilder/RMA smoothing with a classic SMA seed over the first n valid values.

    This matches TradingView's RMA-style initialization while correctly handling
    series whose first valid sample is not at index 0 (e.g. RSI deltas) and series
    whose first sample is already valid (e.g. True Range for ATR).
    """
    out = pd.Series(np.nan, index=values.index, dtype=float)
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if len(arr) == 0 or n <= 0:
        return out

    finite_positions = np.flatnonzero(np.isfinite(arr))
    if len(finite_positions) < n:
        return out

    seed_positions = finite_positions[:n]
    seed = float(np.mean(arr[seed_positions]))
    seed_pos = int(seed_positions[-1])
    result = np.full(len(arr), np.nan, dtype=float)
    result[seed_pos] = seed
    prev = seed

    for i in range(seed_pos + 1, len(arr)):
        x = arr[i]
        if np.isfinite(x):
            prev = (prev * (n - 1) + x) / n
        result[i] = prev

    out.iloc[:] = result
    return out


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    # Wilder RSI (audit fix B1): SMA-seeded smoothing instead of a bare EMA so
    # values match TradingView Pine ta.rsi within < 0.1 point.
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = _wilder_smooth(gain, n)
    avg_loss = _wilder_smooth(loss, n)

    zero_loss = (avg_loss == 0) & (avg_gain > 0)
    zero_both = (avg_loss == 0) & (avg_gain == 0)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.where(~zero_loss, 100.0)
    rsi = rsi.where(~zero_both, 50.0)

    valid_warmup = avg_gain.notna() & avg_loss.notna()
    return rsi.where(valid_warmup, np.nan)


def _bbands(
    series: pd.Series,
    n: int = 20,
    std: float = 2.0,
):
    mid = series.rolling(n, min_periods=n).mean()
    sd = series.rolling(n, min_periods=n).std(ddof=0)
    upper = mid + std * sd
    lower = mid - std * sd
    return upper, mid, lower


def _atr(
    df: pd.DataFrame,
    n: int = 14,
) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return _wilder_smooth(tr, n)


def _pivots(
    df: pd.DataFrame,
    left: int = 3,
    right: int = 2,
):
    lows: list[int] = []
    highs: list[int] = []

    for i in range(left, len(df) - right):
        current_low = float(df["low"].iloc[i])
        previous_lows = df["low"].iloc[i - left:i]
        next_lows = df["low"].iloc[i + 1:i + right + 1]

        current_high = float(df["high"].iloc[i])
        previous_highs = df["high"].iloc[i - left:i]
        next_highs = df["high"].iloc[i + 1:i + right + 1]

        if current_low <= float(previous_lows.min()) and current_low < float(next_lows.min()):
            lows.append(i)

        if current_high >= float(previous_highs.max()) and current_high > float(next_highs.max()):
            highs.append(i)

    return lows, highs


def _event_id(
    symbol: str,
    tf: str,
    typ: str,
    p1_ts: int,
    p2_ts: int,
) -> str:
    fp = f"{symbol}:{tf}:{typ}:{int(p1_ts)}:{int(p2_ts)}"
    return "EVT_" + hashlib.sha256(fp.encode()).hexdigest()[:16].upper()


def add_cvd(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a continuous CVD state without fabricating zero flow.

    A missing/invalid taker-flow bar does not reset the cumulative CVD. The last
    known CVD value is carried forward, while ``taker_flow_valid`` stays False so
    callers can apply a coverage/validity policy. This preserves continuity
    without asserting that missing flow was actually zero.
    """
    work = df.copy()

    if "bar_delta_usdt" not in work.columns:
        work["bingx_cvd"] = float("nan")
        work["taker_flow_valid"] = False
        work["cvd_segment_id"] = 0
        work["cvd_valid_coverage"] = 0.0
        return work

    if "taker_flow_valid" not in work.columns:
        work["taker_flow_valid"] = False

    work["taker_flow_valid"] = work["taker_flow_valid"].fillna(False).astype(bool)
    work["bar_delta_usdt"] = pd.to_numeric(work["bar_delta_usdt"], errors="coerce")

    cvd = []
    coverage = []
    current = float("nan")
    valid_count = 0
    for i in work.index:
        valid = bool(work.at[i, "taker_flow_valid"]) and pd.notna(work.at[i, "bar_delta_usdt"])
        if valid:
            delta = float(work.at[i, "bar_delta_usdt"])
            current = delta if pd.isna(current) else current + delta
            valid_count += 1
        cvd.append(current)
        coverage.append(valid_count / max(1, len(cvd)))

    work["bingx_cvd"] = cvd
    work["cvd_valid_coverage"] = coverage
    # Kept as a stable column for existing event metadata/tests; there are no
    # artificial segments anymore, so every row belongs to the same sequence.
    work["cvd_segment_id"] = 0
    return work


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Append a classic MACD line (ema_fast - ema_slow) with NaN warmup.

    The MACD line is the value compared at swing pivots. ema_slow carries
    min_periods=slow so early unreliable bars are excluded from divergence
    checks (emit_divergence drops NaN pivots).
    """
    work = df.copy()
    close = pd.to_numeric(work.get("close"), errors="coerce")
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean() if close is not None else pd.Series(dtype=float)
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean() if close is not None else pd.Series(dtype=float)
    work["macd"] = ema_fast - ema_slow
    return work


def add_ema(df: pd.DataFrame, periods: tuple[int, ...] = (5, 10, 30, 50, 200)) -> pd.DataFrame:
    work = df.copy()
    close = pd.to_numeric(work.get("close"), errors="coerce")
    if close is None:
        return work
    for n in periods:
        work[f"ema{int(n)}"] = close.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()
    return work


def add_adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    work = df.copy()
    high = pd.to_numeric(work.get("high"), errors="coerce")
    low = pd.to_numeric(work.get("low"), errors="coerce")
    close = pd.to_numeric(work.get("close"), errors="coerce")
    if high is None or low is None or close is None:
        work["adx"] = float("nan")
        return work
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = _wilder_smooth(tr, n)
    plus = 100.0 * _wilder_smooth(plus_dm, n) / atr.replace(0, np.nan)
    minus = 100.0 * _wilder_smooth(minus_dm, n) / atr.replace(0, np.nan)
    dx = 100.0 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)
    work["adx"] = _wilder_smooth(dx, n)
    work["plus_di"] = plus
    work["minus_di"] = minus
    return work


def _trend_context(df: pd.DataFrame, direction: str) -> dict[str, Any]:
    """Describe the prevailing local trend without looking into the future."""
    if len(df) < 55:
        return {"trend_ok": None, "trend": "UNKNOWN"}
    d = str(direction).upper()
    work = add_ema(df, (20, 50, 200))
    close = float(pd.to_numeric(work["close"], errors="coerce").iloc[-1])
    ema20 = float(work["ema20"].iloc[-1]) if pd.notna(work["ema20"].iloc[-1]) else close
    ema50 = float(work["ema50"].iloc[-1]) if pd.notna(work["ema50"].iloc[-1]) else ema20
    slope50 = float(work["ema50"].iloc[-1] - work["ema50"].iloc[-6]) if len(work) >= 56 and work["ema50"].iloc[-6:].notna().all() else 0.0
    if d == "LONG":
        ok = close > ema50 and slope50 >= 0
        trend = "UP" if ok else "DOWN_OR_FLAT"
    else:
        ok = close < ema50 and slope50 <= 0
        trend = "DOWN" if ok else "UP_OR_FLAT"
    return {"trend_ok": bool(ok), "trend": trend, "ema20": ema20, "ema50": ema50, "slope50": slope50}


def detect_macd_4h(df: pd.DataFrame, symbol: str, timeframe: str = "4h") -> list[dict[str, Any]]:
    """Detect a confirmed 4H MACD regime transition as a standalone event."""
    if len(df) < 220 or timeframe.lower() != "4h":
        return []
    d = df.copy()
    for col in ("close", "high", "low", "close_time"):
        if col not in d.columns:
            return []
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["close", "high", "low", "close_time"]).reset_index(drop=True)
    if len(d) < 80:
        return []
    d = add_macd(d)
    d = add_ema(d, (200,))
    macd = d["macd"]
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = macd - signal
    i = len(d) - 1
    if i < 2 or any(pd.isna(x) for x in (macd.iloc[i-1], signal.iloc[i-1], macd.iloc[i], signal.iloc[i], hist.iloc[i])):
        return []
    cross_up = macd.iloc[i-1] <= signal.iloc[i-1] and macd.iloc[i] > signal.iloc[i]
    cross_down = macd.iloc[i-1] >= signal.iloc[i-1] and macd.iloc[i] < signal.iloc[i]
    close = float(d["close"].iloc[i])
    ema200 = float(d["ema200"].iloc[i]) if pd.notna(d["ema200"].iloc[i]) else close
    if pd.isna(d["ema200"].iloc[i]):
        return []
    if cross_up and close > ema200:
        direction, typ = "LONG", "MACD_4H_BULLISH_CROSS"
    elif cross_down and close < ema200:
        direction, typ = "SHORT", "MACD_4H_BEARISH_CROSS"
    else:
        return []
    ts = int(d["close_time"].iloc[i])
    return [{
        "event_id": _event_id(symbol, timeframe, typ, ts, ts),
        "symbol": symbol, "timeframe": timeframe, "direction": direction, "event_type": typ,
        "timestamps": {"pivot_1_ts": ts, "pivot_2_ts": ts, "detected_at_ts": ts},
        "event_fact": {"detection_close_price": close, "macd": float(macd.iloc[i]), "macd_signal": float(signal.iloc[i]),
                        "macd_hist": float(hist.iloc[i]), "ema200": ema200, "distance_ema200_atr": None,
                        "engine": "MACD_4H"},
    }]


def detect_ma_compression_breakout(df: pd.DataFrame, symbol: str, timeframe: str = "1h",
                                   compression_lookback: int = 20, compression_quantile: float = 0.35) -> list[dict[str, Any]]:
    """1H MA compression followed by a structural breakout. Entry later requires retest."""
    if timeframe.lower() != "1h" or len(df) < 80:
        return []
    d = df.copy()
    for col in ("close", "high", "low", "volume", "close_time"):
        if col not in d.columns:
            return []
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["close", "high", "low", "volume", "close_time"]).reset_index(drop=True)
    d = add_ema(d, (5, 10, 30, 50))
    d["atr"] = _atr(d, 14)
    spread = (d["ema5"] - d["ema30"]).abs() / d["atr"].replace(0, np.nan)
    hist = spread.iloc[:-1].dropna().tail(compression_lookback * 3)
    if len(hist) < compression_lookback or pd.isna(spread.iloc[-2]) or pd.isna(spread.iloc[-1]):
        return []
    threshold = float(hist.quantile(compression_quantile))
    was_compressed = float(spread.iloc[-2]) <= threshold
    if not was_compressed:
        return []
    close = float(d["close"].iloc[-1])
    prev = float(d["close"].iloc[-2])
    atr = float(d["atr"].iloc[-1]) if pd.notna(d["atr"].iloc[-1]) else 0.0
    if atr <= 0:
        return []
    prior_high = float(d["high"].iloc[-2])
    prior_low = float(d["low"].iloc[-2])
    vol_mean = d["volume"].iloc[-21:-1].mean() if len(d) >= 21 else np.nan
    vol_ratio = float(d["volume"].iloc[-1] / vol_mean) if pd.notna(vol_mean) and vol_mean > 0 else 0.0
    if close > prior_high and close > float(d["ema30"].iloc[-1]):
        direction, breakout = "LONG", (close - prev) / atr
    elif close < prior_low and close < float(d["ema30"].iloc[-1]):
        direction, breakout = "SHORT", (prev - close) / atr
    else:
        return []
    ts = int(d["close_time"].iloc[-1])
    typ = "MA_COMPRESSION_BREAKOUT"
    return [{
        "event_id": _event_id(symbol, timeframe, typ, ts, ts), "symbol": symbol, "timeframe": timeframe,
        "direction": direction, "event_type": typ, "timestamps": {"pivot_1_ts": ts, "pivot_2_ts": ts, "detected_at_ts": ts},
        "event_fact": {"detection_close_price": close, "compression_ratio": float(spread.iloc[-2] / max(threshold, 1e-9)),
                        "ma_spread_atr": float(spread.iloc[-2]), "compression_threshold": threshold,
                        "breakout_atr": float(breakout), "volume_ratio": vol_ratio, "requires_retest": True, "trigger_level": float(prior_high if direction == "LONG" else prior_low), "engine": "MA_COMPRESSION"},
    }]


def detect_breakout_momentum(df: pd.DataFrame, symbol: str, timeframe: str = "1h", donchian_n: int = 20) -> list[dict[str, Any]]:
    """Breakout + compression + momentum + volume + ADX trend event."""
    if timeframe.lower() not in {"1h", "4h"} or len(df) < max(80, donchian_n + 40):
        return []
    d = df.copy()
    for col in ("close", "high", "low", "volume", "close_time"):
        if col not in d.columns:
            return []
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["close", "high", "low", "volume", "close_time"]).reset_index(drop=True)
    d = add_ema(d, (50,))
    d = add_adx(d, 14)
    bb_u, mid, bb_l = _bbands(d["close"], 20, 2.0)
    atr = _atr(d, 14)
    bb_width = (bb_u - bb_l) / mid.replace(0, np.nan)
    hist = bb_width.iloc[:-1].dropna().tail(100)
    if len(hist) < 30 or pd.isna(atr.iloc[-1]) or pd.isna(d["adx"].iloc[-1]):
        return []
    width_threshold = float(hist.quantile(0.35))
    compressed_recent = bool((bb_width.iloc[-11:-1] <= width_threshold).any())
    close = float(d["close"].iloc[-1]); atr_v = float(atr.iloc[-1])
    dc_high = float(d["high"].iloc[-donchian_n-1:-1].max())
    dc_low = float(d["low"].iloc[-donchian_n-1:-1].min())
    vol_mean = d["volume"].iloc[-21:-1].mean()
    vol_ratio = float(d["volume"].iloc[-1] / vol_mean) if pd.notna(vol_mean) and vol_mean > 0 else 0.0
    adx = float(d["adx"].iloc[-1])
    if not compressed_recent or adx < 20 or vol_ratio < 1.5 or atr_v <= 0:
        return []
    if close > dc_high + 0.10 * atr_v and close > float(d["ema50"].iloc[-1]):
        direction, breakout_atr = "LONG", (close - dc_high) / atr_v
    elif close < dc_low - 0.10 * atr_v and close < float(d["ema50"].iloc[-1]):
        direction, breakout_atr = "SHORT", (dc_low - close) / atr_v
    else:
        return []
    ts = int(d["close_time"].iloc[-1])
    typ = "BREAKOUT_MOMENTUM"
    return [{
        "event_id": _event_id(symbol, timeframe, typ, ts, ts), "symbol": symbol, "timeframe": timeframe,
        "direction": direction, "event_type": typ, "timestamps": {"pivot_1_ts": ts, "pivot_2_ts": ts, "detected_at_ts": ts},
        "event_fact": {"detection_close_price": close, "donchian_high": dc_high, "donchian_low": dc_low,
                        "breakout_atr": float(breakout_atr), "bb_width": float(bb_width.iloc[-1]),
                        "bb_width_threshold": width_threshold, "volume_ratio": vol_ratio, "adx": adx,
                        "requires_retest": True, "trigger_level": float(dc_high if direction == "LONG" else dc_low), "engine": "BREAKOUT_MOMENTUM"},
    }]


def diagnose_15m_retest_trigger(df15: pd.DataFrame, direction: str, reference_level: float,
                                event_detected_at_ts: int, max_delay_min: float = 30.0,
                                volume_mult: float = 1.10, tolerance_pct: float = 0.25) -> dict[str, Any]:
    """Require breakout acceptance through a retest instead of buying the first break."""
    out = {"ok": False, "reason": "no_retest_window", "trigger_bar_close_ts": None, "trigger_price": None,
           "trigger_delay_min": None, "volume_ratio": None, "retest_level": reference_level}
    if not isinstance(df15, pd.DataFrame) or len(df15) < 22:
        out["reason"] = "insufficient_data"; return out
    d = df15.copy()
    for c in ("close_time", "open", "high", "low", "close", "volume"):
        if c not in d.columns:
            out["reason"] = "invalid_data"; return out
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["close_time", "open", "high", "low", "close", "volume"]).sort_values("close_time").reset_index(drop=True)
    ref = float(reference_level)
    tol = abs(ref) * max(0.0, tolerance_pct) / 100.0
    candidate_indices = []
    event_ts = int(event_detected_at_ts)
    for i in range(1, len(d)):
        ts = int(d["close_time"].iloc[i])
        delay = (ts - event_ts) / 60000.0
        if 0 < delay <= max_delay_min:
            candidate_indices.append(i)
    out["bars_considered"] = len(candidate_indices)
    for i in reversed(candidate_indices):
        ts = int(d["close_time"].iloc[i])
        delay = (ts - event_ts) / 60000.0
        prev = d.iloc[i-1]; cur = d.iloc[i]
        vol_mean = d["volume"].iloc[max(0, i-20):i].mean()
        vol_ratio = float(cur["volume"] / vol_mean) if pd.notna(vol_mean) and vol_mean > 0 else 0.0
        if direction.upper() == "LONG":
            retest = float(cur["low"]) <= ref + tol and float(cur["close"]) > ref and float(cur["close"]) > float(cur["open"])
        else:
            retest = float(cur["high"]) >= ref - tol and float(cur["close"]) < ref and float(cur["close"]) < float(cur["open"])
        if retest and vol_ratio >= volume_mult:
            out.update({"ok": True, "reason": "retest_passed", "trigger_bar_close_ts": ts,
                        "trigger_price": float(cur["close"]), "trigger_delay_min": round(delay, 3), "volume_ratio": round(vol_ratio, 6)})
            return out
    return out


def add_stochastic(df: pd.DataFrame, n: int = 14, smooth: int = 3) -> pd.DataFrame:
    """Append slow %K (raw %K smoothed by SMA(smooth))."""
    work = df.copy()
    high = pd.to_numeric(work.get("high"), errors="coerce")
    low = pd.to_numeric(work.get("low"), errors="coerce")
    close = pd.to_numeric(work.get("close"), errors="coerce")

    if high is None or low is None or close is None:
        work["stoch"] = float("nan")
        return work

    hh = high.rolling(n, min_periods=n).max()
    ll = low.rolling(n, min_periods=n).min()
    raw_k = (close - ll) / (hh - ll).replace(0, np.nan) * 100.0
    work["stoch"] = raw_k.rolling(smooth, min_periods=smooth).mean()
    return work


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Append On-Balance Volume (standard implementation of volume divergence)."""
    work = df.copy()
    close = pd.to_numeric(work.get("close"), errors="coerce")
    volume = pd.to_numeric(work.get("volume"), errors="coerce").fillna(0.0) if "volume" in work.columns else None

    if close is None or volume is None:
        work["obv"] = float("nan")
        return work

    direction = np.sign(close.diff()).fillna(0.0)
    work["obv"] = (direction * volume).cumsum()
    return work


def attach_oi_series(df: pd.DataFrame, oi_history: dict[str, float] | None) -> pd.DataFrame:
    """Map the accumulated OI snapshot history onto closed bars.

    oi_history maps "1h bucket index" (close_time // 3_600_000) to the last
    OI snapshot recorded inside that bucket. Snapshots are only written while
    their bucket is active, so a stored value is always from before the bucket
    closed. For 4h bars we take the last 1h bucket strictly before the bar
    close (bucket index of (close_time - 1) // 3_600_000).
    """
    work = df.copy()
    if not isinstance(oi_history, dict) or not oi_history:
        work["oi"] = float("nan")
        return work

    close_time = pd.to_numeric(work.get("close_time"), errors="coerce")
    buckets = (close_time - 1).floordiv(3_600_000)
    work["oi"] = buckets.map(
        lambda b: oi_history.get(str(int(b)), float("nan")) if pd.notna(b) else float("nan")
    )
    return work


def detect_divergences(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "1h",
    left: int = 3,
    right: int = 2,
    min_bars: int = 5,
    max_bars: int = 16,
    min_delta_atr: float = 0.35,
) -> list[dict[str, Any]]:

    if len(df) < 60:
        return []

    required = {"close", "high", "low", "close_time"}
    if not required.issubset(df.columns):
        return []

    work = df.copy()
    for col in ("close", "high", "low", "close_time"):
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["close", "high", "low", "close_time"]).reset_index(drop=True)
    if len(work) < 60:
        return []

    work["rsi"] = _rsi(work["close"], 14)
    work["atr"] = _atr(work, 14)
    # Additional divergence sources (audit gaps B2 / missing features):
    # MACD line, slow Stochastic %K, OBV (volume divergence) and, when the
    # caller attached a historical OI series, Price-vs-OI divergence.
    work = add_macd(work)
    work = add_stochastic(work)
    work = add_obv(work)

    indicators: list[str] = ["rsi", "bingx_cvd", "macd", "stoch", "obv"]
    if "oi" in work.columns:
        indicators.append("oi")

    lows, highs = _pivots(work, left, right)
    events: list[dict[str, Any]] = []

    def emit_divergence(p1i: int, p2i: int, indicator: str, is_low: bool) -> None:
        bars = p2i - p1i
        if not (min_bars <= bars <= max_bars):
            return

        atr_value = work["atr"].iloc[p2i]
        if pd.isna(atr_value):
            return

        atr = float(atr_value)
        if not math.isfinite(atr) or atr <= 0:
            return

        if indicator not in work.columns:
            return

        p1v = work[indicator].iloc[p1i]
        p2v = work[indicator].iloc[p2i]
        if pd.isna(p1v) or pd.isna(p2v):
            return

        detected = p2i + right
        if detected >= len(work):
            return

        if indicator == "bingx_cvd":
            # CVD is continuous across missing bars, but a divergence must still
            # have enough valid taker-flow observations to be trustworthy.
            if "taker_flow_valid" not in work.columns:
                return
            span = work.iloc[p1i:detected + 1]
            valid = span["taker_flow_valid"].astype(bool) & span["bar_delta_usdt"].notna()
            if valid.mean() < 0.80:
                return

        price_column = "low" if is_low else "high"
        p1_price = float(work[price_column].iloc[p1i])
        p2_price = float(work[price_column].iloc[p2i])
        price_delta_atr = abs(p2_price - p1_price) / atr

        if not math.isfinite(price_delta_atr) or price_delta_atr < min_delta_atr:
            return

        typ = None
        direction = None

        if is_low:
            if p2_price < p1_price and float(p2v) > float(p1v):
                typ = "REGULAR_BULLISH_" + indicator.upper()
                direction = "LONG"
            elif p2_price > p1_price and float(p2v) < float(p1v):
                typ = "HIDDEN_BULLISH_" + indicator.upper()
                direction = "LONG"
        else:
            if p2_price > p1_price and float(p2v) < float(p1v):
                typ = "REGULAR_BEARISH_" + indicator.upper()
                direction = "SHORT"
            elif p2_price < p1_price and float(p2v) > float(p1v):
                typ = "HIDDEN_BEARISH_" + indicator.upper()
                direction = "SHORT"

        if not typ or not direction:
            return

        p1_ts = int(work["close_time"].iloc[p1i])
        p2_ts = int(work["close_time"].iloc[p2i])
        detected_ts = int(work["close_time"].iloc[detected])

        events.append(
            {
                "event_id": _event_id(symbol, timeframe, typ, p1_ts, p2_ts),
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction,
                "event_type": typ,
                "timestamps": {
                    "pivot_1_ts": p1_ts,
                    "pivot_2_ts": p2_ts,
                    "detected_at_ts": detected_ts,
                },
                "event_fact": {
                    "detection_close_price": float(work["close"].iloc[detected]),
                    "p1_price": p1_price,
                    "p2_price": p2_price,
                    "p1_indicator": float(p1v),
                    "p2_indicator": float(p2v),
                    "bars_between": bars,
                    "price_delta_atr": float(price_delta_atr),
                },
            }
        )

    def scan_pivots(pivots: list[int], is_low: bool) -> None:
        num_piv = len(pivots)
        for i in range(num_piv):
            for step in (1, 2):
                if i + step >= num_piv:
                    continue
                p1 = pivots[i]
                p2 = pivots[i + step]
                for indicator in indicators:
                    emit_divergence(p1, p2, indicator, is_low)

    if len(lows) >= 2:
        scan_pivots(lows, is_low=True)

    if len(highs) >= 2:
        scan_pivots(highs, is_low=False)

    return events


def detect_squeeze_release(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "1h",
    min_squeeze_bars: int = 3,
    release_lookback_bars: int = 4,
) -> list[dict[str, Any]]:
    """Detect recent volatility-squeeze releases without changing release math.

    The original detector evaluated only the final closed bar. That creates a
    missed-event hole if a scheduled scan is delayed or a runner misses a
    timeframe bucket. We preserve the exact BB/KC transition rule and simply
    inspect a short recent tail. Freshness is enforced by the caller/event TTL.
    """

    if len(df) < 40:
        return []

    required = {"close", "high", "low", "close_time"}
    if not required.issubset(df.columns):
        return []

    b = df.copy()
    for col in ("close", "high", "low", "close_time"):
        b[col] = pd.to_numeric(b[col], errors="coerce")

    b = b.dropna(subset=["close", "high", "low", "close_time"]).reset_index(drop=True)
    if len(b) < 40:
        return []

    bb_u, mid, bb_l = _bbands(b["close"], 20, 2.0)
    atr = _atr(b, 20)
    kc_u = mid + 1.5 * atr
    kc_l = mid - 1.5 * atr

    in_sq = (bb_u <= kc_u) & (bb_l >= kc_l)
    if len(b) < (min_squeeze_bars + 2):
        return []

    lookback = max(1, int(release_lookback_bars))
    first_release_idx = max(1, len(b) - lookback)
    releases: list[dict[str, Any]] = []

    for current_idx in range(first_release_idx, len(b)):
        prev_idx = current_idx - 1
        if pd.isna(bb_u.iloc[current_idx]) or pd.isna(kc_u.iloc[current_idx]):
            continue
        if bool(in_sq.iloc[current_idx]) or not bool(in_sq.iloc[prev_idx]):
            continue

        sq_duration = 0
        idx = prev_idx
        while idx >= 0 and bool(in_sq.iloc[idx]):
            sq_duration += 1
            idx -= 1
        if sq_duration < min_squeeze_bars:
            continue

        close_val = float(b["close"].iloc[current_idx])
        kc_u_val = float(kc_u.iloc[current_idx])
        kc_l_val = float(kc_l.iloc[current_idx])
        if close_val > kc_u_val:
            direction = "LONG"
        elif close_val < kc_l_val:
            direction = "SHORT"
        else:
            continue

        ts = int(b["close_time"].iloc[current_idx])
        typ = "VOLATILITY_SQUEEZE_RELEASE"
        bb_w = float(bb_u.iloc[current_idx] - bb_l.iloc[current_idx])
        kc_w = float(kc_u.iloc[current_idx] - kc_l.iloc[current_idx])
        compression_ratio = bb_w / kc_w if kc_w > 0 else 1.0
        releases.append(
            {
                "event_id": _event_id(symbol, timeframe, typ, ts, ts),
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction,
                "event_type": typ,
                "timestamps": {
                    "pivot_1_ts": ts,
                    "pivot_2_ts": ts,
                    "detected_at_ts": ts,
                },
                "event_fact": {
                    "detection_close_price": close_val,
                    "bb_width": bb_w,
                    "kc_width": kc_w,
                    "squeeze_duration_bars": sq_duration,
                    "compression_ratio": float(compression_ratio),
                },
            }
        )

    return releases


def _liq_squeeze_thresholds() -> dict[str, float]:
    """Environment-tunable thresholds for forced-liquidation squeeze detection.

    Funding-rate units follow Coinalyze's displayed values (0.03 == 0.03%),
    consistent with calculate_setup_score's existing funding handling.
    """

    def _num(name: str, default: float) -> float:
        try:
            value = float(os.environ.get(name, ""))
            return value if math.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    return {
        "liq_oi_ratio": _num("LIQ_SQUEEZE_MIN_LIQ_OI_RATIO", 0.01),
        "oi_chg4h_min": _num("LIQ_SQUEEZE_MIN_OI_CHG4H_PCT", 1.5),
        "funding_extreme": _num("LIQ_SQUEEZE_FUNDING_EXTREME", 0.03),
        "price_spike_atr": _num("LIQ_SQUEEZE_PRICE_SPIKE_ATR", 1.0),
        "ls_short_max": _num("LIQ_SQUEEZE_LS_SHORT_MAX", 0.75),
        "ls_long_min": _num("LIQ_SQUEEZE_LS_LONG_MIN", 2.5),
    }


def detect_liquidation_squeeze(
    row: Any,
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "1h",
) -> list[dict[str, Any]]:
    """Detect forced-liquidation cascades (audit gap B3).

    SHORT_SQUEEZE (trade direction LONG): sharp up-spike on the last closed bar
    plus elevated short liquidations relative to open interest.
    LONG_SQUEEZE (trade direction SHORT): the mirror image for long liquidations.

    Hard factors (both required): price spike (close breaks the previous bar's
    extreme by >= LIQ_SQUEEZE_PRICE_SPIKE_ATR * ATR14) and a liquidation ratio
    liq24/oi >= LIQ_SQUEEZE_MIN_LIQ_OI_RATIO. Soft factors (at least one
    required): 4h OI surge, extreme OI-weighted funding, crowded L/S accounts.
    All factor values are stored in event_fact for later review. Returns []
    whenever Coinalyze data is unavailable instead of guessing.
    """
    if row is None or df is None or not isinstance(df, pd.DataFrame) or len(df) < 30:
        return []

    required = {"close", "high", "low", "close_time"}
    if not required.issubset(df.columns):
        return []

    liq_short = getattr(row, "liq_short24", None)
    liq_long = getattr(row, "liq_long24", None)
    oi = getattr(row, "oi", None)
    if liq_short is None or liq_long is None or oi is None:
        return []

    try:
        liq_short = float(liq_short)
        liq_long = float(liq_long)
        oi = float(oi)
    except (TypeError, ValueError):
        return []
    if oi <= 0 or liq_short < 0 or liq_long < 0:
        return []

    th = _liq_squeeze_thresholds()

    b = df.copy()
    for col in ("close", "high", "low", "close_time"):
        b[col] = pd.to_numeric(b[col], errors="coerce")
    b = b.dropna(subset=["close", "high", "low", "close_time"]).reset_index(drop=True)
    if len(b) < 30:
        return []

    atr_series = _atr(b, 14)
    atr_value = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
    if not math.isfinite(atr_value) or atr_value <= 0:
        return []

    last = b.iloc[-1]
    prev = b.iloc[-2]
    close_val = float(last["close"])
    prev_close = float(prev["close"])
    prev_high = float(prev["high"])
    prev_low = float(prev["low"])
    move = close_val - prev_close

    detected_ts = int(last["close_time"])

    def _soft_factors(direction: str) -> dict[str, Any]:
        oi_chg4h = getattr(row, "oi_chg4h_pct", None)
        fr = getattr(row, "fr_oiw", None)
        ls = getattr(row, "ls_accounts", None)
        try:
            oi_chg4h = float(oi_chg4h) if oi_chg4h is not None else None
        except (TypeError, ValueError):
            oi_chg4h = None
        try:
            fr = float(fr) if fr is not None else None
        except (TypeError, ValueError):
            fr = None
        try:
            ls = float(ls) if ls is not None else None
        except (TypeError, ValueError):
            ls = None

        oi_surge = oi_chg4h is not None and oi_chg4h >= th["oi_chg4h_min"]
        if direction == "LONG":
            # A long trade caused by short squeeze needs crowded/negative funding.
            funding_extreme = fr is not None and fr <= -abs(th["funding_extreme"])
            ls_crowded = ls is not None and ls <= th["ls_short_max"]
        else:
            # A short trade caused by long squeeze needs crowded/positive funding.
            funding_extreme = fr is not None and fr >= abs(th["funding_extreme"])
            ls_crowded = ls is not None and ls >= th["ls_long_min"]

        return {
            "oi_chg4h_pct": oi_chg4h,
            "fr_oiw": fr,
            "ls_accounts": ls,
            "oi_surge": bool(oi_surge),
            "funding_extreme": bool(funding_extreme),
            "ls_crowded": bool(ls_crowded),
        }

    candidates: list[tuple[str, str, float, int]] = []

    # A liquidation cascade can span several consecutive bars. Keep one stable
    # episode/family identity across that run so the event is updated rather
    # than emitted as a new opportunity on every bar.
    def _family_start_index(last_idx: int, direction: str) -> int:
        start = last_idx
        for idx in range(last_idx, 0, -1):
            cur = b.iloc[idx]
            prv = b.iloc[idx - 1]
            cur_close = float(cur["close"])
            prv_close = float(prv["close"])
            cur_move = cur_close - prv_close
            cur_atr = float(atr_series.iloc[idx]) if pd.notna(atr_series.iloc[idx]) else 0.0
            if not math.isfinite(cur_atr) or cur_atr <= 0:
                break
            if direction == "LONG":
                qualifies = cur_move > 0 and cur_close > float(prv["high"]) and cur_move >= th["price_spike_atr"] * cur_atr
            else:
                qualifies = cur_move < 0 and cur_close < float(prv["low"]) and abs(cur_move) >= th["price_spike_atr"] * cur_atr
            if not qualifies:
                break
            start = idx - 1
        return start + 1 if start < last_idx else last_idx

    if move > 0 and close_val > prev_high and move >= th["price_spike_atr"] * atr_value:
        ratio = liq_short / oi
        if ratio >= th["liq_oi_ratio"]:
            family_start = _family_start_index(len(b) - 1, "LONG")
            candidates.append(("SHORT_SQUEEZE", "LONG", ratio, family_start))

    if move < 0 and close_val < prev_low and abs(move) >= th["price_spike_atr"] * atr_value:
        ratio = liq_long / oi
        if ratio >= th["liq_oi_ratio"]:
            family_start = _family_start_index(len(b) - 1, "SHORT")
            candidates.append(("LONG_SQUEEZE", "SHORT", ratio, family_start))

    events: list[dict[str, Any]] = []
    for typ, direction, liq_ratio, family_start_index in candidates:
        family_start_ts = int(b["close_time"].iloc[family_start_index])
        family_id = _event_id(symbol, timeframe, typ + "_FAMILY", family_start_ts, family_start_ts)
        soft = _soft_factors(direction)
        soft_hits = sum(1 for k in ("funding_extreme", "ls_crowded") if soft[k])
        # OI growth alone is leverage buildup, not proof that a liquidation
        # cascade is occurring. Require directional crowding evidence.
        if soft_hits < 1:
            continue

        events.append(
            {
                "event_id": family_id,
                "squeeze_family_id": family_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction,
                "event_type": typ,
                "timestamps": {
                    "pivot_1_ts": detected_ts,
                    "pivot_2_ts": detected_ts,
                    "detected_at_ts": detected_ts,
                    "squeeze_family_start_ts": family_start_ts,
                    "squeeze_family_end_ts": detected_ts,
                },
                "event_fact": {
                    "detection_close_price": close_val,
                    "spike_move": float(move),
                    "spike_atr_mult": float(abs(move) / atr_value),
                    "liq_ratio_24h": float(liq_ratio),
                    "liq_short24": liq_short,
                    "liq_long24": liq_long,
                    "open_interest": oi,
                    **soft,
                },
            }
        )

    return events


def diagnose_15m_trigger(
    df15: pd.DataFrame,
    direction: str,
    event_detected_at_ts: int | None = None,
    max_trigger_delay_min: float = 30.0,
    min_vol_mult: float = 1.05,
    require_event_ts: bool = False,
) -> dict[str, Any]:

    result: dict[str, Any] = {
        "ok": False,
        "reason": None,
        "direction": str(direction).upper(),
        "event_detected_at_ts": event_detected_at_ts,
        "max_trigger_delay_min": float(max_trigger_delay_min),
        # Audit fix B7: make the last-bar fallback explicit so backtests can
        # detect (or forbid) look-ahead instead of silently reading a "future"
        # bar. require_event_ts=True hard-fails the fallback path.
        "event_ts_missing": event_detected_at_ts is None,
        "bars_after_event": 0,
        "bars_considered": 0,
        "trigger_bar_close_ts": None,
        "trigger_delay_min": None,
        "previous_high": None,
        "previous_low": None,
        "current_close": None,
        "current_volume": None,
        "volume_sma20": None,
        "volume_ratio": None,
        "breakout_pass": False,
        "volume_pass": True,
        "data_pass": True,
        "direction_pass": True,
        "scanned_bar_details": [],
    }

    if require_event_ts and event_detected_at_ts is None:
        result["reason"] = "event_ts_required"
        result["data_pass"] = False
        return result

    if not isinstance(df15, pd.DataFrame):
        result["reason"] = "invalid_15m_data"
        result["data_pass"] = False
        return result

    if len(df15) < 2:
        result["reason"] = "insufficient_data"
        result["data_pass"] = False
        return result

    d = str(direction).upper()
    if d not in {"LONG", "SHORT"}:
        result["reason"] = "invalid_direction"
        result["direction_pass"] = False
        return result

    required = {"close", "high", "low"}
    if event_detected_at_ts is not None:
        required.add("close_time")

    missing = required - set(df15.columns)
    if missing:
        result["reason"] = "invalid_15m_data"
        result["data_pass"] = False
        return result

    work = df15.copy()
    for col in ("close", "high", "low"):
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if "volume" in work.columns:
        work["volume"] = pd.to_numeric(work["volume"], errors="coerce")

    if "close_time" in work.columns:
        work["close_time"] = pd.to_numeric(work["close_time"], errors="coerce")

    work = work.dropna(subset=["close", "high", "low"]).reset_index(drop=True)
    if len(work) < 2:
        result["reason"] = "insufficient_data"
        result["data_pass"] = False
        return result

    if event_detected_at_ts is None:
        i = len(work) - 1
        h = work.iloc[i]
        p = work.iloc[i - 1]

        try:
            current_close = float(h["close"])
            previous_high = float(p["high"])
            previous_low = float(p["low"])
        except (TypeError, ValueError):
            result["reason"] = "invalid_15m_data"
            result["data_pass"] = False
            return result

        current_volume = float(h["volume"]) if ("volume" in work.columns and pd.notna(h["volume"])) else None

        if d == "LONG":
            breakout_pass = current_close > previous_high
        else:
            breakout_pass = current_close < previous_low

        volume_sma20 = None
        volume_ratio = None
        volume_pass = True

        if "volume" in work.columns and len(work) >= 20 and min_vol_mult > 0:
            volume_window = pd.to_numeric(work["volume"].iloc[-21:-1], errors="coerce")
            if len(volume_window) == 20 and volume_window.notna().all():
                volume_sma20 = float(volume_window.mean())
                if volume_sma20 > 0 and current_volume is not None:
                    volume_ratio = current_volume / volume_sma20
                    volume_pass = current_volume >= (volume_sma20 * min_vol_mult)

        result["bars_after_event"] = 1
        result["bars_considered"] = 1
        result["breakout_pass"] = bool(breakout_pass)
        result["volume_pass"] = bool(volume_pass)
        result["previous_high"] = previous_high
        result["previous_low"] = previous_low
        result["current_close"] = current_close
        result["current_volume"] = current_volume
        result["volume_sma20"] = volume_sma20
        result["volume_ratio"] = volume_ratio

        if not breakout_pass:
            result["reason"] = "breakout_failed"
            return result
        if not volume_pass:
            result["reason"] = "volume_failed"
            return result

        result["ok"] = True
        result["reason"] = "passed"
        return result

    try:
        event_ts = int(event_detected_at_ts)
    except (TypeError, ValueError):
        result["reason"] = "invalid_event_timestamp"
        result["data_pass"] = False
        return result

    work = work.dropna(subset=["close_time"]).sort_values("close_time").reset_index(drop=True)
    if len(work) < 2:
        result["reason"] = "insufficient_data"
        result["data_pass"] = False
        return result

    candidate_indices: list[int] = []
    for i in range(1, len(work)):
        close_ts = int(work["close_time"].iloc[i])
        if close_ts <= event_ts:
            continue
        delay_min = (close_ts - event_ts) / 60000.0
        if delay_min < 0 or delay_min > max_trigger_delay_min:
            continue
        candidate_indices.append(i)

    result["bars_after_event"] = len(candidate_indices)
    if not candidate_indices:
        result["reason"] = "no_trigger_window"
        return result

    saw_breakout = False
    saw_breakout_without_volume = False
    last_diagnostic = None

    for i in reversed(candidate_indices):
        h = work.iloc[i]
        p = work.iloc[i - 1]
        try:
            current_close = float(h["close"])
            previous_high = float(p["high"])
            previous_low = float(p["low"])
            close_ts = int(h["close_time"])
        except (TypeError, ValueError):
            continue

        current_volume = float(h["volume"]) if ("volume" in work.columns and pd.notna(h["volume"])) else None
        delay_min = (close_ts - event_ts) / 60000.0

        if d == "LONG":
            breakout_pass = current_close > previous_high
        else:
            breakout_pass = current_close < previous_low

        volume_sma20 = None
        volume_ratio = None
        volume_pass = True

        if "volume" in work.columns and i >= 20 and min_vol_mult > 0:
            volume_window = pd.to_numeric(work["volume"].iloc[i - 20:i], errors="coerce")
            if len(volume_window) == 20 and volume_window.notna().all():
                volume_sma20 = float(volume_window.mean())
                if volume_sma20 > 0 and current_volume is not None:
                    volume_ratio = current_volume / volume_sma20
                    volume_pass = current_volume >= (volume_sma20 * min_vol_mult)

        detail = {
            "close_ts": close_ts,
            "delay_min": round(delay_min, 3),
            "previous_high": previous_high,
            "previous_low": previous_low,
            "current_close": current_close,
            "current_volume": current_volume,
            "volume_sma20": volume_sma20,
            "volume_ratio": volume_ratio,
            "breakout_pass": bool(breakout_pass),
            "volume_pass": bool(volume_pass),
        }
        result["scanned_bar_details"].append(detail)
        result["bars_considered"] += 1
        last_diagnostic = detail

        if not breakout_pass:
            continue

        saw_breakout = True
        if not volume_pass:
            saw_breakout_without_volume = True
            continue

        result.update(
            {
                "ok": True,
                "reason": "passed",
                "breakout_pass": True,
                "volume_pass": True,
                "trigger_bar_close_ts": close_ts,
                "trigger_delay_min": round(delay_min, 3),
                "previous_high": previous_high,
                "previous_low": previous_low,
                "current_close": current_close,
                "current_volume": current_volume,
                "volume_sma20": volume_sma20,
                "volume_ratio": volume_ratio,
            }
        )
        return result

    if saw_breakout and saw_breakout_without_volume:
        result["reason"] = "volume_failed"
    else:
        result["reason"] = "breakout_failed"

    if last_diagnostic:
        result.update(
            {
                "previous_high": last_diagnostic["previous_high"],
                "previous_low": last_diagnostic["previous_low"],
                "current_close": last_diagnostic["current_close"],
                "current_volume": last_diagnostic["current_volume"],
                "volume_sma20": last_diagnostic["volume_sma20"],
                "volume_ratio": last_diagnostic["volume_ratio"],
                "breakout_pass": bool(saw_breakout),
                "volume_pass": not saw_breakout_without_volume,
            }
        )

    return result


def build_15m_trigger(
    df15: pd.DataFrame,
    direction: str,
    min_vol_mult: float = 1.0,
    event_detected_at_ts: int | None = None,
    max_trigger_delay_min: float = 30.0,
    require_event_ts: bool = False,
) -> bool:
    diagnostic = diagnose_15m_trigger(
        df15=df15,
        direction=direction,
        event_detected_at_ts=event_detected_at_ts,
        max_trigger_delay_min=max_trigger_delay_min,
        min_vol_mult=min_vol_mult,
        require_event_ts=require_event_ts,
    )
    return bool(diagnostic.get("ok", False))


def check_btc_regime(
    btc_1h_df: pd.DataFrame,
    direction: str,
) -> tuple[bool, str]:

    if len(btc_1h_df) < 5 or "close" not in btc_1h_df.columns:
        return True, "INSUFFICIENT_DATA"

    close = pd.to_numeric(btc_1h_df["close"], errors="coerce")
    if close.isna().any():
        return True, "INSUFFICIENT_DATA"

    last_close = float(close.iloc[-1])
    prev_1h = float(close.iloc[-2])
    prev_4h = float(close.iloc[-5])

    if last_close <= 0 or prev_1h <= 0 or prev_4h <= 0:
        return True, "INSUFFICIENT_DATA"

    chg_1h_pct = ((last_close - prev_1h) / prev_1h) * 100.0
    chg_4h_pct = ((last_close - prev_4h) / prev_4h) * 100.0

    d = str(direction).upper()
    if d == "LONG":
        if chg_1h_pct < -1.2:
            return False, f"BTC_DUMPING_1H ({chg_1h_pct:.2f}%)"
        if chg_4h_pct < -2.5:
            return False, f"BTC_DUMPING_4H ({chg_4h_pct:.2f}%)"
    elif d == "SHORT":
        if chg_1h_pct > 1.5:
            return False, f"BTC_PUMPING_1H (+{chg_1h_pct:.2f}%)"
        if chg_4h_pct > 3.0:
            return False, f"BTC_PUMPING_4H (+{chg_4h_pct:.2f}%)"

    return True, "OK"


def validate_divergence_context(
    ev: dict[str, Any],
    df_htf: pd.DataFrame | None = None,
    context_timeframe: str = "4h",
) -> tuple[bool, str, dict[str, Any]]:
    """Validate divergence semantics: hidden requires aligned trend; regular needs non-trending/exhaustion context.

    The validator is intentionally conservative and causal. When higher-TF data is
    supplied, it is preferred for regime confirmation. Otherwise the event is not
    rejected solely for missing context; the caller can decide whether to require it.
    """
    event_type = str(ev.get("event_type", "")).upper()
    direction = str(ev.get("direction", "")).upper()
    fact = ev.get("event_fact") if isinstance(ev.get("event_fact"), dict) else {}
    meta: dict[str, Any] = {"context_validated": False, "context_source": None}
    if "REGULAR_" not in event_type and "HIDDEN_" not in event_type:
        return True, "NOT_DIVERGENCE", meta

    if isinstance(df_htf, pd.DataFrame) and len(df_htf) >= 60:
        ctx = _trend_context(df_htf, direction)
        meta.update({"context_source": str(context_timeframe).lower(), "htf_trend": ctx.get("trend"), "htf_trend_ok": ctx.get("trend_ok")})
    else:
        # Local event context fallback: do not fabricate a trend decision.
        ctx = _trend_context(pd.DataFrame(), direction)

    if "HIDDEN_" in event_type:
        if ctx.get("trend_ok") is None:
            return False, "HIDDEN_CONTEXT_UNAVAILABLE", meta
        if not bool(ctx["trend_ok"]):
            return False, "HIDDEN_TREND_MISMATCH", meta
        meta["context_validated"] = True
        return True, "HIDDEN_TREND_ALIGNED", meta

    # Regular divergence is a reversal candidate. Avoid forcing a strong-trend veto,
    # but require meaningful price displacement and a non-flat divergence geometry.
    price_delta_atr = float(fact.get("price_delta_atr", 0.0) or 0.0)
    if price_delta_atr < 0.35:
        return False, "REGULAR_WEAK_PRICE_DISPLACEMENT", meta
    meta["context_validated"] = True
    meta["reversal_candidate"] = True
    return True, "REGULAR_CONTEXT_OK", meta
