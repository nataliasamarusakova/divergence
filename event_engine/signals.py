from __future__ import annotations

import hashlib
import time
from typing import Any
import numpy as np
import pandas as pd


def _rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    zero_loss = (avg_loss == 0) & (avg_gain > 0)
    zero_both = (avg_loss == 0) & (avg_gain == 0)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.where(~zero_loss, 100.0)
    rsi = rsi.where(~zero_both, 50.0)
    return rsi.fillna(50.0)


def _bbands(series: pd.Series, n: int = 20, std: float = 2.0):
    mid = series.rolling(n, min_periods=n).mean()
    sd = series.rolling(n, min_periods=n).std(ddof=0)
    return mid + std * sd, mid, mid - std * sd


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df.high - df.low,
            (df.high - df.close.shift()).abs(),
            (df.low - df.close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _pivots(df: pd.DataFrame, left: int = 3, right: int = 2):
    lows, highs = [], []
    for i in range(left, len(df) - right):
        if float(df.low.iloc[i]) <= float(df.low.iloc[i - left : i].min()) and float(df.low.iloc[i]) < float(df.low.iloc[i + 1 : i + right + 1].min()):
            lows.append(i)
        if float(df.high.iloc[i]) >= float(df.high.iloc[i - left : i].max()) and float(df.high.iloc[i]) > float(df.high.iloc[i + 1 : i + right + 1].max()):
            highs.append(i)
    return lows, highs


def _event_id(symbol: str, tf: str, typ: str, p1_ts: int, p2_ts: int) -> str:
    fp = f"{symbol}:{tf}:{typ}:{int(p1_ts)}:{int(p2_ts)}"
    return "EVT_" + hashlib.sha256(fp.encode()).hexdigest()[:16].upper()


def add_cvd(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if "bar_delta_usdt" not in work.columns:
        work["bingx_cvd"] = float("nan")
        work["cvd_segment_id"] = 0
        return work

    if "taker_flow_valid" not in work.columns:
        work["taker_flow_valid"] = False

    work["taker_flow_valid"] = (
        work["taker_flow_valid"].fillna(False).astype(bool)
    )

    work["cvd_segment_id"] = (~work["taker_flow_valid"]).cumsum()
    work["bingx_cvd"] = float("nan")

    for _, idx in work.groupby("cvd_segment_id", sort=False).groups.items():
        valid_idx = [
            i
            for i in idx
            if bool(work.at[i, "taker_flow_valid"])
            and pd.notna(work.at[i, "bar_delta_usdt"])
        ]
        if not valid_idx:
            continue

        work.loc[valid_idx, "bingx_cvd"] = work.loc[valid_idx, "bar_delta_usdt"].cumsum()

    return work


def detect_divergences(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str = "1h",
    left: int = 3,
    right: int = 2,
    min_bars: int = 5,
    max_bars: int = 35,
    min_delta_atr: float = 0.25,
) -> list[dict[str, Any]]:
    if len(df) < 60:
        return []

    work = df.copy()
    work["rsi"] = _rsi(work["close"], 14)
    work["atr"] = _atr(work, 14)
    lows, highs = _pivots(work, left, right)
    events: list[dict[str, Any]] = []

    def emit_divergence(p1i: int, p2i: int, indicator: str, is_low: bool):
        bars = p2i - p1i
        if not (min_bars <= bars <= max_bars):
            return

        atr = float(work.atr.iloc[p2i]) if pd.notna(work.atr.iloc[p2i]) else 0.0
        if atr <= 0:
            return

        p1v = work[indicator].iloc[p1i]
        p2v = work[indicator].iloc[p2i]
        if pd.isna(p1v) or pd.isna(p2v):
            return

        detected = p2i + right
        if detected >= len(work):
            return

        if indicator == "bingx_cvd":
            span = work.iloc[p1i : detected + 1]
            if not span["bar_delta_usdt"].notna().all() or work.cvd_segment_id.iloc[p1i] != work.cvd_segment_id.iloc[detected]:
                return

        p1_price = float(work["low" if is_low else "high"].iloc[p1i])
        p2_price = float(work["low" if is_low else "high"].iloc[p2i])
        price_delta_atr = abs(p2_price - p1_price) / atr

        if price_delta_atr < min_delta_atr:
            return

        typ = None
        direction = None

        if is_low:
            if p2_price < p1_price and p2v > p1v:
                typ = f"REGULAR_BULLISH_{indicator.upper()}"
                direction = "LONG"
            elif p2_price > p1_price and p2v < p1v:
                typ = f"HIDDEN_BULLISH_{indicator.upper()}"
                direction = "LONG"
        else:
            if p2_price > p1_price and p2v < p1v:
                typ = f"REGULAR_BEARISH_{indicator.upper()}"
                direction = "SHORT"
            elif p2_price < p1_price and p2v > p1v:
                typ = f"HIDDEN_BEARISH_{indicator.upper()}"
                direction = "SHORT"

        if not typ or not direction:
            return

        p1_ts = int(work.close_time.iloc[p1i])
        p2_ts = int(work.close_time.iloc[p2i])
        detected_ts = int(work.close_time.iloc[detected])

        events.append({
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
                "detection_close_price": float(work.close.iloc[detected]),
                "p1_price": p1_price,
                "p2_price": p2_price,
                "p1_indicator": float(p1v),
                "p2_indicator": float(p2v),
                "bars_between": bars,
                "price_delta_atr": float(price_delta_atr),
            },
        })

    def scan_pivots(pivots: list[int], is_low: bool):
        num_piv = len(pivots)
        for i in range(num_piv):
            for step in (1, 2):
                if i + step < num_piv:
                    p1, p2 = pivots[i], pivots[i + step]
                    emit_divergence(p1, p2, "rsi", is_low)
                    emit_divergence(p1, p2, "bingx_cvd", is_low)

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
) -> list[dict[str, Any]]:
    if len(df) < 40:
        return []

    b = df.copy()
    bb_u, mid, bb_l = _bbands(b.close, 20, 2.0)
    atr = _atr(b, 20)
    kc_u = mid + 1.5 * atr
    kc_l = mid - 1.5 * atr

    in_sq = (bb_u <= kc_u) & (bb_l >= kc_l)
    if len(b) < min_squeeze_bars + 2 or pd.isna(bb_u.iloc[-1]) or pd.isna(kc_u.iloc[-1]):
        return []

    is_currently_in = bool(in_sq.iloc[-1])
    was_previously_in = bool(in_sq.iloc[-2])
    if is_currently_in or not was_previously_in:
        return []

    sq_duration = 0
    idx = len(in_sq) - 2
    while idx >= 0 and in_sq.iloc[idx]:
        sq_duration += 1
        idx -= 1

    if sq_duration < min_squeeze_bars:
        return []

    close_val = float(b.close.iloc[-1])
    kc_u_val = float(kc_u.iloc[-1])
    kc_l_val = float(kc_l.iloc[-1])

    if close_val > kc_u_val:
        direction = "LONG"
    elif close_val < kc_l_val:
        direction = "SHORT"
    else:
        return []

    ts = int(b.close_time.iloc[-1])
    typ = "VOLATILITY_SQUEEZE_RELEASE"
    bb_w = float(bb_u.iloc[-1] - bb_l.iloc[-1])
    kc_w = float(kc_u.iloc[-1] - kc_l.iloc[-1])

    return [{
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
            "compression_ratio": float(bb_w / kc_w) if kc_w > 0 else 1.0,
        },
    }]


def build_15m_trigger(df15: pd.DataFrame, direction: str, min_vol_mult: float = 1.0) -> bool:
    """15M-триггер с валидацией пробоя ценой и подтверждением всплеска объема."""
    if len(df15) < 2:
        return False

    d = str(direction).upper()
    h = df15.iloc[-1]
    p = df15.iloc[-2]

    # 1. Пробой локального экстремума
    price_confirmed = (float(h.close) > float(p.high)) if d == "LONG" else (float(h.close) < float(p.low))
    if not price_confirmed:
        return False

    # 2. Объемное подтверждение (Volume Filter)
    if "volume" in df15.columns and len(df15) >= 20 and min_vol_mult > 0:
        vol_sma = df15["volume"].iloc[-21:-1].mean()
        if pd.notna(vol_sma) and vol_sma > 0:
            if float(h.volume) < vol_sma * min_vol_mult:
                return False

    return True


def check_btc_regime(btc_1h_df: pd.DataFrame, direction: str) -> tuple[bool, str]:
    """Защита от торговли альткоинами против агрессивного импульса Bitcoin."""
    if len(btc_1h_df) < 5:
        return True, "INSUFFICIENT_DATA"

    close = btc_1h_df["close"]
    last_close = float(close.iloc[-1])
    prev_1h = float(close.iloc[-2])
    prev_4h = float(close.iloc[-5])

    chg_1h_pct = ((last_close - prev_1h) / prev_1h) * 100.0
    chg_4h_pct = ((last_close - prev_4h) / prev_4h) * 100.0

    d = direction.upper()
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
