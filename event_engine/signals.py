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
    avg_gain = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _bbands(series: pd.Series, n: int = 20, std: float = 2.0):
    mid = series.rolling(n, min_periods=n).mean()
    sd = series.rolling(n, min_periods=n).std(ddof=0)
    return mid + std*sd, mid, mid - std*sd



def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([df.high-df.low, (df.high-df.close.shift()).abs(), (df.low-df.close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _pivots(df: pd.DataFrame, left=3, right=2):
    lows, highs = [], []
    for i in range(left, len(df) - right):
        if float(df.low.iloc[i]) <= float(df.low.iloc[i-left:i].min()) and float(df.low.iloc[i]) < float(df.low.iloc[i+1:i+right+1].min()):
            lows.append(i)
        if float(df.high.iloc[i]) >= float(df.high.iloc[i-left:i].max()) and float(df.high.iloc[i]) > float(df.high.iloc[i+1:i+right+1].max()):
            highs.append(i)
    return lows, highs


def _event_id(symbol, tf, typ, p1_ts, p2_ts):
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
        work["taker_flow_valid"]
        .fillna(False)
        .astype(bool)
    )

    # New segment after every invalid taker-flow bar.
    work["cvd_segment_id"] = (
        (~work["taker_flow_valid"])
        .cumsum()
    )

    work["bingx_cvd"] = float("nan")

    for segment_id, idx in work.groupby(
        "cvd_segment_id",
        sort=False,
    ).groups.items():

        valid_idx = [
            i
            for i in idx
            if bool(
                work.at[i, "taker_flow_valid"]
            )
            and pd.notna(
                work.at[i, "bar_delta_usdt"]
            )
        ]

        if not valid_idx:
            continue

        work.loc[
            valid_idx,
            "bingx_cvd"
        ] = work.loc[
            valid_idx,
            "bar_delta_usdt"
        ].cumsum()

    return work


def detect_divergences(df: pd.DataFrame, symbol: str, timeframe: str = "1h", left=3, right=2, min_bars=5, max_bars=35, min_delta_atr=0.25) -> list[dict[str, Any]]:
    if len(df) < 60: return []
    work = df.copy()
    work["rsi"] = _rsi(work["close"], 14)
    work["atr"] = _atr(work)
    lows, highs = _pivots(work, left, right)
    events = []
    def emit(p1i, p2i, direction, indicator, bullish):
        bars = p2i - p1i
        if not min_bars <= bars <= max_bars: return
        atr = float(work.atr.iloc[p2i]) if pd.notna(work.atr.iloc[p2i]) else 0
        if atr <= 0: return
        if bullish:
            if not (work.low.iloc[p2i] < work.low.iloc[p1i]): return
            price_delta_atr = (work.low.iloc[p1i] - work.low.iloc[p2i]) / atr
            if price_delta_atr < min_delta_atr: return
            p1v, p2v = work[indicator].iloc[p1i], work[indicator].iloc[p2i]
            if pd.isna(p1v) or pd.isna(p2v) or not (p2v > p1v): return
            typ = f"REGULAR_BULLISH_{indicator.upper()}"
            direction_ = "LONG"
        else:
            if not (work.high.iloc[p2i] > work.high.iloc[p1i]): return
            price_delta_atr = (work.high.iloc[p2i] - work.high.iloc[p1i]) / atr
            if price_delta_atr < min_delta_atr: return
            p1v, p2v = work[indicator].iloc[p1i], work[indicator].iloc[p2i]
            if pd.isna(p1v) or pd.isna(p2v) or not (p2v < p1v): return
            typ = f"REGULAR_BEARISH_{indicator.upper()}"
            direction_ = "SHORT"
        detected = p2i + right
        if detected >= len(work): return
        if indicator == "bingx_cvd":
            span = work.iloc[p1i:detected+1]
            if not span["bar_delta_usdt"].notna().all() or work.cvd_segment_id.iloc[p1i] != work.cvd_segment_id.iloc[detected]: return
        pfield = "low" if bullish else "high"
        events.append({
            "event_id": _event_id(symbol, timeframe, typ, work.close_time.iloc[p1i], work.close_time.iloc[p2i]),
            "symbol": symbol, "timeframe": timeframe, "direction": direction_, "event_type": typ,
            "timestamps": {"pivot_1_ts": int(work.close_time.iloc[p1i]), "pivot_2_ts": int(work.close_time.iloc[p2i]), "detected_at_ts": int(work.close_time.iloc[detected])},
            "event_fact": {"detection_close_price": float(work.close.iloc[detected]), "p1_price": float(work[pfield].iloc[p1i]), "p2_price": float(work[pfield].iloc[p2i]), "p1_indicator": float(p1v), "p2_indicator": float(p2v), "bars_between": bars, "price_delta_atr": float(price_delta_atr)},
        })
    if len(lows) >= 2:
        for p1,p2 in zip(lows[:-1], lows[1:]):
            emit(p1,p2,"LONG","rsi",True); emit(p1,p2,"LONG","bingx_cvd",True)
    if len(highs) >= 2:
        for p1,p2 in zip(highs[:-1], highs[1:]):
            emit(p1,p2,"SHORT","rsi",False); emit(p1,p2,"SHORT","bingx_cvd",False)
    return events


def detect_squeeze_release(df: pd.DataFrame, symbol: str, timeframe="1h") -> list[dict[str, Any]]:
    if len(df) < 40: return []
    b = df.copy()
    bb_u, mid, bb_l = _bbands(b.close, 20, 2.0)
    atr = _atr(b, 20)
    kc_u = mid + 1.5 * atr
    kc_l = mid - 1.5 * atr
    in_sq = (bb_u <= kc_u) & (bb_l >= kc_l)
    if len(b) < 3 or pd.isna(bb_u.iloc[-1]) or pd.isna(kc_u.iloc[-1]): return []
    release = bool(not in_sq.iloc[-1] and in_sq.iloc[-2])
    if not release: return []
    if b.close.iloc[-1] > kc_u.iloc[-1]: direction="LONG"
    elif b.close.iloc[-1] < kc_l.iloc[-1]: direction="SHORT"
    else: return []
    ts=int(b.close_time.iloc[-1])
    typ="VOLATILITY_SQUEEZE_RELEASE"
    return [{"event_id": _event_id(symbol,timeframe,typ,ts,ts),"symbol":symbol,"timeframe":timeframe,"direction":direction,"event_type":typ,"timestamps":{"pivot_1_ts":ts,"pivot_2_ts":ts,"detected_at_ts":ts},"event_fact":{"detection_close_price":float(b.close.iloc[-1]),"bb_width":float(bb_u.iloc[-1]-bb_l.iloc[-1]),"kc_width":float(kc_u.iloc[-1]-kc_l.iloc[-1])}}]


def build_15m_trigger(df15: pd.DataFrame, direction: str) -> bool:
    if len(df15) < 2: return False
    h = df15.iloc[-1]
    p = df15.iloc[-2]
    if direction == "LONG": return float(h.close) > float(p.high)
    return float(h.close) < float(p.low)
