from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).replace([np.inf, -np.inf], np.nan)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev).abs(),
        (df['low'] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def bbands(close: pd.Series, length: int = 20, std_mult: float = 2.0):
    mid = close.rolling(length, min_periods=length).mean()
    std = close.rolling(length, min_periods=length).std(ddof=0)
    return mid, mid + std_mult * std, mid - std_mult * std


def keltner(df: pd.DataFrame, length: int = 20, atr_mult: float = 1.5):
    mid = df['close'].ewm(span=length, adjust=False, min_periods=length).mean()
    a = atr(df, length)
    return mid, mid + atr_mult * a, mid - atr_mult * a
