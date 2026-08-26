from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr, rsi


def causal_pivots(df: pd.DataFrame, left: int = 3, right: int = 2) -> dict:
    highs: list[dict] = []
    lows: list[dict] = []
    for i in range(left, len(df) - right):
        hi = float(df['high'].iloc[i])
        lo = float(df['low'].iloc[i])
        if hi >= float(df['high'].iloc[i-left:i].max()) and hi > float(df['high'].iloc[i+1:i+right+1].max()):
            highs.append({
                'index': i,
                'price': hi,
                'pivot_ts': int(df['close_time'].iloc[i]),
                'detected_at_ts': int(df['close_time'].iloc[i + right]),
            })
        if lo <= float(df['low'].iloc[i-left:i].min()) and lo < float(df['low'].iloc[i+1:i+right+1].min()):
            lows.append({
                'index': i,
                'price': lo,
                'pivot_ts': int(df['close_time'].iloc[i]),
                'detected_at_ts': int(df['close_time'].iloc[i + right]),
            })
    return {'high_pivots': highs, 'low_pivots': lows}


def _build_event(df: pd.DataFrame, p1: dict, p2: dict, indicator_col: str,
                 indicator_label: str, direction: str, params: dict) -> dict | None:
    p1i, p2i = p1['index'], p2['index']
    atr_p2 = float(df['atr'].iloc[p2i])
    if not np.isfinite(atr_p2) or atr_p2 <= 0:
        return None
    price_delta = abs(float(p2['price']) - float(p1['price']))
    if price_delta / atr_p2 < params['min_price_delta_atr']:
        return None
    i1 = float(df[indicator_col].iloc[p1i])
    i2 = float(df[indicator_col].iloc[p2i])
    if not (np.isfinite(i1) and np.isfinite(i2)):
        return None

    if direction == 'LONG':
        valid = p2['price'] < p1['price'] and i2 > i1
        event_type = f'REGULAR_BULLISH_{indicator_label}'
    else:
        valid = p2['price'] > p1['price'] and i2 < i1
        event_type = f'REGULAR_BEARISH_{indicator_label}'
    if not valid:
        return None

    det_idx = p2i + params['pivot_right']
    if det_idx >= len(df):
        return None
    atr_det = float(df['atr'].iloc[det_idx]) if np.isfinite(df['atr'].iloc[det_idx]) else None
    return {
        'symbol': params['symbol'],
        'timeframe': params['timeframe'],
        'direction': direction,
        'event_type': event_type,
        'detector_params': {
            'pivot_left': params['pivot_left'],
            'pivot_right': params['pivot_right'],
            'min_bars_between': params['min_bars_between'],
            'max_bars_between': params['max_bars_between'],
            'min_price_delta_atr': params['min_price_delta_atr'],
            'pivot_pairing_mode': 'consecutive',
        },
        'timestamps': {
            'pivot_1_ts': int(p1['pivot_ts']),
            'pivot_2_ts': int(p2['pivot_ts']),
            'detected_at_ts': int(p2['detected_at_ts']),
        },
        'event_fact': {
            'detection_close_price': float(df['close'].iloc[det_idx]),
            'p1_price': float(p1['price']),
            'p2_price': float(p2['price']),
            'p1_indicator': i1,
            'p2_indicator': i2,
            'bars_between': int(p2i - p1i),
            'price_delta_pct': (float(p2['price']) - float(p1['price'])) / float(p1['price']) * 100.0,
            'price_delta_atr': price_delta / atr_p2,
            'indicator_delta_raw': i2 - i1,
            'atr_at_pivot_2': atr_p2,
            'atr_at_detection': atr_det,
        },
    }


def _detect(df: pd.DataFrame, symbol: str, timeframe: str, indicator_col: str,
            indicator_label: str, params: dict) -> list[dict]:
    if len(df) < params['max_bars_between'] + params['pivot_left'] + params['pivot_right'] + 15:
        return []
    w = df.copy()
    if indicator_col == 'rsi':
        w['rsi'] = rsi(w['close'], 14)
    w['atr'] = atr(w, 14)
    piv = causal_pivots(w, params['pivot_left'], params['pivot_right'])
    events: list[dict] = []
    for pivots, direction in ((piv['low_pivots'], 'LONG'), (piv['high_pivots'], 'SHORT')):
        for p1, p2 in zip(pivots[:-1], pivots[1:]):
            gap = p2['index'] - p1['index']
            if not (params['min_bars_between'] <= gap <= params['max_bars_between']):
                continue
            if indicator_col == 'bingx_cvd':
                start, end = p1['index'], p2['index'] + params['pivot_right']
                if not w['taker_flow_valid'].iloc[start:end+1].all():
                    continue
                if int(w['cvd_segment_id'].iloc[start]) != int(w['cvd_segment_id'].iloc[end]):
                    continue
            ev = _build_event(w, p1, p2, indicator_col, indicator_label, direction,
                              {**params, 'symbol': symbol, 'timeframe': timeframe})
            if ev:
                events.append(ev)
    return events


def detect_rsi_divergences(df: pd.DataFrame, symbol: str, timeframe: str, params: dict) -> list[dict]:
    return _detect(df, symbol, timeframe, 'rsi', 'RSI', params)


def detect_cvd_divergences(df: pd.DataFrame, symbol: str, timeframe: str, params: dict) -> list[dict]:
    if 'bingx_cvd' not in df.columns or 'taker_flow_valid' not in df.columns:
        return []
    return _detect(df, symbol, timeframe, 'bingx_cvd', 'BINGX_CVD', params)
