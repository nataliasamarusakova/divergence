import numpy as np
import pandas as pd

from event_engine.divergence import causal_pivots, detect_cvd_divergences, detect_rsi_divergences
from event_engine.indicators import atr, rsi
from event_engine.squeeze import detect_volatility_squeeze
from event_engine.telegram import format_signal


def make_df(n=80):
    ts = [1700000000000 + i * 3600000 for i in range(n)]
    close = 100 + np.sin(np.arange(n) / 6.0) * 5
    return pd.DataFrame({
        'open_time': ts, 'close_time': ts,
        'open': close, 'high': close + 1, 'low': close - 1, 'close': close,
        'volume': 1000.0, 'quote_volume': close * 1000,
        'taker_buy_base': 500.0, 'taker_buy_quote': close * 500,
        'taker_flow_valid': True, 'cvd_segment_id': 0,
        'bingx_cvd': np.cumsum(np.ones(n) * 100),
    })


def test_pivots_and_indicators():
    df = make_df(100)
    p = causal_pivots(df, 3, 2)
    assert 'high_pivots' in p and 'low_pivots' in p
    assert rsi(df['close']).dropna().size > 0
    assert atr(df).dropna().size > 0


def test_cvd_gap_is_not_silently_valid():
    df = make_df(100)
    df.loc[40, 'taker_flow_valid'] = False
    df.loc[40, 'bingx_cvd'] = np.nan
    params = {
        'pivot_left': 3, 'pivot_right': 2,
        'min_bars_between': 5, 'max_bars_between': 35,
        'min_price_delta_atr': 0.25,
        'symbol': 'TEST-USDT', 'timeframe': '1h',
    }
    assert isinstance(detect_cvd_divergences(df, 'TEST-USDT', '1h', params), list)


def test_squeeze_and_telegram():
    df = make_df(100)
    assert isinstance(detect_volatility_squeeze(df, 'TEST-USDT', '1h'), list)
    msg = format_signal(
        event={
            'symbol': 'TEST-USDT', 'direction': 'LONG', 'event_type': 'REGULAR_BULLISH_RSI',
            'timeframe': '1h', 'timestamps': {'detected_at_ts': 123},
            'event_fact': {'detection_close_price': 100, 'p1_price': 105, 'p2_price': 100, 'price_delta_atr': 1.2, 'indicator_delta_raw': 5},
        },
        setup={'entry_reference': 100, 'invalidation_price': 98, 'target_price': 104, 'risk_pct': 2, 'rr': 2, 'trigger_ok': True},
    )
    assert '🚨 LONG SIGNAL' in msg


def test_short_message():
    msg = format_signal(
        event={'symbol': 'TEST-USDT','direction':'SHORT','event_type':'REGULAR_BEARISH_RSI','timeframe':'1h','timestamps':{'detected_at_ts':1},'event_fact':{'detection_close_price':100}},
        setup={'entry_reference':100,'invalidation_price':102,'target_price':96,'risk_pct':2,'rr':2,'trigger_ok':True},
    )
    assert '🔻 SHORT SIGNAL' in msg
