from __future__ import annotations

import numpy as np
import pandas as pd
from .indicators import atr


def build_setup(event: dict, df15: pd.DataFrame, sl_atr_buffer: float, rr: float) -> dict | None:
    if df15.empty:
        return None
    detected = int(event['timestamps']['detected_at_ts'])
    w = df15[df15['close_time'] <= detected].copy()
    if len(w) < 30:
        return None
    a = atr(w, 14)
    av = float(a.iloc[-1]) if np.isfinite(a.iloc[-1]) else None
    if av is None or av <= 0:
        return None
    entry = float(event['event_fact']['detection_close_price'])
    recent = w.tail(5)
    direction = event['direction']
    if direction == 'LONG':
        swing = float(recent['low'].min())
        invalidation = swing - sl_atr_buffer * av
        risk = entry - invalidation
        target = entry + rr * risk if risk > 0 else 0
        trigger = float(recent.iloc[-2]['high']) if len(recent) >= 2 else None
        trigger_ok = bool(entry > trigger) if trigger is not None else False
    else:
        swing = float(recent['high'].max())
        invalidation = swing + sl_atr_buffer * av
        risk = invalidation - entry
        target = entry - rr * risk if risk > 0 else 0
        trigger = float(recent.iloc[-2]['low']) if len(recent) >= 2 else None
        trigger_ok = bool(entry < trigger) if trigger is not None else False
    if risk <= 0 or target <= 0:
        return None
    return {
        'event_id': event.get('event_id'),
        'symbol': event['symbol'],
        'direction': direction,
        'created_at_ts': detected,
        'entry_reference': entry,
        'invalidation_price': invalidation,
        'target_price': target,
        'risk_pct': abs(entry - invalidation) / entry * 100.0,
        'rr': rr,
        'trigger_reference': trigger,
        'trigger_ok': trigger_ok,
    }
