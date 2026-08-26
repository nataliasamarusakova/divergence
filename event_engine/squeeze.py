from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import atr, bbands, keltner


def detect_volatility_squeeze(df: pd.DataFrame, symbol: str, timeframe: str,
                              bb_length: int = 20, bb_std: float = 2.0,
                              kc_length: int = 20, kc_atr_mult: float = 1.5,
                              ratio_max: float = 0.80) -> list[dict]:
    if len(df) < max(bb_length, kc_length) + 5:
        return []
    w = df.copy()
    mid, bu, bl = bbands(w['close'], bb_length, bb_std)
    kmid, ku, kl = keltner(w, kc_length, kc_atr_mult)
    bwidth = bu - bl
    kwidth = ku - kl
    ratio = bwidth / kwidth.replace(0, np.nan)
    in_squeeze = (bu <= ku) & (bl >= kl) & (ratio <= ratio_max)
    release = in_squeeze.shift(1, fill_value=False) & (bwidth > kwidth)
    a = atr(w, 14)
    out: list[dict] = []
    for i in np.flatnonzero(release.to_numpy()):
        direction = 'LONG' if w['close'].iloc[i] > ku.iloc[i] else 'SHORT' if w['close'].iloc[i] < kl.iloc[i] else None
        if direction is None:
            continue
        out.append({
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': direction,
            'event_type': 'VOLATILITY_SQUEEZE_RELEASE',
            'detector_params': {
                'bb_length': bb_length,
                'bb_std': bb_std,
                'kc_length': kc_length,
                'kc_atr_mult': kc_atr_mult,
                'ratio_max': ratio_max,
            },
            'timestamps': {'detected_at_ts': int(w['close_time'].iloc[i])},
            'event_fact': {
                'detection_close_price': float(w['close'].iloc[i]),
                'bb_kc_ratio': float(ratio.iloc[i]) if np.isfinite(ratio.iloc[i]) else None,
                'atr_14': float(a.iloc[i]) if np.isfinite(a.iloc[i]) else None,
            },
        })
    return out
