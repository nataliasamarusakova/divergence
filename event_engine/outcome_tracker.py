from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
import time
import pandas as pd

HORIZONS = [15, 30, 60, 120, 240, 480]
MAX_LAG_MIN = 15


def atomic_write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            Path(tmp).unlink(missing_ok=True)
        finally:
            raise


def compute_outcome(event: dict, df15: pd.DataFrame, now_ms: int | None = None) -> dict | None:
    now_ms = now_ms or int(time.time() * 1000)
    if df15.empty:
        return None
    detected = int(event['timestamps']['detected_at_ts'])
    entry = float(event['event_fact']['detection_close_price'])
    direction = event['direction']
    future = df15[df15['close_time'] > detected].copy()
    out = {
        'event_id': event['event_id'],
        'symbol': event['symbol'],
        'direction': direction,
        'detected_at_ts': detected,
        'reference_entry_price': entry,
        'resolution_timeframe': '15m',
        'horizons': {},
    }
    for h in HORIZONS:
        target = detected + h * 60_000
        eligible = future[future['close_time'] >= target]
        if eligible.empty:
            out['horizons'][f'{h}m'] = None
            continue
        bar = eligible.iloc[0]
        actual = int(bar['close_time'])
        if actual - target > MAX_LAG_MIN * 60_000:
            out['horizons'][f'{h}m'] = None
            continue
        window = future[future['close_time'] <= actual]
        close = float(bar['close'])
        if direction == 'LONG':
            ret = (close - entry) / entry * 100
            mfe = (float(window['high'].max()) - entry) / entry * 100
            mae = (float(window['low'].min()) - entry) / entry * 100
        else:
            ret = (entry - close) / entry * 100
            mfe = (entry - float(window['low'].min())) / entry * 100
            mae = (entry - float(window['high'].max())) / entry * 100
        out['horizons'][f'{h}m'] = {
            'available': True,
            'actual_close_ts': actual,
            'lag_ms': actual - target,
            'close_price': close,
            'raw_event_return_pct': round(ret, 4),
            'mfe_pct': round(mfe, 4),
            'mae_pct': round(mae, 4),
        }
    return out
