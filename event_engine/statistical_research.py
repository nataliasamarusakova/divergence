from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Target:
    horizon_min: int
    threshold_pct: float


def load_finalized(path: Path) -> list[dict]:
    if not path.exists(): return []
    out=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            obj=json.loads(line)
            if obj.get('horizons', {}).get('60m', {}).get('available'):
                out.append(obj)
        except Exception:
            pass
    return out


def simple_report(outcomes: list[dict], horizon_min: int = 60, threshold_pct: float = 1.0) -> dict:
    vals=[]
    for o in outcomes:
        h=o.get('horizons', {}).get(f'{horizon_min}m')
        if h and h.get('available'):
            vals.append(float(h['raw_event_return_pct']))
    if not vals: return {'status':'error','error':'no finalized outcomes'}
    a=np.asarray(vals)
    return {
        'status':'ok',
        'n':len(a),
        'p_target':float((a>=threshold_pct).mean()),
        'mean_return_pct':float(a.mean()),
        'median_return_pct':float(np.median(a)),
    }
