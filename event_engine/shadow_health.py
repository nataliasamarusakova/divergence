from __future__ import annotations

from pathlib import Path
import json
import time


def snapshot(events_path: Path, outcomes_path: Path) -> dict:
    events = []
    if events_path.exists():
        for line in events_path.read_text(encoding='utf-8').splitlines():
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    structures = {}
    for e in events:
        t = e.get('timestamps', {})
        p = e.get('detector_params', {})
        key = ':'.join(map(str, [
            e.get('symbol'), e.get('timeframe'), e.get('direction'),
            t.get('pivot_1_ts', 0), t.get('pivot_2_ts', 0),
            p.get('pivot_left', 3), p.get('pivot_right', 2), p.get('pivot_pairing_mode', 'consecutive'),
        ]))
        bucket = structures.setdefault(key, set())
        if 'RSI' in e.get('event_type', ''): bucket.add('RSI')
        if 'CVD' in e.get('event_type', ''): bucket.add('CVD')
    outcomes = []
    if outcomes_path.exists():
        for line in outcomes_path.read_text(encoding='utf-8').splitlines():
            try:
                outcomes.append(json.loads(line))
            except Exception:
                pass
    def cov(h):
        if not outcomes: return 0.0
        return round(100 * sum(1 for o in outcomes if o.get('horizons', {}).get(f'{h}m', {}).get('available')) / len(outcomes), 1)
    rsi_only = sum(v == {'RSI'} for v in structures.values())
    cvd_only = sum(v == {'CVD'} for v in structures.values())
    joint = sum(v == {'RSI','CVD'} for v in structures.values())
    return {
        'timestamp': int(time.time()*1000),
        'events': {
            'total': len(events), 'unique_structures': len(structures),
            'rsi_only': rsi_only, 'cvd_only': cvd_only, 'joint': joint,
        },
        'outcomes': {
            'total': len(outcomes), 'coverage_15m_pct': cov(15), 'coverage_60m_pct': cov(60), 'coverage_480m_pct': cov(480),
        },
        'gate': {
            'rsi_only_ready': rsi_only >= 40,
            'cvd_only_ready': cvd_only >= 40,
            'joint_ready': joint >= 30,
            'coverage_60m_ready': cov(60) >= 95,
        },
    }


def append_health(events_path: Path, outcomes_path: Path, health_path: Path):
    s = snapshot(events_path, outcomes_path)
    health_path.parent.mkdir(parents=True, exist_ok=True)
    with health_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')
    e, o, g = s['events'], s['outcomes'], s['gate']
    print(f"[SHADOW_HEALTH] structures={e['unique_structures']} RSI={e['rsi_only']}/40 CVD={e['cvd_only']}/40 Joint={e['joint']}/30 Cov60={o['coverage_60m_pct']}% Ready={all(g.values())}")
