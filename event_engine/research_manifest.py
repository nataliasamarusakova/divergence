from __future__ import annotations

from pathlib import Path
import json
import time


def write_manifest(path: Path, *, coinalyze_url: str, symbols: list[str], event_count: int,
                   rsi_only: int, cvd_only: int, joint: int, common_blocks: int, coverage_60m: float):
    obj = {
        'manifest_version': 1,
        'freeze_tag': 'EVENT_DRIVEN_ENGINE_1H_15M_TRIGGER',
        'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'coinalyze_url': coinalyze_url,
        'universe': {'symbol_count': len(symbols), 'symbols': symbols},
        'detector_parameters': {
            'pivot_left': 3, 'pivot_right': 2, 'min_bars_between': 5, 'max_bars_between': 35,
            'min_price_delta_atr': 0.25, 'pivot_pairing_mode': 'consecutive',
        },
        'outcome_parameters': {
            'resolution_timeframe': '15m', 'horizons_min': [15,30,60,120,240,480], 'max_lag_min': 15,
        },
        'sample_distribution': {
            'event_count': event_count, 'rsi_only': rsi_only, 'cvd_only': cvd_only,
            'joint_both': joint, 'common_8h_blocks': common_blocks, 'coverage_60m_pct': coverage_60m,
        },
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
