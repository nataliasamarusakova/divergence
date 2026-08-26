from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .bingx import BingXClient
from .coinalyze import CoinalyzeClient
from .config import CONFIG
from .dedup import EventDedup
from .divergence import detect_cvd_divergences, detect_rsi_divergences
from .execution import execute_vst
from .setup import build_setup
from .squeeze import detect_volatility_squeeze
from .telegram import format_signal, send_message

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
EVENTS = DATA / 'events.jsonl'
SETUPS = DATA / 'setups.jsonl'
EXECUTIONS = DATA / 'executions.jsonl'
ERRORS = DATA / 'errors.jsonl'
TELEGRAM_SENT = DATA / 'telegram_sent.json'


def _append(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def _event_id(event: dict) -> str:
    ts = event['timestamps']
    fp = ':'.join([
        event['symbol'].upper(),
        event['timeframe'].lower(),
        event['event_type'].upper(),
        str(ts.get('pivot_1_ts', 0)),
        str(ts.get('pivot_2_ts', 0)),
        str(ts['detected_at_ts']),
    ])
    return 'EVT_' + hashlib.sha256(fp.encode()).hexdigest()[:16].upper()


def _contracts_by_asset(client: BingXClient):
    result = {}
    for c in client.contracts():
        if c.symbol.endswith('-USDT') and c.api_state_open:
            result[c.symbol[:-5].upper()] = c
    return result


def _discovery_rows():
    rows = CoinalyzeClient(CONFIG.coinalyze_url).fetch()
    rows = [r for r in rows if (r.volume_24h or 0) >= CONFIG.min_volume_24h and (r.open_interest or 0) >= CONFIG.min_open_interest]
    rows.sort(key=lambda r: (r.volume_24h or 0), reverse=True)
    return rows[:CONFIG.max_candidates]


def _trigger_15m(df15, event):
    detected = int(event['timestamps']['detected_at_ts'])
    w = df15[df15['close_time'] <= detected].tail(3)
    if len(w) < 2:
        return False
    prev = w.iloc[-2]
    last = w.iloc[-1]
    if event['direction'] == 'LONG':
        return float(last['close']) > float(prev['high'])
    return float(last['close']) < float(prev['low'])


def _candidate_has_cvd_confirmation(event, cvd_events):
    if 'RSI' not in event['event_type']:
        return True
    for other in cvd_events:
        if (
            other['direction'] == event['direction']
            and other['timestamps'].get('pivot_1_ts') == event['timestamps'].get('pivot_1_ts')
            and other['timestamps'].get('pivot_2_ts') == event['timestamps'].get('pivot_2_ts')
        ):
            return True
    return False


def run_once():
    DATA.mkdir(parents=True, exist_ok=True)
    client = BingXClient(CONFIG.bingx_api_key, CONFIG.bingx_secret_key, CONFIG.bingx_env, CONFIG.recv_window)
    contracts = _contracts_by_asset(client)
    rows = _discovery_rows()
    dedup = EventDedup(TELEGRAM_SENT)
    trades_this_cycle = 0

    frozen = {
        'pivot_left': CONFIG.pivot_left,
        'pivot_right': CONFIG.pivot_right,
        'min_bars_between': CONFIG.min_bars_between,
        'max_bars_between': CONFIG.max_bars_between,
        'min_price_delta_atr': CONFIG.min_price_delta_atr,
        'pivot_pairing_mode': 'consecutive',
    }

    print(f'[ENGINE] Coinalyze candidates={len(rows)} execution={CONFIG.execution_enabled} env={CONFIG.execution_mode}')

    for row in rows:
        contract = contracts.get(row.symbol.upper())
        if not contract:
            continue
        try:
            df1h = client.klines(contract.symbol, '1h', CONFIG.kline_limit_1h)
            df15 = client.klines(contract.symbol, '15m', CONFIG.kline_limit_15m)
            if df1h.empty or df15.empty:
                continue

            rsi_events = detect_rsi_divergences(df1h, contract.symbol, '1h', frozen)
            cvd_events = detect_cvd_divergences(df1h, contract.symbol, '1h', frozen)
            squeeze_events = detect_volatility_squeeze(
                df1h, contract.symbol, '1h', CONFIG.bb_length, CONFIG.bb_std,
                CONFIG.kc_length, CONFIG.kc_atr_mult, CONFIG.squeeze_ratio_max,
            )
            events = rsi_events + cvd_events + squeeze_events

            for event in events:
                event['event_id'] = _event_id(event)
                age_min = (int(time.time() * 1000) - int(event['timestamps']['detected_at_ts'])) / 60000.0
                if age_min < 0 or age_min > CONFIG.max_event_age_min:
                    continue
                if CONFIG.require_cvd_confirmation and not _candidate_has_cvd_confirmation(event, cvd_events):
                    continue
                if CONFIG.require_15m_trigger and not _trigger_15m(df15, event):
                    continue

                _append(EVENTS, event)
                setup = build_setup(event, df15, CONFIG.sl_atr_buffer, CONFIG.target_r_multiple)
                if not setup or not setup['trigger_ok']:
                    continue
                _append(SETUPS, setup)

                if trades_this_cycle >= CONFIG.max_trades_per_cycle:
                    continue

                execution = None
                if CONFIG.execution_enabled:
                    if CONFIG.execution_mode.lower() != 'vst':
                        raise RuntimeError('This frozen repository supports only VST execution. Keep EXECUTION_MODE=vst.')
                    execution = execute_vst(client, contract, setup, CONFIG.margin_usdt, CONFIG.leverage, CONFIG.position_mode)
                    _append(EXECUTIONS, {
                        'event_id': event['event_id'],
                        'symbol': contract.symbol,
                        'direction': event['direction'],
                        'ts': int(time.time() * 1000),
                        **execution,
                    })
                    trades_this_cycle += 1

                # Telegram is sent when the setup is actionable. If execution is enabled,
                # the message includes the resulting order ID/status.
                if CONFIG.telegram_enabled and not dedup.seen(event['event_id']):
                    msg = format_signal(event, setup, row, execution)
                    delivered = send_message(msg)
                    if delivered > 0:
                        dedup.add(event['event_id'])
                        _append(DATA / 'telegram_sent.jsonl', {
                            'event_id': event['event_id'],
                            'direction': event['direction'],
                            'symbol': contract.symbol,
                            'delivered_to': delivered,
                            'ts': int(time.time() * 1000),
                        })

        except Exception as exc:
            _append(ERRORS, {
                'ts': int(time.time() * 1000),
                'symbol': contract.symbol,
                'error': repr(exc),
            })
            print(f'[ENGINE] ERROR {contract.symbol}: {exc}')

    print(f'[ENGINE] trades_this_cycle={trades_this_cycle}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    run_once()


if __name__ == '__main__':
    main()
