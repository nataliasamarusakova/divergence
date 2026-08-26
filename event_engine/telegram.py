from __future__ import annotations

import html
import os
from typing import Any
import requests


def _esc(v: Any) -> str:
    return html.escape('—' if v is None else str(v), quote=False)


def _chat_ids() -> list[str]:
    raw = os.getenv('TG_CHAT_IDS') or os.getenv('TG_CHAT_ID') or ''
    return [x.strip() for x in raw.replace(';', ',').split(',') if x.strip()]


def send_message(text: str) -> int:
    token = os.getenv('TG_BOT_TOKEN', '').strip()
    chats = _chat_ids()
    if not token or not chats:
        print('[TELEGRAM] missing TG_BOT_TOKEN/TG_CHAT_IDS')
        return 0
    url = f'https://api.telegram.org/bot{token}/sendMessage'
    ok = 0
    for chat_id in chats:
        try:
            r = requests.post(url, data={
                'chat_id': chat_id,
                'text': text,
                'parse_mode': 'HTML',
                'disable_web_page_preview': 'true',
            }, timeout=15)
            r.raise_for_status()
            if r.json().get('ok'):
                ok += 1
        except Exception as exc:
            print(f'[TELEGRAM] failed chat={chat_id}: {exc}')
    return ok


def format_signal(event: dict, setup: dict, coinalyze=None, execution: dict | None = None) -> str:
    direction = str(event.get('direction', '')).upper()
    title = '🚨 LONG SIGNAL' if direction == 'LONG' else '🔻 SHORT SIGNAL'
    fact = event.get('event_fact', {})
    ts = event.get('timestamps', {})
    lines = [
        f'<b>{title}</b>', '',
        f'<b>{_esc(event.get("symbol"))}</b>',
        f'Event: <code>{_esc(event.get("event_type"))}</code>',
        f'Event TF: <b>{_esc(event.get("timeframe"))}</b>',
        'Trigger TF: <b>15m</b>',
        f'Detection price: <code>{_esc(fact.get("detection_close_price"))}</code>',
        f'Detected: <code>{_esc(ts.get("detected_at_ts"))}</code>',
    ]
    if 'p1_price' in fact:
        lines += [
            '', '<b>Divergence</b>',
            f'P1: <code>{_esc(fact.get("p1_price"))}</code>',
            f'P2: <code>{_esc(fact.get("p2_price"))}</code>',
            f'Price Δ / ATR: <code>{_esc(round(float(fact.get("price_delta_atr", 0)), 3))}</code>',
            f'Indicator Δ: <code>{_esc(round(float(fact.get("indicator_delta_raw", 0)), 4))}</code>',
        ]
    if coinalyze is not None:
        lines += [
            '', '<b>Coinalyze context</b>',
            f'Vol 24H: <code>{_esc(getattr(coinalyze, "volume_24h", None))}</code>',
            f'OI: <code>{_esc(getattr(coinalyze, "open_interest", None))}</code>',
            f'OI Chg 4H: <code>{_esc(getattr(coinalyze, "oi_chg_4h_pct", None))}%</code>',
            f'Funding OI-W: <code>{_esc(getattr(coinalyze, "funding_oiw", None))}</code>',
            f'Short Liqs 24H: <code>{_esc(getattr(coinalyze, "short_liq_24h", None))}</code>',
            f'Long Liqs 24H: <code>{_esc(getattr(coinalyze, "long_liq_24h", None))}</code>',
        ]
    lines += [
        '', '<b>SETUP</b>',
        f'Entry: <code>{_esc(setup.get("entry_reference"))}</code>',
        f'SL: <code>{_esc(setup.get("invalidation_price"))}</code>',
        f'TP: <code>{_esc(setup.get("target_price"))}</code>',
        f'Risk: <code>{_esc(round(float(setup.get("risk_pct", 0)), 3))}%</code>',
        f'R:R: <code>{_esc(setup.get("rr"))}</code>',
        f'15m Trigger: <b>{"✅" if setup.get("trigger_ok") else "❌"}</b>',
    ]
    if execution:
        lines += [
            '', '<b>EXECUTION</b>',
            f'Mode: <b>{_esc(execution.get("mode"))}</b>',
            f'Order: <code>{_esc(execution.get("order_id"))}</code>',
            f'Status: <b>{_esc(execution.get("status"))}</b>',
        ]
    lines += ['', '⚡ Event-driven — 5×5m lifecycle is NOT used']
    return '\n'.join(lines)
