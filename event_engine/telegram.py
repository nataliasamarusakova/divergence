from __future__ import annotations

import html
import os
from typing import Any, Optional

import requests


def _chat_ids() -> list[str]:
    raw = os.environ.get("TG_CHAT_IDS") or os.environ.get("TG_CHAT_ID") or ""
    return [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]


def send(text: str) -> bool:
    token = os.environ.get("TG_BOT_TOKEN", "").strip()
    ids = _chat_ids()
    if not token or not ids:
        print("[TELEGRAM] missing TG_BOT_TOKEN or TG_CHAT_IDS")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    all_ok = True
    for chat_id in ids:
        try:
            r = requests.post(
                url,
                data={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": "true",
                },
                timeout=15,
            )
            r.raise_for_status()
            payload = r.json()
            if not payload.get("ok"):
                all_ok = False
                print(f"[TELEGRAM] API rejected chat_id={chat_id}: {payload}")
        except Exception as exc:
            all_ok = False
            print(f"[TELEGRAM] send failed chat_id={chat_id}: {exc}")
    return all_ok


def format_signal(
    event: dict[str, Any],
    setup: Optional[dict[str, Any]] = None,
    coinalyze_row: Any = None,
    execution: Optional[dict[str, Any]] = None,
) -> str:
    direction = str(event.get("direction", "")).upper()
    title = "🚨 LONG SIGNAL" if direction == "LONG" else "🔻 SHORT SIGNAL"
    fact = event.get("event_fact", {})
    ts = event.get("timestamps", {})
    setup = setup or {}

    def esc(v: Any) -> str:
        return html.escape("—" if v is None else str(v), quote=False)

    lines = [
        f"<b>{title}</b>",
        "",
        f"<b>{esc(getattr(coinalyze_row, 'name', None) or event.get('symbol'))}</b> "
        f"(<code>{esc(event.get('symbol'))}</code>)",
        f"Event: <code>{esc(event.get('event_type'))}</code>",
        f"TF: <b>{esc(event.get('timeframe', '1h'))}</b> + trigger 15m",
        f"Price: <code>{esc(fact.get('detection_close_price'))}</code>",
        f"Detected: <code>{esc(ts.get('detected_at_ts'))}</code>",
    ]

    if "p1_price" in fact:
        lines += [
            "",
            "<b>Divergence</b>",
            f"P1: <code>{esc(fact.get('p1_price'))}</code>",
            f"P2: <code>{esc(fact.get('p2_price'))}</code>",
            f"Price Δ / ATR: <code>{esc(round(float(fact.get('price_delta_atr', 0)), 3))}</code>",
        ]

    if "squeeze_duration_bars" in fact:
        lines += [
            "",
            "<b>Volatility Squeeze</b>",
            f"Duration: <code>{esc(fact.get('squeeze_duration_bars'))} bars</code>",
            f"BB / KC Width: <code>{esc(round(float(fact.get('compression_ratio', 0)), 3))}</code>",
        ]

    if coinalyze_row is not None:
        oi_chg = getattr(coinalyze_row, "oi_chg4h_pct", None)
        if oi_chg is not None:
            if abs(oi_chg) > 1000:
                oi_chg_str = f"${oi_chg:,.0f}"
            else:
                oi_chg_str = f"{oi_chg:.2f}%"
        else:
            oi_chg_str = "—"

        lines += [
            "",
            "<b>Coinalyze</b>",
            f"Vol24H: <code>{esc(getattr(coinalyze_row, 'volume24', None))}</code>",
            f"OI: <code>{esc(getattr(coinalyze_row, 'oi', None))}</code>",
            f"OI Chg 4H: <code>{esc(oi_chg_str)}</code>",
            f"Funding OI-W: <code>{esc(getattr(coinalyze_row, 'fr_oiw', None))}</code>",
        ]

    if setup:
        rr = setup.get("realized_rr", setup.get("target_rr", 2.0))
        lines += [
            "",
            "<b>SETUP</b>",
            f"Entry: <code>{esc(setup.get('entry_reference'))}</code>",
            f"SL: <code>{esc(setup.get('invalidation_price'))}</code>",
            f"TP (Final): <code>{esc(setup.get('target_price'))}</code>",
            f"R:R (Realized): <code>{esc(rr)}</code>",
        ]

    if execution:
        lines += [
            "",
            "<b>EXECUTION</b>",
            f"Mode: <code>{esc(execution.get('mode', 'vst'))}</code>",
            f"Status: <code>{esc(execution.get('status'))}</code>",
            f"Order: <code>{esc(execution.get('order_id'))}</code>",
        ]

    lines += ["", "⚡ Event-driven — 5×5m lifecycle is NOT used"]
    return "\n".join(lines)
