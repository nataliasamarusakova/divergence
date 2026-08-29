# telegram.py

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
        except Exception:
            all_ok = False
    return all_ok


def format_signal(
    event: dict[str, Any],
    setup: Optional[dict[str, Any]] = None,
    coinalyze_row: Any = None,
    execution: Optional[dict[str, Any]] = None,
    score: Optional[float] = None,
) -> str:
    direction = str(event.get("direction", "")).upper()
    header_prefix = "🟢 LONG" if direction == "LONG" else "🔴 SHORT"

    fact = event.get("event_fact", {})
    ts = event.get("timestamps", {})
    setup = setup or {}
    execution = execution or {}

    def esc(v: Any) -> str:
        return html.escape("—" if v is None or v == "" else str(v), quote=False)

    name = getattr(coinalyze_row, "name", None) or event.get("symbol", "")
    symbol = event.get("symbol", "")
    event_type = event.get("event_type", "")
    timeframe = event.get("timeframe", "1h")
    price = fact.get("detection_close_price") or fact.get("close")
    detected_ts = ts.get("detected_at_ts")

    require_trig = os.environ.get("REQUIRE_15M_TRIGGER", "true").lower() == "true"
    trigger_suffix = " + trigger 15m (Vol Confirmed)" if require_trig else ""
    score_str = f"{score:.0f}/100" if score is not None else "—"

    lines = [
        f"<b>{header_prefix} - {esc(name)} ({esc(symbol)})</b>",
        "",
        f"Score: <b>{score_str}</b>",
        f"Event: <code>{esc(event_type)}</code>",
        f"TF: <b>{esc(timeframe)}</b>{trigger_suffix}",
        f"Price: <code>{esc(price)}</code>",
        f"Detected: <code>{esc(detected_ts)}</code>",
    ]

    # Причина открытия сделки (Divergence или Squeeze)
    if "p1_price" in fact:
        lines.extend([
            "",
            "<b>Divergence</b>",
            f"P1: <code>{esc(fact.get('p1_price'))}</code>",
            f"P2: <code>{esc(fact.get('p2_price'))}</code>",
            f"Price Δ / ATR: <code>{esc(round(float(fact.get('price_delta_atr', 0)), 3))}</code>",
        ])
    elif "squeeze_duration_bars" in fact:
        lines.extend([
            "",
            "<b>Volatility Squeeze</b>",
            f"Duration: <code>{esc(fact.get('squeeze_duration_bars'))} bars</code>",
            f"BB / KC Width: <code>{esc(round(float(fact.get('compression_ratio', 0)), 3))}</code>",
        ])

    if setup:
        rr = setup.get("planned_weighted_rr", setup.get("realized_rr", setup.get("target_rr", 1.55)))
        lines.extend([
            "",
            "<b>SETUP</b>",
            f"Entry: <code>{esc(setup.get('entry_reference'))}</code>",
            f"SL: <code>{esc(setup.get('invalidation_price'))}</code>",
            f"TP: <code>{esc(setup.get('target_price'))}</code>",
            f"R:R (Planned Weighted): <code>{esc(rr)}</code>",
        ])

    if execution:
        order_id = execution.get("order_id")
        lines.extend([
            "",
            "<b>EXECUTION</b>",
            f"Mode: <code>{esc(execution.get('mode', 'vst'))}</code>",
            f"Status: <code>{esc(execution.get('status'))}</code>",
            f"Order: <code>{esc(order_id)}</code>",
        ])

    return "\n".join(lines)
