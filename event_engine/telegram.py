from __future__ import annotations

import html
import os
from typing import Any, Optional

import requests


def _chat_ids() -> list[str]:
    raw = os.environ.get("TG_CHAT_IDS") or os.environ.get("TG_CHAT_ID") or ""
    return [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]


def send_detailed(text: str, only_chat_ids: Optional[list[str]] = None) -> dict[str, dict[str, Any]]:
    token = os.environ.get("TG_BOT_TOKEN", "").strip()
    ids = only_chat_ids if only_chat_ids is not None else _chat_ids()
    ids = [str(x).strip() for x in ids if str(x).strip()]
    if not token or not ids:
        print("[TELEGRAM] missing TG_BOT_TOKEN or TG_CHAT_IDS")
        return {str(chat_id): {"sent": False, "error": "missing TG_BOT_TOKEN or TG_CHAT_IDS"} for chat_id in ids}

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    result: dict[str, dict[str, Any]] = {}
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
            try:
                payload = r.json()
            except ValueError:
                payload = {"ok": False, "description": r.text[:500]}
            if payload.get("ok"):
                result[str(chat_id)] = {"sent": True, "message_id": ((payload.get("result") or {}).get("message_id"))}
            else:
                error = str(payload.get("description", "unknown Telegram API error"))
                print(f"[TELEGRAM] API rejected message for chat_id={chat_id}: {error}")
                result[str(chat_id)] = {"sent": False, "error": error}
        except requests.RequestException as exc:
            error = str(exc)
            print(f"[TELEGRAM] Request failed for chat_id={chat_id}: {error}")
            result[str(chat_id)] = {"sent": False, "error": error}
        except Exception as exc:
            error = str(exc)
            print(f"[TELEGRAM] Unexpected send error for chat_id={chat_id}: {error}")
            result[str(chat_id)] = {"sent": False, "error": error}
    return result


def send(text: str) -> bool:
    result = send_detailed(text)
    return bool(result) and all(bool(item.get("sent")) for item in result.values())


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
    ]

    confluence_events = setup.get("confluence_events", []) if isinstance(setup, dict) else []
    if isinstance(confluence_events, list) and confluence_events:
        labels = []
        for item in confluence_events:
            if not isinstance(item, dict):
                continue
            label = f"{str(item.get('timeframe', '1h')).lower()} {str(item.get('event_type', 'EVENT'))}"
            labels.append(f"<code>{esc(label)}</code>")
        if labels:
            lines.append(f"🔗 <b>CONFLUENCE:</b> {' + '.join(labels)}")

    conflict_events = setup.get("conflict_events", []) if isinstance(setup, dict) else []
    if isinstance(conflict_events, list) and conflict_events:
        labels = []
        for item in conflict_events:
            if not isinstance(item, dict):
                continue
            label = f"{str(item.get('timeframe', '1h')).lower()} {str(item.get('direction', ''))}"
            labels.append(f"<code>{esc(label)}</code>")
        if labels:
            lines.append(f"⚠️ <b>CONFLICT:</b> {' + '.join(labels)}")

    lines.extend([
        f"Price: <code>{esc(price)}</code>",
        f"Detected: <code>{esc(detected_ts)}</code>",
    ])

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
    elif "liq_ratio_24h" in fact:
        # Audit B3: forced-liquidation squeeze event card.
        ratio_pct = None
        try:
            ratio_pct = float(fact.get("liq_ratio_24h", 0)) * 100.0
        except (TypeError, ValueError):
            ratio_pct = None
        lines.extend([
            "",
            "<b>Liquidation Squeeze</b>",
            f"Liq/OI 24h: <code>{esc(round(ratio_pct, 3) if ratio_pct is not None else None)}%</code>",
            f"Spike (ATR mult): <code>{esc(round(float(fact.get('spike_atr_mult', 0) or 0), 2))}</code>",
            f"OI chg 4h: <code>{esc(fact.get('oi_chg4h_pct'))}%</code>",
            f"Funding OI-w: <code>{esc(fact.get('fr_oiw'))}</code>",
            f"L/S accounts: <code>{esc(fact.get('ls_accounts'))}</code>",
        ])

    if setup:
        rr = setup.get("effective_weighted_rr", setup.get("planned_weighted_rr", setup.get("realized_rr", setup.get("target_rr", 1.05))))
        tp_mode = setup.get("tp_mode")
        trigger = setup.get("trigger") if isinstance(setup.get("trigger"), dict) else {}
        lines.extend([
            "",
            "<b>SETUP</b>",
            f"Entry: <code>{esc(setup.get('entry_reference'))}</code>",
            f"SL: <code>{esc(setup.get('invalidation_price'))}</code>",
            f"TP: <code>{esc(setup.get('target_price'))}</code>",
            f"R:R (Effective Weighted): <code>{esc(rr)}</code>",
            f"TP Mode: <code>{esc(tp_mode or 'multi_tp')}</code>",
            f"Trigger Price: <code>{esc(trigger.get('trigger_price'))}</code>",
            f"Trigger Delay: <code>{esc(trigger.get('trigger_delay_min'))} min</code>",
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
