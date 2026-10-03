from __future__ import annotations

import html
import os
from typing import Any, Optional

import requests

from .formatting import format_number, format_percent, format_price, format_rr, tp_price_from_pnl


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
        f"Price: <code>{esc(format_price(price))}</code>",
        f"Detected: <code>{esc(detected_ts)}</code>",
    ])

    if "p1_price" in fact:
        lines.extend([
            "",
            "<b>Divergence</b>",
            f"P1: <code>{esc(format_price(fact.get('p1_price')))}</code>",
            f"P2: <code>{esc(format_price(fact.get('p2_price')))}</code>",
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
        trigger = setup.get("trigger") if isinstance(setup.get("trigger"), dict) else {}
        entry_reference = setup.get("entry_reference")
        invalidation_price = setup.get("invalidation_price")
        risk_pct = setup.get("risk_pct")

        tp_levels = setup.get("effective_tp_levels")
        if not isinstance(tp_levels, list) or not tp_levels:
            tp_levels = setup.get("tp_levels") if isinstance(setup.get("tp_levels"), list) else []

        lines.extend([
            "",
            "<b>SETUP</b>",
            f"Entry: <code>{esc(format_price(entry_reference))}</code>",
            f"SL: <code>{esc(format_price(invalidation_price))}</code> <code>({esc(format_percent(-abs(float(risk_pct)), decimals=2, signed=True)) if risk_pct is not None else '—'})</code>",
        ])

        for index, level in enumerate(tp_levels[:3], start=1):
            if not isinstance(level, dict):
                continue
            pnl_pct = level.get("pnl_pct")
            tp_price = tp_price_from_pnl(entry_reference, direction, pnl_pct)
            fraction = level.get("close_fraction")
            fraction_text = ""
            try:
                fraction_text = f" · {float(fraction) * 100:.0f}%" if fraction is not None else ""
            except (TypeError, ValueError):
                fraction_text = ""
            rr_text = ""
            try:
                if risk_pct is not None and float(risk_pct) > 0 and pnl_pct is not None:
                    rr_text = f" · {esc(format_rr(float(pnl_pct) / float(risk_pct))) }"
            except (TypeError, ValueError):
                rr_text = ""
            lines.append(
                f"TP{index}: <code>{esc(format_price(tp_price))}</code> <code>({esc(format_percent(pnl_pct, decimals=2, signed=True))}{rr_text}{fraction_text})</code>"
            )

        if not tp_levels:
            target_price = setup.get("target_price")
            lines.append(f"TP3: <code>{esc(format_price(target_price))}</code>")

        rr_value = setup.get("effective_weighted_rr")
        if rr_value is None:
            rr_value = setup.get("planned_weighted_rr")
        if rr_value is None:
            rr_value = setup.get("realized_rr")
        tp_mode = setup.get("tp_mode") or "multi_tp"
        lines.extend([
            f"R:R: <code>{esc(str(rr_value))}</code>",
            f"TP Mode: <code>{esc(tp_mode)}</code>",
        ])

        sr_context = setup.get("sr_context") if isinstance(setup.get("sr_context"), dict) else {}
        alignment = str(sr_context.get("directional_zone_alignment") or "").upper()
        if alignment == "LONG_IN_DEMAND":
            confirmation = f"🟢 <b>2/2</b> — {direction} {event_type} + <b>DEMAND</b>"
        elif alignment == "SHORT_IN_SUPPLY":
            confirmation = f"🔴 <b>2/2</b> — {direction} {event_type} + <b>SUPPLY</b>"
        elif str(sr_context.get("supporting_zone_context") or "").upper().startswith("SUPPORTIVE_"):
            zone_kind = "DEMAND" if direction == "LONG" else "SUPPLY"
            confirmation = f"🟡 <b>1/2</b> — {direction} {event_type} + {zone_kind} nearby"
        else:
            confirmation = f"⚪ <b>1/2</b> — {direction} {event_type} only"

        lines.extend([
            f"<b>CONFIRMATION:</b> {confirmation}",
            f"Trigger Price: <code>{esc(format_price(trigger.get('trigger_price')))}</code>",
            f"Trigger Delay: <code>{esc(format_number(trigger.get('trigger_delay_min'), decimals=2))} min</code>",
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
