# tracker.py

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

from event_engine.bingx import (
    get_position_directional,
    get_order,
    get_open_protection_directional,
    cancel_order,
    fetch_klines,
    to_bx_symbol,
    get_contract,
    _format_price,
    _format_qty,
    _request,
    ORDER_PATH,
)
from event_engine.telegram import send as send_tg

log = logging.getLogger("event_engine.tracker")

DATA = Path("data")
ACTIVE_TRADES_PATH = DATA / "active_trades.json"


def _load_active_trades() -> dict[str, dict]:
    if not ACTIVE_TRADES_PATH.exists():
        return {}
    try:
        data = json.loads(ACTIVE_TRADES_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.error("[TRACKER_CORRUPT_STATE] Failed to read %s: %s", ACTIVE_TRADES_PATH, exc)
        try:
            corrupt_path = ACTIVE_TRADES_PATH.with_suffix(f".corrupt.{int(time.time())}.json")
            ACTIVE_TRADES_PATH.rename(corrupt_path)
            log.warning("[TRACKER_CORRUPT_BACKUP] Saved to %s", corrupt_path)
        except Exception:
            pass
        return {}


def _save_active_trades(trades: dict[str, dict]) -> None:
    ACTIVE_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ACTIVE_TRADES_PATH.with_name(ACTIVE_TRADES_PATH.name + ".tmp")
    payload = json.dumps(trades, ensure_ascii=False, indent=2)
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, ACTIVE_TRADES_PATH)


def update_active_trade_protection(
    symbol: str,
    direction: str,
    tp_orders: list[dict],
    sl_result: dict,
) -> bool:
    """Update protection on an existing tracked position without resetting lifecycle statistics."""
    trades = _load_active_trades()
    want_symbol = str(symbol).upper().replace("-USDT", "")
    want_direction = str(direction).upper()
    for trade in trades.values():
        trade_symbol = str(trade.get("symbol", "")).upper().replace("-USDT", "")
        trade_direction = str(trade.get("direction", "")).upper()
        if trade_symbol == want_symbol and trade_direction == want_direction:
            trade["tp_orders"] = tp_orders
            trade["sl_order"] = sl_result
            _save_active_trades(trades)
            return True
    return False


def register_active_trade(
    event_id: str,
    symbol: str,
    name: str,
    direction: str,
    entry_price: float,
    qty: float,
    tp_orders: list[dict],
    sl_result: dict,
    event_type: str,
    coinalyze_row: Any = None,
    score: float = 50.0,
) -> None:
    """Регистрирует открытую позицию для сопровождения жизненного цикла."""
    trades = _load_active_trades()
    now_ms = int(time.time() * 1000)

    research = {}
    if coinalyze_row:
        research = {
            "fr_oiw": getattr(coinalyze_row, "fr_oiw", None),
            "liq_short24": getattr(coinalyze_row, "liq_short24", None),
            "liq_long24": getattr(coinalyze_row, "liq_long24", None),
            "ls_accounts": getattr(coinalyze_row, "ls_accounts", None),
            "oi": getattr(coinalyze_row, "oi", None),
        }

    trades[event_id] = {
        "event_id": event_id,
        "symbol": symbol,
        "name": name or symbol,
        "direction": direction.upper(),
        "entry_price": float(entry_price),
        "initial_qty": float(qty),
        "remaining_qty": float(qty),
        "entry_ts": now_ms,
        "tp_orders": tp_orders,
        "sl_order": sl_result,
        "hit_legs": [],
        "be_activated": False,
        "peak_pnl_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "score": score,
        "event_type": event_type,
        "research": research,
        "tp_filled_qty": {},
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,
        "last_tp_exec_price": None,
    }
    _save_active_trades(trades)


def format_tp_hit_message(
    name: str,
    symbol: str,
    leg: str,
    pnl_pct: float,
    exec_price: float,
    closed_qty: float,
    remaining_qty: float,
    remaining_pct: float,
) -> str:
    return (
        f"💰 <b>{name} ({symbol})</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Leg: <b>{leg}</b>\n"
        f"PnL TP: <b>+{pnl_pct:.2f}%</b>\n"
        f"Цена исполнения: <code>{exec_price:.8g}</code>\n"
        f"Закрыто: <code>{closed_qty:.6f}</code>\n"
        f"Осталось: <code>{remaining_qty:.6f} ({remaining_pct:.1f}%)</code>"
    )


def format_be_message(name: str, symbol: str, entry_price: float) -> str:
    return (
        f"🛡 <b>{name} ({symbol}) — БЕЗУБЫТОК</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"TP1 взят! Стоп-лосс перенесен на точку входа: <code>{entry_price:.8g}</code>\n"
        f"Текущий риск по сделке: <b>0.00%</b>"
    )


def format_trade_closed_message(
    name: str,
    symbol: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    duration_min: float,
    peak_pnl: float,
    max_drawdown: float,
    exit_reason: str,
    event_type: str,
    research: dict,
) -> str:
    is_win = pnl_pct >= 0
    emoji = "💚" if is_win else "💔"
    pnl_sign = "+" if pnl_pct > 0 else ""

    lines = [
        f"{emoji} <b>{name} ({symbol}) — сделка закрыта</b>",
        "━━━━━━━━━━━━━━━━━━",
        f"Вход <code>{entry_price:.8g}</code> → Выход <code>{exit_price:.8g}</code>   <b>{pnl_sign}{pnl_pct:.2f}%</b>",
        f"Держали <b>{duration_min:.1f} мин</b> · пик <b>+{peak_pnl:.1f}%</b> · просадка <b>-{abs(max_drawdown):.1f}%</b>",
        f"Выход по: <b>{exit_reason}</b>",
        f"Вход был: <code>{event_type}</code> · TF <b>1h</b>",
    ]

    if research:
        lines.append("━━━━━━━━━━━━━━━━━━")
        lines.append("📊 Research")
        res_parts = []
        if research.get("fr_oiw") is not None:
            res_parts.append(f"FR·OI: <code>{research['fr_oiw']}</code>")
        if research.get("ls_accounts") is not None:
            res_parts.append(f"L/S: <code>{research['ls_accounts']}</code>")
        if research.get("liq_short24") is not None:
            res_parts.append(f"LiqShort: <code>{research['liq_short24']}</code>")
        if research.get("liq_long24") is not None:
            res_parts.append(f"LiqLong: <code>{research['liq_long24']}</code>")
        lines.append(" · ".join(res_parts) if res_parts else "—")

    return "\n".join(lines)


def _move_sl_to_break_even(
    symbol: str,
    direction: str,
    entry_price: float,
    qty: float,
    old_sl_id: str | None,
    trade_id: str | None = None,
) -> dict:
    """Create and verify the BE stop before cancelling the existing SL with deterministic idempotency."""
    bx = to_bx_symbol(symbol)
    contract = get_contract(symbol) or {}
    precision = int(contract.get("quantityPrecision") or 0)
    price_precision = int(contract.get("pricePrecision") or 4)

    if qty <= 0 or entry_price <= 0 or not bx:
        return {"status": "error", "error": "invalid BE parameters", "order_id": "", "stop_price": entry_price}

    be_client_id = f"EVT_BE_{trade_id}" if trade_id else f"EVT_BE_{uuid.uuid4().hex.upper()[:16]}"

    # 1. Idempotency check: check if BE stop already exists on exchange
    verified = get_open_protection_directional(symbol, direction)
    if verified.get("status") == "ok":
        for o in verified.get("sl_orders", []):
            cid = str(o.get("clientOrderId", "")).upper()
            o_price = float(o.get("stopPrice", 0) or o.get("price", 0) or 0)
            if (be_client_id.upper() in cid) or (abs(o_price - entry_price) / max(entry_price, 1e-8) < 0.002):
                existing_be_id = str(o.get("orderId", ""))
                if old_sl_id and str(old_sl_id) != existing_be_id:
                    cancel_order(symbol, old_sl_id)
                return {
                    "status": "created",
                    "order_id": existing_be_id,
                    "client_order_id": cid or be_client_id,
                    "stop_price": entry_price,
                }

    sl_side = "SELL" if direction.upper() == "LONG" else "BUY"
    params = {
        "symbol": bx,
        "side": sl_side,
        "positionSide": direction.upper(),
        "type": "STOP_MARKET",
        "stopPrice": _format_price(entry_price, price_precision),
        "quantity": _format_qty(qty, precision),
        "clientOrderId": be_client_id,
    }

    try:
        resp = _request("POST", ORDER_PATH, params)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "order_id": "", "stop_price": entry_price}

    if not isinstance(resp, dict) or resp.get("code") != 0:
        return {
            "status": "error",
            "error": f"BE stop failed: code={resp.get('code') if isinstance(resp, dict) else None} msg={resp.get('msg') if isinstance(resp, dict) else resp}",
            "order_id": "",
            "stop_price": entry_price,
        }

    order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
    new_order_id = str(order.get("orderId", ""))
    if not new_order_id:
        return {"status": "error", "error": "BE stop response has no orderId", "order_id": "", "stop_price": entry_price}

    verified_after = get_open_protection_directional(symbol, direction)
    if verified_after.get("status") != "ok":
        return {
            "status": "error",
            "error": verified_after.get("error", "BE stop verification failed"),
            "order_id": new_order_id,
            "stop_price": entry_price,
        }

    found = any(str(o.get("orderId", "")) == new_order_id for o in verified_after.get("sl_orders", []))
    if not found:
        return {
            "status": "error",
            "error": "BE stop was created but is not visible on exchange; old SL was kept",
            "order_id": new_order_id,
            "stop_price": entry_price,
        }

    old_cancel = None
    if old_sl_id and str(old_sl_id) != new_order_id:
        old_cancel = cancel_order(symbol, old_sl_id)

    result = {
        "status": "created" if not isinstance(old_cancel, dict) or old_cancel.get("code") in (None, 0) else "created_old_sl_cancel_failed",
        "order_id": new_order_id,
        "client_order_id": order.get("clientOrderId") or be_client_id,
        "stop_price": entry_price,
    }
    if old_cancel is not None and isinstance(old_cancel, dict) and old_cancel.get("code") != 0:
        result["old_sl_cancel_error"] = old_cancel.get("msg") or old_cancel
    return result


def update_active_trades() -> None:
    """Poll exchange positions/orders, track partial TP fills, move BE safely and close trades."""
    trades = _load_active_trades()
    if not trades:
        return

    now_ms = int(time.time() * 1000)
    updated_trades = {}

    for event_id, t in trades.items():
        symbol = t["symbol"]
        direction = str(t["direction"]).upper()
        entry_price = float(t["entry_price"])
        init_qty = float(t["initial_qty"])
        rem_qty = float(t["remaining_qty"])
        hit_legs = set(t.get("hit_legs", []))
        filled_by_leg = {str(k): float(v) for k, v in (t.get("tp_filled_qty") or {}).items()}
        realized_qty = float(t.get("realized_pnl_qty", 0.0) or 0.0)
        realized_weighted = float(t.get("realized_pnl_weighted_sum", 0.0) or 0.0)

        pos = get_position_directional(symbol, direction)
        pos_status = str(pos.get("status", "")).lower()

        # An API error/timeout is not proof that the position is closed. Preserve state.
        if pos_status not in {"found", "not_found"}:
            log.warning("[TRACKER_POSITION_UNCERTAIN] %s %s: %s", symbol, direction, pos.get("error") or pos_status)
            updated_trades[event_id] = t
            continue

        pos_amt = float(pos.get("positionAmt", 0) or 0) if pos_status == "found" else 0.0

        cur_price = entry_price
        current_pnl = 0.0
        try:
            k1m = fetch_klines(symbol, "1m", limit=60)
            if k1m:
                cur_price = float(k1m[-1]["close"])
                peak_so_far = float(t.get("peak_pnl_pct", 0.0) or 0.0)
                max_dd_so_far = float(t.get("max_drawdown_pct", 0.0) or 0.0)

                # Chronological peak and drawdown calculation
                for b in k1m:
                    if direction == "LONG":
                        b_high = ((float(b["high"]) - entry_price) / entry_price) * 100.0
                        b_low = ((float(b["low"]) - entry_price) / entry_price) * 100.0
                    else:
                        b_high = ((entry_price - float(b["low"])) / entry_price) * 100.0
                        b_low = ((entry_price - float(b["high"])) / entry_price) * 100.0

                    peak_so_far = max(peak_so_far, b_high)
                    dd_here = b_low - peak_so_far
                    max_dd_so_far = min(max_dd_so_far, dd_here)

                if direction == "LONG":
                    current_pnl = ((cur_price - entry_price) / entry_price) * 100.0
                else:
                    current_pnl = ((entry_price - cur_price) / entry_price) * 100.0

                peak_so_far = max(peak_so_far, current_pnl)
                t["peak_pnl_pct"] = peak_so_far
                t["max_drawdown_pct"] = max_dd_so_far
        except Exception as exc:
            log.warning("[TRACKER_KLINE_ERROR] %s: %s", symbol, exc)

        # 1. Track TP fills, including PARTIALLY_FILLED quantities.
        for tp in t.get("tp_orders", []):
            leg = str(tp.get("leg", ""))
            order_id = tp.get("order_id")
            if not order_id:
                continue

            order_info = get_order(symbol, order_id)
            if order_info.get("status") == "error":
                continue

            order_status = str(order_info.get("order_status", "")).upper()
            if order_status not in {"PARTIALLY_FILLED", "FILLED"}:
                continue

            executed_qty = max(0.0, float(order_info.get("executed_qty", 0.0) or 0.0))
            previous_qty = max(0.0, float(filled_by_leg.get(leg, 0.0) or 0.0))
            delta_qty = max(0.0, executed_qty - previous_qty)
            if delta_qty <= 0:
                if order_status == "FILLED" and leg not in hit_legs:
                    hit_legs.add(leg)
                continue

            exec_price = float(order_info.get("avg_price") or tp.get("price") or cur_price)
            pnl_tp = float(tp.get("pnl_pct") or (
                ((exec_price - entry_price) / entry_price * 100.0)
                if direction == "LONG"
                else ((entry_price - exec_price) / entry_price * 100.0)
            ))

            rem_qty = max(0.0, rem_qty - delta_qty)
            realized_qty += delta_qty
            realized_weighted += delta_qty * pnl_tp
            filled_by_leg[leg] = executed_qty
            t["remaining_qty"] = rem_qty
            t["realized_pnl_qty"] = realized_qty
            t["realized_pnl_weighted_sum"] = realized_weighted
            t["last_tp_exec_price"] = exec_price

            if order_status == "FILLED" and leg not in hit_legs:
                hit_legs.add(leg)
                rem_pct = (rem_qty / init_qty * 100.0) if init_qty > 0 else 0.0
                send_tg(format_tp_hit_message(
                    name=t["name"],
                    symbol=symbol,
                    leg=leg,
                    pnl_pct=pnl_tp,
                    exec_price=exec_price,
                    closed_qty=delta_qty,
                    remaining_qty=rem_qty,
                    remaining_pct=rem_pct,
                ))

                # Move SL to BE only if there is still quantity to protect.
                if leg == "tp1" and not t.get("be_activated") and rem_qty > 0:
                    new_sl = _move_sl_to_break_even(
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price,
                        qty=rem_qty,
                        old_sl_id=t.get("sl_order", {}).get("order_id"),
                        trade_id=event_id.replace("EVT_", ""),
                    )
                    if new_sl.get("status") in {"created", "created_old_sl_cancel_failed"}:
                        t["sl_order"] = new_sl
                        t["be_activated"] = True
                        send_tg(format_be_message(t["name"], symbol, entry_price))
                    else:
                        log.error("[TRACKER_BE_FAILED] %s %s: %s", symbol, direction, new_sl.get("error"))

        t["hit_legs"] = list(hit_legs)
        t["tp_filled_qty"] = filled_by_leg

        # 2. Close only when the exchange explicitly says the position is gone,
        # or when all tracked quantity has definitely been filled by TP orders.
        closed_by_tp = rem_qty <= 0 and realized_qty > 0
        position_gone = pos_status == "not_found"
        if not position_gone and not closed_by_tp:
            updated_trades[event_id] = t
            continue

        duration_min = (now_ms - t["entry_ts"]) / 60000.0
        exit_price = float(t.get("last_tp_exec_price") or cur_price)
        sl_order_id = t.get("sl_order", {}).get("order_id")
        sl_info = get_order(symbol, sl_order_id) if sl_order_id else {}
        if sl_info.get("status") == "ok" and sl_info.get("order_status") == "FILLED":
            exit_price = float(sl_info.get("avg_price") or exit_price)
            exit_reason = (
                "BREAK_EVEN"
                if t.get("be_activated") and entry_price > 0 and abs(exit_price - entry_price) / entry_price < 0.003
                else "STOP_LOSS"
            )
        elif closed_by_tp and hit_legs:
            exit_reason = "TAKE_PROFIT_FULL"
        else:
            exit_reason = "POSITION_CLOSED"

        # If a stop/other close removed the remaining quantity after partial TP,
        # include that remaining quantity in the weighted PnL.
        if position_gone and rem_qty > 0 and init_qty > 0:
            exit_pnl = (
                ((exit_price - entry_price) / entry_price * 100.0)
                if direction == "LONG"
                else ((entry_price - exit_price) / entry_price * 100.0)
            )
            realized_weighted += rem_qty * exit_pnl
            realized_qty += rem_qty
            rem_qty = 0.0

        final_pnl = (realized_weighted / init_qty) if init_qty > 0 and realized_qty > 0 else current_pnl

        send_tg(format_trade_closed_message(
            name=t["name"],
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl_pct=final_pnl,
            duration_min=duration_min,
            peak_pnl=t.get("peak_pnl_pct", 0.0),
            max_drawdown=t.get("max_drawdown_pct", 0.0),
            exit_reason=exit_reason,
            event_type=t.get("event_type", "DIVERGENCE"),
            research=t.get("research", {}),
        ))

        # Cleanup only orders that are still tracked as open.
        for tp in t.get("tp_orders", []):
            if tp.get("leg") not in hit_legs and tp.get("order_id"):
                cancel_order(symbol, tp["order_id"])
        if sl_order_id:
            cancel_order(symbol, sl_order_id)

    _save_active_trades(updated_trades)
