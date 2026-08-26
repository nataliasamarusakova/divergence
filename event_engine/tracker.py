from __future__ import annotations

import json
import logging
import time
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any

from event_engine.bingx import (
    get_position_directional,
    get_order,
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
    except Exception:
        return {}


def _save_active_trades(trades: dict[str, dict]) -> None:
    ACTIVE_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_TRADES_PATH.write_text(json.dumps(trades, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _move_sl_to_break_even(symbol: str, direction: str, entry_price: float, qty: float, old_sl_id: str | None) -> dict:
    """Переносит Stop Loss на точку входа (Break-Even)."""
    bx = to_bx_symbol(symbol)
    contract = get_contract(symbol) or {}
    precision = int(contract.get("quantityPrecision") or 0)
    price_precision = int(contract.get("pricePrecision") or 4)

    # 1. Отменяем старый стоп-лосс
    if old_sl_id:
        cancel_order(symbol, old_sl_id)

    # 2. Ставим новый STOP_MARKET на цену входа
    sl_side = "SELL" if direction.upper() == "LONG" else "BUY"
    client_order_id = f"EVT_BE_{uuid.uuid4().hex.upper()[:16]}"
    params = {
        "symbol": bx,
        "side": sl_side,
        "positionSide": direction.upper(),
        "type": "STOP_MARKET",
        "stopPrice": _format_price(entry_price, price_precision),
        "quantity": _format_qty(qty, precision),
        "clientOrderId": client_order_id,
        "reduceOnly": "true",
    }
    resp = _request("POST", ORDER_PATH, params)
    order = (resp.get("data") or {}).get("order") or {}
    return {
        "status": "created" if resp.get("code") == 0 else "error",
        "order_id": str(order.get("orderId", "")),
        "stop_price": entry_price,
    }


def update_active_trades() -> None:
    """Опрашивает открытые позиции, фиксирует TP, двигает BE и логирует закрытие."""
    trades = _load_active_trades()
    if not trades:
        return

    now_ms = int(time.time() * 1000)
    updated_trades = {}

    for event_id, t in trades.items():
        symbol = t["symbol"]
        direction = t["direction"]
        entry_price = t["entry_price"]
        init_qty = t["initial_qty"]
        rem_qty = t["remaining_qty"]
        hit_legs = set(t.get("hit_legs", []))

        pos = get_position_directional(symbol, direction)
        pos_amt = float(pos.get("positionAmt", 0) or 0) if pos.get("status") == "found" else 0.0

        cur_price = entry_price
        try:
            k1m = fetch_klines(symbol, "1m", limit=60)
            if k1m:
                cur_high = max(b["high"] for b in k1m)
                cur_low = min(b["low"] for b in k1m)
                cur_price = float(k1m[-1]["close"])

                if direction == "LONG":
                    max_p = ((cur_high - entry_price) / entry_price) * 100.0
                    min_p = ((cur_low - entry_price) / entry_price) * 100.0
                else:
                    max_p = ((entry_price - cur_low) / entry_price) * 100.0
                    min_p = ((entry_price - cur_high) / entry_price) * 100.0

                t["peak_pnl_pct"] = max(t.get("peak_pnl_pct", 0.0), max_p)
                t["max_drawdown_pct"] = min(t.get("max_drawdown_pct", 0.0), min_p)
        except Exception:
            pass

        # 1. Проверка Take Profit ордеров
        for tp in t.get("tp_orders", []):
            leg = tp.get("leg")
            order_id = tp.get("order_id")
            if not order_id or leg in hit_legs:
                continue

            order_info = get_order(symbol, order_id)
            if order_info.get("order_status") == "FILLED":
                hit_legs.add(leg)
                exec_price = float(order_info.get("avg_price") or tp.get("price") or cur_price)
                closed_qty = float(order_info.get("executed_qty") or tp.get("qty") or (init_qty * 0.3))

                rem_qty = max(0.0, rem_qty - closed_qty)
                t["remaining_qty"] = rem_qty
                rem_pct = (rem_qty / init_qty * 100.0) if init_qty > 0 else 0.0
                pnl_tp = tp.get("pnl_pct", abs((exec_price - entry_price) / entry_price * 100.0))

                send_tg(format_tp_hit_message(
                    name=t["name"],
                    symbol=symbol,
                    leg=leg,
                    pnl_pct=pnl_tp,
                    exec_price=exec_price,
                    closed_qty=closed_qty,
                    remaining_qty=rem_qty,
                    remaining_pct=rem_pct,
                ))

                # Автоматический перенос в безубыток при взятии TP1
                if leg == "tp1" and not t.get("be_activated"):
                    new_sl = _move_sl_to_break_even(
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price,
                        qty=rem_qty,
                        old_sl_id=t.get("sl_order", {}).get("order_id"),
                    )
                    t["sl_order"] = new_sl
                    t["be_activated"] = True
                    send_tg(format_be_message(t["name"], symbol, entry_price))

        t["hit_legs"] = list(hit_legs)

        # 2. Проверка закрытия позиции
        if pos_amt <= 0 or rem_qty <= 0:
            duration_min = (now_ms - t["entry_ts"]) / 60000.0
            exit_price = cur_price

            sl_order_id = t.get("sl_order", {}).get("order_id")
            sl_info = get_order(symbol, sl_order_id) if sl_order_id else {}

            if sl_info.get("order_status") == "FILLED":
                exit_reason = "BREAK_EVEN" if t.get("be_activated") and abs(float(sl_info.get("avg_price", 0)) - entry_price) / entry_price < 0.003 else "STOP_LOSS"
                exit_price = float(sl_info.get("avg_price") or exit_price)
            elif len(hit_legs) >= len(t.get("tp_orders", [])) and len(hit_legs) > 0:
                exit_reason = "TAKE_PROFIT_FULL"
            else:
                exit_reason = "POSITION_CLOSED"

            final_pnl = ((exit_price - entry_price) / entry_price * 100.0) if direction == "LONG" else ((entry_price - exit_price) / entry_price * 100.0)

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

            # Очистка висячих ордеров
            for tp in t.get("tp_orders", []):
                if tp.get("leg") not in hit_legs and tp.get("order_id"):
                    cancel_order(symbol, tp["order_id"])
            if sl_order_id:
                cancel_order(symbol, sl_order_id)

            continue

        updated_trades[event_id] = t

    _save_active_trades(updated_trades)
