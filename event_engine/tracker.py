# tracker.py

from __future__ import annotations

import json
import logging
import os
import time
import uuid
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if result != result:
            return default
        if result in (float("inf"), float("-inf")):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _load_active_trades() -> dict[str, dict]:
    if not ACTIVE_TRADES_PATH.exists():
        return {}
    try:
        data = json.loads(ACTIVE_TRADES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
        log.error("[TRACKER_INVALID_STATE] %s is not a JSON object", ACTIVE_TRADES_PATH)
        return {}
    except Exception as exc:
        log.error("[TRACKER_CORRUPT_STATE] Failed to read %s: %s", ACTIVE_TRADES_PATH, exc)
        return {}


def _save_active_trades(trades: dict[str, dict]) -> None:
    ACTIVE_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ACTIVE_TRADES_PATH.with_name(ACTIVE_TRADES_PATH.name + ".tmp")
    payload = json.dumps(trades, ensure_ascii=False, indent=2)
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, ACTIVE_TRADES_PATH)


def _normalize_direction(direction: str) -> str:
    d = str(direction or "").upper()
    if d not in {"LONG", "SHORT"}:
        raise ValueError(f"Invalid direction={direction}")
    return d


def _normalized_symbol(symbol: str) -> str:
    return str(symbol or "").upper().replace("-USDT", "")


def update_active_trade_protection(
    symbol: str,
    direction: str,
    tp_orders: list[dict],
    sl_result: dict,
) -> bool:
    trades = _load_active_trades()
    want_bx = to_bx_symbol(symbol) or _normalized_symbol(symbol)
    want_direction = str(direction).upper()

    for trade in trades.values():
        if trade.get("closed", False):
            continue
        trade_bx = to_bx_symbol(trade.get("symbol", "")) or _normalized_symbol(trade.get("symbol", ""))
        trade_direction = str(trade.get("direction", "")).upper()

        if trade_bx == want_bx and trade_direction == want_direction:
            trade["tp_orders"] = tp_orders if isinstance(tp_orders, list) else []
            trade["sl_order"] = sl_result if isinstance(sl_result, dict) else {}
            trade["protection_last_updated_ts"] = int(time.time() * 1000)
            _save_active_trades(trades)
            return True

    return False


def _extract_setup_metrics(setup: dict | None) -> dict[str, Any]:
    if not isinstance(setup, dict):
        return {
            "planned_risk_pct": None,
            "planned_target_rr": None,
            "planned_weighted_rr": 1.55,
            "entry_reference": None,
            "invalidation_price": None,
            "target_price": None,
            "tp_levels": [],
        }

    return {
        "planned_risk_pct": _safe_float(setup.get("risk_pct"), 0.0) if setup.get("risk_pct") is not None else None,
        "planned_target_rr": _safe_float(setup.get("target_rr"), 0.0) if setup.get("target_rr") is not None else None,
        "planned_weighted_rr": _safe_float(setup.get("planned_weighted_rr", 1.55), 1.55),
        "entry_reference": _safe_float(setup.get("entry_reference"), 0.0) if setup.get("entry_reference") is not None else None,
        "invalidation_price": _safe_float(setup.get("invalidation_price"), 0.0) if setup.get("invalidation_price") is not None else None,
        "target_price": _safe_float(setup.get("target_price"), 0.0) if setup.get("target_price") is not None else None,
        "tp_levels": setup.get("tp_levels") if isinstance(setup.get("tp_levels"), list) else [],
    }


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
    setup: dict | None = None,
    requested_entry_price: float | None = None,
) -> None:
    direction = _normalize_direction(direction)
    trades = _load_active_trades()
    now_ms = int(time.time() * 1000)

    actual_entry_price = _safe_float(entry_price)
    actual_qty = abs(_safe_float(qty))

    if actual_entry_price <= 0 or actual_qty <= 0:
        raise ValueError(f"Cannot register invalid position: entry_price={actual_entry_price} qty={actual_qty}")

    setup_metrics = _extract_setup_metrics(setup)
    research: dict[str, Any] = {}

    if coinalyze_row is not None:
        for attr in (
            "fr_oiw", "pfr_oiw", "liq_short24", "liq_long24",
            "ls_accounts", "oi", "oi_chg24_pct", "oi_chg4h_pct",
            "oi_vol_ratio", "oi_mktcap_ratio", "volume24",
            "btc_corr7d", "cvd24", "lls24",
        ):
            research[attr] = getattr(coinalyze_row, attr, None)

    requested_price = _safe_float(requested_entry_price, 0.0) if requested_entry_price is not None else setup_metrics["entry_reference"]
    entry_slippage_pct = None
    adverse_entry_slippage_pct = None

    if requested_price is not None and requested_price > 0:
        entry_slippage_pct = (actual_entry_price - requested_price) / requested_price * 100.0
        if direction == "LONG":
            adverse_entry_slippage_pct = max(0.0, entry_slippage_pct)
        else:
            adverse_entry_slippage_pct = max(0.0, -entry_slippage_pct)

    trades[event_id] = {
        "event_id": event_id,
        "symbol": symbol,
        "name": name or symbol,
        "direction": direction,
        "entry_price": actual_entry_price,
        "actual_entry_price": actual_entry_price,
        "requested_entry_price": requested_price,
        "entry_slippage_pct": entry_slippage_pct,
        "adverse_entry_slippage_pct": adverse_entry_slippage_pct,
        "initial_qty": actual_qty,
        "remaining_qty": actual_qty,
        "entry_ts": now_ms,
        "tp_orders": tp_orders if isinstance(tp_orders, list) else [],
        "sl_order": sl_result if isinstance(sl_result, dict) else {},
        "hit_legs": [],
        "be_activated": False,
        "be_activation_ts": None,
        "peak_pnl_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "current_pnl_pct": 0.0,
        "score": _safe_float(score, 50.0),
        "event_type": event_type,
        "research": research,
        "setup": setup.copy() if isinstance(setup, dict) else {},
        "planned_risk_pct": setup_metrics["planned_risk_pct"],
        "planned_target_rr": setup_metrics["planned_target_rr"],
        "planned_weighted_rr": setup_metrics["planned_weighted_rr"],
        "planned_entry_reference": setup_metrics["entry_reference"],
        "planned_invalidation_price": setup_metrics["invalidation_price"],
        "planned_target_price": setup_metrics["target_price"],
        "tp_levels": setup_metrics["tp_levels"],
        "tp_filled_qty": {},
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,
        "last_tp_exec_price": None,
        "last_close_exec_price": None,
        "realized_pnl_pct": None,
        "realized_rr": None,
        "exit_price": None,
        "exit_reason": None,
        "closed_ts": None,
        "duration_min": None,
        "closed": False,
        "last_observation_ts": now_ms,
    }

    _save_active_trades(trades)


def format_tp_hit_message(
    name: str, symbol: str, leg: str, pnl_pct: float,
    exec_price: float, closed_qty: float, remaining_qty: float, remaining_pct: float,
) -> str:
    return (
        f"💰 <b>{name} ({symbol})</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Leg: <b>{leg}</b>\n"
        f"PnL TP: <b>+{pnl_pct:.2f}%</b>\n"
        f"Цена исполнения: <code>{exec_price:.8g}</code>\n"
        f"Закрыто: <code>{closed_qty:.8f}</code>\n"
        f"Осталось: <code>{remaining_qty:.8f} ({remaining_pct:.1f}%)</code>"
    )


def format_be_message(name: str, symbol: str, entry_price: float) -> str:
    return (
        f"🛡 <b>{name} ({symbol}) — БЕЗУБЫТОК</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"TP1 взят! Стоп-лосс перенесен на точку входа: <code>{entry_price:.8g}</code>\n"
        f"Текущий риск по сделке: <b>0.00%</b>"
    )


def format_trade_closed_message(
    name: str, symbol: str, direction: str, entry_price: float, exit_price: float,
    pnl_pct: float, realized_rr: float | None, planned_rr: float | None,
    duration_min: float, peak_pnl: float, max_drawdown: float,
    exit_reason: str, event_type: str, research: dict,
) -> str:
    is_win = pnl_pct >= 0.0
    emoji = "💚" if is_win else "💔"
    pnl_sign = "+" if pnl_pct > 0 else ""
    realized_rr_text = f"{realized_rr:.3f}" if realized_rr is not None else "—"
    planned_rr_text = f"{planned_rr:.3f}" if planned_rr is not None else "—"

    lines = [
        f"{emoji} <b>{name} ({symbol}) — сделка закрыта</b>",
        "━━━━━━━━━━━━━━━━━━",
        f"Вход <code>{entry_price:.8g}</code> → Выход <code>{exit_price:.8g}</code>   <b>{pnl_sign}{pnl_pct:.2f}%</b>",
        f"Realized R:R: <b>{realized_rr_text}</b> · Planned Weighted R:R: <b>{planned_rr_text}</b>",
        f"Держали <b>{duration_min:.1f} мин</b> · пик <b>+{peak_pnl:.2f}%</b> · просадка <b>{max_drawdown:.2f}%</b>",
        f"Выход по: <b>{exit_reason}</b>",
        f"Вход был: <code>{event_type}</code> · TF <b>1h</b>",
    ]

    if research:
        lines.append("━━━━━━━━━━━━━━━━━━")
        lines.append("📊 Research")
        res_parts = []
        mappings = [
            ("FR·OI", "fr_oiw"), ("PFR·OI", "pfr_oiw"), ("L/S", "ls_accounts"),
            ("LiqShort", "liq_short24"), ("LiqLong", "liq_long24"),
            ("OI Chg 4H", "oi_chg4h_pct"), ("CVD24", "cvd24"), ("LLS24", "lls24"),
        ]
        for label, key in mappings:
            value = research.get(key)
            if value is not None:
                res_parts.append(f"{label}: <code>{value}</code>")
        lines.append(" · ".join(res_parts) if res_parts else "—")

    return "\n".join(lines)


def _move_sl_to_break_even(
    symbol: str, direction: str, entry_price: float, qty: float, old_sl_id: str | None, trade_id: str | None = None,
) -> dict:
    direction = str(direction).upper()
    bx = to_bx_symbol(symbol)
    contract = get_contract(symbol) or {}

    try:
        precision = int(contract.get("quantityPrecision") or 0)
        price_precision = int(contract.get("pricePrecision") or 4)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": str(exc), "order_id": "", "stop_price": entry_price}

    if qty <= 0 or entry_price <= 0 or not bx:
        return {"status": "error", "error": "invalid BE parameters", "order_id": "", "stop_price": entry_price}

    trade_token = str(trade_id) if trade_id else uuid.uuid4().hex.upper()[:16]
    be_client_id = f"EVT_BE_{trade_token}"

    verified = get_open_protection_directional(symbol, direction)
    if verified.get("status") == "ok":
        for order in verified.get("sl_orders", []):
            cid = str(order.get("clientOrderId", "")).upper()
            order_price = _safe_float(order.get("stopPrice") or order.get("price"), 0.0)
            price_matches = order_price > 0 and abs(order_price - entry_price) / max(entry_price, 1e-12) < 0.002

            if be_client_id.upper() in cid or price_matches:
                existing_id = str(order.get("orderId", ""))
                if old_sl_id and existing_id and str(old_sl_id) != existing_id:
                    try:
                        cancel_order(symbol, old_sl_id)
                    except Exception as exc:
                        log.warning("[TRACKER_BE_OLD_SL_CANCEL_ERROR] %s: %s", symbol, exc)
                return {
                    "status": "created",
                    "order_id": existing_id,
                    "client_order_id": cid or be_client_id,
                    "stop_price": entry_price,
                }

    sl_side = "SELL" if direction == "LONG" else "BUY"
    params = {
        "symbol": bx,
        "side": sl_side,
        "positionSide": direction,
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
            "error": f"BE stop failed: {resp}",
            "order_id": "",
            "stop_price": entry_price,
        }

    order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
    new_order_id = str(order.get("orderId", ""))
    if not new_order_id:
        return {"status": "error", "error": "BE stop response has no orderId", "order_id": "", "stop_price": entry_price}

    verified_after = get_open_protection_directional(symbol, direction)
    if verified_after.get("status") != "ok":
        return {"status": "error", "error": "BE stop verification failed", "order_id": new_order_id, "stop_price": entry_price}

    found = any(str(o.get("orderId", "")) == new_order_id for o in verified_after.get("sl_orders", []))
    if not found:
        return {"status": "error", "error": "BE stop not visible on exchange", "order_id": new_order_id, "stop_price": entry_price}

    if old_sl_id and str(old_sl_id) != new_order_id:
        try:
            cancel_order(symbol, old_sl_id)
        except Exception:
            pass

    return {
        "status": "created",
        "order_id": new_order_id,
        "client_order_id": order.get("clientOrderId") or be_client_id,
        "stop_price": entry_price,
    }


def _calc_trade_pnl_pct(entry_price: float, exit_price: float, direction: str) -> float:
    if entry_price <= 0 or exit_price <= 0:
        return 0.0
    if str(direction).upper() == "LONG":
        return (exit_price - entry_price) / entry_price * 100.0
    return (entry_price - exit_price) / entry_price * 100.0


def _derive_planned_risk_pct(trade: dict) -> float | None:
    direct = trade.get("planned_risk_pct")
    if direct is not None:
        val = _safe_float(direct, 0.0)
        if val > 0:
            return val

    setup = trade.get("setup")
    if isinstance(setup, dict):
        val = _safe_float(setup.get("risk_pct"), 0.0)
        if val > 0:
            return val

    return None


def _calc_realized_rr(pnl_pct: float, risk_pct: float | None) -> float | None:
    if risk_pct is None or risk_pct <= 0:
        return None
    return pnl_pct / risk_pct


def _update_mfe_mae(trade: dict, candles: list[dict]) -> None:
    if not candles:
        return
    entry_price = _safe_float(trade.get("entry_price"))
    direction = str(trade.get("direction", "LONG")).upper()
    if entry_price <= 0:
        return

    peak = _safe_float(trade.get("peak_pnl_pct", 0.0))
    max_drawdown = _safe_float(trade.get("max_drawdown_pct", 0.0))

    for candle in candles:
        high = _safe_float(candle.get("high"), 0.0)
        low = _safe_float(candle.get("low"), 0.0)
        if high <= 0 or low <= 0:
            continue

        if direction == "LONG":
            favorable = (high - entry_price) / entry_price * 100.0
            adverse = (low - entry_price) / entry_price * 100.0
        else:
            favorable = (entry_price - low) / entry_price * 100.0
            adverse = (entry_price - high) / entry_price * 100.0

        peak = max(peak, favorable)
        current_drawdown = adverse - peak
        max_drawdown = min(max_drawdown, current_drawdown)

    trade["peak_pnl_pct"] = peak
    trade["max_drawdown_pct"] = max_drawdown


def _get_exit_from_sl(symbol: str, sl_order_id: str | None) -> tuple[float | None, str | None]:
    if not sl_order_id:
        return None, None
    try:
        sl_info = get_order(symbol, sl_order_id)
    except Exception as exc:
        log.warning("[TRACKER_SL_ORDER_ERROR] %s: %s", symbol, exc)
        return None, None

    if sl_info.get("status") != "ok":
        return None, None

    status = str(sl_info.get("order_status", "")).upper()
    if status != "FILLED":
        return None, None

    exit_price = _safe_float(sl_info.get("avg_price"), 0.0)
    return (exit_price if exit_price > 0 else None, "SL_FILLED")


def update_active_trades() -> None:
    trades = _load_active_trades()
    if not trades:
        return

    now_ms = int(time.time() * 1000)
    updated_trades: dict[str, dict] = {}

    for event_id, trade in trades.items():
        if trade.get("closed", False):
            continue

        try:
            symbol = str(trade.get("symbol", ""))
            direction = _normalize_direction(trade.get("direction", ""))
            entry_price = _safe_float(trade.get("entry_price"))
            init_qty = abs(_safe_float(trade.get("initial_qty")))
            rem_qty = max(0.0, _safe_float(trade.get("remaining_qty")))
            entry_ts = int(_safe_float(trade.get("entry_ts")))

            if not symbol or entry_price <= 0 or init_qty <= 0:
                updated_trades[event_id] = trade
                continue

            hit_legs = set(trade.get("hit_legs", []))
            filled_by_leg = {str(k): max(0.0, _safe_float(v)) for k, v in (trade.get("tp_filled_qty", {}) or {}).items()}
            realized_qty = max(0.0, _safe_float(trade.get("realized_pnl_qty", 0.0)))
            realized_weighted = _safe_float(trade.get("realized_pnl_weighted_sum", 0.0))

            pos = get_position_directional(symbol, direction)
            pos_status = str(pos.get("status", "")).lower()

            if pos_status not in {"found", "not_found"}:
                trade["last_observation_ts"] = now_ms
                updated_trades[event_id] = trade
                continue

            pos_amt = abs(_safe_float(pos.get("positionAmt"))) if pos_status == "found" else 0.0
            if pos_status == "found":
                exchange_avg = _safe_float(pos.get("avgPrice"))
                if exchange_avg > 0:
                    trade["last_exchange_avg_price"] = exchange_avg

            cur_price = entry_price
            try:
                k1m = fetch_klines(symbol, "1m", limit=60)
                if k1m:
                    cur_price = _safe_float(k1m[-1].get("close"), entry_price)
                    _update_mfe_mae(trade, k1m)
            except Exception as exc:
                log.warning("[TRACKER_KLINE_ERROR] %s: %s", symbol, exc)

            current_pnl = _calc_trade_pnl_pct(entry_price, cur_price, direction)
            trade["current_pnl_pct"] = current_pnl
            trade["current_position_qty"] = pos_amt
            trade["last_observation_ts"] = now_ms

            # Проверка исполнения Тейк-Профитов
            for tp in trade.get("tp_orders", []):
                leg = str(tp.get("leg", ""))
                order_id = tp.get("order_id")
                if not leg or not order_id:
                    continue

                try:
                    order_info = get_order(symbol, order_id)
                except Exception:
                    continue

                if order_info.get("status") == "error":
                    continue

                order_status = str(order_info.get("order_status", "")).upper()
                if order_status not in {"PARTIALLY_FILLED", "FILLED"}:
                    continue

                executed_qty = max(0.0, _safe_float(order_info.get("executed_qty", 0.0)))
                previous_qty = max(0.0, _safe_float(filled_by_leg.get(leg, 0.0)))
                delta_qty = max(0.0, executed_qty - previous_qty)

                if delta_qty <= 0:
                    if order_status == "FILLED":
                        hit_legs.add(leg)
                    continue

                exec_price = _safe_float(order_info.get("avg_price") or tp.get("price") or cur_price, cur_price)
                if exec_price <= 0:
                    continue

                pnl_tp = _safe_float(tp.get("pnl_pct"), 0.0)
                if pnl_tp <= 0:
                    pnl_tp = _calc_trade_pnl_pct(entry_price, exec_price, direction)

                rem_qty = max(0.0, rem_qty - delta_qty)
                realized_qty += delta_qty
                realized_weighted += delta_qty * pnl_tp
                filled_by_leg[leg] = executed_qty

                trade["remaining_qty"] = rem_qty
                trade["realized_pnl_qty"] = realized_qty
                trade["realized_pnl_weighted_sum"] = realized_weighted
                trade["last_tp_exec_price"] = exec_price

                if order_status == "FILLED" and leg not in hit_legs:
                    hit_legs.add(leg)
                    rem_pct = rem_qty / init_qty * 100.0 if init_qty > 0 else 0.0

                    try:
                        send_tg(
                            format_tp_hit_message(
                                name=trade.get("name", symbol),
                                symbol=symbol,
                                leg=leg,
                                pnl_pct=pnl_tp,
                                exec_price=exec_price,
                                closed_qty=delta_qty,
                                remaining_qty=rem_qty,
                                remaining_pct=rem_pct,
                            )
                        )
                    except Exception:
                        pass

                    # ПЕРЕНОС В БЕЗУБЫТОК ПОСЛЕ TP1
                    if leg == "tp1" and not trade.get("be_activated") and rem_qty > 0:
                        old_sl_id = trade.get("sl_order", {}).get("order_id") if isinstance(trade.get("sl_order"), dict) else None
                        new_sl = _move_sl_to_break_even(
                            symbol=symbol,
                            direction=direction,
                            entry_price=entry_price,
                            qty=rem_qty,
                            old_sl_id=old_sl_id,
                            trade_id=str(event_id).replace("EVT_", ""),
                        )
                        if new_sl.get("status") in {"created", "created_old_sl_cancel_failed"}:
                            trade["sl_order"] = new_sl
                            trade["be_activated"] = True
                            trade["be_activation_ts"] = now_ms
                            try:
                                send_tg(format_be_message(trade.get("name", symbol), symbol, entry_price))
                            except Exception:
                                pass

            trade["hit_legs"] = sorted(hit_legs)
            trade["tp_filled_qty"] = filled_by_leg

            closed_by_tp = rem_qty <= 1e-12 and realized_qty > 0
            position_gone = pos_status == "not_found"

            if not position_gone and not closed_by_tp:
                updated_trades[event_id] = trade
                continue

            # Фиксация выхода и закрытие
            duration_min = (now_ms - entry_ts) / 60000.0
            exit_price = _safe_float(trade.get("last_tp_exec_price"), cur_price)
            sl_order_id = trade.get("sl_order", {}).get("order_id") if isinstance(trade.get("sl_order"), dict) else None

            sl_exit_price, _ = _get_exit_from_sl(symbol, sl_order_id)
            if sl_exit_price is not None:
                exit_price = sl_exit_price
                if trade.get("be_activated") and abs(exit_price - entry_price) / max(entry_price, 1e-12) < 0.003:
                    exit_reason = "BREAK_EVEN"
                else:
                    exit_reason = "STOP_LOSS"
            elif closed_by_tp:
                exit_reason = "TAKE_PROFIT_FULL"
            else:
                exit_reason = "POSITION_CLOSED"

            if exit_price <= 0:
                exit_price = cur_price

            if position_gone and rem_qty > 0 and init_qty > 0:
                residual_pnl = _calc_trade_pnl_pct(entry_price, exit_price, direction)
                realized_weighted += rem_qty * residual_pnl
                realized_qty += rem_qty
                trade["remaining_qty"] = 0.0

            final_pnl = (realized_weighted / init_qty) if (init_qty > 0 and realized_qty > 0) else current_pnl
            planned_risk_pct = _derive_planned_risk_pct(trade)
            realized_rr = _calc_realized_rr(final_pnl, planned_risk_pct)
            planned_rr = _safe_float(trade.get("planned_weighted_rr", 1.55), 1.55)

            trade["remaining_qty"] = 0.0
            trade["realized_pnl_pct"] = final_pnl
            trade["realized_pnl_qty"] = realized_qty
            trade["realized_pnl_weighted_sum"] = realized_weighted
            trade["realized_rr"] = realized_rr
            trade["exit_price"] = exit_price
            trade["exit_reason"] = exit_reason
            trade["closed_ts"] = now_ms
            trade["duration_min"] = duration_min
            trade["closed"] = True

            try:
                send_tg(
                    format_trade_closed_message(
                        name=trade.get("name", symbol),
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        pnl_pct=final_pnl,
                        realized_rr=realized_rr,
                        planned_rr=planned_rr,
                        duration_min=duration_min,
                        peak_pnl=_safe_float(trade.get("peak_pnl_pct")),
                        max_drawdown=_safe_float(trade.get("max_drawdown_pct")),
                        exit_reason=exit_reason,
                        event_type=trade.get("event_type", "DIVERGENCE"),
                        research=trade.get("research", {}),
                    )
                )
            except Exception:
                pass

            # Отмена оставшихся защитных ордеров
            for tp in trade.get("tp_orders", []):
                if tp.get("leg") not in hit_legs and tp.get("order_id"):
                    try:
                        cancel_order(symbol, tp["order_id"])
                    except Exception:
                        pass

            if sl_order_id:
                try:
                    cancel_order(symbol, sl_order_id)
                except Exception:
                    pass

        except Exception as exc:
            log.exception("[TRACKER_FATAL_TRADE_ERROR] %s: %s", event_id, exc)
            updated_trades[event_id] = trade

    _save_active_trades(updated_trades)
