from __future__ import annotations

import json
import hashlib
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
    _post_protection_order_verified,
)
from event_engine.telegram import send_detailed

log = logging.getLogger("event_engine.tracker")

DATA = Path("data")
ACTIVE_TRADES_PATH = DATA / "active_trades.json"
TRADES_PATH = DATA / "trades.jsonl"
NOTIFICATIONS_PATH = DATA / "notifications.json"




def _load_notifications() -> dict[str, dict]:
    if not NOTIFICATIONS_PATH.exists():
        return {}
    try:
        data = json.loads(NOTIFICATIONS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.error("[TELEGRAM] Notification state read failed: %s", exc)
        return {}


def _save_notifications(data: dict[str, dict]) -> None:
    NOTIFICATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = NOTIFICATIONS_PATH.with_name(NOTIFICATIONS_PATH.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, NOTIFICATIONS_PATH)


def _notification_id(event_id: str, kind: str, leg: str | None = None) -> str:
    raw = f"{event_id}|{kind}|{leg or ''}".encode("utf-8")
    return "TG_" + hashlib.sha256(raw).hexdigest()[:24].upper()


def _deliver_notification(notification_id: str, record: dict) -> bool:
    notifications = _load_notifications()
    item = notifications.get(notification_id)
    if not isinstance(item, dict):
        return False
    item.setdefault("attempts", 0)
    item.setdefault("created_ts", int(time.time() * 1000))
    item.setdefault("chats", {})
    chat_state = item["chats"]
    pending = [cid for cid, state in chat_state.items() if not bool(state.get("sent"))]
    if not pending:
        item["resolved"] = bool(chat_state) and all(bool(state.get("sent")) for state in chat_state.values())
        _save_notifications(notifications)
        return bool(item["resolved"])

    item["attempts"] = int(item.get("attempts", 0)) + 1
    results = send_detailed(str(item.get("text", "")), only_chat_ids=pending)
    now_ms = int(time.time() * 1000)
    for cid in pending:
        state = chat_state.setdefault(cid, {})
        state["attempts"] = int(state.get("attempts", 0)) + 1
        result = results.get(cid, {"sent": False, "error": "no telegram result"})
        state["sent"] = bool(result.get("sent"))
        state["last_attempt_ts"] = now_ms
        state["last_error"] = result.get("error")
        if state["sent"]:
            state["sent_ts"] = now_ms
            state["message_id"] = result.get("message_id")
    item["resolved"] = all(bool(state.get("sent")) for state in chat_state.values()) and bool(chat_state)
    notifications[notification_id] = item
    _save_notifications(notifications)
    return bool(item["resolved"])


def _queue_notification(notification_id: str, *, event_id: str, kind: str, symbol: str, direction: str, text: str, leg: str | None = None) -> bool:
    notifications = _load_notifications()
    item = notifications.get(notification_id)
    raw = os.environ.get("TG_CHAT_IDS") or os.environ.get("TG_CHAT_ID") or ""
    current_chat_ids = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
    if not isinstance(item, dict):
        chat_ids = current_chat_ids
        item = {
            "notification_id": notification_id,
            "event_id": event_id,
            "kind": kind,
            "leg": leg,
            "symbol": symbol,
            "direction": direction,
            "text": text,
            "created_ts": int(time.time() * 1000),
            "attempts": 0,
            "resolved": False,
            "chats": {cid: {"sent": False, "attempts": 0, "last_error": None} for cid in chat_ids},
        }
        notifications[notification_id] = item
        _save_notifications(notifications)
    else:
        chats = item.setdefault("chats", {})
        for cid in current_chat_ids:
            chats.setdefault(cid, {"sent": False, "attempts": 0, "last_error": None})
        item["resolved"] = False if any(not bool(v.get("sent")) for v in chats.values()) else bool(chats)
        notifications[notification_id] = item
        _save_notifications(notifications)
    return _deliver_notification(notification_id, item)


def _retry_pending_notifications() -> None:
    notifications = _load_notifications()
    pending = [
        (nid, item) for nid, item in notifications.items()
        if isinstance(item, dict) and not bool(item.get("resolved"))
    ]
    for nid, item in pending:
        try:
            _deliver_notification(nid, item)
        except Exception as exc:
            log.warning("[TELEGRAM] Pending notification retry failed %s: %s", nid, exc)


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
            normalized = {}
            for event_id, trade in data.items():
                if not isinstance(trade, dict):
                    continue
                t = dict(trade)
                t.setdefault("mae_pct", 0.0)
                t.setdefault("max_drawdown_pct", 0.0)
                t.setdefault("be_required", False)
                t.setdefault("be_last_error", None)
                t.setdefault("sl_order_history", [])
                t.setdefault("tp_mode", "single_tp" if len(t.get("tp_orders", [])) == 1 else "multi_tp")
                t.setdefault("effective_tp_levels", t.get("tp_levels", []))
                t.setdefault("effective_weighted_rr", t.get("planned_weighted_rr", 1.05))
                t.setdefault("close_journal_pending", False)
                t.setdefault("close_notification_pending", False)
                normalized[str(event_id)] = t
            return normalized
        log.error("[TRACKER] Invalid state: %s is not a JSON object", ACTIVE_TRADES_PATH)
        return {}
    except Exception as exc:
        log.error("[TRACKER] Corrupt state in %s: %s", ACTIVE_TRADES_PATH, exc)
        return {}


def _save_active_trades(trades: dict[str, dict]) -> None:
    ACTIVE_TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ACTIVE_TRADES_PATH.with_name(ACTIVE_TRADES_PATH.name + ".tmp")
    payload = json.dumps(trades, ensure_ascii=False, indent=2)
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, ACTIVE_TRADES_PATH)


def _close_record_exists(event_id: str) -> bool:
    if not TRADES_PATH.exists():
        return False
    try:
        with TRADES_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("record_type") == "TRADE_CLOSE" and str(obj.get("event_id")) == str(event_id):
                    return True
    except OSError as exc:
        log.warning("[TRACKER] Could not inspect close journal: %s", exc)
    return False


def _append_trade_record(record: dict) -> None:
    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRADES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    effective_tp_levels: list[dict] | None = None,
    tp_mode: str | None = None,
    effective_weighted_rr: float | None = None,
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
            if effective_tp_levels is not None:
                trade["effective_tp_levels"] = effective_tp_levels
            if tp_mode:
                trade["tp_mode"] = tp_mode
            if effective_weighted_rr is not None:
                trade["effective_weighted_rr"] = _safe_float(effective_weighted_rr, 1.05)
            trade["protection_last_updated_ts"] = int(time.time() * 1000)
            _save_active_trades(trades)
            return True

    return False


def _extract_setup_metrics(setup: dict | None) -> dict[str, Any]:
    if not isinstance(setup, dict):
        return {
            "planned_risk_pct": None,
            "planned_target_rr": None,
            "planned_weighted_rr": 1.05,
            "entry_reference": None,
            "invalidation_price": None,
            "target_price": None,
            "tp_levels": [],
            "effective_tp_levels": [],
            "effective_weighted_rr": 1.05,
            "tp_mode": "multi_tp",
        }

    return {
        "planned_risk_pct": _safe_float(setup.get("risk_pct"), 0.0) if setup.get("risk_pct") is not None else None,
        "planned_target_rr": _safe_float(setup.get("target_rr"), 0.0) if setup.get("target_rr") is not None else None,
        "planned_weighted_rr": _safe_float(setup.get("planned_weighted_rr", 1.05), 1.05),
        "effective_tp_levels": setup.get("effective_tp_levels") if isinstance(setup.get("effective_tp_levels"), list) else [],
        "effective_weighted_rr": _safe_float(setup.get("effective_weighted_rr", setup.get("planned_weighted_rr", 1.05)), 1.05),
        "tp_mode": str(setup.get("tp_mode", "multi_tp")),
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
    timeframe: str | None = None,
    coinalyze_row: Any = None,
    score: float = 50.0,
    setup: dict | None = None,
    requested_entry_price: float | None = None,
    entry_ts_ms: int | None = None,
) -> None:
    direction = _normalize_direction(direction)
    trades = _load_active_trades()
    now_ms = int(time.time() * 1000)
    actual_entry_ts = int(entry_ts_ms) if entry_ts_ms is not None and int(entry_ts_ms) > 0 else now_ms

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
    signal_reference_price = _safe_float((setup or {}).get("signal_price"), 0.0) if isinstance(setup, dict) else 0.0
    if signal_reference_price <= 0:
        signal_reference_price = _safe_float(setup_metrics.get("entry_reference"), 0.0)
    pre_order_reference_price = _safe_float((setup or {}).get("pre_order_reference_price"), 0.0) if isinstance(setup, dict) else 0.0
    entry_slippage_pct = None
    adverse_entry_slippage_pct = None

    if requested_price is not None and requested_price > 0:
        entry_slippage_pct = (actual_entry_price - requested_price) / requested_price * 100.0
        if direction == "LONG":
            adverse_entry_slippage_pct = max(0.0, entry_slippage_pct)
        else:
            adverse_entry_slippage_pct = max(0.0, -entry_slippage_pct)

    trade_id = "TR_" + hashlib.sha256(str(event_id).encode("utf-8")).hexdigest()[:24].upper()
    trades[event_id] = {
        "trade_id": trade_id,
        "event_id": event_id,
        "symbol": symbol,
        "name": name or symbol,
        "direction": direction,
        "entry_price": actual_entry_price,
        "actual_entry_price": actual_entry_price,
        "requested_entry_price": requested_price,
        "signal_reference_price": signal_reference_price if signal_reference_price > 0 else None,
        "pre_order_reference_price": pre_order_reference_price if pre_order_reference_price > 0 else requested_price,
        "entry_slippage_pct": entry_slippage_pct,
        "adverse_entry_slippage_pct": adverse_entry_slippage_pct,
        "initial_qty": actual_qty,
        "remaining_qty": actual_qty,
        "entry_ts": actual_entry_ts,
        "tp_orders": tp_orders if isinstance(tp_orders, list) else [],
        "sl_order": sl_result if isinstance(sl_result, dict) else {},
        "hit_legs": [],
        "be_activated": False,
        "be_activation_ts": None,
        "be_required": False,
        "be_last_error": None,
        "sl_order_history": [],
        "peak_pnl_pct": 0.0,
        "mae_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "current_pnl_pct": 0.0,
        "score": _safe_float(score, 50.0),
        "event_type": event_type,
        "timeframe": str(timeframe or (setup or {}).get("event_timeframe") or (setup or {}).get("timeframe") or "1h").lower(),
        "research": research,
        "setup": setup.copy() if isinstance(setup, dict) else {},
        "planned_risk_pct": setup_metrics["planned_risk_pct"],
        "planned_target_rr": setup_metrics["planned_target_rr"],
        "planned_weighted_rr": setup_metrics["planned_weighted_rr"],
        "planned_entry_reference": setup_metrics["entry_reference"],
        "planned_invalidation_price": setup_metrics["invalidation_price"],
        "planned_target_price": setup_metrics["target_price"],
        "tp_levels": setup_metrics["tp_levels"],
        "effective_tp_levels": setup_metrics["effective_tp_levels"],
        "tp_mode": setup_metrics["tp_mode"],
        "effective_weighted_rr": setup_metrics["effective_weighted_rr"],
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
        f"💰 <b>{name} ({symbol})</b>\n\n"
        f"Leg: <b>{leg}</b>\n"
        f"PnL TP: <b>+{pnl_pct:.2f}%</b>\n"
        f"Цена исполнения: <code>{exec_price:.8g}</code>\n"
        f"Закрыто: <code>{closed_qty:.8f}</code>\n"
        f"Осталось: <code>{remaining_qty:.8f} ({remaining_pct:.1f}%)</code>"
    )


def format_trade_closed_message(
    name: str, symbol: str, direction: str, entry_price: float, exit_price: float,
    pnl_pct: float, realized_rr: float | None, planned_rr: float | None,
    duration_min: float, peak_pnl: float, max_drawdown: float,
    exit_reason: str, event_type: str, timeframe: str = "1h",
) -> str:
    is_win = pnl_pct >= 0.0
    emoji = "💚" if is_win else "💔"
    pnl_sign = "+" if pnl_pct > 0 else ""
    realized_rr_text = f"{realized_rr:.3f}" if realized_rr is not None else "—"
    planned_rr_text = f"{planned_rr:.3f}" if planned_rr is not None else "—"

    lines = [
        f"{emoji} <b>{name} ({symbol}) — сделка закрыта</b>",
        "",
        f"Вход <code>{entry_price:.8g}</code> → Выход <code>{exit_price:.8g}</code>   <b>{pnl_sign}{pnl_pct:.2f}%</b>",
        f"Realized R:R: <b>{realized_rr_text}</b> · Planned Weighted R:R: <b>{planned_rr_text}</b>",
        f"Держали <b>{duration_min:.1f} мин</b> · пик <b>+{peak_pnl:.2f}%</b> · просадка <b>{max_drawdown:.2f}%</b>",
        f"Вход: <code>{event_type}</code> · TF <b>{str(timeframe or '1h').lower()}</b>",
        f"Выход: <b>{exit_reason}</b>",
    ]

    return "\n".join(lines)


def _cancel_old_sl_verified(symbol: str, direction: str, order_id: str, max_attempts: int = 3) -> tuple[bool, str]:
    """Cancel an SL order and verify it actually disappeared (audit fix B4).

    Returns (cancelled, message). After failing DELETE attempts we re-query
    open orders: some exchanges answer an already-executed/expired cancel with
    an error code while the order is in fact gone.
    """
    last_error = ""
    for attempt in range(max_attempts):
        try:
            resp = cancel_order(symbol, order_id)
        except Exception as exc:
            resp = {"code": -1, "msg": str(exc)}
        if isinstance(resp, dict) and resp.get("code") in (0, "0"):
            try:
                latest = get_open_protection_directional(symbol, direction)
                if latest.get("status") == "ok":
                    open_ids = {str(o.get("orderId", "")) for o in (latest.get("sl_orders", []) + latest.get("tp_orders", []))}
                    if str(order_id) not in open_ids:
                        return True, "cancelled_and_verified"
                    last_error = "cancel acknowledged but order is still visible"
                else:
                    last_error = "cancel acknowledged but openOrders verification failed"
            except Exception as exc:
                last_error = f"cancel acknowledged but verification failed: {exc}"
        else:
            last_error = str(resp.get("msg", resp)) if isinstance(resp, dict) else str(resp)
        if attempt + 1 < max_attempts:
            time.sleep(0.3 * (attempt + 1))

    try:
        prot = get_open_protection_directional(symbol, direction)
        if prot.get("status") == "ok":
            open_ids = {str(o.get("orderId", "")) for o in (prot.get("sl_orders", []) + prot.get("tp_orders", []))}
            if str(order_id) not in open_ids:
                return True, "already_gone_from_open_orders"
    except Exception as exc:
        last_error = f"{last_error}; openOrders verify failed: {exc}"

    return False, last_error


def _notify_be_failure(symbol: str, direction: str, detail: str, event_id: str | None = None) -> None:
    """Queue a durable Telegram alert for BE failures."""
    try:
        eid = str(event_id or f"{symbol}:{direction}:BE_FAILURE")
        nid = _notification_id(eid, "BE_FAILED")
        text = (
            f"🛑 <b>BE move failed ({symbol} {direction})</b>\n"
            f"Status: <code>{detail}</code>\n"
            f"Позиция может остаться со старым SL или без SL — требуется проверка."
        )
        _queue_notification(nid, event_id=eid, kind="BE_FAILED", symbol=symbol, direction=direction, text=text)
    except Exception as exc:
        log.error("[TELEGRAM] Could not queue BE failure notification: %s", exc)


def _move_sl_to_break_even(
    symbol: str, direction: str, entry_price: float, qty: float, old_sl_id: str | None, trade_id: str | None = None,
    old_sl_price: float | None = None,
) -> dict:
    """Move the stop-loss to break-even without ever holding two SL orders.

    Audit fix B4 (race condition): the previous implementation created the new
    STOP_MARKET first and cancelled the old SL only afterwards, leaving two
    live stops for a ms-to-seconds window; if both triggered the position
    could be over-closed or flipped. The order is now:

      Step 1: if a BE SL already exists -> keep it, try to cancel the old one.
      Step 2: cancel the OLD SL first and verify the cancellation.
              If it fails -> DO NOT create the new SL (position stays protected
              by the old stop; no double-SL window is possible).
      Step 3: create the new STOP_MARKET at entry price.
      Step 4: verify the new SL on the exchange; on failure restore the old
              stop price (best effort) and raise an error status.
    """
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

    # Always reconcile the actual live quantity immediately before changing the SL.
    live_pos = get_position_directional(symbol, direction)
    if str(live_pos.get("status", "")).lower() != "found":
        return {"status": "position_gone", "error": "position not found before BE", "order_id": "", "stop_price": entry_price}
    live_qty = abs(_safe_float(live_pos.get("positionAmt"), 0.0))
    if live_qty <= 0:
        return {"status": "position_gone", "error": "position quantity is zero before BE", "order_id": "", "stop_price": entry_price}
    qty = live_qty

    trade_token = str(trade_id) if trade_id else uuid.uuid4().hex.upper()[:16]
    be_client_id = f"EVT_BE_{trade_token}"

    verified = get_open_protection_directional(symbol, direction)
    if verified.get("status") == "ok":
        for order in verified.get("sl_orders", []):
            cid = str(order.get("clientOrderId", "")).upper()
            order_price = _safe_float(order.get("stopPrice") or order.get("price"), 0.0)
            order_qty = _safe_float(order.get("origQty") or order.get("quantity"), 0.0)
            price_matches = order_price > 0 and abs(order_price - entry_price) / max(entry_price, 1e-12) < 0.002
            qty_matches = order_qty > 0 and abs(order_qty - qty) <= max(qty * 1e-6, 1e-12)

            if (be_client_id.upper() in cid or price_matches) and qty_matches:
                existing_id = str(order.get("orderId", ""))
                cleanup_ok = True
                for other in verified.get("sl_orders", []):
                    other_id = str(other.get("orderId", ""))
                    if other_id and other_id != existing_id:
                        ok, note = _cancel_old_sl_verified(symbol, direction, other_id)
                        if not ok:
                            cleanup_ok = False
                            log.warning("[TRACKER] BE exists for %s but extra SL %s could not be removed: %s", symbol, other_id, note)
                return {
                    "status": "created" if cleanup_ok else "error",
                    "error": None if cleanup_ok else "duplicate SL cleanup failed",
                    "order_id": existing_id,
                    "client_order_id": cid or be_client_id,
                    "stop_price": entry_price,
                }

    # Step 2: remove EVERY currently visible non-BE SL before creating the new BE SL.
    # Relying only on local old_sl_id is unsafe after a restart/reconciliation because
    # there may be additional stale SLs unknown to local state.
    if verified.get("status") == "ok":
        visible_sl_ids = [str(o.get("orderId", "")) for o in verified.get("sl_orders", []) if o.get("orderId")]
        ids_to_cancel = []
        for oid in visible_sl_ids:
            if oid and oid not in ids_to_cancel:
                ids_to_cancel.append(oid)
        if old_sl_id and str(old_sl_id) not in ids_to_cancel:
            ids_to_cancel.append(str(old_sl_id))
        for oid in ids_to_cancel:
            ok, cancel_note = _cancel_old_sl_verified(symbol, direction, oid)
            if not ok:
                log.error(
                    "[TRACKER] %s %s: SL %s could not be cancelled (%s); new BE SL NOT created.",
                    direction, symbol, oid, cancel_note,
                )
                return {
                    "status": "error",
                    "error": f"old SL cancel failed: {cancel_note}; new SL not created (no double-SL window)",
                    "order_id": "",
                    "stop_price": entry_price,
                    "old_sl_id": oid,
                }

    def _restore_old_sl() -> bool:
        """Best-effort restore of protection when the new SL could not be placed."""
        if not old_sl_price or old_sl_price <= 0 or abs(old_sl_price - entry_price) / max(entry_price, 1e-12) < 1e-9:
            return False
        restore_side = "SELL" if direction == "LONG" else "BUY"
        restore_params = {
            "symbol": bx,
            "side": restore_side,
            "positionSide": direction,
            "type": "STOP_MARKET",
            "stopPrice": _format_price(old_sl_price, price_precision),
            "quantity": _format_qty(qty, precision),
            "clientOrderId": f"EVT_BE_RST_{trade_token}",
        }
        try:
            restore_resp = _post_protection_order_verified(
                symbol, direction, restore_params, restore_params["clientOrderId"], max_attempts=3, retry_delay=0.25
            )
        except Exception as exc:
            log.error("[TRACKER] Old SL restore exception for %s %s: %s", symbol, direction, exc)
            return False
        if not isinstance(restore_resp, dict) or restore_resp.get("code") != 0:
            return False
        restored_order = (restore_resp.get("data") or {}).get("order") or restore_resp.get("data") or {}
        restored_id = str(restored_order.get("orderId", ""))
        if not restored_id:
            return False
        try:
            latest = get_open_protection_directional(symbol, direction)
            if latest.get("status") != "ok":
                return False
            return any(
                str(o.get("orderId", "")) == restored_id
                and abs(_safe_float(o.get("stopPrice") or o.get("price"), 0.0) - old_sl_price) / max(old_sl_price, 1e-12) < 0.002
                and _safe_float(o.get("origQty") or o.get("quantity"), 0.0) > 0
                for o in latest.get("sl_orders", [])
            )
        except Exception:
            return False

    def _fail(error: str) -> dict:
        restored = _restore_old_sl()
        log.error(
            "[TRACKER] BE move failed for %s %s: %s; old SL restore %s.",
            direction, symbol, error, "succeeded" if restored else "FAILED",
        )
        _notify_be_failure(symbol, direction, f"{error}; restore={'ok' if restored else 'failed'}", event_id=trade_id)
        return {
            "status": "error",
            "error": error,
            "order_id": "",
            "stop_price": entry_price,
            "old_sl_restored": bool(restored),
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
        resp = _post_protection_order_verified(
            symbol, direction, params, be_client_id, max_attempts=3, retry_delay=0.25
        )
    except Exception as exc:
        return _fail(f"BE stop request exception: {exc}")

    if not isinstance(resp, dict) or resp.get("code") != 0:
        return _fail(f"BE stop failed: {resp}")

    order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
    new_order_id = str(order.get("orderId", ""))
    if not new_order_id:
        return _fail("BE stop response has no orderId")

    verified_after = get_open_protection_directional(symbol, direction)
    if verified_after.get("status") != "ok":
        return _fail("BE stop verification failed")

    found = any(
        str(o.get("orderId", "")) == new_order_id
        and abs(_safe_float(o.get("stopPrice") or o.get("price"), 0.0) - entry_price) / max(entry_price, 1e-12) < 0.002
        and abs(_safe_float(o.get("origQty") or o.get("quantity"), 0.0) - qty) <= max(qty * 1e-6, 1e-12)
        for o in verified_after.get("sl_orders", [])
    )
    if not found:
        return _fail("BE stop not visible on exchange with exact price/quantity")

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
    entry_ts = int(_safe_float(trade.get("entry_ts"), 0.0))
    if entry_price <= 0:
        return
    peak = _safe_float(trade.get("peak_pnl_pct", 0.0))
    mae = _safe_float(trade.get("mae_pct", 0.0))
    drawdown = _safe_float(trade.get("max_drawdown_pct", 0.0))
    for candle in sorted(candles, key=lambda x: int(_safe_float(x.get("open_time"), x.get("close_time", 0)))):
        open_ts = int(_safe_float(candle.get("open_time"), 0.0))
        close_ts = int(_safe_float(candle.get("close_time"), 0.0))
        if entry_ts and ((open_ts and open_ts < entry_ts) or (not open_ts and close_ts <= entry_ts)):
            continue
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
        prior_peak = peak
        peak = max(peak, favorable)
        mae = min(mae, adverse)
        drawdown = min(drawdown, adverse - max(prior_peak, favorable))
    trade["peak_pnl_pct"] = peak
    trade["mae_pct"] = mae
    trade["max_drawdown_pct"] = drawdown

def _get_exit_from_sl(symbol: str, sl_order_id: str | None) -> tuple[float | None, str | None]:
    if not sl_order_id:
        return None, None
    try:
        sl_info = get_order(symbol, sl_order_id)
    except Exception as exc:
        log.warning("[TRACKER] SL order query error for %s: %s", symbol, exc)
        return None, None

    if sl_info.get("status") != "ok":
        return None, None

    status = str(sl_info.get("order_status", "")).upper()
    if status != "FILLED":
        return None, None

    exit_price = _safe_float(sl_info.get("avg_price"), 0.0)
    return (exit_price if exit_price > 0 else None, "SL_FILLED")




def _get_filled_sl_from_trade(symbol: str, trade: dict) -> tuple[float | None, str | None]:
    candidates: list[tuple[str, str]] = []
    current = trade.get("sl_order") if isinstance(trade.get("sl_order"), dict) else {}
    if current.get("order_id"):
        candidates.append((str(current.get("order_id")), "current"))
    for item in trade.get("sl_order_history", []) or []:
        if isinstance(item, dict) and item.get("order_id"):
            candidates.append((str(item.get("order_id")), "history"))
    seen = set()
    for order_id, _source in candidates:
        if order_id in seen:
            continue
        seen.add(order_id)
        try:
            info = get_order(symbol, order_id)
        except Exception:
            continue
        if str(info.get("status", "")).lower() != "ok":
            continue
        if str(info.get("order_status", "")).upper() == "FILLED":
            px = _safe_float(info.get("avg_price"), 0.0)
            if px > 0:
                return px, order_id
    return None, None


def update_active_trades() -> None:
    _retry_pending_notifications()
    trades = _load_active_trades()
    if not trades:
        return

    now_ms = int(time.time() * 1000)
    updated_trades: dict[str, dict] = {}

    for event_id, trade in trades.items():
        if trade.get("closed", False):
            if trade.get("close_notification_pending"):
                try:
                    nid = _notification_id(str(event_id), "TRADE_CLOSE")
                    _retry_pending_notifications()
                except Exception as exc:
                    log.warning("[TELEGRAM] Closed trade notification retry failed %s: %s", event_id, exc)
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
                log.warning("[TRACKER] Kline fetch error for %s: %s", symbol, exc)

            current_pnl = _calc_trade_pnl_pct(entry_price, cur_price, direction)
            trade["current_pnl_pct"] = current_pnl
            trade["current_position_qty"] = pos_amt
            trade["last_observation_ts"] = now_ms

            # Retry a failed TP1 -> BE transition while the position remains open.
            if "tp1" in set(trade.get("hit_legs", [])) and not trade.get("be_activated") and rem_qty > 0:
                old_sl = trade.get("sl_order", {}) if isinstance(trade.get("sl_order"), dict) else {}
                old_sl_id = old_sl.get("order_id")
                old_sl_price = _safe_float(old_sl.get("stop_price"), 0.0) or None
                retry = _move_sl_to_break_even(symbol, direction, entry_price, rem_qty, old_sl_id, str(event_id).replace("EVT_", ""), old_sl_price=old_sl_price)
                if retry.get("status") == "created":
                    trade["sl_order"] = retry
                    trade["be_activated"] = True
                    trade["be_required"] = False
                    trade["be_last_error"] = None
                    trade["be_activation_ts"] = trade.get("be_activation_ts") or now_ms
                else:
                    trade["be_required"] = True
                    trade["be_last_error"] = retry.get("error")

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

                exec_price = _safe_float(order_info.get("avg_price"), 0.0)
                if exec_price <= 0:
                    log.warning("[TRACKER_TP] %s %s %s has no actual avgPrice; deferring realized PnL", symbol, direction, leg)
                    continue
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

                    log.info(
                        "[TRACKER_TP_HIT] 💰 %s (%s) Leg: %s | PnL: +%.2f%% | Exec: %.8g | Remaining: %.8f (%.1f%%)",
                        trade.get("name", symbol), symbol, leg, pnl_tp, exec_price, rem_qty, rem_pct
                    )

                    try:
                        nid = _notification_id(str(event_id), "TP_HIT", leg)
                        _queue_notification(
                            nid,
                            event_id=str(event_id),
                            kind="TP_HIT",
                            leg=leg,
                            symbol=symbol,
                            direction=direction,
                            text=format_tp_hit_message(
                                name=trade.get("name", symbol),
                                symbol=symbol,
                                leg=leg,
                                pnl_pct=pnl_tp,
                                exec_price=exec_price,
                                closed_qty=delta_qty,
                                remaining_qty=rem_qty,
                                remaining_pct=rem_pct,
                            ),
                        )
                    except Exception as exc:
                        log.error("[TELEGRAM] TP notification queue error %s %s %s: %s", symbol, leg, event_id, exc)

                    # ПЕРЕНОС В БЕЗУБЫТОК ПОСЛЕ TP1 (audit fix B4: cancel-first swap)
                    if leg == "tp1" and not trade.get("be_activated") and rem_qty > 0:
                        old_sl = trade.get("sl_order", {}) if isinstance(trade.get("sl_order"), dict) else {}
                        old_sl_id = old_sl.get("order_id")
                        old_sl_price = _safe_float(old_sl.get("stop_price"), 0.0) or None
                        new_sl = _move_sl_to_break_even(
                            symbol=symbol,
                            direction=direction,
                            entry_price=entry_price,
                            qty=rem_qty,
                            old_sl_id=old_sl_id,
                            trade_id=str(event_id).replace("EVT_", ""),
                            old_sl_price=old_sl_price,
                        )
                        if new_sl.get("status") == "created":
                            if old_sl_id:
                                history = trade.setdefault("sl_order_history", [])
                                if old_sl_id and all(str(x.get("order_id", "")) != str(old_sl_id) for x in history if isinstance(x, dict)):
                                    history.append({"order_id": str(old_sl_id), "stop_price": old_sl_price, "replaced_ts": now_ms})
                            trade["sl_order"] = new_sl
                            trade["be_activated"] = True
                            trade["be_activation_ts"] = now_ms
                            try:
                                nid = _notification_id(str(event_id), "BE_ACTIVATED")
                                _queue_notification(
                                    nid, event_id=str(event_id), kind="BE_ACTIVATED", symbol=symbol, direction=direction,
                                    text=f"✅ <b>BE activated ({trade.get('name', symbol)} {symbol})</b>\n"
                                         f"После {str(leg).upper()} стоп перенесён на <code>{entry_price:.8g}</code>.\n"
                                         f"Остаток позиции: <code>{rem_qty:.8f}</code>"
                                )
                            except Exception as exc:
                                log.error("[TELEGRAM] BE activation notification queue error %s: %s", event_id, exc)
                            log.info(
                                "[TRACKER_BE_ACTIVATED] %s (%s) TP1 taken. Stop-loss moved to Break-Even: %.8g (Risk: 0.00%%)",
                                trade.get("name", symbol), symbol, entry_price
                            )                             
                        else:
                            trade["be_required"] = True
                            trade["be_last_error"] = new_sl.get("error")
                            log.error(
                                "[TRACKER_BE_FAILED] %s (%s) Failed to move SL to Break-Even: %s",
                                trade.get("name", symbol), symbol, new_sl.get("error")
                            )

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

            sl_exit_price, filled_sl_id = _get_filled_sl_from_trade(symbol, trade)
            if closed_by_tp and rem_qty <= 1e-12:
                exit_reason = "TAKE_PROFIT_FULL"
            elif sl_exit_price is not None:
                exit_price = sl_exit_price
                if trade.get("be_activated") and abs(exit_price - entry_price) / max(entry_price, 1e-12) < 0.003:
                    exit_reason = "BREAK_EVEN"
                else:
                    exit_reason = "STOP_LOSS"
            else:
                exit_reason = "POSITION_CLOSED"

            if exit_price <= 0:
                exit_price = cur_price

            tp_closed_qty = realized_qty
            residual_qty = rem_qty if position_gone and rem_qty > 0 and init_qty > 0 else 0.0
            if residual_qty > 0:
                residual_pnl = _calc_trade_pnl_pct(entry_price, exit_price, direction)
                realized_weighted += residual_qty * residual_pnl
                realized_qty += residual_qty
                trade["remaining_qty"] = 0.0

            final_pnl = (realized_weighted / init_qty) if (init_qty > 0 and realized_qty > 0) else current_pnl
            if closed_by_tp and realized_qty > 0 and sl_exit_price is None:
                exit_price = entry_price * (1.0 + final_pnl / 100.0) if direction == "LONG" else entry_price * (1.0 - final_pnl / 100.0)
            planned_risk_pct = _derive_planned_risk_pct(trade)
            realized_rr = _calc_realized_rr(final_pnl, planned_risk_pct)
            stored_sl_price = _safe_float((trade.get("sl_order") or {}).get("stop_price"), 0.0) if isinstance(trade.get("sl_order"), dict) else 0.0
            actual_initial_sl_price = _safe_float(trade.get("planned_invalidation_price"), 0.0)
            actual_initial_sl_risk_pct = None
            if actual_initial_sl_price > 0 and entry_price > 0:
                actual_initial_sl_risk_pct = abs(entry_price - actual_initial_sl_price) / entry_price * 100.0
            elif stored_sl_price > 0 and entry_price > 0:
                actual_initial_sl_risk_pct = abs(entry_price - stored_sl_price) / entry_price * 100.0
            exit_reason_confidence = "confirmed" if exit_reason in {"TAKE_PROFIT_FULL", "STOP_LOSS", "BREAK_EVEN"} and (closed_by_tp or sl_exit_price is not None) else "unknown"
            planned_rr = _safe_float(trade.get("effective_weighted_rr", trade.get("planned_weighted_rr", 1.05)), 1.05)

            trade["remaining_qty"] = 0.0
            trade["realized_pnl_pct"] = final_pnl
            trade["realized_pnl_qty"] = realized_qty
            trade["realized_pnl_weighted_sum"] = realized_weighted
            trade["realized_rr"] = realized_rr
            trade["exit_price"] = exit_price
            trade["exit_reason"] = exit_reason
            trade["closed_ts"] = now_ms
            trade["duration_min"] = duration_min
            trade["closed"] = False
            close_journal_persisted = _close_record_exists(event_id)
            if not close_journal_persisted:
                try:
                    _append_trade_record({
                        "record_type": "TRADE_CLOSE",
                        "trade_id": trade.get("trade_id") or ("TR_" + hashlib.sha256(str(event_id).encode("utf-8")).hexdigest()[:24].upper()),
                        "event_id": event_id,
                        "symbol": symbol,
                        "direction": direction,
                        "event_type": trade.get("event_type"),
                        "closed_ts": now_ms,
                        "exit_reason": exit_reason,
                        "exit_reason_confidence": exit_reason_confidence,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "realized_pnl_pct": final_pnl,
                        "realized_rr": realized_rr,
                        "effective_weighted_rr": planned_rr,
                        "tp_mode": trade.get("tp_mode", "multi_tp"),
                        "effective_tp_levels": trade.get("effective_tp_levels", []),
                        "peak_pnl_pct": _safe_float(trade.get("peak_pnl_pct")),
                        "mae_pct": _safe_float(trade.get("mae_pct")),
                        "max_drawdown_pct": _safe_float(trade.get("max_drawdown_pct")),
                        "duration_min": duration_min,
                        "hit_legs": sorted(hit_legs),
                        "tp_filled_qty": filled_by_leg,
                        "tp_closed_qty": tp_closed_qty,
                        "sl_closed_qty": residual_qty if exit_reason in {"STOP_LOSS", "BREAK_EVEN", "POSITION_CLOSED"} else 0.0,
                        "total_closed_qty": init_qty,
                        "be_activated": bool(trade.get("be_activated")),
                        "research": trade.get("research", {}),
                        "setup": trade.get("setup", {}),
                    })
                    close_journal_persisted = True
                except Exception as exc:
                    log.error("[TRACKER] Trade close journal write failed for %s: %s", event_id, exc)
                    trade["closed"] = False
                    trade["close_journal_pending"] = True
                    updated_trades[event_id] = trade
                    continue

            log.info("[TRACKER_TRADE_CLOSED] %s (%s) | PnL: %+.2f%% | Realized R:R: %s | Planned R:R: %.2f | Exit: %.8g (%s) | Duration: %.1f min", emoji, trade.get("name", symbol), symbol, final_pnl, (f"{realized_rr:.3f}" if realized_rr is not None else "—"), planned_rr, exit_price, exit_reason, duration_min)

            notification_persisted = True
            try:
                nid = _notification_id(str(event_id), "TRADE_CLOSE")
                _queue_notification(
                    nid,
                    event_id=str(event_id),
                    kind="TRADE_CLOSE",
                    symbol=symbol,
                    direction=direction,
                    text=format_trade_closed_message(
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
                        timeframe=trade.get("timeframe") or (trade.get("setup") or {}).get("event_timeframe") or "1h",
                    ),
                )
            except Exception as exc:
                notification_persisted = False
                log.error("[TELEGRAM] Close notification queue error %s: %s", event_id, exc)

            if not notification_persisted:
                trade["closed"] = False
                trade["close_notification_pending"] = True
                updated_trades[event_id] = trade
                continue

            trade["close_notification_pending"] = False
            trade["close_journal_pending"] = False
            trade["closed"] = True
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
            log.exception("[TRACKER] Fatal trade error for event %s: %s", event_id, exc)
            updated_trades[event_id] = trade

    _save_active_trades(updated_trades)
