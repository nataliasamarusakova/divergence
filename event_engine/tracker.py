from __future__ import annotations

import json
import hashlib
import logging
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

from event_engine.bingx import (
    get_position_directional,
    get_order,
    get_order_by_client_order_id,
    get_fill_history,
    get_income_history,
    get_open_protection_directional,
    cancel_order,
    fetch_klines,
    get_live_price,
    to_bx_symbol,
    get_contract,
    _format_price,
    _format_qty,
    _request,
    ORDER_PATH,
    _post_protection_order_verified,
    _cancel_protection_order_verified,
    emergency_close_position,
)
from event_engine.telegram import send_detailed
from event_engine.formatting import format_price, format_rr
from event_engine.json_safe import json_safe

log = logging.getLogger("event_engine.tracker")

DATA = Path("data")
ACTIVE_TRADES_PATH = DATA / "active_trades.json"
TRADES_PATH = DATA / "trades.jsonl"
NOTIFICATIONS_PATH = DATA / "notifications.json"

# Early-loss exits were explicitly disabled by the operator. Keep the old
# thresholds only for historical compatibility/telemetry, but fail closed in
# code even if a stale workflow/environment accidentally sets the flag true.
EARLY_LOSS_CUT_ENV_REQUESTED = os.environ.get("EARLY_LOSS_CUT_ENABLED", "false").strip().lower() == "true"
EARLY_LOSS_CUT_ENABLED = False
if EARLY_LOSS_CUT_ENV_REQUESTED:
    log.error("[CONFIG_GUARD] EARLY_LOSS_CUT_ENABLED=true was requested by the environment but is hard-disabled in this release.")
EARLY_LOSS_CUT_2H_MIN = float(os.environ.get("EARLY_LOSS_CUT_2H_MIN", "120"))
EARLY_LOSS_CUT_2H_PNL = float(os.environ.get("EARLY_LOSS_CUT_2H_PNL", "-1.50"))
EARLY_LOSS_CUT_4H_MIN = float(os.environ.get("EARLY_LOSS_CUT_4H_MIN", "240"))
EARLY_LOSS_CUT_4H_PNL = float(os.environ.get("EARLY_LOSS_CUT_4H_PNL", "-2.50"))

# For the normal two-TP profile, move the remaining stop to break-even after TP1.
# A single-TP micro-position uses the nearest TP1 target for the full position;
# TP2/TP3 remain accepted for legacy tracked positions.
BE_AFTER_LEG = os.environ.get("BE_AFTER_LEG", "tp1").strip().lower()
if BE_AFTER_LEG not in {"tp1", "tp2"}:
    BE_AFTER_LEG = "tp1"


def _be_milestone_reached(hit_legs: set[str], is_single_tp: bool) -> bool:
    """A single-TP position is terminal at its sole configured TP leg."""
    if is_single_tp:
        return bool({"tp1", "tp2", "tp3"} & hit_legs)
    return BE_AFTER_LEG in hit_legs



def _durable_replace_text(path: Path, payload: str) -> None:
    """Atomically replace a JSON state file with file and directory durability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Atomic replacement still applies on filesystems without directory fsync.
            pass
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


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
    payload = json.dumps(json_safe(data), ensure_ascii=False, indent=2, allow_nan=False)
    _durable_replace_text(NOTIFICATIONS_PATH, payload)


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
    changed = False

    # Apply legacy suppressions before delivery. _deliver_notification reloads and
    # persists the latest file for each item; saving this original snapshot after
    # those deliveries would roll sent/resolved flags back and cause Telegram spam.
    for nid, item in notifications.items():
        if not isinstance(item, dict) or bool(item.get("resolved")):
            continue
        if str(item.get("kind", "")).upper() == "BE_ACTIVATED":
            item["resolved"] = True
            item["suppressed"] = True
            notifications[nid] = item
            changed = True
    if changed:
        _save_notifications(notifications)

    pending_ids = [
        nid for nid, item in notifications.items()
        if isinstance(item, dict)
        and not bool(item.get("resolved"))
        and str(item.get("kind", "")).upper() != "BE_ACTIVATED"
    ]
    for nid in pending_ids:
        try:
            _deliver_notification(nid, notifications.get(nid, {}))
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
                    log.error("[TRACKER] Invalid state record %s: expected object, got %s; skipping record", event_id, type(trade).__name__)
                    continue
                try:
                    t = dict(trade)
                    tp_orders = t.get("tp_orders")
                    if tp_orders is None:
                        t["tp_orders"] = []
                    elif not isinstance(tp_orders, list):
                        log.error("[TRACKER] Invalid tp_orders for state record %s: expected list, got %s; normalizing to []", event_id, type(tp_orders).__name__)
                        t["tp_orders"] = []
                    t.setdefault("mae_pct", 0.0)
                    t.setdefault("max_drawdown_pct", 0.0)
                    t.setdefault("be_required", False)
                    t.setdefault("be_last_error", None)
                    t.setdefault("sl_order_history", [])
                    tp_order_count = len(t["tp_orders"])
                    t.setdefault("tp_mode", "single_tp" if tp_order_count == 1 else ("two_tp" if tp_order_count == 2 else "multi_tp"))
                    t.setdefault("effective_tp_levels", t.get("tp_levels", []))
                    t.setdefault("effective_weighted_rr", t.get("planned_weighted_rr", 1.6625))
                    t.setdefault("close_journal_pending", False)
                    t.setdefault("close_notification_pending", False)
                    t.setdefault("close_cleanup_pending", False)
                    t.setdefault("close_cleanup_last_error", None)
                    normalized[str(event_id)] = t
                except Exception as exc:
                    log.error("[TRACKER] Invalid state record %s: %s; skipping record", event_id, exc)
            return normalized
        log.error("[TRACKER] Invalid state: %s is not a JSON object", ACTIVE_TRADES_PATH)
        return {}
    except Exception as exc:
        log.error("[TRACKER] Corrupt state in %s: %s", ACTIVE_TRADES_PATH, exc)
        return {}


def _save_active_trades(trades: dict[str, dict]) -> None:
    payload = json.dumps(json_safe(trades), ensure_ascii=False, indent=2, allow_nan=False)
    _durable_replace_text(ACTIVE_TRADES_PATH, payload)


def queue_protection_cleanup_pending(
    symbol: str, direction: str, reason: str, *, source_event_id: str | None = None
) -> str:
    """Persist rollback protection cleanup when the immediate exchange cleanup is inconclusive.

    A rollback-created order has no normal active-trade lifecycle, so without a
    durable cleanup-only record the next tracker cycle would never retry it. The
    pre-entry gate also uses this record to avoid reusing the symbol-side while
    an orphan conditional order may still exist.
    """
    direction = str(direction or "").upper()
    canonical_symbol = to_bx_symbol(symbol) or str(symbol or "").strip().upper().replace("_", "-").replace("/", "-")
    if direction not in {"LONG", "SHORT"} or not canonical_symbol:
        raise ValueError("valid symbol and LONG/SHORT direction are required for cleanup queue")
    cleanup_id = "ROLLBACK_CLEANUP_" + hashlib.sha256(f"{canonical_symbol}|{direction}".encode("utf-8")).hexdigest()[:20].upper()
    trades = _load_active_trades()
    prior = trades.get(cleanup_id, {}) if isinstance(trades.get(cleanup_id), dict) else {}
    prior.update({
        "event_id": cleanup_id,
        "symbol": canonical_symbol,
        "direction": direction,
        "closed": False,
        "remaining_qty": 0.0,
        "initial_qty": 0.0,
        "entry_price": 0.0,
        "close_cleanup_pending": True,
        "close_cleanup_last_error": str(reason)[:1000],
        "protection_cleanup_only": True,
        "protection_cleanup_source_event_id": str(source_event_id or ""),
        "close_notification_pending": False,
        "close_journal_pending": False,
    })
    trades[cleanup_id] = prior
    _save_active_trades(trades)
    log.critical("[TRACKER] Persisted rollback protection cleanup pending for %s %s (id=%s): %s", canonical_symbol, direction, cleanup_id, reason)
    return cleanup_id


def _load_close_record(event_id: str) -> dict | None:
    """Return the latest persisted close record for a pending notification.

    A close notification can fail to enter notifications.json after the close
    journal has already been written. Keeping enough context to rebuild that
    notification avoids re-running the exchange lifecycle for a position already
    confirmed closed.
    """
    if not TRADES_PATH.exists():
        return None
    found: dict | None = None
    try:
        with TRADES_PATH.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("record_type") == "TRADE_CLOSE" and str(obj.get("event_id")) == str(event_id):
                    found = obj
    except OSError as exc:
        log.warning("[TRACKER] Could not inspect close journal for notification recovery: %s", exc)
    return found


def _close_record_exists(event_id: str) -> bool:
    return _load_close_record(event_id) is not None


def _rebuild_close_notification_text(event_id: str, trade: dict) -> str | None:
    """Rebuild an interrupted close message from persisted journal/state only."""
    close_record = _load_close_record(event_id) or {}
    symbol = str(close_record.get("symbol") or trade.get("symbol") or "").strip()
    direction = str(close_record.get("direction") or trade.get("direction") or "").upper()
    entry_price = _safe_float(close_record.get("entry_price", trade.get("entry_price")), 0.0)
    if not symbol or direction not in {"LONG", "SHORT"} or entry_price <= 0:
        return None

    pnl_raw = close_record.get("realized_pnl_pct", trade.get("realized_pnl_pct"))
    pnl_pct = _safe_float(pnl_raw, None) if pnl_raw is not None else None
    rr_raw = close_record.get("realized_rr", trade.get("realized_rr"))
    realized_rr = _safe_float(rr_raw, None) if rr_raw is not None else None
    planned_raw = close_record.get("effective_weighted_rr", trade.get("planned_weighted_rr"))
    planned_rr = _safe_float(planned_raw, None) if planned_raw is not None else None
    duration = _safe_float(close_record.get("duration_min", trade.get("duration_min")), 0.0)
    peak = _safe_float(close_record.get("peak_pnl_pct", trade.get("peak_pnl_pct")), 0.0)
    drawdown = _safe_float(close_record.get("max_drawdown_pct", trade.get("max_drawdown_pct")), 0.0)
    exit_price_raw = close_record.get("exit_price", trade.get("exit_price"))
    exit_price = _safe_float(exit_price_raw, 0.0) if exit_price_raw is not None else None
    setup = trade.get("setup") if isinstance(trade.get("setup"), dict) else {}
    return format_trade_closed_message(
        name=str(trade.get("name") or close_record.get("name") or symbol),
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl_pct=pnl_pct,
        realized_rr=realized_rr,
        planned_rr=planned_rr,
        duration_min=duration,
        peak_pnl=peak,
        max_drawdown=drawdown,
        exit_reason=str(close_record.get("exit_reason") or trade.get("exit_reason") or "POSITION_CLOSED"),
        event_type=str(close_record.get("event_type") or trade.get("event_type") or "DIVERGENCE"),
        timeframe=str(trade.get("timeframe") or setup.get("event_timeframe") or "1h"),
    )


def _append_trade_record(record: dict) -> None:
    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRADES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")


def _fill_id(value: Any) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _fetch_trade_fill_history(record: dict, opened_record: dict | None, now_ms: int) -> tuple[list[dict], str | None]:
    """Fetch paginated actual fills for one known trade, with bounded time windows."""
    opened = opened_record if isinstance(opened_record, dict) else {}
    execution = opened.get("execution") if isinstance(opened.get("execution"), dict) else {}
    result_open = opened.get("result") if isinstance(opened.get("result"), dict) else {}
    symbol = str(record.get("symbol") or opened.get("symbol") or "").strip()
    if not symbol:
        return [], "missing_symbol"
    close_ts = int(_safe_float(record.get("closed_ts") or record.get("ts"), now_ms))
    entry_ts = int(_safe_float(
        result_open.get("fill_ts_ms") or result_open.get("fillTime") or opened.get("entry_ts_ms")
        or opened.get("ts") or (opened.get("signal") or {}).get("detected_at_ts"), 0.0
    ))
    if close_ts <= 0:
        close_ts = now_ms
    if entry_ts <= 0:
        # A bounded fallback is allowed to query history, but final reconstruction
        # still requires the exact complete close volume and close-time consistency.
        entry_ts = max(1, close_ts - 7 * 24 * 60 * 60 * 1000)
    start_ts = max(1, entry_ts - 60_000)
    end_ts = max(start_ts + 1, close_ts + 120_000)
    if end_ts - start_ts > 90 * 24 * 60 * 60 * 1000:
        return [], "trade_history_window_exceeds_90_days"

    event_id = str(record.get("event_id") or "<unknown>")
    log.info(
        "[TRACKER_RECOVERY] Fill-history scan START event=%s symbol=%s window=%s..%s span_hours=%.2f timeout=5.0s",
        event_id, symbol, start_ts, end_ts, (end_ts - start_ts) / 3_600_000.0,
    )

    # The documented endpoint paginates with pageIndex/pageSize (max 1000).
    # Small windows prevent a large response or undocumented range limit from
    # silently truncating fills for long-lived trades.
    chunk_ms = 24 * 60 * 60 * 1000
    all_fills: list[dict] = []
    seen_ids: set[str] = set()
    cursor = start_ts
    last_request_monotonic = 0.0
    while cursor < end_ts:
        chunk_end = min(end_ts, cursor + chunk_ms - 1)
        page_index = 1
        while page_index <= 10:
            wait = 0.22 - (time.monotonic() - last_request_monotonic)
            if last_request_monotonic > 0 and wait > 0:
                time.sleep(wait)
            request_started = time.monotonic()
            log.info(
                "[TRACKER_RECOVERY] Fill-history request START event=%s symbol=%s chunk=%s..%s page=%d page_size=1000 timeout=5.0s",
                event_id, symbol, cursor, chunk_end, page_index,
            )
            try:
                response = get_fill_history(
                    symbol, cursor, chunk_end, page_index=page_index,
                    page_size=1000, currency="USDT", timeout_sec=5.0,
                )
            except Exception as exc:
                request_elapsed = time.monotonic() - request_started
                log.warning(
                    "[TRACKER_RECOVERY] Fill-history request ERROR event=%s symbol=%s page=%d elapsed=%.2fs error=%s",
                    event_id, symbol, page_index, request_elapsed, exc,
                )
                return [], f"fill_history_exception:{exc}"
            request_elapsed = time.monotonic() - request_started
            last_request_monotonic = time.monotonic()
            if not isinstance(response, dict) or str(response.get("status", "")).lower() != "ok":
                response_error = str(response.get("error", "fill_history_unavailable")) if isinstance(response, dict) else "fill_history_invalid_response"
                log.warning(
                    "[TRACKER_RECOVERY] Fill-history request END event=%s symbol=%s page=%d elapsed=%.2fs status=ERROR error=%s",
                    event_id, symbol, page_index, request_elapsed, response_error,
                )
                return [], response_error
            rows = response.get("fills")
            if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
                log.warning(
                    "[TRACKER_RECOVERY] Fill-history request END event=%s symbol=%s page=%d elapsed=%.2fs status=INVALID_PAYLOAD",
                    event_id, symbol, page_index, request_elapsed,
                )
                return [], "fill_history_invalid_fills_payload"
            log.info(
                "[TRACKER_RECOVERY] Fill-history request END event=%s symbol=%s page=%d elapsed=%.2fs status=OK rows=%d",
                event_id, symbol, page_index, request_elapsed, len(rows),
            )
            for row in rows:
                # De-duplicate repeated IDs across page/time-boundary responses.
                identity = _fill_id(row.get("trade_id"))
                if not identity:
                    identity = "|".join(str(row.get(key) or "") for key in ("order_id", "symbol", "side", "time", "price", "qty"))
                if identity in seen_ids:
                    continue
                seen_ids.add(identity)
                all_fills.append(row)
            if len(rows) < 1000:
                break
            page_index += 1
        if page_index > 10:
            return [], "fill_history_page_limit_exceeded"
        cursor = chunk_end + 1
    return all_fills, None


def _rebuild_data_error_close(record: dict, opened_record: dict | None, fills: list[dict]) -> tuple[dict | None, str | None]:
    """Rebuild DATA_ERROR economics only if exchange fills cover the whole position."""
    opened = opened_record if isinstance(opened_record, dict) else {}
    execution = opened.get("execution") if isinstance(opened.get("execution"), dict) else {}
    open_result = opened.get("result") if isinstance(opened.get("result"), dict) else {}
    protection = open_result.get("protection") if isinstance(open_result.get("protection"), dict) else {}
    symbol = str(record.get("symbol") or opened.get("symbol") or "").upper().replace("-USDT", "")
    direction = str(record.get("direction") or opened.get("direction") or "").upper()
    entry_price = _safe_float(record.get("entry_price"), 0.0) or _safe_float(execution.get("actual_entry_price"), 0.0)
    target_qty = abs(_safe_float(record.get("total_closed_qty") or record.get("initial_qty"), 0.0))
    if target_qty <= 0:
        target_qty = abs(_safe_float(execution.get("actual_qty"), 0.0))
    if not symbol or direction not in {"LONG", "SHORT"} or entry_price <= 0 or target_qty <= 0:
        return None, "missing_trade_identity_or_quantity"

    entry_order_id = _fill_id(record.get("entry_order_id") or execution.get("order_id"))
    # Symbol/time/side/quantity coincidence is not a trade identity. Without a
    # journaled entry order ID, unrelated fills on the same contract can appear
    # to reconstruct this DATA_ERROR close. Leave the record unresolved instead.
    if not entry_order_id:
        return None, "missing_entry_order_id_for_reconstruction"
    sl_result = protection.get("sl_result") if isinstance(protection.get("sl_result"), dict) else {}
    sl_ids = {_fill_id(record.get("sl_order_id")), _fill_id(sl_result.get("order_id") or sl_result.get("orderId"))}
    sl_ids.discard("")
    tp_ids = {_fill_id(x) for x in record.get("tp_order_ids", []) if _fill_id(x)}
    if not tp_ids:
        tp_orders = protection.get("tp_orders") if isinstance(protection.get("tp_orders"), list) else []
        tp_ids = {_fill_id(item.get("order_id") or item.get("orderId")) for item in tp_orders if isinstance(item, dict)}
        tp_ids.discard("")

    close_side = "SELL" if direction == "LONG" else "BUY"
    entry_fills: list[dict] = []
    close_fills: list[dict] = []
    open_ts = int(_safe_float((open_result or {}).get("fill_ts_ms") or opened.get("ts"), 0.0))
    close_ts = int(_safe_float(record.get("closed_ts") or record.get("ts"), 0.0))
    for fill in fills:
        if not isinstance(fill, dict):
            continue
        fill_symbol = str(fill.get("symbol") or "").upper().replace("-USDT", "")
        if fill_symbol and fill_symbol != symbol:
            continue
        fill_ts = int(_safe_float(fill.get("time"), 0.0))
        # Missing time is not accepted for historical reconstruction: without it
        # a fill could belong to another position with the same symbol/direction.
        if fill_ts <= 0:
            continue
        if open_ts > 0 and fill_ts < open_ts - 60_000:
            continue
        if close_ts > 0 and fill_ts > close_ts + 120_000:
            continue
        pos_side = str(fill.get("position_side") or "").upper()
        if pos_side and pos_side not in {direction, "BOTH"}:
            continue
        order_id = _fill_id(fill.get("order_id"))
        side = str(fill.get("side") or "").upper()
        price = _safe_float(fill.get("price"), 0.0)
        qty = abs(_safe_float(fill.get("qty"), 0.0))
        if side not in {"BUY", "SELL"} or price <= 0 or qty <= 0:
            continue
        if entry_order_id and order_id == entry_order_id:
            entry_fills.append(fill)
        elif side == close_side:
            close_fills.append(fill)

    # Only fills from directly journaled exit/protection order IDs can be
    # attributed to this logical trade. Never fall back to every opposite-side
    # fill on the same symbol/time window when these IDs do not match.
    direct_exit_ids = set(sl_ids) | set(tp_ids)
    for key in ("order_id", "orderId", "orderID", "close_order_id", "exit_order_id", "manual_exit_order_id", "filled_sl_order_id"):
        value = _fill_id(record.get(key))
        if value:
            direct_exit_ids.add(value)
    for key in ("close_order_ids", "exit_order_ids", "tp_order_ids", "filled_order_ids"):
        value = record.get(key)
        if isinstance(value, list):
            direct_exit_ids.update(_fill_id(item) for item in value if _fill_id(item))
    direct_exit_ids.discard(entry_order_id)
    if not direct_exit_ids:
        return None, "missing_direct_exit_order_ids_for_reconstruction"
    close_fills = [fill for fill in close_fills if _fill_id(fill.get("order_id")) in direct_exit_ids]
    if not close_fills:
        return None, "no_fills_for_journaled_exit_order_ids"

    filled_qty = sum(abs(_safe_float(fill.get("qty"), 0.0)) for fill in close_fills)
    tolerance = max(target_qty * 0.002, 1e-10)
    entry_qty = sum(abs(_safe_float(fill.get("qty"), 0.0)) for fill in entry_fills)
    if not entry_fills:
        return None, "entry_order_has_no_confirmed_fills"
    if abs(entry_qty - target_qty) > tolerance:
        return None, f"entry_fill_qty_mismatch:{entry_qty:.12g}/{target_qty:.12g}"
    # Recompute entry VWAP from fills belonging to the exact entry order rather
    # than relying on a possibly tracker-derived journal price.
    entry_price = sum(
        abs(_safe_float(fill.get("qty"), 0.0)) * _safe_float(fill.get("price"), 0.0)
        for fill in entry_fills
    ) / entry_qty
    if not math.isfinite(entry_price) or entry_price <= 0:
        return None, "invalid_entry_fill_vwap"
    if filled_qty < target_qty - tolerance:
        return None, f"incomplete_close_fills:{filled_qty:.12g}/{target_qty:.12g}"
    if filled_qty > target_qty + tolerance:
        return None, f"close_fills_exceed_position:{filled_qty:.12g}/{target_qty:.12g}"

    blended_exit = sum(abs(_safe_float(fill.get("qty"), 0.0)) * _safe_float(fill.get("price"), 0.0) for fill in close_fills) / filled_qty
    weighted_pnl_pct = 0.0
    for fill in close_fills:
        price = _safe_float(fill.get("price"), 0.0)
        qty = abs(_safe_float(fill.get("qty"), 0.0))
        pct = ((price - entry_price) / entry_price * 100.0) if direction == "LONG" else ((entry_price - price) / entry_price * 100.0)
        weighted_pnl_pct += qty * pct
    realized_pnl_pct = weighted_pnl_pct / target_qty

    used_order_ids = sorted({_fill_id(fill.get("order_id")) for fill in close_fills if _fill_id(fill.get("order_id"))})
    used_trade_ids = sorted({_fill_id(fill.get("trade_id")) for fill in close_fills if _fill_id(fill.get("trade_id"))})
    used_order_set = set(used_order_ids)
    stop_order_set = used_order_set & sl_ids
    tp_order_set = used_order_set & tp_ids
    be_activated = bool(record.get("be_activated"))
    if stop_order_set:
        stop_fills = [fill for fill in close_fills if _fill_id(fill.get("order_id")) in stop_order_set]
        exit_qty = sum(abs(_safe_float(fill.get("qty"), 0.0)) for fill in stop_fills)
        exit_price = sum(abs(_safe_float(fill.get("qty"), 0.0)) * _safe_float(fill.get("price"), 0.0) for fill in stop_fills) / exit_qty
        exit_price_source = "BINGX_FILL_HISTORY_SL_FILL"
        exit_reason = "BREAK_EVEN" if be_activated and abs(exit_price - entry_price) / max(entry_price, 1e-12) < 0.003 else "STOP_LOSS"
    elif used_order_set and used_order_set.issubset(tp_ids):
        exit_price = blended_exit
        exit_price_source = "BINGX_FILL_HISTORY_TP_WEIGHTED_AVERAGE"
        exit_reason = "TAKE_PROFIT_FULL"
        exit_qty = filled_qty
    else:
        # External/manual market close: require a fully covered position, then use
        # the last actual closing order's average as final exit price. No mark/candle
        # price is introduced here.
        last_fill_ts = max(int(_safe_float(fill.get("time"), 0.0)) for fill in close_fills)
        last_order_fills = [fill for fill in close_fills if int(_safe_float(fill.get("time"), 0.0)) == last_fill_ts]
        last_order_ids = {_fill_id(fill.get("order_id")) for fill in last_order_fills if _fill_id(fill.get("order_id"))}
        if len(last_order_ids) != 1:
            return None, "ambiguous_final_close_order"
        last_order_id = next(iter(last_order_ids))
        final_fills = [fill for fill in close_fills if _fill_id(fill.get("order_id")) == last_order_id]
        exit_qty = sum(abs(_safe_float(fill.get("qty"), 0.0)) for fill in final_fills)
        exit_price = sum(abs(_safe_float(fill.get("qty"), 0.0)) * _safe_float(fill.get("price"), 0.0) for fill in final_fills) / exit_qty
        exit_price_source = "BINGX_FILL_HISTORY_FINAL_ORDER"
        exit_reason = "POSITION_CLOSED_BY_FILL_HISTORY"

    pnl_fields = [fill.get("realized_pnl") for fill in close_fills]
    if pnl_fields and all(value is not None for value in pnl_fields):
        realized_pnl_usdt = sum(float(value) for value in pnl_fields)
        realized_pnl_usdt_source = "BINGX_FILL_REALIZED_PNL"
    else:
        realized_pnl_usdt = None
        realized_pnl_usdt_source = "UNKNOWN_EXCHANGE_REALIZED_PNL_MISSING"

    all_fee_fills = entry_fills + close_fills
    if all(
        fill.get("fee") is not None and str(fill.get("fee_asset") or "").upper() == "USDT"
        for fill in all_fee_fills
    ):
        fees_usdt = sum(float(fill.get("fee")) for fill in all_fee_fills)
    else:
        fees_usdt = None

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_price_source": exit_price_source,
        "blended_exit_price": blended_exit,
        "exit_execution_qty": exit_qty,
        "total_close_fill_qty": filled_qty,
        "realized_pnl_pct": realized_pnl_pct,
        "realized_pnl_usdt": realized_pnl_usdt,
        "realized_pnl_usdt_source": realized_pnl_usdt_source,
        "fees_usdt": fees_usdt,
        "exit_order_ids": used_order_ids,
        "exit_trade_ids": used_trade_ids,
        "exit_reason": exit_reason,
        "fill_count": len(close_fills),
        "entry_fill_count": len(entry_fills),
        "confirmed_qty": target_qty,
    }, None


def _recover_data_error_trade_closes(*, now_ms: int | None = None, max_records: int = 3) -> int:
    """Repair DATA_ERROR close rows only from complete, verified exchange fills."""
    if not TRADES_PATH.exists():
        return 0
    now_ms = int(now_ms or time.time() * 1000)
    try:
        raw_lines = TRADES_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as exc:
        log.warning("[TRACKER_RECOVERY] Trade journal read failed: %s", exc)
        return 0

    cycle_started = time.monotonic()
    parsed: list[dict | None] = []
    open_by_event: dict[str, dict] = {}
    for raw_line in raw_lines:
        try:
            obj = json.loads(raw_line)
        except Exception:
            parsed.append(None)
            continue
        parsed.append(obj if isinstance(obj, dict) else None)
        if isinstance(obj, dict) and obj.get("record_type") == "TRADE_OPEN" and obj.get("event_id"):
            open_by_event[str(obj["event_id"])] = obj

    replacements: dict[int, str] = {}
    attempts_this_cycle = 0
    recovered = 0
    for index, record in enumerate(parsed):
        if attempts_this_cycle >= max(0, int(max_records)):
            break
        if not isinstance(record, dict) or record.get("record_type") != "TRADE_CLOSE":
            continue
        exit_reason_now = str(record.get("exit_reason") or "").upper()
        exit_price_now = _safe_float(record.get("exit_price"), 0.0)
        if exit_reason_now != "DATA_ERROR" and record.get("realized_pnl_pct") is not None and exit_price_now > 0:
            continue
        recovery = record.get("fill_history_recovery") if isinstance(record.get("fill_history_recovery"), dict) else {}
        if recovery.get("status") == "RECOVERED":
            continue
        if int(_safe_float(recovery.get("next_retry_ts"), 0.0)) > now_ms:
            continue

        attempts_this_cycle += 1
        event_id = str(record.get("event_id") or "")
        opened = open_by_event.get(event_id)
        record_started = time.monotonic()
        log.info(
            "[TRACKER_RECOVERY] DATA_ERROR attempt START event=%s attempt_in_cycle=%d/%d",
            event_id, attempts_this_cycle, max(0, int(max_records)),
        )
        fills, query_error = _fetch_trade_fill_history(record, opened, now_ms)
        result, reconstruct_error = (None, query_error) if query_error else _rebuild_data_error_close(record, opened, fills)
        log.info(
            "[TRACKER_RECOVERY] DATA_ERROR attempt END event=%s elapsed=%.2fs fills=%d query_error=%s reconstructed=%s",
            event_id, time.monotonic() - record_started, len(fills), query_error or "none", bool(result),
        )
        attempts = int(_safe_float(recovery.get("attempts"), 0.0)) + 1
        updated = dict(record)
        if result:
            updated.update({
                "entry_price": result["entry_price"],
                "exit_price": result["exit_price"],
                "exit_price_source": result["exit_price_source"],
                "blended_exit_price_recovered": result["blended_exit_price"],
                "exit_execution_qty": result["exit_execution_qty"],
                "total_close_fill_qty_recovered": result["total_close_fill_qty"],
                "exit_order_ids_recovered": result["exit_order_ids"],
                "exit_trade_ids_recovered": result["exit_trade_ids"],
                "realized_pnl_pct": result["realized_pnl_pct"],
                "realized_pnl_pct_source": "BINGX_FILL_HISTORY_PRICE_WEIGHTED",
                "realized_pnl_usdt": result["realized_pnl_usdt"],
                "realized_pnl_usdt_source": result["realized_pnl_usdt_source"],
                "fees_usdt": result["fees_usdt"],
                "funding_usdt": None,  # Unknown funding must remain null, not an assumed zero.
                "net_pnl_usdt": None,
                "pnl_costs_status": "FEES_RECONCILED_FUNDING_UNKNOWN" if result["fees_usdt"] is not None else "FEES_OR_ENTRY_FILLS_UNKNOWN_FUNDING_UNKNOWN",
                "exit_reason": result["exit_reason"],
                "exit_reason_confidence": "confirmed",
                "exit_reason_source": "BINGX_FILL_HISTORY",
                "data_error_previous": record.get("data_error") or record.get("exit_reason"),
                "data_error": None,
                "fill_history_recovery": {
                    "status": "RECOVERED", "attempts": attempts,
                    "recovered_ts": now_ms, "fills": result["fill_count"],
                    "entry_fills": result["entry_fill_count"], "qty_confirmed": result["confirmed_qty"],
                    "order_ids": result["exit_order_ids"], "trade_ids": result["exit_trade_ids"],
                },
            })
            risk_pct = _safe_float(updated.get("planned_risk_pct"), 0.0)
            updated["realized_rr"] = result["realized_pnl_pct"] / risk_pct if risk_pct > 0 else None
            replacements[index] = json.dumps(json_safe(updated), ensure_ascii=False, allow_nan=False) + "\n"
            recovered += 1
            log.warning("[TRACKER_RECOVERY] Recovered DATA_ERROR close %s from %d BingX fill rows (qty=%.12g, final_exit=%.12g).", event_id, result["fill_count"], result["total_close_fill_qty"], result["exit_price"])
        else:
            delay_ms = min(6 * 60 * 60 * 1000, 30 * 60 * 1000 * (2 ** min(attempts - 1, 4)))
            updated["fill_history_recovery"] = {
                "status": "PENDING", "attempts": attempts,
                "last_attempt_ts": now_ms, "next_retry_ts": now_ms + delay_ms,
                "last_error": str(reconstruct_error or "insufficient_or_ambiguous_fills"),
            }
            replacements[index] = json.dumps(json_safe(updated), ensure_ascii=False, allow_nan=False) + "\n"
            log.info("[TRACKER_RECOVERY] DATA_ERROR close %s remains unresolved: %s", event_id, reconstruct_error)

    if replacements:
        tmp_path = TRADES_PATH.with_name(TRADES_PATH.name + ".recovery.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as stream:
                for index, raw_line in enumerate(raw_lines):
                    stream.write(replacements.get(index, raw_line if raw_line.endswith("\n") else raw_line + "\n"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, TRADES_PATH)
        except OSError as exc:
            log.error("[TRACKER_RECOVERY] Atomic journal update failed: %s", exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return 0
    log.info(
        "[TRACKER_RECOVERY] Cycle END elapsed=%.2fs attempted=%d recovered=%d rewritten_rows=%d",
        time.monotonic() - cycle_started, attempts_this_cycle, recovered, len(replacements),
    )
    return recovered


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
                trade["effective_weighted_rr"] = _safe_float(effective_weighted_rr, 1.6625)
            trade["protection_last_updated_ts"] = int(time.time() * 1000)
            _save_active_trades(trades)
            return True

    return False


def _extract_setup_metrics(setup: dict | None) -> dict[str, Any]:
    if not isinstance(setup, dict):
        return {
            "planned_risk_pct": None,
            "planned_target_rr": None,
            "planned_weighted_rr": 1.6625,
            "entry_reference": None,
            "invalidation_price": None,
            "target_price": None,
            "tp_levels": [],
            "effective_tp_levels": [],
            "effective_weighted_rr": 1.6625,
            "tp_mode": "two_tp",
        }

    return {
        "planned_risk_pct": _safe_float(setup.get("risk_pct"), 0.0) if setup.get("risk_pct") is not None else None,
        "planned_target_rr": _safe_float(setup.get("target_rr"), 0.0) if setup.get("target_rr") is not None else None,
        "planned_weighted_rr": _safe_float(setup.get("planned_weighted_rr", 1.6625), 1.6625),
        "effective_tp_levels": setup.get("effective_tp_levels") if isinstance(setup.get("effective_tp_levels"), list) else [],
        "effective_weighted_rr": _safe_float(setup.get("effective_weighted_rr", setup.get("planned_weighted_rr", 1.3875)), 1.3875),
        "tp_mode": str(setup.get("tp_mode", "two_tp")),
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
        "initial_notional_usdt": actual_entry_price * actual_qty,
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
        f"Цена исполнения: <code>{format_price(exec_price)}</code>\n"
        f"Закрыто: <code>{closed_qty:.8f}</code>\n"
        f"Осталось: <code>{remaining_qty:.8f} ({remaining_pct:.1f}%)</code>"
    )


def format_trade_closed_message(
    name: str, symbol: str, direction: str, entry_price: float, exit_price: float,
    pnl_pct: float | None, realized_rr: float | None, planned_rr: float | None,
    duration_min: float, peak_pnl: float, max_drawdown: float,
    exit_reason: str, event_type: str, timeframe: str = "1h",
) -> str:
    is_data_error = pnl_pct is None
    is_win = (pnl_pct >= 0.0) if pnl_pct is not None else False
    emoji = "⚠️" if is_data_error else ("💚" if is_win else "💔")
    pnl_sign = "+" if pnl_pct is not None and pnl_pct > 0 else ""
    pnl_text = "DATA_ERROR" if is_data_error else f"{pnl_sign}{pnl_pct:.2f}%"
    realized_rr_text = format_rr(realized_rr, decimals=2) if realized_rr is not None else "—"
    planned_rr_text = format_rr(planned_rr, decimals=2) if planned_rr is not None else "—"

    exit_text = format_price(exit_price) if exit_price is not None and _safe_float(exit_price, 0.0) > 0 else "UNKNOWN"
    lines = [
        f"{emoji} <b>{name} ({symbol}) — сделка закрыта</b>",
        "",
        f"Вход <code>{format_price(entry_price)}</code> → Выход <code>{exit_text}</code>   <b>{pnl_text}</b>",
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


def _notify_position_state_unknown(
    symbol: str, direction: str, detail: str, event_id: str | None = None, *, kind: str = "POSITION_STATE_UNKNOWN",
) -> None:
    """Queue a durable operator alert when exchange position state is ambiguous."""
    try:
        now_ms = int(time.time() * 1000)
        detail_key = hashlib.sha256(str(detail).strip().encode("utf-8")).hexdigest()[:12].upper()
        incident_key = f"{str(symbol).upper()}:{str(direction).upper()}:{kind}:{detail_key}:{now_ms // 3_600_000}"
        # The trading event id can change when an exchange position is re-adopted.
        # Alert identity is based on the unresolved incident and an hourly bucket,
        # so repeated polling sends at most one copy per hour, while persistent
        # incidents still remind the operator on a later hour.
        eid = incident_key
        nid = _notification_id(eid, kind)
        text = (
            f"🚨 <b>BingX position state requires verification ({symbol} {direction})</b>\n"
            f"Status: <code>{detail}</code>\n"
            "Бот не считает позицию закрытой и не отправляет слепой повторный ордер. "
            "Срочно проверьте фактическую позицию и SL/TP на бирже."
        )
        _queue_notification(nid, event_id=eid, kind=kind, symbol=symbol, direction=direction, text=text)
    except Exception as exc:
        log.error("[TELEGRAM] Could not queue position-state alert: %s", exc)


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
    old_sl_price: float | None = None, *, prior_order_may_exist: bool = False, prior_order_id: str | None = None,
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
      Step 4: verify the new SL on the exchange. Restore the old stop only when
              the new POST is known to have been rejected; after an ambiguous
              POST/verification result, do not recreate a second SL and require
              reconciliation/retry before another mutation.
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
    live_status = str(live_pos.get("status", "")).lower()
    if live_status == "not_found":
        return {"status": "position_gone", "error": "position absence confirmed before BE", "order_id": "", "stop_price": entry_price}
    if live_status != "found":
        return {"status": "error", "error": f"position state unknown before BE: {live_pos.get('error', live_status or 'unknown')}", "order_id": "", "stop_price": entry_price}
    live_qty = abs(_safe_float(live_pos.get("positionAmt"), 0.0))
    if live_qty <= 0:
        return {"status": "error", "error": "position quantity invalid before BE", "order_id": "", "stop_price": entry_price}
    qty = live_qty

    trade_token = str(trade_id) if trade_id else uuid.uuid4().hex.upper()[:16]
    be_client_id = "EVTBE" + hashlib.sha256(str(trade_token).upper().encode()).hexdigest().upper()[:24]

    # Never change SL protection when the open-orders snapshot is not authoritative.
    # Otherwise an API timeout/malformed response can skip old-SL cancellation and
    # still fall through to posting a second break-even SL.
    try:
        verified = get_open_protection_directional(symbol, direction)
    except Exception as exc:
        verified = {"status": "error", "error": str(exc)}
    if not isinstance(verified, dict) or verified.get("status") != "ok":
        detail = str(verified.get("error", verified.get("status", "unknown"))) if isinstance(verified, dict) else "malformed openOrders response"
        error = f"openOrders state unknown before BE; refusing SL change: {detail}"
        log.error("[TRACKER] %s %s: %s", direction, symbol, error)
        _notify_be_failure(symbol, direction, error, event_id=trade_id)
        return {
            "status": "error",
            "error": error,
            "order_id": "",
            "stop_price": entry_price,
            "old_sl_id": str(old_sl_id or ""),
            "protection_state_unknown": True,
            "sl_mutation_attempted": False,
        }

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
                    "new_order_may_exist": not cleanup_ok,
                    "protection_state_unknown": not cleanup_ok,
                }

    # A prior successful/ambiguous BE POST must not be repeated merely because
    # a subsequent openOrders snapshot does not yet display it. The matching
    # branch above can adopt an identifiable BE order; otherwise leave all live
    # orders untouched and let a later reconciliation/operator resolve identity.
    if prior_order_may_exist:
        error = (
            "prior BE stop may already exist but is not identifiable in the current "
            "openOrders snapshot; refusing another SL mutation"
        )
        log.critical(
            "[TRACKER] %s %s: %s (prior_order_id=%s)",
            direction, symbol, error, prior_order_id or "unknown",
        )
        _notify_be_failure(symbol, direction, error, event_id=trade_id)
        return {
            "status": "error", "error": error, "order_id": str(prior_order_id or ""),
            "stop_price": entry_price, "new_order_may_exist": True,
            "protection_state_unknown": True, "sl_mutation_attempted": False,
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
        restore_client_id = "EVTBERST" + hashlib.sha256(str(trade_token).upper().encode()).hexdigest().upper()[:16]
        restore_params = {
            "symbol": bx,
            "side": restore_side,
            "positionSide": direction,
            "type": "STOP_MARKET",
            "stopPrice": _format_price(old_sl_price, price_precision),
            "quantity": _format_qty(qty, precision),
        }
        try:
            restore_resp = _post_protection_order_verified(
                symbol, direction, restore_params, None, max_attempts=3, retry_delay=0.25
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

    def _fail(error: str, *, restore_old: bool = True, new_order_may_exist: bool = False, new_order_id: str = "") -> dict:
        # Restoring the old SL is safe only when we know the new POST was rejected
        # before creating an order. If a successful POST may already have created
        # the BE SL, a failed follow-up query must not recreate a second STOP.
        restored = _restore_old_sl() if restore_old else False
        restore_state = "succeeded" if restored else ("not_attempted_due_to_unknown_new_order" if not restore_old else "FAILED")
        log.error(
            "[TRACKER] BE move failed for %s %s: %s; old SL restore %s; new_order_may_exist=%s new_order_id=%s.",
            direction, symbol, error, restore_state, new_order_may_exist, new_order_id or "unknown",
        )
        _notify_be_failure(symbol, direction, f"{error}; restore={restore_state}; new_order_may_exist={new_order_may_exist}", event_id=trade_id)
        return {
            "status": "error",
            "error": error,
            "order_id": new_order_id,
            "stop_price": entry_price,
            "old_sl_restored": bool(restored),
            "new_order_may_exist": bool(new_order_may_exist),
            "protection_state_unknown": bool(new_order_may_exist or not restore_old),
        }

    sl_side = "SELL" if direction == "LONG" else "BUY"
    params = {
        "symbol": bx,
        "side": sl_side,
        "positionSide": direction,
        "type": "STOP_MARKET",
        "stopPrice": _format_price(entry_price, price_precision),
        "quantity": _format_qty(qty, precision),
    }

    try:
        resp = _post_protection_order_verified(
            symbol, direction, params, None, max_attempts=3, retry_delay=0.25
        )
    except Exception as exc:
        # Transport exceptions are ambiguous: the exchange may have accepted the
        # POST. Do not blindly restore the prior stop and risk two live SLs.
        return _fail(f"BE stop request exception: {exc}", restore_old=False, new_order_may_exist=True)

    if not isinstance(resp, dict) or resp.get("code") not in (0, "0"):
        code = str(resp.get("code", "")) if isinstance(resp, dict) else ""
        explicit_rejection = bool(code and code not in {"-1", "0"} and not resp.get("protection_state_unknown")) if isinstance(resp, dict) else False
        return _fail(
            f"BE stop failed: {resp}",
            restore_old=explicit_rejection,
            new_order_may_exist=not explicit_rejection,
        )

    order = (resp.get("data") or {}).get("order") or resp.get("data") or {}
    new_order_id = str(order.get("orderId", ""))
    if not new_order_id:
        # code=0 without an ID may still mean the conditional order was accepted.
        return _fail("BE stop response has no orderId", restore_old=False, new_order_may_exist=True)

    try:
        verified_after = get_open_protection_directional(symbol, direction)
    except Exception as exc:
        return _fail(
            f"BE stop verification exception: {exc}", restore_old=False,
            new_order_may_exist=True, new_order_id=new_order_id,
        )
    if not isinstance(verified_after, dict) or verified_after.get("status") != "ok":
        return _fail(
            "BE stop verification failed", restore_old=False,
            new_order_may_exist=True, new_order_id=new_order_id,
        )

    found = any(
        str(o.get("orderId", "")) == new_order_id
        and abs(_safe_float(o.get("stopPrice") or o.get("price"), 0.0) - entry_price) / max(entry_price, 1e-12) < 0.002
        and abs(_safe_float(o.get("origQty") or o.get("quantity"), 0.0) - qty) <= max(qty * 1e-6, 1e-12)
        for o in verified_after.get("sl_orders", [])
    )
    if not found:
        # An order accepted by POST may be invisible due to API lag or may have
        # already triggered. Restoring the old SL here can create a duplicate.
        return _fail(
            "BE stop not visible on exchange with exact price/quantity",
            restore_old=False, new_order_may_exist=True, new_order_id=new_order_id,
        )

    return {
        "status": "created",
        "order_id": new_order_id,
        "client_order_id": order.get("clientOrderId") or None,
        "stop_price": entry_price,
    }


def _calc_trade_pnl_pct(entry_price: float, exit_price: float, direction: str) -> float | None:
    if entry_price <= 0 or exit_price <= 0:
        log.warning(
            "[TRACKER_DATA_ERROR] Cannot calculate trade PnL: entry_price=%r exit_price=%r direction=%s",
            entry_price, exit_price, direction,
        )
        return None
    if str(direction).upper() == "LONG":
        return (exit_price - entry_price) / entry_price * 100.0
    return (entry_price - exit_price) / entry_price * 100.0


def _early_loss_cut_guard(symbol: str, direction: str, current_price: float, qty: float, sl_distance_pct: float = 0.50) -> tuple[bool, str, float]:
    """Re-check live position + live SL immediately before an Early Loss Cut MARKET close.

    Returns (safe_to_close, reason, live_qty). A stale/missing exchange state blocks the
    local close rather than risking a second order. If no live SL is present, we do not
    infer safety; the caller should let reconciliation repair protection on the next cycle.
    """
    try:
        pos = get_position_directional(symbol, direction)
    except Exception as exc:
        return False, f"position_recheck_error:{exc}", qty
    pos_status = str(pos.get("status", "")).lower()
    if pos_status == "not_found":
        return False, "position_absence_confirmed_on_recheck", 0.0
    if pos_status != "found":
        return False, f"position_state_unknown_on_recheck:{pos.get('error', pos_status or 'unknown')}", qty
    live_qty = abs(_safe_float(pos.get("positionAmt"), 0.0))
    if live_qty <= 0:
        return False, "position_quantity_invalid_on_recheck", qty

    try:
        protection = get_open_protection_directional(symbol, direction)
    except Exception as exc:
        return False, f"protection_recheck_error:{exc}", live_qty
    if str(protection.get("status", "")).lower() != "ok":
        return False, "protection_state_unknown", live_qty

    sl_orders = [o for o in protection.get("sl_orders", []) if isinstance(o, dict)]
    if len(sl_orders) != 1:
        return False, f"expected_one_live_sl_got_{len(sl_orders)}", live_qty
    try:
        sl_price = _safe_float(sl_orders[0].get("stopPrice") or sl_orders[0].get("price"), 0.0)
    except Exception:
        sl_price = 0.0
    if sl_price <= 0 or current_price <= 0:
        return False, "invalid_live_price_or_sl", live_qty

    distance_pct = abs(current_price - sl_price) / max(abs(sl_price), 1e-12) * 100.0
    if distance_pct <= max(0.0, sl_distance_pct):
        return False, f"near_live_sl:{distance_pct:.4f}%", live_qty

    return True, "safe_to_close", live_qty


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
    if pnl_pct is None or risk_pct is None or risk_pct <= 0:
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

def _update_live_pnl_extrema(trade: dict, current_pnl_pct: float | None) -> None:
    """Fold the live ticker observation into MFE/MAE telemetry.

    ``_update_mfe_mae`` scans only closed candles, while current PnL comes from
    the live ticker. Without this fold, a live price above the last closed-bar
    high could leave ``current_pnl_pct > peak_pnl_pct`` and make MFE incomplete.
    This function is telemetry-only and does not affect exit decisions.
    """
    if current_pnl_pct is None or not math.isfinite(float(current_pnl_pct)):
        return
    current = float(current_pnl_pct)
    peak = _safe_float(trade.get("peak_pnl_pct", 0.0))
    mae = _safe_float(trade.get("mae_pct", 0.0))
    drawdown = _safe_float(trade.get("max_drawdown_pct", 0.0))
    prior_peak = peak
    peak = max(peak, current)
    mae = min(mae, current)
    live_drawdown = current - max(prior_peak, current)
    drawdown = min(drawdown, live_drawdown)
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




def _vwap_actual_fill_rows(
    rows: Any,
    order_id: str,
    direction: str | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    expected_qty: float | None = None,
) -> float | None:
    """Calculate VWAP only from uniquely identifiable, actual fills for one order.

    A trigger price or a malformed/partially paginated list is not execution proof.
    Every row must carry an actual fill/trade identifier, exact order id, valid side,
    positive price/quantity, and a timestamp inside the requested reconciliation window.
    """
    if not isinstance(rows, list) or not rows:
        return None
    weighted = 0.0
    total_qty = 0.0
    wanted_id = str(order_id)
    wanted_side = str(direction or "").upper()
    for fill in rows:
        if not isinstance(fill, dict):
            return None
        row_order = str(fill.get("order_id") or fill.get("orderId") or fill.get("orderID") or "")
        if not row_order or row_order != wanted_id:
            return None
        fill_id = str(
            fill.get("trade_id") or fill.get("tradeId") or fill.get("tradeID")
            or fill.get("fill_id") or fill.get("fillId") or fill.get("id") or ""
        ).strip()
        if not fill_id:
            return None
        pos_side = str(fill.get("position_side") or fill.get("positionSide") or "").upper()
        if wanted_side and pos_side and pos_side not in {wanted_side, "BOTH"}:
            return None
        price = _safe_float(fill.get("price"), 0.0)
        qty = abs(_safe_float(fill.get("qty", fill.get("executedQty", fill.get("vol"))), 0.0))
        timestamp = int(_safe_float(fill.get("time", fill.get("timestamp")), 0.0))
        if 0 < timestamp < 100_000_000_000:
            timestamp *= 1000
        if (price <= 0 or qty <= 0 or timestamp <= 0 or not math.isfinite(price * qty)):
            return None
        if start_ts is not None and timestamp < int(start_ts):
            return None
        if end_ts is not None and timestamp > int(end_ts):
            return None
        weighted += price * qty
        total_qty += qty
    if total_qty <= 0:
        return None
    if expected_qty is not None and expected_qty > 0:
        if abs(total_qty - expected_qty) > max(1e-8, expected_qty * 0.01):
            return None
    vwap = weighted / total_qty
    return vwap if math.isfinite(vwap) and vwap > 0 else None


def _get_filled_sl_from_trade(symbol: str, trade: dict) -> tuple[float | None, str | None]:
    """Recover actual stop execution price, never the trigger price.

    Exchange order average price plus positive executed quantity is accepted for a
    FILLED order. If that information is incomplete, exact fillHistory rows are
    required even when the order status says FILLED. Trigger-only statuses are never
    treated as fills.
    """
    candidates: list[str] = []
    current = trade.get("sl_order") if isinstance(trade.get("sl_order"), dict) else {}
    if current.get("order_id"):
        candidates.append(str(current.get("order_id")))
    for item in trade.get("sl_order_history", []) or []:
        if isinstance(item, dict) and item.get("order_id"):
            candidates.append(str(item.get("order_id")))
    direction = str(trade.get("direction") or "").upper()
    now_ms = int(time.time() * 1000)
    entry_ts = int(_safe_float(trade.get("entry_ts"), 0.0))
    if entry_ts and entry_ts < 100_000_000_000:
        entry_ts *= 1000
    # BingX recent fill history is bounded; don't query an arbitrarily old entry.
    start_ts = max(now_ms - 6 * 24 * 60 * 60 * 1000, entry_ts - 60_000 if entry_ts > 0 else 1)
    end_ts = now_ms + 60_000
    for order_id in dict.fromkeys(candidates):
        try:
            info = get_order(symbol, order_id)
        except Exception:
            info = {}
        if not isinstance(info, dict) or str(info.get("status", "")).lower() != "ok":
            continue
        status = str(info.get("order_status", "")).upper()
        # Triggered/active/partially-filled orders are not final exits. Their actual
        # fills may be useful diagnostics, but cannot explain the disappearance of
        # the whole position as a completed SL execution.
        if status != "FILLED":
            continue
        px = _safe_float(info.get("avg_price"), 0.0)
        executed_qty = _safe_float(info.get("executed_qty"), 0.0)
        if px > 0 and executed_qty > 0 and math.isfinite(px * executed_qty):
            return px, order_id

        raw_order = info.get("raw_order") if isinstance(info.get("raw_order"), dict) else {}
        embedded = next((raw_order.get(k) for k in ("fills", "fill_orders", "fillOrders", "trade_list", "tradeList") if isinstance(raw_order.get(k), list)), None)
        if embedded is not None:
            px = _vwap_actual_fill_rows(embedded, order_id, direction, start_ts, end_ts, executed_qty or None)
            if px is not None:
                return px, order_id

        try:
            response = get_fill_history(
                symbol, start_ts, end_ts, order_id=order_id,
                page_index=1, page_size=1000, timeout_sec=5.0,
            )
        except Exception:
            continue
        if not isinstance(response, dict) or str(response.get("status", "")).lower() != "ok":
            continue
        if response.get("has_more") is True:
            continue
        fills = response.get("fills")
        px = _vwap_actual_fill_rows(fills, order_id, direction, start_ts, end_ts, executed_qty or None)
        if px is not None:
            return px, order_id
    return None, None


def _stop_price_for_order(trade: dict, order_id: str | None) -> float | None:
    """Return the trigger price belonging to the SL order that actually filled."""
    if not order_id:
        return None
    target_id = str(order_id)
    current = trade.get("sl_order") if isinstance(trade.get("sl_order"), dict) else {}
    if str(current.get("order_id") or "") == target_id:
        price = _safe_float(current.get("stop_price"), 0.0)
        if price > 0 and math.isfinite(price):
            return price
    history = trade.get("sl_order_history", [])
    if isinstance(history, list):
        for item in reversed(history):
            if not isinstance(item, dict) or str(item.get("order_id") or "") != target_id:
                continue
            price = _safe_float(item.get("stop_price"), 0.0)
            if price > 0 and math.isfinite(price):
                return price
    return None


def _calc_stop_slippage_pct(direction: str, fill_price: float | None, stop_price: float | None) -> float | None:
    """Measure only adverse execution slippage versus the trigger of the filled stop order."""
    side = str(direction or "").strip().upper()
    fill = _safe_float(fill_price, 0.0)
    stop = _safe_float(stop_price, 0.0)
    if side not in {"LONG", "SHORT"} or not math.isfinite(fill) or not math.isfinite(stop) or fill <= 0 or stop <= 0:
        return None
    if side == "LONG":
        return max(0.0, (stop - fill) / stop * 100.0)
    return max(0.0, (fill - stop) / stop * 100.0)


def _cleanup_closed_trade_protection(trade: dict) -> tuple[bool, str]:
    """Cancel and verify all remaining protection for a position that has closed.

    Stale conditional orders are dangerous in hedge mode because they can affect a
    later position in the same symbol/direction. Cleanup therefore must be verified
    before the lifecycle is finalized.
    """
    symbol = str(trade.get("symbol", ""))
    direction = str(trade.get("direction", "")).upper()
    if not symbol or direction not in {"LONG", "SHORT"}:
        return False, "invalid trade identity for protection cleanup"
    # Cleanup-only rollback records may survive a restart after the immediate
    # close/cleanup sequence was ambiguous. Require two distinct NOT_FOUND reads
    # here too; a single transient empty response must not authorize cancellation
    # of protection belonging to a still-live position.
    for read_index in range(2):
        try:
            live = get_position_directional(symbol, direction)
        except Exception as exc:
            return False, f"position state read failed: {exc}"
        live_status = str(live.get("status", "")).lower()
        if live_status == "found" and abs(_safe_float(live.get("positionAmt"), 0.0)) > 0:
            return False, "refusing cleanup because a live position exists for the same symbol/direction"
        if live_status != "not_found":
            return False, f"refusing cleanup because position state is not proven closed: {live_status or 'unknown'}"
        if read_index == 0:
            time.sleep(0.15)
    try:
        prot = get_open_protection_directional(symbol, direction)
    except Exception as exc:
        return False, f"open protection read failed: {exc}"
    if prot.get("status") != "ok":
        return False, str(prot.get("error", "open protection read failed"))

    orders = list(prot.get("sl_orders", [])) + list(prot.get("tp_orders", []))
    if not orders:
        return True, "no_open_protection"

    failures = []
    for order in orders:
        order_id = str(order.get("orderId", ""))
        if not order_id:
            failures.append("open protection missing orderId")
            continue
        try:
            ok, note = _cancel_protection_order_verified(symbol, direction, order, max_attempts=3)
        except Exception as exc:
            ok, note = False, str(exc)
        if not ok:
            failures.append(f"{order_id}: {note}")

    try:
        final = get_open_protection_directional(symbol, direction)
    except Exception as exc:
        return False, f"final protection verification failed: {exc}"
    if final.get("status") != "ok":
        return False, str(final.get("error", "final protection verification failed"))
    remaining = list(final.get("sl_orders", [])) + list(final.get("tp_orders", []))
    if remaining:
        failures.append("protection still visible after cancellation")
    return (not failures), ("; ".join(failures) if failures else "cleanup_verified")


def _cashflow_symbol(symbol: Any) -> str:
    value = str(symbol or "").strip().upper().replace("_", "-").replace("/", "-")
    if not value:
        return ""
    value = value.replace("--", "-")
    if "-" not in value:
        if value.endswith("USDT") and len(value) > 4:
            value = value[:-4] + "-USDT"
        else:
            value += "-USDT"
    return value


def _event_timestamp_ms(opened: dict | None) -> int:
    if not isinstance(opened, dict):
        return 0
    execution = opened.get("execution") if isinstance(opened.get("execution"), dict) else {}
    result = opened.get("result") if isinstance(opened.get("result"), dict) else {}
    candidates = (
        execution.get("fill_ts_ms"), result.get("fill_ts_ms"),
        opened.get("entry_ts"), opened.get("ts"),
        (opened.get("signal") or {}).get("detected_at_ts") if isinstance(opened.get("signal"), dict) else None,
    )
    for candidate in candidates:
        timestamp = int(_safe_float(candidate, 0.0))
        if timestamp > 0:
            # Normalize Unix seconds while preserving normal millisecond epochs.
            return timestamp * 1000 if timestamp < 100_000_000_000 else timestamp
    return 0


def _overlapping_same_symbol_trade(
    event_id: str,
    symbol: str,
    start_ts: int,
    end_ts: int,
    opened_by_event: dict[str, dict],
    close_ts_by_event: dict[str, int],
) -> str | None:
    wanted_symbol = _cashflow_symbol(symbol)
    for other_event, other_open in opened_by_event.items():
        if other_event == event_id or _cashflow_symbol(other_open.get("symbol")) != wanted_symbol:
            continue
        other_start = _event_timestamp_ms(other_open)
        if other_start <= 0:
            continue
        other_end = int(close_ts_by_event.get(other_event, 0) or 0)
        # An unmatched open is treated as still open; this is intentionally
        # conservative because income rows do not carry an unambiguous side.
        if other_end <= 0:
            other_end = 2**63 - 1
        if other_start <= end_ts and other_end >= start_ts:
            return other_event
    return None


def _cashflow_order_ids(opened: dict, close_record: dict) -> tuple[str | None, set[str]]:
    """Extract exchange order IDs recorded for one local position lifecycle."""
    execution = opened.get("execution") if isinstance(opened.get("execution"), dict) else {}
    open_result = opened.get("result") if isinstance(opened.get("result"), dict) else {}
    prior_recovery = close_record.get("cashflow_recovery") if isinstance(close_record.get("cashflow_recovery"), dict) else {}
    entry_id = str(
        execution.get("order_id") or execution.get("orderId")
        or open_result.get("order_id") or open_result.get("orderId")
        or prior_recovery.get("resolved_entry_order_id") or ""
    ).strip() or None
    order_ids: set[str] = set()
    if entry_id:
        order_ids.add(entry_id)

    def add_order(obj: Any) -> None:
        if isinstance(obj, dict):
            for key in ("order_id", "orderId", "orderID", "sl_order_id", "tp_order_id"):
                value = obj.get(key)
                if value not in (None, ""):
                    order_ids.add(str(value).strip())
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    add_order(value)
        elif isinstance(obj, list):
            for value in obj:
                add_order(value)

    # Only inspect execution/protection structures, not arbitrary signal metadata.
    add_order(open_result.get("protection"))
    add_order(opened.get("protection"))
    add_order(opened.get("sl_order"))
    add_order(opened.get("tp_orders"))
    for key in ("order_id", "orderId", "orderID", "close_order_id", "exit_order_id", "manual_exit_order_id", "filled_sl_order_id"):
        value = close_record.get(key)
        if value not in (None, ""):
            order_ids.add(str(value).strip())
    for key in ("close_order_ids", "exit_order_ids", "tp_order_ids", "filled_order_ids"):
        value = close_record.get(key)
        if isinstance(value, list):
            order_ids.update(str(item).strip() for item in value if item not in (None, ""))
    order_ids.discard("")
    return entry_id, order_ids


def _cashflow_entry_client_order_id(opened: dict) -> str | None:
    """Read deterministic clientOrderId for delayed resolution of a missing ACK id."""
    result = opened.get("result") if isinstance(opened.get("result"), dict) else {}
    ack = result.get("open_result") if isinstance(result.get("open_result"), dict) else {}
    value = ack.get("client_order_id") or ack.get("clientOrderId") or ack.get("clientOrderID")
    return str(value).strip() if value not in (None, "") else None


def _load_exchange_fill_evidence(
    symbol: str, direction: str, query_start: int, query_end: int,
    entry_order_id: str | None, order_ids: set[str],
) -> tuple[list[dict] | None, str | None]:
    """Load complete actual fills for known entry/protection/exit order IDs."""
    if not entry_order_id:
        return None, "abandon:journal_missing_entry_order_id"
    exit_order_ids = order_ids - {entry_order_id}
    if not exit_order_ids:
        return None, "abandon:journal_missing_exit_order_ids"
    if query_end - query_start > 7 * 24 * 60 * 60 * 1000:
        return None, "exchange_fill_query_window_exceeds_seven_days"

    fills_by_id: dict[str, dict] = {}
    order_fill_counts: dict[str, int] = {}
    for order_id in sorted(order_ids):
        try:
            response = get_fill_history(
                symbol, query_start, query_end, order_id=order_id,
                page_index=1, page_size=1000, currency="USDT", timeout_sec=5.0,
            )
        except Exception as exc:
            return None, f"fill_history_exception_for_order:{order_id}:{exc}"
        if not isinstance(response, dict) or str(response.get("status", "")).lower() != "ok":
            error = response.get("error", "fill_history_invalid_response") if isinstance(response, dict) else "fill_history_invalid_response"
            return None, f"fill_history_failed_for_order:{order_id}:{error}"
        if response.get("has_more") is True:
            return None, f"fill_history_truncated_for_order:{order_id}"
        rows = response.get("fills")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            return None, f"fill_history_invalid_rows_for_order:{order_id}"
        for fill in rows:
            fill_order_id = str(fill.get("order_id") or "").strip()
            if fill_order_id != order_id:
                return None, f"fill_history_order_id_mismatch:{order_id}:{fill_order_id or 'MISSING'}"
            if _cashflow_symbol(fill.get("symbol")) != _cashflow_symbol(symbol):
                return None, f"fill_history_symbol_mismatch_for_order:{order_id}"
            fill_ts = int(_safe_float(fill.get("time"), 0.0))
            qty = abs(_safe_float(fill.get("qty"), 0.0))
            price = _safe_float(fill.get("price"), 0.0)
            fee = _safe_float(fill.get("fee"), float("nan"))
            fee_asset = str(fill.get("fee_asset") or "").upper()
            trade_id = str(fill.get("trade_id") or "").strip()
            if not trade_id or fill_ts <= 0 or not query_start <= fill_ts <= query_end:
                return None, f"fill_history_missing_identity_or_time_for_order:{order_id}"
            if qty <= 0 or price <= 0 or not math.isfinite(qty * price):
                return None, f"fill_history_missing_actual_qty_price_for_order:{order_id}"
            if not math.isfinite(fee) or fee_asset != "USDT":
                return None, f"fill_history_fee_unknown_or_non_usdt_for_order:{order_id}"
            pos_side = str(fill.get("position_side") or "").upper()
            if pos_side and pos_side not in {str(direction).upper(), "BOTH"}:
                return None, f"fill_history_position_side_mismatch_for_order:{order_id}"
            realized_raw = fill.get("realized_pnl")
            if realized_raw in (None, ""):
                if order_id == entry_order_id:
                    fill = dict(fill, realized_pnl=0.0)
                else:
                    return None, f"exit_fill_realized_pnl_unknown_for_order:{order_id}"
            realized = _safe_float(fill.get("realized_pnl"), float("nan"))
            if not math.isfinite(realized):
                return None, f"fill_history_realized_pnl_non_finite_for_order:{order_id}"
            normalized = dict(fill)
            normalized.update(order_id=order_id, trade_id=trade_id, time=fill_ts, qty=qty, price=price, fee=fee, fee_asset=fee_asset, realized_pnl=realized)
            previous = fills_by_id.get(trade_id)
            if previous is not None and any(previous.get(k) != normalized.get(k) for k in ("order_id", "qty", "price", "fee", "realized_pnl")):
                return None, f"duplicate_fill_id_conflict:{trade_id}"
            fills_by_id[trade_id] = normalized
            order_fill_counts[order_id] = order_fill_counts.get(order_id, 0) + 1

    if order_fill_counts.get(entry_order_id, 0) <= 0:
        return None, "entry_order_has_no_confirmed_fills"
    if not any(order_fill_counts.get(order_id, 0) > 0 for order_id in exit_order_ids):
        return None, "no_confirmed_exit_fills_for_known_exit_orders"
    return list(fills_by_id.values()), None


def _recover_trade_cashflows(*, now_ms: int | None = None, max_records: int = 6) -> int:
    """Reconcile cashflows only when ledger trade IDs link to exact exchange fills.

    Symbol/time-window totals alone are not position attribution. The recovery
    path requires known entry/exit order IDs, complete actual fill history, direct
    income.tradeId ↔ fill.trade_id links, and PnL/fee checks within $0.02. If any
    identity or coverage is missing, net PnL remains unknown.
    """
    if not TRADES_PATH.exists():
        return 0
    now_ms = int(now_ms or time.time() * 1000)
    try:
        raw_lines = TRADES_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    except OSError as exc:
        log.warning("[TRACKER_CASHFLOW] Cannot read trade journal: %s", exc)
        return 0

    parsed: list[dict | None] = []
    opened_by_event: dict[str, dict] = {}
    close_ts_by_event: dict[str, int] = {}
    close_indices_by_event: dict[str, list[int]] = {}
    for index, line in enumerate(raw_lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            parsed.append(None)
            continue
        parsed.append(record if isinstance(record, dict) else None)
        if not isinstance(record, dict):
            continue
        event_id = str(record.get("event_id") or "")
        if not event_id:
            continue
        if record.get("record_type") == "TRADE_OPEN":
            opened_by_event.setdefault(event_id, record)
        elif record.get("record_type") == "TRADE_CLOSE":
            ts = int(_safe_float(record.get("closed_ts") or record.get("ts"), 0.0))
            if ts and ts < 100_000_000_000:
                ts *= 1000
            close_ts_by_event[event_id] = max(close_ts_by_event.get(event_id, 0), ts)
            close_indices_by_event.setdefault(event_id, []).append(index)

    replacements: dict[int, str] = {}
    attempts_this_cycle = 0
    recovered = 0
    retention_ms = 89 * 24 * 60 * 60 * 1000
    for event_id, indices in close_indices_by_event.items():
        if attempts_this_cycle >= max(0, int(max_records)):
            break
        index = indices[-1]
        record = parsed[index]
        if not isinstance(record, dict):
            continue
        current_status = str(record.get("pnl_costs_status") or "")
        recovery = record.get("cashflow_recovery") if isinstance(record.get("cashflow_recovery"), dict) else {}
        if current_status == "BINGX_ORDER_LINKED_FILLS_AND_INCOME_CONFIRMED" and record.get("net_pnl_usdt") is not None:
            continue

        # Invalidate V10.4.2's symbol/time-window-only totals. They are real account
        # cashflows but cannot be asserted as net PnL for this particular position.
        if current_status in {"BINGX_ACCOUNT_INCOME_CONFIRMED", "BINGX_SYMBOL_WINDOW_CASHFLOWS_ATTRIBUTED"}:
            record = dict(record)
            # Legacy symbol/window aggregates are account facts, not trade facts.
            # Preserve the historical number under an explicitly unattributed name;
            # never let it remain in a per-position realized/net PnL field.
            legacy_gross = record.get("realized_pnl_usdt")
            if legacy_gross is not None:
                record["unattributed_symbol_window_realized_pnl_usdt"] = legacy_gross
                record["unattributed_symbol_window_realized_pnl_source"] = "LEGACY_ACCOUNT_SYMBOL_WINDOW_AGGREGATE_NOT_PER_POSITION"
            record["realized_pnl_usdt"] = None
            record["realized_pnl_usdt_source"] = "UNATTRIBUTED_SYMBOL_WINDOW_AGGREGATE"
            for key in ("fees_usdt", "funding_usdt", "net_pnl_usdt", "other_cashflows_usdt"):
                record[key] = None
            record["pnl_costs_status"] = "UNVERIFIED_POSITION_ATTRIBUTION"
            record["net_pnl_attribution_status"] = "UNKNOWN_DIRECT_FILL_LINK_REQUIRED"
            recovery = {}
            record["cashflow_recovery"] = recovery
            parsed[index] = record
            # Persist the invalidation even if later validation (timestamps, order IDs,
            # overlap checks) prevents a recovery attempt during this cycle.
            replacements[index] = json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False) + "\n"

        if str(recovery.get("status") or "").startswith("ABANDONED") or recovery.get("status") == "RECOVERED":
            continue
        if int(_safe_float(recovery.get("next_retry_ts"), 0.0)) > now_ms:
            continue
        opened = opened_by_event.get(event_id)
        open_ts = _event_timestamp_ms(opened)
        close_ts = int(_safe_float(record.get("closed_ts") or record.get("ts"), 0.0))
        if close_ts and close_ts < 100_000_000_000:
            close_ts *= 1000
        symbol = str(record.get("symbol") or (opened or {}).get("symbol") or "")
        direction = str(record.get("direction") or (opened or {}).get("direction") or "").upper()
        if open_ts <= 0 or close_ts <= open_ts or not symbol or direction not in {"LONG", "SHORT"}:
            continue
        if now_ms - close_ts > retention_ms:
            updated = dict(record)
            updated["cashflow_recovery"] = {
                "status": "ABANDONED_OUTSIDE_EXCHANGE_RETENTION",
                "attempts": int(_safe_float(recovery.get("attempts"), 0.0)),
                "last_error": "exchange_income_history_retention_exceeded",
            }
            replacements[index] = json.dumps(json_safe(updated), ensure_ascii=False, allow_nan=False) + "\n"
            continue
        query_start, query_end = max(1, open_ts - 60_000), close_ts + 60_000
        overlapping = _overlapping_same_symbol_trade(
            event_id, symbol, query_start, query_end, opened_by_event, close_ts_by_event,
        )
        if overlapping:
            updated = dict(record)
            updated["cashflow_recovery"] = {
                "status": "ABANDONED_AMBIGUOUS_SYMBOL_OVERLAP",
                "attempts": int(_safe_float(recovery.get("attempts"), 0.0)),
                "overlapping_event_id": overlapping,
                "last_error": "same_symbol_journal_positions_overlap",
            }
            replacements[index] = json.dumps(json_safe(updated), ensure_ascii=False, allow_nan=False) + "\n"
            continue

        entry_order_id, order_ids = _cashflow_order_ids(opened or {}, record)
        entry_client_order_id = _cashflow_entry_client_order_id(opened or {})
        resolved_entry_order_id = entry_order_id
        if not entry_order_id and entry_client_order_id:
            # A code=0 market response can be missing orderId even while the
            # position is real. Retry a read-only clientOrderId lookup during
            # recovery; never substitute symbol/time-window fills for this link.
            try:
                resolved = get_order_by_client_order_id(symbol, entry_client_order_id)
            except Exception as exc:
                resolved = {"status": "error", "error": str(exc)}
            candidate_id = str(resolved.get("order_id") or "").strip() if isinstance(resolved, dict) else ""
            if isinstance(resolved, dict) and str(resolved.get("status", "")).lower() == "ok" and candidate_id:
                entry_order_id = candidate_id
                resolved_entry_order_id = candidate_id
                order_ids.add(candidate_id)
            else:
                attempts_this_cycle += 1
                updated = dict(record)
                attempts = int(_safe_float(recovery.get("attempts"), 0.0)) + 1
                delay_ms = min(6 * 60 * 60 * 1000, 30 * 60 * 1000 * (2 ** min(attempts - 1, 4)))
                updated["fees_usdt"] = None
                updated["funding_usdt"] = None
                updated["net_pnl_usdt"] = None
                updated["pnl_costs_status"] = "UNVERIFIED_POSITION_ATTRIBUTION"
                updated["net_pnl_attribution_status"] = "UNKNOWN_DIRECT_FILL_LINK_REQUIRED"
                updated["cashflow_recovery"] = {
                    "status": "PENDING_ENTRY_ORDER_ID_RESOLUTION" if attempts < 40 else "ABANDONED_MAX_ATTEMPTS",
                    "attempts": attempts, "last_attempt_ts": now_ms,
                    "next_retry_ts": now_ms + delay_ms if attempts < 40 else None,
                    "client_order_id": entry_client_order_id,
                    "last_error": str((resolved or {}).get("error") or "clientOrderId lookup did not return a direct order id")[:500],
                }
                replacements[index] = json.dumps(json_safe(updated), ensure_ascii=False, allow_nan=False) + "\n"
                log.warning("[TRACKER_CASHFLOW] Trade=%s entry order id not yet resolvable for clientOrderId=%s; retrying without guessing", event_id, entry_client_order_id)
                continue

        attempts_this_cycle += 1
        fills, fill_error = _load_exchange_fill_evidence(
            symbol, direction, query_start, query_end, entry_order_id, order_ids,
        )
        if fill_error:
            updated = dict(record)
            attempts = int(_safe_float(recovery.get("attempts"), 0.0)) + 1
            terminal = fill_error.startswith("abandon:") or attempts >= 40 or "window_exceeds_seven_days" in fill_error
            delay_ms = min(6 * 60 * 60 * 1000, 30 * 60 * 1000 * (2 ** min(attempts - 1, 4)))
            updated["fees_usdt"] = None
            updated["funding_usdt"] = None
            updated["net_pnl_usdt"] = None
            if updated.get("realized_pnl_usdt") is not None and str(updated.get("realized_pnl_usdt_source") or "").startswith("TRACKER_DERIVED"):
                updated["estimated_gross_pnl_usdt"] = updated.get("realized_pnl_usdt")
                updated["estimated_gross_pnl_usdt_source"] = "TRACKER_ESTIMATE_NOT_EXCHANGE_FILL_RECONCILED"
                updated["realized_pnl_usdt"] = None
                updated["realized_pnl_usdt_source"] = "UNKNOWN_DIRECT_FILL_LINK_REQUIRED"
            updated["pnl_costs_status"] = "UNVERIFIED_POSITION_ATTRIBUTION"
            updated["net_pnl_attribution_status"] = "UNKNOWN_DIRECT_FILL_LINK_REQUIRED"
            updated["cashflow_recovery"] = {
                "status": "ABANDONED_NO_DIRECT_ORDER_IDS" if fill_error.startswith("abandon:") else ("ABANDONED_MAX_ATTEMPTS" if terminal else "PENDING"),
                "attempts": attempts, "last_attempt_ts": now_ms,
                "next_retry_ts": None if terminal else now_ms + delay_ms,
                "resolved_entry_order_id": resolved_entry_order_id,
                "client_order_id": entry_client_order_id,
                "last_error": fill_error,
            }
            replacements[index] = json.dumps(json_safe(updated), ensure_ascii=False, allow_nan=False) + "\n"
            log.warning("[TRACKER_CASHFLOW] Trade=%s remains unattributed: %s", event_id, fill_error)
            continue

        try:
            response = get_income_history(symbol, query_start, query_end, limit=1000, timeout_sec=5.0)
        except Exception as exc:
            response = {"status": "error", "error": f"income_history_exception:{exc}", "complete": False, "records": []}
        records = response.get("records") if isinstance(response, dict) else None
        error = None
        if not isinstance(response, dict) or str(response.get("status", "")).lower() != "ok" or response.get("complete") is not True:
            error = str(response.get("error", "income_history_incomplete")) if isinstance(response, dict) else "income_history_invalid_response"
            records = []
        elif not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
            error, records = "income_history_invalid_records", []

        fill_ids = {str(fill.get("trade_id")) for fill in fills}
        order_ids = set(order_ids)
        relevant: list[dict] = []
        recognized_types = {
            "REALIZED_PNL", "TRADING_FEE", "FUNDING_FEE",
            "INSURANCE_CLEAR", "ADL", "SYSTEM_DEDUCTION", "GTD_PRICE",
        }
        if error is None:
            for item in records:
                income_type = str(item.get("income_type") or "").upper()
                # Account transfers are not position PnL and may have no symbol.
                if income_type == "TRANSFER":
                    continue
                row_symbol = _cashflow_symbol(item.get("symbol")) if item.get("symbol") else ""
                row_time = int(_safe_float(item.get("time"), 0.0))
                if row_symbol and row_symbol != _cashflow_symbol(symbol):
                    continue
                if not query_start <= row_time <= query_end:
                    continue
                if income_type not in recognized_types:
                    error = f"income_history_unrecognized_or_unrelated_type:{income_type or 'MISSING'}"
                    break
                if not row_symbol:
                    error = "income_history_missing_symbol_for_relevant_cashflow"
                    break
                if str(item.get("asset") or "").upper() != "USDT":
                    error = "income_history_non_usdt_asset"
                    break
                amount = _safe_float(item.get("amount"), float("nan"))
                if not math.isfinite(amount):
                    error = "income_history_non_finite_amount"
                    break
                trade_ref = str(item.get("trade_id") or "").strip()
                order_ref = str(item.get("order_id") or "").strip()
                if trade_ref not in fill_ids and order_ref not in order_ids:
                    error = f"income_history_cashflow_not_linked_to_position_fills:{item.get('record_id') or 'NO_ID'}"
                    break
                relevant.append(item)

        exit_order_ids = order_ids - ({entry_order_id} if entry_order_id else set())
        exit_fill_ids = {str(fill.get("trade_id")) for fill in fills if str(fill.get("order_id")) in exit_order_ids}
        realized_rows = [item for item in relevant if str(item.get("income_type") or "").upper() == "REALIZED_PNL"]
        fee_rows = [item for item in relevant if str(item.get("income_type") or "").upper() == "TRADING_FEE"]
        if error is None and not realized_rows:
            error = "income_history_missing_realized_pnl_rows"
        if error is None and not fee_rows:
            error = "income_history_missing_trading_fee_rows"
        realized_income_ids = {str(item.get("trade_id") or "") for item in realized_rows}
        fee_income_ids = {str(item.get("trade_id") or "") for item in fee_rows}
        if error is None and not exit_fill_ids.issubset(realized_income_ids):
            error = "income_history_realized_pnl_fill_id_coverage_incomplete"
        if error is None and not fill_ids.issubset(fee_income_ids):
            error = "income_history_fee_fill_id_coverage_incomplete"

        updated = dict(record)
        attempts = int(_safe_float(recovery.get("attempts"), 0.0)) + 1
        if error is None:
            fill_gross = sum(_safe_float(fill.get("realized_pnl"), 0.0) for fill in fills if str(fill.get("order_id")) in exit_order_ids)
            fill_fees = sum(_safe_float(fill.get("fee"), 0.0) for fill in fills)
            income_gross = sum(_safe_float(item.get("amount"), 0.0) for item in realized_rows)
            income_fees = sum(_safe_float(item.get("amount"), 0.0) for item in fee_rows)
            if abs(income_gross - fill_gross) > 0.02:
                error = f"income_vs_fill_realized_pnl_mismatch:{income_gross:.8g}!={fill_gross:.8g}"
            elif abs(income_fees - fill_fees) > 0.02:
                error = f"income_vs_fill_fees_mismatch:{income_fees:.8g}!={fill_fees:.8g}"

        if error is None:
            funding_rows = [item for item in relevant if str(item.get("income_type") or "").upper() == "FUNDING_FEE"]
            funding = sum(_safe_float(item.get("amount"), 0.0) for item in funding_rows)
            other_rows = [item for item in relevant if str(item.get("income_type") or "").upper() in {"INSURANCE_CLEAR", "ADL", "SYSTEM_DEDUCTION", "GTD_PRICE"}]
            other_cashflows = sum(_safe_float(item.get("amount"), 0.0) for item in other_rows)
            net = fill_gross + fill_fees + funding + other_cashflows
            updated.update({
                "realized_pnl_usdt": fill_gross,
                "realized_pnl_usdt_source": "BINGX_ORDER_LINKED_FILL_HISTORY",
                "fees_usdt": fill_fees,
                "fees_usdt_source": "BINGX_ORDER_LINKED_FILL_HISTORY_SIGNED_FEES",
                "funding_usdt": funding,
                "funding_usdt_source": "BINGX_ACCOUNT_INCOME_TRADE_ID_LINKED" if funding_rows else "BINGX_INCOME_COMPLETE_NO_LINKED_FUNDING_ROWS",
                "other_cashflows_usdt": other_cashflows,
                "other_cashflows_usdt_source": "BINGX_ACCOUNT_INCOME_TRADE_ID_LINKED_OTHER_FLOWS",
                "net_pnl_usdt": net,
                "net_pnl_usdt_source": "BINGX_ORDER_LINKED_FILLS_PLUS_TRADE_ID_LINKED_INCOME",
                "net_pnl_attribution_status": "EXACT_EXCHANGE_ORDER_FILL_IDS_MATCHED_TO_ACCOUNT_INCOME_TRADE_IDS",
                "cashflow_attribution_scope": "DIRECT_ORDER_FILL_ID_AND_INCOME_TRADE_ID_LINK",
                "pnl_costs_status": "BINGX_ORDER_LINKED_FILLS_AND_INCOME_CONFIRMED",
                "cashflow_recovery": {
                    "status": "RECOVERED", "attempts": attempts, "recovered_ts": now_ms,
                    "resolved_entry_order_id": resolved_entry_order_id,
                    "source": "BINGX_ORDER_LINKED_FILL_HISTORY_AND_USER_INCOME",
                    "orders": sorted(order_ids), "fill_trade_ids": sorted(fill_ids),
                    "rows": len(relevant), "income_types": sorted({str(item.get("income_type") or "").upper() for item in relevant}),
                    "transaction_ids": sorted({str(item.get("record_id")) for item in relevant if item.get("record_id")}),
                    "cashflow_window_ms": [query_start, query_end],
                    "gross_crosscheck_delta_usdt": income_gross - fill_gross,
                    "fee_crosscheck_delta_usdt": income_fees - fill_fees,
                },
            })
            replacements[index] = json.dumps(json_safe(updated), ensure_ascii=False, allow_nan=False) + "\n"
            recovered += 1
            log.info("[TRACKER_CASHFLOW] Exact exchange-linked net PnL event=%s gross=%.8g fees=%.8g funding=%.8g other=%.8g net=%.8g fills=%d income_rows=%d", event_id, fill_gross, fill_fees, funding, other_cashflows, net, len(fills), len(relevant))
        else:
            delay_ms = min(6 * 60 * 60 * 1000, 30 * 60 * 1000 * (2 ** min(attempts - 1, 4)))
            terminal = attempts >= 40 or error.startswith("abandon:") or "window_exceeds_seven_days" in error
            updated["fees_usdt"] = None
            updated["funding_usdt"] = None
            updated["net_pnl_usdt"] = None
            if updated.get("realized_pnl_usdt") is not None and str(updated.get("realized_pnl_usdt_source") or "").startswith("TRACKER_DERIVED"):
                updated["estimated_gross_pnl_usdt"] = updated.get("realized_pnl_usdt")
                updated["estimated_gross_pnl_usdt_source"] = "TRACKER_ESTIMATE_NOT_EXCHANGE_FILL_RECONCILED"
                updated["realized_pnl_usdt"] = None
                updated["realized_pnl_usdt_source"] = "UNKNOWN_DIRECT_FILL_LINK_REQUIRED"
            updated["pnl_costs_status"] = "UNVERIFIED_POSITION_ATTRIBUTION"
            updated["net_pnl_attribution_status"] = "UNKNOWN_DIRECT_FILL_LINK_REQUIRED"
            updated["cashflow_recovery"] = {
                "status": "ABANDONED_NO_DIRECT_ORDER_IDS" if error.startswith("abandon:") else ("ABANDONED_MAX_ATTEMPTS" if terminal else "PENDING"),
                "attempts": attempts, "last_attempt_ts": now_ms,
                "resolved_entry_order_id": resolved_entry_order_id,
                "client_order_id": entry_client_order_id,
                "next_retry_ts": None if terminal else now_ms + delay_ms,
                "last_error": error,
            }
            replacements[index] = json.dumps(json_safe(updated), ensure_ascii=False, allow_nan=False) + "\n"
            log.warning("[TRACKER_CASHFLOW] Trade=%s cashflow attribution pending/unknown: %s", event_id, error)

    if replacements:
        temp_path = TRADES_PATH.with_name(TRADES_PATH.name + ".cashflow.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as stream:
                for index, raw_line in enumerate(raw_lines):
                    stream.write(replacements.get(index, raw_line if raw_line.endswith("\n") else raw_line + "\n"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, TRADES_PATH)
        except OSError as exc:
            log.error("[TRACKER_CASHFLOW] Failed to persist ledger reconciliation: %s", exc)
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
    return recovered

def update_active_trades() -> None:
    # Reconcile historical DATA_ERROR closes even when no positions remain open.
    # Only fill-history-confirmed prices and quantities are accepted.
    try:
        _recover_data_error_trade_closes()
    except Exception as exc:
        log.error("[TRACKER_RECOVERY] DATA_ERROR fill-history recovery failed safely: %s", exc)
    try:
        _recover_trade_cashflows(max_records=6)
    except Exception as exc:
        log.error("[TRACKER_CASHFLOW] Account-income recovery failed safely: %s", exc)
    _retry_pending_notifications()
    trades = _load_active_trades()
    if not trades:
        return

    now_ms = int(time.time() * 1000)
    updated_trades: dict[str, dict] = {}

    for event_id, trade in trades.items():
        if trade.get("close_notification_pending"):
            # The close journal is already durable; do not re-run position/fill/
            # candle queries just because queue persistence for Telegram failed.
            nid = _notification_id(str(event_id), "TRADE_CLOSE")
            notifications = _load_notifications()
            queued = notifications.get(nid)
            notification_is_durable = (
                isinstance(queued, dict)
                and str(queued.get("event_id", "")) == str(event_id)
                and str(queued.get("kind", "")).upper() == "TRADE_CLOSE"
            )
            if not notification_is_durable:
                text = str(trade.get("close_notification_text") or "").strip()
                if not text:
                    rebuilt = _rebuild_close_notification_text(str(event_id), trade)
                    text = rebuilt or ""
                if text:
                    trade["close_notification_text"] = text
                    try:
                        _queue_notification(
                            nid,
                            event_id=str(event_id),
                            kind="TRADE_CLOSE",
                            symbol=str(trade.get("symbol", "")),
                            direction=str(trade.get("direction", "")),
                            text=text,
                        )
                    except Exception as exc:
                        log.warning("[TELEGRAM] Could not restore close notification queue for %s: %s", event_id, exc)
                    notifications = _load_notifications()
                    queued = notifications.get(nid)
                    notification_is_durable = (
                        isinstance(queued, dict)
                        and str(queued.get("event_id", "")) == str(event_id)
                        and str(queued.get("kind", "")).upper() == "TRADE_CLOSE"
                    )

            # Protection cleanup is independent of Telegram delivery. If it has
            # not already been verified, retry it here while avoiding normal trade
            # lifecycle API calls (position polls, MFE candles and order lookups).
            if not trade.get("close_cleanup_verified"):
                cleanup_ok, cleanup_note = _cleanup_closed_trade_protection(trade)
                if cleanup_ok:
                    trade["close_cleanup_pending"] = False
                    trade["close_cleanup_verified"] = True
                    trade.pop("close_cleanup_last_error", None)
                else:
                    trade["close_cleanup_pending"] = True
                    trade["close_cleanup_last_error"] = cleanup_note
                    trade["close_cleanup_verified"] = False
                    log.error("[TRACKER] Pending close cleanup for %s: %s", event_id, cleanup_note)

            if notification_is_durable:
                trade["close_notification_pending"] = False
                trade.pop("close_notification_text", None)
            else:
                trade["close_notification_pending"] = True
                trade["closed"] = False
                updated_trades[event_id] = trade
                continue

            if trade.get("close_cleanup_verified") and not trade.get("close_cleanup_pending"):
                trade["closed"] = True
                continue

            trade["closed"] = False
            updated_trades[event_id] = trade
            continue

        if trade.get("close_cleanup_pending"):
            cleanup_ok, cleanup_note = _cleanup_closed_trade_protection(trade)
            if cleanup_ok:
                trade["close_cleanup_pending"] = False
                trade["closed"] = True
                trade.pop("close_cleanup_last_error", None)
                log.info("[TRACKER] Closed trade %s protection cleanup verified; removing from active state.", event_id)
                # Active state must not retain a closed row for an extra cycle.
                continue
            trade["closed"] = False
            trade["close_cleanup_pending"] = True
            trade["close_cleanup_last_error"] = cleanup_note
            log.error("[TRACKER] Closed trade %s still has protection cleanup pending: %s", event_id, cleanup_note)
            updated_trades[event_id] = trade
            continue

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
                detail = str(pos.get("error") or pos_status or "unknown")
                trade["position_state_alert_pending"] = True
                trade["position_state_last_error"] = detail
                trade["last_observation_ts"] = now_ms
                _notify_position_state_unknown(symbol, direction, detail, str(event_id))
                updated_trades[event_id] = trade
                continue

            if pos_status == "not_found":
                # One empty snapshot is not proof that the position was closed.
                # Require an independent follow-up read; any error or a reappearing
                # position defers lifecycle closure until the next tracker cycle.
                try:
                    time.sleep(0.25)
                    confirm = get_position_directional(symbol, direction)
                except Exception as exc:
                    trade["position_gone_confirmation"] = {
                        "status": "PENDING", "observed_ts": now_ms,
                        "reason": f"second_position_read_exception:{exc}",
                    }
                    trade["last_observation_ts"] = now_ms
                    trade["position_state_alert_pending"] = True
                    trade["position_state_last_error"] = trade["position_gone_confirmation"]["reason"]
                    _notify_position_state_unknown(symbol, direction, trade["position_state_last_error"], str(event_id))
                    updated_trades[event_id] = trade
                    log.warning("[TRACKER] Deferring close for %s: second position read failed: %s", event_id, exc)
                    continue
                confirm_status = str(confirm.get("status", "")).lower()
                if confirm_status != "not_found":
                    trade["position_gone_confirmation"] = {
                        "status": "PENDING", "observed_ts": now_ms,
                        "reason": f"second_position_read_{confirm_status or 'unknown'}",
                    }
                    trade["last_observation_ts"] = now_ms
                    trade["position_state_alert_pending"] = True
                    trade["position_state_last_error"] = trade["position_gone_confirmation"]["reason"]
                    if confirm_status == "found":
                        trade["current_position_qty"] = abs(_safe_float(confirm.get("positionAmt"), 0.0))
                        trade["last_exchange_avg_price"] = _safe_float(confirm.get("avgPrice"), 0.0) or trade.get("last_exchange_avg_price")
                    _notify_position_state_unknown(symbol, direction, trade["position_state_last_error"], str(event_id))
                    updated_trades[event_id] = trade
                    log.warning("[TRACKER] Deferring close for %s: position disappearance not confirmed (%s).", event_id, confirm_status or "unknown")
                    continue
                trade["position_gone_confirmation"] = {"status": "CONFIRMED", "observed_ts": now_ms, "reads": 2}
                pos = confirm
                pos_status = "not_found"

            pos_amt = abs(_safe_float(pos.get("positionAmt"))) if pos_status == "found" else 0.0
            if pos_status == "found":
                exchange_avg = _safe_float(pos.get("avgPrice"))
                if exchange_avg > 0:
                    trade["last_exchange_avg_price"] = exchange_avg

            cur_price = None
            current_price_source = "UNAVAILABLE"
            estimated_mark_price = None
            live_price = get_live_price(symbol)
            if live_price is not None:
                cur_price = live_price
                current_price_source = "LIVE_TICKER"

            try:
                # Closed klines are historical context only. They may provide a
                # forensic estimate when a position disappears without a known
                # execution fill, but they never become the current execution/
                # reference price used for current PnL or early-loss decisions.
                k1m = fetch_klines(symbol, "1m", limit=60)
                if k1m:
                    last_close = _safe_float(k1m[-1].get("close"), 0.0)
                    if last_close > 0:
                        estimated_mark_price = last_close
                    _update_mfe_mae(trade, k1m)
            except Exception as exc:
                log.warning("[TRACKER] Kline fetch error for %s: %s", symbol, exc)

            trade["current_price_source"] = current_price_source
            current_pnl = (_calc_trade_pnl_pct(entry_price, cur_price, direction)
                           if cur_price is not None else None)
            trade["current_pnl_pct"] = current_pnl
            _update_live_pnl_extrema(trade, current_pnl)
            trade["current_position_qty"] = pos_amt
            trade["last_observation_ts"] = now_ms

            # Experimental early-loss cut: a setup that fails to develop for
            # a sustained period should not be allowed to drift all the way to
            # the full structural stop. The thresholds are configurable and the
            # exit is recorded as a distinct research outcome.
            duration_min = max(0.0, (now_ms - entry_ts) / 60000.0)
            early_loss_reason = None
            if EARLY_LOSS_CUT_ENABLED and current_pnl is not None and pos_status == "found" and pos_amt > 0:
                if duration_min >= EARLY_LOSS_CUT_4H_MIN and current_pnl <= EARLY_LOSS_CUT_4H_PNL:
                    early_loss_reason = "EARLY_LOSS_CUT_4H"
                elif duration_min >= EARLY_LOSS_CUT_2H_MIN and current_pnl <= EARLY_LOSS_CUT_2H_PNL:
                    early_loss_reason = "EARLY_LOSS_CUT_2H"
                if early_loss_reason:
                    cut = {"status": "skipped"}
                    safe_to_close, guard_reason, live_qty = _early_loss_cut_guard(symbol, direction, cur_price, pos_amt)
                    if not safe_to_close:
                        cut["error"] = guard_reason
                        trade["early_loss_cut_last_error"] = guard_reason
                        log.warning("[TRACKER_EARLY_LOSS_CUT] %s %s blocked by live-state guard: %s", symbol, direction, guard_reason)
                    else:
                        log.warning("[TRACKER_EARLY_LOSS_CUT] %s %s pnl=%+.2f%% age=%.1fm reason=%s", symbol, direction, current_pnl, duration_min, early_loss_reason)
                        try:
                            cut = emergency_close_position(symbol, direction, qty=live_qty, reason_token=early_loss_reason)
                        except Exception as exc:
                            cut = {"status": "error", "error": str(exc)}
                    if safe_to_close and str(cut.get("status", "")).lower() == "closed" and cut.get("client_order_id"):
                        trade["manual_exit_reason"] = early_loss_reason
                        # Actual exchange orderId is required for fill reconciliation;
                        # clientOrderId is not interchangeable with exchange fill trade IDs.
                        if cut.get("order_id") not in (None, ""):
                            trade["manual_exit_order_id"] = str(cut.get("order_id"))
                        trade["position_state_alert_pending"] = False
                        manual_execution_price = _safe_float(cut.get("execution_price"), 0.0)
                        if manual_execution_price > 0:
                            trade["manual_exit_price"] = manual_execution_price
                        try:
                            pos = get_position_directional(symbol, direction)
                            pos_status = str(pos.get("status", "")).lower()
                            pos_amt = abs(_safe_float(pos.get("positionAmt"))) if pos_status == "found" else 0.0
                            trade["current_position_qty"] = pos_amt
                        except Exception as exc:
                            log.warning("[TRACKER_EARLY_LOSS_CUT] post-close verification failed for %s: %s", event_id, exc)
                    elif safe_to_close and str(cut.get("status", "")).lower() == "closed":
                        detail = "position_was_already_absent_before_early_loss_cut_order"
                        trade["early_loss_cut_last_error"] = detail
                        trade["position_state_alert_pending"] = True
                        trade["position_state_last_error"] = detail
                        _notify_position_state_unknown(
                            symbol, direction, detail, str(event_id), kind="EARLY_LOSS_CUT_CLOSE_ATTRIBUTION_UNKNOWN",
                        )
                        try:
                            pos = get_position_directional(symbol, direction)
                            pos_status = str(pos.get("status", "")).lower()
                            pos_amt = abs(_safe_float(pos.get("positionAmt"))) if pos_status == "found" else 0.0
                            trade["current_position_qty"] = pos_amt
                        except Exception as exc:
                            log.warning("[TRACKER_EARLY_LOSS_CUT] post-close verification failed for %s: %s", event_id, exc)
                    else:
                        trade["early_loss_cut_last_error"] = cut.get("error") or cut
                        if bool(cut.get("escalation_required")) or str(cut.get("status", "")).lower() in {"unknown", "error"}:
                            detail = str(cut.get("error") or cut.get("status") or "unknown")
                            trade["position_state_alert_pending"] = True
                            trade["position_state_last_error"] = detail
                            _notify_position_state_unknown(
                                symbol, direction, detail, str(event_id), kind="EMERGENCY_CLOSE_UNCERTAIN",
                            )

            # Retry a failed TP -> BE transition after the first realised partial.
            # A single-TP profile activates only when its sole TP order fills;
            # current fallback is TP1, while TP2/TP3 remain legacy-compatible.
            hit_legs = {str(x).lower() for x in trade.get("hit_legs", []) if x}
            is_single_tp = trade.get("tp_mode") == "single_tp"
            should_move_to_be = _be_milestone_reached(hit_legs, is_single_tp)
            if should_move_to_be and not trade.get("be_activated") and rem_qty > 0:
                old_sl = trade.get("sl_order", {}) if isinstance(trade.get("sl_order"), dict) else {}
                old_sl_id = old_sl.get("order_id")
                old_sl_price = _safe_float(old_sl.get("stop_price"), 0.0) or None
                retry = _move_sl_to_break_even(
                    symbol, direction, entry_price, rem_qty, old_sl_id,
                    str(event_id).replace("EVT_", ""), old_sl_price=old_sl_price,
                    prior_order_may_exist=bool(trade.get("be_order_may_exist")),
                    prior_order_id=str(trade.get("be_possible_order_id") or "") or None,
                )
                if retry.get("status") == "created":
                    if old_sl_id:
                        history = trade.setdefault("sl_order_history", [])
                        if all(str(x.get("order_id", "")) != str(old_sl_id) for x in history if isinstance(x, dict)):
                            history.append({"order_id": str(old_sl_id), "stop_price": old_sl_price, "replaced_ts": now_ms})
                    trade["sl_order"] = retry
                    trade["be_activated"] = True
                    trade["be_required"] = False
                    trade["be_last_error"] = None
                    trade["be_activation_ts"] = trade.get("be_activation_ts") or now_ms
                    trade.pop("be_order_may_exist", None)
                    trade.pop("be_possible_order_id", None)
                    trade.pop("be_possible_qty", None)
                    trade.pop("be_possible_stop_price", None)
                else:
                    trade["be_required"] = True
                    trade["be_last_error"] = retry.get("error")
                    if retry.get("new_order_may_exist"):
                        trade["be_order_may_exist"] = True
                        trade["be_possible_order_id"] = str(retry.get("order_id") or trade.get("be_possible_order_id") or "")
                        trade["be_possible_qty"] = rem_qty
                        trade["be_possible_stop_price"] = entry_price

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
                if pnl_tp is None:
                    trade["data_error"] = "INVALID_PRICE_FOR_TP_PNL"
                    trade["exit_reason"] = "DATA_ERROR"
                    log.error(
                        "[TRACKER_DATA_ERROR] %s %s TP leg %s has invalid entry/exit price; realized PnL is not recorded.",
                        symbol, direction, leg,
                    )
                    continue

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

                    is_single_tp = trade.get("tp_mode") == "single_tp"
                    should_move_to_be = _be_milestone_reached(hit_legs, is_single_tp)
                    if should_move_to_be and not trade.get("be_activated") and rem_qty > 0:
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
                            prior_order_may_exist=bool(trade.get("be_order_may_exist")),
                            prior_order_id=str(trade.get("be_possible_order_id") or "") or None,
                        )
                        if new_sl.get("status") == "created":
                            if old_sl_id:
                                history = trade.setdefault("sl_order_history", [])
                                if old_sl_id and all(str(x.get("order_id", "")) != str(old_sl_id) for x in history if isinstance(x, dict)):
                                    history.append({"order_id": str(old_sl_id), "stop_price": old_sl_price, "replaced_ts": now_ms})
                            trade["sl_order"] = new_sl
                            trade["be_activated"] = True
                            trade["be_activation_ts"] = now_ms
                            trade["be_required"] = False
                            trade["be_last_error"] = None
                            trade.pop("be_order_may_exist", None)
                            trade.pop("be_possible_order_id", None)
                            trade.pop("be_possible_qty", None)
                            trade.pop("be_possible_stop_price", None)
                            log.info(
                                "[TRACKER_BE_ACTIVATED] %s (%s) %s milestone reached. Stop-loss moved to Break-Even: %.8g (Risk: 0.00%%)",
                                trade.get("name", symbol), symbol, leg.upper(), entry_price
                            )                             
                        else:
                            trade["be_required"] = True
                            trade["be_last_error"] = new_sl.get("error")
                            if new_sl.get("new_order_may_exist"):
                                trade["be_order_may_exist"] = True
                                trade["be_possible_order_id"] = str(new_sl.get("order_id") or trade.get("be_possible_order_id") or "")
                                trade["be_possible_qty"] = rem_qty
                                trade["be_possible_stop_price"] = entry_price
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

            # Фиксация выхода и закрытие. Never use the last TP execution price
            # as a fallback for an unaccounted residual position: that can turn a
            # real loss after a partial TP into a false profit.
            duration_min = (now_ms - entry_ts) / 60000.0
            exit_price = cur_price
            exit_price_source = "MARKET_ESTIMATE"
            sl_order_id = trade.get("sl_order", {}).get("order_id") if isinstance(trade.get("sl_order"), dict) else None

            sl_exit_price, filled_sl_id = _get_filled_sl_from_trade(symbol, trade)
            if closed_by_tp and rem_qty <= 1e-12:
                exit_reason = "TAKE_PROFIT_FULL"
                exit_price_source = "TP_WEIGHTED_AVERAGE"
            elif sl_exit_price is not None:
                exit_price = sl_exit_price
                exit_price_source = "SL_FILL"
                if trade.get("be_activated") and abs(exit_price - entry_price) / max(entry_price, 1e-12) < 0.003:
                    exit_reason = "BREAK_EVEN"
                else:
                    exit_reason = "STOP_LOSS"
            else:
                exit_reason = "POSITION_CLOSED"

            if trade.get("manual_exit_reason"):
                exit_reason = str(trade.get("manual_exit_reason"))

            manual_exit_price = _safe_float(trade.get("manual_exit_price"), 0.0)
            if manual_exit_price > 0:
                exit_price = manual_exit_price
                exit_price_source = "MANUAL_EXECUTION"

            tp_closed_qty = realized_qty
            residual_qty = rem_qty if position_gone and rem_qty > 0 and init_qty > 0 else 0.0
            unknown_execution = position_gone and not closed_by_tp and sl_exit_price is None and manual_exit_price <= 0
            estimated_exit_price = (cur_price if _safe_float(cur_price, 0.0) > 0 else estimated_mark_price)
            if unknown_execution:
                trade["data_error"] = trade.get("data_error") or "UNKNOWN_EXECUTION_PRICE"
                exit_reason = "DATA_ERROR"
                exit_price = None
                exit_price_source = "UNKNOWN_EXECUTION"
            elif residual_qty > 0:
                residual_pnl = _calc_trade_pnl_pct(entry_price, exit_price, direction)
                if residual_pnl is None:
                    trade["data_error"] = "INVALID_PRICE_FOR_RESIDUAL_PNL"
                    trade["exit_reason"] = "DATA_ERROR"
                else:
                    realized_weighted += residual_qty * residual_pnl
                    realized_qty += residual_qty
                trade["remaining_qty"] = 0.0

            final_pnl = (realized_weighted / init_qty) if (init_qty > 0 and realized_qty > 0) else current_pnl
            if unknown_execution:
                final_pnl = None
            if final_pnl is None:
                trade["data_error"] = trade.get("data_error") or "INVALID_PRICE_FOR_FINAL_PNL"
                exit_reason = "DATA_ERROR"
                trade["exit_reason"] = exit_reason
            if closed_by_tp and realized_qty > 0 and sl_exit_price is None:
                if final_pnl is not None:
                    exit_price = entry_price * (1.0 + final_pnl / 100.0) if direction == "LONG" else entry_price * (1.0 - final_pnl / 100.0)
                    exit_price_source = "TP_WEIGHTED_AVERAGE"
            planned_risk_pct = _derive_planned_risk_pct(trade)
            realized_rr = _calc_realized_rr(final_pnl, planned_risk_pct) if final_pnl is not None else None
            initial_notional_usdt = _safe_float(trade.get("initial_notional_usdt"), entry_price * init_qty)
            realized_pnl_usdt = (realized_weighted / 100.0) * entry_price if entry_price > 0 else None
            planned_risk_usdt = (initial_notional_usdt * planned_risk_pct / 100.0) if planned_risk_pct else None
            stored_sl_price = _safe_float((trade.get("sl_order") or {}).get("stop_price"), 0.0) if isinstance(trade.get("sl_order"), dict) else 0.0
            actual_initial_sl_price = _safe_float(trade.get("planned_invalidation_price"), 0.0)
            # Compare execution with the trigger price of the order that actually
            # filled. The original invalidation is wrong after BE moves the stop.
            filled_sl_stop_price = _stop_price_for_order(trade, filled_sl_id)
            if sl_exit_price is None:
                slippage_reference_price = None
                slippage_reference_source = "NOT_APPLICABLE_NO_SL_FILL"
            elif filled_sl_stop_price is not None:
                slippage_reference_price = filled_sl_stop_price
                slippage_reference_source = "FILLED_SL_ORDER_TRIGGER"
            elif stored_sl_price > 0:
                slippage_reference_price = stored_sl_price
                slippage_reference_source = "CURRENT_SL_ORDER_TRIGGER_FALLBACK"
            elif actual_initial_sl_price > 0:
                slippage_reference_price = actual_initial_sl_price
                slippage_reference_source = "PLANNED_INVALIDATION_FALLBACK"
            else:
                slippage_reference_price = None
                slippage_reference_source = "UNKNOWN"
            stop_slippage_pct = _calc_stop_slippage_pct(direction, sl_exit_price, slippage_reference_price)
            loss_pct = max(0.0, -final_pnl) if final_pnl is not None else None
            tail_loss_multiple = (loss_pct / planned_risk_pct) if loss_pct is not None and planned_risk_pct and planned_risk_pct > 0 else None
            execution_anomaly = bool(
                exit_reason in {"STOP_LOSS", "BREAK_EVEN"}
                and tail_loss_multiple is not None
                and tail_loss_multiple >= 1.50
            )
            actual_initial_sl_risk_pct = None
            if actual_initial_sl_price > 0 and entry_price > 0:
                actual_initial_sl_risk_pct = abs(entry_price - actual_initial_sl_price) / entry_price * 100.0
            elif stored_sl_price > 0 and entry_price > 0:
                actual_initial_sl_risk_pct = abs(entry_price - stored_sl_price) / entry_price * 100.0
            exit_reason_confidence = "confirmed" if (exit_reason in {"TAKE_PROFIT_FULL", "STOP_LOSS", "BREAK_EVEN"} and (closed_by_tp or sl_exit_price is not None)) or exit_reason.startswith("EARLY_LOSS_CUT") else "unknown"
            planned_rr = _safe_float(trade.get("effective_weighted_rr", trade.get("planned_weighted_rr", 1.6625)), 1.6625)
            if final_pnl is None:
                realized_pnl_usdt = None

            trade["remaining_qty"] = 0.0
            trade["realized_pnl_pct"] = final_pnl
            trade["realized_pnl_qty"] = realized_qty
            trade["realized_pnl_weighted_sum"] = realized_weighted
            trade["realized_rr"] = realized_rr
            trade["exit_price"] = exit_price
            trade["exit_price_source"] = exit_price_source
            trade["estimated_exit_price"] = estimated_exit_price
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
                        "exit_price_source": exit_price_source,
                        "estimated_exit_price": estimated_exit_price,
                        # This is algebraically implied by the tracker PnL, not an
                        # independently verified exchange-fill VWAP. Keep the
                        # provenance explicit so auditors do not mistake it for a fill price.
                        "pnl_implied_blended_exit_price": (
                            entry_price * (1.0 + final_pnl / 100.0) if direction == "LONG" else
                            entry_price * (1.0 - final_pnl / 100.0)
                        ) if final_pnl is not None and entry_price > 0 else None,
                        "blended_exit_price_source": "DERIVED_FROM_TRACKER_PNL_NOT_FILL_VWAP" if final_pnl is not None else "UNKNOWN",
                        "realized_pnl_pct": final_pnl,
                        "realized_pnl_pct_source": "TRACKER_WEIGHTED_LEG_PNL_GROSS_OF_COSTS" if final_pnl is not None else "UNKNOWN",
                        "realized_rr": realized_rr,
                        "effective_weighted_rr": planned_rr,
                        "planned_risk_pct": planned_risk_pct,
                        "planned_risk_usdt": planned_risk_usdt,
                        "initial_notional_usdt": initial_notional_usdt,
                        "realized_pnl_usdt": realized_pnl_usdt,
                        "realized_pnl_usdt_source": "TRACKER_DERIVED_FROM_WEIGHTED_PNL_EXCLUDING_COSTS" if realized_pnl_usdt is not None else "UNKNOWN",
                        "fees_usdt": None,
                        "funding_usdt": None,
                        "net_pnl_usdt": None,
                        "pnl_costs_status": "FEES_AND_FUNDING_NOT_ATTRIBUTED",
                        "actual_loss_pct": loss_pct,
                        "tail_loss_multiple": tail_loss_multiple,
                        "stop_slippage_pct": stop_slippage_pct,
                        "filled_sl_order_id": filled_sl_id,
                        "manual_exit_order_id": trade.get("manual_exit_order_id"),
                        "exit_order_ids": sorted({
                            str(x) for x in (
                                [filled_sl_id, trade.get("manual_exit_order_id")]
                                + [item.get("order_id") for item in trade.get("tp_orders", []) if isinstance(item, dict)]
                                + [item.get("order_id") for item in trade.get("sl_order_history", []) if isinstance(item, dict)]
                                + [((trade.get("sl_order") or {}).get("order_id")) if isinstance(trade.get("sl_order"), dict) else None]
                            ) if x not in (None, "")
                        }),
                        "stop_slippage_reference_price": slippage_reference_price,
                        "stop_slippage_reference_source": slippage_reference_source,
                        "execution_anomaly": execution_anomaly,
                        "tp_mode": trade.get("tp_mode", "two_tp"),
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

            close_emoji = "⚠️" if final_pnl is None else ("💚" if final_pnl >= 0.0 else "💔")
            pnl_text = "DATA_ERROR" if final_pnl is None else f"{final_pnl:+.2f}%"
            exit_text = format_price(exit_price) if exit_price is not None and _safe_float(exit_price, 0.0) > 0 else "UNKNOWN"
            log.info("[TRACKER_TRADE_CLOSED] %s (%s/%s) | PnL: %s | Realized R:R: %s | Planned R:R: %s | Exit: %s (%s) | Duration: %.1f min", close_emoji, trade.get("name", symbol), symbol, pnl_text, format_rr(realized_rr, decimals=2), format_rr(planned_rr, decimals=2), exit_text, exit_reason, duration_min)

            notification_persisted = True
            nid = _notification_id(str(event_id), "TRADE_CLOSE")
            close_notification_text = format_trade_closed_message(
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
            )
            # Keep the exact message in the retryable state until its queue record
            # is durable. This covers failures before notifications.json is written.
            trade["close_notification_text"] = close_notification_text
            try:
                _queue_notification(
                    nid,
                    event_id=str(event_id),
                    kind="TRADE_CLOSE",
                    symbol=symbol,
                    direction=direction,
                    text=close_notification_text,
                )
            except Exception as exc:
                notification_persisted = False
                log.error("[TELEGRAM] Close notification queue error %s: %s", event_id, exc)

            trade["close_notification_pending"] = not notification_persisted
            trade["close_journal_pending"] = False
            if notification_persisted:
                trade.pop("close_notification_text", None)

            # A Telegram persistence failure must never bypass cleanup of stale
            # SL/TP orders for a position already confirmed closed.
            cleanup_ok, cleanup_note = _cleanup_closed_trade_protection(trade)
            if cleanup_ok:
                trade["close_cleanup_pending"] = False
                trade["close_cleanup_verified"] = True
                trade["closed"] = bool(notification_persisted)
                trade.pop("close_cleanup_last_error", None)
            else:
                # Keep cleanup retryable. Notification and protection cleanup are
                # separate durable concerns; neither is allowed to erase the other.
                trade["close_cleanup_pending"] = True
                trade["close_cleanup_verified"] = False
                trade["close_cleanup_last_error"] = cleanup_note
                trade["closed"] = False
                log.error("[TRACKER] Trade %s closed but protection cleanup is pending: %s", event_id, cleanup_note)

            if not notification_persisted or not cleanup_ok:
                updated_trades[event_id] = trade

        except Exception as exc:
            log.exception("[TRACKER] Fatal trade error for event %s: %s", event_id, exc)
            updated_trades[event_id] = trade

    _save_active_trades(updated_trades)
