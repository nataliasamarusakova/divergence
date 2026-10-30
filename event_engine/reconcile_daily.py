#!/usr/bin/env python3
"""Read-only daily health/reconciliation check for Divergence.

This script never places, cancels, or modifies orders. Position mismatches are
reported as critical and daily cashflow totals are deliberately marked as
account+symbol aggregates, not attributed to a particular bot position.
Exit status is always zero for reconciliation discrepancies so historical data
issues do not silently suppress a later trading run; API/parse failures are
persisted and alerted instead.
"""
from __future__ import annotations

import json
import math
import os
import hashlib
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from event_engine import bingx  # noqa: E402
from event_engine.telegram import send_detailed  # noqa: E402

DATA_DIR = Path(os.environ.get("DIVERGENCE_DATA_DIR", ROOT / "data"))
ACTIVE_TRADES_PATH = DATA_DIR / "active_trades.json"
TRADES_PATH = DATA_DIR / "trades.jsonl"
HEALTH_PATH = DATA_DIR / "health.jsonl"
ALERT_STATE_PATH = DATA_DIR / "reconciliation_alert_state.json"
ALERT_REPEAT_SUPPRESSION_SEC = max(0.0, min(24 * 3600.0, float(os.environ.get("RECONCILIATION_ALERT_REPEAT_SUPPRESSION_SEC", "3600"))))
MAX_INCOME_SYMBOLS = max(0, min(3, int(os.environ.get("RECONCILIATION_MAX_SYMBOLS", "2"))))
INCOME_TIMEOUT_SEC = max(1.0, min(3.0, float(os.environ.get("RECONCILIATION_INCOME_TIMEOUT_SEC", "3"))))
INCOME_REQUEST_BUDGET = max(1, min(4, int(os.environ.get("RECONCILIATION_INCOME_REQUEST_BUDGET", "4"))))
INCOME_RETRIES = max(0, min(2, int(os.environ.get("RECONCILIATION_INCOME_RETRIES", "1"))))


def _timestamp_ms(value: Any) -> int:
    try:
        stamp = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return 0
    return stamp * 1000 if 0 < stamp < 100_000_000_000 else stamp


def _symbol(value: Any) -> str:
    """Normalize common USDT contract spellings without network lookups.

    BingX can return BASE-USDT while journals/providers may use BASEUSDT.
    Splitting a compact quote suffix avoids producing invalid keys like
    BTCUSDT-USDT, which would otherwise create false reconciliation alarms.
    """
    text = str(value or "").strip().upper().replace("_", "-").replace("/", "-")
    if not text:
        return ""
    if "-" not in text and text.endswith("USDT") and len(text) > len("USDT"):
        text = f"{text[:-len('USDT')]}-USDT"
    elif "-" not in text:
        text += "-USDT"
    return text


# Reconciliation must remain bounded and read-only: symbol normalization must not
# trigger an unbounded contract-catalog request for every row. Keep the known
# Coinalyze/BingX alias here and honor any operator-provided mapping as well.
_KNOWN_POSITION_SYMBOL_ALIASES = {
    "LIT-USDT": "LIGHTER-USDT",
    "LITUSDT": "LIGHTER-USDT",
    "LIT": "LIGHTER-USDT",
    "TRUMP-USDT": "TRUMP-USDT",
    "TRUMPSOL-USDT": "TRUMP-USDT",
}


def _canonical_position_symbol(value: Any) -> str:
    """Normalize known data-provider aliases without performing network I/O.

    Coinalyze uses LIT while BingX's perpetual contract may be exposed as
    LIGHTER-USDT. A reconciliation process must not issue repeated contract-list
    requests while comparing rows; on unknown aliases it preserves the raw key so
    unresolved identity remains visible instead of guessing based on equal sizes.
    """
    raw = _symbol(value)
    if not raw:
        return raw

    # BingX may return the historical Futures API identifier TRUMPSOL-USDT
    # for the token displayed as TRUMPUSDT. This is reconciliation-only; SOL-USDT
    # and the literal TRUMP-SOL-USDT remain independent instrument identities.
    if raw in {"TRUMP-USDT", "TRUMPSOL-USDT"}:
        return "TRUMP-USDT"

    configured = getattr(bingx, "SYMBOL_MAP", {})
    if isinstance(configured, dict):
        for alias, target in configured.items():
            alias_norm, target_norm = _symbol(alias), _symbol(target)
            if raw == alias_norm or raw.replace("-", "") == alias_norm.replace("-", ""):
                return target_norm or raw
            if raw == target_norm:
                return target_norm

    known = _KNOWN_POSITION_SYMBOL_ALIASES.get(raw) or _KNOWN_POSITION_SYMBOL_ALIASES.get(raw.replace("-", ""))
    if known:
        return known

    # Use an already-populated in-process contract catalog only. Never call
    # to_bx_symbol/get_contract here because they may issue a network request.
    cache = getattr(bingx, "CACHE", {})
    catalog = cache.get("data", {}) if isinstance(cache, dict) else {}
    by_name = cache.get("by_display_name", {}) if isinstance(cache, dict) else {}
    if isinstance(catalog, dict) and raw in catalog:
        return raw
    if isinstance(by_name, dict):
        for candidate in (raw, raw.replace("-", ""), raw.replace("-USDT", "USDT")):
            contract = by_name.get(candidate)
            if isinstance(contract, dict) and contract.get("symbol"):
                return _symbol(contract.get("symbol"))
    if isinstance(catalog, dict):
        raw_compact = raw.replace("-", "").replace("/", "")
        for contract in catalog.values():
            if not isinstance(contract, dict):
                continue
            display = _symbol(contract.get("displayName")).replace("-", "").replace("/", "")
            if display == raw_compact and contract.get("symbol"):
                return _symbol(contract.get("symbol"))
    return raw


def _direction(value: Any, qty: float | None = None) -> str:
    side = str(value or "").strip().upper()
    if side in {"LONG", "SHORT"}:
        return side
    if side == "BOTH" and qty is not None:
        return "LONG" if qty > 0 else "SHORT" if qty < 0 else "UNKNOWN"
    return "UNKNOWN"


def _load_json_object(path: Path) -> tuple[dict | None, str | None]:
    if not path.exists():
        return None, "file_missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"read_or_json_error:{exc}"
    if not isinstance(value, dict):
        return None, "root_is_not_object"
    return value, None


def _load_journal(path: Path) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    errors: list[str] = []
    if not path.exists():
        return records, ["trades_journal_missing"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return records, [f"trades_journal_read_error:{exc}"]
    for lineno, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"trades_journal_invalid_json_line:{lineno}:{exc.msg}")
            continue
        if not isinstance(obj, dict):
            errors.append(f"trades_journal_non_object_line:{lineno}")
            continue
        records.append(obj)
    return records, errors


def _active_local_positions(active: dict) -> tuple[dict[tuple[str, str], float], list[str]]:
    positions: dict[tuple[str, str], float] = defaultdict(float)
    errors: list[str] = []
    for event_id, trade in active.items():
        if not isinstance(trade, dict):
            errors.append(f"active_trade_not_object:{event_id}")
            continue
        if bool(trade.get("closed")):
            continue
        try:
            qty = abs(float(trade.get("remaining_qty", 0) or 0))
        except (TypeError, ValueError, OverflowError):
            errors.append(f"active_trade_invalid_qty:{event_id}")
            continue
        if not math.isfinite(qty) or qty <= 1e-12:
            continue
        symbol = _canonical_position_symbol(trade.get("symbol"))
        direction = _direction(trade.get("direction"))
        if not symbol or direction not in {"LONG", "SHORT"}:
            errors.append(f"active_trade_missing_identity:{event_id}")
            continue
        positions[(symbol, direction)] += qty
    return dict(positions), errors


def _exchange_positions(rows: list[dict]) -> tuple[dict[tuple[str, str], float], list[str]]:
    positions: dict[tuple[str, str], float] = defaultdict(float)
    errors: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"exchange_position_not_object:{index}")
            continue
        try:
            qty_signed = float(row.get("positionAmt", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            errors.append(f"exchange_position_invalid_qty:{index}")
            continue
        if not math.isfinite(qty_signed):
            errors.append(f"exchange_position_non_finite_qty:{index}")
            continue
        if abs(qty_signed) <= 1e-12:
            continue
        symbol = _canonical_position_symbol(row.get("symbol"))
        direction = _direction(row.get("positionSide"), qty_signed)
        if not symbol or direction not in {"LONG", "SHORT"}:
            errors.append(f"exchange_position_missing_identity:{index}")
            continue
        positions[(symbol, direction)] += abs(qty_signed)
    return dict(positions), errors


def _protection_from_open_orders_snapshot(snapshot: dict, symbol: str, direction: str) -> dict:
    """Derive one position's protection from a single validated account-wide snapshot."""
    if not isinstance(snapshot, dict) or str(snapshot.get("status", "")).lower() != "ok":
        return {"status": "error", "error": "account_open_orders_snapshot_unavailable", "sl_orders": [], "tp_orders": []}
    orders = snapshot.get("orders")
    if not isinstance(orders, list) or not all(isinstance(order, dict) for order in orders):
        return {"status": "error", "error": "account_open_orders_snapshot_has_invalid_rows", "sl_orders": [], "tp_orders": []}

    sl_orders: list[dict] = []
    tp_orders: list[dict] = []
    for order in orders:
        row_symbol = str(order.get("symbol") or "").strip()
        order_type = str(order.get("type") or "").strip().upper()
        if not row_symbol or not order_type:
            return {"status": "error", "error": "account_open_orders_snapshot_has_unidentifiable_row", "sl_orders": [], "tp_orders": []}
        protective_type = order_type in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET", "STOP", "STOP_MARKET"}
        order_side = str(order.get("positionSide") or "").strip().upper()
        if protective_type and order_side not in {"LONG", "SHORT", "BOTH"}:
            return {"status": "error", "error": "account_open_orders_snapshot_has_unidentifiable_protection_side", "sl_orders": [], "tp_orders": []}
        if _canonical_position_symbol(row_symbol) != symbol:
            continue
        if order_side not in {direction, "BOTH"}:
            continue
        if order_type in {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}:
            tp_orders.append(order)
        elif order_type in {"STOP", "STOP_MARKET"}:
            sl_orders.append(order)
    return {
        "status": "ok", "symbol": symbol, "positionSide": direction,
        "sl_orders": sl_orders, "tp_orders": tp_orders,
        "source": "account_open_orders_snapshot",
    }


def _append_health(path: Path, report: dict) -> str | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return None
    except OSError as exc:
        return f"health_jsonl_write_error:{exc}"


def _alert_signature(discrepancies: list[dict]) -> str:
    """Fingerprint the exact discrepancy state, including quantities and identity."""
    normalized = json.dumps(discrepancies, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _repeat_alert_suppressed(discrepancies: list[dict], path: Path, now_ms: int) -> tuple[bool, str]:
    if not discrepancies:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return False, "no_discrepancies"
    if ALERT_REPEAT_SUPPRESSION_SEC <= 0:
        return False, "suppression_disabled"
    try:
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        saved = {}
    signature = _alert_signature(discrepancies)
    if saved.get("signature") == signature:
        last_sent = int(saved.get("last_sent_ts_ms", 0) or 0)
        age_sec = (now_ms - last_sent) / 1000.0 if last_sent > 0 else float("inf")
        if 0 <= age_sec < ALERT_REPEAT_SUPPRESSION_SEC:
            return True, f"unchanged_discrepancy_suppressed_for_{int(ALERT_REPEAT_SUPPRESSION_SEC)}s"
    return False, "new_or_changed_discrepancy"


def _mark_alert_sent(discrepancies: list[dict], path: Path, now_ms: int) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps({
            "signature": _alert_signature(discrepancies),
            "last_sent_ts_ms": int(now_ms),
        }, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        # Notification dedup is optional telemetry; it must not interrupt reconciliation.
        print(f"[DAILY_RECONCILIATION] Could not persist alert suppression state: {exc}", file=sys.stderr)


def _discrepancy_message_detail(item: dict) -> str:
    if item.get("code") == "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH":
        return (
            f"{item.get('symbol', '?')} {item.get('direction', '?')}: "
            f"local_qty={item.get('local_qty')} exchange_qty={item.get('exchange_qty')} "
            f"delta={item.get('delta_qty')} tolerance={item.get('tolerance_qty')}"
        )
    return str(item.get("detail") or item.get("code") or "unknown")


def run_reconciliation(*, now_ms: int | None = None, data_dir: Path | None = None, notify: bool = True) -> dict:
    """Run one bounded, read-only reconciliation and return a JSON-safe report."""
    global ACTIVE_TRADES_PATH, TRADES_PATH, HEALTH_PATH
    if data_dir is not None:
        data_dir = Path(data_dir)
        active_path, trades_path, health_path = data_dir / "active_trades.json", data_dir / "trades.jsonl", data_dir / "health.jsonl"
        alert_state_path = data_dir / "reconciliation_alert_state.json"
    else:
        active_path, trades_path, health_path = ACTIVE_TRADES_PATH, TRADES_PATH, HEALTH_PATH
        alert_state_path = ALERT_STATE_PATH
    now_ms = int(now_ms or time.time() * 1000)
    utc_now = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
    day_start = int(utc_now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    report: dict[str, Any] = {
        "record_type": "DAILY_RECONCILIATION",
        "ts": now_ms,
        "build_version": str(os.environ.get("DIVERGENCE_BUILD_VERSION", "UNKNOWN")),
        "utc_day": utc_now.date().isoformat(),
        "mode": "READ_ONLY",
        "status": "OK",
        "checks": {},
        "discrepancies": [],
        "notes": ["Daily income totals are account+symbol aggregates and are not per-position attribution."],
    }

    active, active_error = _load_json_object(active_path)
    local_positions: dict[tuple[str, str], float] = {}
    if active is None:
        report["checks"]["local_active_state"] = {"status": "UNKNOWN", "error": active_error}
        report["discrepancies"].append({"level": "WARNING", "code": "LOCAL_ACTIVE_STATE_UNAVAILABLE", "detail": active_error})
    else:
        local_positions, active_errors = _active_local_positions(active)
        report["checks"]["local_active_state"] = {
            "status": "OK" if not active_errors else "WARNING",
            "active_event_records": len(active),
            "active_position_keys": len(local_positions),
            "parse_errors": active_errors,
        }
        for detail in active_errors:
            report["discrepancies"].append({"level": "WARNING", "code": "LOCAL_ACTIVE_STATE_INVALID", "detail": detail})

    try:
        exchange_rows = bingx.get_positions(timeout_sec=3.0, retryable=False)
        exchange_positions, exchange_errors = _exchange_positions(exchange_rows)
        report["checks"]["exchange_positions"] = {
            "status": "OK" if not exchange_errors else "WARNING",
            "active_position_keys": len(exchange_positions),
            "api_rows": len(exchange_rows),
            "parse_errors": exchange_errors,
        }
        for detail in exchange_errors:
            report["discrepancies"].append({"level": "CRITICAL", "code": "EXCHANGE_POSITION_ROW_UNIDENTIFIABLE", "detail": detail})
        if active is not None and not active_errors and not exchange_errors:
            for key in sorted(set(local_positions) | set(exchange_positions)):
                local_qty = local_positions.get(key, 0.0)
                exchange_qty = exchange_positions.get(key, 0.0)
                tolerance = max(1e-8, exchange_qty * 0.01, local_qty * 0.01)
                if abs(local_qty - exchange_qty) > tolerance:
                    report["discrepancies"].append({
                        "level": "CRITICAL", "code": "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH",
                        "symbol": key[0], "direction": key[1], "local_qty": local_qty,
                        "exchange_qty": exchange_qty, "delta_qty": exchange_qty - local_qty,
                        "tolerance_qty": tolerance,
                    })
    except Exception as exc:
        exchange_positions = {}
        report["checks"]["exchange_positions"] = {"status": "ERROR", "error": f"{type(exc).__name__}:{exc}"}
        report["discrepancies"].append({"level": "CRITICAL", "code": "EXCHANGE_POSITION_QUERY_FAILED", "detail": str(exc)})

    # Fetch one account-wide snapshot. Querying openOrders once per position can
    # trip BingX's endpoint trigger-frequency breaker, and symbol-by-symbol retries
    # after code 100410 only amplify the outage. Unknown remains CRITICAL, but the
    # report is aggregated and no fallback requests are made from this read-only job.
    protection_results: dict[str, dict] = {}
    protection_snapshot: dict[str, Any] = {"status": "not_requested", "orders": []}
    protection_snapshot_error: str | None = None
    protection_snapshot_rate_limited = False
    protection_retry_after_ms = None
    if exchange_positions:
        try:
            candidate_snapshot = bingx.get_open_orders_snapshot(timeout_sec=3.0, retryable=False)
        except Exception as exc:
            candidate_snapshot = {"status": "error", "error": f"{type(exc).__name__}:{exc}"}
        if isinstance(candidate_snapshot, dict) and str(candidate_snapshot.get("status", "")).lower() == "ok":
            protection_snapshot = candidate_snapshot
        else:
            protection_snapshot_error = (
                str(candidate_snapshot.get("error") or "account_open_orders_snapshot_unavailable")
                if isinstance(candidate_snapshot, dict) else "invalid_account_open_orders_snapshot_response"
            )
            try:
                protection_code = int(candidate_snapshot.get("code")) if isinstance(candidate_snapshot, dict) else None
            except (TypeError, ValueError, OverflowError):
                protection_code = None
            protection_snapshot_rate_limited = bool(
                isinstance(candidate_snapshot, dict)
                and (candidate_snapshot.get("rate_limited") or candidate_snapshot.get("local_cooldown"))
            ) or protection_code == 100410
            protection_retry_after_ms = candidate_snapshot.get("retry_after_ms") if isinstance(candidate_snapshot, dict) else None

    if protection_snapshot_error is not None:
        affected_positions = []
        for (symbol, direction), exchange_qty in sorted(exchange_positions.items()):
            key = f"{symbol}:{direction}"
            affected_positions.append(key)
            protection_results[key] = {
                "status": "UNKNOWN", "error": protection_snapshot_error,
                "exchange_qty": exchange_qty, "source": "account_open_orders_snapshot",
            }
        if protection_snapshot_rate_limited:
            # Exclude the server's millisecond-changing unblock timestamp from the
            # discrepancy fingerprint so identical 100410 incidents deduplicate.
            detail = (
                "BingX openOrders endpoint rate-limited (code 100410); protection state "
                f"remains unknown for {len(affected_positions)} position(s). "
                "No per-symbol retries were attempted."
            )
        else:
            detail = (
                "Account-wide openOrders snapshot unavailable; protection state remains "
                f"unknown for {len(affected_positions)} position(s). No per-symbol retries were attempted."
            )
        report["discrepancies"].append({
            "level": "CRITICAL", "code": "EXCHANGE_PROTECTION_STATE_UNKNOWN",
            "position_count": len(affected_positions), "positions": affected_positions,
            "rate_limited": protection_snapshot_rate_limited, "detail": detail,
        })
    else:
        # Confirm that each visible exchange position has one identifiable SL.
        # This is read-only; ambiguous openOrders data never becomes "no protection".
        for (symbol, direction), exchange_qty in sorted(exchange_positions.items()):
            key = f"{symbol}:{direction}"
            protection = _protection_from_open_orders_snapshot(protection_snapshot, symbol, direction)
            if not isinstance(protection, dict) or str(protection.get("status", "")).lower() != "ok":
                detail = str(protection.get("error", "invalid_protection_response")) if isinstance(protection, dict) else "invalid_protection_response"
                protection_results[key] = {"status": "ERROR", "error": detail}
                report["discrepancies"].append({"level": "CRITICAL", "code": "EXCHANGE_PROTECTION_STATE_UNKNOWN", "symbol": symbol, "direction": direction, "detail": detail})
                continue
            sl_orders = protection.get("sl_orders")
            tp_orders = protection.get("tp_orders")
            if not isinstance(sl_orders, list) or not isinstance(tp_orders, list):
                protection_results[key] = {"status": "ERROR", "error": "protective_orders_not_lists"}
                report["discrepancies"].append({"level": "CRITICAL", "code": "EXCHANGE_PROTECTION_STATE_UNKNOWN", "symbol": symbol, "direction": direction, "detail": "protective_orders_not_lists"})
                continue
            sl_info = []
            for order in sl_orders:
                if not isinstance(order, dict):
                    sl_info.append({"valid": False, "reason": "sl_row_not_object"})
                    continue
                order_id = str(order.get("orderId") or order.get("orderID") or order.get("clientOrderId") or "")
                order_type = str(order.get("type") or "").strip().upper()
                order_side = str(order.get("positionSide") or "").strip().upper()
                try:
                    order_qty = abs(float(order.get("origQty") or order.get("quantity") or 0))
                except (TypeError, ValueError, OverflowError):
                    order_qty = 0.0
                raw_stop_price = next(
                    (order.get(field) for field in ("stopPrice", "triggerPrice", "stop_price") if order.get(field) not in (None, "")),
                    0,
                )
                try:
                    stop_price = float(raw_stop_price)
                except (TypeError, ValueError, OverflowError):
                    stop_price = 0.0
                qty_ok = math.isfinite(order_qty) and order_qty > 0 and abs(order_qty - exchange_qty) <= max(1e-8, exchange_qty * 0.01)
                trigger_ok = math.isfinite(stop_price) and stop_price > 0
                valid = bool(order_id and order_type in {"STOP", "STOP_MARKET"} and order_side in {direction, "BOTH"} and qty_ok and trigger_ok)
                sl_info.append({"order_id": order_id or None, "type": order_type or None, "position_side": order_side or None, "quantity": order_qty, "quantity_matches_position": qty_ok, "stop_price": stop_price if trigger_ok else None, "trigger_price_valid": trigger_ok, "valid": valid})
            status = "OK"
            if len(sl_orders) != 1 or len(sl_info) != 1 or not sl_info[0].get("valid"):
                status = "ERROR"
                detail = "missing_or_duplicate_stop_loss" if not sl_orders or len(sl_orders) > 1 else "stop_loss_identity_type_side_qty_or_trigger_price_invalid"
                report["discrepancies"].append({"level": "CRITICAL", "code": "EXCHANGE_STOP_LOSS_NOT_VERIFIED", "symbol": symbol, "direction": direction, "detail": detail, "sl_orders": sl_info, "exchange_qty": exchange_qty})
            elif not tp_orders:
                status = "WARNING"
                report["discrepancies"].append({"level": "WARNING", "code": "EXCHANGE_TAKE_PROFIT_NOT_VISIBLE", "symbol": symbol, "direction": direction, "detail": "one verified SL is visible, but no open TP order is visible"})
            protection_results[key] = {"status": status, "sl_orders": sl_info, "tp_order_count": len(tp_orders), "source": "account_open_orders_snapshot"}
    protection_check_status = "OK"
    if any(item.get("status") == "UNKNOWN" for item in protection_results.values()):
        protection_check_status = "UNKNOWN"
    elif any(item.get("status") != "OK" for item in protection_results.values()):
        protection_check_status = "WARNING"
    report["checks"]["exchange_protection"] = {
        "status": protection_check_status,
        "positions_checked": len(protection_results), "by_position": protection_results,
        "source": "account_open_orders_snapshot" if exchange_positions else "not_needed_no_open_positions",
        "error": protection_snapshot_error,
        "rate_limited": protection_snapshot_rate_limited,
        "retry_after_ms": protection_retry_after_ms,
        "note": "Requires exactly one identifiable STOP/STOP_MARKET SL with a finite positive trigger price, expected directional side, and approximate live quantity; uses one account-wide read-only snapshot and never modifies orders.",
    }

    journal, journal_errors = _load_journal(trades_path)
    if journal_errors:
        report["checks"]["trade_journal"] = {"status": "WARNING", "records": len(journal), "parse_errors": journal_errors[:50]}
        report["discrepancies"].extend({"level": "WARNING", "code": "TRADE_JOURNAL_ISSUE", "detail": item} for item in journal_errors[:50])
    else:
        report["checks"]["trade_journal"] = {"status": "OK", "records": len(journal)}

    today_records = []
    for record in journal:
        stamp = _timestamp_ms(record.get("closed_ts") or record.get("ts") or record.get("timestamp"))
        if day_start <= stamp <= now_ms:
            today_records.append(record)
    record_counts = Counter(str(row.get("record_type") or "UNKNOWN") for row in today_records)
    today_data_errors = [
        {"event_id": row.get("event_id"), "symbol": row.get("symbol"), "closed_ts": row.get("closed_ts"), "data_error": row.get("data_error"), "error_code": row.get("error_code")}
        for row in today_records
        if row.get("record_type") == "TRADE_CLOSE" and (
            str(row.get("exit_reason") or "").upper() == "DATA_ERROR"
            or bool(row.get("data_error"))
        )
    ]
    if today_data_errors:
        report["discrepancies"].append({
            "level": "CRITICAL", "code": "DATA_ERROR_CLOSES_IN_UTC_DAY",
            "count": len(today_data_errors), "trades": today_data_errors[:25],
            "detail": "Close outcome or net-cost attribution remains unconfirmed for this UTC day.",
        })
    confirmed_daily_by_symbol: dict[str, dict[str, float]] = defaultdict(lambda: {"net": 0.0, "gross": 0.0, "fees": 0.0, "funding": 0.0, "trades": 0.0})
    for record in today_records:
        if record.get("record_type") != "TRADE_CLOSE":
            continue
        if record.get("pnl_costs_status") != "BINGX_ORDER_LINKED_FILLS_AND_INCOME_CONFIRMED" or record.get("net_pnl_usdt") is None:
            continue
        bucket = confirmed_daily_by_symbol[_symbol(record.get("symbol"))]
        bucket["net"] += float(record["net_pnl_usdt"])
        bucket["gross"] += float(record.get("realized_pnl_usdt") or 0.0)
        bucket["fees"] += float(record.get("fees_usdt") or 0.0)
        bucket["funding"] += float(record.get("funding_usdt") or 0.0)
        bucket["trades"] += 1.0
    report["checks"]["daily_journal_activity"] = {
        "status": "OK" if not journal_errors else "WARNING",
        "records": len(today_records), "record_type_counts": dict(record_counts),
        "confirmed_net_pnl_by_symbol": {key: value for key, value in sorted(confirmed_daily_by_symbol.items())},
        "note": "Only exact exchange-fill/income-linked net PnL records are counted as confirmed.",
    }

    candidate_symbols = Counter()
    for record in today_records:
        sym = _symbol(record.get("symbol"))
        if sym and record.get("record_type") in {"TRADE_OPEN", "TRADE_CLOSE"}:
            candidate_symbols[sym] += 1
    for sym, _ in local_positions:
        candidate_symbols[sym] += 1
    income_summaries: dict[str, dict] = {}
    for symbol, _count in candidate_symbols.most_common(MAX_INCOME_SYMBOLS):
        income: dict[str, Any] = {"status": "error", "error": "income_query_not_attempted", "complete": False, "records": []}
        attempts = 0
        for attempt in range(INCOME_RETRIES + 1):
            attempts = attempt + 1
            try:
                income = bingx.get_income_history(
                    symbol, day_start, max(day_start + 1, now_ms), limit=1000,
                    timeout_sec=INCOME_TIMEOUT_SEC, max_range_requests=INCOME_REQUEST_BUDGET,
                )
            except Exception as exc:
                income = {"status": "error", "error": f"{type(exc).__name__}:{exc}", "complete": False, "records": []}
            if isinstance(income, dict) and income.get("status") == "ok" and income.get("complete") is True:
                break
            if attempt < INCOME_RETRIES:
                # One bounded retry helps with brief API latency spikes without
                # turning read-only reconciliation into an unbounded request loop.
                time.sleep(min(0.25 * (attempt + 1), 0.5))
        if not isinstance(income, dict) or income.get("status") != "ok" or income.get("complete") is not True:
            detail = str(income.get("error", "incomplete_income_response")) if isinstance(income, dict) else "invalid_income_response"
            shape_fingerprint = income.get("shape_fingerprint") if isinstance(income, dict) else None
            shape_summary = income.get("shape_summary") if isinstance(income, dict) else None
            if shape_fingerprint:
                detail += f"; shape_fingerprint={shape_fingerprint}"
            if shape_summary:
                detail += f"; response_shape={shape_summary}"
            income_summaries[symbol] = {
                "status": "ERROR", "error": detail, "attempts": attempts,
                "shape_fingerprint": shape_fingerprint, "shape_summary": shape_summary,
            }
            report["discrepancies"].append({"level": "WARNING", "code": "DAILY_INCOME_QUERY_INCOMPLETE", "symbol": symbol, "detail": f"{detail} (attempts={attempts})"})
            continue
        totals = {"realized_pnl": 0.0, "trading_fees": 0.0, "funding_fees": 0.0, "other_known": 0.0, "records": len(income.get("records", [])), "attempts": attempts}
        for row in income.get("records", []):
            if not isinstance(row, dict) or str(row.get("asset") or "").upper() != "USDT":
                continue
            try:
                amount = float(row.get("amount"))
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(amount):
                continue
            kind = str(row.get("income_type") or "").upper()
            if kind == "REALIZED_PNL":
                totals["realized_pnl"] += amount
            elif kind == "TRADING_FEE":
                totals["trading_fees"] += amount
            elif kind == "FUNDING_FEE":
                totals["funding_fees"] += amount
            elif kind in {"INSURANCE_CLEAR", "ADL", "SYSTEM_DEDUCTION", "GTD_PRICE"}:
                totals["other_known"] += amount
        income_net = sum(totals[key] for key in ("realized_pnl", "trading_fees", "funding_fees", "other_known"))
        totals["aggregate_net"] = income_net
        totals["source_scope"] = "ACCOUNT_SYMBOL_DAILY_AGGREGATE_NOT_PER_POSITION"
        income_summaries[symbol] = {"status": "OK", **totals}
        local_confirmed = confirmed_daily_by_symbol.get(symbol)
        if local_confirmed is not None:
            component_deltas = {
                "gross_delta_usdt": totals["realized_pnl"] - local_confirmed["gross"],
                "fees_delta_usdt": totals["trading_fees"] - local_confirmed["fees"],
                "funding_delta_usdt": totals["funding_fees"] - local_confirmed["funding"],
                "net_delta_usdt": income_net - local_confirmed["net"],
            }
            if any(abs(delta) > 0.02 for delta in component_deltas.values()):
                report["discrepancies"].append({
                    "level": "WARNING", "code": "DAILY_SYMBOL_AGGREGATE_DIFFERS_FROM_CONFIRMED_BOT_TRADES",
                    "symbol": symbol, "exchange_account_symbol_aggregate": totals,
                    "bot_confirmed_trade_totals": local_confirmed, "component_deltas": component_deltas,
                    "detail": "Warning only: account aggregate may include non-bot/manual trades and other open positions.",
                })
    report["checks"]["daily_exchange_income_aggregates"] = {
        "status": "OK" if all(x.get("status") == "OK" for x in income_summaries.values()) else ("UNKNOWN" if not income_summaries else "WARNING"),
        "symbols_checked": income_summaries,
        "scope": "account+symbol aggregate; never used to attribute per-position net PnL",
    }

    levels = [str(item.get("level", "WARNING")).upper() for item in report["discrepancies"]]
    report["status"] = "CRITICAL" if "CRITICAL" in levels else "WARNING" if levels else "OK"
    write_error = _append_health(health_path, report)
    if write_error:
        report["status"] = "CRITICAL"
        report["discrepancies"].append({"level": "CRITICAL", "code": "HEALTH_LOG_WRITE_FAILED", "detail": write_error})
    if notify:
        suppress, suppression_reason = _repeat_alert_suppressed(report["discrepancies"], alert_state_path, now_ms)
        if report["discrepancies"] and suppress:
            report["notification"] = {"status": "SUPPRESSED_REPEAT", "reason": suppression_reason}
        elif report["discrepancies"]:
            lines = [
                f"<b>Divergence daily reconciliation: {report['status']}</b>",
                f"Build: <code>{report.get('build_version', 'UNKNOWN')}</code>",
                f"UTC day: <code>{report['utc_day']}</code>",
            ]
            for item in report["discrepancies"][:12]:
                detail = _discrepancy_message_detail(item)
                lines.append(f"• <b>{item.get('level')}</b> {item.get('code')}: <code>{str(detail)[:350]}</code>")
            lines.append("Read-only check; it does not place/cancel orders or block a later run.")
            try:
                delivery = send_detailed("\n".join(lines))
                delivered = bool(delivery) and all(
                    isinstance(result, dict) and result.get("sent") is True
                    for result in delivery.values()
                )
                if delivered:
                    _mark_alert_sent(report["discrepancies"], alert_state_path, now_ms)
                    report["notification"] = {"status": "SENT", "reason": suppression_reason}
                else:
                    # A failed/missing Telegram delivery must not start the repeat
                    # suppression window; the unchanged critical alert needs retry.
                    report["notification"] = {"status": "FAILED", "error": "telegram_delivery_not_confirmed", "delivery": delivery}
            except Exception as exc:
                report["notification"] = {"status": "FAILED", "error": str(exc)}
                print(f"[DAILY_RECONCILIATION] Telegram alert failed: {exc}", file=sys.stderr)
        else:
            report["notification"] = {"status": "NOT_NEEDED", "reason": "no_discrepancies"}
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report


def main() -> int:
    try:
        run_reconciliation()
    except Exception as exc:  # fail observable, not silently; workflow can continue
        emergency_report = {
            "record_type": "DAILY_RECONCILIATION", "ts": int(time.time() * 1000),
            "mode": "READ_ONLY", "status": "CRITICAL",
            "discrepancies": [{"level": "CRITICAL", "code": "RECONCILIATION_UNHANDLED_EXCEPTION", "detail": f"{type(exc).__name__}:{exc}"}],
        }
        _append_health(HEALTH_PATH, emergency_report)
        try:
            send_detailed(f"<b>Divergence daily reconciliation failed</b>\n<code>{type(exc).__name__}: {str(exc)[:700]}</code>")
        except Exception:
            pass
        print(json.dumps(emergency_report, ensure_ascii=False, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
