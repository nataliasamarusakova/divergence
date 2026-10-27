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
MAX_INCOME_SYMBOLS = max(0, min(3, int(os.environ.get("RECONCILIATION_MAX_SYMBOLS", "2"))))
INCOME_TIMEOUT_SEC = max(1.0, min(3.0, float(os.environ.get("RECONCILIATION_INCOME_TIMEOUT_SEC", "2"))))
INCOME_REQUEST_BUDGET = max(1, min(4, int(os.environ.get("RECONCILIATION_INCOME_REQUEST_BUDGET", "4"))))


def _timestamp_ms(value: Any) -> int:
    try:
        stamp = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return 0
    return stamp * 1000 if 0 < stamp < 100_000_000_000 else stamp


def _symbol(value: Any) -> str:
    text = str(value or "").strip().upper().replace("_", "-").replace("/", "-")
    if text and "-" not in text:
        text += "-USDT"
    return text


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
        symbol = _symbol(trade.get("symbol"))
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
        symbol = _symbol(row.get("symbol"))
        direction = _direction(row.get("positionSide"), qty_signed)
        if not symbol or direction not in {"LONG", "SHORT"}:
            errors.append(f"exchange_position_missing_identity:{index}")
            continue
        positions[(symbol, direction)] += abs(qty_signed)
    return dict(positions), errors


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


def run_reconciliation(*, now_ms: int | None = None, data_dir: Path | None = None, notify: bool = True) -> dict:
    """Run one bounded, read-only reconciliation and return a JSON-safe report."""
    global ACTIVE_TRADES_PATH, TRADES_PATH, HEALTH_PATH
    if data_dir is not None:
        data_dir = Path(data_dir)
        active_path, trades_path, health_path = data_dir / "active_trades.json", data_dir / "trades.jsonl", data_dir / "health.jsonl"
    else:
        active_path, trades_path, health_path = ACTIVE_TRADES_PATH, TRADES_PATH, HEALTH_PATH
    now_ms = int(now_ms or time.time() * 1000)
    utc_now = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
    day_start = int(utc_now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    report: dict[str, Any] = {
        "record_type": "DAILY_RECONCILIATION",
        "ts": now_ms,
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

    # Confirm that each visible exchange position has one identifiable SL.
    # This is read-only; ambiguous openOrders data never becomes "no protection".
    protection_results: dict[str, dict] = {}
    for (symbol, direction), exchange_qty in sorted(exchange_positions.items()):
        key = f"{symbol}:{direction}"
        try:
            protection = bingx.get_open_protection_directional(
                symbol, direction, timeout_sec=3.0, retryable=False,
            )
        except Exception as exc:
            protection = {"status": "error", "error": f"{type(exc).__name__}:{exc}", "sl_orders": [], "tp_orders": []}
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
                (order.get(key) for key in ("stopPrice", "triggerPrice", "stop_price") if order.get(key) not in (None, "")),
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
        protection_results[key] = {"status": status, "sl_orders": sl_info, "tp_order_count": len(tp_orders)}
    report["checks"]["exchange_protection"] = {
        "status": "OK" if all(item.get("status") == "OK" for item in protection_results.values()) else ("UNKNOWN" if not protection_results else "WARNING"),
        "positions_checked": len(protection_results), "by_position": protection_results,
        "note": "Requires exactly one identifiable STOP/STOP_MARKET SL with a finite positive trigger price, expected directional side, and approximate live quantity; no order is modified.",
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
        try:
            income = bingx.get_income_history(
                symbol, day_start, max(day_start + 1, now_ms), limit=1000,
                timeout_sec=INCOME_TIMEOUT_SEC, max_range_requests=INCOME_REQUEST_BUDGET,
            )
        except Exception as exc:
            income = {"status": "error", "error": f"{type(exc).__name__}:{exc}", "complete": False, "records": []}
        if not isinstance(income, dict) or income.get("status") != "ok" or income.get("complete") is not True:
            detail = str(income.get("error", "incomplete_income_response")) if isinstance(income, dict) else "invalid_income_response"
            income_summaries[symbol] = {"status": "ERROR", "error": detail}
            report["discrepancies"].append({"level": "WARNING", "code": "DAILY_INCOME_QUERY_INCOMPLETE", "symbol": symbol, "detail": detail})
            continue
        totals = {"realized_pnl": 0.0, "trading_fees": 0.0, "funding_fees": 0.0, "other_known": 0.0, "records": len(income.get("records", []))}
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
    if notify and report["discrepancies"]:
        lines = [f"<b>Divergence daily reconciliation: {report['status']}</b>", f"UTC day: <code>{report['utc_day']}</code>"]
        for item in report["discrepancies"][:12]:
            detail = item.get("detail") or item.get("code")
            lines.append(f"• <b>{item.get('level')}</b> {item.get('code')}: <code>{str(detail)[:350]}</code>")
        lines.append("Read-only check; it does not place/cancel orders or block a later run.")
        try:
            send_detailed("\n".join(lines))
        except Exception as exc:
            print(f"[DAILY_RECONCILIATION] Telegram alert failed: {exc}", file=sys.stderr)
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
