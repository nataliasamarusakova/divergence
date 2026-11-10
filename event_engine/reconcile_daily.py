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
from event_engine.version import APP_VERSION  # noqa: E402
from event_engine.telegram import send_detailed  # noqa: E402

DATA_DIR = Path(os.environ.get("DIVERGENCE_DATA_DIR", ROOT / "data"))
ACTIVE_TRADES_PATH = DATA_DIR / "active_trades.json"
TRADES_PATH = DATA_DIR / "trades.jsonl"
HEALTH_PATH = DATA_DIR / "health.jsonl"
ALERT_STATE_PATH = DATA_DIR / "reconciliation_alert_state.json"
RUNTIME_STATE_MARKER_PATH = DATA_DIR / "runtime_state_initialized.json"
INCOME_PENDING_STATE_PATH = DATA_DIR / "reconciliation_income_pending.json"
ALERT_REPEAT_SUPPRESSION_SEC = max(0.0, min(24 * 3600.0, float(os.environ.get("RECONCILIATION_ALERT_REPEAT_SUPPRESSION_SEC", "3600"))))
MAX_INCOME_SYMBOLS = max(0, min(3, int(os.environ.get("RECONCILIATION_MAX_SYMBOLS", "2"))))
INCOME_TIMEOUT_SEC = max(1.0, min(3.0, float(os.environ.get("RECONCILIATION_INCOME_TIMEOUT_SEC", "3"))))
INCOME_REQUEST_BUDGET = max(1, min(4, int(os.environ.get("RECONCILIATION_INCOME_REQUEST_BUDGET", "4"))))
INCOME_RETRIES = max(0, min(2, int(os.environ.get("RECONCILIATION_INCOME_RETRIES", "1"))))
INCOME_NULL_GRACE_SEC = max(
    0.0,
    min(6 * 3600.0, float(os.environ.get("RECONCILIATION_INCOME_NULL_GRACE_SEC", "1800"))),
)
# A mismatch can be caused by a fill that occurred after the tracker poll at the
# start of the scan. Verify order executions only for mismatched positions and
# keep the read-only reconciliation bounded.
MAX_TRACKER_FILL_ORDER_QUERIES = max(0, min(32, int(os.environ.get("RECONCILIATION_FILL_ORDER_QUERY_BUDGET", "24"))))
TRACKER_FILL_ORDER_TIMEOUT_SEC = max(1.0, min(5.0, float(os.environ.get("RECONCILIATION_FILL_ORDER_TIMEOUT_SEC", "2"))))
TRACKER_FILL_ORDER_TIME_BUDGET_SEC = max(1.0, min(60.0, float(os.environ.get("RECONCILIATION_FILL_ORDER_TIME_BUDGET_SEC", "20"))))


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

    # Use the common in-memory contract resolver.  It performs no network I/O;
    # run_reconciliation loads the catalog once before comparing positions.
    resolver = getattr(bingx, "canonical_contract_symbol", None)
    if callable(resolver):
        return _symbol(resolver(value))
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


def _atomic_write_text(path: Path, content: str) -> None:
    """Durably replace a small reconciliation state file without clobbering on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Directory fsync is not supported by every filesystem.
            pass
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _write_json_state(path: Path, value: dict) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _has_prior_runtime_state_evidence(
    *, active_path: Path, trades_path: Path, health_path: Path,
) -> bool:
    """Detect evidence that the local trading state was previously initialized.

    A missing pair of files is bootstrap-safe only on a genuinely clean start.
    Existing state files, archived trade journals, or a previous successful
    reconciliation prove that silently recreating an empty active state could
    hide a later state loss.
    """
    if active_path.exists() or trades_path.exists():
        return True
    archive_dir = trades_path.parent / "archive"
    if archive_dir.exists() and any(archive_dir.glob("trades-*.jsonl.gz")):
        return True
    if not health_path.exists():
        return False
    try:
        with health_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A damaged runtime history is not evidence of a clean
                    # first start. Do not bootstrap empty state over ambiguity.
                    return True
                if not isinstance(row, dict) or row.get("record_type") != "DAILY_RECONCILIATION":
                    continue
                checks = row.get("checks") if isinstance(row.get("checks"), dict) else {}
                local = checks.get("local_active_state") if isinstance(checks.get("local_active_state"), dict) else {}
                journal = checks.get("trade_journal") if isinstance(checks.get("trade_journal"), dict) else {}
                # UNKNOWN means the file had not yet been created. OK/WARNING
                # means a real local file was read and therefore existed.
                if str(local.get("status", "")).upper() in {"OK", "WARNING"}:
                    return True
                if str(journal.get("status", "")).upper() == "OK":
                    return True
    except OSError:
        # An unreadable history file is not proof of a clean first start.
        return True
    return False


def _ensure_runtime_state_initialized(
    *, active_path: Path, trades_path: Path, health_path: Path,
    marker_path: Path, active: dict | None, active_error: str | None,
    exchange_positions: dict[tuple[str, str], float], exchange_errors: list[str],
    now_ms: int,
) -> tuple[dict | None, str | None, dict]:
    """Initialize only a confirmed flat, first-run account; never heal state loss."""
    if marker_path.exists():
        marker, marker_error = _load_json_object(marker_path)
        if marker_error or not isinstance(marker, dict) or marker.get("version") != 1 or marker.get("initialized") is not True:
            return active, active_error, {
                "status": "ERROR", "error": marker_error or "runtime_state_marker_schema_invalid",
            }
        return active, active_error, {"status": "OK", "source": "persistent_marker"}

    prior_evidence = _has_prior_runtime_state_evidence(
        active_path=active_path, trades_path=trades_path, health_path=health_path,
    )
    reason = "legacy_state_or_history_present"
    status = "MARKED_EXISTING_STATE"
    if not prior_evidence:
        if exchange_errors:
            # A failed/ambiguous exchange response cannot establish that the
            # account is flat. Leave the marker absent so a later clean snapshot
            # can complete first-run initialization.
            return active, active_error, {
                "status": "UNKNOWN", "reason": "exchange_snapshot_not_clean",
            }
        if exchange_positions:
            # Live exchange exposure with no local files is not a clean start.
            # Persist this fact so a later flat snapshot cannot silently erase it.
            reason = "exchange_exposure_present_without_local_state"
            status = "EXPOSURE_WITHOUT_LOCAL_STATE"
        else:
            try:
                if not active_path.exists() and not trades_path.exists():
                    _atomic_write_text(active_path, "{}\n")
                    _atomic_write_text(trades_path, "")
                    active = {}
                    active_error = None
                    reason = "confirmed_empty_exchange_snapshot"
                    status = "BOOTSTRAPPED_EMPTY_ACCOUNT"
                else:
                    # A partial file set is evidence of a previously initialized
                    # state and must not be completed with invented empty data.
                    reason = "partial_local_state_present"
                    status = "PARTIAL_STATE_NOT_BOOTSTRAPPED"
            except OSError as exc:
                return active, active_error, {
                    "status": "ERROR", "error": f"runtime_state_bootstrap_failed:{type(exc).__name__}:{exc}",
                }

    marker = {
        "version": 1,
        "initialized": True,
        "initialized_at_ts_ms": int(now_ms),
        "reason": reason,
    }
    try:
        _write_json_state(marker_path, marker)
    except OSError as exc:
        return active, active_error, {
            "status": "ERROR", "error": f"runtime_state_marker_write_failed:{type(exc).__name__}:{exc}",
        }
    return active, active_error, {"status": status, **marker}


def _load_income_pending_state(path: Path) -> tuple[dict[str, dict], str | None, bool]:
    """Load persisted null-income grace timestamps without repairing corruption."""
    if not path.exists():
        return {}, None, False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"income_pending_state_read_error:{type(exc).__name__}:{exc}", True
    if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("pending"), dict):
        return {}, "income_pending_state_schema_invalid", True
    pending: dict[str, dict] = {}
    for key, item in value["pending"].items():
        if not isinstance(key, str) or not isinstance(item, dict):
            return {}, "income_pending_state_entry_invalid", True
        since = item.get("since_ts_ms")
        if isinstance(since, bool) or not isinstance(since, int) or since <= 0:
            return {}, f"income_pending_state_timestamp_invalid:{key}", True
        pending[key] = dict(item)
    return pending, None, True


def _save_income_pending_state(path: Path, pending: dict[str, dict]) -> None:
    _write_json_state(path, {"version": 1, "pending": pending})


def _is_closed_or_confirmed_gone_trade(trade: dict) -> bool:
    """Whether an active-state row is no longer a live position exposure.

    The tracker intentionally keeps rows after exchange closure while it retries
    close journaling, Telegram delivery, or protection cleanup. Those rows are
    lifecycle work items, not open position quantity. An unresolved entry-order
    quarantine is deliberately excluded from this shortcut: an entry may still
    fill after an empty position read.
    """
    if not isinstance(trade, dict):
        return False
    if trade.get("entry_order_resolution_pending"):
        return False
    if any(bool(trade.get(flag)) for flag in (
        "closed", "protection_cleanup_only", "close_cleanup_pending",
        "close_notification_pending", "close_journal_pending",
    )):
        return True
    confirmation = trade.get("position_gone_confirmation")
    return isinstance(confirmation, dict) and str(confirmation.get("status", "")).upper() == "CONFIRMED"


def _active_local_positions(active: dict) -> tuple[dict[tuple[str, str], float], list[str]]:
    positions: dict[tuple[str, str], float] = defaultdict(float)
    errors: list[str] = []
    for event_id, trade in active.items():
        if not isinstance(trade, dict):
            errors.append(f"active_trade_not_object:{event_id}")
            continue
        if _is_closed_or_confirmed_gone_trade(trade):
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


def _post_close_lifecycle_pending(active: dict) -> list[dict]:
    """Summarize non-open rows retained for durable post-close processing."""
    pending: list[dict] = []
    for event_id, trade in active.items():
        if not isinstance(trade, dict) or trade.get("closed") or trade.get("entry_order_resolution_pending"):
            continue
        confirmation = trade.get("position_gone_confirmation")
        confirmed_gone = isinstance(confirmation, dict) and str(confirmation.get("status", "")).upper() == "CONFIRMED"
        flags = [name for name in (
            "close_journal_pending", "close_notification_pending", "close_cleanup_pending",
            "protection_cleanup_only",
        ) if bool(trade.get(name))]
        if not confirmed_gone and not flags:
            continue
        pending.append({
            "event_id": str(event_id),
            "symbol": str(trade.get("symbol") or ""),
            "direction": str(trade.get("direction") or ""),
            "pending_flags": flags,
            "position_gone_confirmed": confirmed_gone,
        })
    return pending


def _finite_nonnegative(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0.0 else None


def _pending_exit_fill_evidence(
    active: dict, key: tuple[str, str], expected_delta: float, query_state: dict[str, Any],
) -> tuple[float, list[dict], bool]:
    """Find exchange-confirmed exit fills not yet reflected in tracker remaining_qty.

    TP fills are cumulative per leg in tp_filled_qty, so only the positive
    difference from the last tracker observation is counted. SL/manual exit orders
    are considered only when the exchange says FILLED. This function never writes
    active state and never changes the strict mismatch tolerance.
    """
    candidates: list[tuple[int, str, str, str, str, float]] = []
    for event_id, trade in active.items():
        if not isinstance(trade, dict) or _is_closed_or_confirmed_gone_trade(trade):
            continue
        if _canonical_position_symbol(trade.get("symbol")) != key[0]:
            continue
        if _direction(trade.get("direction")) != key[1]:
            continue

        recorded_by_leg = trade.get("tp_filled_qty") if isinstance(trade.get("tp_filled_qty"), dict) else {}
        tp_orders = trade.get("tp_orders") if isinstance(trade.get("tp_orders"), list) else []
        for tp in tp_orders:
            if not isinstance(tp, dict) or tp.get("order_id") in (None, ""):
                continue
            leg = str(tp.get("leg") or "").strip().lower()
            if not leg:
                continue
            try:
                recorded_qty = float(recorded_by_leg.get(leg, 0.0) or 0.0)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(recorded_qty) or recorded_qty < 0.0:
                continue
            # Query the earliest target first; in the common TP1 race this both
            # gives exact evidence and avoids unnecessary API requests for TP2.
            try:
                leg_rank = int(leg.replace("tp", "")) if leg.startswith("tp") else 99
            except ValueError:
                leg_rank = 99
            candidates.append((
                leg_rank, str(event_id), str(trade.get("symbol") or key[0]),
                str(tp["order_id"]), f"TP:{leg}", recorded_qty,
            ))

        current_sl = trade.get("sl_order") if isinstance(trade.get("sl_order"), dict) else {}
        if current_sl.get("order_id") not in (None, ""):
            candidates.append((
                100, str(event_id), str(trade.get("symbol") or key[0]),
                str(current_sl["order_id"]), "SL", 0.0,
            ))
        manual_exit_id = trade.get("manual_exit_order_id")
        if manual_exit_id not in (None, ""):
            candidates.append((
                101, str(event_id), str(trade.get("symbol") or key[0]),
                str(manual_exit_id), "MANUAL_EXIT", 0.0,
            ))
        sl_history = trade.get("sl_order_history") if isinstance(trade.get("sl_order_history"), list) else []
        for item in sl_history:
            if isinstance(item, dict) and item.get("order_id") not in (None, ""):
                candidates.append((
                    102, str(event_id), str(trade.get("symbol") or key[0]),
                    str(item["order_id"]), "SL_HISTORY", 0.0,
                ))

    candidates.sort(key=lambda item: item[0])
    seen_order_ids: set[str] = set()
    total_pending = 0.0
    evidence: list[dict] = []
    budget_exhausted = False
    evidence_tolerance = max(1e-8, abs(expected_delta) * 1e-8)

    for _rank, event_id, symbol, order_id, source, recorded_qty in candidates:
        if order_id in seen_order_ids:
            continue
        seen_order_ids.add(order_id)
        if query_state["queries_used"] >= query_state["query_budget"]:
            budget_exhausted = True
            break
        elapsed = time.monotonic() - query_state["started_monotonic"]
        remaining_budget = query_state["time_budget_sec"] - elapsed
        # bingx._request clamps every HTTP timeout to at least one second. Do not
        # start a request when less than that remains, or the advertised elapsed
        # budget could be exceeded by the transport layer's minimum timeout.
        if remaining_budget < 1.0:
            budget_exhausted = True
            break
        query_timeout = min(TRACKER_FILL_ORDER_TIMEOUT_SEC, remaining_budget)
        query_state["queries_used"] += 1
        try:
            order = bingx.get_order(symbol, order_id, timeout_sec=query_timeout)
        except Exception:
            continue
        if not isinstance(order, dict) or str(order.get("status", "")).lower() != "ok":
            continue
        order_status = str(order.get("order_status") or "").upper()
        allowed_statuses = {
            "PARTIALLY_FILLED", "FILLED", "CANCELED", "CANCELLED", "EXPIRED", "REJECTED",
        } if source.startswith("TP:") else {"FILLED"}
        if order_status not in allowed_statuses:
            continue
        executed_qty = _finite_nonnegative(order.get("executed_qty"))
        if executed_qty is None:
            continue
        pending_qty = max(0.0, executed_qty - recorded_qty) if source.startswith("TP:") else executed_qty
        if pending_qty <= 1e-12:
            continue
        evidence.append({
            "order_id": order_id, "source": source, "order_status": order_status,
            "executed_qty": executed_qty,
            "previously_recorded_qty": recorded_qty if source.startswith("TP:") else 0.0,
            "unapplied_fill_qty": pending_qty, "event_id": event_id,
        })
        total_pending += pending_qty
        if abs(total_pending - expected_delta) <= evidence_tolerance:
            break
        if total_pending > expected_delta + evidence_tolerance:
            break

    return total_pending, evidence, budget_exhausted


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
        detail = (
            f"{item.get('symbol', '?')} {item.get('direction', '?')}: "
            f"local_qty={item.get('local_qty')} exchange_qty={item.get('exchange_qty')} "
            f"delta={item.get('delta_qty')} tolerance={item.get('tolerance_qty')}"
        )
        if item.get("confirmed_unapplied_fill_qty") is not None:
            detail += f" confirmed_unapplied_exit_fill_qty={item.get('confirmed_unapplied_fill_qty')}"
        return detail
    if item.get("code") == "TRACKER_EXIT_FILL_PENDING":
        evidence = item.get("fill_evidence") if isinstance(item.get("fill_evidence"), list) else []
        order_details = "; ".join(
            f"{row.get('source')}#{row.get('order_id')} qty={row.get('unapplied_fill_qty')}"
            for row in evidence if isinstance(row, dict)
        )
        base = (
            f"{item.get('symbol', '?')} {item.get('direction', '?')}: "
            f"confirmed_unapplied_exit_fill_qty={item.get('confirmed_unapplied_fill_qty')}"
        )
        return f"{base}; orders={order_details}" if order_details else base
    if item.get("code") == "DAILY_INCOME_QUERY_INCOMPLETE":
        return (
            f"{item.get('symbol', '?')}: {item.get('detail') or 'income history incomplete'}"
        )
    if item.get("code") == "DATA_ERROR_CLOSES_UNRESOLVED":
        trades = item.get("trades") if isinstance(item.get("trades"), list) else []
        summaries = []
        for trade in trades[:5]:
            if not isinstance(trade, dict):
                continue
            recovery = trade.get("fill_history_recovery") if isinstance(trade.get("fill_history_recovery"), dict) else {}
            symbol = trade.get("symbol") or "?"
            direction = trade.get("direction") or "?"
            event_id = trade.get("event_id") or "?"
            reason = trade.get("exit_reason") or "DATA_ERROR"
            last_error = recovery.get("last_error") or trade.get("data_error") or trade.get("error_code") or "recovery_not_attempted"
            summaries.append(f"{symbol} {direction} event={event_id} reason={reason} recovery={str(last_error)[:180]}")
        base = f"{item.get('count', len(trades))} unresolved DATA_ERROR close(s)"
        return base + ("; " + " | ".join(summaries) if summaries else "")
    return str(item.get("detail") or item.get("code") or "unknown")


def run_reconciliation(*, now_ms: int | None = None, data_dir: Path | None = None, notify: bool = True) -> dict:
    """Run one bounded, read-only reconciliation and return a JSON-safe report."""
    global ACTIVE_TRADES_PATH, TRADES_PATH, HEALTH_PATH
    if data_dir is not None:
        data_dir = Path(data_dir)
        active_path, trades_path, health_path = data_dir / "active_trades.json", data_dir / "trades.jsonl", data_dir / "health.jsonl"
        alert_state_path = data_dir / "reconciliation_alert_state.json"
        runtime_marker_path = data_dir / "runtime_state_initialized.json"
        income_pending_path = data_dir / "reconciliation_income_pending.json"
    else:
        active_path, trades_path, health_path = ACTIVE_TRADES_PATH, TRADES_PATH, HEALTH_PATH
        alert_state_path = ALERT_STATE_PATH
        runtime_marker_path = RUNTIME_STATE_MARKER_PATH
        income_pending_path = INCOME_PENDING_STATE_PATH
    now_ms = int(now_ms or time.time() * 1000)
    utc_now = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc)
    day_start = int(utc_now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    report: dict[str, Any] = {
        "record_type": "DAILY_RECONCILIATION",
        "ts": now_ms,
        "build_version": APP_VERSION,
        "utc_day": utc_now.date().isoformat(),
        "mode": "READ_ONLY",
        "status": "OK",
        "checks": {},
        "discrepancies": [],
        "notes": [
            "Daily income totals are account+symbol aggregates and are not per-position attribution.",
            "READ_ONLY means no exchange orders or positions are modified; local reconciliation metadata may be persisted.",
        ],
    }

    active, active_error = _load_json_object(active_path)
    local_positions: dict[tuple[str, str], float] = {}
    active_errors: list[str] = []

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
        active, active_error, runtime_state_check = _ensure_runtime_state_initialized(
            active_path=active_path,
            trades_path=trades_path,
            health_path=health_path,
            marker_path=runtime_marker_path,
            active=active,
            active_error=active_error,
            exchange_positions=exchange_positions,
            exchange_errors=exchange_errors,
            now_ms=now_ms,
        )
        report["checks"]["runtime_state_initialization"] = runtime_state_check
        if runtime_state_check.get("status") == "ERROR":
            report["discrepancies"].append({
                "level": "WARNING", "code": "RUNTIME_STATE_INITIALIZATION_UNAVAILABLE",
                "detail": str(runtime_state_check.get("error") or "runtime_state_initialization_failed"),
            })
        if active is None:
            report["checks"]["local_active_state"] = {"status": "UNKNOWN", "error": active_error}
            report["discrepancies"].append({
                "level": "WARNING", "code": "LOCAL_ACTIVE_STATE_UNAVAILABLE",
                "detail": active_error or "active_state_unavailable",
            })
        else:
            local_positions, active_errors = _active_local_positions(active)
            post_close_pending = _post_close_lifecycle_pending(active)
            report["checks"]["local_active_state"] = {
                "status": "OK" if not active_errors else "WARNING",
                "active_event_records": len(active),
                "active_position_keys": len(local_positions),
                "post_close_lifecycle_pending_records": len(post_close_pending),
                "parse_errors": active_errors,
            }
            for detail in active_errors:
                report["discrepancies"].append({"level": "WARNING", "code": "LOCAL_ACTIVE_STATE_INVALID", "detail": detail})
            if post_close_pending:
                report["discrepancies"].append({
                    "level": "WARNING", "code": "LOCAL_CLOSE_LIFECYCLE_PENDING",
                    "detail": (
                        f"{len(post_close_pending)} non-open trade record(s) remain for close journal, "
                        "notification, or protection-cleanup retries; these rows are excluded from live position totals."
                    ),
                    "records": post_close_pending[:20],
                })
        catalog_status = "NOT_NEEDED"
        catalog_count = 0
        catalog_error = None
        identity_unresolved = False
        identity_unresolved_keys: set[tuple[str, str]] = set()
        has_non_ascii_identity = any(
            not str(trade.get("symbol") or "").strip().isascii()
            for trade in (active.values() if isinstance(active, dict) else [])
            if isinstance(trade, dict) and not _is_closed_or_confirmed_gone_trade(trade)
        ) or any(
            not str(row.get("symbol") or "").strip().isascii()
            for row in exchange_rows if isinstance(row, dict)
        )
        # Only refresh the public catalog when local and exchange identity keys
        # do not line up.  This avoids adding a network request to normal runs
        # while resolving display-name aliases (for example 牛来 → NIULAI-USDT)
        # in the separate reconciliation process.  The catalog is fetched at
        # most once and then shared by position and protection normalization.
        if active is not None and not active_errors and not exchange_errors and (
            has_non_ascii_identity
            or (local_positions and exchange_positions and set(local_positions) != set(exchange_positions))
        ):
            catalog_status = "UNKNOWN"
            try:
                contract_catalog = bingx.contracts()
                catalog_count = len(contract_catalog) if isinstance(contract_catalog, dict) else 0
                if catalog_count:
                    local_positions, active_errors = _active_local_positions(active)
                    exchange_positions, exchange_errors = _exchange_positions(exchange_rows)
                    catalog_status = "LOADED"
                    report["checks"]["local_active_state"].update({
                        "status": "OK" if not active_errors else "WARNING",
                        "active_position_keys": len(local_positions),
                        "parse_errors": active_errors,
                    })
                    report["checks"]["exchange_positions"].update({
                        "status": "OK" if not exchange_errors else "WARNING",
                        "active_position_keys": len(exchange_positions),
                        "parse_errors": exchange_errors,
                    })
                else:
                    catalog_error = "empty_or_unavailable_contract_catalog"
            except Exception as exc:
                catalog_error = f"{type(exc).__name__}:{exc}"
            unmatched_keys = set(local_positions) ^ set(exchange_positions)
            non_ascii_keys = {
                key for key in (set(local_positions) | set(exchange_positions))
                if not key[0].isascii()
            }
            # Only a non-Latin display-name identity is inherently ambiguous
            # when the catalog cannot be loaded.  An unmatched ASCII symbol is
            # not presumed to be an alias: retain the strict quantity check so
            # a real missing/extra position cannot be hidden by a catalog outage.
            identity_unresolved = bool(non_ascii_keys)
            identity_unresolved_keys = set(non_ascii_keys)
            if non_ascii_keys:
                # Both sides of a possible displayName/API-symbol pair are
                # unknown until canonicalization succeeds; do not declare the
                # API-side order set missing protection merely because the
                # display-name side could not be mapped.
                identity_unresolved_keys.update(unmatched_keys)
            elif unmatched_keys and catalog_count == 0:
                report["discrepancies"].append({
                    "level": "WARNING", "code": "POSITION_SYMBOL_CATALOG_UNAVAILABLE",
                    "detail": "Contract catalog could not be loaded; unmatched ASCII symbols remain subject to strict position identity/quantity reconciliation.",
                    "catalog_error": catalog_error,
                    "unmatched_keys": [f"{symbol}:{direction}" for symbol, direction in sorted(unmatched_keys)],
                })
            if catalog_count and not identity_unresolved_keys and not unmatched_keys:
                catalog_status = "OK"
            report["checks"]["bingx_contract_catalog"] = {
                "status": catalog_status, "contracts": catalog_count,
                "source": "single_load_on_identity_mismatch", "error": catalog_error,
            }
            if identity_unresolved:
                report["discrepancies"].append({
                    "level": "WARNING", "code": "POSITION_SYMBOL_IDENTITY_UNRESOLVED",
                    "detail": "A display-name identity remains unresolved after the single catalog load; unmatched non-Latin identities were not classified as quantity mismatches.",
                    "catalog_error": catalog_error,
                    "unmatched_keys": [f"{symbol}:{direction}" for symbol, direction in sorted(unmatched_keys)],
                })
        else:
            report["checks"]["bingx_contract_catalog"] = {
                "status": "NOT_NEEDED", "contracts": 0,
                "source": "no_identity_mismatch",
            }
        tracker_fill_query_state = {
            "queries_used": 0,
            "query_budget": MAX_TRACKER_FILL_ORDER_QUERIES,
            "started_monotonic": time.monotonic(),
            "time_budget_sec": TRACKER_FILL_ORDER_TIME_BUDGET_SEC,
        }
        fill_evidence_by_key: dict[tuple[str, str], dict] = {}
        budget_exhausted_keys: set[tuple[str, str]] = set()
        if active is not None and not active_errors and not exchange_errors:
            # Query only keys already discrepant in the first position snapshot.
            # These reads are bounded by both a request-count and elapsed-time budget.
            for key in sorted(set(local_positions) | set(exchange_positions)):
                local_qty = local_positions.get(key, 0.0)
                exchange_qty = exchange_positions.get(key, 0.0)
                tolerance = max(1e-8, exchange_qty * 0.01, local_qty * 0.01)
                expected_exit_delta = local_qty - exchange_qty
                if abs(expected_exit_delta) <= tolerance or expected_exit_delta <= 0:
                    continue
                if MAX_TRACKER_FILL_ORDER_QUERIES <= 0:
                    continue
                confirmed_fill_qty, fill_evidence, budget_exhausted = _pending_exit_fill_evidence(
                    active, key, expected_exit_delta, tracker_fill_query_state,
                )
                if budget_exhausted:
                    budget_exhausted_keys.add(key)
                if fill_evidence:
                    fill_evidence_by_key[key] = {
                        "confirmed_fill_qty": confirmed_fill_qty,
                        "evidence": fill_evidence,
                        "budget_exhausted": budget_exhausted,
                        "expected_delta": expected_exit_delta,
                    }

        # Order details and position snapshots are separate exchange reads. A fill
        # may land between them, so a matching order total alone is not enough to
        # downgrade a CRITICAL mismatch. Take one fresh account-wide position read
        # after exact fill evidence and require the residual position to agree.
        exact_fill_candidates = {
            key: item for key, item in fill_evidence_by_key.items()
            if abs(item["confirmed_fill_qty"] - item["expected_delta"])
            <= max(1e-8, abs(item["expected_delta"]) * 1e-8)
        }
        fill_refresh_status = "NOT_NEEDED"
        fill_refresh_error = None
        if exact_fill_candidates:
            try:
                refreshed_rows = bingx.get_positions(timeout_sec=3.0, retryable=False)
                refreshed_positions, refreshed_errors = _exchange_positions(refreshed_rows)
                if refreshed_errors:
                    fill_refresh_status = "ERROR"
                    fill_refresh_error = ";".join(refreshed_errors[:5])
                else:
                    exchange_positions = refreshed_positions
                    exchange_errors = []
                    fill_refresh_status = "OK"
                    report["checks"]["exchange_positions"].update({
                        "post_fill_refresh": "OK",
                        "post_fill_refresh_api_rows": len(refreshed_rows),
                        "active_position_keys": len(exchange_positions),
                    })
            except Exception as exc:
                fill_refresh_status = "ERROR"
                fill_refresh_error = f"{type(exc).__name__}:{exc}"
            if fill_refresh_status != "OK":
                report["checks"]["exchange_positions"]["post_fill_refresh"] = "ERROR"
                report["checks"]["exchange_positions"]["post_fill_refresh_error"] = fill_refresh_error

        if active is not None and not active_errors and not exchange_errors:
            comparison_keys = set(local_positions) | set(exchange_positions)
            if identity_unresolved:
                comparison_keys = set(local_positions) & set(exchange_positions)
            for key in sorted(comparison_keys):
                local_qty = local_positions.get(key, 0.0)
                exchange_qty = exchange_positions.get(key, 0.0)
                tolerance = max(1e-8, exchange_qty * 0.01, local_qty * 0.01)
                if abs(local_qty - exchange_qty) <= tolerance:
                    continue

                fill_info = fill_evidence_by_key.get(key)
                if fill_info:
                    confirmed_fill_qty = fill_info["confirmed_fill_qty"]
                    fill_evidence = fill_info["evidence"]
                    expected_after_fills = max(0.0, local_qty - confirmed_fill_qty)
                    evidence_tolerance = max(1e-8, abs(expected_after_fills) * 1e-8)
                    refresh_confirms_residual = (
                        fill_refresh_status == "OK"
                        and abs(exchange_qty - expected_after_fills) <= evidence_tolerance
                    )
                    if refresh_confirms_residual and key in exact_fill_candidates:
                        report["discrepancies"].append({
                            "level": "WARNING", "code": "TRACKER_EXIT_FILL_PENDING",
                            "symbol": key[0], "direction": key[1],
                            "local_qty": local_qty, "exchange_qty": exchange_qty,
                            "confirmed_unapplied_fill_qty": confirmed_fill_qty,
                            "fill_evidence": fill_evidence,
                            "detail": "Confirmed order executions explain the full local/exchange delta, and a fresh exchange position read confirms the expected residual quantity. The next tracker pass should persist the fill-adjusted remaining_qty.",
                        })
                        continue

                mismatch = {
                    "level": "CRITICAL", "code": "ACTIVE_POSITION_QTY_OR_IDENTITY_MISMATCH",
                    "symbol": key[0], "direction": key[1], "local_qty": local_qty,
                    "exchange_qty": exchange_qty, "delta_qty": exchange_qty - local_qty,
                    "tolerance_qty": tolerance,
                }
                if fill_info:
                    mismatch["confirmed_unapplied_fill_qty"] = fill_info["confirmed_fill_qty"]
                    mismatch["fill_evidence"] = fill_info["evidence"]
                    mismatch["detail"] = "Confirmed order fills do not fully explain the latest exchange position quantity; the unexplained remainder remains CRITICAL."
                    if fill_refresh_status != "OK":
                        mismatch["position_refresh_error"] = fill_refresh_error or "post_fill_position_refresh_unavailable"
                elif key in budget_exhausted_keys:
                    mismatch["detail"] = "Position mismatch remains CRITICAL; bounded tracker-order verification budget was exhausted before the delta could be explained."
                report["discrepancies"].append(mismatch)

        report["checks"]["tracker_exit_fill_reconciliation"] = {
            "status": "NOT_NEEDED" if tracker_fill_query_state["queries_used"] == 0 else "CHECKED",
            "order_queries_used": tracker_fill_query_state["queries_used"],
            "order_query_budget": tracker_fill_query_state["query_budget"],
            "order_query_timeout_sec": TRACKER_FILL_ORDER_TIMEOUT_SEC,
            "order_query_time_budget_sec": TRACKER_FILL_ORDER_TIME_BUDGET_SEC,
            "position_refresh_status": fill_refresh_status,
            "position_refresh_error": fill_refresh_error,
        }
    except Exception as exc:
        exchange_positions = {}
        report["checks"]["exchange_positions"] = {"status": "ERROR", "error": f"{type(exc).__name__}:{exc}"}
        report["discrepancies"].append({"level": "CRITICAL", "code": "EXCHANGE_POSITION_QUERY_FAILED", "detail": str(exc)})
        report["checks"]["runtime_state_initialization"] = {
            "status": "UNKNOWN", "reason": "exchange_snapshot_failed_no_bootstrap_attempted",
        }
        if active is None:
            report["checks"]["local_active_state"] = {"status": "UNKNOWN", "error": active_error or "file_missing"}
            report["discrepancies"].append({
                "level": "WARNING", "code": "LOCAL_ACTIVE_STATE_UNAVAILABLE",
                "detail": active_error or "active_state_unavailable",
            })
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
            if (symbol, direction) in identity_unresolved_keys:
                protection_results[key] = {
                    "status": "UNKNOWN", "error": "contract_display_name_identity_unresolved",
                    "exchange_qty": exchange_qty, "source": "account_open_orders_snapshot",
                }
                continue
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
    unresolved_data_errors = []
    for row in journal:
        if row.get("record_type") != "TRADE_CLOSE":
            continue
        if str(row.get("exit_reason") or "").upper() != "DATA_ERROR" and not bool(row.get("data_error")):
            continue
        recovery = row.get("fill_history_recovery") if isinstance(row.get("fill_history_recovery"), dict) else {}
        unresolved_data_errors.append({
            "event_id": row.get("event_id"),
            "symbol": row.get("symbol"),
            "direction": row.get("direction"),
            "closed_ts": row.get("closed_ts"),
            "exit_reason": row.get("exit_reason"),
            "data_error": row.get("data_error"),
            "error_code": row.get("error_code"),
            "exit_order_ids": row.get("exit_order_ids", []),
            "entry_order_id": row.get("entry_order_id"),
            "fill_history_recovery": {
                "status": recovery.get("status", "NOT_ATTEMPTED"),
                "attempts": recovery.get("attempts", 0),
                "last_error": recovery.get("last_error"),
                "next_retry_ts": recovery.get("next_retry_ts"),
            },
        })
    if unresolved_data_errors:
        report["discrepancies"].append({
            "level": "CRITICAL", "code": "DATA_ERROR_CLOSES_UNRESOLVED",
            "count": len(unresolved_data_errors), "trades": unresolved_data_errors[:25],
            "detail": "One or more historical trade closes still lack a fill-confirmed exit outcome. This remains CRITICAL across UTC days until fill recovery succeeds; missing fees/funding alone do not trigger it.",
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
    account_wide_income: dict[str, Any] | None = None
    account_wide_income_attempted = False
    income_pending, income_pending_error, income_pending_file_exists = _load_income_pending_state(income_pending_path)
    income_pending_dirty = False
    pending_income_symbols: set[str] = set()
    expired_income_symbols: set[str] = set()
    if income_pending_error:
        report["checks"]["daily_income_pending_state"] = {
            "status": "ERROR", "error": income_pending_error,
        }
        report["discrepancies"].append({
            "level": "WARNING", "code": "DAILY_INCOME_PENDING_STATE_UNAVAILABLE",
            "detail": income_pending_error,
        })
    income_pending_key_prefix = f"{utc_now.date().isoformat()}|"
    # Bound persistent metadata. Expired UTC-day entries are no longer queried
    # by this daily report; prior reports remain in health.jsonl.
    if income_pending_error is None:
        stale_keys = [key for key in income_pending if not key.startswith(income_pending_key_prefix)]
        for key in stale_keys:
            del income_pending[key]
            income_pending_dirty = True
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
        # Some BingX VST responses return code=0 with data=null when the
        # symbol-filtered user/income query has no symbol-scoped payload. The
        # API supports an account-wide query without `symbol`; try it once per
        # reconciliation run, then strictly filter by each row's actual symbol.
        # A null/invalid account-wide response remains UNKNOWN, never zero.
        if not isinstance(income, dict) or income.get("status") != "ok" or income.get("complete") is not True:
            if bingx.is_income_history_null_data_shape(income):
                if not account_wide_income_attempted:
                    account_wide_income_attempted = True
                    try:
                        account_wide_income = bingx.get_income_history(
                            None, day_start, max(day_start + 1, now_ms), limit=1000,
                            timeout_sec=INCOME_TIMEOUT_SEC,
                            max_range_requests=INCOME_REQUEST_BUDGET,
                        )
                    except Exception as exc:
                        account_wide_income = {
                            "status": "error", "complete": False,
                            "error": f"account_wide_income_exception:{type(exc).__name__}:{exc}",
                            "records": [],
                        }
                if isinstance(account_wide_income, dict) and account_wide_income.get("status") == "ok" and account_wide_income.get("complete") is True:
                    income = bingx.filter_income_history_by_symbol(account_wide_income, symbol)
                elif isinstance(account_wide_income, dict):
                    fallback_error = account_wide_income.get("error", "account_wide_income_incomplete")
                    fallback_diagnostics = {
                        key: account_wide_income.get(key) for key in (
                            "response_type", "response_code", "response_msg",
                            "response_keys", "response_data_type", "shape_fingerprint",
                        ) if account_wide_income.get(key) is not None
                    }
                    if isinstance(income, dict):
                        income = dict(income)
                        income["account_wide_fallback_error"] = str(fallback_error)
                        income["account_wide_fallback_diagnostics"] = fallback_diagnostics
            if isinstance(income, dict) and income.get("status") == "ok" and income.get("complete") is True:
                pass
            else:
                detail = str(income.get("error", "incomplete_income_response")) if isinstance(income, dict) else "invalid_income_response"
                shape_fingerprint = income.get("shape_fingerprint") if isinstance(income, dict) else None
                shape_summary = income.get("shape_summary") if isinstance(income, dict) else None
                if shape_fingerprint:
                    detail += f"; shape_fingerprint={shape_fingerprint}"
                if shape_summary:
                    detail += f"; response_shape={shape_summary}"
                response_diagnostics = {
                    key: income.get(key) for key in (
                        "response_type", "response_code", "response_msg",
                        "response_keys", "response_data_type",
                    ) if isinstance(income, dict) and income.get(key) is not None
                }
                if response_diagnostics:
                    detail += "; api_response=" + json.dumps(
                        response_diagnostics, ensure_ascii=False, separators=(",", ":"),
                    )[:700]
                if isinstance(income, dict) and income.get("account_wide_fallback_error"):
                    detail += f"; account_wide_fallback={income.get('account_wide_fallback_error')}"
                    fallback_diagnostics = income.get("account_wide_fallback_diagnostics")
                    if isinstance(fallback_diagnostics, dict) and fallback_diagnostics:
                        detail += "; account_wide_api_response=" + json.dumps(
                            fallback_diagnostics, ensure_ascii=False, separators=(",", ":"),
                        )[:500]
                pending_key = f"{utc_now.date().isoformat()}|{symbol}"
                is_transient_null = bingx.is_income_history_null_data_shape(income)
                if is_transient_null and income_pending_error is None:
                    pending_entry = income_pending.get(pending_key)
                    if pending_entry is None:
                        pending_entry = {
                            "since_ts_ms": int(now_ms),
                            "first_error": "income_history_unexpected_data_shape",
                        }
                        income_pending[pending_key] = pending_entry
                        income_pending_dirty = True
                    since_ts_ms = int(pending_entry["since_ts_ms"])
                    age_sec = max(0.0, (now_ms - since_ts_ms) / 1000.0)
                    if age_sec < INCOME_NULL_GRACE_SEC:
                        pending_income_symbols.add(symbol)
                        income_summaries[symbol] = {
                            "status": "PENDING",
                            "error": detail,
                            "attempts": attempts,
                            "pending_since_ts_ms": since_ts_ms,
                            "pending_age_sec": round(age_sec, 3),
                            "grace_sec": INCOME_NULL_GRACE_SEC,
                            "accounting_value": "UNKNOWN_NOT_ZERO",
                            "shape_fingerprint": shape_fingerprint,
                            "response_diagnostics": response_diagnostics,
                        }
                        continue
                    expired_income_symbols.add(symbol)
                    detail += f"; null_data_pending_for_sec={int(age_sec)}; grace_sec={int(INCOME_NULL_GRACE_SEC)}"
                else:
                    if income_pending_error is None and pending_key in income_pending:
                        del income_pending[pending_key]
                        income_pending_dirty = True
                income_summaries[symbol] = {
                    "status": "ERROR", "error": detail, "attempts": attempts,
                    "shape_fingerprint": shape_fingerprint, "shape_summary": shape_summary,
                    "response_diagnostics": response_diagnostics,
                    "accounting_value": "UNKNOWN_NOT_ZERO",
                }
                report["discrepancies"].append({
                    "level": "WARNING", "code": "DAILY_INCOME_QUERY_INCOMPLETE",
                    "symbol": symbol, "detail": f"{detail} (attempts={attempts})",
                    "response_diagnostics": response_diagnostics,
                })
                continue
        pending_key = f"{utc_now.date().isoformat()}|{symbol}"
        if income_pending_error is None and pending_key in income_pending:
            del income_pending[pending_key]
            income_pending_dirty = True
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
        totals["source_scope"] = (
            "ACCOUNT_WIDE_FILTERED_BY_SYMBOL_DAILY_AGGREGATE_NOT_PER_POSITION"
            if income.get("query_scope") == "ACCOUNT_WIDE_FILTERED_BY_ROW_SYMBOL"
            else "ACCOUNT_SYMBOL_DAILY_AGGREGATE_NOT_PER_POSITION"
        )
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
    income_pending_write_error = None
    if income_pending_error is None and (income_pending_dirty or not income_pending_file_exists):
        try:
            _save_income_pending_state(income_pending_path, income_pending)
        except OSError as exc:
            income_pending_write_error = f"income_pending_state_write_error:{type(exc).__name__}:{exc}"
            report["discrepancies"].append({
                "level": "WARNING", "code": "DAILY_INCOME_PENDING_STATE_UNAVAILABLE",
                "detail": income_pending_write_error,
            })
            for symbol in sorted(pending_income_symbols):
                summary = income_summaries.get(symbol)
                if isinstance(summary, dict) and summary.get("status") == "PENDING":
                    summary["status"] = "ERROR"
                    summary["accounting_value"] = "UNKNOWN_NOT_ZERO"
                    summary["error"] = f"{summary.get('error')}; grace_state_not_persisted"
                    report["discrepancies"].append({
                        "level": "WARNING", "code": "DAILY_INCOME_QUERY_INCOMPLETE",
                        "symbol": symbol,
                        "detail": f"{summary['error']} (pending grace state could not be persisted)",
                    })
    if income_pending_error is None and income_pending_write_error is None:
        pending_state_status = (
            "WARNING" if expired_income_symbols else "PENDING" if pending_income_symbols else "OK"
        )
        report["checks"]["daily_income_pending_state"] = {
            "status": pending_state_status,
            "pending_symbols": sorted(pending_income_symbols),
            "expired_symbols": sorted(expired_income_symbols),
            "grace_sec": INCOME_NULL_GRACE_SEC,
            "note": "Successful-envelope null data remains UNKNOWN, never zero. Transient null responses are pending during the bounded grace period; persistent null responses remain warnings.",
        }
    report["checks"]["daily_exchange_income_aggregates"] = {
        "status": (
            "WARNING" if any(x.get("status") == "ERROR" for x in income_summaries.values())
            else "PENDING" if any(x.get("status") == "PENDING" for x in income_summaries.values())
            else "OK" if income_summaries else "NOT_NEEDED"
        ),
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
            lines.append("Exchange read-only check; persists local reconciliation metadata only. It does not place/cancel orders or block a later run.")
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
