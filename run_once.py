from __future__ import annotations

import json
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import requests

from event_engine.coinalyze import fetch_data
from event_engine.bingx import (
    API_KEY,
    SECRET_KEY,
    BASE_URL,
    refresh_contracts,
    get_contract,
    to_bx_symbol,
    fetch_klines,
    open_market,
    wait_for_position_fill_directional,
    get_positions,
    get_open_protection_directional,
    ensure_directional_protection,
    has_open_position,
    emergency_close_position,
    get_position_directional,
)
from event_engine.signals import (
    add_cvd,
    detect_divergences,
    detect_squeeze_release,
    detect_liquidation_squeeze,
    attach_oi_series,
    build_15m_trigger,
    diagnose_15m_trigger,
    check_btc_regime,
)
from event_engine.telegram import send as send_tg, format_signal
from event_engine.shadow import append_shadow_health
from event_engine.tracker import (
    update_active_trades,
    register_active_trade,
    update_active_trade_protection,
    _load_active_trades,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
log = logging.getLogger("event_engine")

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if pd.notna(x) and abs(x) != float("inf") else default
    except (TypeError, ValueError):
        return default

DATA = Path("data")
DATA.mkdir(exist_ok=True)

EVENTS = DATA / "events.jsonl"
TRADES = DATA / "trades.jsonl"
ACTIONS = DATA / "actions.jsonl"
HEALTH = DATA / "health.jsonl"
TIMEFRAME_STATE = DATA / "timeframe_scan_state.json"
EVENT_CACHE = DATA / "recent_event_cache.json"
# BingX Kline endpoint is rate-limited per IP; serialize heavyweight scan requests.
BAR_CLOSE_GRACE_MIN = float(os.environ.get("BAR_CLOSE_GRACE_MIN", "2"))

MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "0"))
MIN_VOL = float(os.environ.get("MIN_VOLUME_24H", "25000000"))

# Установлен порог Open Interest $10 000 000 по вашему запросу
MIN_OI = float(os.environ.get("MIN_OPEN_INTEREST", "10000000"))

EXECUTION_ENABLED = os.environ.get("EXECUTION_ENABLED", "false").lower() == "true"
REQUIRE_CVD = os.environ.get("REQUIRE_CVD_CONFIRMATION", "false").lower() == "true"
CVD_MIN_CONFIRMATION = float(os.environ.get("MIN_CVD24_CONFIRMATION", "55"))
REQUIRE_TRIGGER = os.environ.get("REQUIRE_15M_TRIGGER", "true").lower() == "true"
MAX_AGE = int(os.environ.get("MAX_EVENT_AGE_MIN", "90"))
MAX_TRIGGER_DELAY = float(os.environ.get("MAX_TRIGGER_DELAY_MIN", "30"))
MAX_ENTRY_DRIFT_PCT = float(os.environ.get("MAX_ENTRY_DRIFT_PCT", "2.00"))
MAX_SQUEEZE_ENTRY_DRIFT_PCT = float(os.environ.get("MAX_SQUEEZE_ENTRY_DRIFT_PCT", "3.00"))
MIN_SCORE = float(os.environ.get("MIN_SETUP_SCORE", "60"))
MIN_SHORT_SCORE = float(os.environ.get("MIN_SHORT_SETUP_SCORE", "85"))
MAX_HOT_OI_CHG24_PCT = float(os.environ.get("MAX_HOT_OI_CHG24_PCT", "50"))
HARD_HOT_OI_CHG24_PCT = float(os.environ.get("HARD_HOT_OI_CHG24_PCT", "75"))
HOT_OI_SCORE_PENALTY = float(os.environ.get("HOT_OI_SCORE_PENALTY", "15"))
SYMBOL_MAX_CONSECUTIVE_LOSSES = int(os.environ.get("SYMBOL_MAX_CONSECUTIVE_LOSSES", "3"))
SYMBOL_QUARANTINE_MIN = float(os.environ.get("SYMBOL_QUARANTINE_MIN", "360"))
MAX_TRADES = int(os.environ.get("MAX_TRADES_PER_CYCLE", "3"))
# Prevent rapid re-entry/churn on the same instrument, including opposite-direction flips.
SYMBOL_ENTRY_COOLDOWN_MIN = float(os.environ.get("SYMBOL_ENTRY_COOLDOWN_MIN", "15"))
SQUEEZE_SYMBOL_ENTRY_COOLDOWN_MIN = float(os.environ.get("SQUEEZE_SYMBOL_ENTRY_COOLDOWN_MIN", "45"))
# Funding values are in percentage points as parsed from Coinalyze
# (e.g. 0.05 == +0.05%, -0.05 == -0.05%).
MAX_SHORT_SQUEEZE_ADVERSE_FUNDING = float(os.environ.get("MAX_SHORT_SQUEEZE_ADVERSE_FUNDING", "-0.10"))
MAX_LONG_SQUEEZE_ADVERSE_FUNDING = float(os.environ.get("MAX_LONG_SQUEEZE_ADVERSE_FUNDING", "0.10"))
EXECUTION_MODE = os.environ.get("EXECUTION_MODE", os.environ.get("BINGX_ENV", "vst"))
POSITION_MODE = os.environ.get("BINGX_POSITION_MODE", "HEDGE").strip().upper()


def _validate_execution_config() -> tuple[bool, str]:
    if not EXECUTION_ENABLED:
        return True, "EXECUTION_DISABLED"
    if not API_KEY or not SECRET_KEY:
        return False, "BINGX credentials are missing while EXECUTION_ENABLED=true"
    mode = str(EXECUTION_MODE or "vst").strip().lower()
    base = str(BASE_URL or "").strip().lower()
    if mode in {"vst", "test", "demo", "simulated"} and "open-api-vst." not in base:
        return False, f"EXECUTION_MODE={mode} requires BingX VST base URL, got {BASE_URL!r}"
    if POSITION_MODE != "HEDGE":
        return False, f"This engine requires BINGX_POSITION_MODE=HEDGE, got {POSITION_MODE!r}"
    if mode in {"live", "prod", "production", "prod-live"} and os.environ.get("ALLOW_LIVE_TRADING", "false").lower() != "true":
        return False, "Live execution requires explicit ALLOW_LIVE_TRADING=true"
    return True, "OK"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _save_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _completed_bucket(interval_ms: int, now_ms: int, grace_min: float) -> int:
    adjusted = max(0, int(now_ms - grace_min * 60_000))
    return (adjusted // interval_ms) - 1


def _load_recent_successful_entries(path: Path, now_ms: int, cooldown_min: float) -> dict[str, int]:
    """Return most recent successful TRADE_OPEN timestamp per symbol.

    Only real opened states count; OPEN_FAILED / protection failures without a
    confirmed position are deliberately ignored so a transient API failure does
    not permanently suppress a symbol.
    """
    if cooldown_min <= 0 or not path.exists():
        return {}
    cutoff = now_ms - int(cooldown_min * 60_000)
    latest: dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("record_type") == "TRADE_CLOSE":
                    symbol = str(obj.get("symbol") or "").strip().upper()
                    if not symbol:
                        continue
                    closed_ts = int(_safe_float(obj.get("closed_ts"), 0.0))
                    if closed_ts >= cutoff and closed_ts <= now_ms:
                        latest[symbol] = max(latest.get(symbol, 0), closed_ts)
                    continue

                if obj.get("record_type") != "TRADE_OPEN":
                    continue
                result = obj.get("result") if isinstance(obj.get("result"), dict) else {}
                execution = obj.get("execution") if isinstance(obj.get("execution"), dict) else {}
                status = str(execution.get("status") or result.get("status") or "").lower()
                position = result.get("position") if isinstance(result.get("position"), dict) else {}
                qty = _safe_float(position.get("positionAmt"), 0.0)
                if status not in {
                    "opened_protected", "opened_protection_check_required", "opened_protection_failed",
                } or qty <= 0:
                    continue
                symbol = str(obj.get("symbol") or "").strip().upper()
                if not symbol:
                    continue
                ts = int(_safe_float(obj.get("ts"), 0.0))
                if ts >= cutoff and ts <= now_ms:
                    latest[symbol] = max(latest.get(symbol, 0), ts)
    except OSError as exc:
        log.warning("[EXECUTION] Failed to inspect recent entries: %s", exc)
    return latest


def _symbol_on_cooldown(symbol: str, latest_entries: dict[str, int], now_ms: int, cooldown_min: float) -> bool:
    key = str(symbol or "").strip().upper().replace("-USDT", "")
    ts = latest_entries.get(key) or latest_entries.get(str(symbol or "").strip().upper())
    if not ts or cooldown_min <= 0:
        return False
    return now_ms - ts < int(cooldown_min * 60_000)


def _mark_local_position_state(
    current_open_positions: dict[tuple[str, str], bool],
    current_positions: dict[tuple[str, str], dict],
    position: dict | None,
    symbol: str,
    direction: str,
) -> None:
    """Immediately update the in-cycle position snapshot after a successful fill."""
    if isinstance(position, dict):
        bx_symbol = str(position.get("symbol") or to_bx_symbol(symbol) or "").upper()
        qty = _safe_float(position.get("positionAmt"), 0.0)
        if bx_symbol and qty > 0:
            key = (bx_symbol, str(direction).upper())
            current_open_positions[key] = True
            current_positions[key] = dict(position)


def _event_is_fresh(ev: dict, now_ms: int, max_age_min: int) -> bool:
    try:
        ts = int(ev.get("timestamps", {}).get("detected_at_ts", 0) or 0)
    except (TypeError, ValueError):
        return False
    age_min = (now_ms - ts) / 60_000.0
    return 0 <= age_min <= max_age_min


def _load_cached_events() -> list[dict]:
    data = _load_json(EVENT_CACHE, {})
    events = data.get("events", []) if isinstance(data, dict) else []
    return [e for e in events if isinstance(e, dict) and e.get("event_id")]


def _load_timeframe_scan_state() -> dict:
    """Load per-symbol/per-timeframe scan state, migrating old global buckets."""
    raw = _load_json(TIMEFRAME_STATE, {})
    if not isinstance(raw, dict):
        return {"symbols": {}}
    symbols = raw.get("symbols")
    if isinstance(symbols, dict):
        return {"version": 2, "symbols": symbols}
    # Legacy format used one bucket for the whole universe. Do not copy it to
    # symbols: doing so would hide newly discovered symbols. Start them from
    # scratch once, then persist their own last-scanned closed bar.
    return {"version": 2, "symbols": {}}


def _save_timeframe_scan_state(state: dict) -> None:
    _save_json_atomic(TIMEFRAME_STATE, state)


def _symbol_scan_due(state: dict, symbol: str, timeframe: str, completed_bucket: int) -> bool:
    symbols = state.setdefault("symbols", {})
    rec = symbols.get(symbol)
    if not isinstance(rec, dict):
        return True
    last = rec.get(timeframe)
    try:
        return int(last) < completed_bucket
    except (TypeError, ValueError):
        return True


def _mark_symbol_scanned(state: dict, symbol: str, timeframe: str, completed_bucket: int) -> None:
    symbols = state.setdefault("symbols", {})
    rec = symbols.setdefault(symbol, {})
    rec[timeframe] = int(completed_bucket)
    rec["updated_ts"] = int(time.time() * 1000)


def _merge_event_cache(existing: list[dict], new_events: list[dict]) -> list[dict]:
    """Merge by event_id while preserving events independent of current universe."""
    by_id: dict[str, dict] = {}
    for ev in existing + new_events:
        if not isinstance(ev, dict):
            continue
        eid = ev.get("event_id")
        if eid:
            by_id[str(eid)] = ev
    return list(by_id.values())


OI_HISTORY = DATA / "oi_history.json"
_OI_HIST_CACHE: dict[str, Any] = {"ts": 0.0, "data": {}, "path": ""}
_OI_HIST_CACHE_TTL = 30.0
_OI_HIST_MAX_BUCKETS = 500


def _load_oi_history() -> dict[str, dict[str, float]]:
    """Cached view of the accumulated OI snapshot history (audit fix B2)."""
    now = time.monotonic()
    cache_path = str(OI_HISTORY.resolve())
    if (
        _OI_HIST_CACHE.get("path") == cache_path
        and now - float(_OI_HIST_CACHE.get("ts", 0.0)) < _OI_HIST_CACHE_TTL
        and "data" in _OI_HIST_CACHE
    ):
        return _OI_HIST_CACHE["data"]
    raw = _load_json(OI_HISTORY, {})
    data = raw if isinstance(raw, dict) else {}
    _OI_HIST_CACHE["ts"] = now
    _OI_HIST_CACHE["path"] = cache_path
    _OI_HIST_CACHE["data"] = data
    return data


def _record_oi_snapshots(rows: list[Any], now_ms: int) -> int:
    """Persist per-symbol OI snapshots keyed by the current 1h bucket.

    Snapshots are written only while their bucket is active, so a stored value
    always predates the bucket close. Divergence detectors can therefore map
    bucket -> OI without look-ahead. Returns the number of symbols updated.
    """
    if not rows:
        return 0
    history = _load_json(OI_HISTORY, {})
    if not isinstance(history, dict):
        history = {}
    bucket = str(int(now_ms // 3_600_000))
    updated = 0
    for r in rows:
        try:
            symbol = str(getattr(r, "symbol", "") or "").upper()
            oi = getattr(r, "oi", None)
            if not symbol or oi is None:
                continue
            oi = float(oi)
        except (TypeError, ValueError):
            continue
        if oi <= 0:
            continue
        rec = history.setdefault(symbol, {})
        if not isinstance(rec, dict):
            rec = history[symbol] = {}
        rec[bucket] = oi
        if len(rec) > _OI_HIST_MAX_BUCKETS:
            for key in sorted(rec, key=lambda k: int(k))[: len(rec) - _OI_HIST_MAX_BUCKETS]:
                del rec[key]
        updated += 1

    if updated:
        _save_json_atomic(OI_HISTORY, history)
        _OI_HIST_CACHE["ts"] = 0.0
    _OI_HIST_CACHE["path"] = str(OI_HISTORY.resolve())
    return updated


def _acquire_scan_slot(min_interval: float) -> None:
    """Process-local pacing for BingX kline scans.

    GitHub-hosted runners are ephemeral and may execute on different VMs.
    Persisting time.monotonic() across runs is invalid because monotonic clocks
    are only comparable within the same boot/environment. Keep the pacing
    timestamp in process memory only.
    """
    min_interval = max(0.0, float(min_interval))
    now = time.monotonic()
    last = getattr(_acquire_scan_slot, "_last_call", None)
    if last is not None:
        wait = min_interval - (now - last)
        if wait > 0:
            time.sleep(wait)
    _acquire_scan_slot._last_call = time.monotonic()

def _file_lock_pace(lock_dir: Path, min_interval: float) -> float:
    """Backward-compatible name; pacing is intentionally process-local.

    The old implementation persisted time.monotonic() in a file. That is
    invalid across ephemeral GitHub Actions runners because monotonic clocks
    are not comparable between different VMs. The lock_dir argument is retained
    only for API/test compatibility and is intentionally unused.
    """
    _ = lock_dir
    _acquire_scan_slot(min_interval)
    return time.monotonic()


def _fetch_klines_scan(symbol: str, timeframe: str, limit: int) -> list[dict]:
    """Fetch Klines with serialized pacing and bounded transient-error retry.

    This protects the runner from burst traffic and BingX application-level
    errors while never marking a symbol/timeframe processed until the caller
    validates the returned dataset. Pacing is process-local because GitHub-hosted
    runners are ephemeral and cross-run monotonic timestamps are not comparable.
    """
    min_interval = float(os.environ.get("BINGX_KLINE_SCAN_MIN_INTERVAL_SEC", "1.05"))
    max_attempts = int(os.environ.get("BINGX_KLINE_RETRY_ATTEMPTS", "3"))
    max_attempts = max(1, min(max_attempts, 5))
    backoff_base = float(os.environ.get("BINGX_KLINE_RETRY_BACKOFF_SEC", "1.0"))

    last_error: Exception | None = None
    for attempt in range(max_attempts):
        log.info(
            "[BINGX_KLINE] START %s/%s attempt %d/%d (rate interval=%.2fs)...",
            symbol, timeframe, attempt + 1, max_attempts, min_interval
        )
        slot_started = time.monotonic()
        log.info("[BINGX_KLINE] WAIT_SLOT %s/%s attempt %d/%d...", symbol, timeframe, attempt + 1, max_attempts)
        _acquire_scan_slot(min_interval)
        log.info(
            "[BINGX_KLINE] slot acquired %s/%s after %.2fs; requesting...",
            symbol, timeframe, time.monotonic() - slot_started
        )
        request_started = time.monotonic()
        try:
            # Scan requests are intentionally fail-fast: the outer loop already
            # provides bounded retries/backoff, so urllib3 must not add another
            # hidden retry chain here.
            result = fetch_klines(
                symbol, timeframe, limit,
                timeout_sec=float(os.environ.get("BINGX_KLINE_HTTP_TIMEOUT_SEC", "5")),
                retryable=False,
            )
            log.info(
                "[BINGX_KLINE] END %s/%s attempt %d/%d in %.2fs; rows=%d.",
                symbol, timeframe, attempt + 1, max_attempts,
                time.monotonic() - request_started, len(result or [])
            )
            return result
        except (RuntimeError, requests.RequestException, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 >= max_attempts:
                break
            delay = max(0.0, backoff_base) * (2 ** attempt)
            log.warning("[BINGX] Kline retry %d/%d for %s/%s after error: %s; sleeping %.1fs", attempt + 1, max_attempts - 1, symbol, timeframe, exc, delay)
            if delay:
                time.sleep(delay)
    raise RuntimeError(f"Kline fetch failed for {symbol}/{timeframe} after {max_attempts} attempts: {last_error}") from last_error


def _tf_stats(stats: dict, timeframe: str) -> dict:
    by_tf = stats.setdefault("by_timeframe", {})
    tf = str(timeframe).lower()
    rec = by_tf.setdefault(tf, {
        "scanned": 0,
        "divergence_events": 0,
        "squeeze_events": 0,
        "scan_errors": 0,
        "fresh_events": 0,
        "fresh_divergence": 0,
        "fresh_squeeze": 0,
        "trigger_passed": 0,
        "trigger_no_window": 0,
        "trigger_breakout_failed": 0,
        "trigger_volume_failed": 0,
        "trigger_direction_failed": 0,
        "rejected_btc": 0,
        "rejected_funding": 0,
        "rejected_cvd": 0,
        "rejected_score": 0,
        "rejected_entry_drift": 0,
        "rejected_hot_oi": 0,
        "rejected_symbol_quarantine": 0,
        "valid_signals": 0,
    })
    return rec


def _refresh_timeframe_events(candidates, timeframe: str, limit: int, now_ms: int, seen_ids: set[str], stats: dict, scan_state: dict, completed_bucket: int) -> list[dict]:
    fresh: list[dict] = []
    for r in candidates:
        symbol = str(r.symbol).upper()
        if not _symbol_scan_due(scan_state, symbol, timeframe, completed_bucket):
            continue
        tf_stats = _tf_stats(stats, timeframe)
        try:
            klines = _fetch_klines_scan(symbol, timeframe, limit)
            if len(klines) < 60:
                log.warning("[SIGNALS] %s %s returned only %d candles; watermark deferred.", timeframe.upper(), symbol, len(klines))
                continue
            d = add_cvd(pd.DataFrame(klines))
            # Audit fix B2: attach the accumulated OI snapshot history so
            # detect_divergences can emit Price-vs-OI divergence when coverage
            # is sufficient (no events until enough buckets are recorded).
            d = attach_oi_series(d, _load_oi_history().get(symbol))
            divs = detect_divergences(d, symbol, timeframe)
            sqs = detect_squeeze_release(d, symbol, timeframe, min_squeeze_bars=3, release_lookback_bars=int(os.environ.get("SQUEEZE_RELEASE_LOOKBACK_BARS", "4")))
            # Audit fix B3: forced-liquidation squeeze from Coinalyze factors.
            liqs = detect_liquidation_squeeze(r, d, symbol, timeframe)
            tf_stats["scanned"] += 1
            stats["divergence_events"] += len(divs)
            stats["squeeze_events"] += len(sqs) + len(liqs)
            tf_stats["divergence_events"] += len(divs)
            tf_stats["squeeze_events"] += len(sqs) + len(liqs)
            stats["events_total"] += len(divs) + len(sqs) + len(liqs)
            for ev in divs + sqs + liqs:
                if not _event_is_fresh(ev, now_ms, MAX_AGE):
                    continue
                fresh.append(ev)
                eid = ev.get("event_id")
                if eid and eid not in seen_ids:
                    emit_event(ev)
                    seen_ids.add(eid)

            # Persist the watermark only AFTER event emission succeeded. If the
            # journal write or state write fails, this symbol remains due for a
            # later run instead of being silently skipped forever. Persisting
            # after every successfully scanned symbol also preserves progress
            # across workflow timeout/interruption.
            _mark_symbol_scanned(scan_state, symbol, timeframe, completed_bucket)
            _save_timeframe_scan_state(scan_state)
        except Exception as exc:
            stats["scan_errors"] += 1
            tf_stats["scan_errors"] += 1
            log.warning("[SIGNALS] %s %s fetch/detection error: %s", timeframe.upper(), symbol, exc)
    return fresh


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            val = obj.get("event_id")
            if val:
                ids.add(str(val))
    return ids


def load_successful_telegram_ids(path: Path) -> set[str]:
    """Return only event IDs for which Telegram actually reported success.

    A failed send must remain retryable on a later cycle. Historically the code
    used load_ids(ACTIONS), which incorrectly treated telegram_sent=false rows
    as already delivered forever.
    """
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if bool(obj.get("telegram_sent")):
            event_id = obj.get("event_id")
            if event_id:
                ids.add(str(event_id))
    return ids


def send_pending_open_trade_notifications(
    current_positions: dict[tuple[str, str], dict],
    successful_ids: set[str],
) -> set[str]:
    """Retry Telegram alerts for already-open trades that were never confirmed delivered.

    This is independent of event freshness: a notification failure must not become
    permanent merely because the 1H event aged out of the scanning window.
    Returns event IDs attempted during this cycle so the normal candidate loop does
    not send the same retry twice.
    """
    attempted: set[str] = set()
    active = _load_active_trades()
    for event_id, trade in active.items():
        event_id = str(event_id)
        if trade.get("closed", False) or event_id in successful_ids:
            continue

        symbol = str(trade.get("symbol", ""))
        direction = str(trade.get("direction", "")).upper()
        bx_symbol = to_bx_symbol(symbol)
        if not bx_symbol or (bx_symbol, direction) not in current_positions:
            continue

        event = {
            "event_id": event_id,
            "symbol": symbol,
            "timeframe": str(trade.get("timeframe") or (trade.get("setup") or {}).get("event_timeframe") or "1h").lower(),
            "direction": direction,
            "event_type": trade.get("event_type", "TRADE_OPEN"),
            "timestamps": {},
            "event_fact": {},
        }
        position = current_positions[(bx_symbol, direction)]
        execution = {
            "status": "OPENED_CONFIRMED_RETRY",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "position": position,
        }
        setup = trade.get("setup", {}) if isinstance(trade.get("setup"), dict) else {}
        msg = format_signal(event, setup=setup, execution=execution, score=trade.get("score"))
        attempted.add(event_id)
        try:
            sent = bool(send_tg(msg))
        except Exception as exc:
            sent = False
            log.error("[TELEGRAM] Pending-open retry exception for %s %s (%s): %s", direction, symbol, event_id, exc)
        record_action({
            "event_id": event_id,
            "symbol": symbol,
            "direction": direction,
            "score": trade.get("score"),
            "event_type": trade.get("event_type"),
            "telegram_sent": sent,
            "telegram_kind": "open_retry",
            "execution_status": "OPENED_CONFIRMED_RETRY",
            "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
        })
        if sent:
            successful_ids.add(event_id)
            log.info("[TELEGRAM] Pending open notification delivered for %s %s (%s).", direction, symbol, event_id)
        else:
            log.error("[TELEGRAM] Pending open notification failed for %s %s (%s); will retry.", direction, symbol, event_id)
    return attempted


def _load_symbol_quarantines(path: Path, now_ms: int, max_consecutive_losses: int, quarantine_min: float) -> dict[str, int]:
    """Return symbols temporarily quarantined after repeated confirmed losses.

    State is derived from the append-only trade journal, so a clean start needs no
    separate legacy file. Only confirmed TRADE_CLOSE records participate.
    """
    if max_consecutive_losses <= 0 or quarantine_min <= 0 or not path.exists():
        return {}
    closes: dict[str, list[dict]] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("record_type") != "TRADE_CLOSE":
                    continue
                symbol = str(obj.get("symbol") or "").strip().upper()
                closed_ts = int(_safe_float(obj.get("closed_ts"), 0.0))
                pnl = _safe_float(obj.get("realized_pnl_pct"), 0.0)
                if not symbol or closed_ts <= 0 or closed_ts > now_ms:
                    continue
                closes.setdefault(symbol, []).append({"ts": closed_ts, "pnl": pnl})
    except OSError:
        return {}

    out: dict[str, int] = {}
    window_ms = int(quarantine_min * 60_000)
    for symbol, rows in closes.items():
        rows.sort(key=lambda x: x["ts"])
        streak = 0
        for row in reversed(rows):
            if row["pnl"] < 0:
                streak += 1
                if streak >= max_consecutive_losses:
                    most_recent_loss_ts = int(rows[-1]["ts"])
                    if now_ms - most_recent_loss_ts < window_ms:
                        out[symbol] = most_recent_loss_ts + window_ms
                    break
            else:
                break
    return out


def _symbol_on_quarantine(symbol: str, quarantines: dict[str, int], now_ms: int) -> bool:
    until = int(quarantines.get(str(symbol).strip().upper(), 0) or 0)
    return until > now_ms


def _entry_drift_pct(signal_price: float, trigger_price: float, direction: str) -> float | None:
    """Absolute percentage distance from signal price to trigger price.

    This is a signal-decay metric, not trade PnL slippage: a SHORT trigger below
    the signal is still a late entry and therefore carries positive drift.
    Direction is accepted for API clarity and future policy expansion.
    """
    _ = direction
    signal_price = _safe_float(signal_price, 0.0)
    trigger_price = _safe_float(trigger_price, 0.0)
    if signal_price <= 0 or trigger_price <= 0:
        return None
    return abs(trigger_price - signal_price) / signal_price * 100.0


def load_successful_trade_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if str(obj.get("record_type", "")) == "EVENT_TERMINAL":
            event_id = obj.get("event_id")
            if event_id:
                ids.add(str(event_id))
            continue

        result = obj.get("result", {})
        if not isinstance(result, dict):
            continue
        status = str(result.get("status", "")).lower()
        if status in {"opened_protected", "opened_protection_check_required", "opened", "opened_protection_failed", "already_executed"}:
            event_id = obj.get("event_id")
            if event_id:
                ids.add(str(event_id))
    return ids


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def emit_event(ev: dict) -> None:
    append_jsonl(EVENTS, ev)


def record_trade(obj: dict) -> None:
    append_jsonl(TRADES, obj)


def record_action(obj: dict) -> None:
    append_jsonl(ACTIONS, obj)


def calculate_execution_slippage(
    signal_price: float, actual_entry_price: float, direction: str,
    pre_order_price: float | None = None,
) -> dict:
    try:
        signal_price = float(signal_price)
        actual_entry_price = float(actual_entry_price)
        pre_order = float(pre_order_price) if pre_order_price is not None else None
    except (TypeError, ValueError):
        return {"slippage_pct": None, "adverse_slippage_pct": None}
    if signal_price <= 0 or actual_entry_price <= 0:
        return {"slippage_pct": None, "adverse_slippage_pct": None}
    d = str(direction).upper()
    signed_total = (actual_entry_price - signal_price) / signal_price * 100.0
    signal_to_order = None
    execution_move = None
    if pre_order and pre_order > 0:
        signal_to_order = (pre_order - signal_price) / signal_price * 100.0
        execution_move = (actual_entry_price - pre_order) / pre_order * 100.0
    def adverse(x):
        if x is None: return None
        return max(0.0, x) if d == "LONG" else max(0.0, -x)
    return {
        "slippage_pct": signed_total,
        "adverse_slippage_pct": adverse(signed_total),
        "signal_to_order_drift_pct": signal_to_order,
        "execution_slippage_pct": execution_move,
        "adverse_execution_slippage_pct": adverse(execution_move),
    }


def check_funding_filter(
    row: Any,
    direction: str,
    event_type: str | None = None,
    max_short_adverse: float | None = None,
    max_long_adverse: float | None = None,
) -> tuple[bool, str]:
    """Block adverse funding only for squeeze events.

    Funding is stored in percentage-point units (0.10 == +0.10%).
    Normal/divergence signals are intentionally not hard-blocked by funding.
    Active squeeze events use +/-0.10% hard limits. Explicit threshold
    arguments remain supported for tests/backward compatibility.
    """
    if row is None:
        return True, "NO_ROW"
    fr = getattr(row, "fr_oiw", None)
    if fr is None:
        return True, "NO_FUNDING_DATA"
    try:
        fr_val = float(fr)
    except (TypeError, ValueError):
        return True, "INVALID_FUNDING_DATA"

    d = str(direction).upper()
    is_squeeze = "SQUEEZE" in str(event_type or "").upper()

    # Funding is a hard entry gate only for squeeze events. For ordinary
    # divergence signals, funding remains a research/context field and must
    # not block an otherwise valid setup.
    if not is_squeeze and max_short_adverse is None and max_long_adverse is None:
        return True, "OK_NORMAL_FUNDING_NOT_FILTERED"

    short_limit = MAX_SHORT_SQUEEZE_ADVERSE_FUNDING if is_squeeze else float("-inf")
    long_limit = MAX_LONG_SQUEEZE_ADVERSE_FUNDING if is_squeeze else float("inf")
    if max_short_adverse is not None:
        short_limit = float(max_short_adverse)
    if max_long_adverse is not None:
        long_limit = float(max_long_adverse)

    if d == "SHORT" and fr_val < short_limit:
        scope = "SQUEEZE" if is_squeeze else "OVERRIDE"
        return False, f"ADVERSE_FUNDING_SHORT_{scope} (fr={fr_val:.4f} < {short_limit:.4f})"
    if d == "LONG" and fr_val > long_limit:
        scope = "SQUEEZE" if is_squeeze else "OVERRIDE"
        return False, f"ADVERSE_FUNDING_LONG_{scope} (fr={fr_val:.4f} > {long_limit:.4f})"

    return True, "OK"


def resolve_symbol_direction_conflicts(opportunities: list[dict]) -> tuple[list[dict], list[dict]]:
    """Allow one direction per symbol; resolve only true LONG/SHORT conflicts.

    The existing best_opportunities_map has already deduplicated multiple events
    in the same (symbol, direction). Here we prevent simultaneous opposite-side
    setups for the same symbol. Score is primary; 4H is the deterministic tie-break.
    """
    by_symbol: dict[str, list[dict]] = {}
    for opp in opportunities:
        by_symbol.setdefault(str(opp.get("symbol", "")), []).append(opp)

    kept: list[dict] = []
    rejected: list[dict] = []
    tf_rank = {"4h": 2, "1h": 1}
    for symbol, items in by_symbol.items():
        directions = {str(x.get("direction", "")).upper() for x in items}
        if len(directions) <= 1:
            kept.extend(items)
            continue

        ranked = sorted(
            items,
            key=lambda x: (
                float(x.get("score", 0.0)),
                tf_rank.get(str(x.get("event", {}).get("timeframe", "1h")).lower(), 0),
                1 if "SQUEEZE" in str(x.get("event", {}).get("event_type", "")).upper() else 0,
                int(x.get("event", {}).get("timestamps", {}).get("detected_at_ts", 0) or 0),
            ),
            reverse=True,
        )
        winner = ranked[0]
        winner.setdefault("conflict_events", [])
        kept.append(winner)
        for loser in ranked[1:]:
            winner["conflict_events"].append({
                "event_id": loser.get("event_id"),
                "event_type": loser.get("event", {}).get("event_type"),
                "timeframe": loser.get("event", {}).get("timeframe", "1h"),
                "direction": loser.get("direction"),
                "score": float(loser.get("score", 0.0)),
            })
            loser["conflict_rejected_against"] = {
                "direction": winner.get("direction"),
                "score": winner.get("score"),
                "timeframe": winner.get("event", {}).get("timeframe"),
                "event_id": winner.get("event_id"),
            }
            rejected.append(loser)
    return kept, rejected


def calculate_setup_score(
    ev: dict,
    coinalyze_row: Any,
    df_15m: pd.DataFrame,
    trigger_diagnostic: dict | None = None,
) -> float:
    score = 50.0
    fact = ev.get("event_fact", {})
    direction = str(ev.get("direction", "LONG")).upper()
    event_type = str(ev.get("event_type", "")).upper()

    try:
        delta_atr = float(fact.get("price_delta_atr", 0))
    except (TypeError, ValueError):
        delta_atr = 0.0

    if delta_atr >= 1.0:
        score += 15.0
    elif delta_atr >= 0.5:
        score += 10.0

    if event_type.endswith(("_MACD", "_STOCH", "_OBV", "_OI")):
        # Audit P1-1/P2-1/P2-2: new divergence families score like CVD.
        score += 15.0

    if "CVD" in event_type:
        score += 15.0

    # Бонус за Сквиз: +25 баллов (лидер по прибыли).
    # Audit P1-2: applies to volatility and forced-liquidation squeezes alike.
    if "SQUEEZE" in event_type:
        score += 25.0

        try:
            comp_ratio = float(fact.get("compression_ratio", 1.0))
        except (TypeError, ValueError):
            comp_ratio = 1.0

        if comp_ratio < 0.65:
            score += 15.0

        try:
            duration = int(fact.get("squeeze_duration_bars", 0))
        except (TypeError, ValueError):
            duration = 0

        if duration >= 5:
            score += 10.0

    if coinalyze_row is not None:
        try:
            oi_chg24 = getattr(coinalyze_row, "oi_chg24_pct", None)
            if oi_chg24 is not None:
                oi_chg24 = float(oi_chg24)
                if oi_chg24 >= MAX_HOT_OI_CHG24_PCT:
                    score -= HOT_OI_SCORE_PENALTY
        except (TypeError, ValueError):
            pass
        try:
            fr = getattr(coinalyze_row, "fr_oiw", None)
            if fr is not None:
                fr = float(fr)
                if direction == "LONG" and fr < 0:
                    score += 15.0
                elif direction == "SHORT" and fr > 0.02:
                    score += 15.0
                elif direction == "LONG" and fr > 0.05:
                    score -= 15.0
                elif direction == "SHORT" and fr < -0.05:
                    score -= 15.0
        except (TypeError, ValueError):
            pass

    try:
        if isinstance(trigger_diagnostic, dict) and trigger_diagnostic.get("volume_ratio") is not None:
            vol_ratio = float(trigger_diagnostic["volume_ratio"])
            if vol_ratio >= 1.5:
                score += 10.0
            elif vol_ratio >= 1.2:
                score += 5.0
        elif "volume" in df_15m.columns and len(df_15m) >= 20:
            # Backward-compatible fallback for direct unit/test callers.
            recent_avg = df_15m["volume"].iloc[-21:-1].mean()
            if pd.notna(recent_avg) and recent_avg > 0:
                vol_ratio = float(df_15m["volume"].iloc[-1]) / float(recent_avg)
                if vol_ratio >= 1.5:
                    score += 10.0
                elif vol_ratio >= 1.2:
                    score += 5.0
    except (TypeError, ValueError):
        pass

    return max(0.0, min(100.0, score))


def build_event_setup(ev: dict, df_1h: pd.DataFrame, entry_price: float) -> dict:
    direction = str(ev.get("direction", "LONG")).upper()
    if direction not in {"LONG", "SHORT"}:
        raise ValueError(f"Invalid direction={direction}")

    entry_price = float(entry_price)
    if entry_price <= 0:
        raise ValueError(f"Invalid entry_price={entry_price}")

    df = df_1h.copy()
    if len(df) < 20:
        raise ValueError("insufficient 1H bars for setup")

    for col in ("high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["high", "low", "close"]].isna().any().any():
        raise ValueError("invalid OHLC data")

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().iloc[-1]
    if pd.isna(atr) or float(atr) <= 0:
        raise ValueError("ATR unavailable")

    atr = float(atr)
    sl_atr_multiplier = 1.5
    risk_pct_raw = (atr * sl_atr_multiplier) / entry_price * 100.0

    if not (float("-inf") < risk_pct_raw < float("inf")):
        raise ValueError("Invalid ATR-derived risk")

    risk_pct = max(0.50, min(risk_pct_raw, 5.00))

    ev_type = str(ev.get("event_type", "")).upper()
    is_squeeze = "SQUEEZE" in ev_type
    target_rr = 3.0 if is_squeeze else 1.75
    planned_weighted_rr = 2.05 if is_squeeze else 1.05

    # TP3 (финальная цель) ставится на target_rr
    if direction == "LONG":
        invalidation = entry_price * (1.0 - risk_pct / 100.0)
        target = entry_price * (1.0 + target_rr * risk_pct / 100.0)
    else:
        invalidation = entry_price * (1.0 + risk_pct / 100.0)
        target = entry_price * (1.0 - target_rr * risk_pct / 100.0)

    return {
        "entry_reference": entry_price,
        "invalidation_price": invalidation,
        "target_price": target,
        "risk_pct": risk_pct,
        "target_rr": target_rr,
        "planned_weighted_rr": planned_weighted_rr,
        "realized_rr": None,
        "trigger_ok": True,
    }


def build_tp_levels(setup: dict, direction: str, event_type: str = "") -> Tuple[float, List[dict]]:
    direction = str(direction).upper()
    entry = float(setup["entry_reference"])
    sl_price = float(setup["invalidation_price"])

    if entry <= 0:
        raise ValueError("entry_reference must be > 0")

    if direction == "LONG":
        sl_pct = (entry - sl_price) / entry * 100.0
    elif direction == "SHORT":
        sl_pct = (sl_price - entry) / entry * 100.0
    else:
        raise ValueError(f"Invalid direction={direction}")

    if sl_pct <= 0:
        raise ValueError("Invalid SL percentage")

    ev_type = str(event_type or setup.get("event_type", "")).upper()
    is_squeeze = "SQUEEZE" in ev_type

    if is_squeeze:
        # Для сквизов тейки шире (импульсный потенциал и защита от преждевременного выбивания по БУ):
        # TP1: 1.00 * SL (30% объема + перевод в БУ после взятия 1.0R)
        # TP2: 2.00 * SL (35% объема)
        # TP3: 3.00 * SL (35% объема)
        # Взвешенный R:R: 0.30 * 1.0 + 0.35 * 2.0 + 0.35 * 3.0 = 2.05
        tp_levels = [
            {"leg": "tp1", "pnl_pct": round(sl_pct * 1.00, 6), "close_fraction": 0.30},
            {"leg": "tp2", "pnl_pct": round(sl_pct * 2.00, 6), "close_fraction": 0.35},
            {"leg": "tp3", "pnl_pct": round(sl_pct * 3.00, 6), "close_fraction": 0.35},
        ]
        planned_weighted_rr = 2.05
        target_rr = 3.0
    else:
        # Оптимальный 3-уровневый каскад для дивергенций:
        # TP1: 0.50 * SL (35% объема + перевод в БУ)
        # TP2: 1.00 * SL (35% объема)
        # TP3: 1.75 * SL (30% объема)
        # Взвешенный R:R: 0.35 * 0.50 + 0.35 * 1.00 + 0.30 * 1.75 = 1.05
        tp_levels = [
            {"leg": "tp1", "pnl_pct": round(sl_pct * 0.50, 6), "close_fraction": 0.35},
            {"leg": "tp2", "pnl_pct": round(sl_pct * 1.00, 6), "close_fraction": 0.35},
            {"leg": "tp3", "pnl_pct": round(sl_pct * 1.75, 6), "close_fraction": 0.30},
        ]
        planned_weighted_rr = 1.05
        target_rr = 1.75

    setup["risk_pct"] = sl_pct
    setup["target_rr"] = target_rr
    setup["planned_weighted_rr"] = planned_weighted_rr
    setup["realized_rr"] = None
    setup["tp_levels"] = tp_levels

    return sl_pct, tp_levels


def install_protection(
    symbol: str,
    direction: str,
    position: dict,
    setup: dict,
    sl_pct: float,
    tp_levels: list,
    trade_id: str,
) -> dict:
    try:
        avg_price = float(position.get("avgPrice", 0) or position.get("entryPrice", 0) or 0)
        qty = abs(float(position.get("positionAmt", 0) or 0))
    except (TypeError, ValueError):
        return {"status": "PROTECTION_INVALID_POSITION", "error": "invalid position values"}

    if avg_price <= 0 or qty <= 0:
        return {"status": "PROTECTION_INVALID_POSITION", "error": f"invalid avgPrice={avg_price} or qty={qty}"}

    try:
        return ensure_directional_protection(
            symbol=symbol,
            direction=direction,
            avg_price=avg_price,
            qty=qty,
            stop_loss_pct=sl_pct,
            tp_levels=tp_levels,
            trade_id=trade_id,
        )
    except Exception as exc:
        return {"status": "PROTECTION_EXCEPTION", "error": str(exc)}


def _tp_orders_to_tracker(
    tp_orders: list[dict],
    *,
    direction: str | None = None,
    avg_price: float | None = None,
    effective_levels: list[dict] | None = None,
) -> list[dict]:
    """Convert live TP orders into tracker records without relying on clientOrderId.

    BingX conditional TP orders do not support clientOrderId. For current orders,
    the trigger price is therefore the authoritative leg identity when the original
    TP profile is known. For an orphan position with no stored profile, retain the
    live protection rather than forcing a repair loop; deterministic ordering gives
    stable fallback labels for tracker state.
    """
    out: list[dict] = []
    pending: list[tuple[dict, float]] = []
    expected = [x for x in (effective_levels or []) if isinstance(x, dict)]

    for order in tp_orders:
        cid = str(order.get("clientOrderId", "")).upper()
        leg = next((x for x in ("tp1", "tp2", "tp3") if x.upper() in cid), None)
        try:
            price = float(order.get("stopPrice", 0) or order.get("price", 0) or 0)
            qty = float(order.get("origQty", 0) or order.get("quantity", 0) or 0)
        except (TypeError, ValueError):
            continue
        if price <= 0 or qty <= 0:
            continue

        if not leg and avg_price and avg_price > 0 and direction in {"LONG", "SHORT"} and expected:
            best = None
            used_expected_legs = {str(x.get("leg", "")).lower() for x in out if x.get("leg")}
            for level in expected:
                candidate_leg = str(level.get("leg", "")).lower()
                try:
                    pnl_pct = float(level.get("pnl_pct", 0))
                except (TypeError, ValueError):
                    continue
                if candidate_leg not in {"tp1", "tp2", "tp3"} or candidate_leg in used_expected_legs or pnl_pct <= 0:
                    continue
                expected_price = avg_price * (1.0 + pnl_pct / 100.0) if direction == "LONG" else avg_price * (1.0 - pnl_pct / 100.0)
                rel = abs(price - expected_price) / max(abs(expected_price), 1e-12)
                if best is None or rel < best[0]:
                    best = (rel, candidate_leg)
            if best is not None and best[0] <= 0.0025:
                leg = best[1]

        row = {
            "leg": leg,
            "status": "already_exists",
            "order_id": str(order.get("orderId", "")),
            "price": price,
            "qty": qty,
        }
        if leg:
            out.append(row)
        else:
            pending.append((row, price))

    if pending:
        # No stored profile exists. Preserve the live orders and assign labels from
        # their favorable distance. A single current TP is the exchange-safe
        # micro-position fallback used by ensure_directional_protection: tp3.
        if avg_price and avg_price > 0:
            pending.sort(key=lambda item: abs(item[1] - avg_price))
        else:
            pending.sort(key=lambda item: item[1])
        fallback_legs = ["tp3"] if len(pending) == 1 else ["tp1", "tp2", "tp3"]
        for (row, _), leg in zip(pending, fallback_legs):
            row["leg"] = leg
            out.append(row)

    return out


def _sl_order_to_tracker(sl_orders: list[dict]) -> dict:
    if not sl_orders:
        return {}
    sl = sl_orders[0]
    return {
        "status": "already_exists",
        "order_id": str(sl.get("orderId", "")),
        "stop_price": float(sl.get("stopPrice", 0) or sl.get("price", 0) or 0),
        "qty": float(sl.get("origQty", 0) or sl.get("quantity", 0) or 0),
    }


def _find_active_trade_for_position(bx_symbol: str, direction: str, active_trades: dict) -> dict | None:
    want_dir = str(direction).upper()
    for trade in active_trades.values():
        if trade.get("closed", False):
            continue
        t_bx = to_bx_symbol(trade.get("symbol", ""))
        t_dir = str(trade.get("direction", "")).upper()
        if t_bx == bx_symbol and t_dir == want_dir:
            return trade
    return None


def reconcile_all_open_positions() -> None:
    # Reconciliation must never monopolize the 5-minute event loop. The normal
    # BingX session intentionally retries GETs, but that can turn one network
    # problem into a long sequence of waits. Reconciliation uses fail-fast GETs
    # and a cycle budget; anything deferred is retried on the next workflow run.
    try:
        recon_timeout = float(os.environ.get("RECONCILIATION_HTTP_TIMEOUT_SEC", "5"))
    except (TypeError, ValueError):
        recon_timeout = 5.0
    recon_timeout = max(2.0, min(recon_timeout, 15.0))
    try:
        recon_budget = float(os.environ.get("RECONCILIATION_MAX_SECONDS", "45"))
    except (TypeError, ValueError):
        recon_budget = 45.0
    recon_budget = max(10.0, min(recon_budget, 180.0))

    started = time.monotonic()
    log.info(
        "[RECONCILIATION] Fetching open positions (timeout=%.1fs, retryable=false, budget=%.1fs)...",
        recon_timeout,
        recon_budget,
    )
    try:
        positions = get_positions(timeout_sec=recon_timeout, retryable=False)
    except Exception as exc:
        log.error("[RECONCILIATION] Failed to fetch positions: %s", exc)
        return

    log.info("[RECONCILIATION] Open-position response received: %d records.", len(positions))
    active_trades = _load_active_trades()
    log.info("[RECONCILIATION] Active trade state loaded: %d records.", len(active_trades))

    for position_index, p in enumerate(positions, start=1):
        elapsed = time.monotonic() - started
        if elapsed >= recon_budget:
            log.warning(
                "[RECONCILIATION] Time budget reached after %.1fs; deferred %d/%d position records to next cycle.",
                elapsed,
                max(0, len(positions) - position_index + 1),
                len(positions),
            )
            break

        bx_symbol = str(p.get("symbol", "")).upper()
        if not bx_symbol:
            continue

        position_side = str(p.get("positionSide", "")).upper()
        try:
            amt = float(p.get("positionAmt", 0) or 0)
            avg_price = float(p.get("avgPrice", 0) or p.get("entryPrice", 0) or 0)
        except (ValueError, TypeError):
            continue

        if amt == 0 or avg_price <= 0:
            continue

        direction = position_side if position_side in {"LONG", "SHORT"} else ("LONG" if amt > 0 else "SHORT")
        qty = abs(amt)

        log.info("[RECONCILIATION] Position %d/%d: %s %s | checking protection...", position_index, len(positions), bx_symbol, direction)
        prot = get_open_protection_directional(
            bx_symbol,
            direction,
            timeout_sec=recon_timeout,
            retryable=False,
        )
        if prot.get("status") != "ok":
            log.warning("[RECONCILIATION] Cannot inspect protection for %s: %s", bx_symbol, prot.get("error"))
            continue

        sl_orders = list(prot.get("sl_orders", []))
        tp_orders = list(prot.get("tp_orders", []))

        matched_trade = _find_active_trade_for_position(bx_symbol, direction, active_trades)
        hit_legs = {str(x).lower() for x in (matched_trade.get("hit_legs", []) if matched_trade else set())}
        be_activated = bool(matched_trade.get("be_activated", False)) if matched_trade else False

        effective_levels = matched_trade.get("effective_tp_levels") if matched_trade else None
        configured_legs = {str(x.get("leg")).lower() for x in effective_levels if isinstance(x, dict) and x.get("leg")} if isinstance(effective_levels, list) and effective_levels else {"tp1", "tp2", "tp3"}
        remaining_expected_legs = configured_legs - hit_legs

        tracker_tp_probe = _tp_orders_to_tracker(
            tp_orders,
            direction=direction,
            avg_price=avg_price,
            effective_levels=effective_levels if isinstance(effective_levels, list) else None,
        )
        known_tp_legs = {str(x.get("leg", "")).lower() for x in tracker_tp_probe if x.get("leg")}

        sl_valid = False
        if sl_orders:
            try:
                sl_price = float(sl_orders[0].get("stopPrice", 0) or sl_orders[0].get("price", 0) or 0)
                sl_amt = float(sl_orders[0].get("origQty", 0) or sl_orders[0].get("quantity", 0) or 0)

                if sl_price > 0 and sl_amt > 0:
                    qty_matches = abs(sl_amt - qty) <= max(qty * 1e-6, 1e-12)
                    if direction == "LONG":
                        price_matches = (sl_price <= avg_price * 1.003) if be_activated else (sl_price < avg_price)
                    elif direction == "SHORT":
                        price_matches = (sl_price >= avg_price * 0.997) if be_activated else (sl_price > avg_price)
                    else:
                        price_matches = False
                    sl_valid = bool(price_matches and qty_matches and len(sl_orders) == 1)
            except (TypeError, ValueError):
                sl_valid = False

        # A position without local tracker state is an orphan. There is no safe
        # original TP profile to compare against, so preserve any currently open
        # TP orders together with a valid SL instead of endlessly creating
        # duplicate/rewritten protection on every cycle. Once registered, the
        # inferred live profile becomes authoritative for future reconciliation.
        if matched_trade is None:
            protection_complete = sl_valid and bool(tracker_tp_probe)
        else:
            protection_complete = sl_valid and remaining_expected_legs.issubset(known_tp_legs)

        if protection_complete:
            tracker_tp = _tp_orders_to_tracker(
                tp_orders,
                direction=direction,
                avg_price=avg_price,
                effective_levels=effective_levels if isinstance(effective_levels, list) else None,
            )
            tracker_sl = _sl_order_to_tracker(sl_orders)
            if tracker_tp and tracker_sl:
                tracked = update_active_trade_protection(
                    symbol=bx_symbol,
                    direction=direction,
                    tp_orders=tracker_tp,
                    sl_result=tracker_sl,
                    effective_tp_levels=matched_trade.get("effective_tp_levels") if matched_trade else None,
                    tp_mode=matched_trade.get("tp_mode") if matched_trade else None,
                    effective_weighted_rr=matched_trade.get("effective_weighted_rr") if matched_trade else None,
                )
                if not tracked and not matched_trade:
                    try:
                        sl_price = _safe_float(sl_orders[0].get("stopPrice") or sl_orders[0].get("price"), 0.0)
                        inferred_risk = abs(avg_price - sl_price) / avg_price * 100.0 if sl_price > 0 else 2.0
                        inferred_risk = max(0.05, min(inferred_risk, 25.0))
                        inferred_levels = []
                        for tp in tracker_tp:
                            tp_price = _safe_float(tp.get("price"), 0.0)
                            if tp_price <= 0:
                                continue
                            pnl_pct = ((tp_price - avg_price) / avg_price * 100.0) if direction == "LONG" else ((avg_price - tp_price) / avg_price * 100.0)
                            if pnl_pct <= 0:
                                continue
                            inferred_levels.append({
                                "leg": str(tp.get("leg", "tp1")),
                                "pnl_pct": pnl_pct,
                                "close_fraction": _safe_float(tp.get("qty"), 0.0) / max(qty, 1e-12),
                            })
                        if not inferred_levels:
                            inferred_levels = [{"leg": "tp1", "pnl_pct": inferred_risk * 1.75, "close_fraction": 1.0}]
                        total_fraction = sum(max(_safe_float(x.get("close_fraction"), 0.0), 0.0) for x in inferred_levels)
                        if total_fraction <= 0:
                            inferred_levels = [{"leg": "tp1", "pnl_pct": inferred_risk * 1.75, "close_fraction": 1.0}]
                        else:
                            for level in inferred_levels:
                                level["close_fraction"] = max(_safe_float(level.get("close_fraction"), 0.0), 0.0) / total_fraction
                        max_rr = max(abs(_safe_float(tp.get("pnl_pct"), 0.0)) / inferred_risk for tp in inferred_levels)
                        inferred_setup = {
                            "risk_pct": inferred_risk,
                            "target_rr": max_rr,
                            "planned_weighted_rr": max_rr,
                            "effective_weighted_rr": max_rr,
                            "tp_mode": "single_tp" if len(inferred_levels) == 1 else "multi_tp",
                            "effective_tp_levels": inferred_levels,
                            "tp_levels": inferred_levels,
                            "entry_reference": avg_price,
                            "invalidation_price": sl_price,
                            "target_price": avg_price,
                            "event_type": "RECONCILED_POSITION",
                        }
                        register_active_trade(
                            event_id=f"RECON_{bx_symbol}_{direction}",
                            symbol=bx_symbol.replace("-USDT", ""),
                            name=bx_symbol.replace("-USDT", ""),
                            direction=direction,
                            entry_price=avg_price,
                            qty=qty,
                            tp_orders=tracker_tp,
                            sl_result=tracker_sl,
                            event_type="RECONCILED_POSITION",
                            timeframe="1h",
                            score=50.0,
                            setup=inferred_setup,
                            requested_entry_price=avg_price,
                        )
                        log.warning("[RECONCILIATION] Registered orphan protected position %s (%s) into tracker.", bx_symbol, direction)
                    except Exception as exc:
                        log.error("[RECONCILIATION] Failed to register protected orphan %s (%s): %s", bx_symbol, direction, exc)
            continue

        log.warning(
            "[RECONCILIATION] %s (%s) Incomplete: SL=%s, TPs=%d/%d (hit: %s). Repairing missing...",
            bx_symbol, direction, "OK" if sl_valid else "MISSING", len(known_tp_legs), len(remaining_expected_legs), list(hit_legs)
        )

        # Audit P1-3 (check/repair skew mitigation): re-inspect protection
        # immediately before repairing. A TP fill, BE move or manual change
        # that happened between the first check and now is picked up here,
        # preventing duplicate repair orders.
        recheck = get_open_protection_directional(
            bx_symbol,
            direction,
            timeout_sec=recon_timeout,
            retryable=False,
        )
        if recheck.get("status") == "ok":
            recheck_tp = list(recheck.get("tp_orders", []))
            recheck_sl = list(recheck.get("sl_orders", []))
            recheck_tracker_probe = _tp_orders_to_tracker(
                recheck_tp,
                direction=direction,
                avg_price=avg_price,
                effective_levels=effective_levels if isinstance(effective_levels, list) else None,
            )
            recheck_known_legs = {str(x.get("leg", "")).lower() for x in recheck_tracker_probe if x.get("leg")}
            recheck_sl_valid = False
            if recheck_sl:
                try:
                    r_sl_price = float(recheck_sl[0].get("stopPrice", 0) or recheck_sl[0].get("price", 0) or 0)
                    r_sl_amt = float(recheck_sl[0].get("origQty", 0) or recheck_sl[0].get("quantity", 0) or 0)
                    r_qty_matches = abs(r_sl_amt - qty) <= max(qty * 1e-6, 1e-12)
                    if direction == "LONG":
                        r_price_matches = (r_sl_price <= avg_price * 1.003) if be_activated else (r_sl_price < avg_price)
                    elif direction == "SHORT":
                        r_price_matches = (r_sl_price >= avg_price * 0.997) if be_activated else (r_sl_price > avg_price)
                    else:
                        r_price_matches = False
                    recheck_sl_valid = bool(r_sl_price > 0 and r_sl_amt > 0 and r_qty_matches and r_price_matches and len(recheck_sl) == 1)
                except (TypeError, ValueError):
                    recheck_sl_valid = False
            if recheck_sl_valid and remaining_expected_legs.issubset(recheck_known_legs):
                log.info("[RECONCILIATION] %s (%s) complete on re-check; skipping repair.", bx_symbol, direction)
                continue

        sl_pct = _safe_float(matched_trade.get("planned_risk_pct"), 0.0) if matched_trade else 0.0
        if sl_pct <= 0:
            sl_pct = 2.0
        try:
            k1 = _fetch_klines_scan(bx_symbol, "1h", limit=30)
            if not matched_trade and len(k1) >= 20:
                df1 = pd.DataFrame(k1)
                for col in ("high", "low", "close"):
                    df1[col] = pd.to_numeric(df1[col], errors="coerce")

                prev_close = df1["close"].shift(1)
                tr = pd.concat(
                    [
                        df1["high"] - df1["low"],
                        (df1["high"] - prev_close).abs(),
                        (df1["low"] - prev_close).abs(),
                    ],
                    axis=1,
                ).max(axis=1)

                atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().iloc[-1]
                if pd.notna(atr) and float(atr) > 0:
                    risk_pct = max(0.50, min(float(atr) * 1.5 / avg_price * 100.0, 5.00))
                    sl_pct = risk_pct
        except Exception as exc:
            log.error("[RECONCILIATION] ATR error for %s: %s", bx_symbol, exc)

        tp_levels = []
        if matched_trade and isinstance(matched_trade.get("effective_tp_levels"), list) and matched_trade.get("effective_tp_levels"):
            # Preserve the exact original protection profile, including squeeze TP1/TP2/TP3
            # distances and micro-position single-TP mode. Never silently replace a squeeze
            # with the ordinary divergence 0.5R/1R/1.75R profile during restart repair.
            for level in matched_trade.get("effective_tp_levels", []):
                if not isinstance(level, dict):
                    continue
                leg = str(level.get("leg", ""))
                if not leg or leg in hit_legs:
                    continue
                try:
                    pnl_pct = float(level.get("pnl_pct", 0))
                    fraction = float(level.get("close_fraction", 0))
                except (TypeError, ValueError):
                    continue
                if pnl_pct > 0 and fraction > 0:
                    tp_levels.append({"leg": leg, "pnl_pct": pnl_pct, "close_fraction": fraction})

        if not tp_levels:
            if "tp1" not in hit_legs:
                tp_levels.append({"leg": "tp1", "pnl_pct": round(sl_pct * 0.50, 6), "close_fraction": 0.35})
            if "tp2" not in hit_legs:
                tp_levels.append({"leg": "tp2", "pnl_pct": round(sl_pct * 1.00, 6), "close_fraction": 0.35})
            if "tp3" not in hit_legs:
                tp_levels.append({"leg": "tp3", "pnl_pct": round(sl_pct * 1.75, 6), "close_fraction": 0.30})

        if not tp_levels:
            tp_levels = [{"leg": "tp3", "pnl_pct": round(sl_pct * 1.75, 6), "close_fraction": 1.0}]

        trade_event_id = matched_trade.get("event_id") if matched_trade else f"REC_{bx_symbol}_{direction}"

        res = ensure_directional_protection(
            symbol=bx_symbol,
            direction=direction,
            avg_price=avg_price,
            qty=qty,
            stop_loss_pct=sl_pct,
            tp_levels=tp_levels,
            trade_id=str(trade_event_id).replace("EVT_", ""),
            stop_loss_price=avg_price if be_activated else None,
        )

        status = str(res.get("status", "")).upper()
        repaired_tp = res.get("tp_orders", [])
        repaired_sl = res.get("sl_result", {})

        if status in {"PROTECTED", "SL_ONLY"} and repaired_tp and repaired_sl:
            tracked = update_active_trade_protection(
                symbol=bx_symbol,
                direction=direction,
                tp_orders=repaired_tp,
                sl_result=repaired_sl,
                effective_tp_levels=res.get("effective_tp_levels"),
                tp_mode=res.get("tp_mode"),
                effective_weighted_rr=res.get("effective_weighted_rr"),
            )
            if not tracked and not matched_trade:
                register_active_trade(
                    event_id=f"RECON_{bx_symbol}_{direction}",
                    symbol=bx_symbol.replace("-USDT", ""),
                    name=bx_symbol.replace("-USDT", ""),
                    direction=direction,
                    entry_price=avg_price,
                    qty=qty,
                    tp_orders=repaired_tp,
                    sl_result=repaired_sl,
                    event_type="RECONCILED_POSITION",
                )

            first_tp = min((float(x.get("pnl_pct", 0)) for x in (repaired_tp or []) if x.get("pnl_pct") is not None), default=0.0)
            log.info("[RECONCILIATION] Protection restored for %s (%s): SL=%.2f%%, first TP=+%.2f%%", bx_symbol, direction, sl_pct, first_tp)
    
    log.info("[RECONCILIATION] Finished in %.1fs.", time.monotonic() - started)


def execute_new_position(symbol: str, direction: str, price: float, setup: dict, event_id: str) -> dict:
    direction = str(direction).upper()
    trade_id = event_id.replace("EVT_", "")

    log.info("[EXECUTION] Opening market position: %s %s at ref price %.8g...", direction, symbol, price)

    try:
        opened = open_market(symbol, direction, price, trade_id)
    except Exception as exc:
        return {"status": "OPEN_EXCEPTION", "mode": EXECUTION_MODE, "order_id": None, "error": str(exc)}

    if not isinstance(opened, dict):
        return {"status": "OPEN_INVALID_RESPONSE", "mode": EXECUTION_MODE, "order_id": None, "raw": repr(opened)}

    open_status = str(opened.get("status", "")).lower()
    if open_status not in {"opened", "success", "ok"}:
        nested_response = opened.get("response") if isinstance(opened.get("response"), dict) else {}
        exchange_code = opened.get("code")
        if exchange_code is None:
            exchange_code = nested_response.get("code")
        error_text = opened.get("error") or opened.get("msg") or nested_response.get("msg") or open_status
        return {
            "status": "EXISTING_POSITION" if open_status == "existing_position" else "OPEN_FAILED",
            "mode": EXECUTION_MODE,
            "order_id": opened.get("order_id"),
            "open_result": opened,
            "error": error_text,
            "bingx_code": exchange_code,
        }

    order_id = opened.get("order_id")

    def _rollback_unprotected_entry(status: str, *, position: dict | None = None, error: str | None = None) -> dict:
        """Fail-safe: after an accepted market entry, never abandon an unknown/unprotected position."""
        rollback = emergency_close_position(
            symbol, direction,
            qty=_safe_float((position or {}).get("positionAmt"), 0.0) if isinstance(position, dict) else None,
            reason_token=f"ENTRYFAIL:{trade_id}:{status}",
        )
        result = {
            "status": status,
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
            "position": position or {},
            "error": error,
            "emergency_close": rollback,
            "rolled_back": rollback.get("status") == "closed",
        }
        if result["rolled_back"]:
            result["position"] = {**(position or {}), "positionAmt": 0.0}
        return result

    try:
        position = wait_for_position_fill_directional(symbol=symbol, direction=direction, timeout_sec=15, poll_interval=0.5)
    except Exception as exc:
        return _rollback_unprotected_entry("POSITION_WAIT_FAILED", error=str(exc))

    if not isinstance(position, dict) or str(position.get("status", "")).lower() != "found":
        return _rollback_unprotected_entry(
            "POSITION_NOT_CONFIRMED",
            position=position if isinstance(position, dict) else {},
            error=str((position or {}).get("error") or (position or {}).get("status") or "position not confirmed"),
        )

    try:
        actual_qty = abs(float(position.get("positionAmt", 0) or 0))
        actual_avg_price = float(position.get("avgPrice", 0) or position.get("entryPrice", 0) or 0)
    except (TypeError, ValueError):
        actual_qty = 0.0
        actual_avg_price = 0.0

    if actual_qty <= 0 or actual_avg_price <= 0:
        return _rollback_unprotected_entry(
            "POSITION_INVALID",
            position=position,
            error=f"invalid confirmed position qty={actual_qty} avgPrice={actual_avg_price}",
        )

    pre_order_price = _safe_float(opened.get("order_reference_price"), 0.0)
    execution_quality = calculate_execution_slippage(
        signal_price=price, actual_entry_price=actual_avg_price, direction=direction,
        pre_order_price=pre_order_price if pre_order_price > 0 else None,
    )
    fill_ts_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    log.info(
        "[EXECUTION] Fill confirmed: %s %s at avgPrice=%.8g (Qty: %.8g, Slippage: %+.2f%%)",
        direction, symbol, actual_avg_price, actual_qty, execution_quality.get("slippage_pct") or 0.0
    )

    event_type_for_risk = str(setup.get("event_type", "")).upper()
    drift_limit = MAX_SQUEEZE_ENTRY_DRIFT_PCT if "SQUEEZE" in event_type_for_risk else MAX_ENTRY_DRIFT_PCT
    signal_price_for_risk = _safe_float(setup.get("signal_price", price), 0.0)
    fill_drift_pct = _entry_drift_pct(signal_price_for_risk, actual_avg_price, direction)
    execution_quality["signal_to_fill_distance_pct"] = fill_drift_pct
    if drift_limit > 0 and fill_drift_pct is not None and fill_drift_pct > drift_limit:
        rollback = emergency_close_position(symbol, direction, actual_qty, reason_token=f"DRIFTFAIL:{trade_id}")
        flattened = rollback.get("status") == "closed"
        log.error(
            "[EXECUTION] Entry drift exceeded for %s %s: %.2f%% > %.2f%%; emergency_close=%s",
            direction, symbol, fill_drift_pct, drift_limit, rollback.get("status"),
        )
        result_position = {**position, "positionAmt": 0.0} if flattened else {**position, "positionAmt": actual_qty}
        return {
            "status": "ENTRY_DRIFT_EXCEEDED",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "position": result_position,
            "open_result": opened,
            "execution_quality": execution_quality,
            "error": f"signal_to_fill_distance_pct={fill_drift_pct:.6f} > limit={drift_limit:.6f}",
            "emergency_close": rollback,
            "rolled_back": flattened,
            "fill_ts_ms": fill_ts_ms,
            "notional_usdt": actual_avg_price * actual_qty,
            "leverage": opened.get("leverage"),
        }

    try:
        setup_for_fill = dict(setup)
        setup_for_fill["entry_reference"] = actual_avg_price
        setup_for_fill["signal_price"] = float(price)
        setup_for_fill["pre_order_reference_price"] = pre_order_price if pre_order_price > 0 else None
        planned_risk_pct = float(setup.get("risk_pct", 0) or 0)

        if not pd.notna(planned_risk_pct) or planned_risk_pct <= 0:
            raise ValueError("invalid planned risk_pct")

        ev_type = str(setup.get("event_type", "")).upper()
        is_squeeze = "SQUEEZE" in ev_type
        target_rr = 3.0 if is_squeeze else 1.75
        planned_weighted_rr = 2.05 if is_squeeze else 1.05

        if direction == "LONG":
            invalidation = actual_avg_price * (1.0 - planned_risk_pct / 100.0)
            target = actual_avg_price * (1.0 + target_rr * planned_risk_pct / 100.0)
        else:
            invalidation = actual_avg_price * (1.0 + planned_risk_pct / 100.0)
            target = actual_avg_price * (1.0 - target_rr * planned_risk_pct / 100.0)

        setup_for_fill["invalidation_price"] = invalidation
        setup_for_fill["target_price"] = target
        setup_for_fill["target_rr"] = target_rr
        setup_for_fill["planned_weighted_rr"] = planned_weighted_rr
        setup_for_fill["realized_rr"] = None

    except (TypeError, ValueError) as exc:
        rollback = emergency_close_position(symbol, direction, actual_qty, reason_token=f"SETUPFAIL:{trade_id}")
        return {
            "status": "PROTECTION_SETUP_INVALID",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "position": {**position, "positionAmt": actual_qty},
            "open_result": opened,
            "execution_quality": execution_quality,
            "error": str(exc),
            "emergency_close": rollback,
            "rolled_back": rollback.get("status") == "closed",
        }

    try:
        sl_pct, tp_levels = build_tp_levels(setup_for_fill, direction, event_type=ev_type)
    except Exception as exc:
        rollback = emergency_close_position(symbol, direction, actual_qty, reason_token=f"TPSETUPFAIL:{trade_id}")
        return {
            "status": "PROTECTION_SETUP_INVALID",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "position": {**position, "positionAmt": actual_qty},
            "open_result": opened,
            "execution_quality": execution_quality,
            "error": str(exc),
            "emergency_close": rollback,
            "rolled_back": rollback.get("status") == "closed",
        }

    protection = install_protection(
        symbol=symbol,
        direction=direction,
        position={**position, "positionAmt": actual_qty, "avgPrice": actual_avg_price},
        setup=setup_for_fill,
        sl_pct=sl_pct,
        tp_levels=tp_levels,
        trade_id=trade_id,
    )

    if protection.get("effective_tp_levels"):
        setup_for_fill["effective_tp_levels"] = protection["effective_tp_levels"]
    setup_for_fill["tp_mode"] = protection.get("tp_mode", "multi_tp")
    if protection.get("effective_weighted_rr") is not None:
        setup_for_fill["effective_weighted_rr"] = protection["effective_weighted_rr"]

    protection_status = str(protection.get("status", "")).upper()
    if protection.get("rolled_back"):
        final_status = "opened_rolled_back"
        try:
            post_close = get_position_directional(symbol, direction)
            if str(post_close.get("status", "")).lower() != "found":
                position = {**position, "positionAmt": 0.0}
        except Exception:
            pass
    elif protection_status == "PROTECTED":
        final_status = "opened_protected"
    elif protection_status == "SL_ONLY":
        final_status = "opened_protection_check_required"
    else:
        final_status = "opened_protection_failed"

    log.info("[EXECUTION] Protection installed for %s %s: Status=%s, SL=%.2f%%, TPs=%d legs", direction, symbol, final_status, sl_pct, len(tp_levels))

    return {
        "status": final_status,
        "mode": EXECUTION_MODE,
        "order_id": order_id,
        "open_result": opened,
        "position": {**position, "positionAmt": actual_qty, "avgPrice": actual_avg_price},
        "protection": protection,
        "sl_pct": sl_pct,
        "tp_levels": tp_levels,
        "execution_quality": execution_quality,
        "setup_used_for_protection": setup_for_fill,
        "fill_ts_ms": fill_ts_ms,
        "notional_usdt": actual_avg_price * actual_qty,
        "leverage": opened.get("leverage"),
        "planned_risk_usdt": (actual_avg_price * actual_qty) * planned_risk_pct / 100.0,
    }


def main() -> None:
    log.info("========== [ENGINE] CYCLE START: %s UTC | Mode: %s | Exec: %s ==========", pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"), EXECUTION_MODE, EXECUTION_ENABLED)

    config_ok, config_reason = _validate_execution_config()
    if not config_ok:
        log.critical("[ENGINE] EXECUTION PREFLIGHT FAILED: %s", config_reason)
        return

    if EXECUTION_ENABLED:
        try:
            log.info("[TRACKER] Checking active trades lifecycle & TP execution...")
            update_active_trades()
        except Exception as exc:
            log.error("[TRACKER] Update error: %s", exc)

        try:
            log.info("[RECONCILIATION] Reconciling open positions and protection...")
            reconcile_all_open_positions()
        except Exception as exc:
            log.error("[RECONCILIATION] Error: %s", exc)

    stats = {
        "coinalyze_rows": 0,
        "liquidity_candidates": 0,
        "contract_candidates": 0,
        "candidates_scanned": 0,
        "divergence_events": 0,
        "squeeze_events": 0,
        "events_total": 0,
        "fresh_events": 0,
        "fresh_long": 0,
        "fresh_short": 0,
        "fresh_divergence": 0,
        "fresh_squeeze": 0,
        "rejected_age": 0,
        "rejected_btc": 0,
        "rejected_funding": 0,
        "rejected_trigger": 0,
        "rejected_cvd": 0,
        "trigger_passed": 0,
        "trigger_no_window": 0,
        "trigger_breakout_failed": 0,
        "trigger_volume_failed": 0,
        "trigger_data_failed": 0,
        "trigger_direction_failed": 0,
        "rejected_score": 0,
        "rejected_short_score": 0,
        "rejected_entry_drift": 0,
        "rejected_hot_oi": 0,
        "rejected_symbol_quarantine": 0,
        "conflict_rejected": 0,
        "valid_signals": 0,
        "execution_attempts": 0,
        "trades": 0,
        "scan_errors": 0,
        "cached_events": 0,
        "timeframe_scanned_symbols_1h": 0,
        "timeframe_scanned_symbols_4h": 0,
        "telegram_pending_retries": 0,
        "telegram_pending_retry_success": 0,
        "by_timeframe": {},
    }

    btc_regime_df = None
    try:
        log.info("[ENGINE_STAGE] BTC regime fetch START (1h, limit=10)...")
        stage_started = time.monotonic()
        btc_klines = _fetch_klines_scan("BTC-USDT", "1h", limit=10)
        log.info("[ENGINE_STAGE] BTC regime fetch END in %.2fs; rows=%d.", time.monotonic() - stage_started, len(btc_klines or []))
        if btc_klines:
            btc_regime_df = pd.DataFrame(btc_klines)
            last_c = float(btc_regime_df["close"].iloc[-1])
            prev_1h = float(btc_regime_df["close"].iloc[-2])
            prev_4h = float(btc_regime_df["close"].iloc[-5])
            chg_1h = ((last_c - prev_1h) / prev_1h) * 100.0
            chg_4h = ((last_c - prev_4h) / prev_4h) * 100.0
            log.info("[BTC_REGIME] BTC: %.1f | 1H: %+.2f%% | 4H: %+.2f%% | Filter: OK", last_c, chg_1h, chg_4h)
    except Exception as exc:
        log.error("[BTC_REGIME] Fetch error: %s", exc)

    rows: list[Any] = []
    try:
        log.info("[ENGINE_STAGE] Coinalyze fetch START...")
        stage_started = time.monotonic()
        rows = fetch_data()
        log.info("[ENGINE_STAGE] Coinalyze fetch END in %.2fs; rows=%d.", time.monotonic() - stage_started, len(rows))
        log.info("[COINALYZE] Ingested %d rows from Coinalyze.", len(rows))
    except Exception as exc:
        stats["scan_errors"] += 1
        log.error("[COINALYZE] Scrape error: %s", exc)

    stats["coinalyze_rows"] = len(rows)

    # Audit fix B2: persist OI snapshots per 1h bucket so Price-vs-OI swing
    # divergence becomes computable once enough history has accumulated.
    try:
        stats["oi_snapshots_recorded"] = _record_oi_snapshots(rows, int(pd.Timestamp.utcnow().timestamp() * 1000))
    except Exception as exc:
        log.error("[OI_HISTORY] Snapshot record error: %s", exc)

    try:
        contracts = refresh_contracts()
        log.info("[BINGX] Refreshed %d active perpetual contracts.", len(contracts))
    except Exception as exc:
        stats["scan_errors"] += 1
        log.error("[BINGX] Contracts refresh error: %s", exc)

    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    current_open_positions: dict[tuple[str, str], bool] = {}
    current_positions: dict[tuple[str, str], dict] = {}
    position_state_unknown = False
    try:
        for position in get_positions():
            bx_sym = str(position.get("symbol", "")).upper()
            side = str(position.get("positionSide", position.get("positionAmt", ""))).upper()
            try:
                amt = float(position.get("positionAmt", 0) or 0)
            except Exception:
                amt = 0.0

            if amt != 0:
                want_dir = side if side in {"LONG", "SHORT"} else ("LONG" if amt > 0 else "SHORT")
                key = (bx_sym, want_dir)
                current_open_positions[key] = True
                current_positions[key] = position
    except Exception as exc:
        position_state_unknown = True
        log.error("[BINGX] Failed to pre-fetch positions for deduplication; NEW ENTRIES BLOCKED: %s", exc)

    recent_entry_ts = _load_recent_successful_entries(
        TRADES, now_ms, max(SYMBOL_ENTRY_COOLDOWN_MIN, SQUEEZE_SYMBOL_ENTRY_COOLDOWN_MIN)
    )
    symbol_quarantines = _load_symbol_quarantines(
        TRADES, now_ms, SYMBOL_MAX_CONSECUTIVE_LOSSES, SYMBOL_QUARANTINE_MIN
    )

    telegram_sent_event_ids = load_successful_telegram_ids(ACTIONS)
    telegram_attempted_this_cycle = send_pending_open_trade_notifications(
        current_positions=current_positions,
        successful_ids=telegram_sent_event_ids,
    )
    stats["telegram_pending_retries"] = len(telegram_attempted_this_cycle)
    stats["telegram_pending_retry_success"] = sum(1 for eid in telegram_attempted_this_cycle if eid in telegram_sent_event_ids)

    candidates: List[Any] = []
    for r in rows:
        try:
            if (
                r.price is None
                or r.price <= 0
                or r.volume24 is None
                or r.volume24 < MIN_VOL
                or r.oi is None
                or r.oi < MIN_OI
            ):
                continue

            stats["liquidity_candidates"] += 1
            if not get_contract(r.symbol):
                continue

            stats["contract_candidates"] += 1
            candidates.append(r)
        except Exception:
            stats["scan_errors"] += 1
            continue

    if MAX_CANDIDATES > 0:
        candidates = candidates[:MAX_CANDIDATES]

    stats["candidates_scanned"] = len(candidates)
    log.info("[UNIVERSE] %d liquidity candidates ($%.0fM Vol, $%.0fM OI) -> %d scanned on BingX.", stats["liquidity_candidates"], MIN_VOL/1e6, MIN_OI/1e6, len(candidates))

    seen_events = load_ids(EVENTS)
    executed_event_ids = load_successful_trade_ids(TRADES)
    best_opportunities_map: dict[tuple[str, str], dict] = {}

    scan_state = _load_timeframe_scan_state()
    event_cache = _load_cached_events()

    completed_1h = _completed_bucket(3_600_000, now_ms, BAR_CLOSE_GRACE_MIN)
    completed_4h = _completed_bucket(14_400_000, now_ms, BAR_CLOSE_GRACE_MIN)

    # Per-symbol watermarks preserve the existing event math while ensuring that a
    # symbol entering the liquidity universe late is scanned immediately for its
    # latest completed bar. Failures do not advance the watermark.
    new_1h = _refresh_timeframe_events(
        candidates, "1h", int(os.environ.get("KLINE_LIMIT_1H", "250")),
        now_ms, seen_events, stats, scan_state, completed_1h,
    )
    new_4h = _refresh_timeframe_events(
        candidates, "4h", int(os.environ.get("KLINE_LIMIT_4H", "250")),
        now_ms, seen_events, stats, scan_state, completed_4h,
    )
    event_cache = _merge_event_cache(event_cache, new_1h + new_4h)
    event_cache = [ev for ev in event_cache if _event_is_fresh(ev, now_ms, MAX_AGE)]
    _save_json_atomic(EVENT_CACHE, {"updated_ts": now_ms, "events": event_cache})
    _save_timeframe_scan_state(scan_state)

    stats["cached_events"] = len(event_cache)
    # These counters describe symbols actually scanned in THIS cycle, not symbols whose
    # persisted watermark already equals the current completed bucket.
    stats["timeframe_scanned_symbols_1h"] = int(_tf_stats(stats, "1h").get("scanned", 0))
    stats["timeframe_scanned_symbols_4h"] = int(_tf_stats(stats, "4h").get("scanned", 0))
    stats["fresh_events"] = 0
    stats["fresh_long"] = 0
    stats["fresh_short"] = 0
    stats["fresh_divergence"] = 0
    stats["fresh_squeeze"] = 0

    events_by_symbol: dict[str, list[dict]] = {}
    for ev in event_cache:
        events_by_symbol.setdefault(str(ev.get("symbol", "")), []).append(ev)

    # Fresh 1H ATR is fetched only after an event passes the cheap 15M trigger + score gate.
    risk_1h_cache: dict[str, pd.DataFrame] = {}
    for r in candidates:
        symbol = str(r.symbol)
        all_events = events_by_symbol.get(symbol, [])
        if not all_events:
            continue

        d15 = None
        for ev in sorted(all_events, key=lambda x: int(x.get("timestamps", {}).get("detected_at_ts", 0) or 0), reverse=True):
            event_id = ev.get("event_id")
            if not event_id or event_id in executed_event_ids:
                continue
            direction = str(ev.get("direction", "")).upper()
            if direction not in {"LONG", "SHORT"}:
                stats["trigger_direction_failed"] += 1
                continue
            bx_symbol = to_bx_symbol(symbol)
            if bx_symbol and current_open_positions.get((bx_symbol, direction)):
                continue
            try:
                detected_at = int(ev.get("timestamps", {}).get("detected_at_ts", 0) or 0)
            except (TypeError, ValueError):
                stats["trigger_data_failed"] += 1
                continue
            if detected_at <= 0:
                stats["trigger_data_failed"] += 1
                continue
            age = (now_ms - detected_at) / 60_000.0
            if age < 0 or age > MAX_AGE:
                continue

            tf = str(ev.get("timeframe", "1h")).lower()
            tf_stats = _tf_stats(stats, tf)

            stats["fresh_events"] += 1
            tf_stats["fresh_events"] += 1
            stats["fresh_long"] += int(direction == "LONG")
            stats["fresh_short"] += int(direction == "SHORT")
            event_type = str(ev.get("event_type", "")).upper()
            stats["fresh_squeeze"] += int("SQUEEZE" in event_type)
            stats["fresh_divergence"] += int("SQUEEZE" not in event_type)
            tf_stats["fresh_squeeze"] += int("SQUEEZE" in event_type)
            tf_stats["fresh_divergence"] += int("SQUEEZE" not in event_type)
            log.info("[SIGNALS] Fresh event: %s %s | TF: %s | Type: %s | Age: %.1fm", direction, symbol, tf, event_type, age)

            if btc_regime_df is not None and symbol != "BTC-USDT":
                btc_ok, btc_reason = check_btc_regime(btc_regime_df, direction)
                if not btc_ok:
                    stats["rejected_btc"] += 1
                    tf_stats["rejected_btc"] += 1
                    continue

            funding_ok, funding_reason = check_funding_filter(
                r, direction, event_type=event_type
            )
            if not funding_ok:
                stats["rejected_funding"] += 1
                tf_stats["rejected_funding"] += 1
                log.info("[SIGNALS] %s %s (%s/%s) rejected by funding filter: %s", direction, symbol, tf, event_type, funding_reason)
                continue

            if d15 is None:
                try:
                    k15 = _fetch_klines_scan(symbol, "15m", int(os.environ.get("KLINE_LIMIT_15M", "250")))
                except Exception as exc:
                    stats["trigger_data_failed"] += 1
                    stats["scan_errors"] += 1
                    log.warning("[SIGNALS] 15M fetch error for %s: %s", symbol, exc)
                    continue
                if len(k15) < 20:
                    stats["trigger_data_failed"] += 1
                    continue
                d15 = pd.DataFrame(k15)

            if not {"close_time", "close", "high", "low", "volume"}.issubset(d15.columns):
                stats["trigger_data_failed"] += 1
                continue

            signal_price = _safe_float(ev.get("event_fact", {}).get("detection_close_price") or r.price, 0.0)
            if signal_price <= 0:
                stats["scan_errors"] += 1
                continue

            if REQUIRE_TRIGGER:
                trigger_diag = diagnose_15m_trigger(d15, direction, event_detected_at_ts=detected_at, max_trigger_delay_min=MAX_TRIGGER_DELAY, min_vol_mult=1.05)
                if not trigger_diag.get("ok"):
                    reason = trigger_diag.get("reason") or "failed"
                    stats["rejected_trigger"] += 1
                    if reason == "no_trigger_window":
                        stats["trigger_no_window"] += 1; tf_stats["trigger_no_window"] += 1
                    elif reason == "breakout_failed":
                        stats["trigger_breakout_failed"] += 1; tf_stats["trigger_breakout_failed"] += 1
                    elif reason == "volume_failed":
                        stats["trigger_volume_failed"] += 1; tf_stats["trigger_volume_failed"] += 1
                    else:
                        stats["trigger_data_failed"] += 1; tf_stats["trigger_data_failed"] += 1
                    log.info("[SIGNALS] %s %s (%s/%s) failed 15m trigger: %s", direction, symbol, tf, event_type, reason)
                    continue
                trigger_price = _safe_float(trigger_diag.get("current_close"), 0.0)
                if trigger_price <= 0:
                    stats["trigger_data_failed"] += 1
                    tf_stats["trigger_data_failed"] += 1
                    continue
                drift_pct = _entry_drift_pct(signal_price, trigger_price, direction)
                drift_limit = MAX_SQUEEZE_ENTRY_DRIFT_PCT if "SQUEEZE" in event_type else MAX_ENTRY_DRIFT_PCT
                if drift_pct is None:
                    stats["trigger_data_failed"] += 1
                    tf_stats["trigger_data_failed"] += 1
                    continue
                if drift_pct > max(0.0, drift_limit):
                    stats["rejected_entry_drift"] += 1
                    tf_stats["rejected_entry_drift"] = tf_stats.get("rejected_entry_drift", 0) + 1
                    log.info("[SIGNALS] %s %s (%s/%s) rejected: entry drift %.2f%% > %.2f%%", direction, symbol, tf, event_type, drift_pct, drift_limit)
                    continue
                trigger_diag["signal_to_trigger_drift_pct"] = round(drift_pct, 6)
                stats["trigger_passed"] += 1
                tf_stats["trigger_passed"] += 1
            else:
                trigger_diag = {"ok": True, "reason": "not_required"}
                stats["trigger_passed"] += 1
                tf_stats["trigger_passed"] += 1

            if REQUIRE_CVD:
                try: cvd24_value = float(getattr(r, "cvd24", 0.0))
                except (TypeError, ValueError): stats["rejected_cvd"] += 1; continue
                if not pd.notna(cvd24_value) or cvd24_value <= CVD_MIN_CONFIRMATION:
                    stats["rejected_cvd"] += 1
                    tf_stats["rejected_cvd"] += 1
                    continue

            if REQUIRE_TRIGGER and trigger_diag.get("signal_to_trigger_drift_pct") is not None:
                setup_drift = float(trigger_diag["signal_to_trigger_drift_pct"])
            else:
                setup_drift = 0.0

            if _symbol_on_quarantine(symbol, symbol_quarantines, now_ms):
                stats["rejected_symbol_quarantine"] += 1
                tf_stats["rejected_symbol_quarantine"] = tf_stats.get("rejected_symbol_quarantine", 0) + 1
                log.info("[RISK] %s %s rejected: symbol quarantine active until %s", direction, symbol, pd.to_datetime(symbol_quarantines.get(symbol.upper()), unit="ms", utc=True).isoformat())
                continue

            oi_chg24 = _safe_float(getattr(r, "oi_chg24_pct", 0.0), 0.0)
            if oi_chg24 >= HARD_HOT_OI_CHG24_PCT and "SQUEEZE" not in event_type:
                stats["rejected_hot_oi"] += 1
                tf_stats["rejected_hot_oi"] = tf_stats.get("rejected_hot_oi", 0) + 1
                log.info("[RISK] %s %s rejected: OI24h %.2f%% >= hard hot-OI %.2f%%", direction, symbol, oi_chg24, HARD_HOT_OI_CHG24_PCT)
                continue

            score = calculate_setup_score(ev=ev, coinalyze_row=r, df_15m=d15, trigger_diagnostic=trigger_diag)
            min_score_for_direction = MIN_SHORT_SCORE if direction == "SHORT" else MIN_SCORE
            if min_score_for_direction > 0 and score < min_score_for_direction:
                stats["rejected_score"] += 1
                stats["rejected_short_score"] += int(direction == "SHORT")
                tf_stats["rejected_score"] += 1
                log.info("[SIGNALS] %s %s (%s/%s) rejected: score %.1f < required %.1f", direction, symbol, tf, event_type, score, min_score_for_direction)
                continue
                stats["rejected_score"] += 1
                tf_stats["rejected_score"] += 1
                log.info("[SIGNALS] %s %s (%s/%s) rejected: score %.1f < MIN_SCORE %.1f", direction, symbol, tf, event_type, score, MIN_SCORE)
                continue

            if symbol not in risk_1h_cache:
                try:
                    k1_risk = _fetch_klines_scan(symbol, "1h", int(os.environ.get("KLINE_LIMIT_1H", "250")))
                    if len(k1_risk) < 20:
                        stats["trigger_data_failed"] += 1
                        continue
                    risk_1h_cache[symbol] = pd.DataFrame(k1_risk)
                except Exception as exc:
                    stats["scan_errors"] += 1
                    log.warning("[RISK] Fresh 1H ATR fetch error for %s: %s", symbol, exc)
                    continue

            try:
                setup = build_event_setup(ev=ev, df_1h=risk_1h_cache[symbol], entry_price=signal_price)
            except (TypeError, ValueError, KeyError) as exc:
                stats["trigger_data_failed"] += 1
                log.warning("[RISK] Invalid setup for %s %s (%s/%s): %s", direction, symbol, tf, event_type, exc)
                continue
            except Exception as exc:
                stats["scan_errors"] += 1
                log.exception("[RISK] Unexpected setup error for %s %s (%s/%s)", direction, symbol, tf, event_type)
                continue
            setup["trigger"] = {
                "event_detected_at_ts": detected_at,
                "trigger_bar_close_ts": trigger_diag.get("trigger_bar_close_ts"),
                "trigger_price": _safe_float(trigger_diag.get("current_close"), 0.0) or None,
                "trigger_delay_min": trigger_diag.get("trigger_delay_min"),
                "signal_to_trigger_drift_pct": trigger_diag.get("signal_to_trigger_drift_pct"),
                "volume_ratio": trigger_diag.get("volume_ratio"),
            }
            setup["signal_price"] = signal_price
            setup["entry_risk"] = {
                "signal_to_trigger_drift_pct": trigger_diag.get("signal_to_trigger_drift_pct"),
                "oi_chg24_pct": oi_chg24,
                "hot_oi_warning": oi_chg24 >= MAX_HOT_OI_CHG24_PCT,
                "short_defensive_mode": direction == "SHORT",
                "symbol_quarantine_until_ts": symbol_quarantines.get(symbol.upper()),
            }
            setup["event_timeframe"] = tf
            setup["event_type"] = event_type
            setup["trigger_ok"] = True

            key = (symbol, direction)
            evidence = {
                "event_id": event_id,
                "event_type": event_type,
                "timeframe": tf,
                "score": float(score),
                "detected_at_ts": detected_at,
            }
            cand = {
                "event": ev, "event_id": event_id, "symbol": symbol, "direction": direction,
                "price": signal_price, "setup": setup, "score": score, "coinalyze_row": r,
                "confluence_events": [evidence],
            }
            if key not in best_opportunities_map:
                best_opportunities_map[key] = cand
            else:
                existing = best_opportunities_map[key]
                existing.setdefault("confluence_events", []).append(evidence)
                if score > existing["score"]:
                    cand["confluence_events"] = existing["confluence_events"]
                    best_opportunities_map[key] = cand

            log.info("[SIGNALS] Signal valid: %s %s | Score: %.0f/100 | TF: %s | Event: %s | Price: %.8g | SL: %.8g | TP: %.8g", direction, symbol, score, tf, event_type, signal_price, setup["invalidation_price"], setup["target_price"])

    opportunities = list(best_opportunities_map.values())
    for opp in opportunities:
        tf = str(opp.get("event", {}).get("timeframe", "1h")).lower()
        _tf_stats(stats, tf)["valid_signals"] += 1
    opportunities, conflict_rejected = resolve_symbol_direction_conflicts(opportunities)
    stats["conflict_rejected"] = len(conflict_rejected)
    stats["valid_signals"] = len(opportunities)
    opportunities.sort(key=lambda x: x["score"], reverse=True)

    for rejected in conflict_rejected:
        loser = rejected.get("direction")
        symbol = rejected.get("symbol")
        against = rejected.get("conflict_rejected_against", {})
        log.info("[RANKING] Conflict rejected: %s %s (Score %.0f, TF %s) vs %s %s (Score %.0f, TF %s).",
                 loser, symbol, float(rejected.get("score", 0)), rejected.get("event", {}).get("timeframe"),
                 against.get("direction"), symbol, float(against.get("score", 0)), against.get("timeframe"))

    log.info("[RANKING] Unique non-conflicting signals ready: %d.", len(opportunities))
    for i, opp in enumerate(opportunities[:5], start=1):
        log.info("  [RANKING] #%d: %s %s | Score: %.0f | %s", i, opp['direction'], opp['symbol'], opp['score'], opp['event'].get('event_type'))

    trades_this_cycle = 0

    for opp in opportunities:
        evidence = list(opp.get("confluence_events", []))
        primary_id = str(opp.get("event_id", ""))
        evidence = [e for e in evidence if str(e.get("event_id", "")) != primary_id]
        evidence.sort(key=lambda e: (-{"4h": 2, "1h": 1}.get(str(e.get("timeframe", "1h")).lower(), 0), str(e.get("event_type", ""))))
        conflicts = list(opp.get("conflict_events", []))
        setup_obj = opp.get("setup") if isinstance(opp.get("setup"), dict) else {}
        setup_obj["confluence_events"] = evidence
        setup_obj["conflict_events"] = conflicts
        opp["setup"] = setup_obj
        event_id = opp["event_id"]
        symbol = opp["symbol"]
        direction = opp["direction"]
        price = opp["price"]
        setup = opp["setup"]
        score = opp["score"]
        r = opp["coinalyze_row"]
        ev = opp["event"]

        bx_symbol = to_bx_symbol(symbol)
        opposite_direction = "SHORT" if direction == "LONG" else "LONG"
        opposite_position_open = bool(bx_symbol and current_open_positions.get((bx_symbol, opposite_direction)))
        is_squeeze_opp = "SQUEEZE" in str(ev.get("event_type", "")).upper()
        effective_cooldown = max(SYMBOL_ENTRY_COOLDOWN_MIN, SQUEEZE_SYMBOL_ENTRY_COOLDOWN_MIN) if is_squeeze_opp else SYMBOL_ENTRY_COOLDOWN_MIN
        symbol_cooldown = _symbol_on_cooldown(symbol, recent_entry_ts, now_ms, effective_cooldown)

        if position_state_unknown and EXECUTION_ENABLED:
            stats["blocked_by_position_state_unknown"] = stats.get("blocked_by_position_state_unknown", 0) + 1
            execution_result = {"status": "POSITION_STATE_UNKNOWN", "mode": EXECUTION_MODE, "order_id": None, "position": {}}
            log.error("[EXECUTION] %s %s blocked: exchange position state is UNKNOWN.", direction, symbol)
        elif event_id in executed_event_ids or (bx_symbol and current_open_positions.get((bx_symbol, direction))) or opposite_position_open or symbol_cooldown:
            existing_position = current_positions.get((bx_symbol, direction), {}) if bx_symbol else {}
            active_trade = None
            try:
                active_trade = next(
                    (x for x in _load_active_trades().values()
                     if not x.get("closed", False) and str(x.get("event_id", "")) == event_id),
                    None,
                )
            except Exception:
                active_trade = None
            execution_result = {
                "status": (
                    "SYMBOL_COOLDOWN" if symbol_cooldown and not existing_position and not opposite_position_open and event_id not in executed_event_ids
                    else ("CONFLICTING_DIRECTION_POSITION" if opposite_position_open and not existing_position else ("ALREADY_EXECUTED_WITH_POSITION" if existing_position else "ALREADY_EXECUTED"))
                ),
                "mode": EXECUTION_MODE,
                "order_id": None,
                "position": existing_position,
                "setup_used_for_protection": (active_trade or {}).get("setup", {}) if active_trade else setup,
            }
            log.info("[EXECUTION] %s (%s) - Already open/executed.%s%s", symbol, direction,
                     " Position confirmed; Telegram retry eligible." if existing_position else "",
                     f" Opposite {opposite_direction} position is already open; new direction blocked." if opposite_position_open else (f" Symbol cooldown active ({effective_cooldown:g}m)." if symbol_cooldown else ""))
        elif EXECUTION_ENABLED and trades_this_cycle < MAX_TRADES:
            stats["execution_attempts"] += 1
            trades_this_cycle += 1
            log.info("[EXECUTION] Attempt #%d/%d: %s %s (Score: %.0f, Ref: %.8g)...", trades_this_cycle, MAX_TRADES, direction, symbol, score, price)
            execution_result = execute_new_position(symbol=symbol, direction=direction, price=price, setup=setup, event_id=event_id)
            actual_position = execution_result.get("position", {}) if isinstance(execution_result, dict) else {}
            actual_qty_for_state = _safe_float(actual_position.get("positionAmt"), 0.0) if isinstance(actual_position, dict) else 0.0
            if actual_qty_for_state > 0:
                _mark_local_position_state(current_open_positions, current_positions, actual_position, symbol, direction)
                executed_event_ids.add(event_id)
                recent_entry_ts[str(symbol).upper()] = now_ms

            err_str = str(execution_result.get("error", "")).lower()
            terminal_reason = None
            if execution_result.get("status") == "ENTRY_DRIFT_EXCEEDED":
                terminal_reason = "ENTRY_DRIFT_EXCEEDED"
                log.warning("[EXECUTION] %s (%s) entry drift exceeded configured limit; terminalizing event %s.", symbol, direction, event_id)
            elif execution_result.get("bingx_code") == 101400 or "clientorderid unique check failed" in err_str:
                terminal_reason = "CLIENT_ORDER_ID_ALREADY_USED"
                log.warning("[EXECUTION] %s (%s) clientOrderId already used on exchange; terminalizing event %s.", symbol, direction, event_id)
            elif "min_qty" in err_str:
                terminal_reason = "MIN_QTY_NOT_REACHABLE"
                log.warning("[EXECUTION] %s (%s) min_qty not met at configured leverage; terminalizing event %s to prevent slot burn.", symbol, direction, event_id)
            elif "min_notional" in err_str or "min_usdt" in err_str or "min_size_usd" in err_str:
                terminal_reason = "MIN_NOTIONAL_NOT_REACHABLE"
                log.warning("[EXECUTION] %s (%s) exchange minimum notional not reachable at configured margin/leverage; terminalizing event %s.", symbol, direction, event_id)

            if terminal_reason:
                executed_event_ids.add(event_id)
                record_trade({
                    "record_type": "EVENT_TERMINAL",
                    "event_id": event_id,
                    "symbol": symbol,
                    "direction": direction,
                    "event_type": ev.get("event_type"),
                    "reason": terminal_reason,
                    "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                })

            actual_entry = float(actual_position.get("avgPrice", 0) or actual_position.get("entryPrice", 0) or price)
            actual_qty = actual_position.get("positionAmt")
            execution_quality = execution_result.get("execution_quality", {}) if isinstance(execution_result, dict) else {}

            execution_status = str(execution_result.get("status", ""))
            confirmed_trade = execution_status in {
                "opened_protected",
                "opened_protection_check_required",
                "opened_protection_failed",
            } and actual_qty_for_state > 0
            record_trade(
                {
                    "record_type": "TRADE_OPEN" if confirmed_trade else "EXECUTION_ATTEMPT",
                    "trade_id": "TR_" + hashlib.sha256(str(event_id).encode("utf-8")).hexdigest()[:24].upper(),
                    "event_id": event_id,
                    "symbol": symbol,
                    "direction": direction,
                    "signal": {
                        "event_type": ev.get("event_type"),
                        "timeframe": ev.get("timeframe"),
                        "signal_price": price,
                        "score": score,
                        "detected_at_ts": ev.get("timestamps", {}).get("detected_at_ts"),
                        "event_fact": ev.get("event_fact", {}),
                    },
                    "execution": {
                        "requested_price": price,
                        "signal_price": price,
                        "pre_order_reference_price": ((execution_result.get("open_result") or {}).get("order_reference_price") if isinstance(execution_result.get("open_result"), dict) else None),
                        "actual_entry_price": actual_entry,
                        "actual_qty": actual_qty,
                        "order_id": execution_result.get("order_id"),
                        "status": execution_result.get("status"),
                        "slippage_pct": execution_quality.get("slippage_pct"),
                        "signal_to_fill_drift_pct": execution_quality.get("slippage_pct"),
                        "adverse_slippage_pct": execution_quality.get("adverse_slippage_pct"),
                        "signal_to_order_drift_pct": execution_quality.get("signal_to_order_drift_pct"),
                        "execution_slippage_pct": execution_quality.get("execution_slippage_pct"),
                        "adverse_execution_slippage_pct": execution_quality.get("adverse_execution_slippage_pct"),
                        "entry_notional_usdt": execution_result.get("notional_usdt"),
                        "leverage": execution_result.get("leverage"),
                        "planned_risk_usdt": execution_result.get("planned_risk_usdt"),
                        "trigger_delay_min": ((setup.get("trigger") or {}).get("trigger_delay_min")),
                        "signal_to_trigger_drift_pct": ((setup.get("trigger") or {}).get("signal_to_trigger_drift_pct")),
                    },
                    "score": score,
                    "event_type": ev.get("event_type"),
                    "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                    "result": execution_result,
                    "setup": setup,
                    "planned_metrics": {
                        "target_rr": (execution_result.get("setup_used_for_protection") or setup).get("target_rr"),
                        "planned_weighted_rr": (execution_result.get("setup_used_for_protection") or setup).get("planned_weighted_rr"),
                        "effective_weighted_rr": (execution_result.get("setup_used_for_protection") or setup).get("effective_weighted_rr"),
                        "tp_mode": (execution_result.get("setup_used_for_protection") or setup).get("tp_mode"),
                        "realized_rr": None,
                    },
                }
            )

            status = str(execution_result.get("status", ""))
            if status in {"opened_protected", "opened_protection_check_required", "opened_protection_failed"}:
                if status in {"opened_protected", "opened_protection_check_required"} or actual_qty_for_state > 0:
                    stats["trades"] += 1

                try:
                    protection = execution_result.get("protection", {})
                    register_active_trade(
                        event_id=event_id,
                        symbol=symbol,
                        name=getattr(r, "name", None) or symbol,
                        direction=direction,
                        entry_price=float(execution_result.get("position", {}).get("avgPrice", price) or price),
                        qty=float(execution_result.get("position", {}).get("positionAmt", 0) or 0),
                        tp_orders=protection.get("tp_orders", []),
                        sl_result=protection.get("sl_result", {}),
                        event_type=ev.get("event_type", ""),
                        timeframe=ev.get("timeframe") or setup.get("event_timeframe") or setup.get("timeframe") or "1h",
                        coinalyze_row=r,
                        score=score,
                        setup=execution_result.get("setup_used_for_protection", setup),
                        requested_entry_price=_safe_float((execution_result.get("open_result") or {}).get("order_reference_price"), price) if isinstance(execution_result.get("open_result"), dict) else price,
                        entry_ts_ms=execution_result.get("fill_ts_ms"),
                    )
                except Exception as exc:
                    log.error("[TRACKER] Registration error for %s: %s", symbol, exc)

        elif not EXECUTION_ENABLED:
            execution_result = {"status": "DISABLED", "mode": EXECUTION_MODE, "order_id": None}
            log.info("[EXECUTION] %s %s skipped: EXECUTION_ENABLED is false.", direction, symbol)
        else:
            execution_result = {"status": "TRADE_LIMIT_REACHED", "mode": EXECUTION_MODE, "order_id": None}
            log.info("[EXECUTION] %s %s skipped: cycle limit reached (%d/%d).", direction, symbol, trades_this_cycle, MAX_TRADES)

        telegram_setup = execution_result.get("setup_used_for_protection") if isinstance(execution_result, dict) else None
        if not isinstance(telegram_setup, dict):
            telegram_setup = setup
        msg = format_signal(
            ev,
            setup=telegram_setup,
            coinalyze_row=r,
            execution=execution_result,
            score=score,
        )

        is_real_execution = execution_result.get("status") in {
            "opened_protected",
            "opened",
            "opened_protection_check_required",
            "opened_protection_failed",
            "ALREADY_EXECUTED_WITH_POSITION",
        }

        telegram_already_sent = event_id in telegram_sent_event_ids
        sent = False
        if is_real_execution and not telegram_already_sent and event_id not in telegram_attempted_this_cycle:
            try:
                sent = bool(send_tg(msg))
            except Exception as exc:
                sent = False
                log.error("[TELEGRAM] Exception while sending %s %s (%s): %s", direction, symbol, event_id, exc)
            if sent:
                telegram_sent_event_ids.add(event_id)
                log.info("[TELEGRAM] Notification sent for %s %s (%s).", direction, symbol, event_id)
            else:
                log.error("[TELEGRAM] Notification NOT sent for %s %s (%s); will retry on a later cycle.", direction, symbol, event_id)

        record_action(
            {
                "event_id": event_id,
                "symbol": symbol,
                "direction": direction,
                "score": score,
                "event_type": ev.get("event_type"),
                "telegram_sent": bool(sent),
                "telegram_required": bool(is_real_execution),
                "telegram_attempted": bool(is_real_execution and event_id not in telegram_attempted_this_cycle),
                "execution_status": execution_result.get("status"),
                "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
            }
        )

    try:
        append_shadow_health(events_path=EVENTS, health_path=HEALTH, trades_path=TRADES)
    except Exception as exc:
        log.error("[SHADOW] Health snapshot error: %s", exc)

    for tf_name, tf_rec in sorted(stats.get("by_timeframe", {}).items()):
        log.info("[TF_STATS] %s %s", tf_name.upper(), " ".join(f"{k}={v}" for k, v in tf_rec.items()))

    summary_str = " ".join(f"{k}={v}" for k, v in stats.items())
    log.info("[SUMMARY] [FORENSIC_SUMMARY] %s", summary_str)
    log.info("[SUMMARY] [ENGINE_SUMMARY] trades_this_cycle=%d %s", trades_this_cycle, summary_str)
    log.info("========== [ENGINE] CYCLE END: trades_this_cycle=%d ==========", trades_this_cycle)


if __name__ == "__main__":
    main()
