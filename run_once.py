from __future__ import annotations

import json
import hashlib
import math
import logging
import os
import time
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import requests

from event_engine.coinalyze import CoinalyzeIncompleteDataError, fetch_data
from event_engine.binance import (
    BinanceRateLimitError,
    BinanceSymbolUnavailableError,
    contract_exists as binance_contract_exists,
    fetch_klines as fetch_binance_klines,
    fetch_price as fetch_binance_price,
)
from event_engine.bingx import (
    API_KEY,
    SECRET_KEY,
    BASE_URL,
    refresh_contracts,
    get_contract,
    contract_exists as bingx_contract_exists,
    to_bx_symbol,
    fetch_klines,
    BingXRateLimitError,
    open_market,
    wait_for_position_fill_directional,
    get_positions,
    get_open_protection_directional,
    ensure_directional_protection,
    has_open_position,
    emergency_close_position,
    get_position_directional,
    _current_close_price,
)
from event_engine.signals import (
    add_cvd,
    detect_divergences,
    detect_volume_profile_divergence,
    detect_harmonic_patterns,
    detect_squeeze_release,
    detect_liquidation_squeeze,
    attach_oi_series,
    attach_funding_series,
    build_15m_trigger,
    diagnose_15m_trigger,
    check_btc_regime,
    detect_macd_4h,
    detect_ma_compression_breakout,
    detect_breakout_momentum,
    detect_donchian_retest,
    detect_liquidity_sweep_reclaim,
    detect_ema_pullback_continuation,
    detect_order_block,
    detect_breaker_block,
    detect_mitigation_block,
    detect_sfp,
    detect_liquidation_cascade_fvg,
    detect_crt,
    diagnose_15m_retest_trigger,
    validate_divergence_context,
    validate_strategy_htf_context,
    _atr as canonical_atr,
)
from event_engine.sr_context import (
    get_cached_sr_snapshot,
    evaluate_sr_room,
    SRSymbolUnavailableError,
)
from event_engine.telegram import send as send_tg, format_signal
from event_engine.formatting import format_number, format_price, format_rr, tp_price_from_pnl
from event_engine.shadow import append_shadow_health, update_divergence_shadow_state, record_divergence_shadow_open
from event_engine.trend_filter import evaluate_trend_filter
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


def _format_execution_ts(ts_ms: float | int | None) -> str:
    """Render an execution/trigger timestamp compactly in UTC logs."""
    ts = _safe_float(ts_ms, 0.0)
    if ts <= 0:
        return "n/a"
    try:
        return pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return "n/a"


def _log_execution_skip(
    *,
    symbol: str,
    direction: str,
    event_id: str,
    execution_result: dict[str, Any],
    trigger_age_min: float | None = None,
    trigger_bar_ts: float = 0.0,
) -> None:
    """Emit one explicit, machine-searchable reason for a non-opened candidate."""
    if not isinstance(execution_result, dict):
        return
    status = str(execution_result.get("status", "")).strip()
    if not status or status in {
        "opened_protected",
        "opened",
        "opened_protection_check_required",
        "opened_protection_failed",
        "ALREADY_EXECUTED_WITH_POSITION",
        "DIVERGENCE_SHADOW",
    }:
        return

    error = str(execution_result.get("error", "")).strip()
    if status == "TRIGGER_STALE":
        age_text = f"{trigger_age_min:.2f}m" if trigger_age_min is not None else "n/a"
        log.info(
            "[EXECUTION] %s %s skipped: TRIGGER_STALE age=%s limit=%.2fm trigger_bar_close_ts=%s event=%s.",
            direction,
            symbol,
            age_text,
            MAX_TRIGGER_TO_ORDER_DELAY_MIN,
            _format_execution_ts(trigger_bar_ts),
            event_id,
        )
        return

    reason = error if error else status
    log.info(
        "[EXECUTION] %s %s skipped: %s reason=%s event=%s.",
        direction,
        symbol,
        status,
        reason,
        event_id,
    )


def _coinalyze_rows_for_new_entries(rows: list[Any], complete: bool) -> list[Any]:
    """Expose only a reconciled/complete derivatives universe to entry logic."""
    return list(rows) if complete else []

DATA = Path("data")
DATA.mkdir(exist_ok=True)

EVENTS = DATA / "events.jsonl"
TRADES = DATA / "trades.jsonl"
ACTIONS = DATA / "actions.jsonl"
HEALTH = DATA / "health.jsonl"
TIMEFRAME_STATE = DATA / "timeframe_scan_state.json"
EVENT_CACHE = DATA / "recent_event_cache.json"
KLINE_RATE_LIMIT_STATE = DATA / "bingx_kline_rate_limit.json"
BINANCE_RATE_LIMIT_STATE = DATA / "binance_market_rate_limit.json"


class EventJournalPersistenceError(RuntimeError):
    """Raised when a newly detected event cannot be durably journaled."""
_KLINE_RATE_LIMIT_CACHE: dict[str, Any] = {"path": "", "cooldown_until_ms": 0, "loaded_ts": 0.0}
_BINANCE_RATE_LIMIT_CACHE: dict[str, Any] = {"path": "", "cooldown_until_ms": 0, "loaded_ts": 0.0}
MARKET_DATA_SOURCE = os.environ.get("MARKET_DATA_SOURCE", "binance").strip().lower()
# Binance is the signal/market-data venue; BingX remains the execution venue.
# Coinalyze remains an independent context/derivatives source.
BAR_CLOSE_GRACE_MIN = float(os.environ.get("BAR_CLOSE_GRACE_MIN", "2"))
# On first sight of a symbol/timeframe, replay only a bounded recent window.
# This recovers fresh events that formed before the symbol entered the liquidity
# universe, without turning first-seen symbols into unrestricted historical replays.
NEW_SYMBOL_BACKFILL_MIN = float(os.environ.get("NEW_SYMBOL_BACKFILL_MIN", "120"))

MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "0"))
MIN_VOL = float(os.environ.get("MIN_VOLUME_24H", "25000000"))

# Установлен порог Open Interest $10 000 000 по вашему запросу
MIN_OI = float(os.environ.get("MIN_OPEN_INTEREST", "10000000"))

EXECUTION_ENABLED = os.environ.get("EXECUTION_ENABLED", "false").lower() == "true"
REQUIRE_CVD = os.environ.get("REQUIRE_CVD_CONFIRMATION", "false").lower() == "true"
ENABLE_VOLUME_PROFILE_DIVERGENCE_ENGINE = os.environ.get("ENABLE_VOLUME_PROFILE_DIVERGENCE_ENGINE", "true").lower() == "true"
ENABLE_HARMONIC_PATTERN_ENGINE = os.environ.get("ENABLE_HARMONIC_PATTERN_ENGINE", "true").lower() == "true"
CVD_MIN_CONFIRMATION = float(os.environ.get("MIN_CVD24_CONFIRMATION", "55"))
REQUIRE_TRIGGER = os.environ.get("REQUIRE_15M_TRIGGER", "true").lower() == "true"
MAX_AGE = int(os.environ.get("MAX_EVENT_AGE_MIN", "60"))
MAX_TRIGGER_DELAY = float(os.environ.get("MAX_TRIGGER_DELAY_MIN", "30"))
MAX_ENTRY_DRIFT_PCT = float(os.environ.get("MAX_ENTRY_DRIFT_PCT", "2.00"))
MAX_SQUEEZE_ENTRY_DRIFT_PCT = float(os.environ.get("MAX_SQUEEZE_ENTRY_DRIFT_PCT", "2.00"))
MIN_SCORE = float(os.environ.get("MIN_SETUP_SCORE", "60"))
MIN_SHORT_SCORE = float(os.environ.get("MIN_SHORT_SETUP_SCORE", "75"))

# Trend Filter v1 is intentionally shadow-only in the current release. It is an
# independent candidate-level diagnostic and never changes event detectors, zones,
# trigger construction, execution, or accounting while mode=shadow.
TREND_FILTER_ENABLED = os.environ.get("TREND_FILTER_ENABLED", "false").lower() == "true"
TREND_FILTER_MODE = os.environ.get("TREND_FILTER_MODE", "off").strip().lower()
if TREND_FILTER_MODE not in {"off", "shadow", "enforce"}:
    TREND_FILTER_MODE = "off"
TREND_FILTER_MIN_1H_BARS = max(200, int(os.environ.get("TREND_FILTER_MIN_1H_BARS", "400")))
TREND_FILTER_MIN_4H_BARS = max(200, int(os.environ.get("TREND_FILTER_MIN_4H_BARS", "400")))
TREND_FILTER_PERSISTENCE_LOOKBACK_1H = max(1, int(os.environ.get("TREND_FILTER_PERSISTENCE_LOOKBACK_1H", "6")))
TREND_FILTER_PERSISTENCE_LOOKBACK_4H = max(1, int(os.environ.get("TREND_FILTER_PERSISTENCE_LOOKBACK_4H", "3")))
TREND_FILTER_SLOPE_LOOKBACK_4H = max(1, int(os.environ.get("TREND_FILTER_SLOPE_LOOKBACK_4H", "6")))
TREND_FILTER_HISTORY_BUFFER_BARS = max(1, int(os.environ.get("TREND_FILTER_HISTORY_BUFFER_BARS", "12")))
TREND_FILTER_REQUIRE_PERSISTENCE = os.environ.get("TREND_FILTER_REQUIRE_PERSISTENCE", "false").lower() == "true"

# Entry-quality policy is intentionally separate from the legacy diagnostic score.
# It can be rolled back via environment variables without changing event detectors
# or the research portfolio-cap policy.
ENTRY_QUALITY_GATE_ENABLED = os.environ.get("ENTRY_QUALITY_GATE_ENABLED", "false").lower() == "true"
ENTRY_QUALITY_MODE = os.environ.get("ENTRY_QUALITY_MODE", "shadow").strip().lower()
if ENTRY_QUALITY_MODE not in {"off", "shadow", "enforce"}:
    ENTRY_QUALITY_MODE = "shadow"
ENTRY_WEAK_ENGINE_BLOCK_ENABLED = os.environ.get("ENTRY_WEAK_ENGINE_BLOCK_ENABLED", "false").lower() == "true"
ENTRY_BLOCKED_EVENT_TYPES = {
    x.strip().upper()
    for x in os.environ.get(
        "ENTRY_BLOCKED_EVENT_TYPES",
        "MA_COMPRESSION_BREAKOUT,CRT_BULLISH,CRT_BEARISH,EMA_PULLBACK_CONTINUATION,LIQUIDITY_SWEEP_RECLAIM,VOLUME_PROFILE_DISTRIBUTION,BREAKER_BLOCK_BEARISH",
    ).split(",")
    if x.strip()
}
SHORT_OI_VETO_ENABLED = os.environ.get("SHORT_OI_VETO_ENABLED", "false").lower() == "true"
SHORT_OI_VETO_PCT = float(os.environ.get("SHORT_OI_VETO_PCT", "-5.0"))
COMPOUND_CVD_LIQ_VETO_ENABLED = os.environ.get("COMPOUND_CVD_LIQ_VETO_ENABLED", "false").lower() == "true"
COMPOUND_CVD_MAX = float(os.environ.get("COMPOUND_CVD_MAX", "25"))
COMPOUND_LIQ_LONG_MIN = float(os.environ.get("COMPOUND_LIQ_LONG_MIN", "150000"))
SYMBOL_LOSS_COOLDOWN_MIN = float(os.environ.get("SYMBOL_LOSS_COOLDOWN_MIN", "0"))
FIXED_STOP_LOSS_PCT = float(os.environ.get("FIXED_STOP_LOSS_PCT", "7.0"))
if not (0.0 < FIXED_STOP_LOSS_PCT < 100.0):
    FIXED_STOP_LOSS_PCT = 7.0
# Kept as compatibility/telemetry knobs; new entries use the fixed stop above.
REJECT_ATR_RISK_CLIP = os.environ.get("REJECT_ATR_RISK_CLIP", "false").lower() == "true"
MAX_ENTRY_RISK_PCT = float(os.environ.get("MAX_ENTRY_RISK_PCT", str(FIXED_STOP_LOSS_PCT)))
MAX_HOT_OI_CHG24_PCT = float(os.environ.get("MAX_HOT_OI_CHG24_PCT", "50"))
HOT_OI_SCORE_PENALTY = float(os.environ.get("HOT_OI_SCORE_PENALTY", "15"))
SYMBOL_MAX_CONSECUTIVE_LOSSES = int(os.environ.get("SYMBOL_MAX_CONSECUTIVE_LOSSES", "3"))
SYMBOL_QUARANTINE_MIN = float(os.environ.get("SYMBOL_QUARANTINE_MIN", "360"))

# Current TP ladder: closer milestones while preserving staged partial exits.
# These RR values are used consistently for new entries and restart fallback.
NORMAL_TP_RR = (0.65, 1.25, 2.00)
NORMAL_TP_FRACTIONS = (0.25, 0.40, 0.35)
NORMAL_PLANNED_WEIGHTED_RR = 1.3625
SQUEEZE_TP_RR = (1.00, 1.50, 2.00)
SQUEEZE_TP_FRACTIONS = (0.30, 0.35, 0.35)
SQUEEZE_PLANNED_WEIGHTED_RR = 1.525

# Ajay R5.41 S/R room is intentionally evaluated lazily for final candidates only.
# Current source is Binance SPOT 1H. Futures can be added later without changing
# the room/gate logic; see V10_3_SR_NOTES.md.
AJAY_SR_ROOM_ENABLED = os.environ.get("AJAY_SR_ROOM_ENABLED", "false").lower() == "true"
AJAY_SR_ROOM_MODE = os.environ.get("AJAY_SR_ROOM_MODE", "shadow").strip().lower()
if AJAY_SR_ROOM_MODE not in {"off", "shadow", "enforce"}:
    AJAY_SR_ROOM_MODE = "shadow"
AJAY_SR_REQUIRE_DATA = os.environ.get("AJAY_SR_REQUIRE_DATA", "false").lower() == "true"

# Research/VST mode deliberately disables entry-cap throttles so valid signals can be
# observed and statistically evaluated. Live-like modes keep the production limits.
_EXECUTION_MODE_HINT = os.environ.get("EXECUTION_MODE", os.environ.get("BINGX_ENV", "vst")).strip().lower()
_RESEARCH_UNLIMITED_ENTRY_MODES = {"vst", "test", "demo", "simulated"}
RESEARCH_UNLIMITED_ENTRIES = os.environ.get(
    "RESEARCH_UNLIMITED_ENTRIES",
    "true" if _EXECUTION_MODE_HINT in _RESEARCH_UNLIMITED_ENTRY_MODES else "false",
).strip().lower() == "true"
PORTFOLIO_CAP_ENABLED = os.environ.get(
    "PORTFOLIO_CAP_ENABLED",
    "false" if RESEARCH_UNLIMITED_ENTRIES else "true",
).strip().lower() == "true"
MAX_TRADES = int(os.environ.get("MAX_TRADES_PER_CYCLE", "0" if RESEARCH_UNLIMITED_ENTRIES else "3"))
# Prevent rapid re-entry/churn on the same instrument, including opposite-direction flips.
SYMBOL_ENTRY_COOLDOWN_MIN = float(os.environ.get("SYMBOL_ENTRY_COOLDOWN_MIN", "15"))
SQUEEZE_SYMBOL_ENTRY_COOLDOWN_MIN = float(os.environ.get("SQUEEZE_SYMBOL_ENTRY_COOLDOWN_MIN", "45"))
# Funding values are in percentage points as parsed from Coinalyze
# (e.g. 0.05 == +0.05%, -0.05 == -0.05%).
MAX_SHORT_SQUEEZE_ADVERSE_FUNDING = float(os.environ.get("MAX_SHORT_SQUEEZE_ADVERSE_FUNDING", "-0.10"))
MAX_LONG_SQUEEZE_ADVERSE_FUNDING = float(os.environ.get("MAX_LONG_SQUEEZE_ADVERSE_FUNDING", "0.10"))
EXTREME_SHORT_FUNDING = float(os.environ.get("EXTREME_SHORT_FUNDING", "-0.50"))
EXTREME_LONG_FUNDING = float(os.environ.get("EXTREME_LONG_FUNDING", "0.50"))
FUNDING_REQUIRED = os.environ.get("FUNDING_REQUIRED", "false").strip().lower() == "true"
MAX_TRIGGER_TO_ORDER_DELAY_MIN = float(os.environ.get("MAX_TRIGGER_TO_ORDER_DELAY_MIN", "8"))
DIVERGENCE_POST_CONFIRM_MAX_AGE_MIN = float(os.environ.get("MAX_DIVERGENCE_POST_CONFIRM_AGE_MIN", os.environ.get("MAX_DIVERGENCE_FORMATION_AGE_MIN", "45")))
MA_COMPRESSION_RETEST_MAX_DELAY_MIN = float(os.environ.get("MA_COMPRESSION_RETEST_MAX_DELAY_MIN", "120"))
BREAKOUT_MOMENTUM_RETEST_MAX_DELAY_MIN = float(os.environ.get("BREAKOUT_MOMENTUM_RETEST_MAX_DELAY_MIN", "90"))
VOLATILITY_SQUEEZE_RETEST_MAX_DELAY_MIN = float(os.environ.get("VOLATILITY_SQUEEZE_RETEST_MAX_DELAY_MIN", "60"))
ORDER_BLOCK_RETEST_MAX_DELAY_MIN = float(os.environ.get("ORDER_BLOCK_RETEST_MAX_DELAY_MIN", "120"))
BREAKER_BLOCK_RETEST_MAX_DELAY_MIN = float(os.environ.get("BREAKER_BLOCK_RETEST_MAX_DELAY_MIN", "120"))
MITIGATION_BLOCK_RETEST_MAX_DELAY_MIN = float(os.environ.get("MITIGATION_BLOCK_RETEST_MAX_DELAY_MIN", "120"))
SFP_RETEST_MAX_DELAY_MIN = float(os.environ.get("SFP_RETEST_MAX_DELAY_MIN", "60"))
# LIQUIDITY_SWEEP emits requires_retest=True but had no window entry, so it fell
# back to MAX_TRIGGER_DELAY (30 min) -- two 15m candidate bars against the 4-8
# every other retest engine gets. 60 min matches its nearest sibling, SFP.
LIQUIDITY_SWEEP_RETEST_MAX_DELAY_MIN = float(os.environ.get("LIQUIDITY_SWEEP_RETEST_MAX_DELAY_MIN", "60"))
LIQUIDATION_CASCADE_FVG_RETEST_MAX_DELAY_MIN = float(os.environ.get("LIQUIDATION_CASCADE_FVG_RETEST_MAX_DELAY_MIN", "90"))
CRT_RETEST_MAX_DELAY_MIN = float(os.environ.get("CRT_RETEST_MAX_DELAY_MIN", "60"))
REQUIRE_4H_CONTEXT_FOR_1H = os.environ.get("REQUIRE_4H_CONTEXT_FOR_1H", "true").lower() == "true"
ENABLE_DIVERGENCE_ENGINE = os.environ.get("ENABLE_DIVERGENCE_ENGINE", "true").lower() == "true"
DIVERGENCE_SHADOW_ONLY = os.environ.get("DIVERGENCE_SHADOW_ONLY", "false").lower() == "true"
DIVERGENCE_SHADOW_STATE = DATA / "divergence_shadow_trades.json"
MAX_PRE_ORDER_DRIFT_REJECTIONS = max(1, int(os.environ.get("MAX_PRE_ORDER_DRIFT_REJECTIONS", "3")))
MAX_CROSS_EXCHANGE_DRIFT_REJECTIONS = max(1, int(os.environ.get("MAX_CROSS_EXCHANGE_DRIFT_REJECTIONS", "3")))
MAX_SR_DATA_REJECTIONS = max(1, int(os.environ.get("MAX_SR_DATA_REJECTIONS", "3")))
CROSS_EXCHANGE_PRICE_GUARD_ENABLED = os.environ.get("CROSS_EXCHANGE_PRICE_GUARD_ENABLED", "true").lower() == "true"
MAX_CROSS_EXCHANGE_DRIFT_PCT = float(os.environ.get("MAX_CROSS_EXCHANGE_DRIFT_PCT", "1.00"))
ENABLE_MACD_4H_ENGINE = os.environ.get("ENABLE_MACD_4H_ENGINE", "true").lower() == "true"
ENABLE_MA_COMPRESSION_ENGINE = os.environ.get("ENABLE_MA_COMPRESSION_ENGINE", "true").lower() == "true"
ENABLE_BREAKOUT_MOMENTUM_ENGINE = os.environ.get("ENABLE_BREAKOUT_MOMENTUM_ENGINE", "true").lower() == "true"
ENABLE_DONCHIAN_RETEST_ENGINE = os.environ.get("ENABLE_DONCHIAN_RETEST_ENGINE", "true").lower() == "true"
ENABLE_LIQUIDITY_SWEEP_ENGINE = os.environ.get("ENABLE_LIQUIDITY_SWEEP_ENGINE", "true").lower() == "true"
ENABLE_EMA_PULLBACK_ENGINE = os.environ.get("ENABLE_EMA_PULLBACK_ENGINE", "true").lower() == "true"
ENABLE_ORDER_BLOCK_ENGINE = os.environ.get("ENABLE_ORDER_BLOCK_ENGINE", "true").lower() == "true"
ENABLE_BREAKER_BLOCK_ENGINE = os.environ.get("ENABLE_BREAKER_BLOCK_ENGINE", "true").lower() == "true"
ENABLE_MITIGATION_BLOCK_ENGINE = os.environ.get("ENABLE_MITIGATION_BLOCK_ENGINE", "true").lower() == "true"
ENABLE_MITIGATION_BLOCK_BULLISH_ENGINE = os.environ.get("ENABLE_MITIGATION_BLOCK_BULLISH_ENGINE", "false").lower() == "true"
ENABLE_SFP_ENGINE = os.environ.get("ENABLE_SFP_ENGINE", "true").lower() == "true"
ENABLE_LIQUIDATION_CASCADE_FVG_ENGINE = os.environ.get("ENABLE_LIQUIDATION_CASCADE_FVG_ENGINE", "false").lower() == "true"
ENABLE_CRT_ENGINE = os.environ.get("ENABLE_CRT_ENGINE", "true").lower() == "true"
ENABLE_LIQUIDATION_SQUEEZE_ENGINE = os.environ.get("LIQ_SQUEEZE_ENGINE_ENABLED", "false").lower() == "true"
MAX_ACTIVE_TRADES = int(os.environ.get("MAX_ACTIVE_TRADES", "12"))
MAX_ACTIVE_LONGS = int(os.environ.get("MAX_ACTIVE_LONGS", "6"))
MAX_ACTIVE_SHORTS = int(os.environ.get("MAX_ACTIVE_SHORTS", "6"))
EXECUTION_MODE = _EXECUTION_MODE_HINT
POSITION_MODE = os.environ.get("BINGX_POSITION_MODE", "HEDGE").strip().upper()

EXPECTED_EVENT_ENGINES = {
    "DIVERGENCE", "VOLATILITY_SQUEEZE", "MACD_4H", "MA_COMPRESSION",
    "BREAKOUT_MOMENTUM", "DONCHIAN_RETEST", "LIQUIDITY_SWEEP", "EMA_PULLBACK",
    "ORDER_BLOCK", "BREAKER_BLOCK", "MITIGATION_BLOCK", "SFP",
    "LIQUIDATION_CASCADE_FVG", "CRT", "VOLUME_PROFILE_DIVERGENCE", "HARMONIC_PATTERN",
}


VALID_EXECUTION_MODES = {
    "vst", "test", "demo", "simulated",
    "live", "prod", "production", "prod-live",
}
VST_EXECUTION_MODES = {"vst", "test", "demo", "simulated"}
LIVE_EXECUTION_MODES = {"live", "prod", "production", "prod-live"}
BINGX_VST_BASE_URL = "https://open-api-vst.bingx.com"
BINGX_LIVE_BASE_URL = "https://open-api.bingx.com"


def _validate_execution_config() -> tuple[bool, str]:
    if not EXECUTION_ENABLED:
        return True, "EXECUTION_DISABLED"
    if not API_KEY or not SECRET_KEY:
        return False, "BINGX credentials are missing while EXECUTION_ENABLED=true"

    mode = str(EXECUTION_MODE or "vst").strip().lower()
    base = str(BASE_URL or "").strip().lower().rstrip("/")

    if mode not in VALID_EXECUTION_MODES:
        return False, f"Unsupported EXECUTION_MODE={mode!r}"

    if mode in VST_EXECUTION_MODES and base != BINGX_VST_BASE_URL:
        return False, (
            f"EXECUTION_MODE={mode} requires BingX VST base URL "
            f"{BINGX_VST_BASE_URL}, got {BASE_URL!r}"
        )

    if mode in LIVE_EXECUTION_MODES:
        if base != BINGX_LIVE_BASE_URL:
            return False, (
                f"EXECUTION_MODE={mode} requires BingX live base URL "
                f"{BINGX_LIVE_BASE_URL}, got {BASE_URL!r}"
            )
        if os.environ.get("ALLOW_LIVE_TRADING", "false").strip().lower() != "true":
            return False, "Live execution requires explicit ALLOW_LIVE_TRADING=true"

    if POSITION_MODE != "HEDGE":
        return False, f"This engine requires BINGX_POSITION_MODE=HEDGE, got {POSITION_MODE!r}"

    return True, "OK"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _save_json_atomic(path: Path, obj: Any) -> None:
    """Durably replace a JSON state file.

    Scheduler state and the event cache are recovery state.  Write the complete
    temporary file, flush it to disk, atomically replace the destination, then
    fsync the parent directory so the rename itself is durable as well.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(obj, ensure_ascii=False, indent=2)
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            # Directory fsync is an additional durability barrier.  The atomic
            # rename above remains the portability baseline on systems where
            # opening/fsyncing the directory is unavailable.
            pass
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


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


def _btc_regime_snapshot(btc_1h_df) -> dict[str, Any]:
    """BTC state at decision time, so regime can be analysed instead of guessed.

    check_btc_regime consumed this and threw it away; btc_corr7d is a 7-day
    correlation, not a regime, and must not be substituted for it.
    """
    out: dict[str, Any] = {"btc_chg_1h_pct": None, "btc_chg_4h_pct": None,
                           "btc_close": None, "btc_available": False}
    try:
        if btc_1h_df is None or len(btc_1h_df) < 5 or "close" not in btc_1h_df.columns:
            return out
        close = pd.to_numeric(btc_1h_df["close"], errors="coerce")
        last, prev_1h, prev_4h = float(close.iloc[-1]), float(close.iloc[-2]), float(close.iloc[-5])
        if (
            not all(math.isfinite(value) for value in (last, prev_1h, prev_4h))
            or last <= 0
            or prev_1h <= 0
            or prev_4h <= 0
        ):
            return out
        out.update({
            "btc_close": last,
            "btc_chg_1h_pct": (last - prev_1h) / prev_1h * 100.0,
            "btc_chg_4h_pct": (last - prev_4h) / prev_4h * 100.0,
            "btc_available": True,
        })
    except Exception:
        return out
    return out


def _entry_context(row: Any, ev: dict, btc_regime: dict[str, Any]) -> dict[str, Any]:
    """Freeze every regime input that the filters used but never journaled."""
    fact = ev.get("event_fact") if isinstance(ev.get("event_fact"), dict) else {}
    def _num(name):
        try:
            value = getattr(row, name, None)
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
    return {
        **btc_regime,
        "fr_oiw": _num("fr_oiw"),
        "pfr_oiw": _num("pfr_oiw"),
        "oi": _num("oi"),
        "oi_chg24_pct": _num("oi_chg24_pct"),
        "oi_chg4h_pct": _num("oi_chg4h_pct"),
        "volume24": _num("volume24"),
        "cvd24": _num("cvd24"),
        "ls_accounts": _num("ls_accounts"),
        "liq_long24": _num("liq_long24"),
        "liq_short24": _num("liq_short24"),
        "btc_corr7d": _num("btc_corr7d"),
        "htf_trend": fact.get("htf_trend"),
        "htf_trend_ok": fact.get("htf_trend_ok"),
        "context_source": fact.get("context_source"),
    }


def _is_divergence_event(ev: dict) -> bool:
    """Engine identity is authoritative; event_type carries the indicator subtype.

    Divergence events are emitted as REGULAR_/HIDDEN_<INDICATOR> (e.g.
    REGULAR_BULLISH_RSI), never as the literal string "DIVERGENCE". Comparing
    event_type to "DIVERGENCE" matched nothing, which silently disabled
    DIVERGENCE_SHADOW_ONLY and ENABLE_DIVERGENCE_ENGINE.
    """
    fact = ev.get("event_fact") if isinstance(ev.get("event_fact"), dict) else {}
    if str(fact.get("engine") or "").upper() == "DIVERGENCE":
        return True
    return str(ev.get("event_type") or "").upper().startswith(("REGULAR_", "HIDDEN_"))


def _is_liquidation_squeeze_event(event_type: str) -> bool:
    return str(event_type or "").upper() in {"SHORT_SQUEEZE", "LONG_SQUEEZE"}


def _is_volatility_squeeze_event(event_type: str) -> bool:
    return str(event_type or "").upper() == "VOLATILITY_SQUEEZE_RELEASE"


def _is_squeeze_event(event_type: str) -> bool:
    return _is_liquidation_squeeze_event(event_type) or _is_volatility_squeeze_event(event_type)


def _event_trigger_max_delay_min(ev: dict) -> float:
    fact = ev.get("event_fact") if isinstance(ev.get("event_fact"), dict) else {}
    engine = str(fact.get("engine") or "").upper()
    event_type = str(ev.get("event_type") or "").upper()
    if engine == "MA_COMPRESSION":
        return MA_COMPRESSION_RETEST_MAX_DELAY_MIN
    if engine == "BREAKOUT_MOMENTUM":
        return BREAKOUT_MOMENTUM_RETEST_MAX_DELAY_MIN
    if engine == "DONCHIAN_RETEST":
        return float(os.environ.get("DONCHIAN_RETEST_MAX_DELAY_MIN", "120"))
    if engine == "EMA_PULLBACK":
        return float(os.environ.get("EMA_PULLBACK_RETEST_MAX_DELAY_MIN", "120"))
    if engine == "ORDER_BLOCK":
        return ORDER_BLOCK_RETEST_MAX_DELAY_MIN
    if engine == "BREAKER_BLOCK":
        return BREAKER_BLOCK_RETEST_MAX_DELAY_MIN
    if engine == "MITIGATION_BLOCK":
        return MITIGATION_BLOCK_RETEST_MAX_DELAY_MIN
    if engine == "SFP":
        return SFP_RETEST_MAX_DELAY_MIN
    if engine == "LIQUIDITY_SWEEP":
        return LIQUIDITY_SWEEP_RETEST_MAX_DELAY_MIN
    if engine == "LIQUIDATION_CASCADE_FVG":
        return LIQUIDATION_CASCADE_FVG_RETEST_MAX_DELAY_MIN
    if engine == "CRT":
        return CRT_RETEST_MAX_DELAY_MIN
    if event_type == "VOLATILITY_SQUEEZE_RELEASE":
        return VOLATILITY_SQUEEZE_RETEST_MAX_DELAY_MIN
    return MAX_TRIGGER_DELAY


def _event_max_age_min(ev: dict) -> float:
    fact = ev.get("event_fact") if isinstance(ev.get("event_fact"), dict) else {}
    if fact.get("requires_retest"):
        return max(float(MAX_AGE), _event_trigger_max_delay_min(ev))
    event_type = str(ev.get("event_type") or "").upper()
    if "REGULAR_" in event_type or "HIDDEN_" in event_type:
        return max(float(MAX_AGE), DIVERGENCE_POST_CONFIRM_MAX_AGE_MIN)
    return float(MAX_AGE)


def _event_is_fresh(ev: dict, now_ms: int, max_age_min: int) -> bool:
    try:
        ts = int(ev.get("timestamps", {}).get("detected_at_ts", 0) or 0)
    except (TypeError, ValueError):
        return False
    age_min = (now_ms - ts) / 60_000.0
    return 0 <= age_min <= max_age_min


def _divergence_post_confirmation_age(ev: dict, now_ms: int) -> tuple[bool, float, float, float]:
    """Return timing validity plus confirmation lag, formation age and post-confirm age."""
    try:
        pivot2_ts = int(ev.get("timestamps", {}).get("pivot_2_ts", 0) or 0)
        confirm_ts = int(ev.get("timestamps", {}).get("detected_at_ts", 0) or 0)
    except (TypeError, ValueError):
        return False, float("inf"), float("inf"), float("inf")
    if pivot2_ts <= 0 or confirm_ts <= 0:
        return False, float("inf"), float("inf"), float("inf")
    confirmation_lag = (confirm_ts - pivot2_ts) / 60_000.0
    formation_age = (now_ms - pivot2_ts) / 60_000.0
    post_confirmation_age = (now_ms - confirm_ts) / 60_000.0
    valid = confirmation_lag >= 0 and post_confirmation_age >= 0
    return valid, formation_age, post_confirmation_age, confirmation_lag


def _load_cached_events(now_ms: int | None = None, terminal_ids: set[str] | None = None) -> list[dict]:
    """Load live event cache and reconcile it from the durable event journal.

    The journal is the source of truth; the JSON cache is only a derived working
    set.  Reconciliation makes a crash after journal append but before cache save
    recoverable on the next run.
    """
    data = _load_json(EVENT_CACHE, {})
    cached = data.get("events", []) if isinstance(data, dict) else []
    events = [e for e in cached if isinstance(e, dict) and e.get("event_id")]
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    terminal_ids = {str(x) for x in (terminal_ids or set()) if x}

    recovered: list[dict] = []
    if EVENTS.exists():
        try:
            with EVENTS.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(ev, dict):
                        continue
                    eid = ev.get("event_id")
                    if not eid or str(eid) in terminal_ids:
                        continue
                    if _event_is_fresh(ev, now_ms, _event_max_age_min(ev)):
                        recovered.append(ev)
        except OSError as exc:
            log.warning("[EVENT_CACHE] Journal reconciliation read failed: %s", exc)

    return _merge_event_cache(events, recovered)


def _load_timeframe_scan_state() -> dict:
    """Load per-symbol/per-timeframe scan state with explicit failure telemetry.

    A missing or malformed state file is fail-safe for trading, but it must never
    be silent: returning ``symbols={}`` causes a full universe rescan. The caller
    therefore gets a structured empty state plus a log explaining why the state
    was not reusable.
    """
    if not TIMEFRAME_STATE.exists():
        log.warning("[SCAN_STATE] missing file path=%s; starting with empty state", TIMEFRAME_STATE)
        return {"version": 2, "symbols": {}}

    try:
        raw = json.loads(TIMEFRAME_STATE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.error("[SCAN_STATE] invalid JSON path=%s line=%d col=%d; starting empty state",
                  TIMEFRAME_STATE, exc.lineno, exc.colno)
        return {"version": 2, "symbols": {}}
    except OSError as exc:
        log.error("[SCAN_STATE] read failed path=%s error=%s; starting empty state",
                  TIMEFRAME_STATE, exc)
        return {"version": 2, "symbols": {}}

    if not isinstance(raw, dict):
        log.error("[SCAN_STATE] invalid root type=%s path=%s; starting empty state",
                  type(raw).__name__, TIMEFRAME_STATE)
        return {"version": 2, "symbols": {}}

    if raw.get("version") != 2:
        log.warning("[SCAN_STATE] unsupported version=%r path=%s; rebuilding per-symbol state",
                    raw.get("version"), TIMEFRAME_STATE)
        return {"version": 2, "symbols": {}}

    symbols = raw.get("symbols")
    if isinstance(symbols, dict):
        # Copy only the fields the scheduler owns; malformed per-symbol records
        # are handled individually by _symbol_scan_due and never suppress a scan.
        return {"version": 2, "symbols": symbols}

    # Legacy format used one bucket for the whole universe. Do not copy it to
    # symbols: doing so would hide newly discovered symbols. Start them from
    # scratch once, then persist their own last-scanned closed bar.
    log.warning("[SCAN_STATE] unsupported/legacy schema path=%s keys=%s; rebuilding per-symbol state",
                TIMEFRAME_STATE, sorted(str(k) for k in raw.keys())[:20])
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
FUNDING_HISTORY = DATA / "funding_history.json"
_OI_HIST_CACHE: dict[str, Any] = {"ts": 0.0, "data": {}, "path": ""}
_OI_HIST_CACHE_TTL = 30.0
_OI_HIST_MAX_BUCKETS = 500
_FUNDING_HIST_CACHE: dict[str, Any] = {"ts": 0.0, "data": {}, "path": ""}
_FUNDING_HIST_CACHE_TTL = 30.0
_FUNDING_HIST_MAX_BUCKETS = 500


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
        # Keep the in-memory cache coherent with the just-persisted snapshot.
        # Invalidating the timestamp alone can still leave callers with stale
        # data during the same monotonic tick if they read immediately.
        _OI_HIST_CACHE["data"] = history
        _OI_HIST_CACHE["ts"] = time.monotonic()
    _OI_HIST_CACHE["path"] = str(OI_HISTORY.resolve())
    return updated


def _load_funding_history() -> dict[str, dict[str, float]]:
    """Cached funding-rate snapshots keyed by symbol and 1h bucket."""
    now = time.monotonic()
    cache_path = str(FUNDING_HISTORY.resolve())
    if (
        _FUNDING_HIST_CACHE.get("path") == cache_path
        and now - float(_FUNDING_HIST_CACHE.get("ts", 0.0)) < _FUNDING_HIST_CACHE_TTL
        and "data" in _FUNDING_HIST_CACHE
    ):
        return _FUNDING_HIST_CACHE["data"]
    raw = _load_json(FUNDING_HISTORY, {})
    data = raw if isinstance(raw, dict) else {}
    _FUNDING_HIST_CACHE["ts"] = now
    _FUNDING_HIST_CACHE["path"] = cache_path
    _FUNDING_HIST_CACHE["data"] = data
    return data


def _record_funding_snapshots(rows: list[Any], now_ms: int) -> int:
    """Persist the latest available OI-weighted funding rate per 1h bucket."""
    if not rows:
        return 0
    history = _load_json(FUNDING_HISTORY, {})
    if not isinstance(history, dict):
        history = {}
    bucket = str(int(now_ms // 3_600_000))
    updated = 0
    for row in rows:
        symbol = str(getattr(row, "symbol", "") or "").upper()
        raw_fr = getattr(row, "fr_oiw", None)
        if not symbol or raw_fr is None:
            continue
        try:
            fr = float(raw_fr)
        except (TypeError, ValueError):
            continue
        if not (-float("inf") < fr < float("inf")):
            continue
        rec = history.setdefault(symbol, {})
        if not isinstance(rec, dict):
            rec = history[symbol] = {}
        rec[bucket] = fr
        if len(rec) > _FUNDING_HIST_MAX_BUCKETS:
            for key in sorted(rec, key=lambda k: int(k))[: len(rec) - _FUNDING_HIST_MAX_BUCKETS]:
                del rec[key]
        updated += 1
    if updated:
        _save_json_atomic(FUNDING_HISTORY, history)
        _FUNDING_HIST_CACHE["data"] = history
        _FUNDING_HIST_CACHE["ts"] = time.monotonic()
    _FUNDING_HIST_CACHE["path"] = str(FUNDING_HISTORY.resolve())
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

def _load_kline_rate_limit_state() -> dict[str, Any]:
    """Load the persisted BingX Kline cooldown shared across workflow runs."""
    cache_path = str(KLINE_RATE_LIMIT_STATE.resolve())
    now = time.monotonic()
    if _KLINE_RATE_LIMIT_CACHE.get("path") == cache_path and now - float(_KLINE_RATE_LIMIT_CACHE.get("loaded_ts", 0.0)) < 5.0:
        return dict(_KLINE_RATE_LIMIT_CACHE)
    raw = _load_json(KLINE_RATE_LIMIT_STATE, {})
    state = raw if isinstance(raw, dict) else {}
    try:
        cooldown_until_ms = int(state.get("cooldown_until_ms", 0) or 0)
    except (TypeError, ValueError):
        cooldown_until_ms = 0
    _KLINE_RATE_LIMIT_CACHE.update(path=cache_path, cooldown_until_ms=max(0, cooldown_until_ms), loaded_ts=now)
    return dict(_KLINE_RATE_LIMIT_CACHE)


def _set_kline_rate_limit_cooldown(exc: BingXRateLimitError) -> int:
    """Persist a server-directed Kline cooldown and return its absolute timestamp."""
    now_ms = int(time.time() * 1000)
    retry_after_ms = int(exc.retry_after_ms or 0)
    try:
        fallback_sec = float(os.environ.get("BINGX_KLINE_RATE_LIMIT_FALLBACK_SEC", "900"))
    except (TypeError, ValueError):
        fallback_sec = 900.0
    fallback_ms = now_ms + int(max(60.0, fallback_sec) * 1000)
    previous = _load_kline_rate_limit_state()
    cooldown_until_ms = retry_after_ms if retry_after_ms > now_ms + 1000 else fallback_ms
    cooldown_until_ms = max(cooldown_until_ms, int(previous.get("cooldown_until_ms", 0) or 0))
    state = {
        "cooldown_until_ms": cooldown_until_ms,
        "saved_at_ms": now_ms,
        "code": exc.code,
        "reason": str(exc),
    }
    _save_json_atomic(KLINE_RATE_LIMIT_STATE, state)
    _KLINE_RATE_LIMIT_CACHE.update(path=str(KLINE_RATE_LIMIT_STATE.resolve()), cooldown_until_ms=cooldown_until_ms, loaded_ts=time.monotonic())
    remaining = max(0, cooldown_until_ms - now_ms)
    log.error("[BINGX_KLINE] RATE_LIMIT cooldown=%.1fs; no further Kline requests will be attempted until %d", remaining / 1000.0, cooldown_until_ms)
    return cooldown_until_ms


def _raise_if_kline_rate_limited(symbol: str, timeframe: str) -> None:
    state = _load_kline_rate_limit_state()
    cooldown_until_ms = int(state.get("cooldown_until_ms", 0) or 0)
    now_ms = int(time.time() * 1000)
    if cooldown_until_ms > now_ms:
        remaining = cooldown_until_ms - now_ms
        raise BingXRateLimitError(
            f"[BINGX] Kline endpoint cooldown active for {symbol}/{timeframe}; retry after {cooldown_until_ms} ({remaining / 1000.0:.1f}s remaining)",
            code=state.get("code"), retry_after_ms=cooldown_until_ms,
        )


def _load_binance_rate_limit_state() -> dict[str, Any]:
    """Load persisted Binance market-data cooldown shared across workflow runs."""
    cache_path = str(BINANCE_RATE_LIMIT_STATE.resolve())
    now = time.monotonic()
    if _BINANCE_RATE_LIMIT_CACHE.get("path") == cache_path and now - float(_BINANCE_RATE_LIMIT_CACHE.get("loaded_ts", 0.0)) < 5.0:
        return dict(_BINANCE_RATE_LIMIT_CACHE)
    raw = _load_json(BINANCE_RATE_LIMIT_STATE, {})
    state = raw if isinstance(raw, dict) else {}
    try:
        cooldown_until_ms = int(state.get("cooldown_until_ms", 0) or 0)
    except (TypeError, ValueError):
        cooldown_until_ms = 0
    _BINANCE_RATE_LIMIT_CACHE.update(path=cache_path, cooldown_until_ms=max(0, cooldown_until_ms), loaded_ts=now)
    return dict(_BINANCE_RATE_LIMIT_CACHE)


def _set_binance_rate_limit_cooldown(exc: BinanceRateLimitError) -> int:
    now_ms = int(time.time() * 1000)
    retry_after_ms = int(exc.retry_after_ms or 0)
    fallback_ms = now_ms + int(60_000)
    previous = _load_binance_rate_limit_state()
    cooldown_until_ms = retry_after_ms if retry_after_ms > now_ms + 1000 else fallback_ms
    cooldown_until_ms = max(cooldown_until_ms, int(previous.get("cooldown_until_ms", 0) or 0))
    state = {
        "cooldown_until_ms": cooldown_until_ms,
        "saved_at_ms": now_ms,
        "code": exc.code,
        "reason": str(exc),
    }
    _save_json_atomic(BINANCE_RATE_LIMIT_STATE, state)
    _BINANCE_RATE_LIMIT_CACHE.update(
        path=str(BINANCE_RATE_LIMIT_STATE.resolve()),
        cooldown_until_ms=cooldown_until_ms,
        loaded_ts=time.monotonic(),
    )
    remaining = max(0, cooldown_until_ms - now_ms)
    log.error(
        "[BINANCE_KLINE] RATE_LIMIT cooldown=%.1fs; no further market-data requests until %d",
        remaining / 1000.0, cooldown_until_ms,
    )
    return cooldown_until_ms


def _raise_if_binance_rate_limited(symbol: str, timeframe: str) -> None:
    state = _load_binance_rate_limit_state()
    cooldown_until_ms = int(state.get("cooldown_until_ms", 0) or 0)
    now_ms = int(time.time() * 1000)
    if cooldown_until_ms > now_ms:
        remaining = cooldown_until_ms - now_ms
        raise BinanceRateLimitError(
            f"[BINANCE] market-data cooldown active for {symbol}/{timeframe}; retry after {cooldown_until_ms} ({remaining / 1000.0:.1f}s remaining)",
            code=state.get("code"),
            retry_after_ms=cooldown_until_ms,
        )


def _fetch_market_klines_scan(symbol: str, timeframe: str, limit: int) -> list[dict]:
    """Fetch signal-side candles from the configured market-data source.

    Production default is Binance. BingX remains available only when explicitly
    selected via MARKET_DATA_SOURCE=bingx, which keeps execution and market-data
    concerns separate without removing the existing BingX client.
    """
    source = MARKET_DATA_SOURCE
    if source == "bingx":
        return _fetch_klines_scan(symbol, timeframe, limit)
    if source != "binance":
        raise RuntimeError(f"Unsupported MARKET_DATA_SOURCE={source!r}; expected 'binance' or 'bingx'")

    _raise_if_binance_rate_limited(symbol, timeframe)
    try:
        result = fetch_binance_klines(symbol, timeframe, limit)
        log.info("[BINANCE_KLINE] END %s/%s rows=%d.", symbol, timeframe, len(result or []))
        return result
    except BinanceRateLimitError as exc:
        _set_binance_rate_limit_cooldown(exc)
        raise


def _is_btc_symbol(symbol: str) -> bool:
    """Recognize BTC consistently across engine and exchange symbol forms."""
    normalized = str(symbol or "").strip().upper().replace("/", "").replace("-", "")
    if normalized.endswith("USDT"):
        normalized = normalized[:-4]
    return normalized == "BTC"


def _fetch_market_price(symbol: str) -> float:
    if MARKET_DATA_SOURCE == "bingx":
        return _current_close_price(symbol)
    if MARKET_DATA_SOURCE != "binance":
        raise RuntimeError(f"Unsupported MARKET_DATA_SOURCE={MARKET_DATA_SOURCE!r}")
    _raise_if_binance_rate_limited(symbol, "ticker")
    try:
        return fetch_binance_price(symbol)
    except BinanceRateLimitError as exc:
        _set_binance_rate_limit_cooldown(exc)
        raise


def _file_lock_pace(lock_dir: Path, min_interval: float) -> float:
    """Backward-compatible name; pacing is intentionally process-local.

    The old implementation persisted time.monotonic() in a file. That is
    invalid across ephemeral GitHub Actions runners because monotonic clocks
    are not comparable between different VMs. The lock_dir argument is retained
    only for API/test compatibility and is intentionally unused. Global
    serialization for the production GitHub workflow is provided by the
    workflow-level ``concurrency.group``; this helper must not be mistaken for
    a cross-process or cross-runner rate limiter.
    """
    _ = lock_dir
    _acquire_scan_slot(min_interval)
    return time.monotonic()


def _fetch_klines_scan(symbol: str, timeframe: str, limit: int) -> list[dict]:
    """Fetch Klines with pacing and bounded transient-error retry.

    BingX rate-limit responses are a hard circuit-breaker condition: they are
    persisted with the server-provided retry timestamp and never retried blindly.
    Ordinary network/transient failures retain the bounded retry path.
    """
    _raise_if_kline_rate_limited(symbol, timeframe)
    min_interval = float(os.environ.get("BINGX_KLINE_SCAN_MIN_INTERVAL_SEC", "1.25"))

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
        except BingXRateLimitError as exc:
            _set_kline_rate_limit_cooldown(exc)
            raise
        except (RuntimeError, requests.RequestException, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 >= max_attempts:
                break
            delay = max(0.0, backoff_base) * (2 ** attempt)
            log.warning("[BINGX] Kline retry %d/%d for %s/%s after error: %s; sleeping %.1fs", attempt + 1, max_attempts - 1, symbol, timeframe, exc, delay)
            if delay:
                time.sleep(delay)
    raise RuntimeError(f"Kline fetch failed for {symbol}/{timeframe} after {max_attempts} attempts: {last_error}") from last_error


def _record_trigger_failure(stats: dict, tf_stats: dict, reason: str) -> None:
    """Classify a 15m trigger rejection without treating a missing window as bad data."""
    stats["rejected_trigger"] += 1
    if reason in {"no_trigger_window", "no_retest_window"}:
        stats["trigger_no_window"] += 1
        tf_stats["trigger_no_window"] += 1
    elif reason == "breakout_failed":
        stats["trigger_breakout_failed"] += 1
        tf_stats["trigger_breakout_failed"] += 1
    elif reason == "volume_failed":
        stats["trigger_volume_failed"] += 1
        tf_stats["trigger_volume_failed"] += 1
    else:
        stats["trigger_data_failed"] += 1
        tf_stats["trigger_data_failed"] += 1


def _record_scan_error(stats: dict, stage: str, tf_stats: dict | None = None) -> None:
    """Record a cycle scan error and preserve its stage for forensic summaries."""
    stats["scan_errors"] += 1
    by_stage = stats.setdefault("scan_errors_by_stage", {})
    by_stage[stage] = int(by_stage.get(stage, 0) or 0) + 1
    if tf_stats is not None:
        tf_stats["scan_errors"] += 1


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
        "trigger_data_failed": 0,
        "trigger_direction_failed": 0,
        "rejected_btc": 0,
        "rejected_funding": 0,
        "rejected_cvd": 0,
        "rejected_score": 0,
        "rejected_entry_quality": 0,
        "rejected_weak_engine": 0,
        "rejected_short_oi": 0,
        "rejected_cvd_liq": 0,
        "rejected_recent_loss": 0,
        "rejected_risk_too_wide": 0,
        "rejected_single_tp": 0,
        "entry_quality_shadow_flags": 0,
        "rejected_entry_drift": 0,
        "rejected_trigger_stale": 0,
        "rejected_portfolio_cap": 0,
        "rejected_hot_oi": 0,
        "rejected_symbol_quarantine": 0,
        "valid_signals": 0,
    })
    return rec


def _timeframe_interval_ms(timeframe: str) -> int:
    value = str(timeframe or "").lower().strip()
    intervals = {"1h": 3_600_000, "4h": 14_400_000}
    try:
        return intervals[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported scan timeframe: {timeframe!r}") from exc


def _scan_buckets_to_process(
    scan_state: dict,
    symbol: str,
    timeframe: str,
    completed_bucket: int,
    available_buckets: set[int],
) -> list[int]:
    """Return contiguous closed buckets that can be replayed safely.

    A new symbol/timeframe replays a bounded, contiguous recent window so a fresh
    event formed before the symbol entered the universe is still recoverable.
    An already-known symbol replays every missing completed bucket for which the
    fetched history is available. We never jump over a missing bucket.
    """
    symbols = scan_state.get("symbols", {}) if isinstance(scan_state, dict) else {}
    rec = symbols.get(symbol) if isinstance(symbols, dict) else None
    if not isinstance(rec, dict) or rec.get(timeframe) in (None, ""):
        eligible = {bucket for bucket in available_buckets if bucket <= completed_bucket}
        if not eligible:
            return []
        # Replay only a bounded recent window, moving backwards from the newest
        # available bucket and stopping at the first historical gap. This keeps
        # first-seen symbols causal and bounded while covering the configured
        # fresh-event/retest lifetime.
        max_buckets = max(1, int(math.ceil(NEW_SYMBOL_BACKFILL_MIN / (_timeframe_interval_ms(timeframe) / 60_000.0))))
        latest = max(eligible)
        targets_rev: list[int] = [latest]
        while len(targets_rev) < max_buckets:
            prev = targets_rev[-1] - 1
            if prev not in eligible:
                break
            targets_rev.append(prev)
        return list(reversed(targets_rev))

    try:
        last_bucket = int(rec.get(timeframe))
    except (TypeError, ValueError):
        return [completed_bucket] if completed_bucket in available_buckets else []

    if last_bucket >= completed_bucket:
        return []

    targets: list[int] = []
    for bucket in range(last_bucket + 1, completed_bucket + 1):
        if bucket not in available_buckets:
            break
        targets.append(bucket)
    return targets


def _frame_through_bucket(klines: list[dict], timeframe: str, bucket: int) -> list[dict]:
    """Keep only closed candles through one exact timeframe bucket."""
    interval_ms = _timeframe_interval_ms(timeframe)
    out: list[dict] = []
    for row in klines:
        try:
            close_ts = int(row.get("close_time"))
        except (TypeError, ValueError, AttributeError):
            continue
        if close_ts // interval_ms <= bucket:
            out.append(row)
    return out


def _frame_through_event_ts(df: pd.DataFrame | None, event_ts: int) -> pd.DataFrame:
    """Return only HTF candles that had closed by the event timestamp."""
    if not isinstance(df, pd.DataFrame) or not event_ts or "close_time" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    close_ts = pd.to_numeric(out["close_time"], errors="coerce")
    out = out.loc[close_ts <= int(event_ts)].copy()
    return out.sort_values("close_time").reset_index(drop=True)


def _trend_history_causal_count(df: pd.DataFrame | None, decision_ts: int) -> int:
    """Count usable bars that were closed by the exact trend decision timestamp."""
    if not isinstance(df, pd.DataFrame) or df.empty or "close_time" not in df.columns:
        return 0
    close_ts = pd.to_numeric(df["close_time"], errors="coerce")
    close = pd.to_numeric(df.get("close"), errors="coerce") if "close" in df.columns else None
    mask = close_ts.notna() & (close_ts <= int(decision_ts))
    if close is not None:
        mask &= close.notna() & (close > 0)
    return int(mask.sum())


def _ensure_trend_history(
    *,
    symbol: str,
    timeframe: str,
    frame: pd.DataFrame,
    decision_ts: int,
    min_bars: int,
) -> tuple[pd.DataFrame, bool]:
    """Top up only the Trend Filter history when causal bars fall below its warmup.

    The normal scanner keeps exactly the configured detector history. Trend Filter
    needs a small causal margin because the latest 15M trigger can occur before the
    latest closed 1H/4H candle. We therefore fetch an expanded frame only when the
    already-cached scan frame cannot supply the required number of bars by the
    trigger timestamp. Detector inputs and their fetch limits remain unchanged.
    """
    if _trend_history_causal_count(frame, decision_ts) >= int(min_bars):
        return frame, False
    base_limit = max(int(min_bars), int(os.environ.get(f"KLINE_LIMIT_{timeframe.upper()}", str(min_bars))))
    expanded_limit = base_limit + TREND_FILTER_HISTORY_BUFFER_BARS
    refreshed = _fetch_market_klines_scan(symbol, timeframe, expanded_limit)
    if isinstance(refreshed, pd.DataFrame):
        refreshed_df = refreshed.copy()
    else:
        refreshed_df = pd.DataFrame(refreshed if refreshed is not None else [])
    return refreshed_df, True


def _refresh_timeframe_events(
    candidates,
    timeframe: str,
    limit: int,
    now_ms: int,
    seen_ids: set[str],
    stats: dict,
    scan_state: dict,
    completed_bucket: int,
    scan_klines_cache: dict[tuple[str, str], list[dict]] | None = None,
) -> list[dict]:
    """Scan each symbol's newly completed buckets without changing detector math."""
    fresh: list[dict] = []
    interval_ms = _timeframe_interval_ms(timeframe)
    for r in candidates:
        symbol = str(r.symbol).upper()
        if not _symbol_scan_due(scan_state, symbol, timeframe, completed_bucket):
            continue
        tf_stats = _tf_stats(stats, timeframe)
        try:
            klines = _fetch_market_klines_scan(symbol, timeframe, limit)
            if scan_klines_cache is not None:
                scan_klines_cache[(symbol, timeframe.lower())] = list(klines or [])
            if len(klines) < 60:
                log.warning("[SIGNALS] %s %s returned only %d candles; watermark deferred.", timeframe.upper(), symbol, len(klines))
                continue

            valid_close_ts: list[int] = []
            for row in klines:
                try:
                    close_ts = int(row.get("close_time"))
                except (TypeError, ValueError, AttributeError):
                    continue
                if close_ts > now_ms:
                    continue
                valid_close_ts.append(close_ts)
            available_buckets = {ts // interval_ms for ts in valid_close_ts}
            targets = _scan_buckets_to_process(
                scan_state, symbol, timeframe, completed_bucket, available_buckets,
            )
            if not targets:
                if _symbol_scan_due(scan_state, symbol, timeframe, completed_bucket):
                    log.warning(
                        "[SCAN_STATE] %s %s has no contiguous closed bucket available through %d; watermark deferred.",
                        symbol, timeframe.upper(), completed_bucket,
                    )
                continue

            symbol_scanned_any = False
            for bucket in targets:
                frame_rows = _frame_through_bucket(klines, timeframe, bucket)
                if len(frame_rows) < 60:
                    log.warning(
                        "[SIGNALS] %s %s bucket=%d has only %d historical candles; watermark deferred at prior bucket.",
                        timeframe.upper(), symbol, bucket, len(frame_rows),
                    )
                    break

                d = add_cvd(pd.DataFrame(frame_rows))
                # Audit fix B2: attach the accumulated OI snapshot history so
                # detect_divergences can emit Price-vs-OI divergence when coverage
                # is sufficient (no events until enough buckets are recorded).
                d = attach_oi_series(d, _load_oi_history().get(symbol))
                d = attach_funding_series(d, _load_funding_history().get(symbol))
                divs = detect_divergences(d, symbol, timeframe)
                vp_events = detect_volume_profile_divergence(d, symbol, timeframe) if ENABLE_VOLUME_PROFILE_DIVERGENCE_ENGINE else []
                harmonic_events = detect_harmonic_patterns(d, symbol, timeframe) if ENABLE_HARMONIC_PATTERN_ENGINE else []
                sqs = detect_squeeze_release(
                    d, symbol, timeframe,
                    min_squeeze_bars=3,
                    release_lookback_bars=int(os.environ.get("SQUEEZE_RELEASE_LOOKBACK_BARS", "4")),
                )
                # Forced-liquidation squeeze is an opt-in research engine until local
                # liquidation time-series coverage is available; the code remains intact.
                liqs = detect_liquidation_squeeze(r, d, symbol, timeframe) if ENABLE_LIQUIDATION_SQUEEZE_ENGINE else []
                strategy_events: list[dict] = []
                if timeframe.lower() == "4h" and ENABLE_MACD_4H_ENGINE:
                    strategy_events.extend(detect_macd_4h(d, symbol, timeframe))
                if timeframe.lower() == "1h" and ENABLE_MA_COMPRESSION_ENGINE:
                    strategy_events.extend(detect_ma_compression_breakout(d, symbol, timeframe))
                if ENABLE_BREAKOUT_MOMENTUM_ENGINE:
                    strategy_events.extend(detect_breakout_momentum(d, symbol, timeframe))
                if ENABLE_DONCHIAN_RETEST_ENGINE:
                    strategy_events.extend(detect_donchian_retest(d, symbol, timeframe))
                if ENABLE_LIQUIDITY_SWEEP_ENGINE:
                    strategy_events.extend(detect_liquidity_sweep_reclaim(d, symbol, timeframe))
                if ENABLE_EMA_PULLBACK_ENGINE:
                    strategy_events.extend(detect_ema_pullback_continuation(d, symbol, timeframe))
                if ENABLE_ORDER_BLOCK_ENGINE:
                    strategy_events.extend(detect_order_block(d, symbol, timeframe))
                if ENABLE_BREAKER_BLOCK_ENGINE:
                    strategy_events.extend(detect_breaker_block(d, symbol, timeframe))
                if ENABLE_MITIGATION_BLOCK_ENGINE:
                    mitigation_events = detect_mitigation_block(d, symbol, timeframe)
                    if not ENABLE_MITIGATION_BLOCK_BULLISH_ENGINE:
                        mitigation_events = [
                            ev for ev in mitigation_events
                            if str(ev.get("event_type", "")).upper() != "MITIGATION_BLOCK_BULLISH"
                        ]
                    strategy_events.extend(mitigation_events)
                if ENABLE_SFP_ENGINE:
                    strategy_events.extend(detect_sfp(d, symbol, timeframe))
                if ENABLE_LIQUIDATION_CASCADE_FVG_ENGINE:
                    strategy_events.extend(detect_liquidation_cascade_fvg(r, d, symbol, timeframe))
                if ENABLE_CRT_ENGINE:
                    strategy_events.extend(detect_crt(d, symbol, timeframe))

                symbol_scanned_any = True
                tf_stats["scanned_buckets"] = int(tf_stats.get("scanned_buckets", 0)) + 1
                all_events = divs + vp_events + harmonic_events + sqs + liqs + strategy_events
                stats["events_total"] += len(all_events)
                stats["divergence_events"] += len(divs)
                stats["squeeze_events"] += len(sqs) + len(liqs)
                tf_stats["divergence_events"] += len(divs)
                tf_stats["squeeze_events"] += len(sqs) + len(liqs)

                for ev in all_events:
                    ev.setdefault("event_fact", {})["market_data_source"] = MARKET_DATA_SOURCE
                    ev["event_fact"]["execution_exchange"] = "BingX"
                    eid = ev.get("event_id")
                    if eid and eid not in seen_ids:
                        if not emit_event(ev):
                            raise EventJournalPersistenceError(
                                f"event journal write failed for {symbol}/{timeframe} bucket={bucket} event_id={eid}"
                            )
                        seen_ids.add(str(eid))
                    elif not eid:
                        log.warning(
                            "[EVENT_JOURNAL] detector emitted event without event_id symbol=%s timeframe=%s bucket=%s; not persisted",
                            symbol, timeframe.upper(), bucket,
                        )

                    # Only live/fresh events are eligible for the entry pipeline.
                    # Older recovered events remain in events.jsonl for audit but do
                    # not get promoted back into the trade cache after their expiry.
                    if _event_is_fresh(ev, now_ms, _event_max_age_min(ev)):
                        fresh.append(ev)

                # The checkpoint advances only after this bucket's detector run AND
                # every newly detected event has been durably appended to the journal.
                _mark_symbol_scanned(scan_state, symbol, timeframe, bucket)
                _save_timeframe_scan_state(scan_state)

            if symbol_scanned_any:
                tf_stats["scanned"] += 1
        except EventJournalPersistenceError as exc:
            _record_scan_error(stats, f"timeframe_{timeframe.lower()}_event_persistence", tf_stats)
            log.error("[SIGNALS] %s %s persistence failure; watermark remains before failed bucket: %s", timeframe.upper(), symbol, exc)
            continue
        except (BinanceRateLimitError, BingXRateLimitError) as exc:
            _record_scan_error(stats, f"timeframe_{timeframe.lower()}_rate_limit", tf_stats)
            venue = "BINANCE" if MARKET_DATA_SOURCE == "binance" else "BINGX"
            log.warning("[SIGNALS] %s %s Kline rate limit reached on %s; stopping this timeframe scan for the cycle: %s", venue, timeframe.upper(), symbol, exc)
            break
        except BinanceSymbolUnavailableError as exc:
            _record_scan_error(stats, f"timeframe_{timeframe.lower()}_symbol_unavailable", tf_stats)
            log.warning("[SIGNALS] %s %s market-data symbol unavailable on Binance; skipping symbol: %s", timeframe.upper(), symbol, exc)
            continue
        except Exception as exc:
            _record_scan_error(stats, f"timeframe_{timeframe.lower()}_fetch_detection", tf_stats)
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


def load_terminal_event_ids(path: Path) -> set[str]:
    """Return event IDs explicitly retired by EVENT_TERMINAL records."""
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
        if isinstance(obj, dict) and obj.get("record_type") == "EVENT_TERMINAL" and obj.get("event_id"):
            ids.add(str(obj["event_id"]))
    return ids


def load_pre_order_drift_failure_counts(path: Path, terminal_ids: set[str] | None = None) -> dict[str, int]:
    """Count repeated PRE_ORDER_DRIFT_EXCEEDED outcomes per event.

    Counts are persisted in trades.jsonl through EXECUTION_ATTEMPT records, so the
    retry budget survives workflow runs without another state file. Terminal events
    are excluded because their retry budget has already been exhausted.
    """
    if not path.exists():
        return {}
    terminal_ids = terminal_ids or set()
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict) or obj.get("record_type") != "EXECUTION_ATTEMPT":
            continue
        event_id = str(obj.get("event_id") or "")
        if not event_id or event_id in terminal_ids:
            continue
        result = obj.get("result") or {}
        if isinstance(result, dict) and str(result.get("status", "")) == "PRE_ORDER_DRIFT_EXCEEDED":
            counts[event_id] = counts.get(event_id, 0) + 1
    return counts


def load_sr_data_failure_counts(path: Path, terminal_ids: set[str] | None = None) -> dict[str, int]:
    """Count persisted retryable S/R data failures per event."""
    if not path.exists():
        return {}
    terminal_ids = terminal_ids or set()
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict) or obj.get("record_type") != "EXECUTION_ATTEMPT":
            continue
        event_id = str(obj.get("event_id") or "")
        if not event_id or event_id in terminal_ids:
            continue
        result = obj.get("result") or {}
        if isinstance(result, dict) and str(result.get("status", "")).upper() == "SR_DATA_UNAVAILABLE":
            counts[event_id] = counts.get(event_id, 0) + 1
    return counts


def load_cross_exchange_drift_failure_counts(path: Path, terminal_ids: set[str] | None = None) -> dict[str, int]:
    """Count repeated cross-exchange drift rejections per event.

    The count is persisted in ``trades.jsonl`` via EXECUTION_ATTEMPT records so a
    workflow restart cannot reset the retry budget and retry the same stale or
    inconsistent cross-exchange event forever.
    """
    if not path.exists():
        return {}
    terminal_ids = terminal_ids or set()
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict) or obj.get("record_type") != "EXECUTION_ATTEMPT":
            continue
        event_id = str(obj.get("event_id") or "")
        if not event_id or event_id in terminal_ids:
            continue
        result = obj.get("result") or {}
        if isinstance(result, dict) and str(result.get("status", "")) == "CROSS_EXCHANGE_DRIFT_EXCEEDED":
            counts[event_id] = counts.get(event_id, 0) + 1
    return counts


def _register_cross_exchange_drift_failure(event_id: str, failure_counts: dict[str, int]) -> tuple[int, bool]:
    """Record one cross-exchange drift rejection and report whether it is exhausted."""
    current_fail_count = failure_counts.get(event_id, 0) + 1
    failure_counts[event_id] = current_fail_count
    return current_fail_count, current_fail_count >= MAX_CROSS_EXCHANGE_DRIFT_REJECTIONS


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


def _load_recent_symbol_losses(path: Path) -> dict[str, int]:
    """Return the latest confirmed LOSS close timestamp per symbol.

    DATA_ERROR/missing PnL is deliberately ignored: unknown results must never
    trigger the repeat-entry safety rule.
    """
    if not path.exists():
        return {}
    latest: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("record_type") != "TRADE_CLOSE":
                continue
            pnl = obj.get("realized_pnl_pct")
            if not isinstance(pnl, (int, float)):
                continue
            if not math.isfinite(float(pnl)) or float(pnl) >= 0:
                continue
            symbol = str(obj.get("symbol") or "").strip().upper()
            try:
                closed_ts = int(obj.get("closed_ts") or 0)
            except (TypeError, ValueError):
                closed_ts = 0
            if symbol and closed_ts > 0:
                latest[symbol] = max(latest.get(symbol, 0), closed_ts)
    except Exception as exc:
        log.warning("[RISK] recent-loss state read failed: %s", exc)
    return latest


def _entry_quality_gate(
    *,
    ev: dict[str, Any],
    row: Any,
    direction: str,
    now_ms: int,
    recent_loss_ts: dict[str, int],
) -> tuple[bool, list[str], bool]:
    """Evaluate targeted entry vetoes without relying on the legacy score.

    Returns (allowed, reasons, terminal). Temporary recent-loss cooldowns are
    non-terminal so the event can be retried after the cooldown window.
    """
    if not ENTRY_QUALITY_GATE_ENABLED:
        return True, [], False

    event_type = str(ev.get("event_type", "")).upper()
    symbol = str(ev.get("symbol", "")).strip().upper()
    d = str(direction).upper()
    try:
        oi24 = float(getattr(row, "oi_chg24_pct", 0.0) or 0.0)
    except (TypeError, ValueError):
        oi24 = 0.0
    cvd24 = None
    try:
        raw_cvd24 = getattr(row, "cvd24", None)
        value = float(raw_cvd24) if raw_cvd24 not in (None, "") else float("nan")
        if math.isfinite(value):
            cvd24 = value
    except (TypeError, ValueError):
        pass
    liq_long24 = None
    try:
        raw_liq_long24 = getattr(row, "liq_long24", None)
        value = float(raw_liq_long24) if raw_liq_long24 not in (None, "") else float("nan")
        if math.isfinite(value):
            liq_long24 = value
    except (TypeError, ValueError):
        pass

    reasons: list[str] = []
    temporary_reasons: list[str] = []
    if ENTRY_WEAK_ENGINE_BLOCK_ENABLED and event_type in ENTRY_BLOCKED_EVENT_TYPES:
        reasons.append(f"WEAK_EVENT_TYPE:{event_type}")
    if SHORT_OI_VETO_ENABLED and d == "SHORT" and oi24 < SHORT_OI_VETO_PCT:
        reasons.append(f"SHORT_OI:{oi24:.2f}%<{SHORT_OI_VETO_PCT:.2f}%")
    if (
        COMPOUND_CVD_LIQ_VETO_ENABLED
        and cvd24 is not None
        and liq_long24 is not None
        and cvd24 < COMPOUND_CVD_MAX
        and liq_long24 >= COMPOUND_LIQ_LONG_MIN
    ):
        reasons.append(f"CVD_LIQ:cvd={cvd24:.2f}<{COMPOUND_CVD_MAX:.2f},liq_long24={liq_long24:.0f}>={COMPOUND_LIQ_LONG_MIN:.0f}")
    if SYMBOL_LOSS_COOLDOWN_MIN > 0 and symbol:
        last_loss_ts = int(recent_loss_ts.get(symbol, 0) or 0)
        age_min = (now_ms - last_loss_ts) / 60_000.0 if last_loss_ts > 0 else None
        if age_min is not None and 0 <= age_min <= SYMBOL_LOSS_COOLDOWN_MIN:
            reason = f"RECENT_SYMBOL_LOSS:{age_min:.1f}m<{SYMBOL_LOSS_COOLDOWN_MIN:.1f}m"
            reasons.append(reason)
            temporary_reasons.append(reason)
    # Only a pure recent-loss cooldown is temporary. Any permanent veto that
    # accompanies it must still terminalize the event in enforce mode; otherwise
    # the same permanently invalid event can be retried on every later cycle.
    terminal = bool(reasons) and not temporary_reasons or (bool(reasons) and len(temporary_reasons) < len(reasons))
    return not reasons, reasons, terminal


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


def append_jsonl(path: Path, obj: dict, *, durable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        if durable:
            os.fsync(f.fileno())


def emit_event(ev: dict) -> bool:
    """Append one event to the durable source-of-truth journal.

    The caller must not advance scheduler state if this returns False.
    """
    try:
        append_jsonl(EVENTS, ev, durable=True)
        return True
    except (OSError, TypeError, ValueError) as exc:
        log.error("[EVENT_JOURNAL] durable write failed event_id=%s error=%s", ev.get("event_id"), exc)
        return False


def record_trade(obj: dict) -> None:
    append_jsonl(TRADES, obj)


def _record_reconciled_trade_open(symbol: str, direction: str, avg_price: float, qty: float,
                                  *, event_id: str, tp_orders: list[dict], sl_result: dict,
                                  setup: dict | None = None) -> None:
    """Journal an exchange position adopted by reconciliation."""
    trade_id = "TR_" + hashlib.sha256(str(event_id).encode("utf-8")).hexdigest()[:24].upper()
    record_trade({
        "record_type": "TRADE_OPEN",
        "trade_id": trade_id,
        "event_id": event_id,
        "symbol": symbol,
        "direction": direction,
        "signal": {"event_type": "RECONCILED_POSITION", "timeframe": "1h", "signal_price": avg_price,
                   "score": None, "detected_at_ts": None, "event_fact": {"reconciliation": True}},
        "execution": {"requested_price": avg_price, "signal_price": avg_price,
                       "actual_entry_price": avg_price, "actual_qty": qty, "order_id": None,
                       "status": "reconciled_adopted", "slippage_pct": None, "adverse_slippage_pct": None},
        "protection": {"tp_orders": tp_orders, "sl_result": sl_result},
        "setup": setup or {"event_type": "RECONCILED_POSITION", "risk_pct": None},
        "reconciliation": True, "status": "reconciled_adopted",
        "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
    })


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
    """Apply directional funding safety vetoes.

    Funding is stored in percentage-point units (0.10 == +0.10%).
    Extreme funding is a safety veto for every engine; liquidation squeezes
    additionally use tighter directional limits. Missing/invalid funding is
    policy-controlled by ``FUNDING_REQUIRED`` rather than silently hidden.
    Explicit thresholds remain supported for tests/backward compatibility.
    """
    if row is None:
        return (False, "FUNDING_REQUIRED_NO_ROW") if FUNDING_REQUIRED else (True, "NO_ROW")
    fr = getattr(row, "fr_oiw", None)
    if fr is None:
        return (False, "FUNDING_REQUIRED_NO_FUNDING_DATA") if FUNDING_REQUIRED else (True, "NO_FUNDING_DATA")
    try:
        fr_val = float(fr)
    except (TypeError, ValueError):
        return (False, "FUNDING_REQUIRED_INVALID_FUNDING_DATA") if FUNDING_REQUIRED else (True, "INVALID_FUNDING_DATA")
    if not math.isfinite(fr_val):
        return (False, "FUNDING_REQUIRED_INVALID_FUNDING_DATA") if FUNDING_REQUIRED else (True, "INVALID_FUNDING_DATA")

    d = str(direction).upper()
    event_upper = str(event_type or "").upper()
    is_liq_squeeze = _is_liquidation_squeeze_event(event_upper)

    # Extreme funding is a universal safety veto. Only actual liquidation-squeeze
    # events get the tighter directional crowding limits; volatility squeeze release
    # is a different mechanism and must not inherit liquidation-funding semantics.
    short_limit = MAX_SHORT_SQUEEZE_ADVERSE_FUNDING if is_liq_squeeze else EXTREME_SHORT_FUNDING
    long_limit = MAX_LONG_SQUEEZE_ADVERSE_FUNDING if is_liq_squeeze else EXTREME_LONG_FUNDING
    if max_short_adverse is not None:
        short_limit = float(max_short_adverse)
    if max_long_adverse is not None:
        long_limit = float(max_long_adverse)

    if d == "SHORT" and fr_val < short_limit:
        scope = "SQUEEZE" if is_liq_squeeze else "OVERRIDE"
        return False, f"ADVERSE_FUNDING_SHORT_{scope} (fr={fr_val:.4f} < {short_limit:.4f})"
    if d == "LONG" and fr_val > long_limit:
        scope = "SQUEEZE" if is_liq_squeeze else "OVERRIDE"
        return False, f"ADVERSE_FUNDING_LONG_{scope} (fr={fr_val:.4f} > {long_limit:.4f})"

    if not is_liq_squeeze:
        return True, "OK_NORMAL_FUNDING_NOT_FILTERED"
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


def _score_gate_passed(score: float, direction: str) -> bool:
    """Return whether the setup meets the configured directional score threshold."""
    value = _safe_float(score, 0.0)
    threshold = MIN_SHORT_SCORE if str(direction).upper() == "SHORT" else MIN_SCORE
    return threshold <= 0 or value >= threshold


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

    if event_type.endswith(("_MACD", "_MACD_HIST", "_STOCH", "_OBV", "_OI", "_FR_OIW_Z", "_MFI", "_CMF")):
        score += 15.0

    if "CVD" in event_type:
        score += 15.0

    # Standalone engines get an explicit identity bonus instead of being hidden
    # inside the generic divergence score. These remain capped at 100.
    if event_type == "MACD_4H_BULLISH_CROSS" or event_type == "MACD_4H_BEARISH_CROSS":
        score += 25.0
    elif event_type == "MA_COMPRESSION_BREAKOUT":
        score += 25.0
    elif event_type == "BREAKOUT_MOMENTUM":
        score += 30.0
    elif event_type == "DONCHIAN_RETEST_BREAKOUT":
        score += 25.0
    elif event_type == "LIQUIDITY_SWEEP_RECLAIM":
        score += 25.0
    elif event_type == "EMA_PULLBACK_CONTINUATION":
        score += 20.0
    elif event_type.startswith("ORDER_BLOCK_"):
        score += 20.0
    elif event_type.startswith("BREAKER_BLOCK_"):
        score += 20.0
    elif event_type.startswith("MITIGATION_BLOCK_"):
        score += 20.0
    elif event_type.startswith("SFP_"):
        score += 20.0
    elif event_type.startswith("LIQUIDATION_CASCADE_FVG_"):
        score += 25.0
    elif event_type.startswith("CRT_"):
        score += 20.0
    elif event_type.startswith("VOLUME_PROFILE_"):
        score += 20.0
    elif event_type.startswith("HARMONIC_"):
        score += 25.0

    # Squeeze families retain a bonus, but volatility compression and liquidation
    # squeeze are separate event types and can be analysed independently.
    if _is_squeeze_event(event_type):
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

    # Stop-loss is intentionally fixed for all new entries. ATR is retained only
    # as diagnostic telemetry when available; it no longer determines or vetoes
    # the entry risk. This makes the requested 7% stop deterministic across engines.
    atr = canonical_atr(df, 14).iloc[-1]
    risk_pct_raw = None
    if pd.notna(atr) and float(atr) > 0:
        atr = float(atr)
        raw = (atr * 1.5) / entry_price * 100.0
        if float("-inf") < raw < float("inf"):
            risk_pct_raw = raw

    risk_pct = float(FIXED_STOP_LOSS_PCT)

    ev_type = str(ev.get("event_type", "")).upper()
    is_squeeze = _is_squeeze_event(ev_type)
    target_rr = SQUEEZE_TP_RR[-1] if is_squeeze else NORMAL_TP_RR[-1]
    planned_weighted_rr = SQUEEZE_PLANNED_WEIGHTED_RR if is_squeeze else NORMAL_PLANNED_WEIGHTED_RR

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
        "risk_pct_raw": risk_pct_raw,
        "risk_was_clipped": False,
        "stop_loss_policy": "fixed",
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
    is_squeeze = _is_squeeze_event(ev_type)

    if is_squeeze:
        # Tighter squeeze cascade: 1.00R / 1.50R / 2.00R.
        tp_levels = [
            {"leg": f"tp{i}", "pnl_pct": round(sl_pct * rr, 6), "close_fraction": fraction}
            for i, (rr, fraction) in enumerate(zip(SQUEEZE_TP_RR, SQUEEZE_TP_FRACTIONS), start=1)
        ]
        planned_weighted_rr = SQUEEZE_PLANNED_WEIGHTED_RR
        target_rr = SQUEEZE_TP_RR[-1]
    else:
        # Tighter normal cascade: 0.65R / 1.25R / 2.00R.
        tp_levels = [
            {"leg": f"tp{i}", "pnl_pct": round(sl_pct * rr, 6), "close_fraction": fraction}
            for i, (rr, fraction) in enumerate(zip(NORMAL_TP_RR, NORMAL_TP_FRACTIONS), start=1)
        ]
        planned_weighted_rr = NORMAL_PLANNED_WEIGHTED_RR
        target_rr = NORMAL_TP_RR[-1]

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


def _reconciliation_event_id(
    bx_symbol: str,
    direction: str,
    avg_price: float,
    qty: float,
    position: dict[str, Any],
    active_trades: dict[str, dict],
) -> str:
    """Return a stable id for one reconciled orphan position.

    Identity uses exchange position fields when available. A collision with a
    previously closed record gets a one-time suffix so reopening the same
    symbol/direction cannot overwrite the historical trade.
    """
    direction = str(direction).upper()
    symbol = str(bx_symbol).upper()
    position_ts = 0
    # Only use stable/opening timestamps for reconciliation identity. An
    # exchange updateTime can change every cycle and would otherwise create a
    # fresh event_id for the same still-open orphan position.
    for key in ("entryTime", "openTime", "positionTime", "time", "timestamp"):
        raw = position.get(key)
        try:
            value = int(float(raw))
        except (TypeError, ValueError):
            continue
        if value > 0:
            position_ts = value
            break

    identity = f"{symbol}|{direction}|{avg_price:.12g}|{position_ts}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12].upper()
    base = f"RECON_{symbol}_{direction}_{digest}"

    if not isinstance(active_trades, dict):
        return base

    existing = active_trades.get(base)
    if existing is None:
        return base
    if not bool(existing.get("closed", False)):
        return base

    # A prior reconciliation of this exact position may already have created a
    # suffixed ID after the base record was closed. Reuse that live ID instead
    # of minting another ID on every reconciliation cycle.
    active_suffixes = sorted(
        str(event_id)
        for event_id, trade in active_trades.items()
        if str(event_id).startswith(base + "_")
        and isinstance(trade, dict)
        and not bool(trade.get("closed", False))
    )
    if active_suffixes:
        return active_suffixes[-1]

    candidate = f"{base}_{int(time.time() * 1000)}"
    if candidate in active_trades:
        candidate = f"{base}_{hashlib.sha256(candidate.encode('utf-8')).hexdigest()[:8].upper()}"
    return candidate


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
                            inferred_levels = [{"leg": "tp3", "pnl_pct": inferred_risk * NORMAL_TP_RR[-1], "close_fraction": 1.0}]
                        total_fraction = sum(max(_safe_float(x.get("close_fraction"), 0.0), 0.0) for x in inferred_levels)
                        if total_fraction <= 0:
                            inferred_levels = [{"leg": "tp3", "pnl_pct": inferred_risk * NORMAL_TP_RR[-1], "close_fraction": 1.0}]
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
                        reconciliation_event_id = _reconciliation_event_id(
                            bx_symbol, direction, avg_price, qty, p, active_trades
                        )
                        register_active_trade(
                            event_id=reconciliation_event_id,
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
                        _record_reconciled_trade_open(
                            bx_symbol.replace("-USDT", ""), direction, avg_price, qty,
                            event_id=reconciliation_event_id,
                            tp_orders=tracker_tp, sl_result=tracker_sl, setup=inferred_setup,
                        )
                        log.warning("[RECONCILIATION] Registered orphan protected position %s (%s) into tracker and journal.", bx_symbol, direction)
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
            # For an orphan/unmatched current position, use the shipped fixed stop
            # policy rather than inventing a legacy ATR-based risk profile.
            sl_pct = float(FIXED_STOP_LOSS_PCT)

        tp_levels = []
        if matched_trade and isinstance(matched_trade.get("effective_tp_levels"), list) and matched_trade.get("effective_tp_levels"):
            # Preserve the exact original protection profile, including squeeze and normal
            # TP distances and micro-position single-TP mode. Never silently rewrite a
            # matched trade during restart repair.
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
            if _is_squeeze_event(str((matched_trade or {}).get("event_type", ""))):
                default_levels = [
                    {"leg": f"tp{i}", "pnl_pct": round(sl_pct * rr, 6), "close_fraction": fraction}
                    for i, (rr, fraction) in enumerate(zip(SQUEEZE_TP_RR, SQUEEZE_TP_FRACTIONS), start=1)
                ]
            else:
                default_levels = [
                    {"leg": f"tp{i}", "pnl_pct": round(sl_pct * rr, 6), "close_fraction": fraction}
                    for i, (rr, fraction) in enumerate(zip(NORMAL_TP_RR, NORMAL_TP_FRACTIONS), start=1)
                ]
            tp_levels.extend([x for x in default_levels if x["leg"] not in hit_legs])

        if not tp_levels:
            final_rr = SQUEEZE_TP_RR[-1] if _is_squeeze_event(str((matched_trade or {}).get("event_type", ""))) else NORMAL_TP_RR[-1]
            tp_levels = [{"leg": "tp3", "pnl_pct": round(sl_pct * final_rr, 6), "close_fraction": 1.0}]

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
                reconciliation_event_id = _reconciliation_event_id(
                    bx_symbol, direction, avg_price, qty, p, active_trades
                )
                register_active_trade(
                    event_id=reconciliation_event_id,
                    symbol=bx_symbol.replace("-USDT", ""),
                    name=bx_symbol.replace("-USDT", ""),
                    direction=direction,
                    entry_price=avg_price,
                    qty=qty,
                    tp_orders=repaired_tp,
                    sl_result=repaired_sl,
                    event_type="RECONCILED_POSITION",
                )
                _record_reconciled_trade_open(
                    bx_symbol.replace("-USDT", ""), direction, avg_price, qty,
                    event_id=reconciliation_event_id,
                    tp_orders=repaired_tp, sl_result=repaired_sl,
                )

            first_tp = min((float(x.get("pnl_pct", 0)) for x in (repaired_tp or []) if x.get("pnl_pct") is not None), default=0.0)
            log.info("[RECONCILIATION] Protection restored for %s (%s): SL=%.2f%%, first TP=+%.2f%%", bx_symbol, direction, sl_pct, first_tp)
    
    log.info("[RECONCILIATION] Finished in %.1fs.", time.monotonic() - started)


def _market_entry_outcome_unknown(execution_result: dict[str, Any]) -> bool:
    """Return True when an accepted/denied entry left the exchange outcome unprovable."""
    if not isinstance(execution_result, dict):
        return False
    if str(execution_result.get("status", "")).upper() != "OPEN_FAILED":
        return False
    open_result = execution_result.get("open_result")
    return isinstance(open_result, dict) and str(open_result.get("status", "")).lower() == "unknown"


def execute_new_position(symbol: str, direction: str, price: float, setup: dict, event_id: str) -> dict:
    direction = str(direction).upper()
    trade_id = event_id.replace("EVT_", "")

    log.info("[EXECUTION] Preparing market entry: %s %s at ref price %.8g...", direction, symbol, price)

    # Hard pre-order drift guard against price movement while the opportunity
    # was being evaluated. The old implementation only compared trigger/event
    # prices and then discovered excessive drift after the fill, which could
    # force an expensive emergency close.
    event_type_for_risk = str(setup.get("event_type", "")).upper()
    drift_limit = MAX_SQUEEZE_ENTRY_DRIFT_PCT if _is_liquidation_squeeze_event(event_type_for_risk) else MAX_ENTRY_DRIFT_PCT
    try:
        # Execution remains on BingX; this reference is the actual BingX market price.
        live_reference = _current_close_price(symbol)
    except Exception as exc:
        return {"status": "PRE_ORDER_PRICE_UNAVAILABLE", "mode": EXECUTION_MODE, "order_id": None, "error": str(exc)}

    execution_quality_guard = {}
    if CROSS_EXCHANGE_PRICE_GUARD_ENABLED and MARKET_DATA_SOURCE == "binance":
        try:
            binance_live_price = fetch_binance_price(symbol)
        except BinanceRateLimitError as exc:
            _set_binance_rate_limit_cooldown(exc)
            return {
                "status": "CROSS_EXCHANGE_PRICE_UNAVAILABLE",
                "mode": EXECUTION_MODE,
                "order_id": None,
                "error": str(exc),
            }
        except BinanceSymbolUnavailableError as exc:
            return {
                "status": "CROSS_EXCHANGE_PRICE_UNAVAILABLE",
                "mode": EXECUTION_MODE,
                "order_id": None,
                "error": str(exc),
            }
        except Exception as exc:
            # Do not place a BingX order when the Binance execution-quality reference cannot be verified.
            return {
                "status": "CROSS_EXCHANGE_PRICE_UNAVAILABLE",
                "mode": EXECUTION_MODE,
                "order_id": None,
                "error": str(exc),
            }
        if live_reference is None or float(live_reference) <= 0 or binance_live_price <= 0:
            return {
                "status": "CROSS_EXCHANGE_PRICE_UNAVAILABLE",
                "mode": EXECUTION_MODE,
                "order_id": None,
                "error": f"invalid BingX/Binance reference prices: bingx={live_reference}, binance={binance_live_price}",
            }
        cross_exchange_drift_pct = abs(float(live_reference) - float(binance_live_price)) / float(binance_live_price) * 100.0
        execution_quality_guard = {
            "binance_live_price": float(binance_live_price),
            "bingx_live_price": float(live_reference),
            "cross_exchange_drift_pct": float(cross_exchange_drift_pct),
            "cross_exchange_drift_limit_pct": float(MAX_CROSS_EXCHANGE_DRIFT_PCT),
        }
        if MAX_CROSS_EXCHANGE_DRIFT_PCT > 0 and cross_exchange_drift_pct > MAX_CROSS_EXCHANGE_DRIFT_PCT:
            return {
                "status": "CROSS_EXCHANGE_DRIFT_EXCEEDED",
                "mode": EXECUTION_MODE,
                "order_id": None,
                "error": f"Binance/BingX price drift={cross_exchange_drift_pct:.6f}% > limit={MAX_CROSS_EXCHANGE_DRIFT_PCT:.6f}%",
                "execution_quality": execution_quality_guard,
            }

    pre_order_drift = _entry_drift_pct(_safe_float(setup.get("signal_price", price), price), live_reference, direction) if live_reference else None
    trigger_price = _safe_float((setup.get("trigger") or {}).get("trigger_price"), 0.0)
    trigger_live_drift = _entry_drift_pct(trigger_price, live_reference, direction) if trigger_price > 0 and live_reference else None
    effective_pre_drift = max(x for x in (pre_order_drift, trigger_live_drift) if x is not None) if (pre_order_drift is not None or trigger_live_drift is not None) else None
    if drift_limit > 0 and effective_pre_drift is not None and effective_pre_drift > drift_limit:
        return {
            "status": "PRE_ORDER_DRIFT_EXCEEDED",
            "mode": EXECUTION_MODE, "order_id": None,
            "error": f"pre_order_drift={effective_pre_drift:.6f} > limit={drift_limit:.6f}",
            "execution_quality": {"signal_to_pre_order_drift_pct": pre_order_drift, "trigger_to_pre_order_drift_pct": trigger_live_drift, "pre_order_price": live_reference, "drift_limit_pct": drift_limit},
        }

    # Final, lazy S/R room check: only now, for a candidate that survived all
    # previous gates and has a current BingX price. The S/R source is deliberately
    # Binance SPOT 1H for now; the rest of the execution path remains unchanged.
    sr_result: dict[str, Any] | None = None
    sr_snapshot: dict[str, Any] | None = None
    if AJAY_SR_ROOM_ENABLED and AJAY_SR_ROOM_MODE != "off":
        # Data retrieval is the only part of the S/R path that may be bypassed
        # when AJAY_SR_REQUIRE_DATA=false. Geometry/evaluation errors are fail-safe: they
        # must never silently turn into an unvalidated market entry.
        try:
            sr_snapshot = get_cached_sr_snapshot(symbol)
        except SRSymbolUnavailableError as exc:
            record_action({
                "event_id": event_id, "symbol": symbol, "direction": direction,
                "event_type": event_type_for_risk,
                "execution_status": "SR_SYMBOL_UNAVAILABLE",
                "error": str(exc),
                "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
            })
            if AJAY_SR_REQUIRE_DATA and AJAY_SR_ROOM_MODE == "enforce":
                return {"status": "SR_SYMBOL_UNAVAILABLE", "mode": EXECUTION_MODE, "order_id": None, "position": {}, "error": str(exc)}
            log.warning("[SR_ROOM] %s %s S/R symbol unavailable; continuing without S/R geometry: %s", direction, symbol, exc)
            sr_snapshot = None
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            record_action({
                "event_id": event_id, "symbol": symbol, "direction": direction,
                "event_type": event_type_for_risk,
                "execution_status": "SR_DATA_UNAVAILABLE",
                "error": str(exc),
                "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
            })
            if AJAY_SR_REQUIRE_DATA and AJAY_SR_ROOM_MODE == "enforce":
                return {"status": "SR_DATA_UNAVAILABLE", "mode": EXECUTION_MODE, "order_id": None, "position": {}, "error": str(exc)}
            log.warning("[SR_ROOM] %s %s S/R data unavailable; continuing without S/R geometry: %s", direction, symbol, exc)
            sr_snapshot = None

        if sr_snapshot is not None:
            try:
                sr_setup = dict(setup)
                sr_setup["entry_reference"] = float(live_reference)
                risk_pct_for_sr = float(setup.get("risk_pct", 0) or 0)
                if risk_pct_for_sr <= 0:
                    raise ValueError("invalid planned risk_pct for SR room")
                if direction == "LONG":
                    sr_setup["invalidation_price"] = float(live_reference) * (1.0 - risk_pct_for_sr / 100.0)
                else:
                    sr_setup["invalidation_price"] = float(live_reference) * (1.0 + risk_pct_for_sr / 100.0)
                sl_pct_preview, preview_tp_levels = build_tp_levels(
                    sr_setup, direction, event_type=event_type_for_risk
                )
                tp_rrs = tuple(
                    float(level.get("pnl_pct", 0) or 0) / max(float(sl_pct_preview), 1e-12)
                    for level in preview_tp_levels
                )
                if len(tp_rrs) != 3 or any(rr <= 0 for rr in tp_rrs):
                    raise ValueError("invalid TP ladder for SR room")
                sr_result = evaluate_sr_room(
                    sr_snapshot,
                    entry_price=float(live_reference),
                    direction=direction,
                    risk_pct=risk_pct_for_sr,
                    target_rrs=(float(tp_rrs[0]), float(tp_rrs[1]), float(tp_rrs[2])),
                )
                setup["sr_context"] = sr_result
                if sr_result.get("supporting_zone_confirmation"):
                    log.info("[SR_ROOM] %s %s supportive directional-zone confirmation: %s", direction, symbol, sr_result.get("directional_zone_alignment"))
                if sr_result.get("reject") and AJAY_SR_ROOM_MODE == "enforce":
                    return {
                        "status": "SR_ROOM_REJECTED",
                        "mode": EXECUTION_MODE,
                        "order_id": None,
                        "position": {},
                        "error": str(sr_result.get("reject_reason") or "SR_ROOM_REJECTED"),
                        "sr_room": sr_result,
                    }
                if sr_result.get("reject") and AJAY_SR_ROOM_MODE == "shadow":
                    record_action({
                        "event_id": event_id, "symbol": symbol, "direction": direction,
                        "event_type": event_type_for_risk,
                        "execution_status": "SR_ROOM_SHADOW_FLAGGED",
                        "sr_room": sr_result,
                        "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                    })
                    log.info("[SR_ROOM_SHADOW] %s %s rejected geometry=%s but NOT blocked", direction, symbol, sr_result.get("room_status"))
                elif AJAY_SR_ROOM_MODE == "shadow":
                    record_action({
                        "event_id": event_id, "symbol": symbol, "direction": direction,
                        "event_type": event_type_for_risk, "execution_status": "SR_ROOM_CHECKED",
                        "sr_room": sr_result, "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                    })
            except Exception as exc:
                record_action({
                    "event_id": event_id, "symbol": symbol, "direction": direction,
                    "event_type": event_type_for_risk, "execution_status": "SR_EVALUATION_FAILED",
                    "error": str(exc), "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                })
                log.exception("[SR_ROOM] %s %s S/R evaluation failed; market entry blocked: %s", direction, symbol, exc)
                return {
                    "status": "SR_EVALUATION_FAILED",
                    "mode": EXECUTION_MODE,
                    "order_id": None,
                    "position": {},
                    "error": str(exc),
                }

    log.info("[EXECUTION] Opening market position after all pre-order gates: %s %s at ref price %.8g...", direction, symbol, live_reference)
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
    execution_quality["signal_to_pre_order_drift_pct"] = pre_order_drift
    execution_quality["trigger_to_pre_order_drift_pct"] = trigger_live_drift
    execution_quality["pre_order_price"] = live_reference
    execution_quality.update(execution_quality_guard)
    fill_ts_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    log.info(
        "[EXECUTION] Fill confirmed: %s %s at avgPrice=%.8g (Qty: %.8g, Slippage: %+.2f%%)",
        direction, symbol, actual_avg_price, actual_qty, execution_quality.get("slippage_pct") or 0.0
    )

    event_type_for_risk = str(setup.get("event_type", "")).upper()
    drift_limit = MAX_SQUEEZE_ENTRY_DRIFT_PCT if _is_liquidation_squeeze_event(event_type_for_risk) else MAX_ENTRY_DRIFT_PCT
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
        is_squeeze = _is_squeeze_event(ev_type)
        target_rr = SQUEEZE_TP_RR[-1] if is_squeeze else NORMAL_TP_RR[-1]
        planned_weighted_rr = SQUEEZE_PLANNED_WEIGHTED_RR if is_squeeze else NORMAL_PLANNED_WEIGHTED_RR

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

        # The pre-order SR check protects the decision to submit the market order.
        # Re-evaluate once on the confirmed average fill so a materially different
        # fill price cannot make TP1 unreachable. The TP ladder itself is never
        # mutated by this S/R check.
        # This uses the already-fetched closed-1H snapshot, so it adds no network call.
        if (
            AJAY_SR_ROOM_MODE == "enforce"
            and sr_snapshot is not None
            and sr_result is not None
        ):
            actual_tp_rrs = tuple(
                float(level.get("pnl_pct", 0) or 0) / max(float(sl_pct), 1e-12)
                for level in tp_levels
            )
            if len(actual_tp_rrs) != 3 or any(rr <= 0 for rr in actual_tp_rrs):
                raise ValueError("invalid actual-fill TP ladder for SR recheck")
            post_fill_sr = evaluate_sr_room(
                sr_snapshot,
                entry_price=float(actual_avg_price),
                direction=direction,
                risk_pct=float(sl_pct),
                target_rrs=actual_tp_rrs,
            )
            sr_result = post_fill_sr
            setup_for_fill["sr_context"] = post_fill_sr
            if post_fill_sr.get("reject"):
                rollback = emergency_close_position(
                    symbol, direction, actual_qty, reason_token=f"SRPOSTFAIL:{trade_id}"
                )
                flattened = rollback.get("status") == "closed"
                return {
                    "status": "SR_ROOM_POST_FILL_REJECTED",
                    "mode": EXECUTION_MODE,
                    "order_id": order_id,
                    "position": ({**position, "positionAmt": 0.0} if flattened else {**position, "positionAmt": actual_qty}),
                    "open_result": opened,
                    "execution_quality": execution_quality,
                    "error": str(post_fill_sr.get("reject_reason") or "SR_ROOM_POST_FILL_REJECTED"),
                    "sr_room": post_fill_sr,
                    "emergency_close": rollback,
                    "rolled_back": flattened,
                }

        # Opposing zones after TP1 do not alter the target ladder. The entry gate
        # only rejects when TP1 itself is blocked; post-TP1 zones are recorded for
        # research/management context without changing the planned targets.
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
    if protection_status not in {"PROTECTED", "SL_ONLY"} and not protection.get("rolled_back"):
        rollback = emergency_close_position(symbol, direction, actual_qty, reason_token=f"PROTECTIONFAIL:{trade_id}")
        protection["emergency_close"] = rollback
        protection["rolled_back"] = rollback.get("status") == "closed"
        protection["rollback_reason"] = "post_entry_protection_not_verified"

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

    effective_tp_count = len(protection.get("effective_tp_levels", [])) if isinstance(protection, dict) else 0
    log.info("[EXECUTION] Protection installed for %s %s: Status=%s, SL=%.2f%%, TPs=%d legs", direction, symbol, final_status, sl_pct, effective_tp_count)

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


def _candidate_is_newer(candidate: dict[str, Any], existing: dict[str, Any]) -> bool:
    """Prefer the newest same-direction event; score breaks exact timestamp ties."""
    candidate_ts = int(candidate.get("event", {}).get("timestamps", {}).get("detected_at_ts", 0) or 0)
    existing_ts = int(existing.get("event", {}).get("timestamps", {}).get("detected_at_ts", 0) or 0)
    return (candidate_ts, float(candidate.get("score", 0.0))) > (existing_ts, float(existing.get("score", 0.0)))


def _trigger_age_min(trigger_meta: dict[str, Any] | None, now_ms: int | None = None) -> float | None:
    """Return trigger age from the actual closed 15M bar, never observation time."""
    meta = trigger_meta if isinstance(trigger_meta, dict) else {}
    trigger_bar_ts = _safe_float(meta.get("trigger_bar_close_ts"), 0.0)
    if trigger_bar_ts <= 0:
        return None
    current_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    return max(0.0, (current_ms - trigger_bar_ts) / 60_000.0)


def _trend_filter_enforce_reject(snapshot: dict[str, Any] | None, mode: str) -> bool:
    """Return whether Trend Filter may block the current candidate evaluation."""
    return str(mode or "off").lower() == "enforce" and (not isinstance(snapshot, dict) or snapshot.get("trend_decision") != "ALIGNED")


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
        "coinalyze_complete": False,
        "coinalyze_new_entries_frozen": False,
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
        "rejected_entry_quality": 0,
        "rejected_weak_engine": 0,
        "rejected_short_oi": 0,
        "rejected_cvd_liq": 0,
        "rejected_recent_loss": 0,
        "rejected_risk_too_wide": 0,
        "rejected_single_tp": 0,
        "entry_quality_shadow_flags": 0,
        "rejected_entry_drift": 0,
        "rejected_trigger_stale": 0,
        "rejected_portfolio_cap": 0,
        "rejected_hot_oi": 0,
        "rejected_symbol_quarantine": 0,
        "rejected_bingx_contract": 0,
        "rejected_binance_contract": 0,
        "conflict_rejected": 0,
        "valid_signals": 0,
        "execution_attempts": 0,
        "trades": 0,
        "scan_errors": 0,
        "scan_errors_by_stage": {},
        "cached_events": 0,
        "timeframe_scanned_symbols_1h": 0,
        "timeframe_scanned_symbols_4h": 0,
        "telegram_pending_retries": 0,
        "telegram_pending_retry_success": 0,
        "by_timeframe": {},
        "trend_shadow_candidates": 0,
        "trend_shadow_aligned": 0,
        "trend_shadow_rejected": 0,
        "trend_shadow_unknown": 0,
        "trend_shadow_persistent": 0,
        "trend_shadow_by_event_type": {},
    }

    btc_regime_df = None
    try:
        log.info("[ENGINE_STAGE] BTC regime fetch START (1h, limit=10)...")
        stage_started = time.monotonic()
        btc_klines = _fetch_market_klines_scan("BTCUSDT", "1h", limit=10)
        log.info("[ENGINE_STAGE] BTC regime fetch END in %.2fs; rows=%d.", time.monotonic() - stage_started, len(btc_klines or []))
        if btc_klines:
            btc_regime_df = pd.DataFrame(btc_klines)
            last_c = float(btc_regime_df["close"].iloc[-1])
            prev_1h = float(btc_regime_df["close"].iloc[-2])
            prev_4h = float(btc_regime_df["close"].iloc[-5])
            if not all(math.isfinite(value) for value in (last_c, prev_1h, prev_4h)):
                log.warning("[BTC_REGIME] Invalid/non-finite close values; regime snapshot unavailable.")
            elif last_c <= 0 or prev_1h <= 0 or prev_4h <= 0:
                log.warning("[BTC_REGIME] Non-positive close values; regime snapshot unavailable.")
            else:
                chg_1h = ((last_c - prev_1h) / prev_1h) * 100.0
                chg_4h = ((last_c - prev_4h) / prev_4h) * 100.0
                log.info("[BTC_REGIME] BTC: %.1f | 1H: %+.2f%% | 4H: %+.2f%% | Filter: OK", last_c, chg_1h, chg_4h)
    except Exception as exc:
        log.error("[BTC_REGIME] Fetch error: %s", exc)

    btc_regime_snapshot = _btc_regime_snapshot(btc_regime_df)
    stats["btc_regime_available"] = bool(btc_regime_snapshot.get("btc_available"))

    rows: list[Any] = []
    coinalyze_complete = False
    try:
        log.info("[ENGINE_STAGE] Coinalyze fetch START...")
        stage_started = time.monotonic()
        rows = fetch_data()
        coinalyze_complete = True
        log.info("[ENGINE_STAGE] Coinalyze fetch END in %.2fs; rows=%d.", time.monotonic() - stage_started, len(rows))
        log.info("[COINALYZE] Ingested %d rows from Coinalyze.", len(rows))
    except CoinalyzeIncompleteDataError as exc:
        rows = list(exc.rows)
        _record_scan_error(stats, "coinalyze_fetch_incomplete")
        log.error(
            "[COINALYZE] Incomplete derivatives universe: retaining %d partial rows for state/history, but freezing NEW ENTRIES: %s",
            len(rows), exc,
        )
    except Exception as exc:
        _record_scan_error(stats, "coinalyze_fetch")
        log.error("[COINALYZE] Scrape error: %s", exc)

    stats["coinalyze_complete"] = coinalyze_complete
    stats["coinalyze_rows"] = len(rows)

    # Maintain persistent paper-trade lifecycle for divergence shadow setups.
    # This never opens exchange positions and is intentionally independent of
    # real active-trade state. Market-price snapshots are used between cycles.
    if DIVERGENCE_SHADOW_ONLY:
        try:
            shadow_prices = {str(r.symbol).upper(): float(r.price) for r in rows if getattr(r, "price", None) and float(r.price) > 0}
            shadow_updates = update_divergence_shadow_state(DIVERGENCE_SHADOW_STATE, shadow_prices, now_ms=int(pd.Timestamp.utcnow().timestamp() * 1000))
            stats["divergence_shadow_active"] = shadow_updates.get("active", 0)
            stats["divergence_shadow_closed"] = shadow_updates.get("closed", 0)
        except Exception as exc:
            _record_scan_error(stats, "shadow_state_update")
            log.warning("[SHADOW] Divergence shadow state update failed: %s", exc)

    # Audit fix B2: persist OI snapshots per 1h bucket so Price-vs-OI swing
    # divergence becomes computable once enough history has accumulated.
    try:
        stats["oi_snapshots_recorded"] = _record_oi_snapshots(rows, int(pd.Timestamp.utcnow().timestamp() * 1000))
    except Exception as exc:
        log.error("[OI_HISTORY] Snapshot record error: %s", exc)

    try:
        stats["funding_snapshots_recorded"] = _record_funding_snapshots(rows, int(pd.Timestamp.utcnow().timestamp() * 1000))
    except Exception as exc:
        log.error("[FUNDING_HISTORY] Snapshot record error: %s", exc)

    bingx_contract_catalog_fresh = False
    try:
        contracts = refresh_contracts()
        bingx_contract_catalog_fresh = True
        log.info("[BINGX] Refreshed %d active perpetual contracts.", len(contracts))
    except Exception as exc:
        _record_scan_error(stats, "bingx_contract_refresh")
        log.error("[BINGX] Contracts refresh error: %s; NEW ENTRIES BLOCKED for this cycle.", exc)

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
    recent_symbol_losses = _load_recent_symbol_losses(TRADES)

    telegram_sent_event_ids = load_successful_telegram_ids(ACTIONS)
    telegram_attempted_this_cycle = send_pending_open_trade_notifications(
        current_positions=current_positions,
        successful_ids=telegram_sent_event_ids,
    )
    stats["telegram_pending_retries"] = len(telegram_attempted_this_cycle)
    stats["telegram_pending_retry_success"] = sum(1 for eid in telegram_attempted_this_cycle if eid in telegram_sent_event_ids)

    candidates: List[Any] = []
    if not coinalyze_complete:
        stats["coinalyze_new_entries_frozen"] = True
        log.warning(
            "[UNIVERSE] Coinalyze context is incomplete; NEW ENTRIES are frozen for this cycle. "
            "Existing-position reconciliation/protection already ran before the derivatives fetch."
        )
    for r in _coinalyze_rows_for_new_entries(rows, coinalyze_complete):
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
            if not bingx_contract_catalog_fresh:
                stats["rejected_bingx_contract"] += 1
                continue
            if not bingx_contract_exists(r.symbol):
                # A symbol may remain visible in cached/Coinalyze universe data
                # while BingX has paused trading. Never create a candidate that
                # can only fail later at execution time.
                stats["rejected_bingx_contract"] += 1
                continue
            if MARKET_DATA_SOURCE == "binance":
                try:
                    if not binance_contract_exists(r.symbol):
                        stats["rejected_binance_contract"] += 1
                        continue
                except BinanceRateLimitError as exc:
                    _set_binance_rate_limit_cooldown(exc)
                    _record_scan_error(stats, "candidate_binance_exchangeinfo_rate_limit")
                    log.error("[BINANCE] exchangeInfo rate limit reached while building candidate universe; stopping candidate build: %s", exc)
                    break
                except BinanceSymbolUnavailableError:
                    stats["rejected_binance_contract"] += 1
                    continue

            stats["contract_candidates"] += 1
            candidates.append(r)
        except Exception as exc:
            _record_scan_error(stats, "candidate_build_exception")
            log.warning("[UNIVERSE] Candidate build error for %s: %s", getattr(r, "symbol", "<unknown>"), exc)
            continue

    if MAX_CANDIDATES > 0:
        candidates = candidates[:MAX_CANDIDATES]

    stats["candidates_scanned"] = len(candidates)
    log.info("[UNIVERSE] %d liquidity candidates ($%.0fM Vol, $%.0fM OI) -> %d scanned via %s / executed on BingX.", stats["liquidity_candidates"], MIN_VOL/1e6, MIN_OI/1e6, len(candidates), MARKET_DATA_SOURCE.upper())

    seen_events = load_ids(EVENTS)
    executed_event_ids = load_successful_trade_ids(TRADES)
    terminal_event_ids = load_terminal_event_ids(TRADES)
    pre_order_drift_fail_counts = load_pre_order_drift_failure_counts(TRADES, terminal_event_ids)
    cross_exchange_drift_fail_counts = load_cross_exchange_drift_failure_counts(TRADES, terminal_event_ids)
    # Persisted S/R-data retry history must never influence candidate lifecycle
    # while the S/R room gate is disabled for the current evaluation run.
    sr_data_fail_counts = (
        load_sr_data_failure_counts(TRADES, terminal_event_ids)
        if AJAY_SR_ROOM_ENABLED and AJAY_SR_ROOM_MODE != "off"
        else {}
    )
    # Older state files may already contain an exhausted cross-exchange retry budget
    # without an EVENT_TERMINAL marker. Retire those events on restart so the next
    # scan cannot issue one more fresh MARKET-entry attempt for the same event.
    for exhausted_event_id, failure_count in cross_exchange_drift_fail_counts.items():
        if failure_count >= MAX_CROSS_EXCHANGE_DRIFT_REJECTIONS:
            terminal_event_ids.add(exhausted_event_id)
            executed_event_ids.add(exhausted_event_id)
    for exhausted_event_id, failure_count in sr_data_fail_counts.items():
        if failure_count >= MAX_SR_DATA_REJECTIONS:
            terminal_event_ids.add(exhausted_event_id)
            executed_event_ids.add(exhausted_event_id)
    best_opportunities_map: dict[tuple[str, str], dict] = {}

    scan_state = _load_timeframe_scan_state()
    event_cache = _load_cached_events(now_ms, terminal_event_ids)

    completed_1h = _completed_bucket(3_600_000, now_ms, BAR_CLOSE_GRACE_MIN)
    completed_4h = _completed_bucket(14_400_000, now_ms, BAR_CLOSE_GRACE_MIN)
    _scan_state_symbols = scan_state.get("symbols", {}) if isinstance(scan_state, dict) else {}
    log.info(
        "[SCAN_STATE] loaded symbols=%d completed_1h_bucket=%d completed_4h_bucket=%d grace_min=%.2f",
        len(_scan_state_symbols) if isinstance(_scan_state_symbols, dict) else 0,
        completed_1h, completed_4h, BAR_CLOSE_GRACE_MIN,
    )

    # Per-symbol watermarks preserve the existing event math while ensuring that a
    # symbol entering the liquidity universe late is scanned immediately for its
    # latest completed bar. Failures do not advance the watermark.
    scan_klines_cache: dict[tuple[str, str], list[dict]] = {}
    new_1h = _refresh_timeframe_events(
        candidates, "1h", int(os.environ.get("KLINE_LIMIT_1H", "250")),
        now_ms, seen_events, stats, scan_state, completed_1h, scan_klines_cache,
    )
    new_4h = _refresh_timeframe_events(
        candidates, "4h", int(os.environ.get("KLINE_LIMIT_4H", "250")),
        now_ms, seen_events, stats, scan_state, completed_4h, scan_klines_cache,
    )
    _scan_state_symbols = scan_state.get("symbols", {}) if isinstance(scan_state, dict) else {}
    log.info(
        "[SCAN_STATE] persisted symbols=%d scanned_1h=%d scanned_4h=%d cache_frames=%d",
        len(_scan_state_symbols) if isinstance(_scan_state_symbols, dict) else 0,
        int(_tf_stats(stats, "1h").get("scanned", 0)),
        int(_tf_stats(stats, "4h").get("scanned", 0)),
        len(scan_klines_cache),
    )
    event_cache = _merge_event_cache(event_cache, new_1h + new_4h)
    event_cache = [ev for ev in event_cache if _event_is_fresh(ev, now_ms, _event_max_age_min(ev))]
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
    htf_context_cache: dict[str, dict[str, Any]] = {}
    trend_1h_cache: dict[str, pd.DataFrame] = {}
    trend_4h_cache: dict[str, pd.DataFrame] = {}
    for r in candidates:
        symbol = str(r.symbol)
        all_events = events_by_symbol.get(symbol, [])
        if not all_events:
            continue

        d15 = None
        d15_rate_limited = False
        for ev in sorted(all_events, key=lambda x: int(x.get("timestamps", {}).get("detected_at_ts", 0) or 0), reverse=True):
            event_id = ev.get("event_id")
            if not event_id or event_id in executed_event_ids or event_id in terminal_event_ids:
                continue

            event_type_name = str(ev.get("event_type", "")).upper()
            if _is_divergence_event(ev) and not ENABLE_DIVERGENCE_ENGINE:
                continue
            direction = str(ev.get("direction", "")).upper()
            if direction not in {"LONG", "SHORT"}:
                stats["trigger_direction_failed"] += 1
                continue
            bx_symbol = to_bx_symbol(symbol)
            if bx_symbol and current_open_positions.get((bx_symbol, direction)):
                continue
            tf = str(ev.get("timeframe", "1h")).lower()
            tf_stats = _tf_stats(stats, tf)
            try:
                detected_at = int(ev.get("timestamps", {}).get("detected_at_ts", 0) or 0)
            except (TypeError, ValueError):
                stats["trigger_data_failed"] += 1
                continue
            if detected_at <= 0:
                stats["trigger_data_failed"] += 1
                continue
            age = (now_ms - detected_at) / 60_000.0
            event_max_age = _event_max_age_min(ev)
            if age < 0 or age > event_max_age:
                continue
            # Divergence pivots are inherently confirmed only after `right` bars.
            # Freshness therefore applies to the post-confirmation lifetime, not
            # to pivot_2 itself. A 1H divergence with right=2 is about 120 minutes
            # old when confirmed; using a raw 45-minute pivot age would reject it.
            if ("REGULAR_" in str(ev.get("event_type", "")).upper() or "HIDDEN_" in str(ev.get("event_type", "")).upper()):
                valid_timing, formation_age, post_confirmation_age, confirmation_lag_min = _divergence_post_confirmation_age(ev, now_ms)
                max_post_confirm_age = DIVERGENCE_POST_CONFIRM_MAX_AGE_MIN
                if not valid_timing or post_confirmation_age > max_post_confirm_age:
                    stats["rejected_stale_formation"] = stats.get("rejected_stale_formation", 0) + 1
                    tf_stats = _tf_stats(stats, str(ev.get("timeframe", "1h")))
                    tf_stats["rejected_stale_formation"] = tf_stats.get("rejected_stale_formation", 0) + 1
                    continue
                ev.setdefault("event_fact", {})["confirmation_lag_min"] = round(confirmation_lag_min, 3)
                ev["event_fact"]["formation_age_min"] = round(formation_age, 3)
                ev["event_fact"]["post_confirmation_age_min"] = round(post_confirmation_age, 3)

            stats["fresh_events"] += 1
            tf_stats["fresh_events"] += 1
            stats["fresh_long"] += int(direction == "LONG")
            stats["fresh_short"] += int(direction == "SHORT")
            event_type = str(ev.get("event_type", "")).upper()
            is_squeeze = _is_squeeze_event(event_type)
            stats["fresh_squeeze"] += int(is_squeeze)
            is_divergence = _is_divergence_event(ev)
            stats["fresh_divergence"] += int(is_divergence)
            tf_stats["fresh_squeeze"] += int(is_squeeze)
            tf_stats["fresh_divergence"] += int(is_divergence)
            log.info("[SIGNALS] Fresh event: %s %s | TF: %s | Type: %s | Age: %.1fm", direction, symbol, tf, event_type, age)

            if btc_regime_df is not None and not _is_btc_symbol(symbol):
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
                if d15_rate_limited:
                    stats["trigger_data_failed"] += 1
                    continue
                try:
                    k15 = _fetch_market_klines_scan(symbol, "15m", int(os.environ.get("KLINE_LIMIT_15M", "250")))
                except (BinanceRateLimitError, BingXRateLimitError) as exc:
                    d15_rate_limited = True
                    stats["trigger_data_failed"] += 1
                    _record_scan_error(stats, "trigger_15m_rate_limit")
                    log.warning("[SIGNALS] 15M market-data rate limited for %s; suppressing further 15M requests for this symbol/cycle: %s", symbol, exc)
                    continue
                except BinanceSymbolUnavailableError as exc:
                    d15_rate_limited = True
                    stats["trigger_data_failed"] += 1
                    log.warning("[SIGNALS] 15M Binance symbol unavailable for %s; suppressing further 15M requests for this symbol/cycle: %s", symbol, exc)
                    continue
                except Exception as exc:
                    stats["trigger_data_failed"] += 1
                    _record_scan_error(stats, "trigger_15m_fetch")
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
                _record_scan_error(stats, "signal_price_invalid")
                log.warning("[SIGNALS] Invalid signal price for %s %s (%s/%s): %.8f", direction, symbol, tf, event_type, signal_price)
                continue

            # 1H divergences require a 4H directional context; hidden divergences
            # are especially strict because they are continuation setups.
            htf_context: dict[str, Any] = {}
            requires_div_context = ("REGULAR_" in event_type or "HIDDEN_" in event_type) and (tf == "4h" or REQUIRE_4H_CONTEXT_FOR_1H)
            requires_strategy_context = bool(ev.get("event_fact", {}).get("requires_htf_context")) and not requires_div_context
            if requires_div_context or requires_strategy_context:
                context_tf = "1d" if tf == "4h" else "4h"
                context_limit = int(os.environ.get("KLINE_LIMIT_1D", "250")) if context_tf == "1d" else int(os.environ.get("KLINE_LIMIT_4H", "250"))
                cache_key = f"{symbol}:{context_tf}"
                htf_context = htf_context_cache.get(cache_key)
                if htf_context is None or (htf_context.get("error") and not htf_context.get("rate_limited")):
                    try:
                        kctx = scan_klines_cache.get((symbol, context_tf))
                        if kctx is None:
                            kctx = _fetch_market_klines_scan(symbol, context_tf, context_limit)
                        cdf = pd.DataFrame(kctx)
                        if len(kctx) >= 60:
                            cdf = add_cvd(cdf)
                            cdf = attach_oi_series(cdf, _load_oi_history().get(symbol))
                            cdf = attach_funding_series(cdf, _load_funding_history().get(symbol))
                            htf_context = {"df": cdf}
                        else:
                            htf_context = {"df": None}
                        htf_context_cache[cache_key] = htf_context
                    except (BinanceRateLimitError, BingXRateLimitError) as exc:
                        htf_context = {"df": None, "error": str(exc), "rate_limited": True}
                        htf_context_cache[cache_key] = htf_context
                    except BinanceSymbolUnavailableError as exc:
                        htf_context = {"df": None, "error": str(exc), "symbol_unavailable": True}
                        htf_context_cache[cache_key] = htf_context
                    except Exception as exc:
                        htf_context = {"df": None, "error": str(exc)}
                        # Do not cache ordinary transient errors; a later event/cycle may succeed.
                        htf_context_cache.pop(cache_key, None)

                raw_ctx_df = (htf_context or {}).get("df") if isinstance(htf_context, dict) else None
                # Historical/recovered events must see only HTF candles that had
                # actually closed by the event timestamp. This removes future HTF
                # information without changing the validator or detector formulas.
                ctx_df = _frame_through_event_ts(raw_ctx_df, detected_at)
                if requires_div_context:
                    valid_ctx, ctx_reason, ctx_meta = validate_divergence_context(ev, ctx_df, context_timeframe=context_tf)
                else:
                    valid_ctx, ctx_reason, ctx_meta = validate_strategy_htf_context(ev, ctx_df, context_timeframe=context_tf)
                if not valid_ctx:
                    if requires_div_context and tf == "1h" and not REQUIRE_4H_CONTEXT_FOR_1H:
                        pass
                    else:
                        stats["rejected_context"] = stats.get("rejected_context", 0) + 1
                        tf_stats["rejected_context"] = tf_stats.get("rejected_context", 0) + 1
                        log.info("[SIGNALS] %s %s (%s/%s) rejected by HTF context: %s", direction, symbol, tf, event_type, ctx_reason)
                        continue
                ev.setdefault("event_fact", {}).update(ctx_meta)

            trigger_observed_at_ts: int | None = None
            if REQUIRE_TRIGGER:
                if bool(ev.get("event_fact", {}).get("requires_retest")):
                    ref_level = _safe_float(ev.get("event_fact", {}).get("trigger_level"), 0.0)
                    if ref_level <= 0:
                        stats["trigger_data_failed"] += 1
                        continue
                    trigger_diag = diagnose_15m_retest_trigger(
                        d15, direction, ref_level, detected_at,
                        max_delay_min=_event_trigger_max_delay_min(ev), volume_mult=1.10
                    )
                else:
                    trigger_diag = diagnose_15m_trigger(d15, direction, event_detected_at_ts=detected_at, max_trigger_delay_min=MAX_TRIGGER_DELAY, min_vol_mult=1.05)
                if not trigger_diag.get("ok"):
                    reason = trigger_diag.get("reason") or "failed"
                    _record_trigger_failure(stats, tf_stats, reason)
                    log.info("[SIGNALS] %s %s (%s/%s) failed 15m trigger: %s", direction, symbol, tf, event_type, reason)
                    continue
                trigger_price = _safe_float(trigger_diag.get("trigger_price") or trigger_diag.get("current_close"), 0.0)
                if trigger_price <= 0:
                    stats["trigger_data_failed"] += 1
                    tf_stats["trigger_data_failed"] += 1
                    continue
                drift_pct = _entry_drift_pct(signal_price, trigger_price, direction)
                drift_limit = MAX_SQUEEZE_ENTRY_DRIFT_PCT if _is_liquidation_squeeze_event(event_type) else MAX_ENTRY_DRIFT_PCT
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
                # Capture the trigger observation time at the moment the trigger
                # actually passes validation. Do not reuse the cycle-start now_ms:
                # the opportunity list may spend seconds/minutes before execution.
                trigger_observed_at_ts = int(time.time() * 1000)
                stats["trigger_passed"] += 1
                tf_stats["trigger_passed"] += 1
            else:
                trigger_diag = {"ok": True, "reason": "not_required"}
                stats["trigger_passed"] += 1
                tf_stats["trigger_passed"] += 1

            # Trend Filter v1 is evaluated only after a valid closed-15M trigger.
            # The causal decision timestamp is the trigger candle close, not the later
            # wall-clock observation time. The latter is retained separately for
            # execution-age telemetry/staleness checks. Every HTF frame is clipped to
            # bars with close_time <= this trigger timestamp, so later 1H/4H bars can
            # never influence the shadow/enforce decision.
            if TREND_FILTER_ENABLED and TREND_FILTER_MODE != "off":
                decision_ts = int(_safe_float((trigger_diag or {}).get("trigger_bar_close_ts"), 0.0))
                try:
                    if symbol not in trend_1h_cache:
                        k1_trend = scan_klines_cache.get((symbol, "1h"))
                        if k1_trend is None:
                            k1_trend = _fetch_market_klines_scan(symbol, "1h", int(os.environ.get("KLINE_LIMIT_1H", "400")))
                        trend_1h_cache[symbol] = pd.DataFrame(k1_trend or [])
                    if symbol not in trend_4h_cache:
                        k4_trend = scan_klines_cache.get((symbol, "4h"))
                        if k4_trend is None:
                            k4_trend = _fetch_market_klines_scan(symbol, "4h", int(os.environ.get("KLINE_LIMIT_4H", "400")))
                        trend_4h_cache[symbol] = pd.DataFrame(k4_trend or [])

                    # The scanner intentionally keeps the detector history at 400 closed bars.
                    # A 15M trigger may close before the latest 1H/4H bar, leaving only 399
                    # causal bars. Top up Trend Filter history only in that edge case.
                    trend_1h_cache[symbol], refetched_1h = _ensure_trend_history(
                        symbol=symbol, timeframe="1h", frame=trend_1h_cache[symbol],
                        decision_ts=decision_ts, min_bars=TREND_FILTER_MIN_1H_BARS,
                    )
                    trend_4h_cache[symbol], refetched_4h = _ensure_trend_history(
                        symbol=symbol, timeframe="4h", frame=trend_4h_cache[symbol],
                        decision_ts=decision_ts, min_bars=TREND_FILTER_MIN_4H_BARS,
                    )
                    if refetched_1h or refetched_4h:
                        log.info(
                            "[TREND_FILTER] topped-up causal history for %s | 1H=%s 4H=%s decision_ts=%s.",
                            symbol, refetched_1h, refetched_4h, _format_execution_ts(decision_ts),
                        )
                    trend_snapshot = evaluate_trend_filter(
                        symbol=symbol,
                        direction=direction,
                        event_type=event_type,
                        df_1h=trend_1h_cache[symbol],
                        df_4h=trend_4h_cache[symbol],
                        btc_1h_df=btc_regime_df,
                        decision_ts_ms=decision_ts,
                        min_bars_1h=TREND_FILTER_MIN_1H_BARS,
                        min_bars_4h=TREND_FILTER_MIN_4H_BARS,
                        persistence_lookback_1h=TREND_FILTER_PERSISTENCE_LOOKBACK_1H,
                        persistence_lookback_4h=TREND_FILTER_PERSISTENCE_LOOKBACK_4H,
                        slope_lookback_4h=TREND_FILTER_SLOPE_LOOKBACK_4H,
                        require_persistence=TREND_FILTER_REQUIRE_PERSISTENCE,
                        mode=TREND_FILTER_MODE,
                    )
                except Exception as exc:
                    trend_snapshot = {
                        "version": "trend-v1-2026-10-04",
                        "enabled": True,
                        "mode": TREND_FILTER_MODE,
                        "symbol": symbol,
                        "event_direction": direction,
                        "event_type": event_type,
                        "decision_ts": decision_ts,
                        "trend_decision": "REJECT",
                        "trend_reject_reason": "TREND_DATA_ERROR",
                        "trend_4h": "UNKNOWN",
                        "trend_1h": "UNKNOWN",
                        "trend_persistence": "UNKNOWN",
                        "error": str(exc),
                    }
                    stats["trend_shadow_unknown"] += 1
                    _record_scan_error(stats, "trend_filter")
                    ev.setdefault("event_fact", {})["trend_filter"] = trend_snapshot
                    log.warning("[TREND_SHADOW] %s %s (%s/%s) unavailable: %s", direction, symbol, tf, event_type, exc)
                    if _trend_filter_enforce_reject(trend_snapshot, TREND_FILTER_MODE):
                        log.info("[TREND_FILTER] %s %s rejected for current evaluation: TREND_DATA_ERROR", direction, symbol)
                        continue
                else:
                    stats["trend_shadow_candidates"] += 1
                    reason = trend_snapshot.get("trend_reject_reason")
                    decision = str(trend_snapshot.get("trend_decision") or "REJECT")
                    if decision == "ALIGNED":
                        stats["trend_shadow_aligned"] += 1
                    else:
                        stats["trend_shadow_rejected"] += 1
                        if str(reason or "").endswith("UNKNOWN") or str(reason or "") in {"TREND_DATA_ERROR", "TREND_DECISION_TS_INVALID"}:
                            stats["trend_shadow_unknown"] += 1
                    if trend_snapshot.get("trend_persistence") == "PERSISTENT":
                        stats["trend_shadow_persistent"] += 1
                    by_event = stats.setdefault("trend_shadow_by_event_type", {})
                    bucket = by_event.setdefault(event_type, {"candidates": 0, "aligned": 0, "rejected": 0})
                    bucket["candidates"] += 1
                    bucket["aligned" if decision == "ALIGNED" else "rejected"] += 1
                    ev.setdefault("event_fact", {})["trend_filter"] = trend_snapshot
                    log.info(
                        "[TREND_SHADOW] %s %s (%s/%s) decision=%s reason=%s 4H=%s 1H=%s persistence=%s BTC=%s decision_ts=%s 4H_bar=%s 1H_bar=%s.",
                        direction, symbol, tf, event_type, decision, reason or "NONE",
                        trend_snapshot.get("trend_4h"), trend_snapshot.get("trend_1h"),
                        trend_snapshot.get("trend_persistence"), trend_snapshot.get("btc_regime"),
                        _format_execution_ts(decision_ts),
                        _format_execution_ts(trend_snapshot.get("trend_4h_bar_close_ts")),
                        _format_execution_ts(trend_snapshot.get("trend_1h_bar_close_ts")),
                    )
                    record_action({
                        "event_id": event_id, "symbol": symbol, "direction": direction,
                        "event_type": event_type, "execution_status": "TREND_SHADOW_" + ("ALIGNED" if decision == "ALIGNED" else "REJECTED"),
                        "trend_filter": trend_snapshot,
                        "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                    })
                    if _trend_filter_enforce_reject(trend_snapshot, TREND_FILTER_MODE):
                        # Even in enforce mode this is a retryable current-evaluation veto;
                        # do not terminalize the parent event because the trend can change.
                        log.info("[TREND_FILTER] %s %s rejected for current evaluation: %s", direction, symbol, reason or "trend_not_aligned")
                        continue
            else:
                trend_snapshot = None

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

            entry_allowed, entry_quality_reasons, entry_quality_terminal = _entry_quality_gate(
                ev=ev, row=r, direction=direction, now_ms=now_ms, recent_loss_ts=recent_symbol_losses
            )
            ev.setdefault("event_fact", {})["entry_quality_gate_enabled"] = bool(ENTRY_QUALITY_GATE_ENABLED)
            ev["event_fact"]["entry_quality_mode"] = ENTRY_QUALITY_MODE
            ev["event_fact"]["entry_quality_allowed"] = bool(entry_allowed)
            ev["event_fact"]["entry_quality_reasons"] = list(entry_quality_reasons)
            if not entry_allowed:
                if ENTRY_QUALITY_MODE == "shadow":
                    stats["entry_quality_shadow_flags"] += 1
                    tf_stats["entry_quality_shadow_flags"] = tf_stats.get("entry_quality_shadow_flags", 0) + 1
                    log.info("[ENTRY_QUALITY_SHADOW] %s %s (%s/%s) flagged but NOT rejected: %s", direction, symbol, tf, event_type, ";".join(entry_quality_reasons))
                    record_action({
                        "event_id": event_id, "symbol": symbol, "direction": direction,
                        "event_type": event_type, "execution_status": "ENTRY_QUALITY_SHADOW_FLAGGED",
                        "entry_quality_reasons": list(entry_quality_reasons),
                        "entry_quality_terminal": bool(entry_quality_terminal),
                        "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                    })
                elif ENTRY_QUALITY_MODE == "enforce":
                    stats["rejected_entry_quality"] += 1
                    tf_stats["rejected_entry_quality"] = tf_stats.get("rejected_entry_quality", 0) + 1
                    if any(x.startswith("WEAK_EVENT_TYPE:") for x in entry_quality_reasons):
                        stats["rejected_weak_engine"] += 1
                        tf_stats["rejected_weak_engine"] = tf_stats.get("rejected_weak_engine", 0) + 1
                    if any(x.startswith("SHORT_OI:") for x in entry_quality_reasons):
                        stats["rejected_short_oi"] += 1
                        tf_stats["rejected_short_oi"] = tf_stats.get("rejected_short_oi", 0) + 1
                    if any(x.startswith("CVD_LIQ:") for x in entry_quality_reasons):
                        stats["rejected_cvd_liq"] += 1
                        tf_stats["rejected_cvd_liq"] = tf_stats.get("rejected_cvd_liq", 0) + 1
                    if any(x.startswith("RECENT_SYMBOL_LOSS:") for x in entry_quality_reasons):
                        stats["rejected_recent_loss"] += 1
                        tf_stats["rejected_recent_loss"] = tf_stats.get("rejected_recent_loss", 0) + 1
                    log.info("[ENTRY_QUALITY] %s %s (%s/%s) rejected: %s", direction, symbol, tf, event_type, ";".join(entry_quality_reasons))
                    record_action({
                        "event_id": event_id, "symbol": symbol, "direction": direction,
                        "event_type": event_type, "execution_status": "ENTRY_QUALITY_REJECTED",
                        "entry_quality_reasons": list(entry_quality_reasons),
                        "entry_quality_terminal": bool(entry_quality_terminal),
                        "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                    })
                    if entry_quality_terminal:
                        terminal_event_ids.add(event_id)
                        executed_event_ids.add(event_id)
                        record_trade({
                            "record_type": "EVENT_TERMINAL", "event_id": event_id,
                            "symbol": symbol, "direction": direction, "event_type": event_type,
                            "reason": "ENTRY_QUALITY_REJECTED",
                            "entry_quality_reasons": list(entry_quality_reasons),
                            "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                        })
                    continue
                else:
                    log.info("[ENTRY_QUALITY_OFF] %s %s (%s/%s) ignored: %s", direction, symbol, tf, event_type, ";".join(entry_quality_reasons))

            # High OI growth remains diagnostic context, but the configured directional
            # score thresholds are now real admission gates. A signal below the threshold
            # is skipped for this trigger evaluation without terminalizing the parent event,
            # because trigger-side diagnostics can legitimately change on a later closed bar.
            score = calculate_setup_score(ev=ev, coinalyze_row=r, df_15m=d15, trigger_diagnostic=trigger_diag)
            min_score_for_direction = MIN_SHORT_SCORE if direction == "SHORT" else MIN_SCORE
            score_gate_passed = _score_gate_passed(score, direction)
            ev.setdefault("event_fact", {})["score"] = score
            ev["event_fact"]["score_gate_threshold"] = min_score_for_direction
            ev["event_fact"]["score_gate_passed"] = bool(score_gate_passed)
            if not score_gate_passed:
                stats["rejected_score"] += 1
                tf_stats["rejected_score"] = tf_stats.get("rejected_score", 0) + 1
                if direction == "SHORT":
                    stats["rejected_short_score"] += 1
                log.info(
                    "[SCORE] %s %s (%s/%s) rejected: score=%.2f < threshold=%.2f",
                    direction, symbol, tf, event_type, score, min_score_for_direction,
                )
                record_action({
                    "event_id": event_id, "symbol": symbol, "direction": direction,
                    "event_type": event_type, "execution_status": "SCORE_REJECTED",
                    "score": score, "score_gate_threshold": min_score_for_direction,
                    "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                })
                continue

            if symbol not in risk_1h_cache:
                try:
                    k1_risk = scan_klines_cache.get((symbol, "1h"))
                    if k1_risk is None:
                        k1_risk = _fetch_market_klines_scan(symbol, "1h", int(os.environ.get("KLINE_LIMIT_1H", "250")))
                    if len(k1_risk) < 20:
                        stats["trigger_data_failed"] += 1
                        continue
                    risk_1h_cache[symbol] = pd.DataFrame(k1_risk)
                except Exception as exc:
                    _record_scan_error(stats, "risk_1h_fetch")
                    log.warning("[RISK] Fresh 1H ATR fetch error for %s: %s", symbol, exc)
                    continue

            try:
                setup = build_event_setup(ev=ev, df_1h=risk_1h_cache[symbol], entry_price=signal_price)
            except (TypeError, ValueError, KeyError) as exc:
                if "ENTRY_RISK_TOO_WIDE" in str(exc):
                    stats["rejected_risk_too_wide"] += 1
                    tf_stats["rejected_risk_too_wide"] = tf_stats.get("rejected_risk_too_wide", 0) + 1
                    record_action({
                        "event_id": event_id, "symbol": symbol, "direction": direction,
                        "event_type": event_type, "execution_status": "ENTRY_RISK_TOO_WIDE",
                        "error": str(exc), "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                    })
                    terminal_event_ids.add(event_id)
                    executed_event_ids.add(event_id)
                    record_trade({
                        "record_type": "EVENT_TERMINAL", "event_id": event_id,
                        "symbol": symbol, "direction": direction, "event_type": event_type,
                        "reason": "ENTRY_RISK_TOO_WIDE",
                        "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                    })
                    continue
                stats["trigger_data_failed"] += 1
                log.warning("[RISK] Invalid setup for %s %s (%s/%s): %s", direction, symbol, tf, event_type, exc)
                continue
            except Exception as exc:
                _record_scan_error(stats, "risk_setup_unexpected")
                log.exception("[RISK] Unexpected setup error for %s %s (%s/%s)", direction, symbol, tf, event_type)
                continue
            setup["trigger"] = {
                "event_detected_at_ts": detected_at,
                "trigger_observed_at_ts": trigger_observed_at_ts,
                "trigger_bar_close_ts": trigger_diag.get("trigger_bar_close_ts"),
                "trigger_price": _safe_float(trigger_diag.get("trigger_price") or trigger_diag.get("current_close"), 0.0) or None,
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
            if isinstance(trend_snapshot, dict):
                setup["trend_filter"] = trend_snapshot
            setup["entry_context"] = _entry_context(r, ev, btc_regime_snapshot)
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
                if _candidate_is_newer(cand, existing):
                    cand["confluence_events"] = existing.get("confluence_events", [])
                    best_opportunities_map[key] = cand

            tp_log_parts = []
            for level in (setup.get("tp_levels") if isinstance(setup.get("tp_levels"), list) else [])[:3]:
                if not isinstance(level, dict):
                    continue
                tp_pct = _safe_float(level.get("pnl_pct"), 0.0)
                tp_price = tp_price_from_pnl(setup.get("entry_reference"), direction, tp_pct)
                risk_for_rr = _safe_float(setup.get("risk_pct"), 0.0)
                rr_text = format_rr(tp_pct / risk_for_rr) if risk_for_rr > 0 and tp_pct > 0 else "—"
                tp_pct_text = f"{format_number(tp_pct, decimals=2)}%"
                tp_log_parts.append(
                    f"{str(level.get('leg', 'TP')).upper()}={format_price(tp_price)} ({tp_pct_text} {rr_text})"
                )
            tp_log_text = " | ".join(tp_log_parts) if tp_log_parts else f"TP3={format_price(setup.get('target_price'))}"
            log.info(
                "[SIGNALS] Signal valid: %s %s | Score: %.0f/100 | TF: %s | Event: %s | Price: %s | SL: %s (-%s) | %s",
                direction, symbol, score, tf, event_type, format_price(signal_price),
                format_price(setup["invalidation_price"]),
                format_number(setup.get("risk_pct"), decimals=2),
                tp_log_text,
            )

    opportunities = list(best_opportunities_map.values())
    for opp in opportunities:
        tf = str(opp.get("event", {}).get("timeframe", "1h")).lower()
        _tf_stats(stats, tf)["valid_signals"] += 1
    opportunities, conflict_rejected = resolve_symbol_direction_conflicts(opportunities)
    stats["conflict_rejected"] = len(conflict_rejected)
    stats["valid_signals"] = len(opportunities)
    def _opportunity_rank_key(item: dict[str, Any]) -> tuple[float, str, str, float]:
        event = item.get("event") if isinstance(item.get("event"), dict) else {}
        ts = _safe_float((event.get("timestamps") or {}).get("detected_at_ts"), 0.0)
        return (-ts, str(item.get("symbol", "")), str(item.get("direction", "")), -_safe_float(item.get("score"), 0.0))
    opportunities.sort(key=_opportunity_rank_key)

    for rejected in conflict_rejected:
        loser = rejected.get("direction")
        symbol = rejected.get("symbol")
        against = rejected.get("conflict_rejected_against", {})
        log.info("[RANKING] Conflict rejected: %s %s (Score %.0f, TF %s) vs %s %s (Score %.0f, TF %s).",
                 loser, symbol, float(rejected.get("score", 0)), rejected.get("event", {}).get("timeframe"),
                 against.get("direction"), symbol, float(against.get("score", 0)), against.get("timeframe"))

    log.info("[RANKING] Unique non-conflicting signals ready: %d (ordered by event recency; score is metadata, not a rank key).", len(opportunities))
    for i, opp in enumerate(opportunities[:5], start=1):
        log.info("  [RANKING] #%d: %s %s | Score: %.0f (not ranked on) | %s", i, opp['direction'], opp['symbol'], opp['score'], opp['event'].get('event_type'))

    execution_attempts_this_cycle = 0
    opened_trades_this_cycle = 0

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
        event_type_name = str(ev.get("event_type", "")).upper()

        bx_symbol = to_bx_symbol(symbol)
        opposite_direction = "SHORT" if direction == "LONG" else "LONG"
        opposite_position_open = bool(bx_symbol and current_open_positions.get((bx_symbol, opposite_direction)))
        is_squeeze_opp = _is_squeeze_event(str(ev.get("event_type", "")))
        effective_cooldown = max(SYMBOL_ENTRY_COOLDOWN_MIN, SQUEEZE_SYMBOL_ENTRY_COOLDOWN_MIN) if is_squeeze_opp else SYMBOL_ENTRY_COOLDOWN_MIN
        symbol_cooldown = _symbol_on_cooldown(symbol, recent_entry_ts, now_ms, effective_cooldown)

        # A valid setup can become stale while the opportunity list is being
        # built or while higher-ranked trades are executed. Do not submit a
        # market order from an old trigger.
        trigger_meta = setup.get("trigger") or {}
        trigger_observed_ts = _safe_float(trigger_meta.get("trigger_observed_at_ts"), 0.0)
        trigger_bar_ts = _safe_float(trigger_meta.get("trigger_bar_close_ts"), 0.0)
        # The cycle-start now_ms is intentionally not used here. The opportunity
        # list may contain many signals, and earlier trades can consume enough time
        # for a previously-valid trigger to become stale before this order is sent.
        execution_now_ms = int(time.time() * 1000)
        trigger_age_min = _trigger_age_min(trigger_meta, execution_now_ms)
        if EXECUTION_ENABLED and _is_divergence_event(ev) and DIVERGENCE_SHADOW_ONLY:
            shadow_ts = int(pd.Timestamp.utcnow().timestamp() * 1000)
            trigger_entry = _safe_float(trigger_meta.get("trigger_price"), 0.0) or float(price)
            try:
                shadow_result = record_divergence_shadow_open(
                    DIVERGENCE_SHADOW_STATE,
                    event_id=event_id,
                    symbol=symbol,
                    direction=direction,
                    event_type=event_type_name,
                    timeframe=str(ev.get("timeframe", "1h")),
                    entry_price=trigger_entry,
                    setup=setup,
                    score=float(score),
                    opened_ts=shadow_ts,
                )
                execution_result = {
                    "status": "DIVERGENCE_SHADOW",
                    "mode": EXECUTION_MODE,
                    "order_id": None,
                    "position": {},
                    "error": "DIVERGENCE_SHADOW_ONLY=true",
                    "shadow": shadow_result,
                }
                record_action({
                    "event_id": event_id, "symbol": symbol, "direction": direction,
                    "score": score, "event_type": ev.get("event_type"),
                    "execution_status": "DIVERGENCE_SHADOW",
                    "shadow_entry_price": trigger_entry,
                    "ts": shadow_ts,
                })
                record_trade({
                    "record_type": "EVENT_TERMINAL",
                    "event_id": event_id,
                    "reason": "DIVERGENCE_SHADOW_OPENED",
                    "symbol": symbol,
                    "direction": direction,
                    "event_type": ev.get("event_type"),
                    "ts": shadow_ts,
                })
                terminal_event_ids.add(event_id)
            except Exception as exc:
                execution_result = {
                    "status": "DIVERGENCE_SHADOW_ERROR",
                    "mode": EXECUTION_MODE,
                    "order_id": None,
                    "position": {},
                    "error": str(exc),
                }
                log.exception("[SHADOW] Failed to record divergence paper trade for %s %s (%s)", direction, symbol, event_id)
        elif EXECUTION_ENABLED and trigger_age_min is not None and trigger_age_min > MAX_TRIGGER_TO_ORDER_DELAY_MIN:
            stats["rejected_trigger_stale"] += 1
            execution_result = {"status": "TRIGGER_STALE", "mode": EXECUTION_MODE, "order_id": None,
                                "error": f"trigger_age={trigger_age_min:.3f}m > limit={MAX_TRIGGER_TO_ORDER_DELAY_MIN:.3f}m"}
            # A stale trigger is a cycle-local skip, not a terminal event. The parent
            # setup may still be valid for a subsequent freshly-detected trigger.
            record_action({"event_id": event_id, "symbol": symbol, "direction": direction,
                           "score": score, "event_type": ev.get("event_type"),
                           "execution_status": "TRIGGER_STALE",
                           "trigger_age_min": trigger_age_min,
                           "trigger_bar_close_ts": trigger_bar_ts,
                           "trigger_bar_close_utc": _format_execution_ts(trigger_bar_ts),
                           "trigger_freshness_limit_min": MAX_TRIGGER_TO_ORDER_DELAY_MIN,
                           "ts": int(pd.Timestamp.utcnow().timestamp() * 1000)})
        elif EXECUTION_ENABLED:
            active_total = sum(1 for p in current_open_positions.values() if p)
            active_longs = sum(1 for (sym, d), p in current_open_positions.items() if p and d == "LONG")
            active_shorts = sum(1 for (sym, d), p in current_open_positions.items() if p and d == "SHORT")
            portfolio_cap_hit = PORTFOLIO_CAP_ENABLED and (
                active_total >= MAX_ACTIVE_TRADES
                or (direction == "LONG" and active_longs >= MAX_ACTIVE_LONGS)
                or (direction == "SHORT" and active_shorts >= MAX_ACTIVE_SHORTS)
            )
            if portfolio_cap_hit:
                stats["rejected_portfolio_cap"] += 1
                execution_result = {"status": "PORTFOLIO_CAP_REACHED", "mode": EXECUTION_MODE, "order_id": None,
                                    "error": f"active={active_total}/{MAX_ACTIVE_TRADES}, longs={active_longs}/{MAX_ACTIVE_LONGS}, shorts={active_shorts}/{MAX_ACTIVE_SHORTS}"}
                log.info("[RISK] %s %s rejected by portfolio cap: %s", direction, symbol, execution_result["error"])
            elif position_state_unknown:
                stats["blocked_by_position_state_unknown"] = stats.get("blocked_by_position_state_unknown", 0) + 1
                execution_result = {"status": "POSITION_STATE_UNKNOWN", "mode": EXECUTION_MODE, "order_id": None, "position": {}}
                log.error("[EXECUTION] %s %s blocked: exchange position state is UNKNOWN.", direction, symbol)
            elif event_id in executed_event_ids or (bx_symbol and current_open_positions.get((bx_symbol, direction))) or opposite_position_open or symbol_cooldown:
                existing_position = current_positions.get((bx_symbol, direction), {}) if bx_symbol else {}
                active_trade = None
                try:
                    active_trade = next((x for x in _load_active_trades().values() if not x.get("closed", False) and str(x.get("event_id", "")) == event_id), None)
                except Exception:
                    active_trade = None
                execution_result = {
                    "status": ("SYMBOL_COOLDOWN" if symbol_cooldown and not existing_position and not opposite_position_open and event_id not in executed_event_ids
                                else ("CONFLICTING_DIRECTION_POSITION" if opposite_position_open and not existing_position
                                      else ("ALREADY_EXECUTED_WITH_POSITION" if existing_position else "ALREADY_EXECUTED"))),
                    "mode": EXECUTION_MODE, "order_id": None, "position": existing_position,
                    "setup_used_for_protection": (active_trade or {}).get("setup", {}) if active_trade else setup,
                }
                log.info("[EXECUTION] %s (%s) - Already open/executed or blocked by cooldown.", symbol, direction)
            elif MAX_TRADES <= 0 or execution_attempts_this_cycle < MAX_TRADES:
                stats["execution_attempts"] += 1
                execution_attempts_this_cycle += 1
                attempt_limit = "unlimited" if MAX_TRADES <= 0 else str(MAX_TRADES)
                log.info("[EXECUTION] Attempt #%d/%s: %s %s (Score: %.0f, Ref: %.8g)...", execution_attempts_this_cycle, attempt_limit, direction, symbol, score, price)
                execution_result = execute_new_position(symbol=symbol, direction=direction, price=price, setup=setup, event_id=event_id)
                actual_position = execution_result.get("position", {}) if isinstance(execution_result, dict) else {}
                actual_qty_for_state = _safe_float(actual_position.get("positionAmt"), 0.0) if isinstance(actual_position, dict) else 0.0
                if actual_qty_for_state > 0:
                    _mark_local_position_state(current_open_positions, current_positions, actual_position, symbol, direction)
                    executed_event_ids.add(event_id)
                    recent_entry_ts[str(symbol).upper()] = now_ms
    
                err_str = str(execution_result.get("error", "")).lower()
                terminal_reason = None
                status_now = str(execution_result.get("status", ""))
                if status_now == "PRE_ORDER_DRIFT_EXCEEDED":
                    current_fail_count = pre_order_drift_fail_counts.get(event_id, 0) + 1
                    pre_order_drift_fail_counts[event_id] = current_fail_count
                    if current_fail_count >= MAX_PRE_ORDER_DRIFT_REJECTIONS:
                        terminal_reason = "ENTRY_DRIFT_EXHAUSTED"
                        log.warning(
                            "[EXECUTION] %s (%s) pre-order drift rejected %d times; terminalizing event %s.",
                            symbol, direction, current_fail_count, event_id,
                        )
                    else:
                        log.info(
                            "[EXECUTION] %s (%s) pre-order drift rejection %d/%d for event %s; keeping event retryable.",
                            symbol, direction, current_fail_count, MAX_PRE_ORDER_DRIFT_REJECTIONS, event_id,
                        )
                elif status_now == "CROSS_EXCHANGE_DRIFT_EXCEEDED":
                    current_fail_count, exhausted = _register_cross_exchange_drift_failure(
                        event_id, cross_exchange_drift_fail_counts
                    )
                    if exhausted:
                        terminal_reason = "CROSS_EXCHANGE_DRIFT_EXHAUSTED"
                        log.warning(
                            "[EXECUTION] %s (%s) cross-exchange drift rejected %d times; terminalizing event %s.",
                            symbol, direction, current_fail_count, event_id,
                        )
                    else:
                        log.info(
                            "[EXECUTION] %s (%s) cross-exchange drift rejection %d/%d for event %s; keeping event retryable.",
                            symbol, direction, current_fail_count, MAX_CROSS_EXCHANGE_DRIFT_REJECTIONS, event_id,
                        )
                elif status_now == "ENTRY_DRIFT_EXCEEDED":
                    terminal_reason = "ENTRY_DRIFT_EXCEEDED"
                    log.warning("[EXECUTION] %s (%s) fill drift exceeded configured limit; terminalizing event %s.", symbol, direction, event_id)
                elif status_now == "SR_SYMBOL_UNAVAILABLE":
                    terminal_reason = "SR_SYMBOL_UNAVAILABLE"
                    stats["rejected_sr_symbol"] = stats.get("rejected_sr_symbol", 0) + 1
                    log.warning("[SR_ROOM] %s (%s) no unambiguous Binance Spot source; terminalizing event %s: %s", symbol, direction, event_id, execution_result.get("error"))
                elif status_now == "SR_DATA_UNAVAILABLE":
                    current_fail_count = sr_data_fail_counts.get(event_id, 0) + 1
                    sr_data_fail_counts[event_id] = current_fail_count
                    stats["rejected_sr_data"] = stats.get("rejected_sr_data", 0) + 1
                    if current_fail_count >= MAX_SR_DATA_REJECTIONS:
                        terminal_reason = "SR_DATA_EXHAUSTED"
                        log.warning("[SR_ROOM] %s (%s) S/R data failure %d/%d; terminalizing event %s.", symbol, direction, current_fail_count, MAX_SR_DATA_REJECTIONS, event_id)
                    else:
                        log.info("[SR_ROOM] %s (%s) S/R data failure %d/%d; event remains retryable.", symbol, direction, current_fail_count, MAX_SR_DATA_REJECTIONS)
                elif status_now in {"SR_ROOM_REJECTED", "SR_ROOM_POST_FILL_REJECTED"}:
                    terminal_reason = status_now
                    stats["rejected_sr_room"] = stats.get("rejected_sr_room", 0) + 1
                    log.info("[SR_ROOM] %s (%s) rejected by opposing-zone room: %s", symbol, direction, execution_result.get("error"))
                elif _market_entry_outcome_unknown(execution_result):
                    # A MARKET entry whose exchange outcome is genuinely UNKNOWN
                    # must never be retried as a fresh signal on a later cycle: the
                    # original order may have been accepted while its ACK was lost.
                    # Reconciliation will adopt/track any position that eventually
                    # appears; a new market entry requires a new, independently
                    # validated event.
                    terminal_reason = "MARKET_ENTRY_OUTCOME_UNKNOWN"
                    log.critical(
                        "[EXECUTION] %s (%s) MARKET entry outcome is UNKNOWN; terminalizing event %s to prevent a duplicate entry. ",
                        symbol, direction, event_id,
                    )
                elif err_str in {"contract_unavailable", "contract_not_found"} or "contract_unavailable" in err_str:
                    terminal_reason = "BINGX_CONTRACT_UNAVAILABLE"
                    stats["rejected_bingx_contract"] = stats.get("rejected_bingx_contract", 0) + 1
                    log.warning("[EXECUTION] %s (%s) BingX contract disappeared/closed after preflight; terminalizing event %s.", symbol, direction, event_id)
                elif execution_result.get("bingx_code") == 101481 or "clientorderid unique check failed" in err_str or "clientorderid has already been used" in err_str:
                    terminal_reason = "CLIENT_ORDER_ID_ALREADY_USED"
                    log.warning("[EXECUTION] %s (%s) clientOrderId already used on exchange; terminalizing event %s.", symbol, direction, event_id)
                elif execution_result.get("bingx_code") == 101400 and ("clientorderid" in err_str or "duplicate" in err_str):
                    terminal_reason = "CLIENT_ORDER_ID_ALREADY_USED"
                    log.warning("[EXECUTION] %s (%s) 101400 reported a duplicate clientOrderId; terminalizing event %s.", symbol, direction, event_id)
                elif execution_result.get("bingx_code") == 101400 and "suspend" in err_str:
                    terminal_reason = "SYMBOL_SUSPENDED"
                    log.warning("[EXECUTION] %s (%s) pair suspended on exchange; terminalizing event %s.", symbol, direction, event_id)
                elif "multi_tp_not_supported" in err_str:
                    terminal_reason = "MULTI_TP_NOT_REACHABLE"
                    stats["rejected_single_tp"] += 1
                    log.warning("[EXECUTION] %s (%s) cannot support three TP legs at configured leverage; terminalizing event %s.", symbol, direction, event_id)
                elif "min_qty" in err_str:
                    terminal_reason = "MIN_QTY_NOT_REACHABLE"
                    log.warning("[EXECUTION] %s (%s) min_qty not met at configured leverage; terminalizing event %s to prevent slot burn.", symbol, direction, event_id)
                elif "min_notional" in err_str or "min_usdt" in err_str or "min_size_usd" in err_str:
                    terminal_reason = "MIN_NOTIONAL_NOT_REACHABLE"
                    log.warning("[EXECUTION] %s (%s) exchange minimum notional not reachable at configured margin/leverage; terminalizing event %s.", symbol, direction, event_id)
    
                if terminal_reason:
                    executed_event_ids.add(event_id)
                    terminal_event_ids.add(event_id)
                    record_trade({
                        "record_type": "EVENT_TERMINAL",
                        "event_id": event_id,
                        "symbol": symbol,
                        "direction": direction,
                        "event_type": ev.get("event_type"),
                        "reason": terminal_reason,
                        "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                    })
    
                actual_entry_raw = _safe_float(
                    actual_position.get("avgPrice", 0) or actual_position.get("entryPrice", 0),
                    0.0,
                )
                actual_entry = actual_entry_raw if actual_entry_raw > 0 else None
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
                        # A confirmed TRADE_OPEN must preserve the exact setup
                        # recalculated from the actual fill, including effective
                        # TP mode/levels and any post-fill SR context. Rejected
                        # execution attempts keep the pre-order setup because no
                        # fill-derived setup exists.
                        "setup": (execution_result.get("setup_used_for_protection") or setup) if confirmed_trade else setup,
                        "planned_metrics": {
                            "target_rr": (execution_result.get("setup_used_for_protection") or setup).get("target_rr"),
                            "planned_weighted_rr": (execution_result.get("setup_used_for_protection") or setup).get("planned_weighted_rr"),
                            "effective_weighted_rr": (execution_result.get("setup_used_for_protection") or setup).get("effective_weighted_rr"),
                            "tp_mode": (execution_result.get("setup_used_for_protection") or setup).get("tp_mode"),
                            "effective_tp_levels": (execution_result.get("setup_used_for_protection") or setup).get("effective_tp_levels", []),
                            "realized_rr": None,
                        },
                    }
                )
    
                status = str(execution_result.get("status", ""))
                if status in {"opened_protected", "opened_protection_check_required", "opened_protection_failed"}:
                    if status in {"opened_protected", "opened_protection_check_required"} or actual_qty_for_state > 0:
                        stats["trades"] += 1
                        opened_trades_this_cycle += 1
    
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
            else:
                execution_result = {"status": "TRADE_LIMIT_REACHED", "mode": EXECUTION_MODE, "order_id": None}
                log.info("[EXECUTION] %s %s skipped: execution-attempt cycle limit reached (%d/%d).", direction, symbol, execution_attempts_this_cycle, MAX_TRADES)
    
        elif not EXECUTION_ENABLED:
            execution_result = {"status": "DISABLED", "mode": EXECUTION_MODE, "order_id": None}
            log.info("[EXECUTION] %s %s skipped: EXECUTION_ENABLED is false.", direction, symbol)

        _log_execution_skip(
            symbol=symbol,
            direction=direction,
            event_id=event_id,
            execution_result=execution_result,
            trigger_age_min=trigger_age_min,
            trigger_bar_ts=trigger_bar_ts,
        )

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
                "error": execution_result.get("error"),
                "sr_room": execution_result.get("sr_room") if isinstance(execution_result.get("sr_room"), dict) else None,
                "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
            }
        )

    try:
        append_shadow_health(events_path=EVENTS, health_path=HEALTH, trades_path=TRADES, divergence_shadow_path=DIVERGENCE_SHADOW_STATE, cycle_stats=stats)
    except Exception as exc:
        log.error("[SHADOW] Health snapshot error: %s", exc)

    for tf_name, tf_rec in sorted(stats.get("by_timeframe", {}).items()):
        log.info("[TF_STATS] %s %s", tf_name.upper(), " ".join(f"{k}={v}" for k, v in tf_rec.items()))

    log.info("[SUMMARY] [SCAN_ERRORS_BY_STAGE] %s", stats.get("scan_errors_by_stage", {}))
    log.info(
        "[TREND_SHADOW_SUMMARY] candidates=%d aligned=%d rejected=%d unknown=%d persistent=%d",
        int(stats.get("trend_shadow_candidates", 0)),
        int(stats.get("trend_shadow_aligned", 0)),
        int(stats.get("trend_shadow_rejected", 0)),
        int(stats.get("trend_shadow_unknown", 0)),
        int(stats.get("trend_shadow_persistent", 0)),
    )

    summary_str = " ".join(f"{k}={v}" for k, v in stats.items())
    log.info("[SUMMARY] [FORENSIC_SUMMARY] %s", summary_str)
    log.info("[SUMMARY] [ENGINE_SUMMARY] opened_trades_this_cycle=%d execution_attempts_this_cycle=%d %s", opened_trades_this_cycle, execution_attempts_this_cycle, summary_str)
    log.info("========== [ENGINE] CYCLE END: opened_trades_this_cycle=%d execution_attempts_this_cycle=%d ==========", opened_trades_this_cycle, execution_attempts_this_cycle)


if __name__ == "__main__":
    main()
