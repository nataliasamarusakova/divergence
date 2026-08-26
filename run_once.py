from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from event_engine.coinalyze import fetch_data
from event_engine.bingx import (
    refresh_contracts,
    get_contract,
    fetch_klines,
    open_market,
    wait_for_position_fill_directional,
    get_position_directional,
    ensure_directional_protection,
)
from event_engine.signals import (
    add_cvd,
    detect_divergences,
    detect_squeeze_release,
    build_15m_trigger,
)
from event_engine.telegram import send as send_tg, format_signal


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

DATA = Path("data")
DATA.mkdir(exist_ok=True)

EVENTS = DATA / "events.jsonl"
TRADES = DATA / "trades.jsonl"
ACTIONS = DATA / "actions.jsonl"

MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "40"))
MIN_VOL = float(os.environ.get("MIN_VOLUME_24H", "1000000"))
MIN_OI = float(os.environ.get("MIN_OPEN_INTEREST", "500000"))

EXECUTION_ENABLED = (
    os.environ.get("EXECUTION_ENABLED", "false").lower() == "true"
)

REQUIRE_CVD = (
    os.environ.get("REQUIRE_CVD_CONFIRMATION", "false").lower() == "true"
)

REQUIRE_TRIGGER = (
    os.environ.get("REQUIRE_15M_TRIGGER", "true").lower() == "true"
)

MAX_AGE = int(os.environ.get("MAX_EVENT_AGE_MIN", "90"))
MAX_TRADES = int(os.environ.get("MAX_TRADES_PER_CYCLE", "1"))

EXECUTION_MODE = os.environ.get(
    "EXECUTION_MODE",
    os.environ.get("BINGX_ENV", "vst"),
)

POSITION_MODE = os.environ.get("BINGX_POSITION_MODE", "HEDGE")


# ============================================================================
# FILE HELPERS
# ============================================================================

def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids: set[str] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        try:
            value = json.loads(line).get("event_id")
        except Exception:
            continue

        if value:
            ids.add(str(value))

    return ids


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                obj,
                ensure_ascii=False,
            )
            + "\n"
        )


def emit_event(ev: dict) -> None:
    append_jsonl(EVENTS, ev)


def record_trade(obj: dict) -> None:
    append_jsonl(TRADES, obj)


def record_action(obj: dict) -> None:
    append_jsonl(ACTIONS, obj)


# ============================================================================
# SETUP
# ============================================================================

def build_event_setup(
    ev: dict,
    df_1h: pd.DataFrame,
    entry_price: float,
) -> dict:
    """
    Формирует торговый Setup независимо от EventObject.

    Приоритет:
    1. готовые SL/TP внутри event;
    2. ATR-based fallback.

    Для LONG:
        SL < Entry < TP

    Для SHORT:
        TP < Entry < SL
    """

    direction = str(
        ev.get("direction", "LONG")
    ).upper()

    if direction not in {"LONG", "SHORT"}:
        raise ValueError(
            f"Unsupported direction={direction}"
        )

    entry_price = float(entry_price)

    if entry_price <= 0:
        raise ValueError(
            f"Invalid entry_price={entry_price}"
        )

    # ----------------------------------------------------------------------
    # 1. Проверяем готовый setup внутри event
    # ----------------------------------------------------------------------

    candidates = [
        ev.get("setup"),
        ev.get("event_fact"),
        ev,
    ]

    invalidation = None
    target = None

    for src in candidates:
        if not isinstance(src, dict):
            continue

        if invalidation is None:
            invalidation = src.get(
                "invalidation_price"
            )

        if target is None:
            target = src.get(
                "target_price"
            )

    if invalidation is not None and target is not None:
        invalidation = float(invalidation)
        target = float(target)

        if direction == "LONG":
            valid_geometry = (
                invalidation < entry_price
                and target > entry_price
            )
        else:
            valid_geometry = (
                invalidation > entry_price
                and target < entry_price
            )

        if valid_geometry:
            risk_pct = (
                abs(entry_price - invalidation)
                / entry_price
                * 100.0
            )

            reward_pct = (
                abs(target - entry_price)
                / entry_price
                * 100.0
            )

            rr = (
                reward_pct / risk_pct
                if risk_pct > 0
                else None
            )

            if rr is not None and rr > 0:
                return {
                    "entry_reference": entry_price,
                    "invalidation_price": invalidation,
                    "target_price": target,
                    "rr": rr,
                    "trigger_ok": True,
                }

    # ----------------------------------------------------------------------
    # 2. ATR fallback
    # ----------------------------------------------------------------------

    df = df_1h.copy()

    if len(df) < 20:
        raise ValueError(
            "insufficient 1H bars for setup"
        )

    for col in ("high", "low", "close"):
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = (
        tr
        .rolling(
            window=14,
            min_periods=14,
        )
        .mean()
        .iloc[-1]
    )

    if pd.isna(atr) or float(atr) <= 0:
        raise ValueError(
            "ATR unavailable"
        )

    atr = float(atr)

    # Ограничиваем SL разумными границами.
    risk_pct = (
        atr / entry_price * 100.0
    )

    risk_pct = max(
        0.50,
        min(risk_pct, 5.00),
    )

    if direction == "LONG":
        invalidation = (
            entry_price
            * (1.0 - risk_pct / 100.0)
        )

        target = (
            entry_price
            * (1.0 + 2.0 * risk_pct / 100.0)
        )

    else:
        invalidation = (
            entry_price
            * (1.0 + risk_pct / 100.0)
        )

        target = (
            entry_price
            * (1.0 - 2.0 * risk_pct / 100.0)
        )

    return {
        "entry_reference": entry_price,
        "invalidation_price": invalidation,
        "target_price": target,
        "rr": 2.0,
        "trigger_ok": True,
    }


def build_tp_levels(
    setup: dict,
    direction: str,
) -> Tuple[float, List[dict]]:
    """
    Формирует модельную защиту.

    SL:
        весь остаток позиции защищается одним стопом.

    TP:
        TP1 = 30%
        TP2 = 30%
        TP3 = 40%

    Важно:
    total close_fraction = 1.0
    """

    direction = str(direction).upper()

    entry = float(
        setup["entry_reference"]
    )

    sl_price = float(
        setup["invalidation_price"]
    )

    final_tp_price = float(
        setup["target_price"]
    )

    if direction == "LONG":
        sl_pct = (
            entry - sl_price
        ) / entry * 100.0

        tp_pct = (
            final_tp_price - entry
        ) / entry * 100.0

    elif direction == "SHORT":
        sl_pct = (
            sl_price - entry
        ) / entry * 100.0

        tp_pct = (
            entry - final_tp_price
        ) / entry * 100.0

    else:
        raise ValueError(
            f"Invalid direction={direction}"
        )

    if sl_pct <= 0:
        raise ValueError(
            f"Invalid SL geometry: {sl_price}"
        )

    if tp_pct <= 0:
        raise ValueError(
            f"Invalid TP geometry: {final_tp_price}"
        )

    tp_levels = [
        {
            "leg": "tp1",
            "pnl_pct": round(
                tp_pct * 0.50,
                6,
            ),
            "close_fraction": 0.30,
        },
        {
            "leg": "tp2",
            "pnl_pct": round(
                tp_pct * 0.75,
                6,
            ),
            "close_fraction": 0.30,
        },
        {
            "leg": "tp3",
            "pnl_pct": round(
                tp_pct,
                6,
            ),
            "close_fraction": 0.40,
        },
    ]

    return sl_pct, tp_levels


# ============================================================================
# POSITION / PROTECTION
# ============================================================================

def reconcile_existing_position(
    symbol: str,
    direction: str,
    setup: dict,
    event_id: str,
) -> dict:
    """
    Если событие уже было исполнено, НЕ открывает позицию повторно.

    Вместо этого:
        existing position
            ↓
        protection check/recovery
    """

    direction = str(direction).upper()

    print(
        f"[EXECUTION_RECONCILE] "
        f"{symbol} direction={direction} "
        f"event={event_id}"
    )

    try:
        position = get_position_directional(
            symbol=symbol,
            direction=direction,
        )
    except TypeError:
        # Поддерживаем также positional implementation.
        position = get_position_directional(
            symbol,
            direction,
        )

    if not position:
        return {
            "status": "ALREADY_EXECUTED_POSITION_NOT_FOUND",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "protection_status": "NOT_VERIFIED",
        }

    print(
        f"[POSITION_FOUND] "
        f"{symbol} direction={direction} "
        f"position={position}"
    )

    try:
        sl_pct, tp_levels = build_tp_levels(
            setup,
            direction,
        )

        protection_result = ensure_directional_protection(
            symbol=symbol,
            direction=direction,
            position=position,
            setup=setup,
            sl_pct=sl_pct,
            tp_levels=tp_levels,
        )

    except TypeError:
        # Совместимость с реализацией,
        # где функция принимает позиционные аргументы.
        protection_result = ensure_directional_protection(
            symbol,
            direction,
            position,
            setup,
            sl_pct,
            tp_levels,
        )

    return {
        "status": "ALREADY_EXECUTED",
        "mode": EXECUTION_MODE,
        "order_id": position.get(
            "order_id"
        ) if isinstance(position, dict) else None,
        "position": position,
        "protection": protection_result,
        "protection_status": (
            protection_result.get("status")
            if isinstance(
                protection_result,
                dict,
            )
            else "UNKNOWN"
        ),
    }


def execute_new_position(
    symbol: str,
    direction: str,
    price: float,
    setup: dict,
    event_id: str,
) -> dict:
    """
    Event-driven execution.

    OPEN
      ↓
    WAIT FOR POSITION
      ↓
    CONFIRMED POSITION
      ↓
    PROTECTION
      ↓
    OPENED_PROTECTED
    """

    direction = str(direction).upper()

    # ----------------------------------------------------------------------
    # 1. Open market order
    # ----------------------------------------------------------------------

    trade_id = event_id.replace(
        "EVT_",
        "",
    )

    print(
        f"[EXECUTION_OPEN] "
        f"{symbol} "
        f"direction={direction} "
        f"entry={price:.8g} "
        f"event={event_id}"
    )

    try:
        opened = open_market(
            symbol,
            direction,
            price,
            trade_id,
        )
    except Exception as exc:
        return {
            "status": "OPEN_EXCEPTION",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "error": str(exc),
        }

    if not isinstance(opened, dict):
        return {
            "status": "OPEN_INVALID_RESPONSE",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "raw": repr(opened),
        }

    open_status = str(
        opened.get("status", "")
    ).lower()

    if open_status not in {
        "opened",
        "success",
        "ok",
    }:
        return {
            "status": "OPEN_FAILED",
            "mode": EXECUTION_MODE,
            "order_id": opened.get(
                "order_id"
            ),
            "open_result": opened,
        }

    order_id = opened.get(
        "order_id"
    )

    print(
        f"[EXECUTION_ORDER_ACCEPTED] "
        f"{symbol} "
        f"direction={direction} "
        f"order={order_id}"
    )

    # ----------------------------------------------------------------------
    # 2. Wait for actual exchange position
    # ----------------------------------------------------------------------

    try:
        position = wait_for_position_fill_directional(
            symbol=symbol,
            direction=direction,
            timeout_sec=15,
            poll_sec=0.5,
        )
    except TypeError:
        try:
            position = wait_for_position_fill_directional(
                symbol,
                direction,
                15,
                0.5,
            )
        except Exception as exc:
            return {
                "status": "POSITION_WAIT_FAILED",
                "mode": EXECUTION_MODE,
                "order_id": order_id,
                "open_result": opened,
                "error": str(exc),
            }
    except Exception as exc:
        return {
            "status": "POSITION_WAIT_FAILED",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
            "error": str(exc),
        }

    if not position:
        return {
            "status": "POSITION_NOT_CONFIRMED",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
        }

    print(
        f"[POSITION_CONFIRMED] "
        f"{symbol} direction={direction} "
        f"position={position}"
    )

    # ----------------------------------------------------------------------
    # 3. Compute protection from actual setup
    # ----------------------------------------------------------------------

    try:
        sl_pct, tp_levels = build_tp_levels(
            setup,
            direction,
        )
    except Exception as exc:
        return {
            "status": "PROTECTION_SETUP_INVALID",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "position": position,
            "error": str(exc),
        }

    print(
        f"[PROTECTION_PLAN] "
        f"{symbol} "
        f"direction={direction} "
        f"sl_pct={sl_pct:.4f} "
        f"tp_levels={tp_levels}"
    )

    # ----------------------------------------------------------------------
    # 4. Install SL + TP
    # ----------------------------------------------------------------------

    try:
        protection = ensure_directional_protection(
            symbol=symbol,
            direction=direction,
            position=position,
            setup=setup,
            sl_pct=sl_pct,
            tp_levels=tp_levels,
        )
    except TypeError:
        try:
            protection = ensure_directional_protection(
                symbol,
                direction,
                position,
                setup,
                sl_pct,
                tp_levels,
            )
        except Exception as exc:
            protection = {
                "status": "PROTECTION_EXCEPTION",
                "error": str(exc),
            }
    except Exception as exc:
        protection = {
            "status": "PROTECTION_EXCEPTION",
            "error": str(exc),
        }

    if not isinstance(protection, dict):
        protection = {
            "status": "UNKNOWN",
            "raw": repr(protection),
        }

    protection_status = str(
        protection.get("status", "")
    ).lower()

    print(
        f"[PROTECTION_RESULT] "
        f"{symbol} "
        f"direction={direction} "
        f"status={protection_status} "
        f"result={protection}"
    )

    if protection_status in {
        "ok",
        "protected",
        "created",
        "reconciled",
        "success",
        "ready",
    }:
        final_status = "opened_protected"
    else:
        final_status = "opened_protection_check_required"

    return {
        "status": final_status,
        "mode": EXECUTION_MODE,
        "order_id": order_id,
        "open_result": opened,
        "position": position,
        "protection": protection,
        "sl_pct": sl_pct,
        "tp_levels": tp_levels,
    }


# ============================================================================
# TELEGRAM
# ============================================================================

def build_fallback_signal_message(
    ev: dict,
    symbol: str,
    name: str,
    setup: dict,
    execution_result: dict,
    price: float,
) -> str:
    direction = str(
        ev.get("direction", "")
    ).upper()

    label = (
        "🚨 LONG SIGNAL"
        if direction == "LONG"
        else "🔻 SHORT SIGNAL"
    )

    sl = setup.get(
        "invalidation_price"
    )
    tp = setup.get(
        "target_price"
    )
    rr = setup.get(
        "rr"
    )

    return (
        f"{label}\n\n"
        f"<b>{name or symbol}</b> "
        f"(<code>{symbol}</code>)\n\n"
        f"Event: <code>{ev.get('event_type')}</code>\n"
        f"TF: <b>{ev.get('timeframe', '1h')}</b> + trigger 15m\n"
        f"Price: <code>{price:.8g}</code>\n"
        f"Detected: <code>"
        f"{ev.get('timestamps', {}).get('detected_at_ts')}"
        f"</code>\n\n"
        f"<b>SETUP</b>\n"
        f"Entry: <code>{price:.8g}</code>\n"
        f"SL: <code>"
        f"{sl:.8g}"
        f"</code>\n"
        f"TP: <code>"
        f"{tp:.8g}"
        f"</code>\n"
        f"R:R: <code>"
        f"{rr:.2f}"
        f"</code>\n\n"
        f"<b>EXECUTION</b>\n"
        f"Mode: <code>{EXECUTION_MODE}</code>\n"
        f"Status: <code>"
        f"{execution_result.get('status')}"
        f"</code>\n"
        f"Order: <code>"
        f"{execution_result.get('order_id') or '—'}"
        f"</code>\n\n"
        f"⚡ Event-driven — 5×5m lifecycle is NOT used"
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    stats = {
        "coinalyze_rows": 0,
        "candidates_before_limit": 0,
        "candidates": 0,
        "bingx_mapped": 0,
        "bingx_unmapped": 0,
        "klines_1h_ok": 0,
        "klines_15m_ok": 0,
        "rsi_events": 0,
        "cvd_events": 0,
        "squeeze_events": 0,
        "events_total": 0,
        "events_recent": 0,
        "events_duplicate": 0,
        "events_cvd_gate_rejected": 0,
        "trigger_pass": 0,
        "trigger_rejected": 0,
        "setup_rejected": 0,
        "setups": 0,
        "execution_attempts": 0,
        "trades": 0,
        "telegram_attempts": 0,
        "telegram_sent": 0,
        "scan_errors": 0,
    }

    # ----------------------------------------------------------------------
    # Coinalyze
    # ----------------------------------------------------------------------

    rows = fetch_data()

    stats["coinalyze_rows"] = len(rows)

    print(
        f"[ENGINE] Coinalyze rows={len(rows)}"
    )

    # ----------------------------------------------------------------------
    # BingX contracts
    # ----------------------------------------------------------------------

    try:
        refresh_contracts()
    except Exception as exc:
        stats["scan_errors"] += 1
        print(
            f"[BINGX] contracts refresh error={exc}"
        )

    candidates: List[Any] = []

    for r in rows:
        try:
            if (
                r.price is None
                or r.price <= 0
            ):
                continue

            if (
                r.volume24 is None
                or r.volume24 < MIN_VOL
            ):
                continue

            if (
                r.oi is None
                or r.oi < MIN_OI
            ):
                continue

            contract = get_contract(
                r.symbol
            )

            if not contract:
                stats["bingx_unmapped"] += 1

                print(
                    f"[MAP] NO_BINGX "
                    f"coinalyze={r.symbol} "
                    f"name={getattr(r, 'name', '')}"
                )

                continue

            stats["bingx_mapped"] += 1

            print(
                f"[MAP] OK "
                f"coinalyze={r.symbol} "
                f"bingx={contract.get('symbol', r.symbol)} "
                f"displayName={contract.get('displayName', '')}"
            )

            candidates.append(r)

        except Exception as exc:
            stats["scan_errors"] += 1

            print(
                f"[CANDIDATE_ERROR] "
                f"{getattr(r, 'symbol', '?')}: {exc}"
            )

    stats["candidates_before_limit"] = len(
        candidates
    )

    candidates = candidates[:MAX_CANDIDATES]

    stats["candidates"] = len(
        candidates
    )

    print(
        f"[ENGINE] Coinalyze candidates="
        f"{len(candidates)} "
        f"execution={EXECUTION_ENABLED} "
        f"env={EXECUTION_MODE}"
    )

    seen_events = load_ids(EVENTS)
    executed_event_ids = load_ids(TRADES)

    trades_this_cycle = 0

    # =========================================================================
    # SYMBOL LOOP
    # =========================================================================

    for r in candidates:
        symbol = r.symbol

        try:
            # ------------------------------------------------------------------
            # KLINES
            # ------------------------------------------------------------------

            k1 = fetch_klines(
                symbol,
                "1h",
                int(
                    os.environ.get(
                        "KLINE_LIMIT_1H",
                        "250",
                    )
                ),
            )

            k15 = fetch_klines(
                symbol,
                "15m",
                int(
                    os.environ.get(
                        "KLINE_LIMIT_15M",
                        "250",
                    )
                ),
            )

            if len(k1) < 60:
                print(
                    f"[DATA_REJECT] "
                    f"{symbol} 1H bars={len(k1)} < 60"
                )
                continue

            if len(k15) < 10:
                print(
                    f"[DATA_REJECT] "
                    f"{symbol} 15m bars={len(k15)} < 10"
                )
                continue

            stats["klines_1h_ok"] += 1
            stats["klines_15m_ok"] += 1

            # ------------------------------------------------------------------
            # DATAFRAMES
            # ------------------------------------------------------------------

            d1 = add_cvd(
                pd.DataFrame(k1)
            )

            d15 = pd.DataFrame(k15)

            # ------------------------------------------------------------------
            # EVENT DETECTION
            # ------------------------------------------------------------------

            divergence_events = detect_divergences(
                d1,
                symbol,
                "1h",
            )

            squeeze_events = detect_squeeze_release(
                d1,
                symbol,
                "1h",
            )

            rsi_events = [
                e
                for e in divergence_events
                if "_RSI" in e.get(
                    "event_type",
                    "",
                )
            ]

            cvd_events = [
                e
                for e in divergence_events
                if "BINGX_CVD" in e.get(
                    "event_type",
                    "",
                )
            ]

            stats["rsi_events"] += len(
                rsi_events
            )

            stats["cvd_events"] += len(
                cvd_events
            )

            stats["squeeze_events"] += len(
                squeeze_events
            )

            stats["events_total"] += (
                len(divergence_events)
                + len(squeeze_events)
            )

            print(
                f"[EVENT_SCAN] {symbol} "
                f"RSI={len(rsi_events)} "
                f"CVD={len(cvd_events)} "
                f"SQUEEZE={len(squeeze_events)}"
            )

            all_events = (
                divergence_events
                + squeeze_events
            )

            # ------------------------------------------------------------------
            # EVENTS
            # ------------------------------------------------------------------

            for ev in all_events:
                event_id = ev.get(
                    "event_id"
                )

                if not event_id:
                    print(
                        f"[EVENT_REJECT] "
                        f"{symbol} event without event_id"
                    )
                    continue

                direction = str(
                    ev.get(
                        "direction",
                        "",
                    )
                ).upper()

                if direction not in {
                    "LONG",
                    "SHORT",
                }:
                    print(
                        f"[EVENT_REJECT_DIRECTION] "
                        f"{symbol} direction={direction}"
                    )
                    continue

                # --------------------------------------------------------------
                # EVENT AGE
                # --------------------------------------------------------------

                detected_at = int(
                    ev.get(
                        "timestamps",
                        {},
                    ).get(
                        "detected_at_ts",
                        0,
                    )
                )

                latest_close = int(
                    d1["close_time"].iloc[-1]
                )

                age = (
                    latest_close - detected_at
                ) / 60000.0

                if (
                    age < 0
                    or age > MAX_AGE
                ):
                    print(
                        f"[EVENT_REJECT_AGE] "
                        f"{symbol} "
                        f"type={ev.get('event_type')} "
                        f"age={age:.1f}m "
                        f"max={MAX_AGE}m"
                    )
                    continue

                stats["events_recent"] += 1

                # --------------------------------------------------------------
                # PERSIST EVENT
                # --------------------------------------------------------------

                already_seen = (
                    event_id in seen_events
                )

                if already_seen:
                    stats["events_duplicate"] += 1
                else:
                    emit_event(ev)
                    seen_events.add(
                        event_id
                    )

                # --------------------------------------------------------------
                # OPTIONAL CVD CONFIRMATION
                # --------------------------------------------------------------

                if (
                    REQUIRE_CVD
                    and "_RSI" in ev.get(
                        "event_type",
                        "",
                    )
                ):
                    pivot_1 = ev.get(
                        "timestamps",
                        {},
                    ).get(
                        "pivot_1_ts"
                    )

                    pivot_2 = ev.get(
                        "timestamps",
                        {},
                    ).get(
                        "pivot_2_ts"
                    )

                    matched_cvd = any(
                        (
                            other.get(
                                "direction"
                            )
                            == direction
                            and
                            other.get(
                                "timestamps",
                                {},
                            ).get(
                                "pivot_1_ts"
                            )
                            == pivot_1
                            and
                            other.get(
                                "timestamps",
                                {},
                            ).get(
                                "pivot_2_ts"
                            )
                            == pivot_2
                        )
                        for other
                        in cvd_events
                    )

                    if not matched_cvd:
                        stats[
                            "events_cvd_gate_rejected"
                        ] += 1

                        print(
                            f"[EVENT_REJECT_CVD] "
                            f"{symbol} "
                            f"type={ev.get('event_type')}"
                        )

                        continue

                # --------------------------------------------------------------
                # 15M TRIGGER
                # --------------------------------------------------------------

                trigger = build_15m_trigger(
                    d15,
                    direction,
                )

                if (
                    REQUIRE_TRIGGER
                    and not trigger
                ):
                    stats[
                        "trigger_rejected"
                    ] += 1

                    print(
                        f"[TRIGGER_REJECT] "
                        f"symbol={symbol} "
                        f"direction={direction} "
                        f"event={ev.get('event_type')} "
                        f"reason=15m_trigger_failed"
                    )

                    continue

                stats["trigger_pass"] += 1

                # --------------------------------------------------------------
                # ENTRY PRICE
                # --------------------------------------------------------------

                fact = ev.get(
                    "event_fact",
                    {},
                )

                price_raw = fact.get(
                    "detection_close_price"
                )

                if price_raw is None:
                    price_raw = (
                        fact.get(
                            "close"
                        )
                        or getattr(
                            r,
                            "price",
                            None,
                        )
                    )

                if price_raw is None:
                    print(
                        f"[SETUP_REJECT] "
                        f"{symbol} "
                        f"event={event_id} "
                        f"reason=no_entry_price"
                    )

                    stats[
                        "setup_rejected"
                    ] += 1

                    continue

                price = float(
                    price_raw
                )

                # --------------------------------------------------------------
                # SETUP
                # --------------------------------------------------------------

                try:
                    setup = build_event_setup(
                        ev=ev,
                        df_1h=d1,
                        entry_price=price,
                    )
                except Exception as exc:
                    stats[
                        "setup_rejected"
                    ] += 1

                    print(
                        f"[SETUP_REJECT] "
                        f"{symbol} "
                        f"event={event_id} "
                        f"reason={exc}"
                    )

                    continue

                stats["setups"] += 1

                print(
                    f"[SETUP_OK] "
                    f"{symbol} "
                    f"direction={direction} "
                    f"entry={setup['entry_reference']:.8g} "
                    f"SL={setup['invalidation_price']:.8g} "
                    f"TP={setup['target_price']:.8g} "
                    f"RR={setup['rr']:.2f}"
                )

                execution_result: Dict[str, Any]

                # ==============================================================
                # EXISTING EVENT / POSITION
                # ==============================================================

                if event_id in executed_event_ids:

                    if EXECUTION_ENABLED:

                        execution_result = (
                            reconcile_existing_position(
                                symbol=symbol,
                                direction=direction,
                                setup=setup,
                                event_id=event_id,
                            )
                        )

                    else:
                        execution_result = {
                            "status": "ALREADY_EXECUTED",
                            "mode": EXECUTION_MODE,
                            "order_id": None,
                            "protection_status": (
                                "NOT_CHECKED_EXECUTION_DISABLED"
                            ),
                        }

                    print(
                        f"[EXECUTION_RESULT] "
                        f"{symbol} "
                        f"event={event_id} "
                        f"status={execution_result.get('status')}"
                    )

                # ==============================================================
                # NEW EXECUTION
                # ==============================================================

                elif (
                    EXECUTION_ENABLED
                    and trades_this_cycle
                    < MAX_TRADES
                ):

                    stats[
                        "execution_attempts"
                    ] += 1

                    execution_result = (
                        execute_new_position(
                            symbol=symbol,
                            direction=direction,
                            price=price,
                            setup=setup,
                            event_id=event_id,
                        )
                    )

                    record_trade(
                        {
                            "event_id": event_id,
                            "symbol": symbol,
                            "direction": direction,
                            "price": price,
                            "event_type": ev.get(
                                "event_type"
                            ),
                            "ts": int(
                                pd.Timestamp.utcnow()
                                .timestamp()
                                * 1000
                            ),
                            "result":
                                execution_result,
                            "setup":
                                setup,
                        }
                    )

                    executed_event_ids.add(
                        event_id
                    )

                    status = str(
                        execution_result.get(
                            "status",
                            "",
                        )
                    )

                    if status == "opened_protected":
                        trades_this_cycle += 1
                        stats["trades"] += 1

                        print(
                            f"[EXECUTION_OPENED_PROTECTED] "
                            f"{symbol} "
                            f"direction={direction} "
                            f"order={execution_result.get('order_id')}"
                        )

                    elif status.startswith(
                        "opened_"
                    ):
                        trades_this_cycle += 1
                        stats["trades"] += 1

                        print(
                            f"[EXECUTION_OPENED] "
                            f"{symbol} "
                            f"direction={direction} "
                            f"status={status}"
                        )

                    else:
                        print(
                            f"[EXECUTION_RESULT] "
                            f"{symbol} "
                            f"status={status}"
                        )

                # ==============================================================
                # EXECUTION DISABLED
                # ==============================================================

                elif not EXECUTION_ENABLED:

                    execution_result = {
                        "status": "DISABLED",
                        "mode": EXECUTION_MODE,
                        "order_id": None,
                        "protection_status": (
                            "NOT_ATTEMPTED"
                        ),
                    }

                # ==============================================================
                # LIMIT REACHED
                # ==============================================================

                else:

                    execution_result = {
                        "status": "TRADE_LIMIT_REACHED",
                        "mode": EXECUTION_MODE,
                        "order_id": None,
                    }

                # --------------------------------------------------------------
                # TELEGRAM
                # --------------------------------------------------------------

                label = (
                    "🚨 LONG SIGNAL"
                    if direction == "LONG"
                    else "🔻 SHORT SIGNAL"
                )

                try:
                    msg = format_signal(
                        ev,
                        setup=setup,
                        coinalyze_row=r,
                        execution=execution_result,
                    )

                    # Гарантируем требуемый заголовок.
                    if not (
                        msg.startswith(
                            "🚨 LONG SIGNAL"
                        )
                        or msg.startswith(
                            "🔻 SHORT SIGNAL"
                        )
                    ):
                        msg = (
                            f"{label}\n\n"
                            + msg
                        )

                except Exception as exc:

                    msg = (
                        build_fallback_signal_message(
                            ev=ev,
                            symbol=symbol,
                            name=(
                                getattr(
                                    r,
                                    "name",
                                    None,
                                )
                                or symbol
                            ),
                            setup=setup,
                            execution_result=(
                                execution_result
                            ),
                            price=price,
                        )
                    )

                    print(
                        f"[TELEGRAM_FORMAT_FALLBACK] "
                        f"{symbol}: {exc}"
                    )

                stats[
                    "telegram_attempts"
                ] += 1

                try:
                    sent = send_tg(msg)
                except Exception as exc:
                    sent = False

                    print(
                        f"[TELEGRAM_ERROR] "
                        f"{symbol}: {exc}"
                    )

                if sent:
                    stats[
                        "telegram_sent"
                    ] += 1

                record_action(
                    {
                        "event_id": event_id,
                        "symbol": symbol,
                        "direction": direction,
                        "event_type": ev.get(
                            "event_type"
                        ),
                        "telegram_sent": bool(
                            sent
                        ),
                        "execution_status":
                            execution_result.get(
                                "status"
                            ),
                        "protection_status":
                            execution_result.get(
                                "protection_status"
                            ),
                        "ts": int(
                            pd.Timestamp.utcnow()
                            .timestamp()
                            * 1000
                        ),
                    }
                )

        except Exception as exc:

            stats["scan_errors"] += 1

            print(
                f"[SCAN_ERROR] "
                f"{symbol}: {exc}"
            )

    print(
        f"[ENGINE] trades_this_cycle="
        f"{trades_this_cycle}"
    )

    print(
        "[ENGINE_SUMMARY] "
        + " ".join(
            f"{key}={value}"
            for key, value in stats.items()
        )
    )


if __name__ == "__main__":
    main()
