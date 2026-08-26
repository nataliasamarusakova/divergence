from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from event_engine.coinalyze import fetch_data
from event_engine.bingx import (
    refresh_contracts,
    get_contract,
    fetch_klines,
    open_market,
)
from event_engine.signals import (
    add_cvd,
    detect_divergences,
    detect_squeeze_release,
    build_15m_trigger,
)
from event_engine.telegram import send as send_tg, format_signal


logging.basicConfig(level=logging.INFO, format="%(message)s")


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

KLINE_LIMIT_1H = int(os.environ.get("KLINE_LIMIT_1H", "250"))
KLINE_LIMIT_15M = int(os.environ.get("KLINE_LIMIT_15M", "250"))

# ---------------------------------------------------------------------
# EXECUTION RISK PARAMETERS
# ---------------------------------------------------------------------

# Структурный buffer для дивергенции.
SWING_BUFFER_ATR = float(os.environ.get("SWING_BUFFER_ATR", "0.25"))

# Для squeeze нет P2 structural swing, поэтому используется ATR.
SQUEEZE_SL_ATR = float(os.environ.get("SQUEEZE_SL_ATR", "1.0"))

# Минимальный R:R.
MIN_RR = float(os.environ.get("MIN_RR", "2.0"))

# TP = entry + RR * risk.
TARGET_R_MULTIPLE = float(os.environ.get("TARGET_R_MULTIPLE", "2.0"))

# Дополнительная защита от ненормальных стопов.
MIN_SL_PCT = float(os.environ.get("MIN_SL_PCT", "0.20"))
MAX_SL_PCT = float(os.environ.get("MAX_SL_PCT", "15.0"))


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids: set[str] = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        try:
            value = json.loads(line).get("event_id")
            if value:
                ids.add(value)
        except Exception:
            continue

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


# =====================================================================
# INDICATORS
# =====================================================================

def calculate_atr(
    df: pd.DataFrame,
    length: int = 14,
) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(
        window=length,
        min_periods=length,
    ).mean()


# =====================================================================
# SETUP BUILDER
# =====================================================================

def build_execution_setup(
    event: dict,
    df_1h: pd.DataFrame,
) -> tuple[Optional[dict], Optional[str]]:
    """
    Строит executable risk geometry ДО open_market().

    Дивергенция:
        LONG  -> SL ниже P2 low
        SHORT -> SL выше P2 high

    Squeeze:
        используется ATR-based invalidation.

    TP рассчитывается от фиксированного R multiple.
    """

    if df_1h.empty or len(df_1h) < 20:
        return None, "INSUFFICIENT_1H_DATA"

    work = df_1h.copy().reset_index(drop=True)

    for col in ("open", "high", "low", "close"):
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(
        subset=["open", "high", "low", "close"]
    ).reset_index(drop=True)

    if len(work) < 20:
        return None, "INVALID_1H_DATA"

    work["atr14"] = calculate_atr(work, 14)

    atr = float(work["atr14"].iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        return None, "INVALID_ATR"

    direction = str(event.get("direction", "")).upper()

    if direction not in {"LONG", "SHORT"}:
        return None, "INVALID_DIRECTION"

    event_type = str(event.get("event_type", ""))

    event_fact = event.get("event_fact", {})

    entry_price = event_fact.get("detection_close_price")

    if entry_price is None:
        entry_price = float(work["close"].iloc[-1])

    entry_price = float(entry_price)

    if not np.isfinite(entry_price) or entry_price <= 0:
        return None, "INVALID_ENTRY_PRICE"

    # ---------------------------------------------------------------
    # 1. STRUCTURAL DIVERGENCE
    # ---------------------------------------------------------------

    p2_idx = None

    timestamps = event.get("timestamps", {})
    pivot_2_ts = timestamps.get("pivot_2_ts")

    if "_RSI" in event_type or "BINGX_CVD" in event_type:
        if pivot_2_ts is not None:
            close_times = pd.to_numeric(
                work["close_time"],
                errors="coerce",
            )

            matches = np.where(
                close_times.to_numpy(dtype=np.int64) == int(pivot_2_ts)
            )[0]

            if len(matches):
                p2_idx = int(matches[-1])

    if p2_idx is not None:
        pivot_high = float(work["high"].iloc[p2_idx])
        pivot_low = float(work["low"].iloc[p2_idx])

        if direction == "LONG":
            invalidation = pivot_low - SWING_BUFFER_ATR * atr

            # Если detection close оказался ниже структуры,
            # структурный SL уже бессмысленен.
            if invalidation >= entry_price:
                invalidation = entry_price - SWING_BUFFER_ATR * atr

        else:
            invalidation = pivot_high + SWING_BUFFER_ATR * atr

            if invalidation <= entry_price:
                invalidation = entry_price + SWING_BUFFER_ATR * atr

        setup_type = "STRUCTURAL_DIVERGENCE"

    # ---------------------------------------------------------------
    # 2. SQUEEZE
    # ---------------------------------------------------------------

    elif event_type == "VOLATILITY_SQUEEZE_RELEASE":
        if direction == "LONG":
            invalidation = entry_price - SQUEEZE_SL_ATR * atr
        else:
            invalidation = entry_price + SQUEEZE_SL_ATR * atr

        setup_type = "ATR_SQUEEZE"

    else:
        return None, "UNSUPPORTED_EVENT_TYPE"

    # ---------------------------------------------------------------
    # 3. RISK GEOMETRY
    # ---------------------------------------------------------------

    if direction == "LONG":
        risk = entry_price - invalidation
    else:
        risk = invalidation - entry_price

    if not np.isfinite(risk) or risk <= 0:
        return None, "INVALID_RISK_DISTANCE"

    risk_pct = risk / entry_price * 100.0

    if risk_pct < MIN_SL_PCT:
        return None, "SL_TOO_TIGHT"

    if risk_pct > MAX_SL_PCT:
        return None, "SL_TOO_WIDE"

    target_distance = risk * TARGET_R_MULTIPLE

    if direction == "LONG":
        target_price = entry_price + target_distance
    else:
        target_price = entry_price - target_distance

    rr = target_distance / risk

    if not np.isfinite(rr) or rr < MIN_RR:
        return None, "RR_BELOW_MINIMUM"

    # Не допускаем неправильного расположения TP.
    if direction == "LONG":
        if target_price <= entry_price or invalidation >= entry_price:
            return None, "INVALID_LONG_GEOMETRY"
    else:
        if target_price >= entry_price or invalidation <= entry_price:
            return None, "INVALID_SHORT_GEOMETRY"

    setup = {
        "setup_type": setup_type,
        "entry_reference": entry_price,
        "invalidation_price": float(invalidation),
        "target_price": float(target_price),
        "risk_distance": float(risk),
        "risk_pct": float(risk_pct),
        "rr": float(rr),
        "target_r_multiple": TARGET_R_MULTIPLE,
        "atr14": float(atr),
        "trigger_ok": True,
    }

    return setup, None


# =====================================================================
# CVD CONFIRMATION
# =====================================================================

def has_cvd_confirmation(
    event: dict,
    cvd_events: list[dict],
) -> bool:
    """
    RSI divergence требует CVD на той же физической паре P1/P2
    и том же направлении.

    Squeeze здесь НЕ проверяется:
    squeeze — отдельный event type.
    """

    event_type = str(event.get("event_type", ""))

    if "BINGX_CVD" in event_type:
        return True

    if "_RSI" not in event_type:
        return True

    direction = event.get("direction")
    timestamps = event.get("timestamps", {})

    p1 = timestamps.get("pivot_1_ts")
    p2 = timestamps.get("pivot_2_ts")

    for other in cvd_events:
        if other.get("direction") != direction:
            continue

        other_ts = other.get("timestamps", {})

        if other_ts.get("pivot_1_ts") != p1:
            continue

        if other_ts.get("pivot_2_ts") != p2:
            continue

        return True

    return False


# =====================================================================
# TELEGRAM
# =====================================================================

def build_fallback_message(
    event: dict,
    setup: Optional[dict],
    row: Any,
    execution: Optional[dict],
    block_reason: Optional[str] = None,
) -> str:

    direction = str(event.get("direction", "")).upper()

    label = (
        "🚨 LONG SIGNAL"
        if direction == "LONG"
        else "🔻 SHORT SIGNAL"
    )

    symbol = event.get("symbol", "?")
    event_type = event.get("event_type", "?")

    price = (
        setup["entry_reference"]
        if setup
        else event.get("event_fact", {}).get(
            "detection_close_price",
            0,
        )
    )

    execution_status = (
        execution.get("status")
        if execution
        else "NOT_ATTEMPTED"
    )

    lines = [
        label,
        "",
        f"<b>{getattr(row, 'name', None) or symbol}</b> "
        f"(<code>{symbol}</code>)",
        "",
        f"Event: <code>{event_type}</code>",
        "TF: <b>1H + trigger 15m</b>",
        f"Price: <code>{float(price):.8g}</code>",
        "",
        "<b>SETUP</b>",
    ]

    if setup:
        lines.extend(
            [
                f"Entry: <code>{setup['entry_reference']:.8g}</code>",
                f"SL: <code>{setup['invalidation_price']:.8g}</code>",
                f"TP: <code>{setup['target_price']:.8g}</code>",
                f"R:R: <code>{setup['rr']:.2f}</code>",
                f"Risk: <code>{setup['risk_pct']:.2f}%</code>",
            ]
        )
    else:
        lines.extend(
            [
                "Entry: <code>—</code>",
                "SL: <code>—</code>",
                "TP: <code>—</code>",
                "R:R: <code>—</code>",
            ]
        )

    lines.extend(
        [
            "",
            "<b>EXECUTION</b>",
            f"Mode: <code>{os.environ.get('EXECUTION_MODE', 'vst')}</code>",
            f"Status: <code>{execution_status}</code>",
        ]
    )

    if execution and execution.get("order_id"):
        lines.append(
            f"Order: <code>{execution['order_id']}</code>"
        )
    else:
        lines.append("Order: <code>—</code>")

    if block_reason:
        lines.extend(
            [
                "",
                f"⛔ <b>BLOCKED:</b> <code>{block_reason}</code>",
            ]
        )

    lines.extend(
        [
            "",
            "⚡ Event-driven — 5×5m lifecycle is NOT used",
        ]
    )

    return "\n".join(lines)


# =====================================================================
# MAIN
# =====================================================================

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

    rows = fetch_data()
    stats["coinalyze_rows"] = len(rows)

    print(
        f"[ENGINE] Coinalyze rows={len(rows)}"
    )

    try:
        refresh_contracts()
    except Exception as exc:
        stats["scan_errors"] += 1
        print(
            f"[BINGX] contracts refresh error={exc}"
        )

    candidates = []

    for r in rows:

        if r.price is None or r.price <= 0:
            continue

        if r.volume24 is None or r.volume24 < MIN_VOL:
            continue

        if r.oi is None or r.oi < MIN_OI:
            continue

        contract = get_contract(r.symbol)

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

    stats["candidates_before_limit"] = len(candidates)

    candidates = candidates[:MAX_CANDIDATES]

    stats["candidates"] = len(candidates)

    print(
        f"[ENGINE] Coinalyze candidates={len(candidates)} "
        f"execution={EXECUTION_ENABLED} "
        f"env={os.environ.get('EXECUTION_MODE', os.environ.get('BINGX_ENV', 'vst'))}"
    )

    seen_events = load_ids(EVENTS)
    executed_event_ids = load_ids(TRADES)

    trades = 0

    for r in candidates:

        symbol = r.symbol

        try:

            k1 = fetch_klines(
                symbol,
                "1h",
                KLINE_LIMIT_1H,
            )

            k15 = fetch_klines(
                symbol,
                "15m",
                KLINE_LIMIT_15M,
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

            d1 = add_cvd(
                pd.DataFrame(k1)
            )

            rsi_events_all = detect_divergences(
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
                e for e in rsi_events_all
                if "_RSI" in e.get("event_type", "")
            ]

            cvd_events = [
                e for e in rsi_events_all
                if "BINGX_CVD" in e.get("event_type", "")
            ]

            stats["rsi_events"] += len(rsi_events)
            stats["cvd_events"] += len(cvd_events)
            stats["squeeze_events"] += len(squeeze_events)

            stats["events_total"] += (
                len(rsi_events_all)
                + len(squeeze_events)
            )

            print(
                f"[EVENT_SCAN] "
                f"{symbol} "
                f"RSI={len(rsi_events)} "
                f"CVD={len(cvd_events)} "
                f"SQUEEZE={len(squeeze_events)}"
            )

            all_events = (
                rsi_events_all
                + squeeze_events
            )

            for ev in all_events:

                event_id = ev.get("event_id")

                if not event_id:
                    print(
                        f"[EVENT_REJECT] "
                        f"{symbol} event without event_id"
                    )
                    continue

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
                    d1.close_time.iloc[-1]
                )

                age = (
                    latest_close - detected_at
                ) / 60000.0

                if age < 0 or age > MAX_AGE:

                    print(
                        f"[EVENT_REJECT_AGE] "
                        f"{symbol} "
                        f"type={ev.get('event_type')} "
                        f"age={age:.1f}m "
                        f"max={MAX_AGE}m"
                    )

                    continue

                stats["events_recent"] += 1

                # -----------------------------------------------------
                # EVENT PERSISTENCE
                # -----------------------------------------------------

                if event_id in seen_events:

                    stats["events_duplicate"] += 1

                else:

                    emit_event(ev)
                    seen_events.add(event_id)

                # -----------------------------------------------------
                # CVD GATE
                # -----------------------------------------------------

                if (
                    REQUIRE_CVD
                    and "_RSI" in ev.get(
                        "event_type",
                        "",
                    )
                ):

                    if not has_cvd_confirmation(
                        ev,
                        cvd_events,
                    ):

                        stats[
                            "events_cvd_gate_rejected"
                        ] += 1

                        print(
                            f"[EVENT_REJECT_CVD] "
                            f"{symbol} "
                            f"type={ev.get('event_type')} "
                            f"reason=no_matching_cvd"
                        )

                        # Не строим setup,
                        # не открываем trade.
                        continue

                # -----------------------------------------------------
                # 15M TRIGGER
                # -----------------------------------------------------

                trigger = build_15m_trigger(
                    pd.DataFrame(k15),
                    ev["direction"],
                )

                if REQUIRE_TRIGGER and not trigger:

                    stats["trigger_rejected"] += 1

                    print(
                        f"[TRIGGER_REJECT] "
                        f"symbol={symbol} "
                        f"direction={ev.get('direction')} "
                        f"event={ev.get('event_type')} "
                        f"reason=15m_trigger_failed"
                    )

                    continue

                stats["trigger_pass"] += 1

                # -----------------------------------------------------
                # SETUP
                # -----------------------------------------------------

                setup, setup_reason = (
                    build_execution_setup(
                        ev,
                        d1,
                    )
                )

                if setup is None:

                    stats["setup_rejected"] += 1

                    print(
                        f"[SETUP_REJECT] "
                        f"{symbol} "
                        f"event={event_id} "
                        f"type={ev.get('event_type')} "
                        f"reason={setup_reason}"
                    )

                    # Сигнал пользователю все равно отправляем,
                    # но execution категорически не пытаемся.
                    execution_result = {
                        "status": "SETUP_REJECTED",
                        "mode": os.environ.get(
                            "EXECUTION_MODE",
                            "vst",
                        ),
                        "order_id": None,
                        "reason": setup_reason,
                    }

                    stats["telegram_attempts"] += 1

                    try:
                        msg = format_signal(
                            ev,
                            setup={
                                "entry_reference":
                                    ev.get(
                                        "event_fact",
                                        {},
                                    ).get(
                                        "detection_close_price"
                                    ),
                                "invalidation_price": None,
                                "target_price": None,
                                "rr": None,
                            },
                            coinalyze_row=r,
                            execution=execution_result,
                        )
                    except Exception:
                        msg = build_fallback_message(
                            ev,
                            None,
                            r,
                            execution_result,
                            setup_reason,
                        )

                    try:
                        sent = bool(send_tg(msg))
                    except Exception as exc:
                        sent = False
                        print(
                            f"[TELEGRAM_ERROR] "
                            f"{symbol}: {exc}"
                        )

                    if sent:
                        stats["telegram_sent"] += 1

                    record_action(
                        {
                            "event_id": event_id,
                            "symbol": symbol,
                            "direction": ev["direction"],
                            "event_type": ev.get(
                                "event_type"
                            ),
                            "telegram_sent": bool(sent),
                            "execution_status":
                                "SETUP_REJECTED",
                            "setup_reject_reason":
                                setup_reason,
                            "ts": int(
                                pd.Timestamp.utcnow()
                                .timestamp()
                                * 1000
                            ),
                        }
                    )

                    continue

                stats["setups"] += 1

                print(
                    f"[SETUP_OK] "
                    f"{symbol} "
                    f"direction={ev['direction']} "
                    f"entry={setup['entry_reference']:.8g} "
                    f"SL={setup['invalidation_price']:.8g} "
                    f"TP={setup['target_price']:.8g} "
                    f"RR={setup['rr']:.2f} "
                    f"risk={setup['risk_pct']:.2f}%"
                )

                # -----------------------------------------------------
                # EXECUTION
                # -----------------------------------------------------

                execution_result = None

                if event_id in executed_event_ids:

                    execution_result = {
                        "status": "ALREADY_EXECUTED",
                        "mode": os.environ.get(
                            "EXECUTION_MODE",
                            "vst",
                        ),
                        "order_id": None,
                    }

                    print(
                        f"[EXECUTION_SKIP_DUPLICATE] "
                        f"{symbol} "
                        f"event={event_id}"
                    )

                elif EXECUTION_ENABLED and trades < MAX_TRADES:

                    stats["execution_attempts"] += 1

                    trade_id = (
                        event_id
                        .replace(
                            "EVT_",
                            "",
                        )
                    )

                    # -------------------------------------------------
                    # FINAL HARD GATE
                    # -------------------------------------------------

                    if not setup.get(
                        "invalidation_price"
                    ):
                        execution_result = {
                            "status":
                                "BLOCKED_NO_SL",
                            "mode":
                                os.environ.get(
                                    "EXECUTION_MODE",
                                    "vst",
                                ),
                            "order_id":
                                None,
                        }

                    elif not setup.get(
                        "target_price"
                    ):
                        execution_result = {
                            "status":
                                "BLOCKED_NO_TP",
                            "mode":
                                os.environ.get(
                                    "EXECUTION_MODE",
                                    "vst",
                                ),
                            "order_id":
                                None,
                        }

                    elif (
                        setup.get("rr") is None
                        or setup["rr"] < MIN_RR
                    ):
                        execution_result = {
                            "status":
                                "BLOCKED_BAD_RR",
                            "mode":
                                os.environ.get(
                                    "EXECUTION_MODE",
                                    "vst",
                                ),
                            "order_id":
                                None,
                        }

                    else:

                        execution_result = open_market(
                            symbol,
                            ev["direction"],
                            setup[
                                "entry_reference"
                            ],
                            trade_id,
                        )

                        record_trade(
                            {
                                "event_id":
                                    event_id,
                                "symbol":
                                    symbol,
                                "direction":
                                    ev["direction"],
                                "price":
                                    setup[
                                        "entry_reference"
                                    ],
                                "sl":
                                    setup[
                                        "invalidation_price"
                                    ],
                                "tp":
                                    setup[
                                        "target_price"
                                    ],
                                "rr":
                                    setup["rr"],
                                "event_type":
                                    ev.get(
                                        "event_type"
                                    ),
                                "ts":
                                    int(
                                        pd.Timestamp.utcnow()
                                        .timestamp()
                                        * 1000
                                    ),
                                "result":
                                    execution_result,
                            }
                        )

                        if execution_result.get(
                            "status"
                        ) == "opened":

                            trades += 1
                            stats["trades"] += 1

                            print(
                                f"[EXECUTION_OPENED] "
                                f"{symbol} "
                                f"direction={ev['direction']} "
                                f"entry={setup['entry_reference']:.8g} "
                                f"SL={setup['invalidation_price']:.8g} "
                                f"TP={setup['target_price']:.8g} "
                                f"RR={setup['rr']:.2f} "
                                f"order={execution_result.get('order_id')}"
                            )

                        else:

                            print(
                                f"[EXECUTION_RESULT] "
                                f"{symbol} "
                                f"status={execution_result.get('status')} "
                                f"message={execution_result.get('msg')}"
                            )

                        executed_event_ids.add(
                            event_id
                        )

                elif not EXECUTION_ENABLED:

                    execution_result = {
                        "status": "DISABLED",
                        "mode": os.environ.get(
                            "EXECUTION_MODE",
                            "vst",
                        ),
                        "order_id": None,
                    }

                else:

                    execution_result = {
                        "status":
                            "TRADE_LIMIT_REACHED",
                        "mode":
                            os.environ.get(
                                "EXECUTION_MODE",
                                "vst",
                            ),
                        "order_id":
                            None,
                    }

                # -----------------------------------------------------
                # TELEGRAM
                # -----------------------------------------------------

                stats["telegram_attempts"] += 1

                try:

                    msg = format_signal(
                        ev,
                        setup=setup,
                        coinalyze_row=r,
                        execution=execution_result,
                    )

                except Exception as exc:

                    print(
                        f"[TELEGRAM_FORMAT_FALLBACK] "
                        f"{symbol}: {exc}"
                    )

                    msg = build_fallback_message(
                        ev,
                        setup,
                        r,
                        execution_result,
                    )

                try:

                    sent = bool(
                        send_tg(msg)
                    )

                except Exception as exc:

                    sent = False

                    print(
                        f"[TELEGRAM_ERROR] "
                        f"{symbol}: {exc}"
                    )

                if sent:
                    stats["telegram_sent"] += 1

                record_action(
                    {
                        "event_id":
                            event_id,
                        "symbol":
                            symbol,
                        "direction":
                            ev["direction"],
                        "event_type":
                            ev.get(
                                "event_type"
                            ),
                        "telegram_sent":
                            bool(sent),
                        "execution_status":
                            execution_result.get(
                                "status"
                            )
                            if execution_result
                            else None,
                        "setup":
                            setup,
                        "ts":
                            int(
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
        f"[ENGINE] "
        f"trades_this_cycle={trades}"
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
