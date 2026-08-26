from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from event_engine.telegram import (
    send as send_tg,
    format_signal,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)


DATA = Path("data")
DATA.mkdir(
    exist_ok=True
)

EVENTS = DATA / "events.jsonl"
TRADES = DATA / "trades.jsonl"
ACTIONS = DATA / "actions.jsonl"


MAX_CANDIDATES = int(
    os.environ.get(
        "MAX_CANDIDATES",
        "40",
    )
)

MIN_VOL = float(
    os.environ.get(
        "MIN_VOLUME_24H",
        "1000000",
    )
)

MIN_OI = float(
    os.environ.get(
        "MIN_OPEN_INTEREST",
        "500000",
    )
)

EXECUTION_ENABLED = (
    os.environ.get(
        "EXECUTION_ENABLED",
        "false",
    ).lower()
    == "true"
)

REQUIRE_CVD = (
    os.environ.get(
        "REQUIRE_CVD_CONFIRMATION",
        "false",
    ).lower()
    == "true"
)

REQUIRE_TRIGGER = (
    os.environ.get(
        "REQUIRE_15M_TRIGGER",
        "true",
    ).lower()
    == "true"
)

MAX_AGE = int(
    os.environ.get(
        "MAX_EVENT_AGE_MIN",
        "90",
    )
)

MAX_TRADES = int(
    os.environ.get(
        "MAX_TRADES_PER_CYCLE",
        "1",
    )
)

EXECUTION_MODE = os.environ.get(
    "EXECUTION_MODE",
    os.environ.get(
        "BINGX_ENV",
        "vst",
    ),
)


# ============================================================================
# FILE HELPERS
# ============================================================================

def load_ids(
    path: Path,
) -> set[str]:

    if not path.exists():
        return set()

    ids: set[str] = set()

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        try:
            value = json.loads(
                line
            ).get(
                "event_id"
            )

        except Exception:
            continue

        if value:
            ids.add(
                str(value)
            )

    return ids


def append_jsonl(
    path: Path,
    obj: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                obj,
                ensure_ascii=False,
            )
            + "\n"
        )


def emit_event(
    ev: dict,
) -> None:

    append_jsonl(
        EVENTS,
        ev,
    )


def record_trade(
    obj: dict,
) -> None:

    append_jsonl(
        TRADES,
        obj,
    )


def record_action(
    obj: dict,
) -> None:

    append_jsonl(
        ACTIONS,
        obj,
    )


# ============================================================================
# SETUP
# ============================================================================

def build_event_setup(
    ev: dict,
    df_1h: pd.DataFrame,
    entry_price: float,
) -> dict:

    direction = str(
        ev.get(
            "direction",
            "LONG",
        )
    ).upper()

    if direction not in {
        "LONG",
        "SHORT",
    }:
        raise ValueError(
            f"Unsupported direction={direction}"
        )

    entry_price = float(
        entry_price
    )

    if entry_price <= 0:
        raise ValueError(
            f"Invalid entry_price={entry_price}"
        )

    candidates = [
        ev.get(
            "setup"
        ),
        ev.get(
            "event_fact"
        ),
        ev,
    ]

    invalidation = None
    target = None

    for source in candidates:

        if not isinstance(
            source,
            dict,
        ):
            continue

        if invalidation is None:
            invalidation = source.get(
                "invalidation_price"
            )

        if target is None:
            target = source.get(
                "target_price"
            )

    if (
        invalidation is not None
        and target is not None
    ):

        invalidation = float(
            invalidation
        )

        target = float(
            target
        )

        if direction == "LONG":

            valid_geometry = (
                invalidation
                < entry_price
                and target
                > entry_price
            )

        else:

            valid_geometry = (
                invalidation
                > entry_price
                and target
                < entry_price
            )

        if valid_geometry:

            risk_pct = (
                abs(
                    entry_price
                    - invalidation
                )
                / entry_price
                * 100.0
            )

            reward_pct = (
                abs(
                    target
                    - entry_price
                )
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
    # ATR fallback
    # ----------------------------------------------------------------------

    df = df_1h.copy()

    if len(df) < 20:
        raise ValueError(
            "insufficient 1H bars for setup"
        )

    for col in (
        "high",
        "low",
        "close",
    ):

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    prev_close = df[
        "close"
    ].shift(1)

    tr = pd.concat(
        [
            df["high"]
            - df["low"],

            (
                df["high"]
                - prev_close
            ).abs(),

            (
                df["low"]
                - prev_close
            ).abs(),
        ],
        axis=1,
    ).max(
        axis=1
    )

    atr = (
        tr
        .rolling(
            window=14,
            min_periods=14,
        )
        .mean()
        .iloc[-1]
    )

    if (
        pd.isna(atr)
        or float(atr) <= 0
    ):
        raise ValueError(
            "ATR unavailable"
        )

    atr = float(
        atr
    )

    risk_pct = (
        atr
        / entry_price
        * 100.0
    )

    risk_pct = max(
        0.50,
        min(
            risk_pct,
            5.00,
        ),
    )

    if direction == "LONG":

        invalidation = (
            entry_price
            * (
                1.0
                - risk_pct / 100.0
            )
        )

        target = (
            entry_price
            * (
                1.0
                + 2.0
                * risk_pct
                / 100.0
            )
        )

    else:

        invalidation = (
            entry_price
            * (
                1.0
                + risk_pct / 100.0
            )
        )

        target = (
            entry_price
            * (
                1.0
                - 2.0
                * risk_pct
                / 100.0
            )
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
) -> Tuple[
    float,
    List[dict],
]:

    direction = str(
        direction
    ).upper()

    entry = float(
        setup[
            "entry_reference"
        ]
    )

    sl_price = float(
        setup[
            "invalidation_price"
        ]
    )

    final_tp_price = float(
        setup[
            "target_price"
        ]
    )

    if direction == "LONG":

        sl_pct = (
            entry
            - sl_price
        ) / entry * 100.0

        tp_pct = (
            final_tp_price
            - entry
        ) / entry * 100.0

    elif direction == "SHORT":

        sl_pct = (
            sl_price
            - entry
        ) / entry * 100.0

        tp_pct = (
            entry
            - final_tp_price
        ) / entry * 100.0

    else:
        raise ValueError(
            f"Invalid direction={direction}"
        )

    if sl_pct <= 0:
        raise ValueError(
            f"Invalid SL geometry: "
            f"{sl_price}"
        )

    if tp_pct <= 0:
        raise ValueError(
            f"Invalid TP geometry: "
            f"{final_tp_price}"
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

    return (
        sl_pct,
        tp_levels,
    )


# ============================================================================
# PROTECTION HELPERS
# ============================================================================

def _protection_is_ok(
    result: dict,
) -> bool:

    if not isinstance(
        result,
        dict,
    ):
        return False

    return (
        str(
            result.get(
                "status",
                "",
            )
        ).upper()
        == "PROTECTED"
    )


def _protection_summary(
    result: dict,
) -> str:

    if not isinstance(
        result,
        dict,
    ):
        return "UNKNOWN"

    status = result.get(
        "status",
        "UNKNOWN",
    )

    tp_orders = (
        result.get(
            "tp_orders"
        )
        or []
    )

    sl_result = (
        result.get(
            "sl_result"
        )
        or {}
    )

    tp_ok = sum(
        1
        for item in tp_orders
        if str(
            item.get(
                "status",
                "",
            )
        ).lower()
        in {
            "created",
            "already_exists",
        }
    )

    sl_ok = (
        str(
            sl_result.get(
                "status",
                "",
            )
        ).lower()
        in {
            "created",
            "already_exists",
        }
    )

    return (
        f"status={status} "
        f"tp={tp_ok}/{len(tp_orders)} "
        f"sl={'OK' if sl_ok else 'FAIL'}"
    )


# ============================================================================
# EXISTING POSITION RECONCILIATION
# ============================================================================

def reconcile_existing_position(
    symbol: str,
    direction: str,
    setup: dict,
    event_id: str,
) -> dict:

    direction = str(
        direction
    ).upper()

    print(
        f"[EXECUTION_RECONCILE] "
        f"{symbol} "
        f"direction={direction} "
        f"event={event_id}"
    )

    try:

        position = (
            get_position_directional(
                symbol,
                direction,
            )
        )

    except Exception as exc:

        return {
            "status": (
                "RECONCILE_POSITION_ERROR"
            ),
            "mode": EXECUTION_MODE,
            "order_id": None,
            "protection_status": (
                "UNKNOWN"
            ),
            "error": str(
                exc
            ),
        }

    if not isinstance(
        position,
        dict,
    ):

        return {
            "status": (
                "ALREADY_EXECUTED"
            ),
            "mode": EXECUTION_MODE,
            "order_id": None,
            "protection_status": (
                "NOT_FOUND"
            ),
        }

    if position.get(
        "status"
    ) != "found":

        return {
            "status": (
                "ALREADY_EXECUTED"
            ),
            "mode": EXECUTION_MODE,
            "order_id": None,
            "position": position,
            "protection_status": (
                "NOT_FOUND"
            ),
        }

    avg_price = float(
        position.get(
            "avgPrice",
            0,
        )
        or position.get(
            "entryPrice",
            0,
        )
        or 0
    )

    qty = abs(
        float(
            position.get(
                "positionAmt",
                0,
            )
            or 0
        )
    )

    print(
        f"[POSITION_FOUND] "
        f"{symbol} "
        f"direction={direction} "
        f"avgPrice={avg_price} "
        f"qty={qty}"
    )

    try:

        sl_pct, tp_levels = (
            build_tp_levels(
                setup,
                direction,
            )
        )

        protection = (
            ensure_directional_protection(
                symbol=symbol,
                direction=direction,
                avg_price=avg_price,
                qty=qty,
                stop_loss_pct=sl_pct,
                tp_levels=tp_levels,
                trade_id=event_id.replace(
                    "EVT_",
                    "",
                ),
            )
        )

    except Exception as exc:

        print(
            f"[PROTECTION_RECONCILE_ERROR] "
            f"{symbol}: {exc}"
        )

        return {
            "status": (
                "ALREADY_EXECUTED"
            ),
            "mode": EXECUTION_MODE,
            "order_id": None,
            "position": position,
            "protection_status": (
                "ERROR"
            ),
            "protection": {
                "status": (
                    "PROTECTION_EXCEPTION"
                ),
                "error": str(
                    exc
                ),
            },
        }

    protected = _protection_is_ok(
        protection
    )

    print(
        f"[PROTECTION_RECONCILE_RESULT] "
        f"{symbol} "
        f"{direction} "
        f"{_protection_summary(protection)}"
    )

    return {
        "status": (
            "ALREADY_EXECUTED"
        ),
        "mode": EXECUTION_MODE,
        "order_id": None,
        "position": position,
        "protection": protection,
        "protection_status": (
            "PROTECTED"
            if protected
            else "INCOMPLETE"
        ),
    }


# ============================================================================
# NEW POSITION EXECUTION
# ============================================================================

def execute_new_position(
    symbol: str,
    direction: str,
    price: float,
    setup: dict,
    event_id: str,
) -> dict:

    direction = str(
        direction
    ).upper()

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

    # ----------------------------------------------------------------------
    # MARKET ENTRY
    # ----------------------------------------------------------------------

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
            "error": str(
                exc
            ),
        }

    if not isinstance(
        opened,
        dict,
    ):

        return {
            "status": (
                "OPEN_INVALID_RESPONSE"
            ),
            "mode": EXECUTION_MODE,
            "order_id": None,
            "raw": repr(
                opened
            ),
        }

    open_status = str(
        opened.get(
            "status",
            "",
        )
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

    order_id = (
        opened.get(
            "order_id"
        )
        or (
            (
                opened.get(
                    "response"
                )
                or {}
            )
            .get(
                "data",
                {},
            )
            .get(
                "order",
                {},
            )
            .get(
                "orderId"
            )
        )
    )

    print(
        f"[EXECUTION_ORDER_ACCEPTED] "
        f"{symbol} "
        f"direction={direction} "
        f"order={order_id}"
    )

    # ----------------------------------------------------------------------
    # WAIT REAL POSITION
    # ----------------------------------------------------------------------

    try:

        position = (
            wait_for_position_fill_directional(
                symbol,
                direction,
                timeout_sec=30,
                poll_interval=1.0,
            )
        )

    except Exception as exc:

        return {
            "status": (
                "POSITION_WAIT_FAILED"
            ),
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
            "error": str(
                exc
            ),
        }

    if not isinstance(
        position,
        dict,
    ):

        return {
            "status": (
                "POSITION_NOT_CONFIRMED"
            ),
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
        }

    if position.get(
        "status"
    ) != "found":

        return {
            "status": (
                "POSITION_NOT_CONFIRMED"
            ),
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
            "position": position,
        }

    avg_price = float(
        position.get(
            "avgPrice",
            0,
        )
        or position.get(
            "entryPrice",
            0,
        )
        or 0
    )

    actual_qty = abs(
        float(
            position.get(
                "positionAmt",
                0,
            )
            or 0
        )
    )

    if avg_price <= 0:

        return {
            "status": (
                "POSITION_INVALID_AVG_PRICE"
            ),
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "position": position,
        }

    if actual_qty <= 0:

        return {
            "status": (
                "POSITION_INVALID_QTY"
            ),
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "position": position,
        }

    print(
        f"[POSITION_CONFIRMED] "
        f"{symbol} "
        f"direction={direction} "
        f"avgPrice={avg_price} "
        f"qty={actual_qty}"
    )

    # ----------------------------------------------------------------------
    # PROTECTION PLAN
    # ----------------------------------------------------------------------

    try:

        sl_pct, tp_levels = (
            build_tp_levels(
                setup,
                direction,
            )
        )

    except Exception as exc:

        return {
            "status": (
                "PROTECTION_SETUP_INVALID"
            ),
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "position": position,
            "error": str(
                exc
            ),
        }

    print(
        f"[PROTECTION_PLAN] "
        f"{symbol} "
        f"direction={direction} "
        f"sl_pct={sl_pct:.6f} "
        f"tp_levels={tp_levels}"
    )

    # ----------------------------------------------------------------------
    # INSTALL SL + TP
    # ----------------------------------------------------------------------

    try:

        protection = (
            ensure_directional_protection(
                symbol=symbol,
                direction=direction,
                avg_price=avg_price,
                qty=actual_qty,
                stop_loss_pct=sl_pct,
                tp_levels=tp_levels,
                trade_id=trade_id,
            )
        )

    except Exception as exc:

        protection = {
            "status": (
                "PROTECTION_EXCEPTION"
            ),
            "error": str(
                exc
            ),
        }

    if not isinstance(
        protection,
        dict,
    ):

        protection = {
            "status": (
                "PROTECTION_INVALID_RESPONSE"
            ),
            "raw": repr(
                protection
            ),
        }

    protected = _protection_is_ok(
        protection
    )

    print(
        f"[PROTECTION_RESULT] "
        f"{symbol} "
        f"direction={direction} "
        f"{_protection_summary(protection)}"
    )

    return {
        "status": (
            "opened_protected"
            if protected
            else
            "opened_protection_check_required"
        ),
        "mode": EXECUTION_MODE,
        "order_id": order_id,
        "open_result": opened,
        "position": position,
        "protection": protection,
        "protection_status": (
            "PROTECTED"
            if protected
            else "INCOMPLETE"
        ),
        "sl_pct": sl_pct,
        "tp_levels": tp_levels,
    }


# ============================================================================
# TELEGRAM FALLBACK
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
        ev.get(
            "direction",
            "",
        )
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
        f"Event: "
        f"<code>{ev.get('event_type')}</code>\n"
        f"TF: "
        f"<b>{ev.get('timeframe', '1h')}</b>"
        f" + trigger 15m\n"
        f"Price: <code>{price:.8g}</code>\n"
        f"Detected: "
        f"<code>"
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
        f"Mode: "
        f"<code>{EXECUTION_MODE}</code>\n"
        f"Status: "
        f"<code>{execution_result.get('status')}</code>\n"
        f"Protection: "
        f"<code>"
        f"{execution_result.get('protection_status', '—')}"
        f"</code>\n"
        f"Order: "
        f"<code>"
        f"{execution_result.get('order_id') or '—'}"
        f"</code>\n\n"
        f"⚡ Event-driven — "
        f"5×5m lifecycle is NOT used"
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
    # COINALYZE
    # ----------------------------------------------------------------------

    rows = fetch_data()

    stats[
        "coinalyze_rows"
    ] = len(rows)

    print(
        f"[ENGINE] "
        f"Coinalyze rows={len(rows)}"
    )

    # ----------------------------------------------------------------------
    # BINGX CONTRACTS
    # ----------------------------------------------------------------------

    try:
        refresh_contracts()

    except Exception as exc:

        stats[
            "scan_errors"
        ] += 1

        print(
            f"[BINGX] "
            f"contracts refresh error={exc}"
        )

    candidates: List[
        Any
    ] = []

    for row in rows:

        try:

            if (
                row.price is None
                or row.price <= 0
            ):
                continue

            if (
                row.volume24 is None
                or row.volume24 < MIN_VOL
            ):
                continue

            if (
                row.oi is None
                or row.oi < MIN_OI
            ):
                continue

            contract = get_contract(
                row.symbol
            )

            if not contract:

                stats[
                    "bingx_unmapped"
                ] += 1

                print(
                    f"[MAP] NO_BINGX "
                    f"coinalyze={row.symbol} "
                    f"name="
                    f"{getattr(row, 'name', '')}"
                )

                continue

            stats[
                "bingx_mapped"
            ] += 1

            print(
                f"[MAP] OK "
                f"coinalyze={row.symbol} "
                f"bingx="
                f"{contract.get('symbol', row.symbol)} "
                f"displayName="
                f"{contract.get('displayName', '')}"
            )

            candidates.append(
                row
            )

        except Exception as exc:

            stats[
                "scan_errors"
            ] += 1

            print(
                f"[CANDIDATE_ERROR] "
                f"{getattr(row, 'symbol', '?')}: "
                f"{exc}"
            )

    stats[
        "candidates_before_limit"
    ] = len(
        candidates
    )

    candidates = candidates[
        :MAX_CANDIDATES
    ]

    stats[
        "candidates"
    ] = len(
        candidates
    )

    print(
        f"[ENGINE] "
        f"Coinalyze candidates="
        f"{len(candidates)} "
        f"execution="
        f"{EXECUTION_ENABLED} "
        f"env="
        f"{EXECUTION_MODE}"
    )

    seen_events = load_ids(
        EVENTS
    )

    executed_event_ids = load_ids(
        TRADES
    )

    trades_this_cycle = 0

    # =========================================================================
    # SYMBOL LOOP
    # =========================================================================

    for row in candidates:

        symbol = row.symbol

        try:

            # ----------------------------------------------------------------
            # KLINES
            # ----------------------------------------------------------------

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
                    f"{symbol} "
                    f"1H bars={len(k1)} < 60"
                )

                continue

            if len(k15) < 10:

                print(
                    f"[DATA_REJECT] "
                    f"{symbol} "
                    f"15m bars={len(k15)} < 10"
                )

                continue

            stats[
                "klines_1h_ok"
            ] += 1

            stats[
                "klines_15m_ok"
            ] += 1

            d1 = add_cvd(
                pd.DataFrame(
                    k1
                )
            )

            d15 = pd.DataFrame(
                k15
            )

            # ----------------------------------------------------------------
            # EVENTS
            # ----------------------------------------------------------------

            divergence_events = (
                detect_divergences(
                    d1,
                    symbol,
                    "1h",
                )
            )

            squeeze_events = (
                detect_squeeze_release(
                    d1,
                    symbol,
                    "1h",
                )
            )

            rsi_events = [
                event
                for event
                in divergence_events
                if "_RSI"
                in event.get(
                    "event_type",
                    "",
                )
            ]

            cvd_events = [
                event
                for event
                in divergence_events
                if "BINGX_CVD"
                in event.get(
                    "event_type",
                    "",
                )
            ]

            stats[
                "rsi_events"
            ] += len(
                rsi_events
            )

            stats[
                "cvd_events"
            ] += len(
                cvd_events
            )

            stats[
                "squeeze_events"
            ] += len(
                squeeze_events
            )

            stats[
                "events_total"
            ] += (
                len(
                    divergence_events
                )
                + len(
                    squeeze_events
                )
            )

            print(
                f"[EVENT_SCAN] "
                f"{symbol} "
                f"RSI={len(rsi_events)} "
                f"CVD={len(cvd_events)} "
                f"SQUEEZE={len(squeeze_events)}"
            )

            all_events = (
                divergence_events
                + squeeze_events
            )

            # ----------------------------------------------------------------
            # EVENT LOOP
            # ----------------------------------------------------------------

            for ev in all_events:

                event_id = ev.get(
                    "event_id"
                )

                if not event_id:

                    print(
                        f"[EVENT_REJECT] "
                        f"{symbol} "
                        f"event without event_id"
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
                        f"{symbol} "
                        f"direction={direction}"
                    )

                    continue

                # ------------------------------------------------------------
                # EVENT AGE
                # ------------------------------------------------------------

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
                    d1[
                        "close_time"
                    ].iloc[-1]
                )

                age = (
                    latest_close
                    - detected_at
                ) / 60000.0

                if (
                    age < 0
                    or age > MAX_AGE
                ):

                    print(
                        f"[EVENT_REJECT_AGE] "
                        f"{symbol} "
                        f"type="
                        f"{ev.get('event_type')} "
                        f"age={age:.1f}m "
                        f"max={MAX_AGE}m"
                    )

                    continue

                stats[
                    "events_recent"
                ] += 1

                # ------------------------------------------------------------
                # PERSIST EVENT
                # ------------------------------------------------------------

                already_seen = (
                    event_id
                    in seen_events
                )

                if already_seen:

                    stats[
                        "events_duplicate"
                    ] += 1

                else:

                    emit_event(
                        ev
                    )

                    seen_events.add(
                        event_id
                    )

                # ------------------------------------------------------------
                # CVD CONFIRMATION
                # ------------------------------------------------------------

                if (
                    REQUIRE_CVD
                    and "_RSI"
                    in ev.get(
                        "event_type",
                        "",
                    )
                ):

                    timestamps = ev.get(
                        "timestamps",
                        {},
                    )

                    pivot_1 = timestamps.get(
                        "pivot_1_ts"
                    )

                    pivot_2 = timestamps.get(
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
                            f"type="
                            f"{ev.get('event_type')}"
                        )

                        continue

                # ------------------------------------------------------------
                # 15M TRIGGER
                # ------------------------------------------------------------

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
                        f"event="
                        f"{ev.get('event_type')} "
                        f"reason="
                        f"15m_trigger_failed"
                    )

                    continue

                stats[
                    "trigger_pass"
                ] += 1

                # ------------------------------------------------------------
                # ENTRY PRICE
                # ------------------------------------------------------------

                event_fact = ev.get(
                    "event_fact",
                    {},
                )

                price_raw = (
                    event_fact.get(
                        "detection_close_price"
                    )
                )

                if price_raw is None:

                    price_raw = (
                        event_fact.get(
                            "close"
                        )
                        or getattr(
                            row,
                            "price",
                            None,
                        )
                    )

                if price_raw is None:

                    stats[
                        "setup_rejected"
                    ] += 1

                    print(
                        f"[SETUP_REJECT] "
                        f"{symbol} "
                        f"event={event_id} "
                        f"reason="
                        f"no_entry_price"
                    )

                    continue

                price = float(
                    price_raw
                )

                # ------------------------------------------------------------
                # SETUP
                # ------------------------------------------------------------

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

                stats[
                    "setups"
                ] += 1

                print(
                    f"[SETUP_OK] "
                    f"{symbol} "
                    f"direction={direction} "
                    f"entry="
                    f"{setup['entry_reference']:.8g} "
                    f"SL="
                    f"{setup['invalidation_price']:.8g} "
                    f"TP="
                    f"{setup['target_price']:.8g} "
                    f"RR="
                    f"{setup['rr']:.2f}"
                )

                execution_result: Dict[
                    str,
                    Any,
                ]

                # ============================================================
                # EVENT ALREADY EXECUTED
                # ============================================================

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
                            "status": (
                                "ALREADY_EXECUTED"
                            ),
                            "mode": EXECUTION_MODE,
                            "order_id": None,
                            "protection_status": (
                                "NOT_CHECKED"
                            ),
                        }

                    print(
                        f"[EXECUTION_RESULT] "
                        f"{symbol} "
                        f"event={event_id} "
                        f"status="
                        f"{execution_result.get('status')}"
                    )

                # ============================================================
                # NEW EXECUTION
                # ============================================================

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
                            "result": (
                                execution_result
                            ),
                            "setup": setup,
                        }
                    )

                    status = str(
                        execution_result.get(
                            "status",
                            "",
                        )
                    )

                    protection_status = str(
                        execution_result.get(
                            "protection_status",
                            "",
                        )
                    )

                    # --------------------------------------------------------
                    # CRITICAL:
                    # event is considered executed ONLY if protected.
                    # --------------------------------------------------------

                    if (
                        status
                        == "opened_protected"
                        and
                        protection_status
                        == "PROTECTED"
                    ):

                        executed_event_ids.add(
                            event_id
                        )

                        trades_this_cycle += 1
                        stats[
                            "trades"
                        ] += 1

                        print(
                            f"[EXECUTION_OPENED_PROTECTED] "
                            f"{symbol} "
                            f"direction={direction} "
                            f"order="
                            f"{execution_result.get('order_id')}"
                        )

                    else:

                        print(
                            f"[EXECUTION_NOT_PROTECTED] "
                            f"{symbol} "
                            f"direction={direction} "
                            f"status={status} "
                            f"protection="
                            f"{protection_status}"
                        )

                # ============================================================
                # EXECUTION DISABLED
                # ============================================================

                elif not EXECUTION_ENABLED:

                    execution_result = {
                        "status": "DISABLED",
                        "mode": EXECUTION_MODE,
                        "order_id": None,
                        "protection_status": (
                            "NOT_ATTEMPTED"
                        ),
                    }

                # ============================================================
                # TRADE LIMIT
                # ============================================================

                else:

                    execution_result = {
                        "status": (
                            "TRADE_LIMIT_REACHED"
                        ),
                        "mode": EXECUTION_MODE,
                        "order_id": None,
                        "protection_status": (
                            "NOT_ATTEMPTED"
                        ),
                    }

                # ------------------------------------------------------------
                # TELEGRAM
                # ------------------------------------------------------------

                label = (
                    "🚨 LONG SIGNAL"
                    if direction == "LONG"
                    else "🔻 SHORT SIGNAL"
                )

                try:

                    message = format_signal(
                        ev,
                        setup=setup,
                        coinalyze_row=row,
                        execution=execution_result,
                    )

                    if not (
                        message.startswith(
                            "🚨 LONG SIGNAL"
                        )
                        or
                        message.startswith(
                            "🔻 SHORT SIGNAL"
                        )
                    ):

                        message = (
                            label
                            + "\n\n"
                            + message
                        )

                except Exception as exc:

                    message = (
                        build_fallback_signal_message(
                            ev=ev,
                            symbol=symbol,
                            name=(
                                getattr(
                                    row,
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

                    sent = send_tg(
                        message
                    )

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

            stats[
                "scan_errors"
            ] += 1

            print(
                f"[SCAN_ERROR] "
                f"{symbol}: "
                f"{exc}"
            )

    print(
        f"[ENGINE] "
        f"trades_this_cycle="
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
