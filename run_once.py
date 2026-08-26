from __future__ import annotations

import inspect
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

POSITION_MODE = os.environ.get(
    "BINGX_POSITION_MODE",
    "HEDGE",
)


# ============================================================================
# FILE HELPERS
# ============================================================================

def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids: set[str] = set()

    for line in path.read_text(
        encoding="utf-8",
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
        ev.get("setup"),
        ev.get("event_fact"),
        ev,
    ]

    invalidation = None
    target = None

    for src in candidates:

        if not isinstance(
            src,
            dict,
        ):
            continue

        if invalidation is None:
            invalidation = src.get(
                "invalidation_price"
            )

        if target is None:
            target = src.get(
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

            if (
                rr is not None
                and rr > 0
            ):

                return {
                    "entry_reference": entry_price,
                    "invalidation_price": invalidation,
                    "target_price": target,
                    "rr": rr,
                    "trigger_ok": True,
                }

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
            df["high"] - df["low"],
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
) -> Tuple[float, List[dict]]:

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

    return (
        sl_pct,
        tp_levels,
    )


# ============================================================================
# BINGX PROTECTION COMPATIBILITY
# ============================================================================

def _get_position_avg_price(
    position: dict,
) -> float:

    return float(
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


def _get_position_qty(
    position: dict,
) -> float:

    return abs(
        float(
            position.get(
                "positionAmt",
                0,
            )
            or 0
        )
    )


def install_protection(
    symbol: str,
    direction: str,
    position: dict,
    setup: dict,
    sl_pct: float,
    tp_levels: list,
    trade_id: str,
) -> dict:

    avg_price = _get_position_avg_price(
        position
    )

    qty = _get_position_qty(
        position
    )

    if avg_price <= 0:
        return {
            "status": "PROTECTION_INVALID_POSITION",
            "error": (
                f"invalid avgPrice={avg_price}"
            ),
        }

    if qty <= 0:
        return {
            "status": "PROTECTION_INVALID_POSITION",
            "error": (
                f"invalid positionAmt={qty}"
            ),
        }

    try:
        signature = inspect.signature(
            ensure_directional_protection
        )

        parameters = set(
            signature.parameters
        )

    except Exception:
        parameters = set()

    print(
        f"[PROTECTION_CALL] "
        f"symbol={symbol} "
        f"direction={direction} "
        f"avg_price={avg_price} "
        f"qty={qty} "
        f"sl_pct={sl_pct} "
        f"tp_levels={tp_levels!r} "
        f"trade_id={trade_id} "
        f"signature_params={sorted(parameters)!r}"
    )

    if {
        "avg_price",
        "qty",
        "stop_loss_pct",
        "tp_levels",
    }.issubset(
        parameters
    ):

        try:

            result = (
                ensure_directional_protection(
                    symbol=symbol,
                    direction=direction,
                    avg_price=avg_price,
                    qty=qty,
                    stop_loss_pct=sl_pct,
                    tp_levels=tp_levels,
                    trade_id=trade_id,
                )
            )

            print(
                f"[PROTECTION_RAW_RESULT] "
                f"symbol={symbol} "
                f"direction={direction} "
                f"result={result!r}"
            )

            return result

        except Exception as exc:

            print(
                f"[PROTECTION_CALL_EXCEPTION] "
                f"symbol={symbol} "
                f"direction={direction} "
                f"exception="
                f"{type(exc).__name__}: {exc}"
            )

            return {
                "status": "PROTECTION_EXCEPTION",
                "error": str(exc),
                "exception_type": (
                    type(exc).__name__
                ),
            }

    if {
        "position",
        "setup",
        "sl_pct",
        "tp_levels",
    }.issubset(
        parameters
    ):

        try:

            result = (
                ensure_directional_protection(
                    symbol=symbol,
                    direction=direction,
                    position=position,
                    setup=setup,
                    sl_pct=sl_pct,
                    tp_levels=tp_levels,
                )
            )

            print(
                f"[PROTECTION_RAW_RESULT] "
                f"symbol={symbol} "
                f"direction={direction} "
                f"result={result!r}"
            )

            return result

        except Exception as exc:

            print(
                f"[PROTECTION_CALL_EXCEPTION] "
                f"symbol={symbol} "
                f"direction={direction} "
                f"exception="
                f"{type(exc).__name__}: {exc}"
            )

            return {
                "status": "PROTECTION_EXCEPTION",
                "error": str(exc),
                "exception_type": (
                    type(exc).__name__
                ),
            }

    try:

        result = (
            ensure_directional_protection(
                symbol,
                direction,
                avg_price,
                qty,
                sl_pct,
                tp_levels,
                trade_id,
            )
        )

        print(
            f"[PROTECTION_RAW_RESULT] "
            f"symbol={symbol} "
            f"direction={direction} "
            f"result={result!r}"
        )

        return result

    except TypeError:

        try:

            result = (
                ensure_directional_protection(
                    symbol,
                    direction,
                    position,
                    setup,
                    sl_pct,
                    tp_levels,
                )
            )

            print(
                f"[PROTECTION_RAW_RESULT] "
                f"symbol={symbol} "
                f"direction={direction} "
                f"result={result!r}"
            )

            return result

        except Exception as exc:

            print(
                f"[PROTECTION_CALL_EXCEPTION] "
                f"symbol={symbol} "
                f"direction={direction} "
                f"exception="
                f"{type(exc).__name__}: {exc}"
            )

            return {
                "status": "PROTECTION_EXCEPTION",
                "error": str(exc),
                "exception_type": (
                    type(exc).__name__
                ),
            }

    except Exception as exc:

        print(
            f"[PROTECTION_CALL_EXCEPTION] "
            f"symbol={symbol} "
            f"direction={direction} "
            f"exception="
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "status": "PROTECTION_EXCEPTION",
            "error": str(exc),
            "exception_type": (
                type(exc).__name__
            ),
        }


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
                symbol=symbol,
                direction=direction,
            )
        )

    except TypeError:

        try:

            position = (
                get_position_directional(
                    symbol,
                    direction,
                )
            )

        except Exception as exc:

            print(
                f"[RECONCILE_POSITION_EXCEPTION] "
                f"{symbol} "
                f"direction={direction} "
                f"exception="
                f"{type(exc).__name__}: {exc}"
            )

            return {
                "status": "ALREADY_EXECUTED_POSITION_ERROR",
                "mode": EXECUTION_MODE,
                "order_id": None,
                "protection_status": (
                    "NOT_VERIFIED"
                ),
                "error": str(exc),
            }

    except Exception as exc:

        print(
            f"[RECONCILE_POSITION_EXCEPTION] "
            f"{symbol} "
            f"direction={direction} "
            f"exception="
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "status": "ALREADY_EXECUTED_POSITION_ERROR",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "protection_status": (
                "NOT_VERIFIED"
            ),
            "error": str(exc),
        }

    print(
        f"[RECONCILE_POSITION_RESULT] "
        f"{symbol} "
        f"direction={direction} "
        f"position={position!r}"
    )

    if not isinstance(
        position,
        dict,
    ):

        return {
            "status": (
                "ALREADY_EXECUTED_POSITION_NOT_FOUND"
            ),
            "mode": EXECUTION_MODE,
            "order_id": None,
            "protection_status": (
                "NOT_VERIFIED"
            ),
            "position": position,
        }

    position_status = str(
        position.get(
            "status",
            "",
        )
    ).lower()

    if position_status != "found":

        print(
            f"[RECONCILE_POSITION_NOT_FOUND] "
            f"{symbol} "
            f"direction={direction} "
            f"status={position.get('status')} "
            f"position={position!r}"
        )

        return {
            "status": (
                "ALREADY_EXECUTED_POSITION_NOT_FOUND"
            ),
            "mode": EXECUTION_MODE,
            "order_id": None,
            "protection_status": (
                "NOT_VERIFIED"
            ),
            "position": position,
        }

    avg_price = _get_position_avg_price(
        position
    )

    qty = _get_position_qty(
        position
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

    except Exception as exc:

        print(
            f"[RECONCILE_SETUP_INVALID] "
            f"{symbol} "
            f"direction={direction} "
            f"exception="
            f"{type(exc).__name__}: {exc} "
            f"setup={setup!r}"
        )

        return {
            "status": "ALREADY_EXECUTED",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "position": position,
            "protection_status": (
                "SETUP_INVALID"
            ),
            "error": str(exc),
        }

    print(
        f"[RECONCILE_PROTECTION_PLAN] "
        f"{symbol} "
        f"direction={direction} "
        f"sl_pct={sl_pct:.6f} "
        f"tp_levels={tp_levels!r} "
        f"avg_price={avg_price} "
        f"qty={qty}"
    )

    trade_id = event_id.replace(
        "EVT_",
        "",
    )

    protection_result = (
        install_protection(
            symbol=symbol,
            direction=direction,
            position=position,
            setup=setup,
            sl_pct=sl_pct,
            tp_levels=tp_levels,
            trade_id=trade_id,
        )
    )

    if not isinstance(
        protection_result,
        dict,
    ):

        protection_result = {
            "status": "UNKNOWN",
            "raw": repr(
                protection_result
            ),
        }

    protection_status = str(
        protection_result.get(
            "status",
            "",
        )
    ).upper()

    print(
        f"[PROTECTION_RECONCILE_RESULT] "
        f"{symbol} "
        f"{direction} "
        f"status={protection_status} "
        f"result={protection_result!r}"
    )

    return {
        "status": "ALREADY_EXECUTED",
        "mode": EXECUTION_MODE,
        "order_id": None,
        "position": position,
        "protection": protection_result,
        "protection_status": (
            protection_result.get(
                "status"
            )
        ),
    }


# ============================================================================
# NEW EXECUTION
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

    try:

        opened = open_market(
            symbol,
            direction,
            price,
            trade_id,
        )

    except Exception as exc:

        print(
            f"[OPEN_EXCEPTION] "
            f"symbol={symbol} "
            f"direction={direction} "
            f"event={event_id} "
            f"exception="
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "status": "OPEN_EXCEPTION",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "error": str(exc),
            "exception_type": (
                type(exc).__name__
            ),
        }

    print(
        f"[OPEN_RAW_RESULT] "
        f"symbol={symbol} "
        f"direction={direction} "
        f"event={event_id} "
        f"result={opened!r}"
    )

    if not isinstance(
        opened,
        dict,
    ):

        print(
            f"[OPEN_INVALID_RESPONSE] "
            f"symbol={symbol} "
            f"direction={direction} "
            f"event={event_id} "
            f"type={type(opened).__name__} "
            f"raw={opened!r}"
        )

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

        print(
            f"[OPEN_FAILED] "
            f"symbol={symbol} "
            f"direction={direction} "
            f"event={event_id} "
            f"status={opened.get('status')} "
            f"error={opened.get('error')} "
            f"msg={opened.get('msg')} "
            f"code={opened.get('code')} "
            f"symbol_bingx={opened.get('symbol')} "
            f"order_id={opened.get('order_id')} "
            f"full={opened!r}"
        )

        return {
            "status": "OPEN_FAILED",
            "mode": EXECUTION_MODE,
            "order_id": opened.get(
                "order_id"
            ),
            "open_result": opened,
            "error": (
                opened.get(
                    "error"
                )
                or opened.get(
                    "msg"
                )
                or "unknown_open_error"
            ),
            "bingx_code": opened.get(
                "code"
            ),
        }

    order_id = opened.get(
        "order_id"
    )

    print(
        f"[EXECUTION_ORDER_ACCEPTED] "
        f"{symbol} "
        f"direction={direction} "
        f"order={order_id} "
        f"raw={opened!r}"
    )

    try:

        wait_signature = inspect.signature(
            wait_for_position_fill_directional
        )

        wait_params = set(
            wait_signature.parameters
        )

    except Exception:

        wait_params = set()

    print(
        f"[POSITION_WAIT_START] "
        f"{symbol} "
        f"direction={direction} "
        f"order={order_id} "
        f"wait_params={sorted(wait_params)!r}"
    )

    try:

        if "poll_interval" in wait_params:

            position = (
                wait_for_position_fill_directional(
                    symbol=symbol,
                    direction=direction,
                    timeout_sec=15,
                    poll_interval=0.5,
                )
            )

        elif "poll_sec" in wait_params:

            position = (
                wait_for_position_fill_directional(
                    symbol=symbol,
                    direction=direction,
                    timeout_sec=15,
                    poll_sec=0.5,
                )
            )

        else:

            position = (
                wait_for_position_fill_directional(
                    symbol,
                    direction,
                    15,
                    0.5,
                )
            )

    except Exception as exc:

        print(
            f"[POSITION_WAIT_EXCEPTION] "
            f"symbol={symbol} "
            f"direction={direction} "
            f"order={order_id} "
            f"exception="
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "status": (
                "POSITION_WAIT_FAILED"
            ),
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
            "error": str(exc),
        }

    print(
        f"[POSITION_WAIT_RESULT] "
        f"symbol={symbol} "
        f"direction={direction} "
        f"order={order_id} "
        f"position={position!r}"
    )

    if not isinstance(
        position,
        dict,
    ):

        print(
            f"[POSITION_INVALID_RESPONSE] "
            f"symbol={symbol} "
            f"direction={direction} "
            f"order={order_id} "
            f"raw={position!r}"
        )

        return {
            "status": (
                "POSITION_NOT_CONFIRMED"
            ),
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
            "position_raw": repr(
                position
            ),
        }

    position_status = str(
        position.get(
            "status",
            "",
        )
    ).lower()

    if position_status != "found":

        print(
            f"[POSITION_NOT_CONFIRMED] "
            f"symbol={symbol} "
            f"direction={direction} "
            f"order={order_id} "
            f"status={position.get('status')} "
            f"position={position!r}"
        )

        return {
            "status": (
                "POSITION_NOT_CONFIRMED"
            ),
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
            "position": position,
        }

    avg_price = _get_position_avg_price(
        position
    )

    qty = _get_position_qty(
        position
    )

    print(
        f"[POSITION_CONFIRMED] "
        f"{symbol} "
        f"direction={direction} "
        f"avgPrice={avg_price} "
        f"qty={qty} "
        f"order={order_id}"
    )

    try:

        sl_pct, tp_levels = (
            build_tp_levels(
                setup,
                direction,
            )
        )

    except Exception as exc:

        print(
            f"[PROTECTION_SETUP_INVALID] "
            f"symbol={symbol} "
            f"direction={direction} "
            f"order={order_id} "
            f"exception="
            f"{type(exc).__name__}: {exc} "
            f"setup={setup!r}"
        )

        return {
            "status": (
                "PROTECTION_SETUP_INVALID"
            ),
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "position": position,
            "open_result": opened,
            "error": str(exc),
        }

    print(
        f"[PROTECTION_PLAN] "
        f"{symbol} "
        f"direction={direction} "
        f"sl_pct={sl_pct:.6f} "
        f"tp_levels={tp_levels!r} "
        f"avg_price={avg_price} "
        f"qty={qty}"
    )

    protection = install_protection(
        symbol=symbol,
        direction=direction,
        position=position,
        setup=setup,
        sl_pct=sl_pct,
        tp_levels=tp_levels,
        trade_id=trade_id,
    )

    if not isinstance(
        protection,
        dict,
    ):

        protection = {
            "status": "UNKNOWN",
            "raw": repr(
                protection
            ),
        }

    protection_status = str(
        protection.get(
            "status",
            "",
        )
    ).lower()

    print(
        f"[PROTECTION_RESULT] "
        f"{symbol} "
        f"direction={direction} "
        f"order={order_id} "
        f"status={protection.get('status')} "
        f"result={protection!r}"
    )

    if protection_status in {
        "ok",
        "protected",
        "created",
        "reconciled",
        "success",
        "ready",
    }:

        final_status = (
            "opened_protected"
        )

    else:

        final_status = (
            "opened_protection_check_required"
        )

    print(
        f"[EXECUTION_FINAL] "
        f"{symbol} "
        f"direction={direction} "
        f"event={event_id} "
        f"order={order_id} "
        f"status={final_status} "
        f"position_confirmed=true "
        f"protection_status="
        f"{protection.get('status')}"
    )

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
        f"<b>{ev.get('timeframe', '1h')}</b> "
        f"+ trigger 15m\n"
        f"Price: "
        f"<code>{price:.8g}</code>\n"
        f"Detected: "
        f"<code>"
        f"{ev.get('timestamps', {}).get('detected_at_ts')}"
        f"</code>\n\n"
        f"<b>SETUP</b>\n"
        f"Entry: "
        f"<code>{price:.8g}</code>\n"
        f"SL: "
        f"<code>"
        f"{sl:.8g}"
        f"</code>\n"
        f"TP: "
        f"<code>"
        f"{tp:.8g}"
        f"</code>\n"
        f"R:R: "
        f"<code>"
        f"{rr:.2f}"
        f"</code>\n\n"
        f"<b>EXECUTION</b>\n"
        f"Mode: "
        f"<code>{EXECUTION_MODE}</code>\n"
        f"Status: "
        f"<code>"
        f"{execution_result.get('status')}"
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
        "telegram_suppressed_duplicate": 0,
        "scan_errors": 0,
    }

    # =========================================================================
    # COINALYZE
    # =========================================================================

    rows = fetch_data()

    stats[
        "coinalyze_rows"
    ] = len(rows)

    print(
        f"[ENGINE] "
        f"Coinalyze rows={len(rows)}"
    )

    # =========================================================================
    # BINGX CONTRACTS
    # =========================================================================

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

                stats[
                    "bingx_unmapped"
                ] += 1

                print(
                    f"[MAP] NO_BINGX "
                    f"coinalyze={r.symbol} "
                    f"name="
                    f"{getattr(r, 'name', '')}"
                )

                continue

            stats[
                "bingx_mapped"
            ] += 1

            print(
                f"[MAP] OK "
                f"coinalyze={r.symbol} "
                f"bingx="
                f"{contract.get('symbol', r.symbol)} "
                f"displayName="
                f"{contract.get('displayName', '')}"
            )

            candidates.append(
                r
            )

        except Exception as exc:

            stats[
                "scan_errors"
            ] += 1

            print(
                f"[CANDIDATE_ERROR] "
                f"{getattr(r, 'symbol', '?')}: "
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
        f"execution={EXECUTION_ENABLED} "
        f"env={EXECUTION_MODE}"
    )

    seen_events = load_ids(
        EVENTS
    )

    executed_event_ids = load_ids(
        TRADES
    )

    telegram_sent_event_ids = load_ids(
        ACTIONS
    )

    print(
        f"[TELEGRAM_STATE] "
        f"sent_event_ids="
        f"{len(telegram_sent_event_ids)}"
    )

    trades_this_cycle = 0

    # =========================================================================
    # SYMBOL LOOP
    # =========================================================================

    for r in candidates:

        symbol = r.symbol

        try:

            # =================================================================
            # KLINES
            # =================================================================

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
                    f"1H bars={len(k1)} "
                    f"< 60"
                )

                continue

            if len(k15) < 10:

                print(
                    f"[DATA_REJECT] "
                    f"{symbol} "
                    f"15m bars={len(k15)} "
                    f"< 10"
                )

                continue

            stats[
                "klines_1h_ok"
            ] += 1

            stats[
                "klines_15m_ok"
            ] += 1

            # =================================================================
            # DATAFRAMES
            # =================================================================

            d1 = add_cvd(
                pd.DataFrame(k1)
            )

            d15 = pd.DataFrame(
                k15
            )

            # =================================================================
            # EVENT DETECTION
            # =================================================================

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
                e
                for e
                in divergence_events
                if "_RSI"
                in e.get(
                    "event_type",
                    "",
                )
            ]

            cvd_events = [
                e
                for e
                in divergence_events
                if "BINGX_CVD"
                in e.get(
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
                f"RSI="
                f"{len(rsi_events)} "
                f"CVD="
                f"{len(cvd_events)} "
                f"SQUEEZE="
                f"{len(squeeze_events)}"
            )

            all_events = (
                divergence_events
                + squeeze_events
            )

            # =================================================================
            # EVENTS
            # =================================================================

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

                # =============================================================
                # AGE
                # =============================================================

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

                # =============================================================
                # EVENT PERSISTENCE
                # =============================================================

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

                # =============================================================
                # CVD
                # =============================================================

                if (
                    REQUIRE_CVD
                    and "_RSI"
                    in ev.get(
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
                            f"type="
                            f"{ev.get('event_type')}"
                        )

                        continue

                # =============================================================
                # 15M TRIGGER
                # =============================================================

                trigger = (
                    build_15m_trigger(
                        d15,
                        direction,
                    )
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

                # =============================================================
                # ENTRY PRICE
                # =============================================================

                fact = ev.get(
                    "event_fact",
                    {},
                )

                price_raw = (
                    fact.get(
                        "detection_close_price"
                    )
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
                        f"reason="
                        f"no_entry_price"
                    )

                    stats[
                        "setup_rejected"
                    ] += 1

                    continue

                price = float(
                    price_raw
                )

                # =============================================================
                # SETUP
                # =============================================================

                try:

                    setup = (
                        build_event_setup(
                            ev=ev,
                            df_1h=d1,
                            entry_price=price,
                        )
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

                # =============================================================
                # ALREADY EXECUTED
                # =============================================================

                if (
                    event_id
                    in executed_event_ids
                ):

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
                            "status":
                                "ALREADY_EXECUTED",
                            "mode":
                                EXECUTION_MODE,
                            "order_id":
                                None,
                            "protection_status":
                                (
                                    "NOT_CHECKED_"
                                    "EXECUTION_DISABLED"
                                ),
                        }

                    print(
                        f"[EXECUTION_RESULT] "
                        f"{symbol} "
                        f"event={event_id} "
                        f"status="
                        f"{execution_result.get('status')} "
                        f"protection="
                        f"{execution_result.get('protection_status')}"
                    )

                # =============================================================
                # NEW EXECUTION
                # =============================================================

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
                            "event_id":
                                event_id,
                            "symbol":
                                symbol,
                            "direction":
                                direction,
                            "price":
                                price,
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
                            "setup":
                                setup,
                        }
                    )

                    status = str(
                        execution_result.get(
                            "status",
                            "",
                        )
                    )

                    execution_confirmed = (
                        status
                        in {
                            "opened_protected",
                            "opened_protection_check_required",
                            "opened",
                        }
                    )

                    if execution_confirmed:

                        executed_event_ids.add(
                            event_id
                        )

                        print(
                            f"[EVENT_MARKED_EXECUTED] "
                            f"symbol={symbol} "
                            f"event={event_id} "
                            f"status={status}"
                        )

                    else:

                        print(
                            f"[EVENT_NOT_MARKED_EXECUTED] "
                            f"symbol={symbol} "
                            f"event={event_id} "
                            f"status={status}"
                        )

                    if status == (
                        "opened_protected"
                    ):

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

                    elif status == (
                        "opened_protection_check_required"
                    ):

                        trades_this_cycle += 1

                        stats[
                            "trades"
                        ] += 1

                        print(
                            f"[EXECUTION_OPENED_"
                            f"PROTECTION_CHECK_REQUIRED] "
                            f"{symbol} "
                            f"direction={direction} "
                            f"order="
                            f"{execution_result.get('order_id')}"
                        )

                    elif status == (
                        "OPEN_FAILED"
                    ):

                        print(
                            f"[EXECUTION_OPEN_FAILED] "
                            f"{symbol} "
                            f"direction={direction} "
                            f"event={event_id} "
                            f"error="
                            f"{execution_result.get('error')} "
                            f"code="
                            f"{execution_result.get('bingx_code')} "
                            f"open_result="
                            f"{execution_result.get('open_result')!r}"
                        )

                    else:

                        print(
                            f"[EXECUTION_FAILED] "
                            f"{symbol} "
                            f"direction={direction} "
                            f"event={event_id} "
                            f"status={status} "
                            f"error="
                            f"{execution_result.get('error')} "
                            f"result="
                            f"{execution_result!r}"
                        )

                # =============================================================
                # EXECUTION DISABLED
                # =============================================================

                elif not EXECUTION_ENABLED:

                    execution_result = {
                        "status":
                            "DISABLED",
                        "mode":
                            EXECUTION_MODE,
                        "order_id":
                            None,
                        "protection_status":
                            "NOT_ATTEMPTED",
                    }

                # =============================================================
                # TRADE LIMIT
                # =============================================================

                else:

                    execution_result = {
                        "status":
                            "TRADE_LIMIT_REACHED",
                        "mode":
                            EXECUTION_MODE,
                        "order_id":
                            None,
                    }

                # =============================================================
                # TELEGRAM
                # =============================================================

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
                        f"{symbol}: "
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                # =============================================================
                # TELEGRAM IDEMPOTENCY
                # =============================================================

                telegram_already_sent = (
                    event_id
                    in telegram_sent_event_ids
                )

                if telegram_already_sent:

                    sent = False

                    stats[
                        "telegram_suppressed_duplicate"
                    ] += 1

                    print(
                        f"[TELEGRAM_SUPPRESS_DUPLICATE] "
                        f"symbol={symbol} "
                        f"event={event_id} "
                        f"direction={direction} "
                        f"reason=event_id_already_sent"
                    )

                else:

                    stats[
                        "telegram_attempts"
                    ] += 1

                    print(
                        f"[TELEGRAM_ATTEMPT] "
                        f"symbol={symbol} "
                        f"event={event_id} "
                        f"direction={direction} "
                        f"execution_status="
                        f"{execution_result.get('status')} "
                        f"protection_status="
                        f"{execution_result.get('protection_status')}"
                    )

                    try:

                        sent = bool(
                            send_tg(
                                msg
                            )
                        )

                    except Exception as exc:

                        sent = False

                        print(
                            f"[TELEGRAM_ERROR] "
                            f"symbol={symbol} "
                            f"event={event_id} "
                            f"exception="
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )

                    if sent:

                        stats[
                            "telegram_sent"
                        ] += 1

                        telegram_sent_event_ids.add(
                            event_id
                        )

                        print(
                            f"[TELEGRAM_SENT] "
                            f"symbol={symbol} "
                            f"event={event_id}"
                        )

                    else:

                        print(
                            f"[TELEGRAM_NOT_SENT] "
                            f"symbol={symbol} "
                            f"event={event_id}"
                        )

                record_action(
                    {
                        "event_id":
                            event_id,
                        "symbol":
                            symbol,
                        "direction":
                            direction,
                        "event_type":
                            ev.get(
                                "event_type"
                            ),
                        "telegram_sent":
                            bool(
                                sent
                            ),
                        "telegram_suppressed_duplicate":
                            bool(
                                telegram_already_sent
                            ),
                        "execution_status":
                            execution_result.get(
                                "status"
                            ),
                        "protection_status":
                            execution_result.get(
                                "protection_status"
                            ),
                        "order_id":
                            execution_result.get(
                                "order_id"
                            ),
                        "ts":
                            int(
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
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    # =========================================================================
    # FINAL
    # =========================================================================

    print(
        f"[ENGINE] "
        f"trades_this_cycle="
        f"{trades_this_cycle}"
    )

    print(
        "[ENGINE_SUMMARY] "
        + " ".join(
            f"{key}={value}"
            for key, value
            in stats.items()
        )
    )


if __name__ == "__main__":
    main()
