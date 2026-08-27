from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

from event_engine.coinalyze import fetch_data
from event_engine.bingx import (
    refresh_contracts,
    get_contract,
    fetch_klines,
    open_market,
    wait_for_position_fill_directional,
    get_positions,
    get_position_directional,
    get_open_protection_directional,
    ensure_directional_protection,
)
from event_engine.signals import (
    add_cvd,
    detect_divergences,
    detect_squeeze_release,
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
)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

DATA = Path("data")
DATA.mkdir(exist_ok=True)

EVENTS = DATA / "events.jsonl"
TRADES = DATA / "trades.jsonl"
ACTIONS = DATA / "actions.jsonl"
HEALTH = DATA / "health.jsonl"


# 0 or negative means scan ALL eligible liquidity/contract candidates.
MAX_CANDIDATES = int(
    os.environ.get("MAX_CANDIDATES", "0")
)

MIN_VOL = float(
    os.environ.get("MIN_VOLUME_24H", "10000000")
)

MIN_OI = float(
    os.environ.get("MIN_OPEN_INTEREST", "5000000")
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

CVD_MIN_CONFIRMATION = float(
    os.environ.get(
        "MIN_CVD24_CONFIRMATION",
        "55",
    )
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
        "3",
    )
)

EXECUTION_MODE = os.environ.get(
    "EXECUTION_MODE",
    os.environ.get(
        "BINGX_ENV",
        "vst",
    ),
)


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids: set[str] = set()

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        try:
            value = json.loads(line).get(
                "event_id"
            )
        except Exception:
            continue

        if value:
            ids.add(str(value))

    return ids


def load_successful_trade_ids(
    path: Path,
) -> set[str]:
    """
    Считывает только успешно открытые сделки,
    позволяя retry при временных сбоях API.
    """
    if not path.exists():
        return set()

    ids: set[str] = set()

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        try:
            obj = json.loads(line)

            status = str(
                obj.get(
                    "result",
                    {},
                ).get(
                    "status",
                    "",
                )
            ).lower()

            if status in {
                "opened_protected",
                "opened_protection_check_required",
                "opened",
                "already_executed",
            }:
                value = obj.get(
                    "event_id"
                )

                if value:
                    ids.add(str(value))

        except Exception:
            continue

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


def emit_event(ev: dict) -> None:
    append_jsonl(
        EVENTS,
        ev,
    )


def record_trade(obj: dict) -> None:
    append_jsonl(
        TRADES,
        obj,
    )


def record_action(obj: dict) -> None:
    append_jsonl(
        ACTIONS,
        obj,
    )


def calculate_setup_score(
    ev: dict,
    coinalyze_row: Any,
    df_15m: pd.DataFrame,
) -> float:

    score = 50.0

    fact = ev.get(
        "event_fact",
        {},
    )

    direction = str(
        ev.get(
            "direction",
            "LONG",
        )
    ).upper()

    try:
        delta_atr = float(
            fact.get(
                "price_delta_atr",
                0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        delta_atr = 0.0

    if delta_atr >= 1.0:
        score += 15.0

    elif delta_atr >= 0.5:
        score += 10.0

    if "CVD" in str(
        ev.get(
            "event_type",
            "",
        )
    ):
        score += 15.0

    if "VOLATILITY_SQUEEZE_RELEASE" in str(
        ev.get(
            "event_type",
            "",
        )
    ):
        try:
            comp_ratio = float(
                fact.get(
                    "compression_ratio",
                    1.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            comp_ratio = 1.0

        if comp_ratio < 0.65:
            score += 15.0

        try:
            duration = int(
                fact.get(
                    "squeeze_duration_bars",
                    0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            duration = 0

        if duration >= 5:
            score += 10.0

    if coinalyze_row is not None:

        fr = getattr(
            coinalyze_row,
            "fr_oiw",
            None,
        )

        if fr is not None:

            try:
                fr = float(fr)
            except (
                TypeError,
                ValueError,
            ):
                fr = None

        if fr is not None:

            if (
                direction == "LONG"
                and fr < 0
            ):
                score += 15.0

            elif (
                direction == "SHORT"
                and fr > 0.02
            ):
                score += 15.0

            elif (
                direction == "LONG"
                and fr > 0.05
            ):
                score -= 15.0

            elif (
                direction == "SHORT"
                and fr < -0.05
            ):
                score -= 15.0

    if (
        "volume" in df_15m.columns
        and len(df_15m) >= 20
    ):
        recent_avg = (
            pd.to_numeric(
                df_15m["volume"],
                errors="coerce",
            )
            .iloc[-21:-1]
            .mean()
        )

        if (
            pd.notna(recent_avg)
            and float(recent_avg) > 0
        ):
            try:
                vol_ratio = (
                    float(
                        df_15m[
                            "volume"
                        ].iloc[-1]
                    )
                    / float(recent_avg)
                )
            except (
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                vol_ratio = 0.0

            if vol_ratio >= 1.5:
                score += 10.0

            elif vol_ratio >= 1.2:
                score += 5.0

    return max(
        0.0,
        min(
            100.0,
            score,
        ),
    )


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

    entry_price = float(
        entry_price
    )

    if entry_price <= 0:
        raise ValueError(
            f"Invalid entry_price={entry_price}"
        )

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

    prev_close = df["close"].shift(1)

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
    ).max(axis=1)

    atr = (
        tr.rolling(
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

    atr = float(atr)

    risk_pct = (
        atr / entry_price
    ) * 100.0

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

    elif direction == "SHORT":

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

    else:
        raise ValueError(
            f"Invalid direction={direction}"
        )

    return {
        "entry_reference": entry_price,
        "invalidation_price": invalidation,
        "target_price": target,
        "target_rr": 2.0,
        "planned_weighted_rr": 1.55,
        "realized_rr": 1.55,
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
            (entry - sl_price)
            / entry
            * 100.0
        )

        tp_pct = (
            (final_tp_price - entry)
            / entry
            * 100.0
        )

    elif direction == "SHORT":

        sl_pct = (
            (sl_price - entry)
            / entry
            * 100.0
        )

        tp_pct = (
            (entry - final_tp_price)
            / entry
            * 100.0
        )

    else:
        raise ValueError(
            f"Invalid direction={direction}"
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


def install_protection(
    symbol: str,
    direction: str,
    position: dict,
    setup: dict,
    sl_pct: float,
    tp_levels: list,
    trade_id: str,
) -> dict:

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

    if (
        avg_price <= 0
        or qty <= 0
    ):
        return {
            "status": "PROTECTION_INVALID_POSITION",
            "error": (
                f"invalid avgPrice={avg_price} "
                f"or qty={qty}"
            ),
        }

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

        return {
            "status": "PROTECTION_EXCEPTION",
            "error": str(exc),
        }


def _tp_orders_to_tracker(
    tp_orders: list[dict],
) -> list[dict]:

    out: list[dict] = []

    for order in tp_orders:

        cid = str(
            order.get(
                "clientOrderId",
                "",
            )
        ).upper()

        leg = next(
            (
                x
                for x in (
                    "tp1",
                    "tp2",
                    "tp3",
                )
                if x.upper() in cid
            ),
            None,
        )

        if not leg:
            continue

        out.append(
            {
                "leg": leg,
                "status": "already_exists",
                "order_id": str(
                    order.get(
                        "orderId",
                        "",
                    )
                ),
                "price": float(
                    order.get(
                        "stopPrice",
                        0,
                    )
                    or order.get(
                        "price",
                        0,
                    )
                    or 0
                ),
                "qty": float(
                    order.get(
                        "origQty",
                        0,
                    )
                    or order.get(
                        "quantity",
                        0,
                    )
                    or 0
                ),
            }
        )

    return out


def _sl_order_to_tracker(
    sl_orders: list[dict],
) -> dict:

    if not sl_orders:
        return {}

    sl = sl_orders[0]

    return {
        "status": "already_exists",
        "order_id": str(
            sl.get(
                "orderId",
                "",
            )
        ),
        "stop_price": float(
            sl.get(
                "stopPrice",
                0,
            )
            or sl.get(
                "price",
                0,
            )
            or 0
        ),
        "qty": float(
            sl.get(
                "origQty",
                0,
            )
            or sl.get(
                "quantity",
                0,
            )
            or 0
        ),
    }


def _expected_tp_leg_count(
    position_qty: float,
    symbol: str,
) -> int:

    contract = (
        get_contract(symbol)
        or {}
    )

    try:
        min_qty = float(
            contract.get(
                "tradeMinQuantity"
            )
            or contract.get(
                "minQty"
            )
            or 0
        )
    except (
        TypeError,
        ValueError,
    ):
        min_qty = 0.0

    return (
        1
        if (
            min_qty > 0
            and position_qty
            < min_qty * 3
        )
        else 3
    )


def reconcile_all_open_positions() -> None:

    try:
        positions = get_positions()

    except Exception as exc:

        print(
            "[RECONCILIATION_ERROR] "
            f"Failed to fetch positions: {exc}"
        )

        return

    for p in positions:

        bx_symbol = str(
            p.get(
                "symbol",
                "",
            )
        ).upper()

        if not bx_symbol:
            continue

        position_side = str(
            p.get(
                "positionSide",
                "",
            )
        ).upper()

        try:

            amt = float(
                p.get(
                    "positionAmt",
                    0,
                )
                or 0
            )

            avg_price = float(
                p.get(
                    "avgPrice",
                    0,
                )
                or p.get(
                    "entryPrice",
                    0,
                )
                or 0
            )

        except (
            ValueError,
            TypeError,
        ):
            continue

        if (
            amt == 0
            or avg_price <= 0
        ):
            continue

        direction = (
            position_side
            if position_side
            in {"LONG", "SHORT"}
            else (
                "LONG"
                if amt > 0
                else "SHORT"
            )
        )

        qty = abs(amt)

        prot = (
            get_open_protection_directional(
                bx_symbol,
                direction,
            )
        )

        if prot.get(
            "status"
        ) != "ok":

            print(
                "[RECONCILIATION] "
                f"Cannot inspect protection for "
                f"{bx_symbol}: "
                f"{prot.get('error', 'unknown error')}"
            )

            continue

        sl_orders = list(
            prot.get(
                "sl_orders",
                [],
            )
        )

        tp_orders = list(
            prot.get(
                "tp_orders",
                [],
            )
        )

        expected_tp_count = (
            _expected_tp_leg_count(
                qty,
                bx_symbol,
            )
        )

        known_tp_legs = {
            leg
            for order in tp_orders
            for leg in (
                "tp1",
                "tp2",
                "tp3",
            )
            if leg.upper()
            in str(
                order.get(
                    "clientOrderId",
                    "",
                )
            ).upper()
        }

        sl_valid = False

        if sl_orders:

            sl_price = float(
                sl_orders[0].get(
                    "stopPrice",
                    0,
                )
                or sl_orders[0].get(
                    "price",
                    0,
                )
                or 0
            )

            sl_amt = float(
                sl_orders[0].get(
                    "origQty",
                    0,
                )
                or sl_orders[0].get(
                    "quantity",
                    0,
                )
                or 0
            )

            if (
                direction == "LONG"
                and 0 < sl_price < avg_price
                and sl_amt > 0
            ):
                sl_valid = True

            elif (
                direction == "SHORT"
                and sl_price > avg_price > 0
                and sl_amt > 0
            ):
                sl_valid = True

        protection_complete = (
            sl_valid
            and len(known_tp_legs)
            >= expected_tp_count
        )

        if protection_complete:

            print(
                "[RECONCILIATION] "
                f"{bx_symbol} ({direction}) "
                f"protection OK: "
                f"SL=1 "
                f"TP={len(known_tp_legs)}; "
                "no changes"
            )

            tracker_tp = (
                _tp_orders_to_tracker(
                    tp_orders
                )
            )

            tracker_sl = (
                _sl_order_to_tracker(
                    sl_orders
                )
            )

            if (
                tracker_tp
                and tracker_sl
            ):

                tracked = (
                    update_active_trade_protection(
                        symbol=bx_symbol,
                        direction=direction,
                        tp_orders=tracker_tp,
                        sl_result=tracker_sl,
                    )
                )

                if not tracked:

                    register_active_trade(
                        event_id=(
                            f"RECON_{bx_symbol}_"
                            f"{direction}"
                        ),
                        symbol=bx_symbol.replace(
                            "-USDT",
                            "",
                        ),
                        name=bx_symbol.replace(
                            "-USDT",
                            "",
                        ),
                        direction=direction,
                        entry_price=avg_price,
                        qty=qty,
                        tp_orders=tracker_tp,
                        sl_result=tracker_sl,
                        event_type=(
                            "RECONCILED_POSITION"
                        ),
                    )

            continue

        print(
            "[RECONCILIATION] "
            f"{bx_symbol} ({direction}) "
            "protection incomplete: "
            f"SL={len(sl_orders)} "
            f"TP={len(known_tp_legs)}/"
            f"{expected_tp_count}; "
            "repairing only missing protection..."
        )

        sl_pct = 2.0
        tp_pct = 4.0

        try:

            k1 = fetch_klines(
                bx_symbol,
                "1h",
                limit=30,
            )

            if len(k1) >= 20:

                df1 = pd.DataFrame(k1)

                for col in (
                    "high",
                    "low",
                    "close",
                ):
                    df1[col] = pd.to_numeric(
                        df1[col],
                        errors="coerce",
                    )

                prev_close = (
                    df1["close"].shift(1)
                )

                tr = pd.concat(
                    [
                        (
                            df1["high"]
                            - df1["low"]
                        ),
                        (
                            df1["high"]
                            - prev_close
                        ).abs(),
                        (
                            df1["low"]
                            - prev_close
                        ).abs(),
                    ],
                    axis=1,
                ).max(axis=1)

                atr = (
                    tr.rolling(
                        14,
                        min_periods=14,
                    )
                    .mean()
                    .iloc[-1]
                )

                if (
                    pd.notna(atr)
                    and float(atr) > 0
                ):

                    risk_pct = max(
                        0.50,
                        min(
                            float(atr)
                            / avg_price
                            * 100.0,
                            5.00,
                        ),
                    )

                    sl_pct = risk_pct
                    tp_pct = (
                        risk_pct * 2.0
                    )

        except Exception as exc:

            print(
                "[RECONCILIATION_ATR_ERROR] "
                f"{bx_symbol}: {exc}"
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

        res = ensure_directional_protection(
            symbol=bx_symbol,
            direction=direction,
            avg_price=avg_price,
            qty=qty,
            stop_loss_pct=sl_pct,
            tp_levels=tp_levels,
            trade_id=(
                f"REC_{bx_symbol}_{direction}"
            ),
        )

        status = str(
            res.get(
                "status",
                "",
            )
        ).upper()

        created_sl = (
            str(
                res.get(
                    "sl_result",
                    {},
                ).get(
                    "status",
                    "",
                )
            ).lower()
            == "created"
        )

        created_tp = any(
            str(
                x.get(
                    "status",
                    "",
                )
            ).lower()
            == "created"
            for x in res.get(
                "tp_orders",
                [],
            )
        )

        changed = (
            created_sl
            or created_tp
        )

        print(
            "[RECONCILIATION] "
            f"{bx_symbol} "
            f"result={status} "
            f"changed={changed}"
        )

        if status in {
            "PROTECTED",
            "SL_ONLY",
        }:

            updated_existing = False

            repaired_tp = res.get(
                "tp_orders",
                [],
            )

            repaired_sl = res.get(
                "sl_result",
                {},
            )

            if (
                repaired_tp
                and repaired_sl
            ):

                updated_existing = (
                    update_active_trade_protection(
                        symbol=bx_symbol,
                        direction=direction,
                        tp_orders=repaired_tp,
                        sl_result=repaired_sl,
                    )
                )

            if (
                not updated_existing
                and repaired_tp
                and repaired_sl
            ):

                register_active_trade(
                    event_id=(
                        f"RECON_{bx_symbol}_"
                        f"{direction}"
                    ),
                    symbol=bx_symbol.replace(
                        "-USDT",
                        "",
                    ),
                    name=bx_symbol.replace(
                        "-USDT",
                        "",
                    ),
                    direction=direction,
                    entry_price=avg_price,
                    qty=qty,
                    tp_orders=repaired_tp,
                    sl_result=repaired_sl,
                    event_type=(
                        "RECONCILED_POSITION"
                    ),
                )

            if changed:

                send_tg(
                    f"🛡 <b>[ЗАЩИТА ВОССТАНОВЛЕНА] "
                    f"{bx_symbol}</b>\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"Недостающая защита "
                    f"восстановлена:\n"
                    f"• Направление: "
                    f"<b>{direction}</b>\n"
                    f"• Цена входа: "
                    f"<code>{avg_price:.8g}</code>\n"
                    f"• SL: "
                    f"<code>{sl_pct:.2f}%</code>\n"
                    f"• TP: "
                    f"<code>+{tp_pct:.2f}%</code> "
                    f"(каскад)"
                )


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

    if not isinstance(
        opened,
        dict,
    ):

        return {
            "status": "OPEN_INVALID_RESPONSE",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "raw": repr(opened),
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
            "error": (
                opened.get("error")
                or opened.get("msg")
                or "unknown_open_error"
            ),
            "bingx_code": opened.get(
                "code"
            ),
        }

    order_id = opened.get(
        "order_id"
    )

    try:

        position = (
            wait_for_position_fill_directional(
                symbol=symbol,
                direction=direction,
                timeout_sec=15,
                poll_interval=0.5,
            )
        )

    except Exception as exc:

        return {
            "status": "POSITION_WAIT_FAILED",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
            "error": str(exc),
        }

    if (
        not isinstance(
            position,
            dict,
        )
        or str(
            position.get(
                "status",
                "",
            )
        ).lower()
        != "found"
    ):

        return {
            "status": "POSITION_NOT_CONFIRMED",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
            "position": position,
        }

    try:

        actual_qty = abs(
            float(
                position.get(
                    "positionAmt",
                    0,
                )
                or 0
            )
        )

        actual_avg_price = float(
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

    except (
        TypeError,
        ValueError,
    ):

        actual_qty = 0.0
        actual_avg_price = 0.0

    if (
        actual_qty <= 0
        or actual_avg_price <= 0
    ):

        return {
            "status": "POSITION_INVALID",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
            "position": position,
        }

    try:

        sl_pct, tp_levels = (
            build_tp_levels(
                setup,
                direction,
            )
        )

    except Exception as exc:

        return {
            "status": "PROTECTION_SETUP_INVALID",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "position": position,
            "open_result": opened,
            "error": str(exc),
        }

    protection = install_protection(
        symbol=symbol,
        direction=direction,
        position={
            **position,
            "positionAmt": actual_qty,
            "avgPrice": actual_avg_price,
        },
        setup=setup,
        sl_pct=sl_pct,
        tp_levels=tp_levels,
        trade_id=trade_id,
    )

    protection_status = str(
        protection.get(
            "status",
            "",
        )
    ).upper()

    if (
        protection_status
        == "PROTECTED"
    ):
        final_status = (
            "opened_protected"
        )

    elif (
        protection_status
        == "SL_ONLY"
    ):
        final_status = (
            "opened_protection_check_required"
        )

    else:
        final_status = (
            "opened_protection_failed"
        )

    return {
        "status": final_status,
        "mode": EXECUTION_MODE,
        "order_id": order_id,
        "open_result": opened,
        "position": {
            **position,
            "positionAmt": actual_qty,
            "avgPrice": actual_avg_price,
        },
        "protection": protection,
        "sl_pct": sl_pct,
        "tp_levels": tp_levels,
    }


def main() -> None:

    # ============================================================
    # 1. RECONCILIATION
    # ============================================================

    if EXECUTION_ENABLED:

        try:
            reconcile_all_open_positions()

        except Exception as exc:

            print(
                "[RECONCILIATION_ERROR] "
                f"{exc}"
            )

        try:
            update_active_trades()

        except Exception as exc:

            print(
                "[TRACKER_ERROR] "
                f"{exc}"
            )

    # ============================================================
    # 2. PIPELINE STATS
    # ============================================================

    stats = {
        "coinalyze_rows": 0,
        "liquidity_candidates": 0,
        "contract_candidates": 0,
        "candidates_scanned": 0,

        "divergence_events": 0,
        "squeeze_events": 0,
        "events_total": 0,

        "fresh_events": 0,

        "rejected_age": 0,
        "rejected_btc": 0,
        "rejected_trigger": 0,
        "rejected_cvd": 0,

        "trigger_breakout_failed": 0,
        "trigger_volume_failed": 0,
        "trigger_data_failed": 0,
        "trigger_direction_failed": 0,
        "trigger_passed": 0,

        "fresh_long": 0,
        "fresh_short": 0,

        "fresh_divergence": 0,
        "fresh_squeeze": 0,

        "valid_signals": 0,
        "execution_attempts": 0,
        "trades": 0,
        "scan_errors": 0,
    }

    # ============================================================
    # 3. BTC REGIME DATA
    # ============================================================

    btc_regime_df = None

    try:

        btc_klines = fetch_klines(
            "BTC-USDT",
            "1h",
            limit=10,
        )

        if btc_klines:
            btc_regime_df = pd.DataFrame(
                btc_klines
            )

    except Exception as exc:

        print(
            "[BTC_FETCH_ERROR] "
            f"{exc}"
        )

    # ============================================================
    # 4. COINALYZE
    # ============================================================

    rows = []

    try:

        rows = fetch_data()

    except Exception as exc:

        stats["scan_errors"] += 1

        print(
            "[COINALYZE_SCRAPE_ERROR] "
            f"{exc}"
        )

    stats[
        "coinalyze_rows"
    ] = len(rows)

    # ============================================================
    # 5. BINGX CONTRACTS
    # ============================================================

    try:

        refresh_contracts()

    except Exception as exc:

        stats["scan_errors"] += 1

        print(
            "[BINGX] contracts refresh "
            f"error={exc}"
        )

    # ============================================================
    # 6. UNIVERSE FILTER
    # ============================================================

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

            stats[
                "liquidity_candidates"
            ] += 1

            if not get_contract(
                r.symbol
            ):
                continue

            stats[
                "contract_candidates"
            ] += 1

            candidates.append(r)

        except Exception as exc:

            print(
                "[CANDIDATE_FILTER_ERROR] "
                f"{getattr(r, 'symbol', '?')}: "
                f"{exc}"
            )

            continue

    # ============================================================
    # IMPORTANT:
    # MAX_CANDIDATES=0 => scan ALL.
    # We do not remove this feature completely because a positive
    # explicit limit may still be useful for controlled testing.
    # ============================================================

    if MAX_CANDIDATES > 0:

        candidates = candidates[
            :MAX_CANDIDATES
        ]

    stats[
        "candidates_scanned"
    ] = len(candidates)

    # ============================================================
    # 7. EVENT / EXECUTION STATE
    # ============================================================

    seen_events = load_ids(
        EVENTS
    )

    executed_event_ids = (
        load_successful_trade_ids(
            TRADES
        )
    )

    telegram_sent_event_ids = (
        load_ids(
            ACTIONS
        )
    )

    opportunities: List[
        dict
    ] = []

    # ============================================================
    # FORENSIC DIAGNOSTICS
    # ============================================================

    fresh_event_details: list[
        dict
    ] = []

    fresh_event_type_counts: dict[
        str,
        int,
    ] = {}

    fresh_direction_counts: dict[
        str,
        int,
    ] = {}

    trigger_reason_counts: dict[
        str,
        int,
    ] = {
        "breakout_failed": 0,
        "volume_failed": 0,
        "insufficient_data": 0,
        "invalid_direction": 0,
        "invalid_15m_data": 0,
        "passed": 0,
        "unknown": 0,
    }

    squeeze_details: list[
        dict
    ] = []

    # ============================================================
    # 8. SCAN ALL CANDIDATES
    # ============================================================

    for r in candidates:

        symbol = r.symbol

        try:

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

            if len(k1) < 60:
                continue

            d1 = add_cvd(
                pd.DataFrame(k1)
            )

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
                    min_squeeze_bars=3,
                )
            )

            stats[
                "divergence_events"
            ] += len(
                divergence_events
            )

            stats[
                "squeeze_events"
            ] += len(
                squeeze_events
            )

            all_events = (
                divergence_events
                + squeeze_events
            )

            stats[
                "events_total"
            ] += len(all_events)

            if not all_events:
                continue

            # ----------------------------------------------------
            # 15M data is fetched only if a fresh 1H event exists.
            # ----------------------------------------------------

            d15 = None

            for ev in all_events:

                event_id = ev.get(
                    "event_id"
                )

                if not event_id:
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
                    continue

                try:
                    detected_at = int(
                        ev.get(
                            "timestamps",
                            {},
                        ).get(
                            "detected_at_ts",
                            0,
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    stats[
                        "rejected_age"
                    ] += 1
                    continue

                try:
                    latest_close = int(
                        d1[
                            "close_time"
                        ].iloc[-1]
                    )
                except (
                    KeyError,
                    IndexError,
                    TypeError,
                    ValueError,
                ):
                    stats[
                        "scan_errors"
                    ] += 1
                    continue

                age = (
                    latest_close
                    - detected_at
                ) / 60000.0

                # -----------------------------------------------
                # Freshness gate
                # -----------------------------------------------

                if (
                    age < 0
                    or age > MAX_AGE
                ):
                    stats[
                        "rejected_age"
                    ] += 1
                    continue

                # -----------------------------------------------
                # Fresh-event telemetry
                # -----------------------------------------------

                stats[
                    "fresh_events"
                ] += 1

                event_type = str(
                    ev.get(
                        "event_type",
                        "UNKNOWN",
                    )
                )

                fresh_event_type_counts[
                    event_type
                ] = (
                    fresh_event_type_counts.get(
                        event_type,
                        0,
                    )
                    + 1
                )

                fresh_direction_counts[
                    direction
                ] = (
                    fresh_direction_counts.get(
                        direction,
                        0,
                    )
                    + 1
                )

                if direction == "LONG":

                    stats[
                        "fresh_long"
                    ] += 1

                elif direction == "SHORT":

                    stats[
                        "fresh_short"
                    ] += 1

                if (
                    event_type
                    == "VOLATILITY_SQUEEZE_RELEASE"
                ):

                    stats[
                        "fresh_squeeze"
                    ] += 1

                    fact = ev.get(
                        "event_fact",
                        {},
                    )

                    squeeze_details.append(
                        {
                            "symbol": symbol,
                            "direction": direction,
                            "event_id": event_id,
                            "age_min": round(
                                age,
                                2,
                            ),
                            "compression_ratio": (
                                fact.get(
                                    "compression_ratio"
                                )
                            ),
                            "duration_bars": (
                                fact.get(
                                    "squeeze_duration_bars"
                                )
                            ),
                            "price": fact.get(
                                "detection_close_price"
                            ),
                        }
                    )

                else:

                    stats[
                        "fresh_divergence"
                    ] += 1

                # -----------------------------------------------
                # Persist event
                # -----------------------------------------------

                if event_id not in seen_events:

                    emit_event(ev)
                    seen_events.add(
                        event_id
                    )

                # -----------------------------------------------
                # BTC regime
                # -----------------------------------------------

                if (
                    btc_regime_df is not None
                    and symbol != "BTC-USDT"
                ):

                    btc_ok, _ = (
                        check_btc_regime(
                            btc_regime_df,
                            direction,
                        )
                    )

                    if not btc_ok:

                        stats[
                            "rejected_btc"
                        ] += 1

                        continue

                # -----------------------------------------------
                # Fetch 15M only after fresh 1H event.
                # -----------------------------------------------

                if d15 is None:

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

                    if len(k15) < 20:

                        stats[
                            "trigger_data_failed"
                        ] += 1

                        trigger_reason_counts[
                            "insufficient_data"
                        ] += 1

                        fresh_event_details.append(
                            {
                                "symbol": symbol,
                                "direction": direction,
                                "event_type": event_type,
                                "event_id": event_id,
                                "age_min": round(
                                    age,
                                    2,
                                ),
                                "price_delta_atr": (
                                    ev.get(
                                        "event_fact",
                                        {},
                                    ).get(
                                        "price_delta_atr"
                                    )
                                ),
                                "trigger_reason": (
                                    "insufficient_data"
                                ),
                                "previous_high": None,
                                "previous_low": None,
                                "current_close": None,
                                "current_volume": None,
                                "volume_sma20": None,
                                "volume_ratio": None,
                            }
                        )

                        if REQUIRE_TRIGGER:

                            stats[
                                "rejected_trigger"
                            ] += 1

                        continue

                    d15 = pd.DataFrame(
                        k15
                    )

                # -----------------------------------------------
                # Causal 15M timing validation
                # -----------------------------------------------

                try:

                    latest_15m_close_ts = int(
                        d15[
                            "close_time"
                        ].iloc[-1]
                    )

                except (
                    KeyError,
                    IndexError,
                    TypeError,
                    ValueError,
                ):

                    stats[
                        "rejected_trigger"
                    ] += 1

                    stats[
                        "trigger_data_failed"
                    ] += 1

                    trigger_reason_counts[
                        "invalid_15m_data"
                    ] += 1

                    fresh_event_details.append(
                        {
                            "symbol": symbol,
                            "direction": direction,
                            "event_type": event_type,
                            "event_id": event_id,
                            "age_min": round(
                                age,
                                2,
                            ),
                            "price_delta_atr": (
                                ev.get(
                                    "event_fact",
                                    {},
                                ).get(
                                    "price_delta_atr"
                                )
                            ),
                            "trigger_reason": (
                                "invalid_15m_data"
                            ),
                            "previous_high": None,
                            "previous_low": None,
                            "current_close": None,
                            "current_volume": None,
                            "volume_sma20": None,
                            "volume_ratio": None,
                        }
                    )

                    continue

                trigger_delay_min = (
                    latest_15m_close_ts
                    - detected_at
                ) / 60000.0

                # -----------------------------------------------
                # Strict temporal causality
                # -----------------------------------------------

                if (
                    trigger_delay_min < 0
                    or trigger_delay_min > MAX_AGE
                ):

                    stats[
                        "rejected_age"
                    ] += 1

                    continue

                # -----------------------------------------------
                # 15M trigger
                # -----------------------------------------------

                if REQUIRE_TRIGGER:

                    trigger_diag = (
                        diagnose_15m_trigger(
                            d15,
                            direction,
                            min_vol_mult=1.05,
                        )
                    )

                    trigger_reason = str(
                        trigger_diag.get(
                            "reason",
                            "unknown",
                        )
                    )

                    trigger_reason_counts[
                        trigger_reason
                    ] = (
                        trigger_reason_counts.get(
                            trigger_reason,
                            0,
                        )
                        + 1
                    )

                    if trigger_diag.get(
                        "ok"
                    ):

                        stats[
                            "trigger_passed"
                        ] += 1

                    else:

                        stats[
                            "rejected_trigger"
                        ] += 1

                        if (
                            trigger_reason
                            == "breakout_failed"
                        ):

                            stats[
                                "trigger_breakout_failed"
                            ] += 1

                        elif (
                            trigger_reason
                            == "volume_failed"
                        ):

                            stats[
                                "trigger_volume_failed"
                            ] += 1

                        elif (
                            trigger_reason
                            in {
                                "insufficient_data",
                                "invalid_15m_data",
                            }
                        ):

                            stats[
                                "trigger_data_failed"
                            ] += 1

                        elif (
                            trigger_reason
                            == "invalid_direction"
                        ):

                            stats[
                                "trigger_direction_failed"
                            ] += 1

                        fact = ev.get(
                            "event_fact",
                            {},
                        )

                        fresh_event_details.append(
                            {
                                "symbol": symbol,
                                "direction": direction,
                                "event_type": event_type,
                                "event_id": event_id,
                                "age_min": round(
                                    age,
                                    2,
                                ),
                                "price_delta_atr": (
                                    fact.get(
                                        "price_delta_atr"
                                    )
                                ),
                                "trigger_reason": (
                                    trigger_reason
                                ),
                                "previous_high": (
                                    trigger_diag.get(
                                        "previous_high"
                                    )
                                ),
                                "previous_low": (
                                    trigger_diag.get(
                                        "previous_low"
                                    )
                                ),
                                "current_close": (
                                    trigger_diag.get(
                                        "current_close"
                                    )
                                ),
                                "current_volume": (
                                    trigger_diag.get(
                                        "current_volume"
                                    )
                                ),
                                "volume_sma20": (
                                    trigger_diag.get(
                                        "volume_sma20"
                                    )
                                ),
                                "volume_ratio": (
                                    trigger_diag.get(
                                        "volume_ratio"
                                    )
                                ),
                            }
                        )

                        continue

                else:

                    trigger_diag = {
                        "ok": True,
                        "reason": "disabled",
                    }

                # -----------------------------------------------
                # Optional CVD gate
                # -----------------------------------------------

                if REQUIRE_CVD:

                    cvd24 = getattr(
                        r,
                        "cvd24",
                        None,
                    )

                    try:
                        cvd24_value = float(
                            cvd24
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        stats[
                            "rejected_cvd"
                        ] += 1

                        continue

                    if (
                        cvd24_value
                        <= CVD_MIN_CONFIRMATION
                    ):

                        stats[
                            "rejected_cvd"
                        ] += 1

                        continue

                # -----------------------------------------------
                # Setup + scoring
                # -----------------------------------------------

                fact = ev.get(
                    "event_fact",
                    {},
                )

                try:

                    price = float(
                        fact.get(
                            "detection_close_price"
                        )
                        or fact.get(
                            "close"
                        )
                        or r.price
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    stats[
                        "scan_errors"
                    ] += 1

                    continue

                setup = build_event_setup(
                    ev=ev,
                    df_1h=d1,
                    entry_price=price,
                )

                score = (
                    calculate_setup_score(
                        ev=ev,
                        coinalyze_row=r,
                        df_15m=d15,
                    )
                )

                opportunities.append(
                    {
                        "event": ev,
                        "event_id": event_id,
                        "symbol": symbol,
                        "direction": direction,
                        "price": price,
                        "setup": setup,
                        "score": score,
                        "coinalyze_row": r,
                    }
                )

        except Exception as exc:

            stats[
                "scan_errors"
            ] += 1

            print(
                "[SCAN_ERROR] "
                f"{symbol}: {exc}"
            )

    # ============================================================
    # 9. SORT FINAL OPPORTUNITIES
    # ============================================================

    stats[
        "valid_signals"
    ] = len(
        opportunities
    )

    opportunities.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # ============================================================
    # 10. EXECUTION
    # ============================================================

    trades_this_cycle = 0

    for opp in opportunities:

        event_id = opp[
            "event_id"
        ]

        symbol = opp[
            "symbol"
        ]

        direction = opp[
            "direction"
        ]

        price = opp[
            "price"
        ]

        setup = opp[
            "setup"
        ]

        score = opp[
            "score"
        ]

        r = opp[
            "coinalyze_row"
        ]

        ev = opp[
            "event"
        ]

        if (
            event_id
            in executed_event_ids
        ):

            execution_result = {
                "status": "ALREADY_EXECUTED",
                "mode": EXECUTION_MODE,
                "order_id": None,
            }

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
                    "score": score,
                    "event_type": ev.get(
                        "event_type"
                    ),
                    "ts": int(
                        pd.Timestamp.utcnow()
                        .timestamp()
                        * 1000
                    ),
                    "result": execution_result,
                    "setup": setup,
                }
            )

            status = str(
                execution_result.get(
                    "status",
                    "",
                )
            )

            if status in {
                "opened_protected",
                "opened_protection_check_required",
                "opened_protection_failed",
            }:

                trades_this_cycle += 1

                if status in {
                    "opened_protected",
                    "opened_protection_check_required",
                }:

                    executed_event_ids.add(
                        event_id
                    )

                    stats[
                        "trades"
                    ] += 1

                else:

                    print(
                        "[CRITICAL_PROTECTION_FAILED] "
                        f"Position opened but "
                        f"protection failed for "
                        f"{symbol}: "
                        f"{execution_result.get('protection')}"
                    )

                try:

                    register_active_trade(
                        event_id=event_id,
                        symbol=symbol,
                        name=(
                            getattr(
                                r,
                                "name",
                                None,
                            )
                            or symbol
                        ),
                        direction=direction,
                        entry_price=float(
                            execution_result.get(
                                "position",
                                {},
                            ).get(
                                "avgPrice",
                                price,
                            )
                        ),
                        qty=float(
                            execution_result.get(
                                "position",
                                {},
                            ).get(
                                "positionAmt",
                                0,
                            )
                        ),
                        tp_orders=(
                            execution_result.get(
                                "protection",
                                {},
                            ).get(
                                "tp_orders",
                                [],
                            )
                        ),
                        sl_result=(
                            execution_result.get(
                                "protection",
                                {},
                            ).get(
                                "sl_result",
                                {},
                            )
                        ),
                        event_type=ev.get(
                            "event_type",
                            "",
                        ),
                        coinalyze_row=r,
                        score=score,
                    )

                except Exception as exc:

                    print(
                        "[REGISTER_ACTIVE_TRADE_ERROR] "
                        f"{symbol}: {exc}"
                    )

        elif not EXECUTION_ENABLED:

            execution_result = {
                "status": "DISABLED",
                "mode": EXECUTION_MODE,
                "order_id": None,
            }

        else:

            execution_result = {
                "status": "TRADE_LIMIT_REACHED",
                "mode": EXECUTION_MODE,
                "order_id": None,
            }

        # --------------------------------------------------------
        # Telegram
        # --------------------------------------------------------

        label = (
            "🚨 LONG SIGNAL"
            if direction == "LONG"
            else "🔻 SHORT SIGNAL"
        )

        msg = format_signal(
            ev,
            setup=setup,
            coinalyze_row=r,
            execution=execution_result,
            score=score,
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

        is_real_execution = (
            execution_result.get(
                "status"
            )
            in {
                "opened_protected",
                "opened",
                "opened_protection_check_required",
                "opened_protection_failed",
            }
        )

        telegram_already_sent = (
            event_id
            in telegram_sent_event_ids
        ) and not is_real_execution

        sent = False

        if not telegram_already_sent:

            try:

                sent = bool(
                    send_tg(msg)
                )

            except Exception:

                sent = False

            if sent:
                telegram_sent_event_ids.add(
                    event_id
                )

        record_action(
            {
                "event_id": event_id,
                "symbol": symbol,
                "direction": direction,
                "score": score,
                "event_type": ev.get(
                    "event_type"
                ),
                "telegram_sent": bool(
                    sent
                ),
                "execution_status": (
                    execution_result.get(
                        "status"
                    )
                ),
                "ts": int(
                    pd.Timestamp.utcnow()
                    .timestamp()
                    * 1000
                ),
            }
        )

    # ============================================================
    # 11. SHADOW HEALTH
    # ============================================================

    try:

        append_shadow_health(
            events_path=EVENTS,
            health_path=HEALTH,
            trades_path=TRADES,
        )

    except Exception as exc:

        print(
            "[SHADOW_HEALTH_ERROR] "
            f"{exc}"
        )

    # ============================================================
    # 12. FORENSIC TRIGGER BREAKDOWN
    # ============================================================

    print("")
    print(
        "================ "
        "FORENSIC TRIGGER BREAKDOWN "
        "================"
    )

    print(
        f"fresh_events="
        f"{stats['fresh_events']} "
        f"divergence="
        f"{stats['fresh_divergence']} "
        f"squeeze="
        f"{stats['fresh_squeeze']} "
        f"LONG="
        f"{stats['fresh_long']} "
        f"SHORT="
        f"{stats['fresh_short']}"
    )

    print(
        f"trigger_passed="
        f"{stats['trigger_passed']} "
        f"breakout_failed="
        f"{stats['trigger_breakout_failed']} "
        f"volume_failed="
        f"{stats['trigger_volume_failed']} "
        f"data_failed="
        f"{stats['trigger_data_failed']} "
        f"direction_failed="
        f"{stats['trigger_direction_failed']}"
    )

    print("")
    print("TRIGGER REASONS:")

    for reason, count in sorted(
        trigger_reason_counts.items(),
        key=lambda x: (
            -x[1],
            x[0],
        ),
    ):

        print(
            f"  {reason}: {count}"
        )

    print("")
    print("EVENT TYPES:")

    if fresh_event_type_counts:

        for (
            event_type,
            count,
        ) in sorted(
            fresh_event_type_counts.items(),
            key=lambda x: (
                -x[1],
                x[0],
            ),
        ):

            print(
                f"  {event_type}: {count}"
            )

    else:

        print("  none")

    print("")
    print("DIRECTIONS:")

    if fresh_direction_counts:

        for (
            direction,
            count,
        ) in sorted(
            fresh_direction_counts.items()
        ):

            print(
                f"  {direction}: {count}"
            )

    else:

        print("  none")

    print("")
    print("SQUEEZE DETAILS:")

    if squeeze_details:

        for item in squeeze_details:

            print(
                "  "
                f"{item['symbol']} "
                f"{item['direction']} "
                f"age={item['age_min']}m "
                f"compression="
                f"{item['compression_ratio']} "
                f"duration="
                f"{item['duration_bars']} "
                f"price="
                f"{item['price']}"
            )

    else:

        print("  none")

    print("")
    print("TRIGGER FAIL DETAILS:")

    if fresh_event_details:

        for item in fresh_event_details:

            print(
                "  "
                f"{item['symbol']} "
                f"{item['direction']} "
                f"{item['event_type']} "
                f"reason="
                f"{item['trigger_reason']} "
                f"age="
                f"{item['age_min']}m "
                f"close="
                f"{item['current_close']} "
                f"prev_high="
                f"{item['previous_high']} "
                f"prev_low="
                f"{item['previous_low']} "
                f"vol="
                f"{item['current_volume']} "
                f"sma20="
                f"{item['volume_sma20']} "
                f"vol_ratio="
                f"{item['volume_ratio']}"
            )

    else:

        print("  none")

    print(
        "============================================================"
    )

    # ============================================================
    # 13. ENGINE SUMMARY
    # ============================================================

    print(
        f"[ENGINE] "
        f"trades_this_cycle="
        f"{trades_this_cycle}"
    )

    print(
        "[ENGINE_SUMMARY] "
        + " ".join(
            f"{k}={v}"
            for k, v in stats.items()
        )
    )


if __name__ == "__main__":
    main()
