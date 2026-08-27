from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from event_engine.coinalyze import fetch_data
from event_engine.bingx import (
    refresh_contracts,
    get_contract,
    fetch_klines,
    open_market,
    wait_for_position_fill_directional,
    get_positions,
    get_open_protection_directional,
    ensure_directional_protection,
)
from event_engine.signals import (
    add_cvd,
    detect_divergences,
    detect_squeeze_release,
    build_15m_trigger,
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

# 0 or negative = scan ALL eligible candidates.
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
    os.environ.get("EXECUTION_ENABLED", "false").lower()
    == "true"
)

REQUIRE_CVD = (
    os.environ.get("REQUIRE_CVD_CONFIRMATION", "false").lower()
    == "true"
)

CVD_MIN_CONFIRMATION = float(
    os.environ.get("MIN_CVD24_CONFIRMATION", "55")
)

REQUIRE_TRIGGER = (
    os.environ.get("REQUIRE_15M_TRIGGER", "true").lower()
    == "true"
)

MAX_AGE = int(
    os.environ.get("MAX_EVENT_AGE_MIN", "90")
)

MAX_TRADES = int(
    os.environ.get("MAX_TRADES_PER_CYCLE", "3")
)

EXECUTION_MODE = os.environ.get(
    "EXECUTION_MODE",
    os.environ.get("BINGX_ENV", "vst"),
)


def load_ids(path: Path) -> set[str]:
    """
    Generic event_id loader.

    Used for events.jsonl where presence itself is meaningful.
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

            if not isinstance(obj, dict):
                continue

            value = obj.get("event_id")

        except Exception:
            continue

        if value:
            ids.add(str(value))

    return ids


def load_successful_trade_ids(
    path: Path,
) -> set[str]:
    """
    Load only event IDs that reached an accepted/opened state.

    Failed operations remain retryable.
    """
    if not path.exists():
        return set()

    ids: set[str] = set()

    successful_statuses = {
        "opened_protected",
        "opened_protection_check_required",
        "opened",
        "already_executed",
    }

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        try:
            obj = json.loads(line)

            if not isinstance(obj, dict):
                continue

            result = obj.get("result") or {}
            status = str(
                result.get("status", "")
            ).lower()

            if status not in successful_statuses:
                continue

            value = obj.get("event_id")

        except Exception:
            continue

        if value:
            ids.add(str(value))

    return ids


def load_successful_telegram_ids(
    path: Path,
) -> set[str]:
    """
    IMPORTANT:
    Only telegram_sent=True counts as successfully notified.

    A failed Telegram request must remain retryable.
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

            if not isinstance(obj, dict):
                continue

            if obj.get("telegram_sent") is not True:
                continue

            value = obj.get("event_id")

        except Exception:
            continue

        if value:
            ids.add(str(value))

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
    """
    Composite signal quality score.

    This score does NOT determine actual execution prices.
    It is calculated from the signal/event context.
    """
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
            ""
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
            if direction == "LONG" and fr < 0:
                score += 15.0
            elif direction == "SHORT" and fr > 0.02:
                score += 15.0
            elif direction == "LONG" and fr > 0.05:
                score -= 15.0
            elif direction == "SHORT" and fr < -0.05:
                score -= 15.0

    if (
        "volume" in df_15m.columns
        and len(df_15m) >= 20
    ):
        recent_avg = df_15m[
            "volume"
        ].iloc[-21:-1].mean()

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
    """
    Build setup relative to the supplied entry price.

    IMPORTANT:
    Before execution this is the signal/reference setup.
    After exchange fill this function is called again using actual avgPrice.
    """
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
            f"Invalid direction={direction}"
        )

    entry_price = float(entry_price)

    if (
        not math.isfinite(entry_price)
        or entry_price <= 0
    ):
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
        if col not in df.columns:
            raise ValueError(
                f"missing required 1H column: {col}"
            )

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
                + 2.0 * risk_pct / 100.0
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
                - 2.0 * risk_pct / 100.0
            )
        )

    return {
        "entry_reference": entry_price,
        "invalidation_price": invalidation,
        "target_price": target,

        "target_rr": 2.0,

        # 30% @ 0.5R
        # 30% @ 0.75R
        # 40% @ 1.0R
        #
        # Weighted reward relative to the full 2R target:
        # 2 * (0.30*0.50 + 0.30*0.75 + 0.40*1.00)
        # = 1.55R
        "planned_weighted_rr": 1.55,

        # Nothing has been realized at setup/open time.
        "realized_rr": None,

        "risk_pct": risk_pct,

        "trigger_ok": True,
    }


def build_tp_levels(
    setup: dict,
    direction: str,
):
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
    try:
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

    except (
        TypeError,
        ValueError,
    ) as exc:
        return {
            "status": "PROTECTION_INVALID_POSITION",
            "error": str(exc),
        }

    if (
        not math.isfinite(avg_price)
        or not math.isfinite(qty)
        or avg_price <= 0
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

        leg = None

        for candidate in (
            "tp1",
            "tp2",
            "tp3",
        ):
            if (
                f"_{candidate.upper()}_" in
                f"_{cid}_"
            ):
                leg = candidate
                break

        if not leg:
            continue

        try:
            price = float(
                order.get(
                    "stopPrice",
                    0,
                )
                or order.get(
                    "price",
                    0,
                )
                or 0
            )
        except (
            TypeError,
            ValueError,
        ):
            price = 0.0

        try:
            qty = float(
                order.get(
                    "origQty",
                    0,
                )
                or order.get(
                    "quantity",
                    0,
                )
                or 0
            )
        except (
            TypeError,
            ValueError,
        ):
            qty = 0.0

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
                "client_order_id": cid,
                "price": price,
                "qty": qty,
            }
        )

    return out


def _sl_order_to_tracker(
    sl_orders: list[dict],
) -> dict:
    if not sl_orders:
        return {}

    sl = sl_orders[0]

    try:
        stop_price = float(
            sl.get(
                "stopPrice",
                0,
            )
            or sl.get(
                "price",
                0,
            )
            or 0
        )
    except (
        TypeError,
        ValueError,
    ):
        stop_price = 0.0

    try:
        qty = float(
            sl.get(
                "origQty",
                0,
            )
            or sl.get(
                "quantity",
                0,
            )
            or 0
        )
    except (
        TypeError,
        ValueError,
    ):
        qty = 0.0

    return {
        "status": "already_exists",
        "order_id": str(
            sl.get(
                "orderId",
                "",
            )
        ),
        "client_order_id": str(
            sl.get(
                "clientOrderId",
                "",
            )
        ),
        "stop_price": stop_price,
        "qty": qty,
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


def _validate_existing_sl(
    direction: str,
    avg_price: float,
    sl_orders: list[dict],
) -> bool:
    """
    Existing SL is valid only when:
    - correct direction
    - positive trigger
    - trigger on the protective side
    - positive quantity
    - STOP/STOP_MARKET order type
    """
    if not sl_orders:
        return False

    direction = str(
        direction
    ).upper()

    for order in sl_orders:
        order_type = str(
            order.get(
                "type",
                "",
            )
        ).upper()

        if order_type not in {
            "STOP",
            "STOP_MARKET",
        }:
            continue

        try:
            sl_price = float(
                order.get(
                    "stopPrice",
                    0,
                )
                or order.get(
                    "price",
                    0,
                )
                or 0
            )

            sl_qty = float(
                order.get(
                    "origQty",
                    0,
                )
                or order.get(
                    "quantity",
                    0,
                )
                or 0
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            sl_price <= 0
            or sl_qty <= 0
        ):
            continue

        if direction == "LONG":
            if sl_price < avg_price:
                return True

        elif direction == "SHORT":
            if sl_price > avg_price:
                return True

    return False


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
            in {
                "LONG",
                "SHORT",
            }
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

        if prot.get("status") != "ok":
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

        known_tp_legs: set[str] = set()

        for order in tp_orders:
            cid = str(
                order.get(
                    "clientOrderId",
                    "",
                )
            ).upper()

            for leg in (
                "tp1",
                "tp2",
                "tp3",
            ):
                if (
                    f"_{leg.upper()}_"
                    in f"_{cid}_"
                ):
                    known_tp_legs.add(leg)

        sl_valid = _validate_existing_sl(
            direction=direction,
            avg_price=avg_price,
            sl_orders=sl_orders,
        )

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
                f"SL=1 TP={len(known_tp_legs)}; "
                f"no changes"
            )

            tracker_tp = _tp_orders_to_tracker(
                tp_orders
            )

            tracker_sl = _sl_order_to_tracker(
                sl_orders
            )

            if tracker_tp and tracker_sl:
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
                            f"RECON_{bx_symbol}_{direction}"
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
            f"protection incomplete: "
            f"SL={len(sl_orders)} "
            f"TP={len(known_tp_legs)}"
            f"/{expected_tp_count}; "
            f"repairing only missing protection..."
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

                prev_close = df1[
                    "close"
                ].shift(1)

                tr = pd.concat(
                    [
                        df1["high"]
                        - df1["low"],
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
                ).max(
                    axis=1
                )

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

            if repaired_tp and repaired_sl:
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
                        f"RECON_{bx_symbol}_{direction}"
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
                    risk_pct=sl_pct,
                )

            if changed:
                send_tg(
                    f"🛡 <b>[ЗАЩИТА ВОССТАНОВЛЕНА] "
                    f"{bx_symbol}</b>\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"Недостающая защита восстановлена:\n"
                    f"• Направление: <b>{direction}</b>\n"
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
    ev: dict,
    df_1h: pd.DataFrame,
) -> dict:
    """
    Open position, wait for exchange-confirmed fill,
    rebuild setup from actual fill, then install protection.
    """
    direction = str(
        direction
    ).upper()

    if direction not in {
        "LONG",
        "SHORT",
    }:
        return {
            "status": "OPEN_INVALID_DIRECTION",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "error": (
                f"invalid direction={direction}"
            ),
        }

    trade_id = event_id.replace(
        "EVT_",
        "",
    )

    signal_price = float(price)

    try:
        opened = open_market(
            symbol,
            direction,
            signal_price,
            trade_id,
        )

    except Exception as exc:
        return {
            "status": "OPEN_EXCEPTION",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "error": str(exc),
            "signal_price": signal_price,
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
            "signal_price": signal_price,
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
            "signal_price": signal_price,
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
            "signal_price": signal_price,
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
            "signal_price": signal_price,
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
        not math.isfinite(
            actual_qty
        )
        or not math.isfinite(
            actual_avg_price
        )
        or actual_qty <= 0
        or actual_avg_price <= 0
    ):
        return {
            "status": "POSITION_INVALID",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
            "position": position,
            "signal_price": signal_price,
        }

    # ---------------------------------------------------------
    # CRITICAL:
    # Build the actual trade setup from exchange-confirmed
    # avgPrice, not from the original signal/reference price.
    # ---------------------------------------------------------
    try:
        actual_setup = build_event_setup(
            ev=ev,
            df_1h=df_1h,
            entry_price=actual_avg_price,
        )

        sl_pct, tp_levels = build_tp_levels(
            actual_setup,
            direction,
        )

    except Exception as exc:
        return {
            "status": "PROTECTION_SETUP_INVALID",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "position": position,
            "open_result": opened,
            "error": str(exc),
            "signal_price": signal_price,
            "actual_avg_price": actual_avg_price,
        }

    protection = install_protection(
        symbol=symbol,
        direction=direction,
        position={
            **position,
            "positionAmt": actual_qty,
            "avgPrice": actual_avg_price,
        },
        setup=actual_setup,
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

    if protection_status == "PROTECTED":
        final_status = (
            "opened_protected"
        )

    elif protection_status == "SL_ONLY":
        final_status = (
            "opened_protection_check_required"
        )

    else:
        final_status = (
            "opened_protection_failed"
        )

    entry_price_deviation_pct = (
        (
            actual_avg_price
            - signal_price
        )
        / signal_price
        * 100.0
        if signal_price > 0
        else None
    )

    actual_setup = {
        **actual_setup,
        "signal_price": signal_price,
        "actual_avg_price": actual_avg_price,
        "entry_price_deviation_pct": (
            entry_price_deviation_pct
        ),
        "realized_rr": None,
    }

    return {
        "status": final_status,
        "mode": EXECUTION_MODE,
        "order_id": order_id,

        "signal_price": signal_price,
        "actual_avg_price": actual_avg_price,
        "entry_price_deviation_pct": (
            entry_price_deviation_pct
        ),

        "open_result": opened,

        "position": {
            **position,
            "positionAmt": actual_qty,
            "avgPrice": actual_avg_price,
        },

        "protection": protection,

        "sl_pct": sl_pct,
        "tp_levels": tp_levels,

        "setup": actual_setup,
    }


def main() -> None:
    # ---------------------------------------------------------
    # 1. Reconciliation / tracker first.
    # ---------------------------------------------------------
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

    stats = {
        "coinalyze_rows": 0,
        "liquidity_candidates": 0,
        "contract_candidates": 0,
        "candidates_scanned": 0,
        "divergence_events": 0,
        "squeeze_events": 0,
        "events_total": 0,
        "rejected_age": 0,
        "rejected_btc": 0,
        "rejected_trigger": 0,
        "rejected_cvd": 0,
        "valid_signals": 0,
        "execution_attempts": 0,
        "trades": 0,
        "scan_errors": 0,
    }

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

    rows = []

    try:
        rows = fetch_data()

    except Exception as exc:
        stats["scan_errors"] += 1

        print(
            "[COINALYZE_SCRAPE_ERROR] "
            f"{exc}"
        )

    stats["coinalyze_rows"] = len(
        rows
    )

    try:
        refresh_contracts()

    except Exception as exc:
        stats["scan_errors"] += 1

        print(
            "[BINGX] contracts refresh "
            f"error={exc}"
        )

    candidates: list[Any] = []

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

        except Exception:
            continue

    if MAX_CANDIDATES > 0:
        candidates = candidates[
            :MAX_CANDIDATES
        ]

    stats[
        "candidates_scanned"
    ] = len(candidates)

    seen_events = load_ids(
        EVENTS
    )

    executed_event_ids = (
        load_successful_trade_ids(
            TRADES
        )
    )

    telegram_sent_event_ids = (
        load_successful_telegram_ids(
            ACTIONS
        )
    )

    opportunities: list[dict] = []

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
                    stats[
                        "rejected_age"
                    ] += 1
                    continue

                if (
                    event_id
                    not in seen_events
                ):
                    emit_event(ev)
                    seen_events.add(
                        event_id
                    )

                if (
                    btc_regime_df
                    is not None
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
                        continue

                    d15 = pd.DataFrame(
                        k15
                    )

                latest_15m_close_ts = int(
                    d15[
                        "close_time"
                    ].iloc[-1]
                )

                trigger_delay_min = (
                    latest_15m_close_ts
                    - detected_at
                ) / 60000.0

                if (
                    trigger_delay_min < 0
                    or trigger_delay_min > MAX_AGE
                ):
                    stats[
                        "rejected_age"
                    ] += 1
                    continue

                if (
                    REQUIRE_TRIGGER
                    and not build_15m_trigger(
                        d15,
                        direction,
                        min_vol_mult=1.05,
                    )
                ):
                    stats[
                        "rejected_trigger"
                    ] += 1
                    continue

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

                fact = ev.get(
                    "event_fact",
                    {},
                )

                price = float(
                    fact.get(
                        "detection_close_price"
                    )
                    or fact.get(
                        "close"
                    )
                    or r.price
                )

                setup = build_event_setup(
                    ev=ev,
                    df_1h=d1,
                    entry_price=price,
                )

                score = calculate_setup_score(
                    ev=ev,
                    coinalyze_row=r,
                    df_15m=d15,
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
                        "df_1h": d1,
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

    stats[
        "valid_signals"
    ] = len(opportunities)

    opportunities.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

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

        d1 = opp[
            "df_1h"
        ]

        # Default output setup is signal/reference setup.
        output_setup = setup

        if event_id in executed_event_ids:
            execution_result = {
                "status": "ALREADY_EXECUTED",
                "mode": EXECUTION_MODE,
                "order_id": None,
            }

        elif (
            EXECUTION_ENABLED
            and trades_this_cycle < MAX_TRADES
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
                    ev=ev,
                    df_1h=d1,
                )
            )

            # Actual-fill setup takes precedence in persistence/output.
            output_setup = (
                execution_result.get(
                    "setup"
                )
                or setup
            )

            record_trade(
                {
                    "event_id": event_id,
                    "symbol": symbol,
                    "direction": direction,

                    # Original signal/reference price.
                    "signal_price": price,

                    # Exchange-confirmed values.
                    "actual_avg_price": (
                        execution_result.get(
                            "actual_avg_price"
                        )
                    ),
                    "actual_qty": (
                        execution_result.get(
                            "position",
                            {},
                        ).get(
                            "positionAmt"
                        )
                    ),

                    "entry_price_deviation_pct": (
                        execution_result.get(
                            "entry_price_deviation_pct"
                        )
                    ),

                    "score": score,
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

                    "setup": output_setup,
                }
            )

            status = str(
                execution_result.get(
                    "status",
                    ""
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
                    position_data = (
                        execution_result.get(
                            "position",
                            {},
                        )
                    )

                    protection_data = (
                        execution_result.get(
                            "protection",
                            {},
                        )
                    )

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

                        # ACTUAL exchange-confirmed entry.
                        entry_price=float(
                            execution_result.get(
                                "actual_avg_price"
                            )
                            or position_data.get(
                                "avgPrice",
                                price,
                            )
                        ),

                        # ACTUAL exchange-confirmed quantity.
                        qty=float(
                            position_data.get(
                                "positionAmt",
                                0,
                            )
                        ),

                        tp_orders=(
                            protection_data.get(
                                "tp_orders",
                                [],
                            )
                        ),

                        sl_result=(
                            protection_data.get(
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

                        risk_pct=(
                            execution_result.get(
                                "sl_pct"
                            )
                        ),

                        signal_price=(
                            execution_result.get(
                                "signal_price",
                                price,
                            )
                        ),

                        entry_price_deviation_pct=(
                            execution_result.get(
                                "entry_price_deviation_pct"
                            )
                        ),
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

        label = (
            "🚨 LONG SIGNAL"
            if direction == "LONG"
            else "🔻 SHORT SIGNAL"
        )

        msg = format_signal(
            ev,
            setup=output_setup,
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
            and not is_real_execution
        )

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

    print(
        f"[ENGINE] "
        f"trades_this_cycle={trades_this_cycle}"
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
