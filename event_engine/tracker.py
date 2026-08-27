from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from event_engine.bingx import (
    get_position_directional,
    get_order,
    get_open_protection_directional,
    cancel_order,
    fetch_klines,
    to_bx_symbol,
    get_contract,
    _format_price,
    _format_qty,
    _request,
    ORDER_PATH,
)
from event_engine.telegram import send as send_tg


log = logging.getLogger("event_engine.tracker")


DATA = Path("data")
ACTIVE_TRADES_PATH = DATA / "active_trades.json"


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)
        if result != result:
            return default
        if result in (float("inf"), float("-inf")):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _load_active_trades() -> dict[str, dict]:
    if not ACTIVE_TRADES_PATH.exists():
        return {}

    try:
        data = json.loads(
            ACTIVE_TRADES_PATH.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(data, dict):
            return {}

        return data

    except Exception as exc:
        log.error(
            "[TRACKER_CORRUPT_STATE] Failed to read %s: %s",
            ACTIVE_TRADES_PATH,
            exc,
        )

        try:
            corrupt_path = ACTIVE_TRADES_PATH.with_suffix(
                f".corrupt.{int(time.time())}.json"
            )

            ACTIVE_TRADES_PATH.rename(
                corrupt_path
            )

            log.warning(
                "[TRACKER_CORRUPT_BACKUP] Saved to %s",
                corrupt_path,
            )

        except Exception:
            pass

        return {}


def _save_active_trades(
    trades: dict[str, dict]
) -> None:
    ACTIVE_TRADES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = ACTIVE_TRADES_PATH.with_name(
        ACTIVE_TRADES_PATH.name + ".tmp"
    )

    payload = json.dumps(
        trades,
        ensure_ascii=False,
        indent=2,
    )

    tmp_path.write_text(
        payload,
        encoding="utf-8",
    )

    os.replace(
        tmp_path,
        ACTIVE_TRADES_PATH,
    )


def update_active_trade_protection(
    symbol: str,
    direction: str,
    tp_orders: list[dict],
    sl_result: dict,
) -> bool:
    """
    Обновляет только exchange-side protection.

    Lifecycle-статистика сделки не сбрасывается.
    """

    trades = _load_active_trades()

    want_symbol = (
        str(symbol)
        .upper()
        .replace("-USDT", "")
    )

    want_direction = str(
        direction
    ).upper()

    for trade in trades.values():
        trade_symbol = (
            str(
                trade.get(
                    "symbol",
                    "",
                )
            )
            .upper()
            .replace(
                "-USDT",
                "",
            )
        )

        trade_direction = str(
            trade.get(
                "direction",
                "",
            )
        ).upper()

        if (
            trade_symbol == want_symbol
            and trade_direction == want_direction
        ):
            trade["tp_orders"] = (
                tp_orders
            )

            trade["sl_order"] = (
                sl_result
            )

            _save_active_trades(
                trades
            )

            return True

    return False


def register_active_trade(
    event_id: str,
    symbol: str,
    name: str,
    direction: str,
    entry_price: float,
    qty: float,
    tp_orders: list[dict],
    sl_result: dict,
    event_type: str,
    coinalyze_row: Any = None,
    score: float = 50.0,
) -> None:
    """
    Регистрирует открытую позицию.

    Важно:
    - entry_price = фактическая exchange avgPrice;
    - qty = фактический exchange positionAmt;
    - planned_weighted_rr хранится отдельно;
    - realized_rr до закрытия сделки неизвестен.
    """

    trades = _load_active_trades()

    now_ms = int(
        time.time() * 1000
    )

    research: dict[str, Any] = {}

    if coinalyze_row is not None:
        research = {
            "fr_oiw": getattr(
                coinalyze_row,
                "fr_oiw",
                None,
            ),
            "pfr_oiw": getattr(
                coinalyze_row,
                "pfr_oiw",
                None,
            ),
            "liq_short24": getattr(
                coinalyze_row,
                "liq_short24",
                None,
            ),
            "liq_long24": getattr(
                coinalyze_row,
                "liq_long24",
                None,
            ),
            "ls_accounts": getattr(
                coinalyze_row,
                "ls_accounts",
                None,
            ),
            "oi": getattr(
                coinalyze_row,
                "oi",
                None,
            ),
            "oi_chg24_pct": getattr(
                coinalyze_row,
                "oi_chg24_pct",
                None,
            ),
            "oi_chg4h_pct": getattr(
                coinalyze_row,
                "oi_chg4h_pct",
                None,
            ),
            "oi_vol_ratio": getattr(
                coinalyze_row,
                "oi_vol_ratio",
                None,
            ),
            "oi_mktcap_ratio": getattr(
                coinalyze_row,
                "oi_mktcap_ratio",
                None,
            ),
            "volume24": getattr(
                coinalyze_row,
                "volume24",
                None,
            ),
            "btc_corr7d": getattr(
                coinalyze_row,
                "btc_corr7d",
                None,
            ),
            "cvd24": getattr(
                coinalyze_row,
                "cvd24",
                None,
            ),
            "lls24": getattr(
                coinalyze_row,
                "lls24",
                None,
            ),
        }

    planned_weighted_rr = 1.55

    trades[event_id] = {
        "event_id": event_id,
        "symbol": symbol,
        "name": name or symbol,
        "direction": str(
            direction
        ).upper(),

        "entry_price": _safe_float(
            entry_price
        ),

        "initial_qty": abs(
            _safe_float(qty)
        ),

        "remaining_qty": abs(
            _safe_float(qty)
        ),

        "entry_ts": now_ms,

        "tp_orders": (
            tp_orders
            if isinstance(
                tp_orders,
                list,
            )
            else []
        ),

        "sl_order": (
            sl_result
            if isinstance(
                sl_result,
                dict,
            )
            else {}
        ),

        "hit_legs": [],
        "be_activated": False,

        "peak_pnl_pct": 0.0,
        "max_drawdown_pct": 0.0,

        "score": _safe_float(
            score,
            50.0,
        ),

        "event_type": event_type,

        "research": research,

        # Incremental TP execution accounting.
        "tp_filled_qty": {},

        # Qty-weighted realized PnL in percentage points.
        "realized_pnl_qty": 0.0,
        "realized_pnl_weighted_sum": 0.0,

        "last_tp_exec_price": None,

        # Planned setup metrics.
        "planned_target_rr": 2.0,
        "planned_weighted_rr": planned_weighted_rr,

        # Unknown until close.
        "realized_rr": None,
        "realized_pnl_pct": None,

        # Final close information.
        "exit_price": None,
        "exit_reason": None,
        "closed_ts": None,
        "duration_min": None,

        # Actual execution information.
        "actual_entry_price": _safe_float(
            entry_price
        ),
        "requested_entry_price": None,
        "entry_slippage_pct": None,
        "adverse_entry_slippage_pct": None,
    }

    _save_active_trades(
        trades
    )


def format_tp_hit_message(
    name: str,
    symbol: str,
    leg: str,
    pnl_pct: float,
    exec_price: float,
    closed_qty: float,
    remaining_qty: float,
    remaining_pct: float,
) -> str:
    return (
        f"💰 <b>{name} ({symbol})</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Leg: <b>{leg}</b>\n"
        f"PnL TP: <b>+{pnl_pct:.2f}%</b>\n"
        f"Цена исполнения: "
        f"<code>{exec_price:.8g}</code>\n"
        f"Закрыто: "
        f"<code>{closed_qty:.6f}</code>\n"
        f"Осталось: "
        f"<code>{remaining_qty:.6f} "
        f"({remaining_pct:.1f}%)</code>"
    )


def format_be_message(
    name: str,
    symbol: str,
    entry_price: float,
) -> str:
    return (
        f"🛡 <b>{name} ({symbol}) — БЕЗУБЫТОК</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"TP1 взят! Стоп-лосс перенесен "
        f"на точку входа: "
        f"<code>{entry_price:.8g}</code>\n"
        f"Текущий риск по сделке: "
        f"<b>0.00%</b>"
    )


def format_trade_closed_message(
    name: str,
    symbol: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    realized_rr: float | None,
    planned_rr: float | None,
    duration_min: float,
    peak_pnl: float,
    max_drawdown: float,
    exit_reason: str,
    event_type: str,
    research: dict,
) -> str:
    is_win = pnl_pct >= 0

    emoji = (
        "💚"
        if is_win
        else "💔"
    )

    pnl_sign = (
        "+"
        if pnl_pct > 0
        else ""
    )

    rr_text = (
        f"{realized_rr:.3f}"
        if realized_rr is not None
        else "—"
    )

    planned_rr_text = (
        f"{planned_rr:.3f}"
        if planned_rr is not None
        else "—"
    )

    lines = [
        f"{emoji} <b>{name} ({symbol}) — "
        f"сделка закрыта</b>",
        "━━━━━━━━━━━━━━━━━━",
        (
            f"Вход <code>{entry_price:.8g}</code> "
            f"→ Выход "
            f"<code>{exit_price:.8g}</code>   "
            f"<b>{pnl_sign}{pnl_pct:.2f}%</b>"
        ),
        (
            f"Realized R:R: "
            f"<b>{rr_text}</b> · "
            f"Planned Weighted R:R: "
            f"<b>{planned_rr_text}</b>"
        ),
        (
            f"Держали <b>{duration_min:.1f} мин</b> · "
            f"пик <b>+{peak_pnl:.1f}%</b> · "
            f"просадка "
            f"<b>{max_drawdown:.1f}%</b>"
        ),
        f"Выход по: <b>{exit_reason}</b>",
        (
            f"Вход был: "
            f"<code>{event_type}</code> · "
            f"TF <b>1h</b>"
        ),
    ]

    if research:
        lines.append(
            "━━━━━━━━━━━━━━━━━━"
        )

        lines.append(
            "📊 Research"
        )

        res_parts = []

        if research.get(
            "fr_oiw"
        ) is not None:
            res_parts.append(
                "FR·OI: "
                f"<code>{research['fr_oiw']}</code>"
            )

        if research.get(
            "ls_accounts"
        ) is not None:
            res_parts.append(
                "L/S: "
                f"<code>{research['ls_accounts']}</code>"
            )

        if research.get(
            "liq_short24"
        ) is not None:
            res_parts.append(
                "LiqShort: "
                f"<code>{research['liq_short24']}</code>"
            )

        if research.get(
            "liq_long24"
        ) is not None:
            res_parts.append(
                "LiqLong: "
                f"<code>{research['liq_long24']}</code>"
            )

        if research.get(
            "cvd24"
        ) is not None:
            res_parts.append(
                "CVD24: "
                f"<code>{research['cvd24']}</code>"
            )

        if research.get(
            "lls24"
        ) is not None:
            res_parts.append(
                "LLS24: "
                f"<code>{research['lls24']}</code>"
            )

        lines.append(
            " · ".join(
                res_parts
            )
            if res_parts
            else "—"
        )

    return "\n".join(
        lines
    )


def _move_sl_to_break_even(
    symbol: str,
    direction: str,
    entry_price: float,
    qty: float,
    old_sl_id: str | None,
    trade_id: str | None = None,
) -> dict:
    """
    Создаёт BE stop, проверяет его наличие на бирже,
    и только после проверки удаляет старый SL.
    """

    bx = to_bx_symbol(
        symbol
    )

    contract = (
        get_contract(symbol)
        or {}
    )

    try:
        precision = int(
            contract.get(
                "quantityPrecision"
            )
            or 0
        )

        price_precision = int(
            contract.get(
                "pricePrecision"
            )
            or 4
        )

    except (
        TypeError,
        ValueError,
    ):
        return {
            "status": "error",
            "error": "invalid contract precision",
            "order_id": "",
            "stop_price": entry_price,
        }

    if (
        qty <= 0
        or entry_price <= 0
        or not bx
    ):
        return {
            "status": "error",
            "error": "invalid BE parameters",
            "order_id": "",
            "stop_price": entry_price,
        }

    trade_token = (
        str(trade_id)
        if trade_id
        else uuid.uuid4().hex.upper()[
            :16
        ]
    )

    be_client_id = (
        f"EVT_BE_{trade_token}"
    )

    # First check whether an identical BE already exists.
    verified = (
        get_open_protection_directional(
            symbol,
            direction,
        )
    )

    if verified.get(
        "status"
    ) == "ok":

        for order in verified.get(
            "sl_orders",
            [],
        ):
            cid = str(
                order.get(
                    "clientOrderId",
                    "",
                )
            ).upper()

            order_price = _safe_float(
                order.get(
                    "stopPrice"
                )
                or order.get(
                    "price"
                )
            )

            price_matches = (
                order_price > 0
                and abs(
                    order_price
                    - entry_price
                )
                / max(
                    entry_price,
                    1e-12,
                )
                < 0.002
            )

            if (
                be_client_id.upper()
                in cid
                or price_matches
            ):
                existing_id = str(
                    order.get(
                        "orderId",
                        "",
                    )
                )

                if (
                    old_sl_id
                    and str(old_sl_id)
                    != existing_id
                ):
                    cancel_order(
                        symbol,
                        old_sl_id,
                    )

                return {
                    "status": "created",
                    "order_id": existing_id,
                    "client_order_id": (
                        cid
                        or be_client_id
                    ),
                    "stop_price": entry_price,
                }

    sl_side = (
        "SELL"
        if direction.upper()
        == "LONG"
        else "BUY"
    )

    params = {
        "symbol": bx,
        "side": sl_side,
        "positionSide": (
            direction.upper()
        ),
        "type": "STOP_MARKET",
        "stopPrice": _format_price(
            entry_price,
            price_precision,
        ),
        "quantity": _format_qty(
            qty,
            precision,
        ),
        "clientOrderId": be_client_id,
    }

    try:
        resp = _request(
            "POST",
            ORDER_PATH,
            params,
        )

    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "order_id": "",
            "stop_price": entry_price,
        }

    if (
        not isinstance(
            resp,
            dict,
        )
        or resp.get(
            "code"
        )
        != 0
    ):
        return {
            "status": "error",
            "error": (
                "BE stop failed: "
                f"code="
                f"{resp.get('code') if isinstance(resp, dict) else None} "
                f"msg="
                f"{resp.get('msg') if isinstance(resp, dict) else resp}"
            ),
            "order_id": "",
            "stop_price": entry_price,
        }

    order = (
        (resp.get("data") or {})
        .get("order")
        or resp.get("data")
        or {}
    )

    new_order_id = str(
        order.get(
            "orderId",
            "",
        )
    )

    if not new_order_id:
        return {
            "status": "error",
            "error": (
                "BE stop response "
                "has no orderId"
            ),
            "order_id": "",
            "stop_price": entry_price,
        }

    verified_after = (
        get_open_protection_directional(
            symbol,
            direction,
        )
    )

    if (
        verified_after.get(
            "status"
        )
        != "ok"
    ):
        return {
            "status": "error",
            "error": (
                verified_after.get(
                    "error",
                    "BE stop verification failed",
                )
            ),
            "order_id": new_order_id,
            "stop_price": entry_price,
        }

    found = any(
        str(
            o.get(
                "orderId",
                "",
            )
        )
        == new_order_id
        for o in verified_after.get(
            "sl_orders",
            [],
        )
    )

    if not found:
        return {
            "status": "error",
            "error": (
                "BE stop was created but "
                "is not visible on exchange; "
                "old SL was kept"
            ),
            "order_id": new_order_id,
            "stop_price": entry_price,
        }

    old_cancel = None

    if (
        old_sl_id
        and str(old_sl_id)
        != new_order_id
    ):
        old_cancel = cancel_order(
            symbol,
            old_sl_id,
        )

    result = {
        "status": (
            "created"
            if (
                not isinstance(
                    old_cancel,
                    dict,
                )
                or old_cancel.get(
                    "code"
                )
                in (None, 0)
            )
            else "created_old_sl_cancel_failed"
        ),
        "order_id": new_order_id,
        "client_order_id": (
            order.get(
                "clientOrderId"
            )
            or be_client_id
        ),
        "stop_price": entry_price,
    }

    if (
        isinstance(
            old_cancel,
            dict,
        )
        and old_cancel.get(
            "code"
        )
        != 0
    ):
        result[
            "old_sl_cancel_error"
        ] = (
            old_cancel.get(
                "msg"
            )
            or old_cancel
        )

    return result


def _calc_trade_pnl_pct(
    entry_price: float,
    exit_price: float,
    direction: str,
) -> float:
    if (
        entry_price <= 0
        or exit_price <= 0
    ):
        return 0.0

    if (
        str(direction).upper()
        == "LONG"
    ):
        return (
            (
                exit_price
                - entry_price
            )
            / entry_price
            * 100.0
        )

    return (
        (
            entry_price
            - exit_price
        )
        / entry_price
        * 100.0
    )


def _calc_realized_rr(
    realized_pnl_pct: float,
    planned_risk_pct: float | None,
) -> float | None:
    """
    Realized R:R = actual realized return / planned initial risk.

    planned_risk_pct должен быть процентом SL
    относительно actual entry.
    """

    if (
        planned_risk_pct is None
        or planned_risk_pct <= 0
    ):
        return None

    return (
        realized_pnl_pct
        / planned_risk_pct
    )


def _update_mfe_mae(
    trade: dict,
    candles: list[dict],
) -> None:
    if not candles:
        return

    entry_price = _safe_float(
        trade.get(
            "entry_price"
        )
    )

    direction = str(
        trade.get(
            "direction",
            "LONG",
        )
    ).upper()

    if entry_price <= 0:
        return

    peak = _safe_float(
        trade.get(
            "peak_pnl_pct",
            0.0,
        )
    )

    max_dd = _safe_float(
        trade.get(
            "max_drawdown_pct",
            0.0,
        )
    )

    for candle in candles:
        high = _safe_float(
            candle.get("high")
        )

        low = _safe_float(
            candle.get("low")
        )

        if (
            high <= 0
            or low <= 0
        ):
            continue

        if direction == "LONG":
            favorable = (
                (
                    high
                    - entry_price
                )
                / entry_price
                * 100.0
            )

            adverse = (
                (
                    low
                    - entry_price
                )
                / entry_price
                * 100.0
            )

        else:
            favorable = (
                (
                    entry_price
                    - low
                )
                / entry_price
                * 100.0
            )

            adverse = (
                (
                    entry_price
                    - high
                )
                / entry_price
                * 100.0
            )

        peak = max(
            peak,
            favorable,
        )

        drawdown = (
            adverse
            - peak
        )

        max_dd = min(
            max_dd,
            drawdown,
        )

    trade[
        "peak_pnl_pct"
    ] = peak

    trade[
        "max_drawdown_pct"
    ] = max_dd


def update_active_trades() -> None:
    """
    Полностью сопровождает активные сделки.

    Гарантии:

    1. Ошибка API при чтении позиции не считается закрытием.
    2. TP учитывается инкрементально.
    3. PARTIALLY_FILLED не приводит к двойному учёту.
    4. BE создаётся и проверяется до удаления старого SL.
    5. Итоговый PnL считается по фактическим execution prices.
    6. Реальный R:R появляется только после закрытия.
    """

    trades = _load_active_trades()

    if not trades:
        return

    now_ms = int(
        time.time() * 1000
    )

    updated_trades: dict[
        str,
        dict,
    ] = {}

    for event_id, trade in trades.items():
        try:
            symbol = str(
                trade.get(
                    "symbol",
                    "",
                )
            )

            direction = str(
                trade.get(
                    "direction",
                    "LONG",
                )
            ).upper()

            entry_price = _safe_float(
                trade.get(
                    "entry_price"
                )
            )

            init_qty = abs(
                _safe_float(
                    trade.get(
                        "initial_qty"
                    )
                )
            )

            rem_qty = max(
                0.0,
                _safe_float(
                    trade.get(
                        "remaining_qty"
                    )
                )
            )

            entry_ts = int(
                _safe_float(
                    trade.get(
                        "entry_ts"
                    )
                )
            )

            if (
                not symbol
                or direction
                not in {
                    "LONG",
                    "SHORT",
                }
                or entry_price <= 0
                or init_qty <= 0
            ):
                log.error(
                    "[TRACKER_INVALID_TRADE] "
                    "event_id=%s symbol=%s",
                    event_id,
                    symbol,
                )

                updated_trades[
                    event_id
                ] = trade

                continue

            hit_legs = set(
                trade.get(
                    "hit_legs",
                    [],
                )
            )

            filled_by_leg = {
                str(k): max(
                    0.0,
                    _safe_float(v),
                )
                for k, v in (
                    trade.get(
                        "tp_filled_qty",
                        {}
                    )
                    or {}
                ).items()
            }

            realized_qty = max(
                0.0,
                _safe_float(
                    trade.get(
                        "realized_pnl_qty",
                        0.0,
                    )
                )
            )

            realized_weighted = (
                _safe_float(
                    trade.get(
                        "realized_pnl_weighted_sum",
                        0.0,
                    )
                )
            )

            # ---------------------------------------------------------
            # POSITION SOURCE OF TRUTH
            # ---------------------------------------------------------

            pos = get_position_directional(
                symbol,
                direction,
            )

            pos_status = str(
                pos.get(
                    "status",
                    "",
                )
            ).lower()

            if pos_status not in {
                "found",
                "not_found",
            }:
                log.warning(
                    "[TRACKER_POSITION_UNCERTAIN] "
                    "%s %s: %s",
                    symbol,
                    direction,
                    pos.get(
                        "error"
                    )
                    or pos_status,
                )

                updated_trades[
                    event_id
                ] = trade

                continue

            pos_amt = (
                abs(
                    _safe_float(
                        pos.get(
                            "positionAmt"
                        )
                    )
                )
                if pos_status
                == "found"
                else 0.0
            )

            # ---------------------------------------------------------
            # CURRENT PRICE / MFE / MAE
            # ---------------------------------------------------------

            cur_price = entry_price

            try:
                k1m = fetch_klines(
                    symbol,
                    "1m",
                    limit=60,
                )

                if k1m:
                    cur_price = _safe_float(
                        k1m[-1].get(
                            "close"
                        ),
                        entry_price,
                    )

                    _update_mfe_mae(
                        trade,
                        k1m,
                    )

            except Exception as exc:
                log.warning(
                    "[TRACKER_KLINE_ERROR] "
                    "%s: %s",
                    symbol,
                    exc,
                )

            current_pnl = (
                _calc_trade_pnl_pct(
                    entry_price,
                    cur_price,
                    direction,
                )
            )

            trade[
                "current_pnl_pct"
            ] = current_pnl

            trade[
                "current_position_qty"
            ] = pos_amt

            trade[
                "last_observation_ts"
            ] = now_ms

            # ---------------------------------------------------------
            # TP FILL PROCESSING
            # ---------------------------------------------------------

            for tp in trade.get(
                "tp_orders",
                [],
            ):
                leg = str(
                    tp.get(
                        "leg",
                        "",
                    )
                )

                order_id = tp.get(
                    "order_id"
                )

                if (
                    not leg
                    or not order_id
                ):
                    continue

                order_info = get_order(
                    symbol,
                    order_id,
                )

                if (
                    order_info.get(
                        "status"
                    )
                    == "error"
                ):
                    continue

                order_status = str(
                    order_info.get(
                        "order_status",
                        "",
                    )
                ).upper()

                if order_status not in {
                    "PARTIALLY_FILLED",
                    "FILLED",
                }:
                    continue

                executed_qty = max(
                    0.0,
                    _safe_float(
                        order_info.get(
                            "executed_qty",
                            0.0,
                        )
                    )
                )

                previous_qty = max(
                    0.0,
                    _safe_float(
                        filled_by_leg.get(
                            leg,
                            0.0,
                        )
                    )
                )

                delta_qty = max(
                    0.0,
                    executed_qty
                    - previous_qty,
                )

                if (
                    delta_qty <= 0
                ):
                    if (
                        order_status
                        == "FILLED"
                    ):
                        hit_legs.add(
                            leg
                        )

                    continue

                exec_price = _safe_float(
                    order_info.get(
                        "avg_price"
                    )
                    or tp.get(
                        "price"
                    )
                    or cur_price,
                    cur_price,
                )

                if exec_price <= 0:
                    log.warning(
                        "[TRACKER_INVALID_EXEC_PRICE] "
                        "%s %s order=%s",
                        symbol,
                        leg,
                        order_id,
                    )
                    continue

                pnl_tp = _safe_float(
                    tp.get(
                        "pnl_pct"
                    )
                )

                if pnl_tp <= 0:
                    pnl_tp = (
                        _calc_trade_pnl_pct(
                            entry_price,
                            exec_price,
                            direction,
                        )
                    )

                # IMPORTANT:
                # only delta_qty is booked, so repeated polling cannot
                # double-count already executed quantity.
                rem_qty = max(
                    0.0,
                    rem_qty
                    - delta_qty,
                )

                realized_qty += (
                    delta_qty
                )

                realized_weighted += (
                    delta_qty
                    * pnl_tp
                )

                filled_by_leg[
                    leg
                ] = executed_qty

                trade[
                    "remaining_qty"
                ] = rem_qty

                trade[
                    "realized_pnl_qty"
                ] = realized_qty

                trade[
                    "realized_pnl_weighted_sum"
                ] = realized_weighted

                trade[
                    "last_tp_exec_price"
                ] = exec_price

                # Record every execution event.
                tp_exec_history = trade.setdefault(
                    "tp_execution_history",
                    [],
                )

                already_recorded = any(
                    (
                        str(
                            item.get(
                                "order_id",
                                "",
                            )
                        )
                        == str(
                            order_id
                        )
                        and abs(
                            _safe_float(
                                item.get(
                                    "executed_qty_total",
                                    0,
                                )
                            )
                            - executed_qty
                        )
                        < 1e-12
                    )
                    for item in tp_exec_history
                )

                if not already_recorded:
                    tp_exec_history.append(
                        {
                            "ts": now_ms,
                            "leg": leg,
                            "order_id": str(
                                order_id
                            ),
                            "executed_qty_total": executed_qty,
                            "delta_qty": delta_qty,
                            "exec_price": exec_price,
                            "pnl_pct": pnl_tp,
                            "order_status": order_status,
                        }
                    )

                # -----------------------------------------------------
                # FULL TP LEG
                # -----------------------------------------------------

                if (
                    order_status
                    == "FILLED"
                    and leg
                    not in hit_legs
                ):
                    hit_legs.add(
                        leg
                    )

                    rem_pct = (
                        rem_qty
                        / init_qty
                        * 100.0
                        if init_qty > 0
                        else 0.0
                    )

                    send_tg(
                        format_tp_hit_message(
                            name=trade.get(
                                "name",
                                symbol,
                            ),
                            symbol=symbol,
                            leg=leg,
                            pnl_pct=pnl_tp,
                            exec_price=exec_price,
                            closed_qty=delta_qty,
                            remaining_qty=rem_qty,
                            remaining_pct=rem_pct,
                        )
                    )

                    # Move SL to BE only after TP1 FULL fill.
                    if (
                        leg == "tp1"
                        and not trade.get(
                            "be_activated"
                        )
                        and rem_qty > 0
                    ):
                        old_sl_id = (
                            trade.get(
                                "sl_order",
                                {}
                            )
                            .get(
                                "order_id"
                            )
                        )

                        new_sl = (
                            _move_sl_to_break_even(
                                symbol=symbol,
                                direction=direction,
                                entry_price=entry_price,
                                qty=rem_qty,
                                old_sl_id=old_sl_id,
                                trade_id=(
                                    str(
                                        event_id
                                    ).replace(
                                        "EVT_",
                                        "",
                                    )
                                ),
                            )
                        )

                        if new_sl.get(
                            "status"
                        ) in {
                            "created",
                            "created_old_sl_cancel_failed",
                        }:
                            trade[
                                "sl_order"
                            ] = new_sl

                            trade[
                                "be_activated"
                            ] = True

                            trade[
                                "be_activation_ts"
                            ] = now_ms

                            send_tg(
                                format_be_message(
                                    trade.get(
                                        "name",
                                        symbol,
                                    ),
                                    symbol,
                                    entry_price,
                                )
                            )

                        else:
                            log.error(
                                "[TRACKER_BE_FAILED] "
                                "%s %s: %s",
                                symbol,
                                direction,
                                new_sl.get(
                                    "error"
                                ),
                            )

            trade[
                "hit_legs"
            ] = sorted(
                hit_legs
            )

            trade[
                "tp_filled_qty"
            ] = filled_by_leg

            # ---------------------------------------------------------
            # DETERMINE WHETHER POSITION IS CLOSED
            # ---------------------------------------------------------

            closed_by_tp = (
                rem_qty <= 1e-12
                and realized_qty > 0
            )

            position_gone = (
                pos_status
                == "not_found"
            )

            if (
                not position_gone
                and not closed_by_tp
            ):
                updated_trades[
                    event_id
                ] = trade

                continue

            # ---------------------------------------------------------
            # DETERMINE EXIT
            # ---------------------------------------------------------

            duration_min = (
                now_ms
                - entry_ts
            ) / 60000.0

            exit_price = _safe_float(
                trade.get(
                    "last_tp_exec_price"
                ),
                cur_price,
            )

            sl_order_id = (
                trade.get(
                    "sl_order",
                    {}
                ).get(
                    "order_id"
                )
            )

            sl_info = (
                get_order(
                    symbol,
                    sl_order_id,
                )
                if sl_order_id
                else {}
            )

            if (
                sl_info.get(
                    "status"
                )
                == "ok"
                and str(
                    sl_info.get(
                        "order_status",
                        "",
                    )
                ).upper()
                == "FILLED"
            ):
                exit_price = _safe_float(
                    sl_info.get(
                        "avg_price"
                    ),
                    exit_price,
                )

                if (
                    trade.get(
                        "be_activated"
                    )
                    and entry_price > 0
                    and abs(
                        exit_price
                        - entry_price
                    )
                    / entry_price
                    < 0.003
                ):
                    exit_reason = (
                        "BREAK_EVEN"
                    )
                else:
                    exit_reason = (
                        "STOP_LOSS"
                    )

            elif (
                closed_by_tp
                and hit_legs
            ):
                exit_reason = (
                    "TAKE_PROFIT_FULL"
                )

            elif position_gone:
                exit_reason = (
                    "POSITION_CLOSED"
                )

            else:
                exit_reason = (
                    "POSITION_CLOSED"
                )

            if exit_price <= 0:
                exit_price = cur_price

            # ---------------------------------------------------------
            # ACCOUNT FOR REMAINING POSITION
            # ---------------------------------------------------------

            if (
                position_gone
                and rem_qty > 0
                and init_qty > 0
            ):
                exit_pnl = (
                    _calc_trade_pnl_pct(
                        entry_price,
                        exit_price,
                        direction,
                    )
                )

                realized_weighted += (
                    rem_qty
                    * exit_pnl
                )

                realized_qty += (
                    rem_qty
                )

                rem_qty = 0.0

                trade[
                    "remaining_qty"
                ] = 0.0

                trade[
                    "realized_pnl_qty"
                ] = realized_qty

                trade[
                    "realized_pnl_weighted_sum"
                ] = realized_weighted

                trade[
                    "last_close_exec_price"
                ] = exit_price

            # ---------------------------------------------------------
            # FINAL REALIZED PNL
            # ---------------------------------------------------------

            if (
                init_qty > 0
                and realized_qty > 0
            ):
                final_pnl = (
                    realized_weighted
                    / init_qty
                )
            else:
                final_pnl = current_pnl

            # Determine planned initial risk.
            planned_risk_pct = None

            stored_setup = (
                trade.get(
                    "setup",
                    {}
                )
            )

            if isinstance(
                stored_setup,
                dict,
            ):
                planned_risk_pct = (
                    _safe_float(
                        stored_setup.get(
                            "risk_pct"
                        ),
                        0.0,
                    )
                )

            if (
                not planned_risk_pct
                or planned_risk_pct <= 0
            ):
                sl_order = trade.get(
                    "sl_order",
                    {}
                )

                sl_price = _safe_float(
                    sl_order.get(
                        "stop_price"
                    )
                    if isinstance(
                        sl_order,
                        dict,
                    )
                    else 0,
                    0.0,
                )

                if sl_price > 0:
                    if direction == "LONG":
                        planned_risk_pct = (
                            (
                                entry_price
                                - sl_price
                            )
                            / entry_price
                            * 100.0
                        )
                    else:
                        planned_risk_pct = (
                            (
                                sl_price
                                - entry_price
                            )
                            / entry_price
                            * 100.0
                        )

            realized_rr = (
                _calc_realized_rr(
                    final_pnl,
                    planned_risk_pct,
                )
            )

            planned_rr = _safe_float(
                trade.get(
                    "planned_weighted_rr",
                    1.55,
                ),
                1.55,
            )

            # ---------------------------------------------------------
            # FINAL STATE
            # ---------------------------------------------------------

            trade[
                "remaining_qty"
            ] = 0.0

            trade[
                "realized_pnl_pct"
            ] = final_pnl

            trade[
                "realized_rr"
            ] = realized_rr

            trade[
                "exit_price"
            ] = exit_price

            trade[
                "exit_reason"
            ] = exit_reason

            trade[
                "closed_ts"
            ] = now_ms

            trade[
                "duration_min"
            ] = duration_min

            trade[
                "planned_risk_pct"
            ] = planned_risk_pct

            trade[
                "closed"
            ] = True

            trade[
                "close_metrics"
            ] = {
                "realized_pnl_pct": final_pnl,
                "realized_rr": realized_rr,
                "planned_weighted_rr": planned_rr,
                "realized_qty": realized_qty,
                "initial_qty": init_qty,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "duration_min": duration_min,
                "peak_pnl_pct": _safe_float(
                    trade.get(
                        "peak_pnl_pct"
                    )
                ),
                "max_drawdown_pct": _safe_float(
                    trade.get(
                        "max_drawdown_pct"
                    )
                ),
            }

            # ---------------------------------------------------------
            # TELEGRAM
            # ---------------------------------------------------------

            send_tg(
                format_trade_closed_message(
                    name=trade.get(
                        "name",
                        symbol,
                    ),
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pct=final_pnl,
                    realized_rr=realized_rr,
                    planned_rr=planned_rr,
                    duration_min=duration_min,
                    peak_pnl=_safe_float(
                        trade.get(
                            "peak_pnl_pct"
                        )
                    ),
                    max_drawdown=_safe_float(
                        trade.get(
                            "max_drawdown_pct"
                        )
                    ),
                    exit_reason=exit_reason,
                    event_type=trade.get(
                        "event_type",
                        "DIVERGENCE",
                    ),
                    research=trade.get(
                        "research",
                        {},
                    ),
                )
            )

            # ---------------------------------------------------------
            # CANCEL REMAINING ORDERS
            # ---------------------------------------------------------

            for tp in trade.get(
                "tp_orders",
                [],
            ):
                leg = tp.get(
                    "leg"
                )

                if (
                    leg not in hit_legs
                    and tp.get(
                        "order_id"
                    )
                ):
                    try:
                        cancel_order(
                            symbol,
                            tp[
                                "order_id"
                            ],
                        )
                    except Exception as exc:
                        log.warning(
                            "[TRACKER_CANCEL_TP_ERROR] "
                            "%s %s: %s",
                            symbol,
                            leg,
                            exc,
                        )

            if sl_order_id:
                try:
                    cancel_order(
                        symbol,
                        sl_order_id,
                    )
                except Exception as exc:
                    log.warning(
                        "[TRACKER_CANCEL_SL_ERROR] "
                        "%s: %s",
                        symbol,
                        exc,
                    )

            # Closed trades are intentionally removed
            # from active_trades.json after their final
            # statistics have been calculated.
            log.info(
                "[TRACKER_CLOSED] "
                "%s %s pnl=%.4f%% "
                "realized_rr=%s "
                "reason=%s",
                symbol,
                direction,
                final_pnl,
                (
                    f"{realized_rr:.4f}"
                    if realized_rr is not None
                    else "n/a"
                ),
                exit_reason,
            )

        except Exception as exc:
            log.exception(
                "[TRACKER_FATAL_TRADE_ERROR] "
                "event_id=%s symbol=%s: %s",
                event_id,
                trade.get(
                    "symbol",
                    "",
                ),
                exc,
            )

            updated_trades[
                event_id
            ] = trade

    _save_active_trades(
        updated_trades
    )
