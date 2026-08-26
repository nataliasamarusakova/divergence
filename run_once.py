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
from event_engine.shadow import append_shadow_health


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

POSITION_MODE = os.environ.get(
    "BINGX_POSITION_MODE",
    "HEDGE",
)


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
            value = json.loads(line).get("event_id")
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
    append_jsonl(EVENTS, ev)


def record_trade(obj: dict) -> None:
    append_jsonl(TRADES, obj)


def record_action(obj: dict) -> None:
    append_jsonl(ACTIONS, obj)


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

    entry_price = float(entry_price)

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
        if not isinstance(src, dict):
            continue

        if invalidation is None:
            invalidation = src.get("invalidation_price")

        if target is None:
            target = src.get("target_price")

    if invalidation is not None and target is not None:
        invalidation = float(invalidation)
        target = float(target)

        if direction == "LONG":
            valid_geometry = invalidation < entry_price and target > entry_price
        else:
            valid_geometry = invalidation > entry_price and target < entry_price

        if valid_geometry:
            risk_pct = abs(entry_price - invalidation) / entry_price * 100.0
            reward_pct = abs(target - entry_price) / entry_price * 100.0
            rr = reward_pct / risk_pct if risk_pct > 0 else None

            if rr is not None and rr > 0:
                return {
                    "entry_reference": entry_price,
                    "invalidation_price": invalidation,
                    "target_price": target,
                    "rr": rr,
                    "trigger_ok": True,
                }

    df = df_1h.copy()

    if len(df) < 20:
        raise ValueError("insufficient 1H bars for setup")

    for col in ("high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

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
        tr.rolling(window=14, min_periods=14)
        .mean()
        .iloc[-1]
    )

    if pd.isna(atr) or float(atr) <= 0:
        raise ValueError("ATR unavailable")

    atr = float(atr)
    risk_pct = atr / entry_price * 100.0
    risk_pct = max(0.50, min(risk_pct, 5.00))

    if direction == "LONG":
        invalidation = entry_price * (1.0 - risk_pct / 100.0)
        target = entry_price * (1.0 + 2.0 * risk_pct / 100.0)
    else:
        invalidation = entry_price * (1.0 + risk_pct / 100.0)
        target = entry_price * (1.0 - 2.0 * risk_pct / 100.0)

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
    direction = str(direction).upper()
    entry = float(setup["entry_reference"])
    sl_price = float(setup["invalidation_price"])
    final_tp_price = float(setup["target_price"])

    if direction == "LONG":
        sl_pct = (entry - sl_price) / entry * 100.0
        tp_pct = (final_tp_price - entry) / entry * 100.0
    elif direction == "SHORT":
        sl_pct = (sl_price - entry) / entry * 100.0
        tp_pct = (entry - final_tp_price) / entry * 100.0
    else:
        raise ValueError(f"Invalid direction={direction}")

    if sl_pct <= 0:
        raise ValueError(f"Invalid SL geometry: {sl_price}")

    if tp_pct <= 0:
        raise ValueError(f"Invalid TP geometry: {final_tp_price}")

    tp_levels = [
        {
            "leg": "tp1",
            "pnl_pct": round(tp_pct * 0.50, 6),
            "close_fraction": 0.30,
        },
        {
            "leg": "tp2",
            "pnl_pct": round(tp_pct * 0.75, 6),
            "close_fraction": 0.30,
        },
        {
            "leg": "tp3",
            "pnl_pct": round(tp_pct, 6),
            "close_fraction": 0.40,
        },
    ]

    return sl_pct, tp_levels


def _get_position_avg_price(position: dict) -> float:
    return float(
        position.get("avgPrice", 0)
        or position.get("entryPrice", 0)
        or 0
    )


def _get_position_qty(position: dict) -> float:
    return abs(float(position.get("positionAmt", 0) or 0))


def install_protection(
    symbol: str,
    direction: str,
    position: dict,
    setup: dict,
    sl_pct: float,
    tp_levels: list,
    trade_id: str,
) -> dict:
    avg_price = _get_position_avg_price(position)
    qty = _get_position_qty(position)

    if avg_price <= 0 or qty <= 0:
        return {
            "status": "PROTECTION_INVALID_POSITION",
            "error": f"invalid avgPrice={avg_price} or qty={qty}",
        }

    try:
        signature = inspect.signature(ensure_directional_protection)
        parameters = set(signature.parameters)
    except Exception:
        parameters = set()

    if {"avg_price", "qty", "stop_loss_pct", "tp_levels"}.issubset(parameters):
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

    try:
        return ensure_directional_protection(
            symbol,
            direction,
            avg_price,
            qty,
            sl_pct,
            tp_levels,
            trade_id,
        )
    except Exception as exc:
        return {"status": "PROTECTION_EXCEPTION", "error": str(exc)}


def reconcile_existing_position(
    symbol: str,
    direction: str,
    setup: dict,
    event_id: str,
) -> dict:
    direction = str(direction).upper()
    try:
        position = get_position_directional(symbol=symbol, direction=direction)
    except Exception as exc:
        return {
            "status": "ALREADY_EXECUTED_POSITION_ERROR",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "protection_status": "NOT_VERIFIED",
            "error": str(exc),
        }

    if not isinstance(position, dict) or str(position.get("status", "")).lower() != "found":
        return {
            "status": "ALREADY_EXECUTED_POSITION_NOT_FOUND",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "protection_status": "NOT_VERIFIED",
            "position": position,
        }

    try:
        sl_pct, tp_levels = build_tp_levels(setup, direction)
    except Exception as exc:
        return {
            "status": "ALREADY_EXECUTED",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "position": position,
            "protection_status": "SETUP_INVALID",
            "error": str(exc),
        }

    trade_id = event_id.replace("EVT_", "")
    protection_result = install_protection(
        symbol=symbol,
        direction=direction,
        position=position,
        setup=setup,
        sl_pct=sl_pct,
        tp_levels=tp_levels,
        trade_id=trade_id,
    )

    return {
        "status": "ALREADY_EXECUTED",
        "mode": EXECUTION_MODE,
        "order_id": None,
        "position": position,
        "protection": protection_result,
        "protection_status": str(protection_result.get("status", "UNKNOWN")),
    }


def execute_new_position(
    symbol: str,
    direction: str,
    price: float,
    setup: dict,
    event_id: str,
) -> dict:
    direction = str(direction).upper()
    trade_id = event_id.replace("EVT_", "")

    try:
        opened = open_market(symbol, direction, price, trade_id)
    except Exception as exc:
        return {
            "status": "OPEN_EXCEPTION",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "error": str(exc),
            "exception_type": type(exc).__name__,
        }

    if not isinstance(opened, dict):
        return {
            "status": "OPEN_INVALID_RESPONSE",
            "mode": EXECUTION_MODE,
            "order_id": None,
            "raw": repr(opened),
        }

    open_status = str(opened.get("status", "")).lower()
    if open_status not in {"opened", "success", "ok"}:
        return {
            "status": "OPEN_FAILED",
            "mode": EXECUTION_MODE,
            "order_id": opened.get("order_id"),
            "open_result": opened,
            "error": opened.get("error") or opened.get("msg") or "unknown_open_error",
            "bingx_code": opened.get("code"),
        }

    order_id = opened.get("order_id")

    try:
        position = wait_for_position_fill_directional(
            symbol=symbol,
            direction=direction,
            timeout_sec=15,
            poll_interval=0.5,
        )
    except Exception as exc:
        return {
            "status": "POSITION_WAIT_FAILED",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
            "error": str(exc),
        }

    if not isinstance(position, dict) or str(position.get("status", "")).lower() != "found":
        return {
            "status": "POSITION_NOT_CONFIRMED",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "open_result": opened,
            "position": position,
        }

    try:
        sl_pct, tp_levels = build_tp_levels(setup, direction)
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
        position=position,
        setup=setup,
        sl_pct=sl_pct,
        tp_levels=tp_levels,
        trade_id=trade_id,
    )

    protection_status = str(protection.get("status", "")).lower()
    if protection_status in {"ok", "protected", "created", "reconciled", "success", "ready"}:
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


def build_fallback_signal_message(
    ev: dict,
    symbol: str,
    name: str,
    setup: dict,
    execution_result: dict,
    price: float,
) -> str:
    direction = str(ev.get("direction", "")).upper()
    label = "🚨 LONG SIGNAL" if direction == "LONG" else "🔻 SHORT SIGNAL"
    sl = setup.get("invalidation_price")
    tp = setup.get("target_price")
    rr = setup.get("rr")

    return (
        f"{label}\n\n"
        f"<b>{name or symbol}</b> (<code>{symbol}</code>)\n\n"
        f"Event: <code>{ev.get('event_type')}</code>\n"
        f"TF: <b>{ev.get('timeframe', '1h')}</b> + trigger 15m\n"
        f"Price: <code>{price:.8g}</code>\n"
        f"Detected: <code>{ev.get('timestamps', {}).get('detected_at_ts')}</code>\n\n"
        f"<b>SETUP</b>\n"
        f"Entry: <code>{price:.8g}</code>\n"
        f"SL: <code>{sl:.8g}</code>\n"
        f"TP: <code>{tp:.8g}</code>\n"
        f"R:R: <code>{rr:.2f}</code>\n\n"
        f"<b>EXECUTION</b>\n"
        f"Mode: <code>{EXECUTION_MODE}</code>\n"
        f"Status: <code>{execution_result.get('status')}</code>\n"
        f"Order: <code>{execution_result.get('order_id') or '—'}</code>\n\n"
        f"⚡ Event-driven — 5×5m lifecycle is NOT used"
    )


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

    rows = fetch_data()
    stats["coinalyze_rows"] = len(rows)
    print(f"[ENGINE] Coinalyze rows={len(rows)}")

    try:
        refresh_contracts()
    except Exception as exc:
        stats["scan_errors"] += 1
        print(f"[BINGX] contracts refresh error={exc}")

    candidates: List[Any] = []
    for r in rows:
        try:
            if r.price is None or r.price <= 0:
                continue
            if r.volume24 is None or r.volume24 < MIN_VOL:
                continue
            if r.oi is None or r.oi < MIN_OI:
                continue

            contract = get_contract(r.symbol)
            if not contract:
                stats["bingx_unmapped"] += 1
                continue

            stats["bingx_mapped"] += 1
            candidates.append(r)
        except Exception as exc:
            stats["scan_errors"] += 1
            print(f"[CANDIDATE_ERROR] {getattr(r, 'symbol', '?')}: {exc}")

    stats["candidates_before_limit"] = len(candidates)
    candidates = candidates[:MAX_CANDIDATES]
    stats["candidates"] = len(candidates)

    print(
        f"[ENGINE] Coinalyze candidates={len(candidates)} "
        f"execution={EXECUTION_ENABLED} env={EXECUTION_MODE}"
    )

    seen_events = load_ids(EVENTS)
    executed_event_ids = load_ids(TRADES)
    telegram_sent_event_ids = load_ids(ACTIONS)

    trades_this_cycle = 0

    for r in candidates:
        symbol = r.symbol
        try:
            k1 = fetch_klines(
                symbol,
                "1h",
                int(os.environ.get("KLINE_LIMIT_1H", "250")),
            )
            k15 = fetch_klines(
                symbol,
                "15m",
                int(os.environ.get("KLINE_LIMIT_15M", "250")),
            )

            if len(k1) < 60 or len(k15) < 10:
                continue

            stats["klines_1h_ok"] += 1
            stats["klines_15m_ok"] += 1

            d1 = add_cvd(pd.DataFrame(k1))
            d15 = pd.DataFrame(k15)

            divergence_events = detect_divergences(d1, symbol, "1h")
            squeeze_events = detect_squeeze_release(d1, symbol, "1h", min_squeeze_bars=3)

            rsi_events = [e for e in divergence_events if "_RSI" in e.get("event_type", "")]
            cvd_events = [e for e in divergence_events if "BINGX_CVD" in e.get("event_type", "")]

            stats["rsi_events"] += len(rsi_events)
            stats["cvd_events"] += len(cvd_events)
            stats["squeeze_events"] += len(squeeze_events)
            stats["events_total"] += len(divergence_events) + len(squeeze_events)

            all_events = divergence_events + squeeze_events

            for ev in all_events:
                event_id = ev.get("event_id")
                if not event_id:
                    continue

                direction = str(ev.get("direction", "")).upper()
                if direction not in {"LONG", "SHORT"}:
                    continue

                detected_at = int(ev.get("timestamps", {}).get("detected_at_ts", 0))
                latest_close = int(d1["close_time"].iloc[-1])
                age = (latest_close - detected_at) / 60000.0

                if age < 0 or age > MAX_AGE:
                    continue

                stats["events_recent"] += 1

                if event_id in seen_events:
                    stats["events_duplicate"] += 1
                else:
                    emit_event(ev)
                    seen_events.add(event_id)

                if REQUIRE_CVD and "_RSI" in ev.get("event_type", ""):
                    pivot_1 = ev.get("timestamps", {}).get("pivot_1_ts")
                    pivot_2 = ev.get("timestamps", {}).get("pivot_2_ts")
                    matched_cvd = any(
                        (
                            other.get("direction") == direction
                            and other.get("timestamps", {}).get("pivot_1_ts") == pivot_1
                            and other.get("timestamps", {}).get("pivot_2_ts") == pivot_2
                        )
                        for other in cvd_events
                    )
                    if not matched_cvd:
                        stats["events_cvd_gate_rejected"] += 1
                        continue

                # 15M Trigger Latency Window Check: ensure trigger sync within [-15, 45] minutes
                latest_15m_close_ts = int(d15["close_time"].iloc[-1])
                trigger_delay_min = (latest_15m_close_ts - detected_at) / 60000.0

                if trigger_delay_min < -15 or trigger_delay_min > 45:
                    stats["trigger_rejected"] += 1
                    continue

                trigger = build_15m_trigger(d15, direction)
                if REQUIRE_TRIGGER and not trigger:
                    stats["trigger_rejected"] += 1
                    continue

                stats["trigger_pass"] += 1

                fact = ev.get("event_fact", {})
                price_raw = fact.get("detection_close_price") or fact.get("close") or getattr(r, "price", None)
                if price_raw is None:
                    stats["setup_rejected"] += 1
                    continue

                price = float(price_raw)

                try:
                    setup = build_event_setup(ev=ev, df_1h=d1, entry_price=price)
                except Exception:
                    stats["setup_rejected"] += 1
                    continue

                stats["setups"] += 1

                if event_id in executed_event_ids:
                    if EXECUTION_ENABLED:
                        execution_result = reconcile_existing_position(
                            symbol=symbol,
                            direction=direction,
                            setup=setup,
                            event_id=event_id,
                        )
                    else:
                        execution_result = {
                            "status": "ALREADY_EXECUTED",
                            "mode": EXECUTION_MODE,
                            "order_id": None,
                            "protection_status": "NOT_CHECKED_EXECUTION_DISABLED",
                        }
                elif EXECUTION_ENABLED and trades_this_cycle < MAX_TRADES:
                    stats["execution_attempts"] += 1
                    execution_result = execute_new_position(
                        symbol=symbol,
                        direction=direction,
                        price=price,
                        setup=setup,
                        event_id=event_id,
                    )

                    record_trade({
                        "event_id": event_id,
                        "symbol": symbol,
                        "direction": direction,
                        "price": price,
                        "event_type": ev.get("event_type"),
                        "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                        "result": execution_result,
                        "setup": setup,
                    })

                    status = str(execution_result.get("status", ""))
                    if status in {"opened_protected", "opened_protection_check_required", "opened"}:
                        executed_event_ids.add(event_id)
                        trades_this_cycle += 1
                        stats["trades"] += 1
                elif not EXECUTION_ENABLED:
                    execution_result = {
                        "status": "DISABLED",
                        "mode": EXECUTION_MODE,
                        "order_id": None,
                        "protection_status": "NOT_ATTEMPTED",
                    }
                else:
                    execution_result = {
                        "status": "TRADE_LIMIT_REACHED",
                        "mode": EXECUTION_MODE,
                        "order_id": None,
                    }

                label = "🚨 LONG SIGNAL" if direction == "LONG" else "🔻 SHORT SIGNAL"
                try:
                    msg = format_signal(
                        ev,
                        setup=setup,
                        coinalyze_row=r,
                        execution=execution_result,
                    )
                    if not (msg.startswith("🚨 LONG SIGNAL") or msg.startswith("🔻 SHORT SIGNAL")):
                        msg = f"{label}\n\n" + msg
                except Exception:
                    msg = build_fallback_signal_message(
                        ev=ev,
                        symbol=symbol,
                        name=getattr(r, "name", None) or symbol,
                        setup=setup,
                        execution_result=execution_result,
                        price=price,
                    )

                telegram_already_sent = event_id in telegram_sent_event_ids
                if telegram_already_sent:
                    sent = False
                    stats["telegram_suppressed_duplicate"] += 1
                else:
                    stats["telegram_attempts"] += 1
                    try:
                        sent = bool(send_tg(msg))
                    except Exception:
                        sent = False

                    if sent:
                        stats["telegram_sent"] += 1
                        telegram_sent_event_ids.add(event_id)

                record_action({
                    "event_id": event_id,
                    "symbol": symbol,
                    "direction": direction,
                    "event_type": ev.get("event_type"),
                    "telegram_sent": bool(sent),
                    "telegram_suppressed_duplicate": bool(telegram_already_sent),
                    "execution_status": execution_result.get("status"),
                    "protection_status": execution_result.get("protection_status"),
                    "order_id": execution_result.get("order_id"),
                    "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                })

        except Exception as exc:
            stats["scan_errors"] += 1
            print(f"[SCAN_ERROR] {symbol}: {exc}")

    # =========================================================================
    # SHADOW HEALTH DIAGNOSTICS LOGGING
    # =========================================================================
    try:
        append_shadow_health(
            events_path=EVENTS,
            health_path=HEALTH,
            trades_path=TRADES,
        )
    except Exception as exc:
        print(f"[SHADOW_HEALTH_ERROR] {exc}")

    print(f"[ENGINE] trades_this_cycle={trades_this_cycle}")
    print("[ENGINE_SUMMARY] " + " ".join(f"{k}={v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
