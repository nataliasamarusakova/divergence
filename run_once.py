# run_once.py

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
    to_bx_symbol,
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
    _load_active_trades,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
log = logging.getLogger("event_engine")

DATA = Path("data")
DATA.mkdir(exist_ok=True)

EVENTS = DATA / "events.jsonl"
TRADES = DATA / "trades.jsonl"
ACTIONS = DATA / "actions.jsonl"
HEALTH = DATA / "health.jsonl"

MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "0"))
MIN_VOL = float(os.environ.get("MIN_VOLUME_24H", "50000000"))
MIN_OI = float(os.environ.get("MIN_OPEN_INTEREST", "5000000"))
EXECUTION_ENABLED = os.environ.get("EXECUTION_ENABLED", "false").lower() == "true"
REQUIRE_CVD = os.environ.get("REQUIRE_CVD_CONFIRMATION", "false").lower() == "true"
CVD_MIN_CONFIRMATION = float(os.environ.get("MIN_CVD24_CONFIRMATION", "55"))
REQUIRE_TRIGGER = os.environ.get("REQUIRE_15M_TRIGGER", "true").lower() == "true"
MAX_AGE = int(os.environ.get("MAX_EVENT_AGE_MIN", "90"))
MAX_TRADES = int(os.environ.get("MAX_TRADES_PER_CYCLE", "3"))
EXECUTION_MODE = os.environ.get("EXECUTION_MODE", os.environ.get("BINGX_ENV", "vst"))


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
        result = obj.get("result", {})
        if not isinstance(result, dict):
            continue
        status = str(result.get("status", "")).lower()
        if status in {"opened_protected", "opened_protection_check_required", "opened", "already_executed"}:
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


def calculate_execution_slippage(signal_price: float, actual_entry_price: float, direction: str) -> dict:
    try:
        signal_price = float(signal_price)
        actual_entry_price = float(actual_entry_price)
    except (TypeError, ValueError):
        return {"slippage_pct": None, "adverse_slippage_pct": None}

    direction = str(direction).upper()
    if signal_price <= 0 or actual_entry_price <= 0:
        return {"slippage_pct": None, "adverse_slippage_pct": None}

    raw_pct = (actual_entry_price - signal_price) / signal_price * 100.0
    if direction == "LONG":
        adverse = max(0.0, raw_pct)
    elif direction == "SHORT":
        adverse = max(0.0, -raw_pct)
    else:
        adverse = None

    return {"slippage_pct": raw_pct, "adverse_slippage_pct": adverse}


def calculate_setup_score(ev: dict, coinalyze_row: Any, df_15m: pd.DataFrame) -> float:
    score = 50.0
    fact = ev.get("event_fact", {})
    direction = str(ev.get("direction", "LONG")).upper()

    try:
        delta_atr = float(fact.get("price_delta_atr", 0))
    except (TypeError, ValueError):
        delta_atr = 0.0

    if delta_atr >= 1.0:
        score += 15.0
    elif delta_atr >= 0.5:
        score += 10.0

    if "CVD" in str(ev.get("event_type", "")):
        score += 15.0

    if "VOLATILITY_SQUEEZE_RELEASE" in str(ev.get("event_type", "")):
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

    if "volume" in df_15m.columns and len(df_15m) >= 20:
        try:
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

    atr = tr.rolling(window=14, min_periods=14).mean().iloc[-1]
    if pd.isna(atr) or float(atr) <= 0:
        raise ValueError("ATR unavailable")

    atr = float(atr)
    sl_atr_multiplier = 1.5
    risk_pct_raw = (atr * sl_atr_multiplier) / entry_price * 100.0

    if not (float("-inf") < risk_pct_raw < float("inf")):
        raise ValueError("Invalid ATR-derived risk")

    risk_pct = max(0.50, min(risk_pct_raw, 5.00))

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
        "risk_pct": risk_pct,
        "target_rr": 2.0,
        "planned_weighted_rr": 1.55,
        "realized_rr": None,
        "trigger_ok": True,
    }


def build_tp_levels(setup: dict, direction: str) -> Tuple[float, List[dict]]:
    direction = str(direction).upper()
    entry = float(setup["entry_reference"])
    sl_price = float(setup["invalidation_price"])
    final_tp_price = float(setup["target_price"])

    if entry <= 0:
        raise ValueError("entry_reference must be > 0")

    if direction == "LONG":
        sl_pct = (entry - sl_price) / entry * 100.0
        tp_pct = (final_tp_price - entry) / entry * 100.0
    elif direction == "SHORT":
        sl_pct = (sl_price - entry) / entry * 100.0
        tp_pct = (entry - final_tp_price) / entry * 100.0
    else:
        raise ValueError(f"Invalid direction={direction}")

    if sl_pct <= 0 or tp_pct <= 0:
        raise ValueError("Invalid SL/TP percentages")

    tp_levels = [
        {"leg": "tp1", "pnl_pct": round(tp_pct * 0.50, 6), "close_fraction": 0.30},
        {"leg": "tp2", "pnl_pct": round(tp_pct * 0.75, 6), "close_fraction": 0.30},
        {"leg": "tp3", "pnl_pct": round(tp_pct, 6), "close_fraction": 0.40},
    ]

    setup["risk_pct"] = sl_pct
    setup["planned_weighted_rr"] = 1.55
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


def _tp_orders_to_tracker(tp_orders: list[dict]) -> list[dict]:
    out: list[dict] = []
    for order in tp_orders:
        cid = str(order.get("clientOrderId", "")).upper()
        leg = next((x for x in ("tp1", "tp2", "tp3") if x.upper() in cid), None)
        if not leg:
            continue
        out.append(
            {
                "leg": leg,
                "status": "already_exists",
                "order_id": str(order.get("orderId", "")),
                "price": float(order.get("stopPrice", 0) or order.get("price", 0) or 0),
                "qty": float(order.get("origQty", 0) or order.get("quantity", 0) or 0),
            }
        )
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
    try:
        positions = get_positions()
    except Exception as exc:
        log.error("[RECONCILIATION] Failed to fetch positions: %s", exc)
        return

    active_trades = _load_active_trades()

    for p in positions:
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

        prot = get_open_protection_directional(bx_symbol, direction)
        if prot.get("status") != "ok":
            log.warning("[RECONCILIATION] Cannot inspect protection for %s: %s", bx_symbol, prot.get("error"))
            continue

        sl_orders = list(prot.get("sl_orders", []))
        tp_orders = list(prot.get("tp_orders", []))

        matched_trade = _find_active_trade_for_position(bx_symbol, direction, active_trades)
        hit_legs = set(matched_trade.get("hit_legs", [])) if matched_trade else set()
        be_activated = bool(matched_trade.get("be_activated", False)) if matched_trade else False

        all_possible_legs = {"tp1", "tp2", "tp3"}
        remaining_expected_legs = all_possible_legs - hit_legs

        known_tp_legs = {
            leg
            for order in tp_orders
            for leg in ("tp1", "tp2", "tp3")
            if leg.upper() in str(order.get("clientOrderId", "")).upper()
        }

        sl_valid = False
        if sl_orders:
            try:
                sl_price = float(sl_orders[0].get("stopPrice", 0) or sl_orders[0].get("price", 0) or 0)
                sl_amt = float(sl_orders[0].get("origQty", 0) or sl_orders[0].get("quantity", 0) or 0)

                if sl_price > 0 and sl_amt > 0:
                    if direction == "LONG":
                        sl_valid = (sl_price <= avg_price * 1.003) if be_activated else (sl_price < avg_price)
                    elif direction == "SHORT":
                        sl_valid = (sl_price >= avg_price * 0.997) if be_activated else (sl_price > avg_price)
            except (TypeError, ValueError):
                sl_valid = False

        protection_complete = sl_valid and remaining_expected_legs.issubset(known_tp_legs)

        if protection_complete:
            tracker_tp = _tp_orders_to_tracker(tp_orders)
            tracker_sl = _sl_order_to_tracker(sl_orders)
            if tracker_tp and tracker_sl:
                update_active_trade_protection(
                    symbol=bx_symbol,
                    direction=direction,
                    tp_orders=tracker_tp,
                    sl_result=tracker_sl,
                )
            continue

        log.warning(
            "[RECONCILIATION] %s (%s) Incomplete: SL=%s, TPs=%d/%d (hit: %s). Repairing missing...",
            bx_symbol, direction, "OK" if sl_valid else "MISSING", len(known_tp_legs), len(remaining_expected_legs), list(hit_legs)
        )

        sl_pct = 2.0
        tp_pct = 4.0

        try:
            k1 = fetch_klines(bx_symbol, "1h", limit=30)
            if len(k1) >= 20:
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

                atr = tr.rolling(14, min_periods=14).mean().iloc[-1]
                if pd.notna(atr) and float(atr) > 0:
                    risk_pct = max(0.50, min(float(atr) * 1.5 / avg_price * 100.0, 5.00))
                    sl_pct = risk_pct
                    tp_pct = risk_pct * 2.0
        except Exception as exc:
            log.error("[RECONCILIATION] ATR error for %s: %s", bx_symbol, exc)

        tp_levels = []
        if "tp1" not in hit_legs:
            tp_levels.append({"leg": "tp1", "pnl_pct": round(tp_pct * 0.50, 6), "close_fraction": 0.30})
        if "tp2" not in hit_legs:
            tp_levels.append({"leg": "tp2", "pnl_pct": round(tp_pct * 0.75, 6), "close_fraction": 0.30})
        if "tp3" not in hit_legs:
            tp_levels.append({"leg": "tp3", "pnl_pct": round(tp_pct, 6), "close_fraction": 0.40})

        if not tp_levels:
            tp_levels = [{"leg": "tp3", "pnl_pct": round(tp_pct, 6), "close_fraction": 1.0}]

        trade_event_id = matched_trade.get("event_id") if matched_trade else f"REC_{bx_symbol}_{direction}"

        res = ensure_directional_protection(
            symbol=bx_symbol,
            direction=direction,
            avg_price=avg_price,
            qty=qty,
            stop_loss_pct=sl_pct,
            tp_levels=tp_levels,
            trade_id=str(trade_event_id).replace("EVT_", ""),
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

            log.info("[RECONCILIATION] Protection restored for %s (%s): SL=%.2f%%, TP=+%.2f%%", bx_symbol, direction, sl_pct, tp_pct)

            send_tg(
                f"🛡 <b>[ЗАЩИТА ВОССТАНОВЛЕНА] {bx_symbol}</b>\n\n"
                f"Недостающая защита восстановлена:\n"
                f"• Направление: <b>{direction}</b>\n"
                f"• Цена входа: <code>{avg_price:.8g}</code>\n"
                f"• SL: <code>{sl_pct:.2f}%</code>\n"
                f"• TP: <code>+{tp_pct:.2f}%</code> (каскад)"
            )


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
        position = wait_for_position_fill_directional(symbol=symbol, direction=direction, timeout_sec=15, poll_interval=0.5)
    except Exception as exc:
        return {"status": "POSITION_WAIT_FAILED", "mode": EXECUTION_MODE, "order_id": order_id, "open_result": opened, "error": str(exc)}

    if not isinstance(position, dict) or str(position.get("status", "")).lower() != "found":
        return {"status": "POSITION_NOT_CONFIRMED", "mode": EXECUTION_MODE, "order_id": order_id, "open_result": opened, "position": position}

    try:
        actual_qty = abs(float(position.get("positionAmt", 0) or 0))
        actual_avg_price = float(position.get("avgPrice", 0) or position.get("entryPrice", 0) or 0)
    except (TypeError, ValueError):
        actual_qty = 0.0
        actual_avg_price = 0.0

    if actual_qty <= 0 or actual_avg_price <= 0:
        return {"status": "POSITION_INVALID", "mode": EXECUTION_MODE, "order_id": order_id, "open_result": opened, "position": position}

    execution_quality = calculate_execution_slippage(signal_price=price, actual_entry_price=actual_avg_price, direction=direction)
    log.info(
        "[EXECUTION] Fill confirmed: %s %s at avgPrice=%.8g (Qty: %.8g, Slippage: %+.2f%%)",
        direction, symbol, actual_avg_price, actual_qty, execution_quality.get("slippage_pct") or 0.0
    )

    try:
        setup_for_fill = dict(setup)
        setup_for_fill["entry_reference"] = actual_avg_price
        planned_risk_pct = float(setup.get("risk_pct", 0) or 0)

        if not pd.notna(planned_risk_pct) or planned_risk_pct <= 0:
            raise ValueError("invalid planned risk_pct")

        if direction == "LONG":
            invalidation = actual_avg_price * (1.0 - planned_risk_pct / 100.0)
            target = actual_avg_price * (1.0 + 2.0 * planned_risk_pct / 100.0)
        else:
            invalidation = actual_avg_price * (1.0 + planned_risk_pct / 100.0)
            target = actual_avg_price * (1.0 - 2.0 * planned_risk_pct / 100.0)

        setup_for_fill["invalidation_price"] = invalidation
        setup_for_fill["target_price"] = target
        setup_for_fill["realized_rr"] = None

    except (TypeError, ValueError) as exc:
        return {
            "status": "PROTECTION_SETUP_INVALID",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "position": position,
            "open_result": opened,
            "execution_quality": execution_quality,
            "error": str(exc),
        }

    try:
        sl_pct, tp_levels = build_tp_levels(setup_for_fill, direction)
    except Exception as exc:
        return {
            "status": "PROTECTION_SETUP_INVALID",
            "mode": EXECUTION_MODE,
            "order_id": order_id,
            "position": position,
            "open_result": opened,
            "execution_quality": execution_quality,
            "error": str(exc),
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

    protection_status = str(protection.get("status", "")).upper()
    if protection_status == "PROTECTED":
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
    }


def main() -> None:
    log.info("========== [ENGINE] CYCLE START: %s UTC | Mode: %s | Exec: %s ==========", pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"), EXECUTION_MODE, EXECUTION_ENABLED)

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
        "rejected_trigger": 0,
        "rejected_cvd": 0,
        "trigger_passed": 0,
        "trigger_no_window": 0,
        "trigger_breakout_failed": 0,
        "trigger_volume_failed": 0,
        "trigger_data_failed": 0,
        "trigger_direction_failed": 0,
        "valid_signals": 0,
        "execution_attempts": 0,
        "trades": 0,
        "scan_errors": 0,
    }

    btc_regime_df = None
    try:
        btc_klines = fetch_klines("BTC-USDT", "1h", limit=10)
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
        rows = fetch_data()
        log.info("[COINALYZE] Ingested %d rows from Coinalyze.", len(rows))
    except Exception as exc:
        stats["scan_errors"] += 1
        log.error("[COINALYZE] Scrape error: %s", exc)

    stats["coinalyze_rows"] = len(rows)

    try:
        contracts = refresh_contracts()
        log.info("[BINGX] Refreshed %d active perpetual contracts.", len(contracts))
    except Exception as exc:
        stats["scan_errors"] += 1
        log.error("[BINGX] Contracts refresh error: %s", exc)

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
    telegram_sent_event_ids = load_ids(ACTIONS)

    opportunities: List[dict] = []
    opportunity_keys: set[tuple] = set()

    for r in candidates:
        symbol = r.symbol
        try:
            k1 = fetch_klines(symbol, "1h", int(os.environ.get("KLINE_LIMIT_1H", "250")))
            if len(k1) < 60:
                continue

            d1 = add_cvd(pd.DataFrame(k1))
            divergence_events = detect_divergences(d1, symbol, "1h")
            squeeze_events = detect_squeeze_release(d1, symbol, "1h", min_squeeze_bars=3)

            stats["divergence_events"] += len(divergence_events)
            stats["squeeze_events"] += len(squeeze_events)
            all_events = divergence_events + squeeze_events
            stats["events_total"] += len(all_events)

            if not all_events:
                continue

            d15 = None
            for ev in all_events:
                event_id = ev.get("event_id")
                if not event_id:
                    continue

                direction = str(ev.get("direction", "")).upper()
                if direction not in {"LONG", "SHORT"}:
                    stats["trigger_direction_failed"] += 1
                    continue

                timestamps = ev.get("timestamps", {})
                try:
                    detected_at = int(timestamps.get("detected_at_ts", 0) or 0)
                except (TypeError, ValueError):
                    stats["trigger_data_failed"] += 1
                    continue

                if detected_at <= 0:
                    stats["trigger_data_failed"] += 1
                    continue

                try:
                    latest_close = int(d1["close_time"].iloc[-1])
                except (TypeError, ValueError, IndexError, KeyError):
                    stats["scan_errors"] += 1
                    continue

                age = (latest_close - detected_at) / 60000.0
                if age < 0 or age > MAX_AGE:
                    stats["rejected_age"] += 1
                    continue

                stats["fresh_events"] += 1
                if direction == "LONG":
                    stats["fresh_long"] += 1
                else:
                    stats["fresh_short"] += 1

                event_type = str(ev.get("event_type", "")).upper()
                if "SQUEEZE" in event_type:
                    stats["fresh_squeeze"] += 1
                else:
                    stats["fresh_divergence"] += 1

                log.info("[SIGNALS] Fresh event: %s %s | Type: %s | Age: %.1fm", direction, symbol, event_type, age)

                if event_id not in seen_events:
                    emit_event(ev)
                    seen_events.add(event_id)

                if btc_regime_df is not None and symbol != "BTC-USDT":
                    btc_ok, btc_reason = check_btc_regime(btc_regime_df, direction)
                    if not btc_ok:
                        stats["rejected_btc"] += 1
                        log.info("[SIGNALS] %s %s blocked by BTC: %s", direction, symbol, btc_reason)
                        continue

                if d15 is None:
                    try:
                        k15 = fetch_klines(symbol, "15m", int(os.environ.get("KLINE_LIMIT_15M", "250")))
                    except Exception as exc:
                        stats["trigger_data_failed"] += 1
                        stats["scan_errors"] += 1
                        log.warning("[SIGNALS] 15M fetch error for %s: %s", symbol, exc)
                        continue

                    if len(k15) < 20:
                        stats["trigger_data_failed"] += 1
                        continue
                    d15 = pd.DataFrame(k15)

                required_15m = {"close_time", "close", "high", "low", "volume"}
                if not required_15m.issubset(d15.columns):
                    stats["trigger_data_failed"] += 1
                    continue

                try:
                    latest_15m_close_ts = int(d15["close_time"].iloc[-1])
                    trigger_delay_min = (latest_15m_close_ts - detected_at) / 60000.0
                except (TypeError, ValueError, IndexError):
                    stats["trigger_data_failed"] += 1
                    continue

                if trigger_delay_min < 0 or trigger_delay_min > MAX_AGE:
                    stats["trigger_no_window"] += 1
                    continue

                if REQUIRE_TRIGGER:
                    trigger_ok = build_15m_trigger(d15, direction, min_vol_mult=1.05)
                    if not trigger_ok:
                        stats["rejected_trigger"] += 1
                        log.info("[SIGNALS] %s %s (%s) failed 15m trigger.", direction, symbol, event_type)
                        continue
                    stats["trigger_passed"] += 1
                else:
                    stats["trigger_passed"] += 1

                if REQUIRE_CVD:
                    cvd24 = getattr(r, "cvd24", None)
                    try:
                        cvd24_value = float(cvd24)
                    except (TypeError, ValueError):
                        stats["rejected_cvd"] += 1
                        continue

                    if not pd.notna(cvd24_value) or cvd24_value <= CVD_MIN_CONFIRMATION:
                        stats["rejected_cvd"] += 1
                        log.info("[SIGNALS] %s %s CVD low (%.1f <= %.1f)", direction, symbol, cvd24_value or 0, CVD_MIN_CONFIRMATION)
                        continue

                fact = ev.get("event_fact", {})
                raw_price = fact.get("detection_close_price") or fact.get("close") or r.price
                try:
                    signal_price = float(raw_price)
                except (TypeError, ValueError):
                    stats["scan_errors"] += 1
                    continue

                if not pd.notna(signal_price) or signal_price <= 0:
                    stats["scan_errors"] += 1
                    continue

                setup = build_event_setup(ev=ev, df_1h=d1, entry_price=signal_price)
                score = calculate_setup_score(ev=ev, coinalyze_row=r, df_15m=d15)

                opportunity_key = (
                    symbol,
                    direction,
                    event_type,
                    timestamps.get("pivot_1_ts"),
                    timestamps.get("pivot_2_ts"),
                )

                if opportunity_key in opportunity_keys:
                    continue
                opportunity_keys.add(opportunity_key)

                log.info("[SIGNALS] Signal valid: %s %s | Score: %.0f/100 | Event: %s | Price: %.8g | SL: %.8g | TP: %.8g",
                         direction, symbol, score, event_type, signal_price, setup['invalidation_price'], setup['target_price'])

                opportunities.append(
                    {
                        "event": ev,
                        "event_id": event_id,
                        "symbol": symbol,
                        "direction": direction,
                        "price": signal_price,
                        "setup": setup,
                        "score": score,
                        "coinalyze_row": r,
                    }
                )

        except Exception as exc:
            stats["scan_errors"] += 1
            log.error("[SCAN] Error scanning %s: %s", symbol, exc)

    stats["valid_signals"] = len(opportunities)
    opportunities.sort(key=lambda x: x["score"], reverse=True)

    log.info("[RANKING] Valid signals count: %d.", len(opportunities))
    for i, opp in enumerate(opportunities[:5], start=1):
        log.info("  [RANKING] #%d: %s %s | Score: %.0f | %s", i, opp['direction'], opp['symbol'], opp['score'], opp['event'].get('event_type'))

    trades_this_cycle = 0

    for opp in opportunities:
        event_id = opp["event_id"]
        symbol = opp["symbol"]
        direction = opp["direction"]
        price = opp["price"]
        setup = opp["setup"]
        score = opp["score"]
        r = opp["coinalyze_row"]
        ev = opp["event"]

        if event_id in executed_event_ids:
            execution_result = {"status": "ALREADY_EXECUTED", "mode": EXECUTION_MODE, "order_id": None}
            log.info("[EXECUTION] %s (%s) - Already executed previously.", symbol, event_id)
        elif EXECUTION_ENABLED and trades_this_cycle < MAX_TRADES:
            stats["execution_attempts"] += 1
            log.info("[EXECUTION] Attempt #%d/%d: %s %s (Score: %.0f, Ref: %.8g)...", trades_this_cycle + 1, MAX_TRADES, direction, symbol, score, price)
            execution_result = execute_new_position(symbol=symbol, direction=direction, price=price, setup=setup, event_id=event_id)

            actual_position = execution_result.get("position", {}) if isinstance(execution_result, dict) else {}
            actual_entry = float(actual_position.get("avgPrice", 0) or actual_position.get("entryPrice", 0) or price)
            actual_qty = actual_position.get("positionAmt")
            execution_quality = execution_result.get("execution_quality", {}) if isinstance(execution_result, dict) else {}

            record_trade(
                {
                    "record_type": "TRADE_OPEN",
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
                        "actual_entry_price": actual_entry,
                        "actual_qty": actual_qty,
                        "order_id": execution_result.get("order_id"),
                        "status": execution_result.get("status"),
                        "slippage_pct": execution_quality.get("slippage_pct"),
                        "adverse_slippage_pct": execution_quality.get("adverse_slippage_pct"),
                    },
                    "score": score,
                    "event_type": ev.get("event_type"),
                    "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
                    "result": execution_result,
                    "setup": setup,
                    "planned_metrics": {
                        "target_rr": setup.get("target_rr"),
                        "planned_weighted_rr": setup.get("planned_weighted_rr"),
                        "realized_rr": None,
                    },
                }
            )

            status = str(execution_result.get("status", ""))
            if status in {"opened_protected", "opened_protection_check_required", "opened_protection_failed"}:
                trades_this_cycle += 1
                if status in {"opened_protected", "opened_protection_check_required"}:
                    executed_event_ids.add(event_id)
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
                        coinalyze_row=r,
                        score=score,
                        setup=setup,
                        requested_entry_price=price,
                    )
                except Exception as exc:
                    log.error("[TRACKER] Registration error for %s: %s", symbol, exc)

        elif not EXECUTION_ENABLED:
            execution_result = {"status": "DISABLED", "mode": EXECUTION_MODE, "order_id": None}
            log.info("[EXECUTION] %s %s skipped: EXECUTION_ENABLED is false.", direction, symbol)
        else:
            execution_result = {"status": "TRADE_LIMIT_REACHED", "mode": EXECUTION_MODE, "order_id": None}
            log.info("[EXECUTION] %s %s skipped: cycle limit reached (%d/%d).", direction, symbol, trades_this_cycle, MAX_TRADES)

        msg = format_signal(
            ev,
            setup=setup,
            coinalyze_row=r,
            execution=execution_result,
            score=score,
        )

        is_real_execution = execution_result.get("status") in {
            "opened_protected",
            "opened",
            "opened_protection_check_required",
            "opened_protection_failed",
        }

        telegram_already_sent = event_id in telegram_sent_event_ids and not is_real_execution
        sent = False
        if not telegram_already_sent:
            try:
                sent = bool(send_tg(msg))
            except Exception:
                sent = False
            if sent:
                telegram_sent_event_ids.add(event_id)
                log.info("[TELEGRAM] Notification sent for %s %s (%s).", direction, symbol, event_id)

        record_action(
            {
                "event_id": event_id,
                "symbol": symbol,
                "direction": direction,
                "score": score,
                "event_type": ev.get("event_type"),
                "telegram_sent": bool(sent),
                "execution_status": execution_result.get("status"),
                "ts": int(pd.Timestamp.utcnow().timestamp() * 1000),
            }
        )

    try:
        append_shadow_health(events_path=EVENTS, health_path=HEALTH, trades_path=TRADES)
    except Exception as exc:
        log.error("[SHADOW] Health snapshot error: %s", exc)

    summary_str = " ".join(f"{k}={v}" for k, v in stats.items())
    log.info("[SUMMARY] [FORENSIC_SUMMARY] %s", summary_str)
    log.info("[SUMMARY] [ENGINE_SUMMARY] trades_this_cycle=%d %s", trades_this_cycle, summary_str)
    log.info("========== [ENGINE] CYCLE END: trades_this_cycle=%d ==========", trades_this_cycle)


if __name__ == "__main__":
    main()
