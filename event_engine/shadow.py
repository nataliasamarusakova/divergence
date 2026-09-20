from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    return rows


def _load_state(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(path: Path, state: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _symbol_key(value: str) -> str:
    s = str(value or "").upper().strip()
    for suffix in ("-USDT", "USDT", "-USD", "USD"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def _paper_pnl_pct(entry: float, price: float, direction: str) -> float:
    if entry <= 0 or price <= 0:
        return 0.0
    if direction == "LONG":
        return (price - entry) / entry * 100.0
    return (entry - price) / entry * 100.0


def _reached(price: float, level: float, direction: str, kind: str) -> bool:
    if direction == "LONG":
        return price <= level if kind == "SL" else price >= level
    return price >= level if kind == "SL" else price <= level


def record_divergence_shadow_open(
    path: Path,
    *,
    event_id: str,
    symbol: str,
    direction: str,
    event_type: str,
    timeframe: str,
    entry_price: float,
    setup: dict[str, Any],
    score: float,
    opened_ts: int,
) -> dict[str, Any]:
    state = _load_state(path)
    key = str(event_id)
    if key in state:
        return {"status": "already_opened", **state[key]}

    entry = float(entry_price)
    risk_pct = float(setup.get("risk_pct", 0.0) or 0.0)
    if entry <= 0 or risk_pct <= 0:
        raise ValueError(f"Invalid shadow setup for {event_id}")
    # Rebase protection/target to the paper entry price rather than the original
    # signal reference. This mirrors live protection, which is rebuilt from the
    # actual fill price after trigger confirmation.
    if str(direction).upper() == "LONG":
        sl = entry * (1.0 - risk_pct / 100.0)
        target = entry * (1.0 + 2.50 * risk_pct / 100.0)
    else:
        sl = entry * (1.0 + risk_pct / 100.0)
        target = entry * (1.0 - 2.50 * risk_pct / 100.0)

    trade = {
        "event_id": key,
        "symbol": str(symbol),
        "symbol_key": _symbol_key(symbol),
        "direction": str(direction).upper(),
        "event_type": str(event_type),
        "timeframe": str(timeframe),
        "score": float(score),
        "entry_price": entry,
        "sl_price": sl,
        "tp3_price": target,
        "risk_pct": risk_pct,
        "opened_ts": int(opened_ts),
        "updated_ts": int(opened_ts),
        "status": "ACTIVE",
        "peak_pnl_pct": 0.0,
        "mae_pct": 0.0,
        "current_price": entry,
        "hit_legs": [],
    }
    state[key] = trade
    _save_state(path, state)
    result = dict(trade)
    result["status"] = "opened"
    result["state_status"] = "ACTIVE"
    return result


def update_divergence_shadow_state(path: Path, market_prices: dict[str, float], now_ms: int) -> dict[str, int]:
    state = _load_state(path)
    if not state:
        return {"active": 0, "closed": 0, "updated": 0}

    by_symbol = {_symbol_key(k): float(v) for k, v in market_prices.items() if v is not None and float(v) > 0}
    closed = 0
    updated = 0

    for event_id, trade in list(state.items()):
        if str(trade.get("status")) != "ACTIVE":
            continue
        price = by_symbol.get(str(trade.get("symbol_key", "")))
        if price is None:
            continue

        direction = str(trade.get("direction", "")).upper()
        entry = float(trade.get("entry_price", 0.0) or 0.0)
        sl = float(trade.get("sl_price", 0.0) or 0.0)
        tp3 = float(trade.get("tp3_price", 0.0) or 0.0)
        pnl_pct = _paper_pnl_pct(entry, price, direction)
        trade["current_price"] = price
        trade["updated_ts"] = int(now_ms)
        trade["peak_pnl_pct"] = max(float(trade.get("peak_pnl_pct", 0.0) or 0.0), pnl_pct)
        trade["mae_pct"] = min(float(trade.get("mae_pct", 0.0) or 0.0), pnl_pct)

        # The persistent snapshot feed does not expose intrabar ordering, so when
        # both levels could have been crossed in the same snapshot, SL wins for
        # conservative paper accounting. TP levels are not used to trigger BE; this
        # shadow model is only measuring the setup without changing live management.
        exit_reason = None
        if _reached(price, sl, direction, "SL"):
            exit_reason = "SHADOW_SL"
        elif _reached(price, tp3, direction, "TP"):
            exit_reason = "SHADOW_TP3"
            trade["hit_legs"] = ["tp1", "tp2", "tp3"]
        else:
            rr = pnl_pct / float(trade.get("risk_pct", 1.0) or 1.0)
            if rr >= 1.50:
                trade["hit_legs"] = sorted(set(trade.get("hit_legs", [])) | {"tp1", "tp2"})
            elif rr >= 0.75:
                trade["hit_legs"] = sorted(set(trade.get("hit_legs", [])) | {"tp1"})

        if exit_reason:
            trade["status"] = "CLOSED"
            trade["closed_ts"] = int(now_ms)
            trade["exit_reason"] = exit_reason
            trade["exit_price"] = price
            trade["realized_pnl_pct"] = pnl_pct
            trade["realized_rr"] = pnl_pct / float(trade.get("risk_pct", 1.0) or 1.0)
            closed += 1
        updated += 1

    _save_state(path, state)
    active = sum(1 for t in state.values() if str(t.get("status")) == "ACTIVE")
    return {"active": active, "closed": closed, "updated": updated}


def generate_shadow_health_snapshot(events_path: Path, trades_path: Path | None = None, divergence_shadow_path: Path | None = None, cycle_stats: dict | None = None) -> dict:
    now_ms = int(time.time() * 1000)
    events = _load_jsonl(events_path)
    trades = _load_jsonl(trades_path) if trades_path else []
    shadow = _load_state(divergence_shadow_path) if divergence_shadow_path else {}

    structures: dict[str, set[str]] = {}
    for event in events:
        ts = event.get("timestamps", {})
        params = event.get("detector_params", {})
        key = ":".join(map(str, (event.get("symbol"), event.get("timeframe"), event.get("direction"), ts.get("pivot_1_ts", 0), ts.get("pivot_2_ts", 0), params.get("pivot_left", 3), params.get("pivot_right", 2), params.get("pivot_pairing_mode", "multi"))))
        bucket = structures.setdefault(key, set())
        et = str(event.get("event_type", ""))
        for tag in ("RSI", "CVD", "MACD", "STOCH", "OBV"):
            if tag in et:
                bucket.add(tag)
        if et.endswith("_OI"):
            bucket.add("OI")

    rsi_only = sum(v == {"RSI"} for v in structures.values())
    cvd_only = sum(v == {"CVD"} for v in structures.values())
    joint = sum(v == {"RSI", "CVD"} for v in structures.values())
    latest_event = max((e.get("timestamps", {}).get("detected_at_ts", 0) for e in events), default=0)
    types = [str(e.get("event_type", "")) for e in events]
    open_records = [t for t in trades if t.get("record_type") == "TRADE_OPEN"]
    close_records = [t for t in trades if t.get("record_type") == "TRADE_CLOSE"]
    confirmed_open = []
    for t in open_records:
        execution = t.get("execution") if isinstance(t.get("execution"), dict) else {}
        result = t.get("result") if isinstance(t.get("result"), dict) else {}
        position = result.get("position") if isinstance(result.get("position"), dict) else {}
        status = str(execution.get("status") or result.get("status") or "").lower()
        try:
            qty = abs(float(position.get("positionAmt", 0) or 0))
        except (TypeError, ValueError):
            qty = 0.0
        if status in {"opened", "opened_protected", "opened_protection_check_required", "opened_protection_failed"} and qty > 0:
            confirmed_open.append(t)
    unique_confirmed_ids = {str(t.get("trade_id") or t.get("event_id")) for t in confirmed_open if t.get("trade_id") or t.get("event_id")}
    return {
        "timestamp": now_ms,
        "events": {
            "total": len(events), "unique_structures": len(structures),
            "rsi_events": sum("RSI" in t for t in types), "cvd_events": sum("CVD" in t for t in types),
            "macd_events": sum("MACD" in t for t in types), "stoch_events": sum("STOCH" in t for t in types),
            "obv_events": sum("OBV" in t for t in types), "oi_events": sum(t.endswith("_OI") for t in types),
            "liq_squeeze_events": sum(t in {"SHORT_SQUEEZE", "LONG_SQUEEZE"} for t in types),
            "rsi_only_structures": rsi_only, "cvd_only_structures": cvd_only, "joint_structures": joint,
            "latest_event_ts": latest_event,
            "event_feed_age_min": round((now_ms - latest_event) / 60000.0, 1) if latest_event else None,
        },
        "trades": {
            "journal_records": len(trades),
            "total": len(unique_confirmed_ids),
            "opened": len(unique_confirmed_ids),
            "open_records": len(open_records),
            "confirmed_open_records": len(confirmed_open),
            "close_records": len(close_records),
        },
        "divergence_shadow": {
            "total": len(shadow),
            "active": sum(1 for t in shadow.values() if t.get("status") == "ACTIVE"),
            "closed": sum(1 for t in shadow.values() if t.get("status") == "CLOSED"),
            "wins": sum(1 for t in shadow.values() if t.get("status") == "CLOSED" and float(t.get("realized_pnl_pct", 0.0) or 0.0) > 0),
            "losses": sum(1 for t in shadow.values() if t.get("status") == "CLOSED" and float(t.get("realized_pnl_pct", 0.0) or 0.0) <= 0),
        },
        # CVD needs taker-flow fields that BingX v3 klines do not return, so
        # cvd_only is structurally 0 and must not hold all_criteria_met hostage.
        "gate_readiness": {
            "rsi_only_ready": rsi_only >= 40,
            "cvd_only_ready": cvd_only >= 40,
            "cvd_feed_available": cvd_only > 0,
            "joint_ready": joint >= 30,
            "all_criteria_met": rsi_only >= 40 and (cvd_only == 0 or (cvd_only >= 40 and joint >= 30)),
        },
        "cycle_stats": cycle_stats or {},
    }


def append_shadow_health(events_path: Path, health_path: Path, trades_path: Path | None = None, divergence_shadow_path: Path | None = None, cycle_stats: dict | None = None) -> dict:
    snapshot = generate_shadow_health_snapshot(events_path, trades_path, divergence_shadow_path, cycle_stats)
    health_path.parent.mkdir(parents=True, exist_ok=True)
    with health_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    ev = snapshot["events"]
    tr = snapshot["trades"]
    ds = snapshot["divergence_shadow"]
    print(f"[SHADOW_HEALTH] structures={ev['unique_structures']} RSI={ev['rsi_only_structures']}/40 CVD={ev['cvd_only_structures']}/40 Joint={ev['joint_structures']}/30 Trades={tr['open_records']}/{tr['close_records']} DIVERGENCE_SHADOW={ds['active']} active/{ds['closed']} closed")
    return snapshot
