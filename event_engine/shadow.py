# shadow.py

from __future__ import annotations

import json
import time
from pathlib import Path


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


def generate_shadow_health_snapshot(events_path: Path, trades_path: Path | None = None) -> dict:
    now_ms = int(time.time() * 1000)
    events = _load_jsonl(events_path)
    trades = _load_jsonl(trades_path) if trades_path else []

    structures: dict[str, set[str]] = {}
    for event in events:
        ts = event.get("timestamps", {})
        params = event.get("detector_params", {})
        key = ":".join(
            map(
                str,
                (
                    event.get("symbol"),
                    event.get("timeframe"),
                    event.get("direction"),
                    ts.get("pivot_1_ts", 0),
                    ts.get("pivot_2_ts", 0),
                    params.get("pivot_left", 3),
                    params.get("pivot_right", 2),
                    params.get("pivot_pairing_mode", "multi"),
                ),
            )
        )
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
        qty = 0.0
        try:
            qty = abs(float(position.get("positionAmt", 0) or 0))
        except (TypeError, ValueError):
            qty = 0.0
        if status in {"opened", "opened_protected", "opened_protection_check_required", "opened_protection_failed"} and qty > 0:
            confirmed_open.append(t)

    unique_trade_ids = {str(t.get("trade_id") or t.get("event_id")) for t in confirmed_open if t.get("trade_id") or t.get("event_id")}

    return {
        "timestamp": now_ms,
        "events": {
            "total": len(events),
            "unique_structures": len(structures),
            "rsi_events": sum("RSI" in t for t in types),
            "cvd_events": sum("CVD" in t for t in types),
            "macd_events": sum("MACD" in t for t in types),
            "stoch_events": sum("STOCH" in t for t in types),
            "obv_events": sum("OBV" in t for t in types),
            "oi_events": sum(t.endswith("_OI") for t in types),
            "liq_squeeze_events": sum(t in {"SHORT_SQUEEZE", "LONG_SQUEEZE"} for t in types),
            "rsi_only_structures": rsi_only,
            "cvd_only_structures": cvd_only,
            "joint_structures": joint,
            "latest_event_ts": latest_event,
            "event_feed_age_min": round((now_ms - latest_event) / 60000.0, 1) if latest_event else None,
        },
        "trades": {
            "journal_records": len(trades),
            "total": len(unique_trade_ids),
            "opened": len(unique_trade_ids),
            "open_records": len(open_records),
            "confirmed_open_records": len(confirmed_open),
            "close_records": len(close_records),
            "unique_closed_trade_ids": len({str(t.get("trade_id") or t.get("event_id")) for t in close_records if t.get("trade_id") or t.get("event_id")}),
        },
        "gate_readiness": {
            "rsi_only_ready": rsi_only >= 40,
            "cvd_only_ready": cvd_only >= 40,
            "joint_ready": joint >= 30,
            "all_criteria_met": rsi_only >= 40 and cvd_only >= 40 and joint >= 30,
        },
    }


def append_shadow_health(events_path: Path, health_path: Path, trades_path: Path | None = None) -> dict:
    snapshot = generate_shadow_health_snapshot(events_path, trades_path)
    health_path.parent.mkdir(parents=True, exist_ok=True)
    with health_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    ev = snapshot["events"]
    tr = snapshot["trades"]
    print(
        f"[SHADOW_HEALTH] structures={ev['unique_structures']} "
        f"RSI={ev['rsi_only_structures']}/40 "
        f"CVD={ev['cvd_only_structures']}/40 "
        f"Joint={ev['joint_structures']}/30 "
        f"Trades={tr['opened']}/{tr['total']}"
    )
    return snapshot

