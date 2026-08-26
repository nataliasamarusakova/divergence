"""
shadow.py
"""

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
        if "RSI" in et:
            bucket.add("RSI")
        if "CVD" in et:
            bucket.add("CVD")

    rsi_only = sum(v == {"RSI"} for v in structures.values())
    cvd_only = sum(v == {"CVD"} for v in structures.values())
    joint = sum(v == {"RSI", "CVD"} for v in structures.values())
    latest_event = max((e.get("timestamps", {}).get("detected_at_ts", 0) for e in events), default=0)

    return {
        "timestamp": now_ms,
        "events": {
            "total": len(events),
            "unique_structures": len(structures),
            "rsi_events": sum("RSI" in str(e.get("event_type", "")) for e in events),
            "cvd_events": sum("CVD" in str(e.get("event_type", "")) for e in events),
            "rsi_only_structures": rsi_only,
            "cvd_only_structures": cvd_only,
            "joint_structures": joint,
            "latest_event_ts": latest_event,
            "event_feed_age_min": round((now_ms - latest_event) / 60000.0, 1) if latest_event else None,
        },
        "trades": {
            "total": len(trades),
            "opened": sum(t.get("result", {}).get("status") in {"opened", "opened_protected"} for t in trades),
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
