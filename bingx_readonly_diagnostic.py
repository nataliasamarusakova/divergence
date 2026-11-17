from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from event_engine import bingx


def shape(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(value).__name__}
    if isinstance(value, dict):
        result["keys"] = sorted(str(k) for k in value.keys())[:80]
        result["key_count"] = len(value)
        # Do not emit values: API payloads can contain account/order information.
        nested = {}
        for key in ("list", "rows", "records", "positions", "position", "income", "data", "result"):
            if key in value:
                nested[key] = shape(value[key])
        if nested:
            result["nested"] = nested
    elif isinstance(value, list):
        result["length"] = len(value)
        result["sample_item_types"] = [type(x).__name__ for x in value[:3]]
        result["sample_item_keys"] = [sorted(str(k) for k in x.keys())[:80] if isinstance(x, dict) else None for x in value[:3]]
    return result


def _canonical_symbol(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip().upper().replace("_", "-").replace("/", "-")
    raw = re.sub(r"-+", "-", raw)
    if not raw:
        return None
    if "-" not in raw and raw.endswith("USDT") and len(raw) > 4:
        raw = f"{raw[:-4]}-USDT"
    elif "-" not in raw and raw.isalnum():
        raw = f"{raw}-USDT"
    return raw if re.fullmatch(r"[A-Z0-9]+-USDT", raw) else None


def income_semantics(data: Any) -> dict[str, Any]:
    """Emit only aggregate classifications; never emit account row values."""
    if not isinstance(data, list):
        return {"available": False, "data_type": type(data).__name__}
    targets = {"TRX-USDT", "WLFI-USDT"}
    match_counts = Counter()
    income_types = Counter()
    missing_symbol_types = Counter()
    symbol_types = Counter()
    key_presence = Counter()
    for row in data:
        if not isinstance(row, dict):
            continue
        kind_raw = row.get("incomeType")
        kind = str(kind_raw).strip().upper() if kind_raw not in (None, "") else "MISSING"
        income_types[kind] += 1
        if "symbol" in row:
            key_presence["symbol_key_present"] += 1
        else:
            key_presence["symbol_key_missing"] += 1
        raw_symbol = row.get("symbol")
        symbol_types[type(raw_symbol).__name__] += 1
        normalized = _canonical_symbol(raw_symbol)
        if normalized in targets:
            match_counts[normalized] += 1
        if normalized is None:
            missing_symbol_types[kind] += 1
    return {
        "available": True,
        "row_count": len(data),
        "target_symbol_match_counts": {key: match_counts.get(key, 0) for key in sorted(targets)},
        "income_type_counts": dict(sorted(income_types.items())),
        "rows_with_unresolved_or_missing_symbol_by_income_type": dict(sorted(missing_symbol_types.items())),
        "symbol_value_type_counts": dict(sorted(symbol_types.items())),
        "symbol_key_presence_counts": dict(sorted(key_presence.items())),
        "non_dict_row_count": sum(1 for row in data if not isinstance(row, dict)),
    }


def probe(name: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    started = time.monotonic()
    try:
        response = bingx._request(
            "GET", path, params, signed=True, timeout_sec=8.0,
            retryable=False, response_metadata=metadata, close_connection=True,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        safe = {
            "name": name,
            "transport": "response",
            "elapsed_ms": elapsed_ms,
            "http_status": metadata.get("http_status"),
            "request_host": metadata.get("request_host"),
            "endpoint": metadata.get("endpoint"),
            "method": metadata.get("method"),
            "response_type": type(response).__name__,
            "response_keys": sorted(str(k) for k in response.keys())[:80] if isinstance(response, dict) else [],
            "response_code": response.get("code") if isinstance(response, dict) else None,
            "response_msg_type": type(response.get("msg")).__name__ if isinstance(response, dict) else None,
            "data": shape(response.get("data")) if isinstance(response, dict) and "data" in response else {"type": "missing"},
        }
        if name == "income_account_wide_24h" and isinstance(response, dict):
            safe["sanitized_row_summary"] = income_semantics(response.get("data"))
        return safe
    except Exception as exc:
        return {
            "name": name, "transport": "exception", "elapsed_ms": int((time.monotonic() - started) * 1000),
            "exception_type": type(exc).__name__,
            # Exception text may contain a fully signed request URL; never publish it.
            "exception_message": "redacted",
            "metadata": {k: metadata.get(k) for k in ("http_status", "endpoint", "method", "request_host", "transport")},
        }


def main() -> int:
    if not bingx.API_KEY or not bingx.SECRET_KEY:
        print(json.dumps({"status": "ERROR", "error": "BINGX_API_KEY or BINGX_SECRET_KEY is missing"}))
        return 2

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - 24 * 60 * 60 * 1000
    output: dict[str, Any] = {
        "diagnostic": "BINGX_READ_ONLY_SEMANTIC_DIAGNOSTIC",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_host": bingx.BASE_URL.split("//", 1)[-1].split("/", 1)[0],
        "position_endpoint": bingx.POSITION_PATH,
        "income_endpoint": bingx.INCOME_HISTORY_PATH,
        "income_window": {"start_ms": start_ms, "end_ms": now_ms, "hours": 24},
        "probes": [],
        "safety": "Signed GET only. No order, cancel, leverage, or position mutation endpoints are called.",
        "redaction": "No raw account row, symbol, amount, quantity, ID, API key, signature, or message is emitted. The account-wide income probe adds only aggregate counts for the two target symbols and row classifications.",
    }
    output["probes"].append(probe("positions_read_1", bingx.POSITION_PATH, {}))
    time.sleep(1.5)
    output["probes"].append(probe("positions_read_2_fresh_connection", bingx.POSITION_PATH, {}))
    output["probes"].append(probe("income_account_wide_24h", bingx.INCOME_HISTORY_PATH, {"startTime": start_ms, "endTime": now_ms, "limit": 1000}))
    for symbol in ("TRX-USDT", "WLFI-USDT"):
        output["probes"].append(probe(f"income_symbol_{symbol}_24h", bingx.INCOME_HISTORY_PATH, {"symbol": symbol, "startTime": start_ms, "endTime": now_ms, "limit": 1000}))

    failures = sum(1 for item in output["probes"] if item.get("transport") == "exception")
    output["status"] = "PARTIAL" if failures else "COMPLETE"
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
