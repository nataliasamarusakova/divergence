from __future__ import annotations

import json
import os
import sys
import time
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
        "diagnostic": "BINGX_READ_ONLY_SCHEMA_DIAGNOSTIC",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_host": bingx.BASE_URL.split("//", 1)[-1].split("/", 1)[0],
        "position_endpoint": bingx.POSITION_PATH,
        "income_endpoint": bingx.INCOME_HISTORY_PATH,
        "income_window": {"start_ms": start_ms, "end_ms": now_ms, "hours": 24},
        "probes": [],
        "safety": "Signed GET only. No order, cancel, leverage, or position mutation endpoints are called.",
        "redaction": "Response values, symbols, quantities, IDs, API keys, signatures, and raw messages are not emitted; only envelope and schema metadata are reported.",
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
