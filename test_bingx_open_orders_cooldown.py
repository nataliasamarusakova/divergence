"""Regression tests for BingX openOrders endpoint disable periods."""
import json
import time

from event_engine import bingx as bx


class _Response:
    def __init__(self, payload):
        self._payload = payload
        self.headers = {}

    def json(self):
        return dict(self._payload)


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _Response(self.payload)


def test_open_orders_100410_cooldown_is_persisted_and_prevents_further_http_calls(tmp_path, monkeypatch):
    now_ms = int(time.time() * 1000)
    unblock_ms = now_ms + 120_000
    path = tmp_path / "bingx_open_orders_cooldown.json"
    session = _Session({
        "code": 100410,
        "msg": f"The endpoint trigger frequency limit rule is currently in the disabled period and will be unblocked after {unblock_ms}",
        "data": None,
    })
    monkeypatch.setattr(bx, "OPEN_ORDERS_COOLDOWN_STATE_PATH", path)
    monkeypatch.setattr(bx, "FAST_SESSION", session)
    monkeypatch.setattr(bx, "_OPEN_ORDERS_COOLDOWN_UNTIL_MS", 0)
    monkeypatch.setattr(bx, "_OPEN_ORDERS_COOLDOWN_REASON", "")

    first = bx._request("GET", bx.OPEN_ORDERS_PATH, {}, signed=False, retryable=False)
    assert first["code"] == 100410
    assert first["rate_limited"] is True
    assert first["retry_after_ms"] == unblock_ms
    assert len(session.calls) == 1
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["cooldown_until_ms"] == unblock_ms
    assert persisted["endpoint"] == bx.OPEN_ORDERS_PATH

    # Simulate a new workflow process: the persistent file, not module memory,
    # must stop the next request from reaching BingX during the disable period.
    monkeypatch.setattr(bx, "_OPEN_ORDERS_COOLDOWN_UNTIL_MS", 0)
    second = bx._request("GET", bx.OPEN_ORDERS_PATH, {"symbol": "BTC-USDT"}, signed=False, retryable=False)
    assert second["code"] == 100410
    assert second["local_cooldown"] is True
    assert second["retry_after_ms"] == unblock_ms
    assert len(session.calls) == 1

    assert bx._read_open_orders_cooldown(now_ms=unblock_ms + 1) is None
    assert not path.exists()


def test_open_orders_100410_without_timestamp_uses_conservative_fallback(tmp_path, monkeypatch):
    path = tmp_path / "bingx_open_orders_cooldown.json"
    session = _Session({"code": 100410, "msg": "Request rate limit exceeded", "data": None})
    monkeypatch.setattr(bx, "OPEN_ORDERS_COOLDOWN_STATE_PATH", path)
    monkeypatch.setattr(bx, "FAST_SESSION", session)
    monkeypatch.setattr(bx, "_OPEN_ORDERS_COOLDOWN_UNTIL_MS", 0)
    monkeypatch.setattr(bx, "_OPEN_ORDERS_COOLDOWN_REASON", "")

    response = bx._request("GET", bx.OPEN_ORDERS_PATH, {}, signed=False, retryable=False)
    now_ms = int(time.time() * 1000)
    assert response["code"] == 100410
    assert now_ms < response["retry_after_ms"] <= now_ms + int(bx.OPEN_ORDERS_COOLDOWN_FALLBACK_SEC * 1000) + 1000
    assert path.exists()


def test_engine_reconciliation_does_not_fan_out_after_account_snapshot_100410(monkeypatch):
    import run_once as ro

    positions = [
        {"symbol": "BTC-USDT", "positionSide": "LONG", "positionAmt": "0.02", "avgPrice": "60000"},
        {"symbol": "ETH-USDT", "positionSide": "SHORT", "positionAmt": "-0.5", "avgPrice": "3000"},
        {"symbol": "SOL-USDT", "positionSide": "LONG", "positionAmt": "3", "avgPrice": "100"},
    ]
    monkeypatch.setattr(ro, "get_positions", lambda **kwargs: positions)
    monkeypatch.setattr(ro, "_load_active_trades", lambda: {})
    snapshot_calls = []
    monkeypatch.setattr(ro, "get_open_orders_snapshot", lambda **kwargs: snapshot_calls.append(kwargs) or {
        "status": "error", "code": 100410, "rate_limited": True,
        "retry_after_ms": int(time.time() * 1000) + 60_000,
        "error": "openOrders snapshot failed: code=100410:endpoint disabled",
    })
    monkeypatch.setattr(ro, "get_open_protection_directional", lambda *a, **k: (_ for _ in ()).throw(AssertionError("per-symbol fallback must be skipped after 100410")))
    actions = []
    monkeypatch.setattr(ro, "record_action", lambda row: actions.append(row))

    ro.reconcile_all_open_positions()

    assert len(snapshot_calls) == 1
    assert len(actions) == len(positions)
    assert all(row["record_type"] == "PROTECTION_STATE_UNKNOWN" for row in actions)
    assert all("cooldown active" in row["error"] for row in actions)
