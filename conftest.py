"""Pytest isolation for production-only runtime gates.

The GitHub Actions workflow sets production VST environment variables for the
whole job. Unit tests must not inherit those execution gates implicitly: tests
that are specifically about S/R explicitly enable the S/R gate themselves.
"""
from __future__ import annotations

import ipaddress
import socket

import pytest


def _is_loopback_host(host):
    if host in (None, "localhost", "localhost."):
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def isolate_production_entry_gates(monkeypatch, tmp_path):
    # Unit/integration tests may use local HTTP servers for deterministic
    # redirect/transport tests, but must never resolve or connect to public
    # services. Record attempts as well as raising: broad exception handlers
    # in production code must not silently swallow an attempted network call.
    external_attempts = []
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyname = socket.gethostbyname
    original_gethostbyname_ex = socket.gethostbyname_ex
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_getaddrinfo(host, *args, **kwargs):
        if not _is_loopback_host(host):
            external_attempts.append(f"dns:{host!r}")
            raise AssertionError(f"pytest attempted external DNS lookup: {host!r}")
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_gethostbyname(host):
        if not _is_loopback_host(host):
            external_attempts.append(f"dns:{host!r}")
            raise AssertionError(f"pytest attempted external DNS lookup: {host!r}")
        return original_gethostbyname(host)

    def guarded_gethostbyname_ex(host):
        if not _is_loopback_host(host):
            external_attempts.append(f"dns:{host!r}")
            raise AssertionError(f"pytest attempted external DNS lookup: {host!r}")
        return original_gethostbyname_ex(host)

    def guarded_connect(sock, address):
        host = address[0] if isinstance(address, tuple) and address else address
        if not _is_loopback_host(host):
            external_attempts.append(f"connect:{address!r}")
            raise AssertionError(f"pytest attempted external socket connection: {address!r}")
        return original_connect(sock, address)

    def guarded_connect_ex(sock, address):
        host = address[0] if isinstance(address, tuple) and address else address
        if not _is_loopback_host(host):
            external_attempts.append(f"connect:{address!r}")
            raise AssertionError(f"pytest attempted external socket connection: {address!r}")
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", guarded_gethostbyname)
    monkeypatch.setattr(socket, "gethostbyname_ex", guarded_gethostbyname_ex)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)

    import run_once as ro
    import event_engine.bingx as bingx
    import event_engine.tracker as tracker
    import event_engine.reconcile_daily as reconciliation

    # Never let an accidentally unmocked notification path or inherited
    # credentials authorize a real message from a test.
    for name in (
        "TG_BOT_TOKEN", "TG_CHAT_IDS", "TG_CHAT_ID",
        "BINGX_API_KEY", "BINGX_SECRET_KEY",
        "COINALYZE_URL", "COINALYZE_P_SID", "COINALYZE_CHAT_SID",
    ):
        monkeypatch.delenv(name, raising=False)
    # Module constants are captured at import time, so clearing the environment
    # alone is insufficient when tests are collected before fixtures execute.
    # Force the exchange client and engine into a credential-free, no-execution
    # state; individual tests may opt in with explicit dummy values.
    monkeypatch.setattr(bingx, "API_KEY", "")
    monkeypatch.setattr(bingx, "SECRET_KEY", "")
    monkeypatch.setattr(ro, "API_KEY", "")
    monkeypatch.setattr(ro, "SECRET_KEY", "")
    monkeypatch.setattr(ro, "EXECUTION_ENABLED", False)
    monkeypatch.setattr(ro, "send_tg", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(tracker, "send_detailed", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(reconciliation, "send_detailed", lambda *_args, **_kwargs: {})

    # Keep runtime-state writes inside pytest's temporary directory. Rollback
    # safety logic intentionally persists unresolved cleanup records, and tests
    # must not leak those records into unrelated candidates or into the repo.
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", tmp_path / "active_trades.json")
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")
    monkeypatch.setattr(tracker, "NOTIFICATIONS_PATH", tmp_path / "notifications.json")

    # Keep ordinary unit/integration tests deterministic even when CI exports
    # production VST settings such as AJAY_SR_ROOM_MODE=enforce.
    monkeypatch.setattr(ro, "SR_3H_DIRECTIONAL_GATE_ENABLED", False)
    monkeypatch.setattr(ro, "AJAY_SR_ROOM_ENABLED", False)
    monkeypatch.setattr(ro, "AJAY_SR_ROOM_MODE", "off")
    monkeypatch.setattr(ro, "AJAY_SR_REQUIRE_DATA", False)
    # Tests that exercise candidate rejection/execution helpers must never append
    # to the checked-in runtime journal. Individual tests may override this path
    # with their own fixture when asserting journal contents.
    monkeypatch.setattr(ro, "ACTIONS", tmp_path / "actions.jsonl")
    # Default to the established per-symbol reconciliation path in tests. Tests
    # for the account-wide snapshot opt in explicitly with their own response,
    # so ordinary unit tests can never make an exchange/network request.
    monkeypatch.setattr(ro, "get_open_orders_snapshot", lambda **kwargs: {
        "status": "error", "error": "account-wide snapshot disabled by test fixture", "orders": [],
    })
    yield
    if external_attempts:
        raise AssertionError(
            "pytest attempted external network access (blocked before connection): "
            + "; ".join(external_attempts)
        )
