"""Pytest isolation for production-only runtime gates.

The GitHub Actions workflow sets production VST environment variables for the
whole job. Unit tests must not inherit those execution gates implicitly: tests
that are specifically about S/R explicitly enable the S/R gate themselves.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_production_entry_gates(monkeypatch, tmp_path):
    import run_once as ro
    import event_engine.tracker as tracker

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
