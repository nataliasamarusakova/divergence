"""Pytest isolation for production-only runtime gates.

The GitHub Actions workflow sets production VST environment variables for the
whole job. Unit tests must not inherit those execution gates implicitly: tests
that are specifically about S/R explicitly enable the S/R gate themselves.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_production_entry_gates(monkeypatch):
    import run_once as ro

    # Keep ordinary unit/integration tests deterministic even when CI exports
    # production VST settings such as AJAY_SR_ROOM_MODE=enforce.
    monkeypatch.setattr(ro, "AJAY_SR_ROOM_ENABLED", False)
    monkeypatch.setattr(ro, "AJAY_SR_ROOM_MODE", "off")
    monkeypatch.setattr(ro, "AJAY_SR_REQUIRE_DATA", False)
    yield
