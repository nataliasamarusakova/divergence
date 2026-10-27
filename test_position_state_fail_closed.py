"""Regression tests: an unknown BingX position response must never mean flat."""

import os
import subprocess
import sys
from pathlib import Path


def test_position_reader_rejects_unrecognized_payload_instead_of_not_found(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": {"unexpected": []}})
    result = bingx.get_position_directional("BTC", "LONG")
    assert result["status"] == "error"
    assert result["status"] != "not_found"
    assert "unrecognized_data_shape" in result["error"]


def test_position_reader_keeps_active_position_found_when_avg_price_is_missing(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": [{
        "symbol": "BTC-USDT", "positionSide": "LONG", "positionAmt": "0.25",
    }]})
    result = bingx.get_position_directional("BTC", "LONG")
    assert result["status"] == "found"
    assert result["positionAmt"] == 0.25
    assert result["avgPrice"] == 0.0
    assert result["price_status"] == "unknown"


def test_emergency_close_does_not_report_closed_when_initial_position_read_errors(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "POSITION_MODE", "HEDGE")
    monkeypatch.setattr(bingx, "get_position_mode", lambda: {"status": "ok", "mode": "HEDGE"})
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "get_contract", lambda _symbol: {"quantityPrecision": 3})
    monkeypatch.setattr(bingx, "get_position_directional", lambda *a, **k: {
        "status": "error", "error": "simulated API response parsing failure",
    })
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not POST")))

    result = bingx.emergency_close_position("BTC", "LONG", 0.25, reason_token="TEST_UNKNOWN_POSITION")
    assert result["status"] == "unknown"
    assert result["escalation_required"] is True
    assert "simulated API response parsing failure" in result["error"]


def test_emergency_close_does_not_report_closed_when_post_close_read_errors(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "POSITION_MODE", "HEDGE")
    monkeypatch.setattr(bingx, "get_position_mode", lambda: {"status": "ok", "mode": "HEDGE"})
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "get_contract", lambda _symbol: {"quantityPrecision": 3})
    states = iter([
        {"status": "found", "positionAmt": 0.25, "avgPrice": 100.0},
        {"status": "error", "error": "simulated verification timeout"},
    ])
    monkeypatch.setattr(bingx, "get_position_directional", lambda *a, **k: next(states))
    monkeypatch.setattr(bingx, "_request", lambda method, path, params=None, **kwargs: {
        "code": 0, "data": {"order": {"avgPrice": "99.8"}},
    } if method == "POST" else {"code": 0, "data": {"dualSidePosition": True}})

    result = bingx.emergency_close_position("BTC", "LONG", 0.25, reason_token="TEST_POST_VERIFY")
    assert result["status"] == "unknown"
    assert result["escalation_required"] is True
    assert "verification" in result["error"] or "state is unknown" in result["error"]


def test_emergency_close_requires_two_explicit_not_found_reads(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "POSITION_MODE", "HEDGE")
    monkeypatch.setattr(bingx, "get_position_mode", lambda: {"status": "ok", "mode": "HEDGE"})
    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "get_contract", lambda _symbol: {"quantityPrecision": 3})
    calls = []
    states = iter([
        {"status": "not_found"},
        {"status": "not_found"},
    ])
    monkeypatch.setattr(bingx, "get_position_directional", lambda *a, **k: calls.append(1) or next(states))
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not POST")))
    monkeypatch.setattr(bingx.time, "sleep", lambda *_: None)

    result = bingx.emergency_close_position("BTC", "LONG", 0.25, reason_token="TEST_ABSENT")
    assert result["status"] == "closed"
    assert result["verification"] == "two_not_found_reads"
    assert len(calls) == 2


def test_research_unlimited_mode_overrides_all_entry_cooldown_and_quarantine_env(monkeypatch):
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.update({
        # A stale explicit false must not override the VST research invariant.
        "RESEARCH_UNLIMITED_ENTRIES": "false",
        "EXECUTION_MODE": "vst",
        "MAX_TRADES_PER_CYCLE": "2",
        "MAX_ACTIVE_TRADES": "12",
        "MAX_ACTIVE_LONGS": "6",
        "MAX_ACTIVE_SHORTS": "6",
        "PORTFOLIO_CAP_ENABLED": "true",
        "SYMBOL_ENTRY_COOLDOWN_MIN": "25",
        "SQUEEZE_SYMBOL_ENTRY_COOLDOWN_MIN": "45",
        "SYMBOL_LOSS_COOLDOWN_MIN": "360",
        "SYMBOL_MAX_CONSECUTIVE_LOSSES": "3",
        "SYMBOL_QUARANTINE_MIN": "360",
    })
    code = (
        "import run_once; "
        "assert run_once.RESEARCH_UNLIMITED_ENTRIES is True; "
        "assert run_once.MAX_TRADES == 0; "
        "assert run_once.PORTFOLIO_CAP_ENABLED is False; "
        "assert run_once.MAX_ACTIVE_TRADES == 0; "
        "assert run_once.MAX_ACTIVE_LONGS == 0; "
        "assert run_once.MAX_ACTIVE_SHORTS == 0; "
        "assert run_once.SYMBOL_ENTRY_COOLDOWN_MIN == 0; "
        "assert run_once.SQUEEZE_SYMBOL_ENTRY_COOLDOWN_MIN == 0; "
        "assert run_once.SYMBOL_LOSS_COOLDOWN_MIN == 0; "
        "assert run_once.SYMBOL_MAX_CONSECUTIVE_LOSSES == 0; "
        "assert run_once.SYMBOL_QUARANTINE_MIN == 0"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr or result.stdout


def test_account_position_reader_rejects_unrecognized_payload_instead_of_empty_list(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": {"unexpected": []}})
    try:
        bingx.get_positions()
    except RuntimeError as exc:
        assert "unrecognized data shape" in str(exc)
    else:
        raise AssertionError("unknown positions response must not become an empty position list")


def test_account_position_reader_accepts_explicit_empty_collection(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": []})
    assert bingx.get_positions() == []


def test_protection_reader_rejects_unrecognized_payload_instead_of_no_protection(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": {"unexpected": []}})
    result = bingx.get_open_protection_directional("BTC", "LONG")
    assert result["status"] == "error"
    assert "unrecognized data shape" in result["error"]


def test_protection_reader_accepts_explicit_empty_order_collection(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": {"orders": []}})
    result = bingx.get_open_protection_directional("BTC", "LONG")
    assert result["status"] == "ok"
    assert result["tp_orders"] == []
    assert result["sl_orders"] == []


def test_protection_reader_rejects_unidentifiable_order_row(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "to_bx_symbol", lambda _symbol: "BTC-USDT")
    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": [{"type": "LIMIT"}]})
    result = bingx.get_open_protection_directional("BTC", "LONG")
    assert result["status"] == "error"
    assert "not identifiable" in result["error"]


def test_account_wide_open_orders_rejects_rows_missing_type_or_symbol(monkeypatch):
    from event_engine import bingx

    monkeypatch.setattr(bingx, "_request", lambda *a, **k: {"code": 0, "data": [{"orderId": "1", "symbol": "BTC-USDT"}]})
    result = bingx.get_open_orders_snapshot()
    assert result["status"] == "error"
    assert "missing type or symbol" in result["error"]
