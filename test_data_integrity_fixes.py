"""Regression tests for data persistence and reconciliation analytics integrity."""
from __future__ import annotations

import json
import math
from dataclasses import fields

import pytest


def _coinalyze_row(raw_note: str = "x" * 900):
    from event_engine.coinalyze import CoinalyzeRow

    return CoinalyzeRow(
        symbol="BTC", name="Bitcoin", price=100.0, price_chg24=1.25,
        volume24=50_000_000.0, oi=15_000_000.0, oi_chg24_pct=-2.5,
        oi_chg4h_pct=0.5, oi_vol_ratio=0.3, oi_mktcap_ratio=0.1,
        fr_oiw=0.01, pfr_oiw=None, liq_short24=100_000.0,
        liq_long24=80_000.0, ls_accounts=1.1, btc_corr7d=0.8,
        cvd24=-10_000.0, lls24=2_000.0,
        raw={"price": 100.0, "missing": None, "nested": {"note": raw_note, "bad_number": float("nan")}},
    )


def test_journal_safe_preserves_full_dataclass_as_structured_json():
    import run_once as ro

    row = _coinalyze_row()
    safe = ro._journal_safe(row)
    expected_fields = {field.name for field in fields(row)}

    assert isinstance(safe, dict)
    assert set(safe) == expected_fields
    assert safe["symbol"] == "BTC"
    assert safe["pfr_oiw"] is None
    assert safe["raw"]["nested"]["note"] == "x" * 900
    assert safe["raw"]["nested"]["bad_number"] is None
    decoded = json.loads(json.dumps(safe, ensure_ascii=False, allow_nan=False))
    assert decoded["raw"]["nested"]["note"] == "x" * 900


def test_record_action_sanitizes_dataclass_and_nonfinite_numbers(tmp_path, monkeypatch):
    import run_once as ro

    journal = tmp_path / "actions.jsonl"
    monkeypatch.setattr(ro, "ACTIONS", journal)
    monkeypatch.setattr(ro, "ACTION_JOURNAL_ROTATE_BYTES", 80 * 1024 * 1024)
    row = _coinalyze_row()

    ro.record_action({
        "record_type": "CANDIDATE_FILTER_DECISION",
        "raw_market_features": row,
        "diagnostics": {"inf": float("inf"), "nan": float("nan")},
    })

    record = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(record["raw_market_features"], dict)
    assert record["raw_market_features"]["raw"]["nested"]["note"] == "x" * 900
    assert record["raw_market_features"]["raw"]["price"] == pytest.approx(100.0)
    assert record["diagnostics"] == {"inf": None, "nan": None}


def test_oi_snapshots_reject_nan_and_infinities(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import run_once as ro

    history_path = tmp_path / "oi_history.json"
    monkeypatch.setattr(ro, "OI_HISTORY", history_path)
    monkeypatch.setattr(ro, "_OI_HIST_CACHE", {"ts": 0.0, "data": {}, "path": ""})
    rows = [
        SimpleNamespace(symbol="VALID", oi=123.0),
        SimpleNamespace(symbol="NAN", oi=float("nan")),
        SimpleNamespace(symbol="POS_INF", oi=float("inf")),
        SimpleNamespace(symbol="NEG_INF", oi=float("-inf")),
        SimpleNamespace(symbol="ZERO", oi=0.0),
        SimpleNamespace(symbol="MISSING", oi=None),
    ]

    updated = ro._record_oi_snapshots(rows, 1_700_000_000_000)
    history = json.loads(history_path.read_text(encoding="utf-8"))

    assert updated == 1
    assert set(history) == {"VALID"}
    assert all(math.isfinite(float(v)) for values in history.values() for v in values.values())


def test_reconciled_setup_calculates_fraction_weighted_rr_not_max_rr():
    import run_once as ro

    setup = ro._build_reconciled_setup(
        avg_price=100.0,
        direction="LONG",
        risk_pct=7.0,
        risk_pct_source="LIVE_STOP_DISTANCE",
        invalidation_price=93.0,
        tp_levels=[
            {"leg": "tp1", "price": 104.55, "qty": 0.25},
            {"leg": "tp2", "price": 110.5, "qty": 0.75},
        ],
    )

    assert setup["target_rr"] == pytest.approx(1.5)
    assert setup["planned_weighted_rr"] == pytest.approx(1.2875)
    assert setup["effective_weighted_rr"] == pytest.approx(1.2875)
    assert [x["close_fraction"] for x in setup["effective_tp_levels"]] == pytest.approx([0.25, 0.75])
    assert setup["risk_pct_source"] == "LIVE_STOP_DISTANCE"
    assert setup["target_price"] == pytest.approx(110.5)


def test_reconciled_setup_computes_short_tp_distance_from_entry_price():
    import run_once as ro

    setup = ro._build_reconciled_setup(
        avg_price=100.0,
        direction="SHORT",
        risk_pct=7.0,
        risk_pct_source="LIVE_STOP_DISTANCE",
        invalidation_price=107.0,
        tp_levels=[
            {"leg": "tp1", "price": 95.45, "qty": 0.30},
            {"leg": "tp2", "price": 89.5, "qty": 0.70},
        ],
    )

    assert setup["target_rr"] == pytest.approx(1.5)
    assert setup["planned_weighted_rr"] == pytest.approx(1.245)
    assert setup["effective_tp_levels"][0]["pnl_pct"] == pytest.approx(4.55)
    assert setup["effective_tp_levels"][1]["pnl_pct"] == pytest.approx(10.5)
    assert setup["target_price"] == pytest.approx(89.5)


def test_reconciled_trade_open_uses_canonical_position_schema_for_readers(tmp_path, monkeypatch):
    import run_once as ro

    journal = tmp_path / "trades.jsonl"
    written = []
    monkeypatch.setattr(ro, "record_trade", lambda record: written.append(record))
    ts = 1_800_000_000_000
    setup = ro._build_reconciled_setup(
        avg_price=100.0,
        direction="SHORT",
        risk_pct=7.0,
        risk_pct_source="FIXED_STOP_POLICY_FALLBACK",
        invalidation_price=107.0,
        tp_levels=[
            {"leg": "tp1", "pnl_pct": 4.55, "close_fraction": 0.25},
            {"leg": "tp2", "pnl_pct": 10.5, "close_fraction": 0.75},
        ],
    )

    ro._record_reconciled_trade_open(
        "BTC", "SHORT", 100.0, 2.0,
        event_id="REC_BTC_SHORT_TEST",
        tp_orders=[{"orderId": "TP1"}, {"orderId": "TP2"}],
        sl_result={"order_id": "SL1"},
        setup=setup,
    )
    record = written[0]
    journal.write_text(json.dumps(record, allow_nan=False) + "\n", encoding="utf-8")

    assert record["record_type"] == "TRADE_OPEN"
    assert record["execution"]["status"] == "reconciled_adopted"
    assert record["result"]["status"] == "reconciled_adopted"
    assert record["result"]["position"]["positionAmt"] == pytest.approx(2.0)
    assert record["result"]["position"]["positionSide"] == "SHORT"
    assert record["setup"]["planned_weighted_rr"] == pytest.approx(1.2875)
    assert record["planned_metrics"]["effective_weighted_rr"] == pytest.approx(1.2875)
    assert record["entry_time_source"] == "RECONCILIATION_OBSERVED"

    now_ms = record["ts"] + 60_000
    entries = ro._load_recent_successful_entries(journal, now_ms, 15)
    assert entries["BTC"] == record["ts"]
    assert "REC_BTC_SHORT_TEST" in ro.load_successful_trade_ids(journal)


def test_shadow_health_counts_reconciled_position_as_confirmed_open(tmp_path):
    from event_engine.shadow import generate_shadow_health_snapshot

    events = tmp_path / "events.jsonl"
    trades = tmp_path / "trades.jsonl"
    events.write_text("", encoding="utf-8")
    trades.write_text(json.dumps({
        "record_type": "TRADE_OPEN",
        "trade_id": "TR_REC",
        "event_id": "REC_ETH_LONG",
        "symbol": "ETH",
        "direction": "LONG",
        "execution": {"status": "reconciled_adopted"},
        "result": {"status": "reconciled_adopted", "position": {"positionAmt": 1.0}},
    }) + "\n", encoding="utf-8")

    snapshot = generate_shadow_health_snapshot(events, trades)
    assert snapshot["trades"]["confirmed_open_records"] == 1
    assert snapshot["trades"]["opened"] == 1


def test_reconciliation_skips_nonfinite_exchange_position_fields(monkeypatch):
    import run_once as ro

    actions = []
    monkeypatch.setattr(ro, "get_positions", lambda **kwargs: [
        {"symbol": "BAD1-USDT", "positionSide": "LONG", "positionAmt": "NaN", "avgPrice": "100"},
        {"symbol": "BAD2-USDT", "positionSide": "SHORT", "positionAmt": "2", "avgPrice": "Infinity"},
    ])
    monkeypatch.setattr(ro, "_load_active_trades", lambda: {})
    monkeypatch.setattr(ro, "record_action", lambda row: actions.append(row))
    monkeypatch.setattr(
        ro, "get_open_protection_directional",
        lambda *args, **kwargs: pytest.fail("invalid exchange positions must be rejected before order queries"),
    )

    ro.reconcile_all_open_positions()

    assert len(actions) == 2
    assert all(row["record_type"] == "RECONCILIATION_POSITION_INVALID" for row in actions)
    assert all(row["reason"] == "NON_FINITE_OR_NON_POSITIVE_POSITION_FIELDS" for row in actions)


def test_common_jsonl_writer_sanitizes_nonfinite_values(tmp_path):
    import run_once as ro

    path = tmp_path / "trades.jsonl"
    ro.append_jsonl(path, {
        "record_type": "TRADE_CLOSE",
        "realized_pnl": float("nan"),
        "fee": float("inf"),
        "nested": {"funding": float("-inf")},
    })
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record == {
        "record_type": "TRADE_CLOSE",
        "realized_pnl": None,
        "fee": None,
        "nested": {"funding": None},
    }


def test_tracker_conversion_rejects_nonfinite_tp_and_sl_orders():
    import run_once as ro

    tp = ro._tp_orders_to_tracker([
        {"orderId": "TP-NAN", "stopPrice": float("nan"), "origQty": 1.0},
        {"orderId": "TP-INF", "stopPrice": float("inf"), "origQty": 1.0},
        {"orderId": "TP-QTY", "stopPrice": 105.0, "origQty": float("inf")},
        {"orderId": "TP-OK", "stopPrice": 105.0, "origQty": 1.0},
    ], direction="LONG", avg_price=100.0)
    assert len(tp) == 1
    assert tp[0]["order_id"] == "TP-OK"
    assert math.isfinite(tp[0]["price"])
    assert math.isfinite(tp[0]["qty"])

    assert ro._sl_order_to_tracker([
        {"orderId": "SL-NAN", "stopPrice": float("nan"), "origQty": 1.0},
    ]) == {}
    assert ro._sl_order_to_tracker([
        {"orderId": "SL-INF", "stopPrice": float("inf"), "origQty": 1.0},
    ]) == {}
    assert ro._sl_order_to_tracker([
        {"orderId": "SL-QTY", "stopPrice": 95.0, "origQty": float("inf")},
    ]) == {}


def test_atomic_json_writer_sanitizes_nonfinite_values(tmp_path):
    import run_once as ro

    path = tmp_path / "state.json"
    ro._save_json_atomic(path, {
        "open_interest": float("nan"),
        "nested": {"avg_price": float("inf")},
    })
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state == {"open_interest": None, "nested": {"avg_price": None}}


def test_tracker_and_shadow_state_writers_normalize_nonfinite_numbers(tmp_path, monkeypatch):
    import event_engine.tracker as tracker
    from event_engine.shadow import _save_state

    monkeypatch.setattr(tracker, "NOTIFICATIONS_PATH", tmp_path / "notifications.json")
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", tmp_path / "active_trades.json")
    monkeypatch.setattr(tracker, "TRADES_PATH", tmp_path / "trades.jsonl")

    tracker._save_notifications({"event": {"pnl": float("nan")}})
    tracker._save_active_trades({"event": {"stop_price": float("inf")}})
    tracker._append_trade_record({"record_type": "TRADE_CLOSE", "pnl": float("-inf")})
    _save_state(tmp_path / "shadow.json", {"event": {"score": float("nan")}})

    assert json.loads((tmp_path / "notifications.json").read_text(encoding="utf-8")) == {
        "event": {"pnl": None}
    }
    assert json.loads((tmp_path / "active_trades.json").read_text(encoding="utf-8")) == {
        "event": {"stop_price": None}
    }
    assert json.loads((tmp_path / "trades.jsonl").read_text(encoding="utf-8")) == {
        "record_type": "TRADE_CLOSE", "pnl": None
    }
    assert json.loads((tmp_path / "shadow.json").read_text(encoding="utf-8")) == {
        "event": {"score": None}
    }


def test_tracker_conversion_preserves_three_or_more_unlabelled_live_tp_orders():
    import run_once as ro

    orders = [
        {"orderId": "TP-105", "stopPrice": 105.0, "origQty": 1.0},
        {"orderId": "TP-110", "stopPrice": 110.0, "origQty": 1.0},
        {"orderId": "TP-115", "stopPrice": 115.0, "origQty": 1.0},
        {"orderId": "TP-120", "stopPrice": 120.0, "origQty": 1.0},
    ]
    tp = ro._tp_orders_to_tracker(orders, direction="LONG", avg_price=100.0)
    assert len(tp) == 4
    assert {row["order_id"] for row in tp} == {"TP-105", "TP-110", "TP-115", "TP-120"}
    assert len({row["leg"] for row in tp}) == 4
    assert {row["leg"] for row in tp[:3]} == {"tp1", "tp2", "tp3"}
    assert any(row["leg"].startswith("tp_extra_") for row in tp)


def test_tracker_conversion_uses_unused_leg_when_one_pending_tp_has_known_tp2():
    import run_once as ro

    tp = ro._tp_orders_to_tracker([
        {"orderId": "TP-KNOWN-2", "clientOrderId": "EVT_TP2_ABC", "stopPrice": 110.0, "origQty": 1.0},
        {"orderId": "TP-UNKNOWN", "stopPrice": 105.0, "origQty": 1.0},
    ], direction="LONG", avg_price=100.0)
    assert len(tp) == 2
    assert {row["order_id"] for row in tp} == {"TP-KNOWN-2", "TP-UNKNOWN"}
    assert len({row["leg"] for row in tp}) == 2
    assert next(row for row in tp if row["order_id"] == "TP-KNOWN-2")["leg"] == "tp2"
    assert next(row for row in tp if row["order_id"] == "TP-UNKNOWN")["leg"] == "tp1"


def test_active_trade_notification_and_shadow_snapshots_fsync_before_replace(tmp_path, monkeypatch):
    import os
    import stat
    import event_engine.tracker as tracker
    import event_engine.shadow as shadow

    active_path = tmp_path / "active_trades.json"
    notifications_path = tmp_path / "notifications.json"
    shadow_path = tmp_path / "shadow_state.json"
    monkeypatch.setattr(tracker, "ACTIVE_TRADES_PATH", active_path)
    monkeypatch.setattr(tracker, "NOTIFICATIONS_PATH", notifications_path)

    original_fsync = os.fsync
    regular_file_fsyncs = []

    def tracked_fsync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            regular_file_fsyncs.append(fd)
        return original_fsync(fd)

    monkeypatch.setattr(os, "fsync", tracked_fsync)
    tracker._save_active_trades({"EVT1": {"symbol": "TEST", "qty": 1.0}})
    tracker._save_notifications({"TG1": {"resolved": True, "chats": {}}})
    shadow._save_state(shadow_path, {"EVT1": {"status": "ACTIVE"}})

    assert len(regular_file_fsyncs) >= 3
    assert json.loads(active_path.read_text(encoding="utf-8"))["EVT1"]["qty"] == 1.0
    assert json.loads(notifications_path.read_text(encoding="utf-8"))["TG1"]["resolved"] is True
    assert json.loads(shadow_path.read_text(encoding="utf-8"))["EVT1"]["status"] == "ACTIVE"
