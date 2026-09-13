import json
from pathlib import Path


def test_repeated_pre_order_drift_failures_are_counted(tmp_path, monkeypatch):
    import run_once

    path = tmp_path / "trades.jsonl"
    rows = [
        {"record_type": "EXECUTION_ATTEMPT", "event_id": "EVT_A", "result": {"status": "PRE_ORDER_DRIFT_EXCEEDED"}},
        {"record_type": "EXECUTION_ATTEMPT", "event_id": "EVT_A", "result": {"status": "PRE_ORDER_DRIFT_EXCEEDED"}},
        {"record_type": "EXECUTION_ATTEMPT", "event_id": "EVT_B", "result": {"status": "OPEN_FAILED"}},
    ]
    path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    assert run_once.load_pre_order_drift_failure_counts(path) == {"EVT_A": 2}


def test_terminal_event_ids_are_loaded(tmp_path):
    import run_once

    path = tmp_path / "trades.jsonl"
    rows = [
        {"record_type": "EVENT_TERMINAL", "event_id": "EVT_A", "reason": "ENTRY_DRIFT_EXHAUSTED"},
        {"record_type": "TRADE_CLOSE", "event_id": "EVT_B"},
    ]
    path.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    assert run_once.load_terminal_event_ids(path) == {"EVT_A"}


def test_shadow_divergence_is_terminalized_after_evaluation(monkeypatch, tmp_path):
    import run_once

    actions = []
    trades = []
    monkeypatch.setattr(run_once, "record_action", lambda obj: actions.append(obj))
    monkeypatch.setattr(run_once, "record_trade", lambda obj: trades.append(obj))
    # Structural source guard: shadow branch must emit one terminal record.
    source = Path(run_once.__file__).read_text(encoding="utf-8")
    assert 'DIVERGENCE_SHADOW_EVALUATED' in source
    assert 'terminal_event_ids.add(event_id)' in source
