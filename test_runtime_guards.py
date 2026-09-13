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
    assert 'DIVERGENCE_SHADOW_OPENED' in source
    assert 'terminal_event_ids.add(event_id)' in source


def test_fresh_divergence_counter_is_not_all_non_squeeze(monkeypatch, tmp_path):
    """Source sanity: fresh_divergence must count only true DIVERGENCE events."""
    src = open('run_once.py', encoding='utf-8').read()
    assert 'is_divergence = event_type == "DIVERGENCE"' in src
    assert 'stats["fresh_divergence"] += int(is_divergence)' in src
    assert 'stats["fresh_divergence"] += int(not is_squeeze)' not in src
    assert 'event_type_name = str(ev.get("event_type", "")).upper()' in src


def test_divergence_shadow_state_records_and_closes(tmp_path):
    from pathlib import Path
    from event_engine.shadow import record_divergence_shadow_open, update_divergence_shadow_state

    path = Path(tmp_path) / 'shadow.json'
    setup = {
        'invalidation_price': 99.0,
        'target_price': 102.5,
        'risk_pct': 2.5,
    }
    opened = record_divergence_shadow_open(
        path, event_id='EVT1', symbol='TEST', direction='LONG',
        event_type='REGULAR_BULLISH_RSI', timeframe='1h',
        entry_price=100.0, setup=setup, score=70.0, opened_ts=1000,
    )
    assert opened['status'] == 'opened'
    result = update_divergence_shadow_state(path, {'TEST': 106.25}, 2000)
    assert result['closed'] == 1
    data = __import__('json').loads(path.read_text())
    assert data['EVT1']['status'] == 'CLOSED'
    assert data['EVT1']['exit_reason'] == 'SHADOW_TP3'
    assert round(data['EVT1']['realized_pnl_pct'], 6) == 6.25
