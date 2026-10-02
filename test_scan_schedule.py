import run_once


def test_same_five_minute_window_does_not_advance_completed_buckets():
    # With the production 2-minute close grace, 13:02 and 13:12 UTC belong
    # to the same last safely-completed 1H and 4H buckets used by scan state.
    a = 13 * 3_600_000 + 2 * 60_000
    b = 13 * 3_600_000 + 12 * 60_000
    assert run_once._completed_bucket(3_600_000, a, 2.0) == 12
    assert run_once._completed_bucket(3_600_000, b, 2.0) == 12
    assert run_once._completed_bucket(14_400_000, a, 2.0) == 2
    assert run_once._completed_bucket(14_400_000, b, 2.0) == 2


def test_symbol_scan_due_false_when_watermark_matches_bucket():
    state = {"version": 2, "symbols": {"FIL": {"1h": 12, "4h": 2}}}
    assert run_once._symbol_scan_due(state, "FIL", "1h", 12) is False
    assert run_once._symbol_scan_due(state, "FIL", "4h", 2) is False
    assert run_once._symbol_scan_due(state, "FIL", "1h", 13) is True
    assert run_once._symbol_scan_due(state, "FIL", "4h", 3) is True


def test_timeframe_scan_state_loads_persisted_v2_symbols(tmp_path, monkeypatch):
    state_path = tmp_path / "timeframe_scan_state.json"
    state_path.write_text(
        '{"version": 2, "symbols": {"FIL": {"1h": 12, "4h": 2}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(run_once, "TIMEFRAME_STATE", state_path)

    state = run_once._load_timeframe_scan_state()

    assert state["version"] == 2
    assert state["symbols"]["FIL"]["1h"] == 12
    assert state["symbols"]["FIL"]["4h"] == 2


def test_timeframe_scan_state_missing_file_is_explicitly_empty(tmp_path, monkeypatch):
    state_path = tmp_path / "missing.json"
    monkeypatch.setattr(run_once, "TIMEFRAME_STATE", state_path)

    state = run_once._load_timeframe_scan_state()

    assert state == {"version": 2, "symbols": {}}


def test_timeframe_scan_state_malformed_file_is_explicitly_empty(tmp_path, monkeypatch):
    state_path = tmp_path / "broken.json"
    state_path.write_text('{broken', encoding="utf-8")
    monkeypatch.setattr(run_once, "TIMEFRAME_STATE", state_path)

    state = run_once._load_timeframe_scan_state()

    assert state == {"version": 2, "symbols": {}}


def test_timeframe_scan_state_unsupported_version_is_explicitly_empty(tmp_path, monkeypatch):
    state_path = tmp_path / "future.json"
    state_path.write_text(
        '{"version": 99, "symbols": {"FIL": {"1h": 12, "4h": 2}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(run_once, "TIMEFRAME_STATE", state_path)

    state = run_once._load_timeframe_scan_state()

    assert state == {"version": 2, "symbols": {}}
