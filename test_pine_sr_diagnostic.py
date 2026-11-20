from __future__ import annotations

import sys
from pathlib import Path

import pytest

import pine_sr_diagnostic as diagnostic


def test_format_pine_price_preserves_raw_level_without_exchange_tick_rounding():
    assert diagnostic.format_pine_price(84875.2) == "84875.2"
    assert diagnostic.format_pine_price(81188.0) == "81188"
    assert diagnostic.format_pine_price(75690.2) == "75690.2"
    assert diagnostic.format_pine_price(None) == "None"


def _state():
    return {
        "source_symbol": "BTCUSDT",
        "history_to_keep": 20,
        "active_zones": [
            {"kind": "H", "top": 81800.0, "bottom": 81629.8, "center": 81714.9},
            {"kind": "H", "top": 84875.2, "bottom": 84701.7, "center": 84788.4},
            {"kind": "L", "top": 62000.0, "bottom": 61900.0, "center": 61950.0},
        ],
        "horizontal_sr": {
            "status": "OK",
            "high_level": 84875.2,
            "low_level": 61956.3,
            "zone_half_width": 472.4,
            "level_records": [
                {"level": 81188.0, "band_lower": 80715.6, "band_upper": 81660.4},
                {"level": 75690.2, "band_lower": 75217.8, "band_upper": 76162.6},
                {"level": 72691.0, "band_lower": 72218.6, "band_upper": 73163.4},
            ],
        },
    }


def test_main_prints_only_levels_and_all_zones_sorted_high_to_low(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["pine_sr_diagnostic.py", "--symbol", "BTCUSDT.P"])
    monkeypatch.setattr(diagnostic, "resolve_futures_symbol", lambda _symbol: {"source_symbol": "BTCUSDT"})
    snapshot_calls = []
    def get_snapshot(symbol, limit, include_horizontal_sr):
        snapshot_calls.append((symbol, limit, include_horizontal_sr))
        return _state()
    monkeypatch.setattr(diagnostic, "get_cached_sr_snapshot", get_snapshot)

    assert diagnostic.main() == 0
    assert snapshot_calls == [("BTCUSDT.P", diagnostic.DIAGNOSTIC_SOURCE_1H_LIMIT, True)]
    output = capsys.readouterr().out.splitlines()
    zone_lines = output
    levels = [
        float(line.split("=", 1)[1].split(" ", 1)[0])
        if line.startswith(("HIGH_LEVEL=", "LOW_LEVEL="))
        else float(line.split("level=", 1)[1].split(" ", 1)[0])
        for line in zone_lines
    ]
    assert any(line.startswith("HIGH_LEVEL=84875.2 | zone=") for line in zone_lines)
    assert any(line.startswith("LOW_LEVEL=61956.3 | zone=") for line in zone_lines)
    assert levels == sorted(levels, reverse=True)
    assert zone_lines[0].startswith("HIGH_LEVEL=84875.2")
    low_level_index = next(i for i, line in enumerate(zone_lines) if line.startswith("LOW_LEVEL="))
    lower_demand_index = next(i for i, line in enumerate(zone_lines) if line.startswith("DEMAND | level=61950"))
    assert low_level_index < lower_demand_index  # 61956.3 must precede the lower 61950 zone
    assert any(line.startswith("SUPPLY |") for line in zone_lines)
    assert any(line.startswith("DEMAND |") for line in zone_lines)
    assert any(line.startswith("HORIZONTAL_SR |") for line in zone_lines)
    assert len(output) == 8
    assert not list(tmp_path.iterdir())  # no CSV, report or other output files
    assert not any("GATE_" in line or "CSV" in line or "WARNING" in line for line in output)




def test_main_rejects_incomplete_pine_state_without_printing_partial_zones(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pine_sr_diagnostic.py", "--symbol", "BTCUSDT.P"])
    monkeypatch.setattr(diagnostic, "resolve_futures_symbol", lambda _symbol: {"source_symbol": "BTCUSDT"})
    state = _state()
    state["horizontal_sr"]["status"] = "INCOMPLETE_PIVOT_WARMUP_AT_LAST_REFRESH"
    monkeypatch.setattr(diagnostic, "get_cached_sr_snapshot", lambda *args, **kwargs: state)
    with pytest.raises(RuntimeError, match="Insufficient history.*INCOMPLETE_PIVOT_WARMUP"):
        diagnostic.main()
    assert capsys.readouterr().out == ""



def test_main_rejects_snapshot_from_different_futures_contract(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["pine_sr_diagnostic.py", "--symbol", "BTCUSDT.P"])
    monkeypatch.setattr(diagnostic, "resolve_futures_symbol", lambda _symbol: {"source_symbol": "BTCUSDT"})
    monkeypatch.setattr(diagnostic, "get_cached_sr_snapshot", lambda *args, **kwargs: {"source_symbol": "ETHUSDT"})
    with pytest.raises(RuntimeError, match="does not match"):
        diagnostic.main()
    assert capsys.readouterr().out == ""

def test_diagnostic_workflow_displays_to_logs_and_does_not_save_artifacts():
    diagnostic_workflow = Path(".github/workflows/pine-sr-diagnostic.yml").read_text(encoding="utf-8")
    engine_workflow = Path(".github/workflows/event-engine.yml").read_text(encoding="utf-8")
    assert 'default: "BTCUSDT.P"' in diagnostic_workflow
    assert 'python pine_sr_diagnostic.py --symbol "$DIAG_SYMBOL"' in diagnostic_workflow
    assert "actions/upload-artifact" not in diagnostic_workflow
    assert "pine-sr-zones.csv" not in diagnostic_workflow
    assert "pine-sr-clusters.csv" not in diagnostic_workflow
    assert "GITHUB_STEP_SUMMARY" not in diagnostic_workflow
    assert "Display Pine SR zones (highest to lowest)" in diagnostic_workflow
    assert 'print(json.dumps' not in diagnostic_workflow
    assert 'wireproxy" --version' not in diagnostic_workflow
    assert 'AJAY_SR_KLINE_LIMIT_1H: "1500"' in diagnostic_workflow
    assert 'AJAY_SR_ROOM_ENABLED' not in diagnostic_workflow
    assert 'AJAY_SR_ENTRY_BUFFER_PCT' not in diagnostic_workflow
    assert 'AJAY_SR_KLINE_LIMIT_1H: "1000"' in engine_workflow
    assert "WIREGUARD_CONF: ${{ secrets.WIREGUARD_CONF }}" in diagnostic_workflow
    assert "BINANCE_VPN_ENABLED=true" in diagnostic_workflow
    assert "BINANCE_HTTP_PROXY=http://127.0.0.1:18080" in diagnostic_workflow
    assert "WIREPROXY_SHA256: e88c1d090740373fc606c1bafd81d9a5eadc642cce5667616e20e9d7a444f51c" in diagnostic_workflow
