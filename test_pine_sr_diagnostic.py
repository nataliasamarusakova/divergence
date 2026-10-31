from __future__ import annotations

import sys

import pytest

import pine_sr_diagnostic as diagnostic


@pytest.mark.parametrize(
    ("value", "tick_size", "expected"),
    [
        (None, 0.05, "None"),
        (1.234, 0.01, "1.23"),
        # Tick size is a multiple, not only a number of decimal places.
        (1.234, 0.05, "1.25"),
        (1.225, 0.05, "1.25"),
        (1.22, 0.05, "1.20"),
        (123.4, 5.0, "125"),
        (1.234, None, "1.234"),
        (1.234, 0.0, "1.234"),
    ],
)
def test_format_price(value, tick_size, expected):
    assert diagnostic.format_price(value, tick_size) == expected


def test_source_tick_size_uses_price_filter_and_logical_price_scale(monkeypatch):
    import event_engine.sr_context as sr

    monkeypatch.setattr(
        sr,
        "_spot_symbol_catalog",
        lambda: {
            "SHIBUSDT": {
                "filters": [
                    {"filterType": "LOT_SIZE", "tickSize": "0.1"},
                    {"filterType": "PRICE_FILTER", "tickSize": "0.00001"},
                ]
            }
        },
    )
    assert diagnostic._source_tick_size(
        {"source_symbol": "SHIBUSDT", "source_price_scale": 1000.0}
    ) == pytest.approx(0.01)


def test_source_tick_size_returns_none_when_price_filter_is_missing(monkeypatch):
    import event_engine.sr_context as sr

    monkeypatch.setattr(sr, "_spot_symbol_catalog", lambda: {"FOOUSDT": {"filters": []}})
    assert diagnostic._source_tick_size(
        {"source_symbol": "FOOUSDT", "source_price_scale": 1.0}
    ) is None


def test_source_tick_size_fails_closed_on_catalog_error(monkeypatch):
    import event_engine.sr_context as sr

    def fail():
        raise RuntimeError("offline test")

    monkeypatch.setattr(sr, "_spot_symbol_catalog", fail)
    assert diagnostic._source_tick_size(
        {"source_symbol": "FOOUSDT", "source_price_scale": 1.0}
    ) is None


def test_main_prints_resolved_snapshot_and_exports_zone_bands_without_network(monkeypatch, capsys, tmp_path):
    output_csv = tmp_path / "zones.csv"
    monkeypatch.setattr(sys, "argv", ["pine_sr_diagnostic.py", "--symbol", "BTC-USDT", "--limit", "1000", "--output", str(output_csv)])
    monkeypatch.setattr(
        diagnostic,
        "resolve_spot_symbol",
        lambda symbol: {
            "requested_symbol": "BTCUSDT",
            "source_symbol": "BTCUSDT",
            "source_alias_kind": "EXACT",
            "source_price_scale": 1.0,
            "source_volume_scale": 1.0,
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "get_cached_sr_snapshot",
        lambda symbol, limit: {"highestph": 100.0, "levels": [99.5, 98.75], "lowestpl": 97.0, "cwidth": 0.02, "zone_scale": 1.0, "timeframe": "3h", "source": "binance_spot", "source_symbol": "BTCUSDT", "accepted": [{"price": 99.5, "kind": "L", "points": 3, "countpp": 1}], "latest_closed_timestamp": 0, "closed_candle_count_3h": 333, "latest_closed_age_min": 12.0},
    )
    monkeypatch.setattr(diagnostic, "_source_tick_size", lambda resolved: 0.25)
    monkeypatch.setattr(diagnostic, "get_spot_current_price", lambda symbol, snapshot: 100.0)

    assert diagnostic.main() == 0
    output = capsys.readouterr().out
    assert "REQUESTED=BTCUSDT" in output
    assert "SOURCE_SYMBOL=BTCUSDT" in output
    assert "TIMEFRAME=3h | SOURCE_KLINE_INTERVAL=1h_AGGREGATED" in output
    assert "HIGH LEVEL: 100.00" in output
    assert "99.50" in output
    assert "98.75" in output
    assert "LOW LEVEL: 97.00" in output
    assert "CWIDTH: 0.02" in output
    assert "DEMAND | center=99.50 | band=[99.50, 99.50]" in output
    assert "CLOSED_3H_BARS=333" in output
    assert "GATE_LONG=ALLOW REASON=ALLOWED" in output
    assert "GATE_SHORT=REJECT REASON=DIRECTION_CONTRADICTS_CURRENT_ZONE" in output
    assert "ZONE_COUNT=1" in output
    assert output_csv.exists()
    import csv
    with output_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["zone_type"] == "DEMAND"
    assert float(rows[0]["center"]) == pytest.approx(99.5)
    assert rows[0]["long_gate_decision"] == "ALLOW"
    assert rows[0]["long_gate_reason"] == "ALLOWED"
    assert rows[0]["long_selected"] == "True"
    assert rows[0]["short_gate_decision"] == "REJECT"
    assert rows[0]["short_gate_reason"] == "DIRECTION_CONTRADICTS_CURRENT_ZONE"
    assert rows[0]["short_selected"] == "True"


def test_main_rejects_non_production_source_window(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pine_sr_diagnostic.py", "--symbol", "BTC-USDT", "--limit", "999"])
    with pytest.raises(SystemExit) as exc:
        diagnostic.main()
    assert exc.value.code == 2


def test_diagnostic_uses_the_exact_production_zone_builder():
    import event_engine.sr_context as sr
    assert diagnostic._build_zones is sr._build_zones


def test_main_fails_when_current_spot_price_is_unavailable(monkeypatch, tmp_path):
    output_csv = tmp_path / "must-not-exist.csv"
    monkeypatch.setattr(sys, "argv", ["pine_sr_diagnostic.py", "--symbol", "BTC-USDT", "--limit", "1000", "--output", str(output_csv)])
    monkeypatch.setattr(diagnostic, "resolve_spot_symbol", lambda symbol: {
        "requested_symbol": "BTCUSDT", "source_symbol": "BTCUSDT", "source_alias_kind": "EXACT",
        "source_price_scale": 1.0, "source_volume_scale": 1.0,
    })
    monkeypatch.setattr(diagnostic, "get_cached_sr_snapshot", lambda symbol, limit: {
        "timeframe": "3h", "highestph": 1.0, "levels": [], "lowestpl": 0.5, "cwidth": 0.01,
        "accepted": [], "latest_closed_timestamp": 0, "closed_candle_count_3h": 333,
    })
    monkeypatch.setattr(diagnostic, "get_spot_current_price", lambda symbol, snapshot: (_ for _ in ()).throw(RuntimeError("ticker unavailable")))
    with pytest.raises(RuntimeError, match="ticker unavailable"):
        diagnostic.main()
    assert not output_csv.exists()


def test_diagnostic_workflow_uses_production_sr_parameters_and_source_window():
    from pathlib import Path
    import re

    diagnostic_workflow = Path(".github/workflows/pine-sr-diagnostic.yml").read_text(encoding="utf-8")
    engine_workflow = Path(".github/workflows/event-engine.yml").read_text(encoding="utf-8")

    def read_sr_env(text):
        values = {}
        for key, value in re.findall(r"^\s+(AJAY_SR_[A-Z0-9_]+):\s*[\"']?([^\"'\n]+)[\"']?\s*$", text, flags=re.MULTILINE):
            if key != "AJAY_SR_HTTP_PROXY":
                values[key] = value.strip()
        return values

    diagnostic_values = read_sr_env(diagnostic_workflow)
    engine_values = read_sr_env(engine_workflow)
    assert diagnostic_values == engine_values
    assert 'python pine_sr_diagnostic.py --symbol "$DIAG_SYMBOL" --limit 1000' in diagnostic_workflow
    assert "WIREGUARD_CONF: ${{ secrets.WIREGUARD_CONF }}" in diagnostic_workflow
    assert "AJAY_SR_HTTP_PROXY=http://127.0.0.1:18080" in diagnostic_workflow
    assert "WIREPROXY_SHA256: e88c1d090740373fc606c1bafd81d9a5eadc642cce5667616e20e9d7a444f51c" in diagnostic_workflow
