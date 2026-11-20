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


def test_source_tick_size_uses_futures_exchange_price_filter(monkeypatch):
    import event_engine.binance as bn

    monkeypatch.setattr(bn, "refresh_exchange_info", lambda: {
        "PIEVERSEUSDT": {
            "filters": [
                {"filterType": "LOT_SIZE", "tickSize": "0.1"},
                {"filterType": "PRICE_FILTER", "tickSize": "0.0001"},
            ]
        }
    })
    assert diagnostic._source_tick_size({"source_symbol": "PIEVERSEUSDT"}) == pytest.approx(0.0001)


def test_format_pine_price_suppresses_insignificant_trailing_zeroes():
    assert diagnostic.format_pine_price(84875.2, 0.1) == "84875.2"
    assert diagnostic.format_pine_price(81188.0, 0.1) == "81188"
    assert diagnostic.format_pine_price(75690.2, 0.1) == "75690.2"


def test_source_tick_size_returns_none_when_price_filter_is_missing(monkeypatch):
    import event_engine.binance as bn

    monkeypatch.setattr(bn, "refresh_exchange_info", lambda: {"FOOUSDT": {"filters": []}})
    assert diagnostic._source_tick_size({"source_symbol": "FOOUSDT"}) is None


def test_source_tick_size_fails_closed_on_catalog_error(monkeypatch):
    import event_engine.binance as bn

    def fail():
        raise RuntimeError("offline test")

    monkeypatch.setattr(bn, "refresh_exchange_info", fail)
    assert diagnostic._source_tick_size({"source_symbol": "FOOUSDT"}) is None


def test_main_prints_resolved_snapshot_and_exports_zone_bands_without_network(monkeypatch, capsys, tmp_path):
    output_csv = tmp_path / "zones.csv"
    cluster_csv = tmp_path / "clusters.csv"
    monkeypatch.setattr(sys, "argv", ["pine_sr_diagnostic.py", "--symbol", "BTCUSDT.P", "--limit", "1000", "--output", str(output_csv), "--cluster-output", str(cluster_csv)])
    monkeypatch.setattr(
        diagnostic,
        "resolve_futures_symbol",
        lambda symbol: {
            "requested_symbol": "BTCUSDT",
            "source_symbol": "BTCUSDT",
            "source_alias_kind": "EXACT_FUTURES",
            "contract_type": "PERPETUAL",
            "quote_asset": "USDT",
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "get_cached_sr_snapshot",
        lambda symbol, limit, include_horizontal_sr=True: {
            "highestph": 100.0, "levels": [99.5, 98.75], "lowestpl": 97.0,
            "cwidth": 0.02, "zone_scale": 1.0, "timeframe": "3h",
            "source": "binance_usds_m_futures", "source_symbol": "BTCUSDT",
            "active_zones": [{"top": 99.5, "bottom": 99.5, "center": 99.5, "kind": "L", "points": 1, "countpp": 1, "box_width": 0.0}],
            "supply_zones": [], "demand_zones": [], "latest_closed_close": 99.8,
            "latest_closed_timestamp": 0, "closed_candle_count_3h": 333,
            "latest_closed_age_min": 12.0, "atr": 0.08,
            "horizontal_sr": {
                "algorithm": "AJAY_R541_HORIZONTAL_SR_CLUSTERS", "status": "OK",
                "high_level": 100.0, "low_level": 97.0,
                "levels": [99.0, 98.0], "zone_half_width": 0.5, "zone_width": 1.0,
                "cluster_half_width_at_update": 2.0, "latest_update_timestamp": 0,
                "latest_update_bars_ago": 1,
                "level_records": [
                    {"slot": 1, "level": 99.0, "band_lower": 98.5, "band_upper": 99.5,
                     "points": 3, "pivot_event_count": 2, "candidate_kind": "HIGH",
                     "candidate_pivot_timestamp": 0, "candidate_confirmation_timestamp": 0,
                     "cluster_lower": 97.0, "cluster_upper": 101.0, "cluster_half_width": 2.0}
                ],
            },
        },
    )
    monkeypatch.setattr(diagnostic, "_source_tick_size", lambda resolved: 0.25)
    monkeypatch.setattr(diagnostic, "get_futures_current_price", lambda symbol, snapshot: 100.0)

    assert diagnostic.main() == 0
    output = capsys.readouterr().out
    assert "INPUT_SYMBOL=BTCUSDT.P" in output
    assert "REQUESTED=BTCUSDT | SOURCE=BINANCE_USDS_M_FUTURES" in output
    assert "TRADINGVIEW_SYMBOL=BINANCE:BTCUSDT.P" in output
    assert "SOURCE_SYMBOL=BTCUSDT" in output
    assert "TIMEFRAME=3h | SOURCE_KLINE_INTERVAL=1h_AGGREGATED" in output
    assert "PINE SUBSYSTEM 1: SWING DEMAND / SUPPLY BOXES" in output
    assert "PINE SUBSYSTEM 2: HORIZONTAL SUPPORT / RESISTANCE CLUSTERS" in output
    assert "HIGH_LEVEL=100" in output
    assert "LOW_LEVEL=97" in output
    assert "SR_CLUSTER | slot=01 | level=99" in output
    assert "VISIBLE_ZONE_HALF_WIDTH_300BARS=0.5" in output
    assert "DEMAND | center=99.50 | band=[99.50, 99.50]" in output
    assert "CLOSED_3H_BARS=333" in output
    assert "GATE_LONG=ALLOW REASON=ALLOWED" in output
    assert "GATE_SHORT=REJECT REASON=DIRECTION_CONTRADICTS_CURRENT_ZONE" in output
    assert "DEMAND_SUPPLY_ZONE_COUNT=1" in output
    assert output_csv.exists()
    assert cluster_csv.exists()
    import csv
    with output_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["zone_type"] == "DEMAND"
    assert float(rows[0]["center"]) == pytest.approx(99.5)
    assert float(rows[0]["box_width"]) == pytest.approx(0.0)
    assert rows[0]["long_gate_decision"] == "ALLOW"
    assert rows[0]["long_gate_reason"] == "ALLOWED"
    assert rows[0]["long_selected"] == "True"
    assert rows[0]["short_gate_decision"] == "REJECT"
    assert rows[0]["short_gate_reason"] == "DIRECTION_CONTRADICTS_CURRENT_ZONE"
    assert rows[0]["short_selected"] == "True"
    with cluster_csv.open(encoding="utf-8", newline="") as handle:
        cluster_rows = list(csv.DictReader(handle))
    assert len(cluster_rows) == 3  # one SR cluster plus High Level and Low Level
    assert cluster_rows[0]["record_type"] == "SR_CLUSTER"
    assert cluster_rows[0]["tradingview_symbol"] == "BINANCE:BTCUSDT.P"
    assert float(cluster_rows[0]["level"]) == pytest.approx(99.0)
    assert float(cluster_rows[0]["band_lower"]) == pytest.approx(98.5)
    assert {row["record_type"] for row in cluster_rows[1:]} == {"HIGH_LEVEL", "LOW_LEVEL"}


def test_main_rejects_non_production_source_window(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pine_sr_diagnostic.py", "--symbol", "BTCUSDT.P", "--limit", "999"])
    with pytest.raises(SystemExit) as exc:
        diagnostic.main()
    assert exc.value.code == 2


def test_diagnostic_uses_the_exact_production_zone_builder():
    import event_engine.sr_context as sr
    assert diagnostic._build_zones is sr._build_zones


def test_main_fails_when_current_futures_price_is_unavailable(monkeypatch, tmp_path):
    output_csv = tmp_path / "must-not-exist.csv"
    monkeypatch.setattr(sys, "argv", ["pine_sr_diagnostic.py", "--symbol", "BTC-USDT", "--limit", "1000", "--output", str(output_csv)])
    monkeypatch.setattr(diagnostic, "resolve_futures_symbol", lambda symbol: {
        "requested_symbol": "BTCUSDT", "source_symbol": "BTCUSDT", "source_alias_kind": "EXACT_FUTURES",
        "contract_type": "PERPETUAL", "quote_asset": "USDT",
    })
    monkeypatch.setattr(diagnostic, "get_cached_sr_snapshot", lambda symbol, limit, include_horizontal_sr=True: {
        "timeframe": "3h", "highestph": 1.0, "levels": [], "lowestpl": 0.5, "cwidth": 0.01,
        "active_zones": [], "accepted": [], "latest_closed_timestamp": 0, "closed_candle_count_3h": 333, "atr": 0.01,
    })
    # This test targets the unavailable ticker path; source tick-size lookup is
    # unrelated and must not trigger a real Binance Futures exchangeInfo request.
    monkeypatch.setattr(diagnostic, "_source_tick_size", lambda _resolved: None)
    monkeypatch.setattr(diagnostic, "get_futures_current_price", lambda symbol, snapshot: (_ for _ in ()).throw(RuntimeError("ticker unavailable")))
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
            values[key] = value.strip()
        return values

    diagnostic_values = read_sr_env(diagnostic_workflow)
    engine_values = read_sr_env(engine_workflow)
    assert diagnostic_values == engine_values
    assert "AJAY_SR_SPOT_BASE_URL" not in diagnostic_workflow
    assert "AJAY_SR_SPOT_BASE_URL" not in engine_workflow
    assert 'default: "BTCUSDT.P"' in diagnostic_workflow
    assert 'python pine_sr_diagnostic.py --symbol "$DIAG_SYMBOL" --limit 1000 --output pine-sr-zones.csv --cluster-output pine-sr-clusters.csv' in diagnostic_workflow
    assert 'pine-sr-clusters.csv' in diagnostic_workflow
    assert "WIREGUARD_CONF: ${{ secrets.WIREGUARD_CONF }}" in diagnostic_workflow
    assert "BINANCE_VPN_ENABLED=true" in diagnostic_workflow
    assert "BINANCE_HTTP_PROXY=http://127.0.0.1:18080" in diagnostic_workflow
    assert "WIREPROXY_SHA256: e88c1d090740373fc606c1bafd81d9a5eadc642cce5667616e20e9d7a444f51c" in diagnostic_workflow
