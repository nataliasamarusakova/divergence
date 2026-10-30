from __future__ import annotations

import pytest


def _snapshot(*zones, cwidth=0.5, timeframe="3h"):
    accepted = [
        {"price": float(center), "kind": kind, "points": 3, "countpp": i + 1}
        for i, (center, kind) in enumerate(zones)
    ]
    return {
        "source": "binance_spot",
        "timeframe": timeframe,
        "interval_ms": 10_800_000,
        "source_symbol": "TESTUSDT",
        "source_price_scale": 1.0,
        "source_volume_scale": 1.0,
        "latest_closed_timestamp": 1_800_000_000_000,
        "latest_closed_age_min": 10.0,
        "cwidth": cwidth,
        "zone_scale": 1.0,
        "accepted": accepted,
    }


def test_demand_zone_allows_long_within_five_percent_without_touch():
    from event_engine.sr_context import evaluate_sr_directional_zone

    out = evaluate_sr_directional_zone(
        _snapshot((95.0, "L")), current_price=100.0, direction="LONG"
    )
    assert out["reject"] is False
    assert out["zone_type"] == "DEMAND"
    assert out["allowed_direction"] == "LONG"
    assert out["distance_pct"] == pytest.approx(4.5)
    assert out["alignment"] == "LONG_IN_DEMAND"


def test_demand_zone_hard_blocks_short_even_when_price_is_within_five_percent():
    from event_engine.sr_context import evaluate_sr_directional_zone

    out = evaluate_sr_directional_zone(
        _snapshot((95.0, "L")), current_price=100.0, direction="SHORT"
    )
    assert out["reject"] is True
    assert out["reject_reason"] == "DIRECTION_CONTRADICTS_CURRENT_ZONE"
    assert out["allowed_direction"] == "LONG"


def test_supply_zone_allows_short_and_hard_blocks_long():
    from event_engine.sr_context import evaluate_sr_directional_zone

    supply = _snapshot((105.0, "H"))
    short = evaluate_sr_directional_zone(supply, current_price=100.0, direction="SHORT")
    long = evaluate_sr_directional_zone(supply, current_price=100.0, direction="LONG")
    assert short["reject"] is False
    assert short["zone_type"] == "SUPPLY"
    assert short["allowed_direction"] == "SHORT"
    assert short["distance_pct"] == pytest.approx(4.5)
    assert long["reject"] is True
    assert long["reject_reason"] == "DIRECTION_CONTRADICTS_CURRENT_ZONE"


def test_exact_five_percent_distance_is_allowed_but_more_than_five_is_rejected():
    from event_engine.sr_context import evaluate_sr_directional_zone

    boundary = evaluate_sr_directional_zone(
        _snapshot((94.5, "L")), current_price=100.0, direction="LONG"
    )
    too_far = evaluate_sr_directional_zone(
        _snapshot((94.49, "L")), current_price=100.0, direction="LONG"
    )
    assert boundary["reject"] is False
    assert boundary["distance_pct"] == pytest.approx(5.0)
    assert too_far["reject"] is True
    assert too_far["reject_reason"] == "NO_DIRECTIONAL_ZONE_WITHIN_DISTANCE"
    assert too_far["distance_pct"] > 5.0


def test_price_inside_demand_or_supply_zone_has_zero_distance():
    from event_engine.sr_context import evaluate_sr_directional_zone

    demand = evaluate_sr_directional_zone(_snapshot((100.0, "L")), current_price=100.0, direction="LONG")
    supply = evaluate_sr_directional_zone(_snapshot((100.0, "H")), current_price=100.0, direction="SHORT")
    assert demand["reject"] is False and demand["distance_pct"] == 0.0
    assert supply["reject"] is False and supply["distance_pct"] == 0.0


def test_broken_demand_or_supply_zone_is_not_used_as_current_zone():
    from event_engine.sr_context import evaluate_sr_directional_zone

    broken_demand = evaluate_sr_directional_zone(
        _snapshot((95.0, "L")), current_price=93.0, direction="LONG"
    )
    broken_supply = evaluate_sr_directional_zone(
        _snapshot((105.0, "H")), current_price=107.0, direction="SHORT"
    )
    assert broken_demand["reject"] is True
    assert broken_demand["reject_reason"] == "NO_VALID_DEMAND_OR_SUPPLY_ZONE"
    assert broken_supply["reject"] is True
    assert broken_supply["reject_reason"] == "NO_VALID_DEMAND_OR_SUPPLY_ZONE"


def test_nearest_zone_selects_direction_and_mixed_type_tie_fails_closed():
    from event_engine.sr_context import evaluate_sr_directional_zone

    closer_supply = evaluate_sr_directional_zone(
        _snapshot((95.0, "L"), (104.5, "H")), current_price=100.0, direction="SHORT"
    )
    tied = evaluate_sr_directional_zone(
        _snapshot((94.5, "L"), (105.5, "H")), current_price=100.0, direction="LONG"
    )
    assert closer_supply["reject"] is False
    assert closer_supply["zone_type"] == "SUPPLY"
    assert tied["reject"] is True
    assert tied["reject_reason"] == "AMBIGUOUS_NEAREST_DEMAND_SUPPLY_ZONE"


def test_directional_gate_rejects_missing_or_non_3h_snapshot():
    from event_engine.sr_context import evaluate_sr_directional_zone

    no_timeframe = evaluate_sr_directional_zone(
        _snapshot((95.0, "L"), timeframe="1h"), current_price=100.0, direction="LONG"
    )
    assert no_timeframe["reject"] is True
    assert no_timeframe["reject_reason"] == "SR_TIMEFRAME_MISMATCH"


def test_spot_snapshot_aggregates_three_closed_1h_candles_into_3h(monkeypatch):
    import event_engine.sr_context as sr

    monkeypatch.setattr(sr, "SR_PRD", 0)
    monkeypatch.setattr(sr, "SR_RB", 0)
    now_ms = 26_000_000
    monkeypatch.setattr(sr.time, "time", lambda: now_ms / 1000.0)
    resolved = {
        "requested_symbol": "TESTUSDT", "source_symbol": "TESTUSDT", "underlying_asset": "TEST",
        "source_price_scale": 1.0, "source_volume_scale": 1.0,
        "source_contract_multiplier": 1.0, "source_alias_kind": "EXACT",
    }
    response = type("Response", (), {
        "status_code": 200,
        "raise_for_status": lambda self: None,
        "json": lambda self: [
            [0, "10", "11", "9", "10.5", "100", 3_599_999],
            [3_600_000, "10.5", "12", "10", "11", "120", 7_199_999],
            [7_200_000, "11", "12", "10", "11.5", "90", 10_799_999],
            [10_800_000, "11.5", "13", "11", "12", "70", 14_399_999],
            [14_400_000, "12", "14", "11.5", "13", "80", 17_999_999],
            [18_000_000, "13", "15", "12", "14", "95", 21_599_999],
            # A seventh closed 1h candle is a partial 3h bucket and must not be emitted.
            [21_600_000, "14", "16", "13", "15", "50", 25_199_999],
            # This current 1h candle is still open and is excluded before aggregation.
            [25_200_000, "15", "17", "14", "16", "40", 28_799_999],
        ],
    })()
    calls = []
    class Session:
        def get(self, url, **kwargs):
            calls.append((url, kwargs))
            return response
    monkeypatch.setattr(sr, "resolve_spot_symbol", lambda symbol, **kwargs: dict(resolved))
    monkeypatch.setattr(sr, "_SESSION", Session())
    monkeypatch.setattr(sr, "_rate_limit_before_request", lambda: None)

    candles, meta = sr._fetch_closed_3h_spot("TEST")
    assert len(candles) == 2
    assert calls[0][1]["params"]["interval"] == "1h"  # Binance Spot has no native 3h interval.
    assert calls[0][1]["params"]["symbol"] == "TESTUSDT"
    assert candles[0].ts == 0
    assert candles[0].open == pytest.approx(10.0)
    assert candles[0].high == pytest.approx(12.0)
    assert candles[0].low == pytest.approx(9.0)
    assert candles[0].close == pytest.approx(11.5)
    assert candles[0].volume == pytest.approx(310.0)
    assert candles[1].ts == 10_800_000
    assert candles[1].open == pytest.approx(11.5)
    assert candles[1].high == pytest.approx(15.0)
    assert candles[1].low == pytest.approx(11.0)
    assert candles[1].close == pytest.approx(14.0)
    assert candles[1].volume == pytest.approx(245.0)
    assert meta["source_symbol"] == "TESTUSDT"


def test_3h_aggregation_drops_incomplete_or_misaligned_hour_groups():
    import event_engine.sr_context as sr

    hourly = [
        sr.Candle(0, 10, 11, 9, 10, 1),
        sr.Candle(3_600_000, 10, 12, 9, 11, 2),
        # Missing the third hour in first bucket, so no 3h bar is emitted.
        sr.Candle(10_800_000, 11, 12, 10, 11.5, 3),
        sr.Candle(14_400_000, 11.5, 13, 11, 12, 4),
        sr.Candle(18_000_000, 12, 14, 11.5, 13.5, 5),
    ]
    candles = sr._aggregate_closed_3h_candles(hourly)
    assert len(candles) == 1
    assert candles[0].ts == 10_800_000
    assert candles[0].open == pytest.approx(11.0)
    assert candles[0].high == pytest.approx(14.0)
    assert candles[0].low == pytest.approx(10.0)
    assert candles[0].close == pytest.approx(13.5)
    assert candles[0].volume == pytest.approx(12.0)


def test_get_spot_current_price_uses_spot_ticker_and_alias_price_scale(monkeypatch):
    import event_engine.sr_context as sr

    captured = {}
    class Response:
        def raise_for_status(self): return None
        def json(self): return {"symbol": "SHIBUSDT", "price": "0.000005"}
    class Session:
        def get(self, url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return Response()
    monkeypatch.setattr(sr, "_SESSION", Session())
    monkeypatch.setattr(sr, "_rate_limit_before_request", lambda: None)
    out = sr.get_spot_current_price("1000SHIB", snapshot={
        "source_symbol": "SHIBUSDT", "source_price_scale": 1000.0,
    })
    assert out == pytest.approx(0.005)
    assert captured["params"]["symbol"] == "SHIBUSDT"
    assert captured["url"].endswith("/api/v3/ticker/price")


def test_execute_new_position_hard_blocks_direction_conflict_before_bingx_order(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "SR_3H_DIRECTIONAL_GATE_ENABLED", True)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", False)
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: _snapshot((95.0, "L")))
    monkeypatch.setattr(ro, "get_spot_current_price", lambda symbol, snapshot=None: 100.0)
    monkeypatch.setattr(ro, "record_action", lambda *args, **kwargs: None)
    monkeypatch.setattr(ro, "open_market", lambda *args, **kwargs: pytest.fail("opposing entry must not reach BingX"))
    setup = {"signal_price": 100.0, "event_type": "DONCHIAN_RETEST_BREAKOUT"}

    out = ro.execute_new_position("TEST", "SHORT", 100.0, setup, "EVT_3H_WRONG_SIDE")
    assert out["status"] == "SR_3H_DIRECTIONAL_REJECTED"
    assert out["sr_3h"]["allowed_direction"] == "LONG"


def test_execute_new_position_fails_closed_when_3h_spot_data_unavailable(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "SR_3H_DIRECTIONAL_GATE_ENABLED", True)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", False)
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: (_ for _ in ()).throw(RuntimeError("spot down")))
    monkeypatch.setattr(ro, "record_action", lambda *args, **kwargs: None)
    monkeypatch.setattr(ro, "open_market", lambda *args, **kwargs: pytest.fail("missing 3H context must fail closed"))
    out = ro.execute_new_position("TEST", "LONG", 100.0, {"signal_price": 100.0}, "EVT_3H_NO_DATA")
    assert out["status"] == "SR_DATA_UNAVAILABLE"


def test_execute_new_position_allows_matching_zone_to_reach_bingx_order(monkeypatch):
    import run_once as ro

    calls = {"open": 0}
    monkeypatch.setattr(ro, "SR_3H_DIRECTIONAL_GATE_ENABLED", True)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", False)
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: _snapshot((95.0, "L")))
    monkeypatch.setattr(ro, "get_spot_current_price", lambda symbol, snapshot=None: 100.0)
    monkeypatch.setattr(ro, "record_action", lambda *args, **kwargs: None)
    monkeypatch.setattr(ro, "open_market", lambda *args, **kwargs: calls.__setitem__("open", calls["open"] + 1) or {
        "status": "error", "error": "STOP_AFTER_PROOF_GATE_REACHED",
    })
    out = ro.execute_new_position("TEST", "LONG", 100.0, {"signal_price": 100.0}, "EVT_3H_RIGHT_SIDE")
    assert calls["open"] == 1
    assert out["status"] == "OPEN_FAILED"


def test_3h_fetch_fails_closed_when_incomplete_bucket_creates_history_gap(monkeypatch):
    import event_engine.sr_context as sr

    monkeypatch.setattr(sr, "SR_PRD", 0)
    monkeypatch.setattr(sr, "SR_RB", 0)
    resolved = {
        "requested_symbol": "TESTUSDT", "source_symbol": "TESTUSDT", "underlying_asset": "TEST",
        "source_price_scale": 1.0, "source_volume_scale": 1.0,
        "source_contract_multiplier": 1.0, "source_alias_kind": "EXACT",
    }
    # Complete bucket at 00:00, incomplete bucket at 03:00, then complete bucket at 06:00.
    hourly = [
        sr.Candle(0, 10, 11, 9, 10, 1),
        sr.Candle(3_600_000, 10, 11, 9, 10, 1),
        sr.Candle(7_200_000, 10, 11, 9, 10, 1),
        sr.Candle(10_800_000, 10, 11, 9, 10, 1),
        # Missing 04:00 candle means the 03:00 3h bucket is dropped.
        sr.Candle(18_000_000, 10, 11, 9, 10, 1),
        sr.Candle(21_600_000, 10, 11, 9, 10, 1),
        sr.Candle(25_200_000, 10, 11, 9, 10, 1),
        sr.Candle(28_800_000, 10, 11, 9, 10, 1),
    ]
    monkeypatch.setattr(sr, "_fetch_closed_spot", lambda *args, **kwargs: (hourly, resolved))
    with pytest.raises(ValueError, match="Incomplete 3H Spot history"):
        sr._fetch_closed_3h_spot("TEST")
