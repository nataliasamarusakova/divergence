from __future__ import annotations

import pytest


def _snapshot(*zones, cwidth=0.5, timeframe="3h"):
    accepted = [
        {"price": float(center), "kind": kind, "points": 3, "countpp": i + 1}
        for i, (center, kind) in enumerate(zones)
    ]
    return {
        "source": "binance_usds_m_futures",
        "timeframe": timeframe,
        "interval_ms": 10_800_000,
        "source_symbol": "TESTUSDT",
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


def test_futures_snapshot_aggregates_three_closed_1h_candles_into_3h(monkeypatch):
    import event_engine.sr_context as sr

    monkeypatch.setattr(sr, "SR_MIN_CANDLES", 2)
    monkeypatch.setattr(sr, "SR_ATR_LENGTH", 1)
    monkeypatch.setattr(sr, "SR_SWING_LENGTH", 1)
    resolved = {
        "requested_symbol": "TESTUSDT", "source_symbol": "TESTUSDT", "underlying_asset": "TEST",
        "source_alias_kind": "EXACT_FUTURES", "contract_type": "PERPETUAL", "quote_asset": "USDT",
    }
    hourly = [
        sr.Candle(0, 10, 11, 9, 10.5, 100),
        sr.Candle(3_600_000, 10.5, 12, 10, 11, 120),
        sr.Candle(7_200_000, 11, 12, 10, 11.5, 90),
        sr.Candle(10_800_000, 11.5, 13, 11, 12, 70),
        sr.Candle(14_400_000, 12, 14, 11.5, 13, 80),
        sr.Candle(18_000_000, 13, 15, 12, 14, 95),
        # A seventh closed 1h candle is a partial 3h bucket and is not emitted.
        sr.Candle(21_600_000, 14, 16, 13, 15, 50),
    ]
    calls = []
    monkeypatch.setattr(
        sr, "_fetch_closed_1h_futures",
        lambda symbol, **kwargs: calls.append((symbol, kwargs)) or (hourly, resolved),
    )

    candles, meta = sr._fetch_closed_3h_futures("TEST")
    assert len(candles) == 2
    assert calls == [("TEST", {"resolved": None, "limit": None})]
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


def test_get_futures_current_price_uses_same_perpetual_symbol_without_scaling(monkeypatch):
    import event_engine.sr_context as sr

    captured = {}
    monkeypatch.setattr(sr, "fetch_futures_price", lambda symbol: captured.setdefault("symbol", symbol) or 2.9459)
    # setdefault returns the symbol, so use an explicit deterministic stub.
    monkeypatch.setattr(sr, "fetch_futures_price", lambda symbol: captured.update({"symbol": symbol}) or 2.9459)
    out = sr.get_futures_current_price("PIEVERSE", snapshot={"source_symbol": "PIEVERSEUSDT"})
    assert out == pytest.approx(2.9459)
    assert captured["symbol"] == "PIEVERSEUSDT"


def test_execute_new_position_hard_blocks_direction_conflict_before_bingx_order(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "SR_3H_DIRECTIONAL_GATE_ENABLED", True)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", False)
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: _snapshot((95.0, "L")))
    monkeypatch.setattr(ro, "get_futures_current_price", lambda symbol, snapshot=None: 100.0)
    monkeypatch.setattr(ro, "record_action", lambda *args, **kwargs: None)
    monkeypatch.setattr(ro, "open_market", lambda *args, **kwargs: pytest.fail("opposing entry must not reach BingX"))
    setup = {"signal_price": 100.0, "event_type": "DONCHIAN_RETEST_BREAKOUT"}

    out = ro.execute_new_position("TEST", "SHORT", 100.0, setup, "EVT_3H_WRONG_SIDE")
    assert out["status"] == "SR_3H_DIRECTIONAL_REJECTED"
    assert out["sr_3h"]["allowed_direction"] == "LONG"


def test_execute_new_position_fails_closed_when_3h_futures_data_unavailable(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "SR_3H_DIRECTIONAL_GATE_ENABLED", True)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", False)
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: (_ for _ in ()).throw(RuntimeError("Futures data down")))
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
    monkeypatch.setattr(ro, "get_futures_current_price", lambda symbol, snapshot=None: 100.0)
    monkeypatch.setattr(ro, "record_action", lambda *args, **kwargs: None)
    monkeypatch.setattr(ro, "open_market", lambda *args, **kwargs: calls.__setitem__("open", calls["open"] + 1) or {
        "status": "error", "error": "STOP_AFTER_PROOF_GATE_REACHED",
    })
    out = ro.execute_new_position("TEST", "LONG", 100.0, {"signal_price": 100.0}, "EVT_3H_RIGHT_SIDE")
    assert calls["open"] == 1
    assert out["status"] == "OPEN_FAILED"


def test_3h_fetch_fails_closed_when_incomplete_bucket_creates_history_gap(monkeypatch):
    import event_engine.sr_context as sr

    monkeypatch.setattr(sr, "SR_MIN_CANDLES", 2)
    monkeypatch.setattr(sr, "SR_ATR_LENGTH", 1)
    monkeypatch.setattr(sr, "SR_SWING_LENGTH", 1)
    resolved = {
        "requested_symbol": "TESTUSDT", "source_symbol": "TESTUSDT", "underlying_asset": "TEST",
        "source_alias_kind": "EXACT_FUTURES", "contract_type": "PERPETUAL", "quote_asset": "USDT",
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
    monkeypatch.setattr(sr, "_fetch_closed_1h_futures", lambda *args, **kwargs: (hourly, resolved))
    with pytest.raises(ValueError, match="Incomplete Futures 3H history"):
        sr._fetch_closed_3h_futures("TEST")
