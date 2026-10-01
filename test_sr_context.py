from __future__ import annotations

import pytest



def _snapshot(*zones):
    accepted = []
    for center, kind in zones:
        accepted.append({"price": center, "kind": kind, "points": 2, "countpp": len(accepted) + 1})
    return {
        "source": "binance_spot",
        "source_symbol": "TESTUSDT",
        "latest_closed_timestamp": 1_000_000,
        "latest_closed_age_min": 10.0,
        "cwidth": 0.5,
        "zone_scale": 1.0,
        "accepted": accepted,
    }


def test_long_zone_between_tp2_and_tp3_is_allowed_and_caps_tp3(monkeypatch):
    import event_engine.sr_context as sr
    monkeypatch.setattr(sr, "SR_TARGET_BUFFER_PCT", 0.05)
    monkeypatch.setattr(sr, "SR_TARGET_BUFFER_R", 0.10)
    monkeypatch.setattr(sr, "SR_MIN_PARTIAL_TP3_R", 0.25)
    out = sr.evaluate_sr_room(
        _snapshot((104.5, "H")),
        entry_price=100.0,
        direction="LONG",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is False
    assert out["room_status"] == "PARTIAL_ROOM"
    assert out["tp3_capped"] is True
    assert out["effective_tp3_price"] < 104.0
    assert out["effective_tp3_price"] > 103.0


def test_long_opposing_zone_before_tp2_is_rejected():
    import event_engine.sr_context as sr
    out = sr.evaluate_sr_room(
        _snapshot((103.0, "H")),
        entry_price=100.0,
        direction="LONG",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is True
    assert out["room_status"] == "INSUFFICIENT_ROOM"
    assert out["reject_reason"] == "OPPOSING_ZONE_BEFORE_TP2"


def test_long_entry_inside_support_is_not_opposing_and_is_recorded():
    import event_engine.sr_context as sr
    out = sr.evaluate_sr_room(
        _snapshot((99.8, "L")),
        entry_price=100.0,
        direction="LONG",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is False
    assert out["supporting_zone_context"] == "SUPPORTIVE_INSIDE"


def test_nearby_long_resistance_inside_execution_buffer_is_still_opposing():
    import event_engine.sr_context as sr
    out = sr.evaluate_sr_room(
        _snapshot((100.60, "H")),
        entry_price=100.0,
        direction="LONG",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["nearest_opposing_zone"] is not None
    assert out["opposing_zone_distance_r"] >= 0.0
    assert out["reject"] is True


def test_nearby_short_support_inside_execution_buffer_is_still_opposing():
    import event_engine.sr_context as sr
    out = sr.evaluate_sr_room(
        _snapshot((99.40, "L")),
        entry_price=100.0,
        direction="SHORT",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["nearest_opposing_zone"] is not None
    assert out["opposing_zone_distance_r"] >= 0.0
    assert out["reject"] is True


def test_short_wide_support_zone_uses_pivot_kind_for_overlap():
    import event_engine.sr_context as sr
    out = sr.evaluate_sr_room(
        _snapshot((100.2, "L")),
        entry_price=100.0,
        direction="SHORT",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is True
    assert out["room_status"] == "ENTRY_IN_OPPOSING_ZONE"


def test_long_entry_inside_opposing_zone_is_rejected():
    import event_engine.sr_context as sr
    out = sr.evaluate_sr_room(
        _snapshot((100.2, "H")),
        entry_price=100.0,
        direction="LONG",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is True
    assert out["room_status"] == "ENTRY_IN_OPPOSING_ZONE"


def test_short_rule_is_mirror_of_long():
    import event_engine.sr_context as sr
    out = sr.evaluate_sr_room(
        _snapshot((95.75, "L")),
        entry_price=100.0,
        direction="SHORT",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is False
    assert out["room_status"] == "PARTIAL_ROOM"
    assert out["tp3_capped"] is True
    assert out["effective_tp3_price"] > 96.0
    assert out["effective_tp3_price"] < 97.0


def test_opposing_zone_behind_entry_does_not_block():
    import event_engine.sr_context as sr
    out = sr.evaluate_sr_room(
        _snapshot((94.0, "L")),
        entry_price=100.0,
        direction="LONG",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is False
    assert out["nearest_opposing_zone"] is None


def test_apply_sr_tp3_cap_recomputes_rr():
    import event_engine.sr_context as sr
    setup = {"risk_pct": 2.0, "target_rr": 2.5, "planned_weighted_rr": 1.6625}
    tp_levels = [
        {"leg": "tp1", "pnl_pct": 1.5, "close_fraction": 0.25},
        {"leg": "tp2", "pnl_pct": 3.0, "close_fraction": 0.40},
        {"leg": "tp3", "pnl_pct": 5.0, "close_fraction": 0.35},
    ]
    sr_result = {"tp3_capped": True, "effective_tp3_price": 103.8}
    adjusted, rr = sr.apply_sr_tp3_cap(
        setup,
        direction="LONG",
        tp_levels=tp_levels,
        sr_result=sr_result,
        actual_entry_price=100.0,
    )
    assert adjusted[2]["pnl_pct"] == 3.8
    assert abs(rr - 1.9) < 1e-9
    assert setup["target_rr"] == 1.9
    assert setup["sr_tp3_capped"] is True


def test_execute_new_position_blocks_before_order_when_sr_room_fails(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "AJAY_SR_ROOM_ENABLED", True)
    monkeypatch.setattr(ro, "AJAY_SR_ROOM_MODE", "enforce")
    monkeypatch.setattr(ro, "AJAY_SR_REQUIRE_DATA", True)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "fetch_binance_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: _snapshot((102.8, "H")))
    called = {"open": False}
    monkeypatch.setattr(ro, "open_market", lambda *args, **kwargs: called.__setitem__("open", True) or {"status": "opened"})
    monkeypatch.setattr(ro, "record_action", lambda *args, **kwargs: None)

    setup = {
        "entry_reference": 100.0,
        "invalidation_price": 98.0,
        "risk_pct": 2.0,
        "signal_price": 100.0,
        "event_type": "DONCHIAN_RETEST_BREAKOUT",
        "trigger": {"trigger_price": 100.0},
    }
    out = ro.execute_new_position("TEST", "LONG", 100.0, setup, "EVT_SR_REJECT")
    assert out["status"] == "SR_ROOM_REJECTED"
    assert called["open"] is False


def test_lazy_sr_fetch_is_cached_per_latest_closed_bar(monkeypatch):
    import event_engine.sr_context as sr
    sr.clear_sr_cache()
    calls = {"n": 0}
    candles = [sr.Candle(i * 3_600_000, 1, 2, 0.5, 1.5, 10) for i in range(300)]
    fake_now = (candles[-1].ts + 3_599_999 + 60_000) / 1000.0
    monkeypatch.setattr(sr.time, "time", lambda: fake_now)
    monkeypatch.setattr(sr, "_fetch_closed_1h_spot", lambda symbol: calls.__setitem__("n", calls["n"] + 1) or candles)
    monkeypatch.setattr(sr, "compute_current_sr", lambda *args, **kwargs: {
        "levels": [1.0], "sr_levels": [None, 1.0], "highestph": 2.0, "lowestpl": 0.5,
        "cwidth": 0.1, "event_idx": 250, "event_timestamp": candles[250].ts, "accepted": [],
    })
    first = sr.get_cached_sr_snapshot("TEST-USDT")
    second = sr.get_cached_sr_snapshot("TEST-USDT")
    assert first["source"] == "binance_spot"
    assert second["source"] == "binance_spot"
    assert calls["n"] == 1


def test_long_wide_resistance_zone_uses_pivot_kind_for_overlap():
    import event_engine.sr_context as sr
    # The H center is below entry, but the wide H zone still overlaps entry.
    # It must remain opposing rather than being misclassified as support.
    out = sr.evaluate_sr_room(
        _snapshot((99.8, "H")),
        entry_price=100.0,
        direction="LONG",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is True
    assert out["room_status"] == "ENTRY_IN_OPPOSING_ZONE"


def test_shadow_sr_does_not_mutate_tp3(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "AJAY_SR_ROOM_ENABLED", True)
    monkeypatch.setattr(ro, "AJAY_SR_ROOM_MODE", "shadow")
    monkeypatch.setattr(ro, "AJAY_SR_REQUIRE_DATA", True)
    monkeypatch.setattr(ro, "MARKET_DATA_SOURCE", "binance")
    monkeypatch.setattr(ro, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", True)
    monkeypatch.setattr(ro, "MAX_CROSS_EXCHANGE_DRIFT_PCT", 1.0)
    monkeypatch.setattr(ro, "MAX_ENTRY_DRIFT_PCT", 3.0)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "fetch_binance_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: _snapshot((104.5, "H")))
    monkeypatch.setattr(ro, "record_action", lambda *args, **kwargs: None)
    monkeypatch.setattr(ro, "open_market", lambda *args, **kwargs: {
        "status": "opened", "order_id": "O1", "order_reference_price": 100.0, "leverage": 10,
    })
    monkeypatch.setattr(ro, "wait_for_position_fill_directional", lambda **kwargs: {
        "status": "found", "positionAmt": "1", "avgPrice": "100", "entryPrice": "100",
    })
    captured = {}
    monkeypatch.setattr(ro, "install_protection", lambda **kwargs: (
        captured.update(kwargs) or {
            "status": "PROTECTED",
            "tp_mode": "multi_tp",
            "effective_tp_levels": kwargs["tp_levels"],
            "effective_weighted_rr": kwargs["setup"]["planned_weighted_rr"],
        }
    ))
    monkeypatch.setattr(ro, "emergency_close_position", lambda *args, **kwargs: pytest.fail("shadow SR must not force a rollback"))

    out = ro.execute_new_position(
        "TEST", "LONG", 100.0,
        {"risk_pct": 2.0, "signal_price": 100.0, "event_type": "DONCHIAN_RETEST_BREAKOUT"},
        "EVT_SR_SHADOW",
    )
    assert out["status"] == "opened_protected"
    assert captured["tp_levels"][2]["pnl_pct"] == pytest.approx(5.0)


def test_enforced_sr_data_failure_blocks_candidate_without_sending_order(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "AJAY_SR_ROOM_ENABLED", True)
    monkeypatch.setattr(ro, "AJAY_SR_ROOM_MODE", "enforce")
    monkeypatch.setattr(ro, "AJAY_SR_REQUIRE_DATA", True)
    monkeypatch.setattr(ro, "MARKET_DATA_SOURCE", "binance")
    monkeypatch.setattr(ro, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", True)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "fetch_binance_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: (_ for _ in ()).throw(RuntimeError("spot unavailable")))
    monkeypatch.setattr(ro, "record_action", lambda *args, **kwargs: None)
    monkeypatch.setattr(ro, "open_market", lambda *args, **kwargs: pytest.fail("order must not be sent when required SR data is unavailable"))

    out = ro.execute_new_position(
        "TEST", "LONG", 100.0,
        {"risk_pct": 2.0, "signal_price": 100.0, "event_type": "DONCHIAN_RETEST_BREAKOUT"},
        "EVT_SR_DATA_FAIL",
    )
    assert out["status"] == "SR_DATA_UNAVAILABLE"


def test_post_fill_sr_recheck_can_flatten_when_fill_destroys_tp2_room(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "AJAY_SR_ROOM_ENABLED", True)
    monkeypatch.setattr(ro, "AJAY_SR_ROOM_MODE", "enforce")
    monkeypatch.setattr(ro, "AJAY_SR_REQUIRE_DATA", True)
    monkeypatch.setattr(ro, "MARKET_DATA_SOURCE", "binance")
    monkeypatch.setattr(ro, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", True)
    monkeypatch.setattr(ro, "MAX_CROSS_EXCHANGE_DRIFT_PCT", 1.0)
    monkeypatch.setattr(ro, "MAX_ENTRY_DRIFT_PCT", 2.0)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "fetch_binance_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: _snapshot((104.5, "H")))
    monkeypatch.setattr(ro, "record_action", lambda *args, **kwargs: None)
    monkeypatch.setattr(ro, "open_market", lambda *args, **kwargs: {
        "status": "opened", "order_id": "O1", "order_reference_price": 100.0, "leverage": 10,
    })
    monkeypatch.setattr(ro, "wait_for_position_fill_directional", lambda **kwargs: {
        "status": "found", "positionAmt": "1", "avgPrice": "102", "entryPrice": "102",
    })
    monkeypatch.setattr(ro, "install_protection", lambda **kwargs: pytest.fail("protection must not be installed after post-fill SR rejection"))
    monkeypatch.setattr(ro, "emergency_close_position", lambda *args, **kwargs: {"status": "closed"})

    out = ro.execute_new_position(
        "TEST", "LONG", 100.0,
        {"risk_pct": 2.0, "signal_price": 100.0, "event_type": "DONCHIAN_RETEST_BREAKOUT"},
        "EVT_SR_POST_FILL",
    )
    assert out["status"] == "SR_ROOM_POST_FILL_REJECTED"
    assert out["rolled_back"] is True
    assert out["sr_room"]["room_status"] == "INSUFFICIENT_ROOM"


def test_post_fill_partial_room_recomputes_tp3_from_actual_fill(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "AJAY_SR_ROOM_ENABLED", True)
    monkeypatch.setattr(ro, "AJAY_SR_ROOM_MODE", "enforce")
    monkeypatch.setattr(ro, "AJAY_SR_REQUIRE_DATA", True)
    monkeypatch.setattr(ro, "MARKET_DATA_SOURCE", "binance")
    monkeypatch.setattr(ro, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", True)
    monkeypatch.setattr(ro, "MAX_CROSS_EXCHANGE_DRIFT_PCT", 1.0)
    monkeypatch.setattr(ro, "MAX_ENTRY_DRIFT_PCT", 3.0)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "fetch_binance_price", lambda symbol: 100.0)
    # H zone is [104.8, 105.8]. Pre-order TP3 is 105.0, so the zone forces a cap.
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: _snapshot((105.3, "H")))
    monkeypatch.setattr(ro, "record_action", lambda *args, **kwargs: None)
    monkeypatch.setattr(ro, "open_market", lambda *args, **kwargs: {
        "status": "opened", "order_id": "O1", "order_reference_price": 100.0, "leverage": 10,
    })
    # Worse but still acceptable fill: 100.5. The same zone now has a materially
    # different R-distance, so TP3 must be recomputed from the actual fill.
    monkeypatch.setattr(ro, "wait_for_position_fill_directional", lambda **kwargs: {
        "status": "found", "positionAmt": "1", "avgPrice": "100.5", "entryPrice": "100.5",
    })
    captured = {}
    monkeypatch.setattr(ro, "install_protection", lambda **kwargs: (
        captured.update(kwargs) or {
            "status": "PROTECTED",
            "tp_mode": "multi_tp",
            "effective_tp_levels": kwargs["tp_levels"],
            "effective_weighted_rr": kwargs["setup"]["planned_weighted_rr"],
        }
    ))
    monkeypatch.setattr(ro, "emergency_close_position", lambda *args, **kwargs: pytest.fail("partial room with valid fill must not rollback"))

    out = ro.execute_new_position(
        "TEST", "LONG", 100.0,
        {"risk_pct": 2.0, "signal_price": 100.0, "event_type": "DONCHIAN_RETEST_BREAKOUT"},
        "EVT_SR_POST_FILL_CAP",
    )
    assert out["status"] == "opened_protected"
    expected_cap = 104.8 - max(100.5 * 0.0005, (100.5 * 0.02) * 0.10)
    expected_pnl_pct = round((expected_cap - 100.5) / 100.5 * 100.0, 6)
    assert captured["tp_levels"][2]["pnl_pct"] == pytest.approx(expected_pnl_pct, abs=1e-9)
    assert captured["tp_levels"][2]["pnl_pct"] < 4.8
