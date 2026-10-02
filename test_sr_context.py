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


def test_long_opposing_zone_after_tp1_is_allowed_without_tp3_mutation():
    import event_engine.sr_context as sr
    out = sr.evaluate_sr_room(
        _snapshot((104.5, "H")),
        entry_price=100.0,
        direction="LONG",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is False
    assert out["room_status"] == "POST_TP1_OPPOSING_ZONE"
    assert out["tp3_capped"] is False
    assert out["effective_tp3_price"] == pytest.approx(105.0)


def test_long_opposing_zone_before_tp1_is_rejected():
    import event_engine.sr_context as sr
    out = sr.evaluate_sr_room(
        _snapshot((100.8, "H")),
        entry_price=100.0,
        direction="LONG",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is True
    assert out["room_status"] == "OPPOSING_ZONE_BEFORE_TP1"
    assert out["reject_reason"] == "OPPOSING_ZONE_BEFORE_TP1"


def test_long_zone_just_after_tp1_is_allowed_without_buffer_veto():
    import event_engine.sr_context as sr
    # TP1 = 101.5. Zone [101.51, 102.51] does not reach TP1 and must be allowed.
    out = sr.evaluate_sr_room(
        _snapshot((102.01, "H")),
        entry_price=100.0,
        direction="LONG",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is False
    assert out["room_status"] == "POST_TP1_OPPOSING_ZONE"


def test_long_opposing_zone_between_tp1_and_tp2_is_allowed():
    import event_engine.sr_context as sr
    out = sr.evaluate_sr_room(
        _snapshot((102.5, "H")),
        entry_price=100.0,
        direction="LONG",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is False
    assert out["room_status"] == "POST_TP1_OPPOSING_ZONE"


def test_long_entry_inside_support_has_directional_confirmation():
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
    assert out["supporting_zone_confirmation"] is True
    assert out["directional_zone_alignment"] == "LONG_IN_DEMAND"


def test_short_entry_inside_supply_has_directional_confirmation():
    import event_engine.sr_context as sr
    out = sr.evaluate_sr_room(
        _snapshot((100.2, "H")),
        entry_price=100.0,
        direction="SHORT",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is False
    assert out["supporting_zone_context"] == "SUPPORTIVE_INSIDE"
    assert out["supporting_zone_confirmation"] is True
    assert out["directional_zone_alignment"] == "SHORT_IN_SUPPLY"


def test_nearby_long_resistance_is_opposing_and_blocks_tp1():
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
    assert out["reject_reason"] == "OPPOSING_ZONE_BEFORE_TP1"


def test_nearby_short_support_is_opposing_and_blocks_tp1():
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
    assert out["reject_reason"] == "OPPOSING_ZONE_BEFORE_TP1"


def test_short_opposing_zone_after_tp1_is_allowed_without_tp3_mutation():
    import event_engine.sr_context as sr
    out = sr.evaluate_sr_room(
        _snapshot((95.75, "L")),
        entry_price=100.0,
        direction="SHORT",
        risk_pct=2.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is False
    assert out["room_status"] == "POST_TP1_OPPOSING_ZONE"
    assert out["tp3_capped"] is False
    assert out["effective_tp3_price"] == pytest.approx(95.0)


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


def test_current_sr_policy_has_no_tp3_cap_helper():
    import event_engine.sr_context as sr
    assert not hasattr(sr, "apply_sr_tp3_cap")


def test_execute_new_position_blocks_before_order_when_sr_room_fails(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "AJAY_SR_ROOM_ENABLED", True)
    monkeypatch.setattr(ro, "AJAY_SR_ROOM_MODE", "enforce")
    monkeypatch.setattr(ro, "AJAY_SR_REQUIRE_DATA", True)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "fetch_binance_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: _snapshot((101.2, "H")))
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


def test_post_fill_sr_recheck_allows_zone_after_tp1(monkeypatch):
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
    # H zone starts above TP1 for a 2% test risk, so it must not block the fill.
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: _snapshot((104.0, "H")))
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
    monkeypatch.setattr(ro, "emergency_close_position", lambda *args, **kwargs: pytest.fail("zone after TP1 must not rollback"))

    out = ro.execute_new_position(
        "TEST", "LONG", 100.0,
        {"risk_pct": 2.0, "signal_price": 100.0, "event_type": "DONCHIAN_RETEST_BREAKOUT"},
        "EVT_SR_POST_FILL_ALLOWED",
    )
    assert out["status"] == "opened_protected"
    assert captured["tp_levels"][0]["pnl_pct"] == pytest.approx(1.5)
    assert captured["tp_levels"][2]["pnl_pct"] == pytest.approx(5.0)

def test_post_fill_sr_does_not_mutate_tp3_when_zone_is_after_tp1(monkeypatch):
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
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: _snapshot((104.0, "H")))
    monkeypatch.setattr(ro, "record_action", lambda *args, **kwargs: None)
    monkeypatch.setattr(ro, "open_market", lambda *args, **kwargs: {
        "status": "opened", "order_id": "O1", "order_reference_price": 100.0, "leverage": 10,
    })
    monkeypatch.setattr(ro, "wait_for_position_fill_directional", lambda **kwargs: {
        "status": "found", "positionAmt": "1", "avgPrice": "101", "entryPrice": "101",
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

    out = ro.execute_new_position(
        "TEST", "LONG", 100.0,
        {"risk_pct": 2.0, "signal_price": 100.0, "event_type": "DONCHIAN_RETEST_BREAKOUT"},
        "EVT_SR_POST_FILL_NO_MUTATION",
    )
    assert out["status"] == "opened_protected"
    assert captured["tp_levels"][0]["pnl_pct"] == pytest.approx(1.5)
    assert captured["tp_levels"][1]["pnl_pct"] == pytest.approx(3.0)
    assert captured["tp_levels"][2]["pnl_pct"] == pytest.approx(5.0)

