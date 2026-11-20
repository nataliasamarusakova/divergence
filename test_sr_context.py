from __future__ import annotations

import pytest



@pytest.fixture(autouse=True)
def _disable_network_symbol_resolution(monkeypatch):
    """Execution-path unit tests use normalized test symbols and must stay offline."""
    import run_once as ro

    def normalize(symbol):
        text = str(symbol or "").strip().upper().replace("_", "-").replace("/", "-")
        if not text:
            return None
        if "-" not in text and text.endswith("USDT") and len(text) > 4:
            return f"{text[:-4]}-USDT"
        if "-" not in text:
            return text + "-USDT"
        return text

    monkeypatch.setattr(ro, "to_bx_symbol", normalize)


def _snapshot(*zones):
    accepted = []
    for center, kind in zones:
        accepted.append({"price": center, "kind": kind, "points": 2, "countpp": len(accepted) + 1})
    return {
        "source": "binance_usds_m_futures",
        "timeframe": "3h",
        "interval_ms": 10_800_000,
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
    candles = [sr.Candle(i * 10_800_000, 1, 2, 0.5, 1.5, 10) for i in range(300)]
    fake_now = (candles[-1].ts + 10_799_999 + 60_000) / 1000.0
    monkeypatch.setattr(sr.time, "time", lambda: fake_now)
    monkeypatch.setattr(sr, "_fetch_closed_3h_futures", lambda symbol, **kwargs: calls.__setitem__("n", calls["n"] + 1) or (candles, {
        "requested_symbol": "TESTUSDT", "source_symbol": "TESTUSDT", "underlying_asset": "TEST",
        "source_alias_kind": "EXACT_FUTURES", "contract_type": "PERPETUAL", "quote_asset": "USDT",
    }))
    monkeypatch.setattr(sr, "compute_current_sr", lambda *args, **kwargs: {
        "levels": [1.0], "sr_levels": [None, 1.0], "highestph": 2.0, "lowestpl": 0.5,
        "cwidth": 0.1, "event_idx": 250, "event_timestamp": candles[250].ts, "accepted": [],
    })
    first = sr.get_cached_sr_snapshot("TEST-USDT")
    second = sr.get_cached_sr_snapshot("TEST-USDT")
    assert first["source"] == "binance_usds_m_futures"
    assert second["source"] == "binance_usds_m_futures"
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


def test_shadow_sr_does_not_mutate_two_tp_ladder(monkeypatch):
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
    assert [x["leg"] for x in captured["tp_levels"]] == ["tp1", "tp2"]
    assert captured["tp_levels"][1]["pnl_pct"] == pytest.approx(2.9)


def test_enforced_sr_data_failure_blocks_candidate_without_sending_order(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "AJAY_SR_ROOM_ENABLED", True)
    monkeypatch.setattr(ro, "AJAY_SR_ROOM_MODE", "enforce")
    monkeypatch.setattr(ro, "AJAY_SR_REQUIRE_DATA", True)
    monkeypatch.setattr(ro, "MARKET_DATA_SOURCE", "binance")
    monkeypatch.setattr(ro, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", True)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "fetch_binance_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: (_ for _ in ()).throw(RuntimeError("Futures data unavailable")))
    monkeypatch.setattr(ro, "record_action", lambda *args, **kwargs: None)
    monkeypatch.setattr(ro, "open_market", lambda *args, **kwargs: pytest.fail("order must not be sent when required SR data is unavailable"))

    out = ro.execute_new_position(
        "TEST", "LONG", 100.0,
        {"risk_pct": 2.0, "signal_price": 100.0, "event_type": "DONCHIAN_RETEST_BREAKOUT"},
        "EVT_SR_DATA_FAIL",
    )
    assert out["status"] == "SR_DATA_UNAVAILABLE"


def test_sr_data_failure_does_not_block_entry_when_data_is_optional(monkeypatch):
    import run_once as ro

    monkeypatch.setattr(ro, "AJAY_SR_ROOM_ENABLED", True)
    monkeypatch.setattr(ro, "AJAY_SR_ROOM_MODE", "enforce")
    monkeypatch.setattr(ro, "AJAY_SR_REQUIRE_DATA", False)
    monkeypatch.setattr(ro, "MARKET_DATA_SOURCE", "binance")
    monkeypatch.setattr(ro, "CROSS_EXCHANGE_PRICE_GUARD_ENABLED", True)
    monkeypatch.setattr(ro, "_current_close_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "fetch_binance_price", lambda symbol: 100.0)
    monkeypatch.setattr(ro, "get_cached_sr_snapshot", lambda symbol: (_ for _ in ()).throw(RuntimeError("Futures data unavailable")))
    monkeypatch.setattr(ro, "record_action", lambda *args, **kwargs: None)
    called = {"open": 0}
    monkeypatch.setattr(ro, "open_market", lambda *args, **kwargs: called.__setitem__("open", called["open"] + 1) or {
        "status": "error", "error": "TEST_STOP_AFTER_OPEN_CALL"
    })

    out = ro.execute_new_position(
        "TEST", "LONG", 100.0,
        {"risk_pct": 2.0, "signal_price": 100.0, "event_type": "DONCHIAN_RETEST_BREAKOUT"},
        "EVT_SR_DATA_OPTIONAL",
    )
    assert called["open"] == 1
    assert out["status"] != "SR_DATA_UNAVAILABLE"


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
    assert [x["leg"] for x in captured["tp_levels"]] == ["tp1", "tp2"]
    assert captured["tp_levels"][0]["pnl_pct"] == pytest.approx(1.2)
    assert captured["tp_levels"][1]["pnl_pct"] == pytest.approx(2.9)

def test_post_fill_sr_does_not_mutate_tp2_when_zone_is_after_tp1(monkeypatch):
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
    assert [x["leg"] for x in captured["tp_levels"]] == ["tp1", "tp2"]
    assert captured["tp_levels"][0]["pnl_pct"] == pytest.approx(1.2)
    assert captured["tp_levels"][1]["pnl_pct"] == pytest.approx(2.9)



def test_futures_resolver_uses_exact_active_perpetual_symbol(monkeypatch):
    import event_engine.sr_context as sr

    calls = []
    monkeypatch.setattr(sr, "resolve_binance_futures_symbol", lambda symbol: calls.append(symbol) or symbol)
    out = sr.resolve_futures_symbol("1000SHIB-USDT")
    assert out["requested_symbol"] == "1000SHIBUSDT"
    assert out["source_symbol"] == "1000SHIBUSDT"
    assert out["source_alias_kind"] == "EXACT_FUTURES"
    assert out["contract_type"] == "PERPETUAL"
    assert calls == ["1000SHIBUSDT"]


def test_futures_resolver_uses_direct_contract_without_cross_venue_scaling(monkeypatch):
    import event_engine.sr_context as sr

    monkeypatch.setattr(sr, "resolve_binance_futures_symbol", lambda symbol: symbol)
    for logical in ("1000SHIB", "1000PEPE", "1000BONK", "1INCH"):
        resolved = sr.resolve_futures_symbol(logical)
        assert resolved["source_symbol"] == f"{logical}USDT"
        assert resolved["source_alias_kind"] == "EXACT_FUTURES"
        assert "source_price_scale" not in resolved
        assert "source_volume_scale" not in resolved


def test_futures_resolver_fails_closed_when_contract_is_not_listed(monkeypatch):
    import event_engine.sr_context as sr

    monkeypatch.setattr(sr, "resolve_binance_futures_symbol", lambda _symbol: None)
    with pytest.raises(sr.SRSymbolUnavailableError, match="Binance USDⓈ-M Futures"):
        sr.resolve_futures_symbol("NOTLISTED")


def test_futures_1h_kline_fetch_uses_shared_binance_client_without_price_scaling(monkeypatch):
    import event_engine.sr_context as sr

    monkeypatch.setattr(sr, "SR_MIN_CANDLES", 2)
    monkeypatch.setattr(sr, "resolve_futures_symbol", lambda _symbol: {
        "requested_symbol": "1000SHIBUSDT", "source_symbol": "1000SHIBUSDT",
        "underlying_asset": "1000SHIB", "source_alias_kind": "EXACT_FUTURES",
        "contract_type": "PERPETUAL", "quote_asset": "USDT",
    })
    rows = [
        {"open_time": i * sr.SR_SOURCE_INTERVAL_MS, "open": 1.0 + i, "high": 1.2 + i,
         "low": 0.8 + i, "close": 1.1 + i, "volume": 5000.0 + i}
        for i in range(6)
    ]
    calls = []
    monkeypatch.setattr(sr, "fetch_futures_klines", lambda symbol, interval, limit: calls.append((symbol, interval, limit)) or rows)

    candles, meta = sr._fetch_closed_1h_futures("1000SHIB", limit=6)
    assert calls == [("1000SHIBUSDT", "1h", 6)]
    assert len(candles) == 6
    assert candles[0].open == pytest.approx(1.0)
    assert candles[0].volume == pytest.approx(5000.0)
    assert meta["source_symbol"] == "1000SHIBUSDT"


def test_futures_live_price_uses_resolved_perpetual_symbol(monkeypatch):
    import event_engine.sr_context as sr

    calls = []
    monkeypatch.setattr(sr, "fetch_futures_price", lambda symbol: calls.append(symbol) or 2.9459)
    out = sr.get_futures_current_price("PIEVERSE", snapshot={"source_symbol": "PIEVERSEUSDT"})
    assert out == pytest.approx(2.9459)
    assert calls == ["PIEVERSEUSDT"]


def test_sr_semantics_do_not_confuse_pivot_high_below_long_entry_with_support():
    import event_engine.sr_context as sr
    out = sr.evaluate_sr_room(
        _snapshot((95.9, "H")),
        entry_price=100.0,
        direction="LONG",
        risk_pct=7.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert out["reject"] is False
    assert out["nearest_opposing_zone"] is None
    assert out["nearest_supporting_zone"] is None
    assert out["supporting_zone_confirmation"] is False


def test_sr_semantics_for_short_use_low_as_opposing_high_as_supporting():
    import event_engine.sr_context as sr
    opposing = sr.evaluate_sr_room(
        _snapshot((95.9, "L")),
        entry_price=100.0,
        direction="SHORT",
        risk_pct=7.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    supporting = sr.evaluate_sr_room(
        _snapshot((104.9, "H")),
        entry_price=100.0,
        direction="SHORT",
        risk_pct=7.0,
        target_rrs=(0.75, 1.50, 2.50),
    )
    assert opposing["nearest_opposing_zone"] is not None
    assert supporting["nearest_supporting_zone"] is not None


def test_custom_sr_limit_is_passed_to_futures_klines_without_using_cache(monkeypatch):
    import event_engine.sr_context as sr
    sr.clear_sr_cache()
    seen = {}
    candles = [sr.Candle(i * 10_800_000, 1, 2, 0.5, 1.5, 10) for i in range(300)]

    def fake_fetch(symbol, **kwargs):
        seen.update(kwargs)
        return candles, {
            "requested_symbol": sr.normalize_futures_symbol(symbol),
            "source_symbol": "TESTUSDT",
            "underlying_asset": "TEST",
            "source_alias_kind": "EXACT_FUTURES",
            "contract_type": "PERPETUAL",
            "quote_asset": "USDT",
        }

    monkeypatch.setattr(sr, "_fetch_closed_3h_futures", fake_fetch)
    monkeypatch.setattr(sr.time, "time", lambda: (candles[-1].ts + 10_799_999 + 60_000) / 1000.0)
    monkeypatch.setattr(sr, "compute_current_sr", lambda *args, **kwargs: {
        "levels": [1.0], "sr_levels": [None, 1.0], "highestph": 2.0, "lowestpl": 0.5,
        "cwidth": 0.1, "event_idx": 250, "event_timestamp": candles[250].ts, "accepted": [],
    })
    out = sr.get_cached_sr_snapshot("TEST", limit=777)
    assert seen["limit"] == 777
    assert out["source_symbol"] == "TESTUSDT"
    assert out["source"] == "binance_usds_m_futures"


def test_entry_overlapping_support_zone_wins_over_closer_nonoverlapping_zone(monkeypatch):
    import event_engine.sr_context as sr

    monkeypatch.setattr(sr, "_build_zones", lambda snapshot: [
        {"center": 99.55, "lower": 99.40, "upper": 99.70, "kind": "L", "distance_from_entry": 0.30},
        {"center": 100.0, "lower": 99.50, "upper": 100.50, "kind": "L"},
    ])
    out = sr.evaluate_sr_room(
        _snapshot((99.5, "L")), entry_price=100.0, direction="LONG",
        risk_pct=2.0, target_rrs=(0.65, 1.50),
    )
    assert out["supporting_zone_confirmation"] is True
    assert out["directional_zone_alignment"] == "LONG_IN_DEMAND"
    assert out["nearest_supporting_zone"]["entry_overlap"] == "supporting"


def test_normalize_futures_symbol_accepts_tradingview_perpetual_suffix():
    import event_engine.sr_context as sr

    assert sr.normalize_futures_symbol("BTCUSDT.P") == "BTCUSDT"
    assert sr.normalize_futures_symbol("BINANCE:BTCUSDT.P") == "BTCUSDT"
    assert sr.normalize_futures_symbol("BTCUSDT.PERP") == "BTCUSDT"


def test_resolve_futures_symbol_maps_fully_qualified_tradingview_perpetual_ticker(monkeypatch):
    import event_engine.sr_context as sr

    received = []

    def resolve(symbol):
        received.append(symbol)
        return symbol

    monkeypatch.setattr(sr, "resolve_binance_futures_symbol", resolve)
    resolved = sr.resolve_futures_symbol("BINANCE:BTCUSDT.P")
    assert received == ["BTCUSDT"]
    assert resolved["requested_symbol"] == "BTCUSDT"
    assert resolved["source_symbol"] == "BTCUSDT"
    assert resolved["contract_type"] == "PERPETUAL"
    assert resolved["quote_asset"] == "USDT"
