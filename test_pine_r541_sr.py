from __future__ import annotations

import pytest

from pine_r541_sr import (
    Candle,
    _is_pivot_high,
    _is_pivot_low,
    build_confirmed_pivots,
    calculate_atr,
    calculate_true_range,
    compute_current_sr,
)


def _candle(ts: int, high: float, low: float, close: float | None = None) -> Candle:
    close = (high + low) / 2.0 if close is None else float(close)
    return Candle(ts=ts, open=close, high=high, low=low, close=close, volume=100.0)


def test_pivot_helpers_require_left_and_right_confirmation_bars():
    candles = [
        _candle(0, 5.0, 4.0),
        _candle(1, 9.0, 3.0),
        _candle(2, 6.0, 3.5),
    ]
    assert _is_pivot_high(candles, 0, 1) is False
    assert _is_pivot_low(candles, 0, 1) is False
    assert _is_pivot_high(candles, 1, 1) is True
    # A wide-range candle can legitimately be both a swing high and low.
    assert _is_pivot_low(candles, 1, 1) is True


def test_pivot_ties_match_tradingview_asymmetric_rule():
    # Older/left equal extreme is allowed; newer/right equal extreme wins.
    high_left_tie = [_candle(0, 10, 8), _candle(1, 10, 7), _candle(2, 9, 8)]
    high_right_tie = [_candle(0, 8, 7), _candle(1, 10, 6), _candle(2, 10, 8)]
    low_left_tie = [_candle(0, 12, 5), _candle(1, 13, 5), _candle(2, 12, 6)]
    low_right_tie = [_candle(0, 12, 7), _candle(1, 13, 5), _candle(2, 12, 5)]
    assert _is_pivot_high(high_left_tie, 1, 1) is True
    assert _is_pivot_high(high_right_tie, 1, 1) is False
    assert _is_pivot_low(low_left_tie, 1, 1) is True
    assert _is_pivot_low(low_right_tie, 1, 1) is False


def test_build_confirmed_pivots_emits_on_confirmation_bar():
    candles = [
        _candle(0, 2, 1),
        _candle(1, 9, 4),
        _candle(2, 5, 2),
        _candle(3, 3, 1),
    ]
    ph, pl = build_confirmed_pivots(candles, swing_length=1)
    assert ph[1] is None
    assert ph[2] == 9.0
    assert pl[2] is None


def test_true_range_first_bar_uses_high_minus_low_and_later_bars_include_gaps():
    candles = [
        Candle(0, 10, 12, 9, 11, 1),
        Candle(1, 15, 16, 14, 15, 1),
        Candle(2, 12, 13, 10, 11, 1),
    ]
    assert calculate_true_range(candles) == pytest.approx([3.0, 5.0, 5.0])


def test_atr_is_sma_seeded_rma_of_true_range():
    candles = [_candle(i, 10 + i, 8 + i) for i in range(4)]
    atr = calculate_atr(candles, length=2)
    # TR values are 2, 2, 2, 2; the SMA seed and RMA recurrence both remain 2.
    assert atr[:1] == [None]
    assert atr[1:] == pytest.approx([2.0, 2.0, 2.0])


def test_supply_zone_uses_quarter_atr_and_is_created_only_on_confirmation_bar():
    candles = [
        _candle(0, 10, 8, 9),
        _candle(1, 15, 9, 12),  # swing high
        _candle(2, 11, 7, 9),  # confirms pivot; TR=5, box thickness=1.25
        _candle(3, 12, 8, 10),
    ]
    out = compute_current_sr(candles, swing_length=1, history_to_keep=20, box_width=2.5, atr_length=1)
    supply = out["supply_zones"]
    assert len(supply) == 1
    zone = supply[0]
    assert zone["top"] == pytest.approx(15.0)
    assert zone["bottom"] == pytest.approx(13.75)
    assert zone["center"] == pytest.approx(14.375)
    assert zone["pivot_index"] == 1
    assert zone["confirmation_index"] == 2
    assert zone["created_timestamp"] == candles[2].ts
    assert zone["pivot_timestamp"] == candles[1].ts


def test_demand_zone_uses_quarter_atr_and_bos_removes_zone_from_active_set():
    candles = [
        _candle(0, 12, 9, 10),
        _candle(1, 11, 7, 9),  # swing low
        _candle(2, 13, 8, 10),  # confirms pivot; ATR=5, band=1.25
        _candle(3, 9, 6, 6.9),  # close <= bottom (7.0): BOS
    ]
    out = compute_current_sr(candles, swing_length=1, box_width=2.5, atr_length=1)
    assert out["demand_zones"] == []
    assert out["demand_bos"]
    assert out["demand_bos"][0]["bos_type"] == "DEMAND_BROKEN"
    assert out["demand_bos"][0]["bos_timestamp"] == 3
    assert out["demand_slots"][0] is None


def test_wick_through_zone_does_not_break_it_without_close_through():
    candles = [
        _candle(0, 10, 8, 9),
        _candle(1, 15, 9, 12),
        _candle(2, 11, 7, 9),
        _candle(3, 16, 8, 14),  # high crosses supply but close remains below top=15
    ]
    out = compute_current_sr(candles, swing_length=1, box_width=2.5, atr_length=1)
    assert len(out["supply_zones"]) == 1
    assert out["supply_bos"] == []


def test_simultaneous_swing_high_and_low_obeys_pine_if_else_if_high_precedence():
    candles = [
        _candle(0, 10, 9, 9.5),
        _candle(1, 15, 5, 10),  # both local high and low
        _candle(2, 11, 8, 9.5),
    ]
    out = compute_current_sr(candles, swing_length=1, box_width=2.5, atr_length=1)
    assert len(out["supply_zones"]) == 1
    assert out["demand_zones"] == []


def test_active_zone_overlap_is_same_type_only_and_uses_two_atr_around_poi():
    candles = [
        _candle(0, 10, 8, 9),
        _candle(1, 15, 9, 12),
        _candle(2, 11, 7, 9),
        _candle(3, 10, 8, 9),
        _candle(4, 15.5, 9, 12),  # second high candidate; close does not break first box
        _candle(5, 11, 7, 9),
    ]
    out = compute_current_sr(candles, swing_length=1, box_width=2.5, atr_length=1)
    # With ATR around 4-5, the second POI sits inside first POI +/- 2*ATR.
    assert len(out["supply_zones"]) == 1


def test_history_queue_is_bounded_and_active_zones_are_newest_first():
    candles = []
    # Generate alternating pivot highs/lows far enough apart to avoid same-type overlap.
    highs = [10, 20, 10, 12, 24, 12, 14, 28, 14, 16, 32, 16]
    for idx, high in enumerate(highs):
        low = high - 5
        candles.append(_candle(idx, high, low, high - 2))
    out = compute_current_sr(candles, swing_length=1, history_to_keep=2, box_width=2.5, atr_length=1)
    assert len(out["supply_slots"]) == 2
    assert len(out["demand_slots"]) == 2


def test_compute_current_sr_enforces_atr_and_pivot_warmup_minimum():
    candles = [_candle(i, 2.0 + i, 1.0 + i) for i in range(4)]
    with pytest.raises(ValueError, match=r"Need at least 5 candles, got 4"):
        compute_current_sr(candles, swing_length=2, atr_length=5)


def test_empty_snapshot_has_explicit_empty_zones_when_no_confirmed_extreme_is_active():
    candles = [_candle(i, 10.0 + i, 1.0 + i) for i in range(50)]
    out = compute_current_sr(candles, swing_length=10, atr_length=50)
    assert out["algorithm"] == "AJAY_R541_DEMAND_SUPPLY"
    assert out["active_zones"] == []
    assert out["supply_zones"] == []
    assert out["demand_zones"] == []
    assert out["atr"] is not None
    assert out["latest_closed_timestamp"] == candles[-1].ts
    assert out["latest_closed_close"] == candles[-1].close


def test_compute_current_sr_returns_only_zones_still_active_at_latest_bar():
    candles = [
        _candle(0, 10, 8, 9),
        _candle(1, 15, 9, 12),
        _candle(2, 11, 7, 9),
        _candle(3, 16, 8, 15),  # close >= supply top, breaks zone
    ]
    out = compute_current_sr(candles, swing_length=1, box_width=2.5, atr_length=1)
    assert out["supply_zones"] == []
    assert out["supply_bos"][0]["bos_type"] == "SUPPLY_BROKEN"
    assert out["latest_closed_timestamp"] == 3
